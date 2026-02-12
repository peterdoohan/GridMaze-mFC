"""
Lets try and decode both place-direction and distance-to-goal simultaenously and see if there is
structure in the decoding outputs
@peterdoohan
"""

# %% imports
import json
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from copy import deepcopy
from matplotlib import axes, pyplot as plt
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pingouin import multivariate_ttest, circ_rayleigh
from scipy.ndimage import gaussian_filter
from sympy import true


from GridMaze.maze import representations as mr
from GridMaze.analysis.core import folds
from GridMaze.analysis.core import convert
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.distance_to_goal import distributions as dd
from GridMaze.analysis.place_direction import future_decoding as fd

from GridMaze.analysis.theta_mod import trajectory_decoding as tpd
from GridMaze.analysis.distance_to_goal import theta_mod_decoder as tdd

# %% global variables
FRAME_RATE = 60

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "theta_mod" / "double_decoding"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)


# %%
def plot_double_theta_mod_decoding(results_df, demean="subject", print_stats=True, ax=None):
    """ """
    # calculate decoding bias (pred - true)
    df = results_df.copy()
    bias_dfs = []
    for rep in ["from_distance", "from_place"]:
        if rep == "from_distance":
            bias_df = df.decoded_distance[rep].sub(df.distance_bin_mid, axis=0)
        else:
            bias_df = df.decoded_distance[rep]  # as error from place
        if demean == "sample":
            bias_df = bias_df.sub(bias_df.mean(axis=1), axis=0)
        bias_df.columns = pd.MultiIndex.from_product([["decoding_bias"], [rep], bias_df.columns])
        bias_dfs.append(bias_df)
    df = pd.concat([df] + bias_dfs, axis=1)

    subject_avg = df.groupby(["subject_ID"]).decoding_bias.mean()
    dist_bias = subject_avg.decoding_bias["from_distance"]
    place_bias = subject_avg.decoding_bias["from_place"]
    if demean:
        dist_bias = dist_bias.sub(dist_bias.mean(axis=1), axis=0)
        place_bias = place_bias.sub(place_bias.mean(axis=1), axis=0)

    _plot_double_decoding_bias(dist_bias, place_bias, print_stats=print_stats, ax=ax)


# %%  Exp level function


def get_double_decoding_df(verbose=True, n_jobs=-1, save=False):
    """
    versions (hacky):
    "... _close_low_res": first attempt that was at 0.2 res (same for samples and sum_spikes) w/ max dist = 0.8
    """
    save_path = RESULTS_DIR / "double_decoding_df.parquet"
    if save_path.exists() and not save:
        if verbose:
            print(f"Loading existing double results from {save_path} ...")
        return pd.read_parquet(save_path)
    results_dfs = []
    for subject_ID in SUBJECT_IDS:
        for maze_name in ["maze_1", "maze_2", "rooms_maze"]:
            if verbose:
                print(f"Loading sessions for {subject_ID} - {maze_name} ...")
            sessions = gs.get_maze_sessions(
                subject_IDs=[subject_ID],
                maze_names=[maze_name],
                days_on_maze="late",
                with_data=[
                    "navigation_df",
                    "cluster_metrics",
                    "trials_df",
                    "navigation_theta_spike_counts_df",
                ],
                must_have_data=True,
            )
            if n_jobs:
                dfs = Parallel(n_jobs=n_jobs)(delayed(get_session_double_decoding_df)(session) for session in sessions)
            else:
                dfs = [get_session_double_decoding_df(session) for session in sessions]
            results_dfs.extend(dfs)
    results_df = pd.concat(results_dfs, axis=0, ignore_index=True)
    if save:
        results_df.to_parquet(save_path)
        if verbose:
            print(f"Saved double decoding results to {save_path}.")
    return results_df


# %% session level double decoding function


def get_session_double_decoding_df(
    session,
    resolution=0.1,
    sum_spike_window=0.4,
    moving_only=True,
    bin_spacing=0.08,
    max_distance=None,
    max_steps_from_goal=30,
    place_offset=4,
    n_folds=8,
    sqrt_spikes=True,
    normalise_X=True,
    alpha="opt",
    output="weighted",
    distance_ref="pos",
    restrict_to_traj=True,
    verbose=True,
):
    """ """
    # load data
    if verbose:
        print(f"{session.name}: loading input data...")
    input_data = get_input_data(
        session,
        resolution=resolution,
        sum_spike_window=sum_spike_window,
        moving_only=moving_only,
        bin_spacing=bin_spacing,
        max_distance=max_distance,
        max_steps_to_goal=max_steps_from_goal,
        place_offset=place_offset,
    )

    # generate variables to be used across folds, reg validation etc.
    distances = np.sort(input_data.distance_bin_mid.unique())  # in order corresponding to bin_id [0, 1, ...]
    distance_bin_ids = np.sort(input_data.distance_bin_id.unique())
    lfp_phases = input_data.spike_count.columns.get_level_values(1).unique().values
    all_pairs_path_length = _get_all_pairs_path_length(session)
    folds_df = folds.get_folds_df(
        session,
        goal_stratified=False,
        n_folds=n_folds,
        return_unique_IDs=False,
    )

    # init results df
    results_df = pd.concat(
        [
            input_data.drop(["spike_count", "past", "future"], axis=1, level=0).copy(),
            pd.DataFrame(
                index=input_data.index,
                columns=pd.MultiIndex.from_product((["decoded_distance"], ["from_distance", "from_place"], lfp_phases)),
            ),
        ],
        axis=1,
    )
    results_df[("place_decoding_info", "traj_defined", "")] = False
    results_df[("place_decoding_info", "in_train", "")] = False

    # over cv folds train separate decoders to predict distance_to_goal from population place rep or from distance rep
    # on average activity across theta phases, then test on each theta phase separately
    _folds = folds_df.columns.get_level_values(0).unique()
    for fold in _folds:
        if verbose:
            print(fold)
        fold_df = folds_df[fold]
        train_trials, test_trials = [fold_df[t].unstack().dropna().values for t in ["train", "test"]]
        train_df, test_df = [input_data[input_data.trial.isin(trials)] for trials in [train_trials, test_trials]]
        # train decoder on mean spikes across theta phases
        X_train_mean, X_test_mean = [df.spike_count.T.groupby(level=0).mean().T.values for df in [train_df, test_df]]
        if sqrt_spikes:
            X_train_mean, X_test_mean = np.sqrt(X_train_mean), np.sqrt(X_test_mean)
        if normalise_X:
            scaler = StandardScaler().fit(X_train_mean)
            X_train_mean, X_test_mean = scaler.transform(X_train_mean), scaler.transform(X_test_mean)
        # decoder either distace-to-goal or place (we will set up different decoders for each)
        Yd_train, Yd_test = [df.distance_bin_id.values for df in [train_df, test_df]]
        Yp_train, Yp_test = [df.maze_position.simple.values for df in [train_df, test_df]]
        # optionaly find optimal xval regularisation
        if alpha == "opt":
            if verbose:
                print("    Finding optimal alpha for distance decoder...")
            d_alpha = get_opt_alpha(
                fold_df,
                train_df,
                var="distance_to_goal",
                normalise_X=normalise_X,
                sqrt_spikes=sqrt_spikes,
                distances=distances,
                distance_bin_ids=distance_bin_ids,
                output=output,
                verbose=verbose,
            )
            if verbose:
                print("    Finding optimal alpha for place decoder...")
            p_alpha = get_opt_alpha(
                fold_df,
                train_df,
                var="place",
                normalise_X=normalise_X,
                sqrt_spikes=sqrt_spikes,
                output=output,
                distance_ref=distance_ref,
                restrict_to_traj=restrict_to_traj,
                all_pairs_path_length=all_pairs_path_length,
                verbose=verbose,
            )
        else:
            d_alpha, p_alpha = alpha, alpha
        # train decoders
        d_decoder = LogisticRegression(C=d_alpha, random_state=0, max_iter=10_000, class_weight="balanced")
        d_decoder.fit(X_train_mean, Yd_train)
        train_distances_bin_ids = d_decoder.classes_
        p_decoder = LogisticRegression(C=p_alpha, random_state=0, max_iter=10_000, class_weight="balanced")
        p_decoder.fit(X_train_mean, Yp_train)
        train_locations = p_decoder.classes_
        # test on spikes at each theta phase
        for phase in lfp_phases:
            X_theta_test = test_df.spike_count.xs(phase, level=1, axis=1).values
            if sqrt_spikes:
                X_theta_test = np.sqrt(X_theta_test)
            if normalise_X:
                X_theta_test = scaler.transform(X_theta_test)
            Yd_prob = d_decoder.predict_proba(X_theta_test)
            d_pred = _get_distance_pred_distance(
                Yd_prob,
                distances=distances,
                distance_bin_ids=distance_bin_ids,
                decoder_classes=train_distances_bin_ids,
                output=output,
            )
            results_df.loc[test_df.index, ("decoded_distance", "from_distance", phase)] = d_pred
            Yp_prob = p_decoder.predict_proba(X_theta_test)
            p_pred, p_traj_defined, p_in_train = _get_place_pred_distance(
                Yp_prob,
                Yp_test,
                test_df,
                decoder_classes=train_locations,
                all_pairs_path_length=all_pairs_path_length,
                output=output,
                distance_ref=distance_ref,
                return_as="all",
            )
            results_df.loc[test_df.index, ("decoded_distance", "from_place", phase)] = p_pred
        # places in training data and whether full trajectory is same for test at each theta phase
        results_df.loc[test_df.index, ("place_decoding_info", "traj_defined", "")] = p_traj_defined
        results_df.loc[test_df.index, ("place_decoding_info", "in_train", "")] = p_in_train

    return results_df.reset_index(drop=True)


def get_opt_alpha(
    fold_df,
    train_df,
    var="distance_to_goal",
    normalise_X=True,
    sqrt_spikes=True,
    reg_range=np.logspace(-4, 4, 10),
    output="weighted",
    distance_ref="pos",
    restrict_to_traj=True,
    all_pairs_path_length=None,
    distances=None,
    distance_bin_ids=None,
    verbose=False,
):
    """ """
    # check inputs
    if var not in ["distance_to_goal", "place"]:
        raise ValueError(f"var must be 'distance_to_goal' or 'place'.")
    if var == "place" and all_pairs_path_length is None:
        raise ValueError(f"Must provide all_pairs_path_length for place decoding.")
    if var == "distance_to_goal" and distances is None:
        raise ValueError(f"Must provide distances for distance_to_goal decoding.")

    vfolds_df = fold_df.train
    vfolds = vfolds_df.columns
    results = np.zeros((len(vfolds), len(reg_range)))
    for i, vfold in enumerate(vfolds):
        if verbose:
            print(f"        vfold: {i}")
        val_trials = vfolds_df[vfold].dropna().values
        train_trials = vfolds_df[[t for t in vfolds if t != vfold]].unstack().dropna().values
        _train_df = train_df[train_df.trial.isin(train_trials)]
        _val_df = train_df[train_df.trial.isin(val_trials)]
        # train and test on average spikes over theta phases (reg search is theta independent)
        X_train, X_val = [df.spike_count.T.groupby(level=0).mean().T.values for df in [_train_df, _val_df]]
        if X_train.shape[0] == 0 or X_val.shape[0] == 0:
            continue
        if var == "place":
            Y_train, Y_val = [df.maze_position.simple.values for df in [_train_df, _val_df]]
        if var == "distance_to_goal":
            Y_train, Y_val = [df.distance_bin_id.values for df in [_train_df, _val_df]]
        val_distance_to_goal = _val_df.distance_bin_mid.values
        # standardise
        if sqrt_spikes:
            X_train, X_val = np.sqrt(X_train), np.sqrt(X_val)
        if normalise_X:
            scaler = StandardScaler().fit(X_train)
            X_train, X_val = scaler.transform(X_train), scaler.transform(X_val)
        # fit model
        for j, alpha in enumerate(reg_range):
            decoder = LogisticRegression(C=alpha, random_state=0, max_iter=10_000, class_weight="balanced")
            decoder.fit(X_train, Y_train)
            decoder_classes = decoder.classes_
            Yprob = decoder.predict_proba(X_val)
            if var == "distance_to_goal":
                pred_dist = _get_distance_pred_distance(
                    Yprob=Yprob,
                    distances=distances,
                    distance_bin_ids=distance_bin_ids,
                    decoder_classes=decoder_classes,
                    output=output,
                )
            if var == "place":
                pred_dist = _get_place_pred_distance(
                    Yprob,
                    Y_val,
                    _val_df,
                    decoder_classes,
                    all_pairs_path_length,
                    output=output,
                    distance_ref=distance_ref,
                    restrict_to_traj=restrict_to_traj,
                    return_as="dist",
                )
            scores = val_distance_to_goal - pred_dist
            if not np.isfinite(scores).any():
                results[i, j] = np.nan
                continue
            results[i, j] = np.nanmean(scores**2)
    opt_alpha = reg_range[np.nanmean(results, axis=0).argmin()]
    return opt_alpha


def _get_distance_pred_distance(Yprob, distances, decoder_classes, distance_bin_ids, output="weighted"):
    """ """
    # filter distances_bins for those that were in training
    _distances = distances[np.isin(distance_bin_ids, decoder_classes)]
    if output == "weighted":
        pred_dist = Yprob.dot(_distances)
    elif output == "max":
        pred_dist = _distances[np.argmax(Yprob, axis=1)]
    return pred_dist


def _get_place_pred_distance(
    Yprob,
    Y_test,
    test_df,
    decoder_classes,
    all_pairs_path_length,
    restrict_to_traj=True,
    output="weighted",  # 'max' or 'weighted'
    return_as="dist",  # value to return
    distance_ref="pos",  # pos or goal
):
    """
    This is complicated, should write doc-string
    """
    # check inputs
    if output not in ["max", "weighted"]:
        raise ValueError(f"Output must be 'max' or 'weighted'.")
    if distance_ref not in ["pos", "goal"]:
        raise ValueError(f"distance_ref must be 'pos' or 'goal'.")
    if return_as not in ["dist", "all"]:
        raise ValueError(f"return_as must be 'dist' or 'all'.")

    # extract rel info
    traj_envelope = test_df[["past", "future"]]  # past & future parts of trajectory
    goals = test_df.goal.values  # goal active on each sample

    # init outputs
    samples = Yprob.shape[0]
    pred_dist = np.zeros(samples)
    full_traj_defined = np.ones(samples, dtype=bool)
    in_train = np.ones(samples, dtype=bool)
    for i in range(samples):
        y = Y_test[i]  # place
        goal = goals[i]
        if y not in decoder_classes:
            full_traj_defined[i] = False
            pred_dist[i] = np.nan
            in_train[i] = False
            continue
        probs = Yprob[i]
        loc2prob = dict(zip(decoder_classes, probs))
        if restrict_to_traj:
            traj = traj_envelope.iloc[i]
            if traj.isnull().any():
                full_traj_defined[i] = False
            include_locs = traj.dropna().unique()
            past_locs = traj.loc["past"].iloc[1:].dropna().unique()
            future_locs = traj.loc["future"].iloc[1:].dropna().unique()
            if distance_ref == "goal":
                probs = np.array([loc2prob[loc] if loc in loc2prob.keys() else np.nan for loc in include_locs])
                distances = np.array([all_pairs_path_length[loc][goal] for loc in include_locs])
                if output == "weighted":
                    pred_dist[i] = np.nansum(probs * distances) / np.nansum(probs)
                if output == "max":
                    pred_dist[i] = distances[np.nanargmax(probs)]
            else:  # pos
                assert output == "weighted", "max output not implemented for position ('pos') ref"
                past_probs = [loc2prob[loc] if loc in loc2prob.keys() else np.nan for loc in past_locs]
                past_dists = np.array([all_pairs_path_length[y][loc] for loc in past_locs])
                future_probs = [loc2prob[loc] if loc in loc2prob.keys() else np.nan for loc in future_locs]
                future_dists = np.array([all_pairs_path_length[y][loc] for loc in future_locs])
                # calculate decoding error distance over the trajectory (+ve = more past, -ve = more future)
                weighted_past, weighted_future = np.nansum(future_probs * future_dists), np.nansum(
                    past_probs * past_dists
                )
                pred_dist[i] = weighted_past - weighted_future
        else:
            include_locs = decoder_classes
            if distance_ref == "goal":
                distances = np.array([all_pairs_path_length[loc][goal] for loc in include_locs])
                if output == "weighted":
                    pred_dist[i] = np.nansum(probs * distances) / np.nansum(probs)
                if output == "max":
                    pred_dist[i] = distances[np.nanargmax(probs)]
            else:  # pos
                distances = np.array([all_pairs_path_length[y][loc] for loc in include_locs])
                if output == "weighted":
                    pred_dist[i] = np.nansum(probs * distances) / np.nansum(probs)
                if output == "max":
                    pred_dist[i] = distances[np.nanargmax(probs)]

    if return_as == "dist":
        return pred_dist
    else:
        return (pred_dist, full_traj_defined, in_train)


# %% get input data


def get_input_data(
    session,
    theta_split=True,
    resolution=0.1,
    sum_spike_window=0.4,
    metric=("distance_to_goal", "geodesic"),
    include_multiunits=True,
    moving_only=True,
    navigation_only=True,
    remove_time_at_goal=True,
    max_steps_to_goal=30,
    bin_spacing=0.04,
    bin_method="uniform",
    max_distance=0.8,
    n_log_bins=25,
    place_offset=2,
):
    """ """
    # load data
    navigation_df = session.navigation_df.copy()
    if theta_split:
        spike_counts_df = session.navigation_theta_spike_counts_df  # [frames, clusters * 12 lfp phase bins]
        spike_counts_df.reset_index(inplace=True, drop=True)
    else:
        spike_counts_df = session.navigation_spike_counts_df  # [frames, clusters]
        spike_counts_df.reset_index(inplace=True, drop=True)
        spike_counts_df.columns = pd.MultiIndex.from_tuples([(*c, "") for c in spike_counts_df.columns])

    # filter clusters
    keep_clusters = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=include_multiunits,
    )
    spike_counts_df = spike_counts_df[
        spike_counts_df.columns[spike_counts_df.columns.get_level_values(1).isin(keep_clusters)]
    ]

    if sum_spike_window is None or sum_spike_window == resolution:
        # sum spikes and downsample behaviour to same resolution
        ds_nav_df, ds_spike_counts_df = ds.downsample_nav_spikes_data(
            navigation_df,
            spike_counts_df,
            resolution=resolution,
            distance_metrics=[("steps_to_goal", "future"), metric],
        )

    else:
        # sum spikes over spike_window (smooth)
        sum_frames = int(sum_spike_window * FRAME_RATE)
        spike_counts_df = spike_counts_df.rolling(window=sum_frames, center=True).sum().fillna(0).astype(int)
        # downsample (usually higher rate than sum_spikes)
        every_n_frames = int(resolution * FRAME_RATE)
        ds_spike_counts_df = spike_counts_df.iloc[::every_n_frames].reset_index(drop=True)
        ds_nav_df = navigation_df.iloc[::every_n_frames].reset_index(drop=True)
    ds_nav_df.columns = pd.MultiIndex.from_tuples([(*c, "") for c in ds_nav_df.columns])
    input_df = pd.concat([ds_nav_df, ds_spike_counts_df], axis=1).reset_index(drop=True)

    # add future, past state (place) information
    input_df[("place_direction", "", "")] = input_df.maze_position.simple + "_" + input_df.cardinal_movement_direction
    offset_df = fd.get_past_and_future_states(
        input_df, state_type="place", past_offset=place_offset, future_offset=place_offset
    )
    offset_df.columns = pd.MultiIndex.from_tuples([(*col, "") for col in offset_df.columns])
    input_df = pd.concat([input_df, offset_df], axis=1)
    input_df = input_df.sort_index(axis=1)  # sort columns for easier indexing later

    # get binned distance to goal
    if max_distance is None:
        max_distance = dd.get_distance_percentile(metric, 0.85)
    if bin_method == "uniform":
        n_bins = int(max_distance / bin_spacing)
    elif bin_method == "log":
        n_bins = n_log_bins
    input_df = input_df[input_df[metric] < max_distance]
    bins = convert._get_distance_bins(
        binning_method=bin_method,
        n_distance_bins=n_bins,
        distance_metrics=metric,
        max_distance=max_distance,
    )
    input_df.loc[:, ("distance_bin", "", "")] = pd.cut(input_df[metric], bins=bins, include_lowest=True).to_numpy()
    input_df.loc[:, ("distance_bin_mid", "", "")] = input_df.distance_bin.apply(lambda x: x.mid)
    input_df.loc[:, ("distance_bin_id", "", "")] = input_df.distance_bin.map({b: i for i, b in enumerate(bins)})

    # filter data
    input_df = filt.filter_navigation_rates_df(
        input_df,
        navigation_only=navigation_only,
        moving_only=moving_only,
        exclude_time_at_goal=remove_time_at_goal,
        max_steps_to_goal=max_steps_to_goal,
    )
    # add other info
    input_df[("subject_ID", "", "")] = session.subject_ID
    input_df[("maze_name", "", "")] = session.maze_name
    input_df[("day_on_maze", "", "")] = session.day_on_maze
    return input_df


def _get_all_pairs_path_length(session):
    skeleton_maze = session.skeleton_maze()
    dists = dict(nx.all_pairs_dijkstra_path_length(skeleton_maze, weight="weight"))
    coord2label = mr.get_maze_coord2label(skeleton_maze)
    _dists = {}
    for src in dists.keys():
        if src[-1] == 0:  # center of each tower/bridge
            src_dists = dists[src]
            __dists = {}
            for targ in src_dists.keys():
                if targ[-1] == 0:
                    __dists[coord2label[targ].split("_")[0]] = src_dists[targ]
                _dists[coord2label[src].split("_")[0]] = __dists
    return _dists


# %%


def compare_previous_decoding_profiles(print_stats=True, ax=None):
    """
    load data separate summary dfs from:
        - theta mod distance decoding
        - theta mod trajectory decoding
    and compare the modulation sinusoids and offsets
    """

    # load data: note summary dfs formated slighly differently
    distance_mod_df = tdd.load_decoding_results(lfp_type="theta_mid")
    place_mod_df = tpd.get_summary_df(verbose=False)
    # note signed error in place_mod_df is future-past, so pos is closer to goal
    # but in distance_mod_df, error is pred-true, so pos is further from goal, FIX
    place_mod_df.loc[:, "signed_error"] *= -1

    # process distance to get subject by norm decoding bias
    dist_bias = distance_mod_df.groupby(["subject_ID"]).lfp_phase.mean().lfp_phase
    dist_bias_norm = dist_bias.sub(dist_bias.mean(axis=1), axis=0)

    # process place to get same
    place_bias = place_mod_df.groupby(["subject_ID", "theta_phase"])[f"signed_error"].mean().unstack(0)
    place_bias_norm = place_bias.sub(place_bias.mean(), axis=1).T

    # plot a nice summary
    _plot_double_decoding_bias(
        dist_bias_norm,
        place_bias_norm,
        print_stats=print_stats,
        ax=ax,
    )


def _plot_double_decoding_bias(dist_bias, place_bias, print_stats=True, ax=None):
    # set up figure
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.axhline(0, color="k", ls="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("theta phase")
    ax.set_ylabel("decoding bias \n (norm.)")
    ax.set_xticks(np.arange(-np.pi, np.pi + 0.1, np.pi / 2))
    ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])

    phases = dist_bias.columns.values.astype(float)

    # plot a nice summary
    for _df, color, label in zip(
        [dist_bias, place_bias],
        ["darkviolet", "darkred"],
        ["distance", "place"],
    ):
        # normalise df to max
        plot_df = _df / np.max(np.abs(_df.values))
        mean = plot_df.mean().values
        sem = plot_df.sem().values
        # plot datapoints
        ax.errorbar(
            phases,
            mean,
            yerr=sem,
            fmt="o",
            color=color,
            markersize=5,
            linewidth=None,
            capsize=None,
            elinewidth=1.5,
        )
        # plot curvefit
        _x, _y = fit_sinusoid(phases, mean, fit_constant=True, return_as="curve")
        ax.plot(_x, _y, color=color, linewidth=1.5, label=label)

        # test sinusoidal random effects across subjects
        if print_stats:
            print(label)
            _get_decoding_bias_stats(_df)

    ax.legend(frameon=False, fontsize=8)

    # for each subject fit each modulation with a sinusoid and compare offsets
    if print_stats:
        offsets = []
        for subject in SUBJECT_IDS:
            _dist_curve = dist_bias.loc[subject].values
            dist_fit = fit_sinusoid(phases, _dist_curve, fit_constant=True, return_as="params")
            _place_curve = place_bias.loc[subject].values
            place_fit = fit_sinusoid(phases, _place_curve, fit_constant=True, return_as="params")
            # get phase offset
            off = place_fit["phi"] - dist_fit["phi"]
            # wrap to [-pi, pi]
            w_off = (off + np.pi) % (2 * np.pi) - np.pi
            offsets.append(w_off)
        z, p = circ_rayleigh(offsets, d=np.pi / 6)
        print(f"offset rayleigh test: z={z:.3f}, p={p:.3f}")


def fit_sinusoid(x, y, fit_constant=True, return_as="params"):
    """
    Fit y(x) ≈ alpha*sin(x) + beta*cos(x) + C  (period = 2π -> ω = 1)
    Returns dict with alpha, beta, C, A, phi (radians), residuals.
    Notes:
      - A = sqrt(alpha^2 + beta^2)
      - phi = atan2(beta, alpha)  (so model = A * sin(x + phi) + C)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    # design matrix: columns [sin(x), cos(x), (1)]
    X = np.column_stack([np.sin(x), np.cos(x)])
    if fit_constant:
        X = np.column_stack([X, np.ones_like(x)])
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    alpha = coeffs[0]
    beta = coeffs[1]
    C = coeffs[2] if fit_constant else 0.0
    A = np.hypot(alpha, beta)
    phi = np.atan2(beta, alpha)  # returns phase in radians
    if return_as == "params":
        return {"alpha": alpha, "beta": beta, "C": C, "A": A, "phi": phi}
    elif return_as == "curve":
        _x = np.linspace(-np.pi, np.pi, 100)
        _y = A * np.sin(_x + phi) + C
        return _x, _y
    else:
        raise ValueError(f"return_as must be 'params' or 'curve'.")


def _get_decoding_bias_stats(phase_mean_decoding):
    """ """
    phis = phase_mean_decoding.columns.astype(float)
    data = phase_mean_decoding.values
    beta_cos = data.dot(np.cos(phis))
    beta_sin = data.dot(np.sin(phis))
    betas = np.column_stack([beta_cos, beta_sin])
    zeros = np.zeros_like(betas)
    mv_test = multivariate_ttest(betas, zeros, paired=False)
    return print(mv_test)
