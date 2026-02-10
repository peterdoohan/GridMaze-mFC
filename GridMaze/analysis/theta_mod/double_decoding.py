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
from matplotlib import pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_1samp
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.multitest import multipletests
from matplotlib.ticker import ScalarFormatter

from GridMaze.analysis.core import folds
from GridMaze.analysis.core import convert
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.distance_to_goal import distributions as dd
from GridMaze.analysis.distance_to_goal import logreg_decoder as ld
from GridMaze.analysis.place_direction import future_decoding as fd
from GridMaze.analysis.place_direction.future_decoding import get_decision_points
from GridMaze.analysis.theta_mod import distance_to_goal_tuning as gt
from GridMaze.maze import representations as mr
from GridMaze.maze import plotting as mp

# %% global variables
FRAME_RATE = 60


# %% session level double decoding function


def get_session_double_decoding_df(
    session,
    resolution=0.2,
    moving_only=True,
    bin_spacing=0.04,
    max_distance=0.8,
    max_steps_from_goal=30,
    place_offset=6,
    n_folds=10,
    sqrt_spikes=True,
    normalise_X=True,
    alpha="opt",
    output="weighted",
    distance_ref="goal",
    restrict_to_traj=True,
    verbose=True,
):
    """ """
    # load data
    if verbose:
        print("Loading input data...")
    input_data = get_input_data(
        session,
        resolution=resolution,
        moving_only=moving_only,
        bin_spacing=bin_spacing,
        max_distance=max_distance,
        max_steps_to_goal=max_steps_from_goal,
        place_offset=place_offset,
    )

    # generate variables to be used across folds, reg validation etc.
    distances = np.sort(input_data.distance_bin_mid.unique())  # in order corresponding to bin_id [0, 1, ...]
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
            d_pred = _get_distance_pred_distance(Yd_prob, distances=distances)
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

    return results_df


def get_opt_alpha(
    fold_df,
    train_df,
    var="distance_to_goal",
    normalise_X=True,
    sqrt_spikes=True,
    reg_range=np.logspace(-4, 4, 10),
    output="weighted",
    distance_ref="goal",
    restrict_to_traj=True,
    all_pairs_path_length=None,
    distances=None,
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
                pred_dist = _get_distance_pred_distance(Yprob=Yprob, distances=distances, output=output)
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
            results[i, j] = np.nanmean(scores**2)
    opt_alpha = reg_range[np.nanmean(results, axis=0).argmin()]
    return opt_alpha


def _get_distance_pred_distance(Yprob, distances, output="weighted"):
    """ """
    if output == "weighted":
        pred_dist = Yprob.dot(distances)
    elif output == "max":
        pred_dist = distances[np.argmax(Yprob, axis=1)]
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
    distance_ref="goal",  # pos or goal
):
    """ """
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
        if y not in decoder_classes:
            full_traj_defined[i] = False
            pred_dist[i] = np.nan
            in_train[i] = False
            continue
        probs = Yprob[i]
        loc2prob = dict(zip(decoder_classes, probs))
        _measure2 = goals[i] if distance_ref == "goal" else y
        if restrict_to_traj:
            traj = traj_envelope.iloc[i]
            if traj.isnull().any():
                full_traj_defined[i] = False
            include_locs = traj.dropna().unique()
        else:
            include_locs = decoder_classes
        probs = np.array([loc2prob[loc] if loc in loc2prob.keys() else np.nan for loc in include_locs])
        distances = np.array([all_pairs_path_length[loc][_measure2] for loc in include_locs])
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
    resolution=0.4,
    metric=("distance_to_goal", "geodesic"),
    include_multiunits=True,
    moving_only=True,
    navigation_only=True,
    remove_time_at_goal=True,
    max_steps_to_goal=30,
    bin_spacing=0.04,
    bin_method="uniform",
    max_distance=1.6,
    n_log_bins=25,
    place_offset=6,
):
    """ """
    # load data
    navigation_df = session.navigation_df.copy()
    spike_counts_df = session.navigation_theta_spike_counts_df  # [frames, clusters * 12 lfp phase bins]
    spike_counts_df.reset_index(inplace=True, drop=True)

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

    # downsample data
    ds_nav_df, ds_spike_counts_df = ds.downsample_nav_spikes_data(
        navigation_df,
        spike_counts_df,
        resolution=resolution,
        distance_metrics=[("steps_to_goal", "future"), metric],
    )
    ds_nav_df.columns = pd.MultiIndex.from_tuples([(*c, "") for c in ds_nav_df.columns])
    metric = (*metric, "")
    input_df = pd.concat([ds_nav_df, ds_spike_counts_df], axis=1)

    # add future, past state (place) information
    input_df[("place_direction", "", "")] = input_df.maze_position.simple + "_" + input_df.cardinal_movement_direction
    offset_df = fd.get_past_and_future_states(
        input_df, state_type="place", past_offset=place_offset, future_offset=place_offset
    )
    offset_df.columns = pd.MultiIndex.from_tuples([(*col, "") for col in offset_df.columns])
    input_df = pd.concat([input_df, offset_df], axis=1)

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
