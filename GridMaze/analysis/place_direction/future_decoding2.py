"""
Can we decode the future position/place-direction of the animal from neural avtivity
(above that predicted by the current place-direction)?
@krisjensen @peterdoohan
"""

# %% Imports
from cProfile import label
import json
import copy
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from py import process
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import folds
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import convert

from GridMaze.maze import representations as mr
from GridMaze.maze import plotting as mp

# %% Global Variables

from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

RESULTS_DIR = RESULTS_PATH / "place_direction" / "future_decoding"

# %% Functions


def _get_stats_df(future_df, past_df):
    """ """
    stats_df = pd.DataFrame(index=np.arange(1, future_df.offset.max() + 1), columns=["future", "past"])
    for mode, df in zip(["future", "past"], [future_df, past_df]):
        if df is None:
            continue
        scores_df = df.groupby(["subject_ID", "regressors", "offset"]).score.mean().unstack(level=(1, 2))
        diff = scores_df["spikes_place_direction"] - scores_df["place_direction"]
        diff = diff.drop(columns=[0])
        # ttest each offset
        p_values = ttest_1samp(diff.values, 0, axis=0, alternative="greater").pvalue
        # correct for multiple comparisons
        reject, pvals_corrected, _, _ = multipletests(p_values, method="fdr_bh", alpha=0.05)
        stats_df.loc[:, mode] = pvals_corrected
    return stats_df


def plot_place_deocoding_summary(
    future_df, past_df=None, normalise=False, colors=["violet", "lightskyblue"], print_stats=True, ax=None
):
    """ """
    # set up figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Steps in the past/future")
    ax.set_ylabel("Decoding accuracy \n (chance norm.)")

    for mode, df, color in zip(["future", "past"], [future_df, past_df], colors):
        if df is None:
            continue
        # average over folds
        scores_df = (
            df.groupby(["subject_ID", "regressors", "offset"]).score.mean().unstack(level=(1, 2))
        )  # [subjects, regressors x offsets]
        pd_scores, spike_scores, pd_spikes_scores = (
            scores_df["place_direction"],
            scores_df["spikes"],
            scores_df["spikes_place_direction"],
        )

        diff = pd_spikes_scores - pd_scores
        # normalise
        if normalise:
            metric = diff / (1 - pd_scores)
        else:
            metric = diff
        # plot
        mean = metric.mean()
        mean_ = mean[mean.index > 0]
        mean_0 = mean[mean.index == 0]
        sem = metric.sem()
        sem_ = sem[sem.index > 0]
        sem_0 = sem[sem.index == 0]
        x_0 = mean_0.index.values
        x_ = mean_.index.values
        if mode == "past":
            x_ = -x_
        ax.errorbar(
            x_0,
            mean_0.values,
            yerr=sem_0.values,
            marker="o",
            color="grey",
        )
        ax.errorbar(
            x_,
            mean_.values,
            yerr=sem_.values,
            label=mode,
            marker="o",
            color=color,
            linestyle="-",
        )
    max_offset = future_df.offset.max()
    ax.set_xticks(np.arange(-max_offset, max_offset + 1, 2))
    ax.set_xticklabels(np.arange(-max_offset, max_offset + 1, 2))
    if print_stats:
        stats_df = _get_stats_df(future_df, past_df)
        print("offset pvalues:")
        print(stats_df)


# %%


def get_place_decoding_summary(
    mode="future",
    max_offset=8,
    subjects="all",
    maze_names=["maze_1", "maze_2"],
    days_on_maze="late",
    save=False,
    verbose=False,
):
    """ """
    save_path = RESULTS_DIR / f"{mode}_place_decoding_summary.csv"
    if not save and save_path.exists():
        if verbose:
            print(f"Loading existing results from {save_path}")
        return pd.read_csv(save_path, index_col=0)
    if verbose:
        print("Loading sessions ...")
    sessions = gs.get_maze_sessions(
        subject_IDs=subjects,
        maze_names=maze_names,
        days_on_maze=days_on_maze,
        with_data=[
            "navigation_df",
            "navigation_spike_counts_df",
            "cluster_metrics",
        ],
        must_have_data=True,
    )
    dfs = []
    for session in sessions:
        if verbose:
            print(session.name)
        results_df = get_session_future_place_decoding(
            session, mode=mode, offset=max_offset, state_type="place_direction"
        )  # defualt settings
        results_df["subject_ID"] = session.subject_ID
        results_df["maze_name"] = session.maze_name
        results_df["day_on_maze"] = session.day_on_maze
        dfs.append(results_df)
    summary_df = pd.concat(dfs, axis=0)
    if save:
        summary_df.to_csv(save_path)
        if verbose:
            print(f"Saving results to {save_path}")
    return summary_df


# %%


def quick_plot(df):
    """ """
    _df = df.groupby(["offset", "regressors"]).score.mean().unstack()
    diff = _df["spikes_place_direction"] - _df["place_direction"]
    diff.plot()


# %% Dev new core decoding function


def test(
    session,
    include_multi_units=True,
    max_steps_to_goal=30,
    resolution=0.2,
    modes=["future", "past"],
    offset=12,
    state_type="place_direction",
    min_spikes=300,
    sqrt_spikes=True,
    n_folds=5,
    alpha="opt",
    normalise_X=True,
    spikes_reg_weight=None,
    n_jobs=-1,
    verbose=True,
):
    """ """
    # input data (see get_input_df for details)
    input_df = get_input_df(
        session,
        include_multi_units,
        max_steps_to_goal,
        resolution,
        modes,
        offset,
        state_type,
        min_spikes,
    )
    simple_maze = session.simple_maze()
    _process_kwargs = {
        "session": session,
        "input_df": input_df,
        "sqrt_spikes": sqrt_spikes,
        "spikes_reg_weight": spikes_reg_weight,
        "alpha": alpha,
        "simple_maze": simple_maze,
        "state_type": state_type,
        "n_folds": n_folds,
        "normalise_X": normalise_X,
        "verbose": verbose,
    }
    if n_jobs is not None:
        results = Parallel(n_jobs=n_jobs, verbose=True)(
            delayed(_process_offset)(mode, off, **_process_kwargs)
            for mode in modes
            for off in range(0, offset + 1)
            if off != 0 or mode == modes[0]
        )
    else:
        results = [
            _process_offset(mode, off, **_process_kwargs)
            for mode in modes
            for off in range(0, offset + 1)
            if off != 0 or mode == modes[0]
        ]
    return results


def _process_offset(
    mode,
    off,
    session,
    input_df,
    sqrt_spikes,
    spikes_reg_weight,
    alpha,
    simple_maze,
    state_type,
    n_folds,
    normalise_X,
    verbose,
):
    if verbose:
        print(f"processing: {mode}, offset: {off}")
    offset_results = []
    # filter input df for times where mode-offsets are defined
    _input_df = input_df[~input_df[mode][off].isnull()]
    # gather data for decoding
    S = _input_df.spike_count.values  # spikes
    if sqrt_spikes:
        S = np.sqrt(S)
    if spikes_reg_weight is not None:
        assert alpha != "opt", "use spike_reg_weight when not cv optimsing reg across train data"
        S = S * spikes_reg_weight
    N = convert.place_direction2onehot(
        _input_df.place_direction.values, simple_maze=simple_maze
    )  # nusance regressors for current pd
    Y_label = _input_df[mode][off].values  # future/past place/place-direction (what we are predicting from spikes)
    if state_type == "place":
        Y = np.array([x.split("_")[0] for x in Y_label])
    elif state_type == "place_direction":
        Y = Y_label
    else:
        raise ValueError(f"Unknown state type: {state_type}. Must be 'place' or 'place_direction'.")
    feature_set2X = {
        "spatial_spikes": np.concat([N, S], axis=1),  # baseline + spikes
        "spatial": N,  # baseline
    }
    # define cv folds based on trials available in this subsampling of the data
    folds_df = folds.get_folds_df(
        session,
        goal_stratified=False,
        valid_trials=_input_df.trial.unique(),
        n_folds=n_folds,
        return_unique_IDs=True,
    )
    _folds = folds_df.columns.get_level_values(0).unique()
    for fold in _folds:
        fold_df = folds_df[fold]
        train_trials, test_trials = [fold_df[t].unstack().dropna().values for t in ["train", "test"]]
        train_mask, test_mask = [
            _input_df.trial_unique_ID.isin(trials).values for trials in [train_trials, test_trials]
        ]
        Y_train, Y_test = Y[train_mask], Y[test_mask]
        # init results df (contains info rel to sample predictions eg, trial, moving etc.)
        res = _input_df[test_mask].drop(columns=["spike_count", "past", "future"], level=0)
        if alpha == "opt":
            feature_set2alpha = search_reg(fold_df, _input_df, Y, feature_set2X, normalise_X)
        else:
            feature_set2alpha = {label: alpha for label, alpha in feature_set2X.keys()}
        for label, X in feature_set2X.items():
            X_train, X_test = X[train_mask, :], X[test_mask, :]
            if normalise_X:
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
            # fit model
            model = LogisticRegression(
                C=feature_set2alpha[label], random_state=0, max_iter=10_000, class_weight="balanced"
            )
            model.fit(X_train, Y_train)
            # evaluate model
            Y_hat = model.predict(X_test)
            acc = (Y_test == Y_hat).astype(int)
            res[("accuracy", label)] = acc
            res[("fold", "")] = fold
            res[("mode", "")] = mode
            res[("offset", "")] = off
            offset_results.append(res)
    return offset_results


def search_reg(fold_df, _input_df, Y, feature_set2X, normalise_X, reg_range=np.logspace(-4, 4, 10)):
    v_df = fold_df.train
    v_folds = v_df.columns.values
    results = np.zeros((len(v_folds), len(feature_set2X), len(reg_range)))
    for i, v in enumerate(v_folds):
        val_trials = v_df[[col for col in v_folds if col != v]].stack().dropna().values
        test_trials = v_df[v].dropna().values
        val_mask, test_mask = (
            _input_df.trial_unique_ID.isin(val_trials).values,
            _input_df.trial_unique_ID.isin(test_trials).values,
        )
        y_val, y_test = Y[val_mask], Y[test_mask]
        for j, X in enumerate(feature_set2X.values()):
            X_val, X_test = X[val_mask], X[test_mask]
            if normalise_X:
                scaler = StandardScaler()
                scaler.fit(X_val)
                X_val, X_test = scaler.transform(X_val), scaler.transform(X_test)
            for k, alpha in enumerate(reg_range):
                print(f"fold: {i}, feat: {j}, alpha: {alpha}")
                model = LogisticRegression(C=alpha, random_state=0, max_iter=10_000, class_weight="balanced")
                model.fit(X_val, y_val)
                results[i, j, k] = model.score(X_test, y_test)
    opt_alphas = reg_range[results.mean(0).argmax(1)]
    return {label: alpha for label, alpha in zip(feature_set2X.keys(), opt_alphas)}


# %%


def get_session_future_place_decoding(
    session,
    include_multi_units=True,
    max_steps_to_goal=30,
    resolution=0.1,
    mode="future",
    offset=8,
    state_type="place_direction",
    min_spikes=300,
    sqrt_spikes=True,
    n_folds=5,
    normalise_X=True,
    spikes_reg_weight=0.1,
    max_jobs=20,
    verbose=True,
):
    # input data (see get_input_df for details)
    navigation_spikes_df = get_input_df(
        session, include_multi_units, max_steps_to_goal, resolution, mode, offset, state_type, min_spikes
    )
    simple_maze = session.simple_maze()
    # prep data for decoding
    spike_counts = navigation_spikes_df.spike_count.values
    if sqrt_spikes:
        spike_counts = np.sqrt(spike_counts)
    # add current place-direction and goal nuissance regressor array
    PD_1hot = convert.place_direction2onehot(navigation_spikes_df.place_direction.values, simple_maze=simple_maze)
    # target values are the location we are at now, and that we will be at at different points in the past/future
    if mode == "future":
        Y = navigation_spikes_df.future.values
    elif mode == "past":
        Y = navigation_spikes_df.past.values
    #  convert to one-hot for regression
    if state_type == "place":
        Y_1hot = np.array([convert.place2onehot(Y[:, i], simple_maze) for i in range(Y.shape[-1])])
    elif state_type == "place_direction":
        Y_1hot = np.array([convert.place_direction2onehot(Y[:, i], simple_maze) for i in range(Y.shape[-1])])
    else:
        raise ValueError(f"Unknown state type: {state_type}. Must be 'place' or 'place_direction'.")
    if verbose:
        print(
            f"amount of data for the {mode} at different delays:\n{mode}:",
            Y_1hot.sum((-1, -2)),
        )
    # CRITICALLY: filter data for decision points where past and future up to max_offset are available
    decision_points = get_decision_points(simple_maze, mode, return_as="strings", plot=False)
    at_decision_point = np.array([sa in decision_points for sa in navigation_spikes_df.place_direction.values])
    future_and_past_avail = Y_1hot.sum(-1).mean(0) == 1
    keep_inds = np.where(future_and_past_avail & at_decision_point)[0]
    if verbose:
        print("keeping", len(keep_inds), "data points")

    X_spikes = spike_counts[keep_inds, :]  # spike counts for relevant data
    Y_final = Y_1hot[..., keep_inds, :].argmax(-1)  # future location for relevant data
    X_SA = PD_1hot[keep_inds, :]  # state-action regressors for relevant data (can use X_SA or X_SAG)
    trials = navigation_spikes_df.trial.values[keep_inds]  # trial numbers

    # run cv decoding
    # split trial into cv folds
    unique_trials = np.unique(trials)
    trial_splits = [[] for _ in range(n_folds)]
    for trial in unique_trials:
        trial_splits[int(trial) % n_folds].append(trial)

    # data indices corresponding to each fold
    trial_split_inds = [
        np.concatenate([np.where(trials == trial_id)[0] for trial_id in trial_split]) for trial_split in trial_splits
    ]
    if normalise_X:
        X_spikes, X_SA = [(X - X.mean(0)[None, :]) / (1e-10 + X.std(0)[None, :]) for X in [X_spikes, X_SA]]  # normalize

    # try to decode from either just spikes, just state-actions, or both
    possible_Xs = [
        X_spikes,
        X_SA,
        np.concatenate(
            [spikes_reg_weight * X_spikes, X_SA], axis=-1
        ),  # weight spikes to increase effective reg strength w/ more regressors
    ]  # for first one (this worked)
    # run regression for each of these models. Total result shape is (regressions, future vs past, offset, fold)
    results = Parallel(n_jobs=max_jobs)(
        delayed(_process_fold)(Y_final, trial_split_inds, n_folds, fold, X, ishift, mode, label, verbose)
        for X, label in zip(possible_Xs, ["spikes", "place_direction", "spikes_place_direction"])
        for ishift in np.arange(0, offset + 1)
        for fold in range(n_folds)
    )

    return pd.DataFrame(results)


def _process_fold(Y_final, trial_split_inds, n_folds, fold, X, ishift, mode, label, verbose):
    """"""
    if verbose:
        print(f"Processing fold {fold}, mode {mode}, offset {ishift} with {label} regressors")
    y = Y_final[ishift, :]
    # training and test indices
    test, train = trial_split_inds[fold], np.concatenate([trial_split_inds[f] for f in range(n_folds) if f != fold])
    # could do nested crossvalidation to set the regularization strength, but just doing something simple to start
    clf = LogisticRegression(C=1e-0, max_iter=10_000)
    clf.fit(X[train, :], y[train])  # fit the model
    score = clf.score(X[test, :], y[test])
    return {
        "mode": mode,
        "offset": ishift,
        "fold": fold,
        "regressors": label,
        "score": score,
    }


def get_input_df(
    session,
    include_multi_units=True,
    max_steps_to_goal=30,
    resolution=0.1,
    modes=["future", "past"],
    offset=12,
    state_type="place",
    min_spikes=300,
):
    """
    Note slightly hacky way of grabing the future/past states but don't want to change Kris' original code
    """
    # load data
    navigation_df = session.navigation_df
    spike_counts_df = session.navigation_spike_counts_df.reset_index(drop=True)

    # filter clusters
    keep_clusters = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=include_multi_units,
    )
    spike_counts_df = spike_counts_df[spike_counts_df.columns[spike_counts_df.spike_count.columns.isin(keep_clusters)]]

    # downsample data
    ds_nav_df, ds_spikes_df = ds.downsample_nav_spikes_data(
        navigation_df, spike_counts_df, resolution=resolution, distance_metrics=[("steps_to_goal", "future")]
    )
    navigation_spikes_df = pd.concat([ds_nav_df, ds_spikes_df], axis=1)

    # add place_direction column
    navigation_spikes_df[("place_direction", "")] = (
        navigation_spikes_df.maze_position.simple + "_" + navigation_spikes_df.cardinal_movement_direction
    )

    # add future, past state information
    future_past_dfs = []
    for mode in modes:
        future_past_df = get_past_and_future_states(
            navigation_spikes_df, state_type=state_type, past_offset=offset, future_offset=offset
        )
        future_past_dfs.append(future_past_df.xs(mode, axis=1, level=0, drop_level=False))
    navigation_spikes_df = pd.concat([navigation_spikes_df, *future_past_dfs], axis=1)

    # filter data
    navigation_spikes_df = filt.filter_navigation_rates_df(
        navigation_spikes_df,
        navigation_only=True,
        moving_only=False,
        exclude_time_at_goal=True,
        max_steps_to_goal=max_steps_to_goal,
    )
    # filter out clustes with min activity during navigation
    if min_spikes is not None:
        _spikes = navigation_spikes_df.spike_count
        reject_clusters = _spikes.columns[_spikes.sum(axis=0) < min_spikes].values
        navigation_spikes_df = navigation_spikes_df.drop(columns=reject_clusters, level=1, axis=1)

    return navigation_spikes_df


def get_past_and_future_states(
    navigation_spikes_df,
    state_type="place",
    past_offset=6,
    future_offset=6,
):
    """ """
    future_offsets = np.arange(1, future_offset + 1)
    past_offsets = np.arange(1, past_offset + 1)

    if state_type == "place":
        all_states = navigation_spikes_df.maze_position.simple.values
    elif state_type == "place_direction":
        all_states = navigation_spikes_df.place_direction.values
    else:
        raise ValueError(f"Unknown state type: {state_type}. Must be 'place' or 'place_direction'.")

    # only keep data from navigation
    phases = navigation_spikes_df.trial_phase.values
    all_states[phases != "navigation"] = None  # remove data not during navigation

    # also remove any data where the mouse is at the goal location
    all_trial_nums = navigation_spikes_df.trial.unique()
    for trial in all_trial_nums[~np.isnan(all_trial_nums)]:
        trial_inds = np.where(navigation_spikes_df.trial == trial)[0]
        goal_inds = trial_inds[
            (
                navigation_spikes_df.maze_position.simple.values[trial_inds]
                == navigation_spikes_df.goal.values[trial_inds[0]]
            )
        ]
        all_states[goal_inds] = None

    # find the df indices corresponding to every time the mouse moves to a different state
    boundaries = np.concatenate(
        [np.zeros(1).astype(int), np.where(all_states[1:] != all_states[:-1])[0] + 1]
    )  # first index in each state

    # also instantiate an array in which we will store current, future, and past states
    # this will have shape (2, max_offset+1, num_datapoints)
    # it will store the state (tower or bridge) that the mouse will be in after (0,1,...,num_offset) actions, and (0,1, ..., num_offset) actions before
    offset_array = np.array(
        [
            np.vstack(
                [copy.deepcopy(all_states)]
                + [np.array([None for _ in range(len(all_states))]) for _ in range((past_offset + future_offset))]
            )
            for _ in range(2)
        ]
    )  # future and past

    # populate this array
    for itype, (type_, offsets) in enumerate(
        zip(["future", "past"], [future_offsets, past_offsets])
    ):  # first consider future, then past states
        sign = +1 if type_ == "future" else -1  # the sign of our offset
        # boundaries between states going either forwards (to compute future states) or backwards (to compute past)
        dir_boundaries = (
            boundaries if (type_ == "future") else np.flip(boundaries) - 1
        )  # go either forwards or backwards
        for i_off, offset in enumerate(offsets):  # for every offset we are interested in
            ref = offset_array[
                itype, offset - 1, :
            ]  # what is the state at the previous offset (e.g. what is the previous state if we now want the state two before)
            next_ = offset_array[
                itype, offset, :
            ]  # the array we're trying to populate corresponding to the current offset
            for i_b, b in enumerate(dir_boundaries[:-1]):  # for every state transition
                inds = np.arange(b, dir_boundaries[i_b + 1], sign)  # indices of the df  where I was in that state
                cur_state = ref[inds[0]]  # where will I be in 'offset-1' steps
                next_state = ref[inds[-1] + sign]  # where will I be in 'offset' steps

                if None in [cur_state, next_state]:  # if we're at a trial boundary
                    next_[inds] = None  # don't have a next state
                else:
                    assert cur_state != next_state  # make sure that we have actually moved
                    next_[inds] = next_state  # store the next state
    output_df = pd.DataFrame(index=navigation_spikes_df.index)  # create a new dataframe to store the results
    for offset in [0] + list(future_offsets):  # make this data part of the big dataframe
        output_df[("future", offset)] = offset_array[0, offset, :]
    for offset in [0] + list(past_offsets):  # make this data part of the big dataframe
        output_df[("past", offset)] = offset_array[1, offset, :]
    # convert to multiindex
    output_df.columns = pd.MultiIndex.from_tuples(output_df.columns)
    return output_df


def get_decision_points(simple_maze, mode="future", return_as="strings", plot=False):
    """
    Computes and returns the set of decision point identifiers in a given maze.

    A decision point is defined as a position (or an intermediate bridge) in the maze from which
    a move in a specific cardinal direction (North, South, East, West) leads to a node having three
    or more neighbors. The identifiers are generated by concatenating a label (from the maze's coordinate-label
    mapping) with an underscore and the corresponding direction.

    Returns:
        A set of strings, each representing a decision point in the maze. Each string is formed by combining
        the label (either of a node or an intermediate position along an edge) with the direction taken from that node.
    """
    coord2label = coord2label = mr.get_maze_coord2label(simple_maze)
    deltas = {(1, 0): "E", (-1, 0): "W", (0, 1): "N", (0, -1): "S"}  # mapping from action vector to direction
    decision_points = set()
    for node1 in simple_maze.nodes:
        for node2 in simple_maze.neighbors(node1):
            check_node = node2 if mode == "future" else node1
            other_node = node1 if mode == "future" else node2
            if len(list(simple_maze.neighbors(check_node))) >= 3:
                dir_ = deltas[tuple(np.array(node2) - np.array(node1))]
                decision_points.add(
                    (coord2label[other_node], dir_)
                )  # going in this direction from node1 yields a decision point
                try:
                    edge = coord2label[(node1, node2)]
                except:
                    edge = coord2label[(node2, node1)]
                decision_points.add((edge, dir_))
    if plot:
        dps = pd.Series(index=pd.MultiIndex.from_tuples(list(decision_points)), data=1)
        mp.plot_directed_heatmap(simple_maze, dps, colormap="Greys", colorbar=False)
    if return_as == "tuples":
        return decision_points
    elif return_as == "strings":
        return {f"{label}_{dir_}" for label, dir_ in decision_points}
    else:
        raise ValueError(f"Unknown return_as type: {return_as}. Must be 'tuples' or 'strings'.")
