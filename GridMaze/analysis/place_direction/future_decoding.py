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
    return


# %%


def plot_future_decoding_summary(summary_df, decision_points="future", steps_to_goal=None, plot_as="diff", ax=None):
    """ """
    # filter for decision points
    if decision_points:
        # filter decoded samples for only those at decision points where future is less predicted
        # by current location
        dfs = []
        for maze_name in ["maze_1", "maze_2"]:
            maze_df = summary_df[summary_df.maze_name == maze_name]
            simple_maze = mr.get_simple_maze(maze_name)
            if decision_points == "future":
                decision_points = get_decision_points(
                    simple_maze, mode="future", edges_only=True, node_only=False, return_as="strings", plot=False
                )
            elif decision_points == "past":
                decision_points = get_decision_points(
                    simple_maze, mode="past", edges_only=False, node_only=True, return_as="strings", plot=False
                )
            dfs.append(maze_df[maze_df.place_direction.isin(decision_points)])
        df = pd.concat(dfs, axis=0)
    # filter for steps to goal
    if steps_to_goal is not None:
        # update steps to goal
        df[("steps_to_goal", "future")] = df.steps_to_goal.future.astype(int)
        df = df[df.steps_to_goal.future.between(*steps_to_goal)]
    # process for plotting
    subject_means = df.groupby(["subject_ID", "mode", "offset"]).accuracy.mean().accuracy
    # plot
    if plot_as == "diff":
        _plot_decoding_diff(subject_means, ax=ax)
    elif plot_as == "raw":
        _plot_decoding_raw(subject_means, ax=ax)
    else:
        raise ValueError(f"Unknown plot_as: {plot_as}. Must be 'diff' or 'raw'.")


def _plot_decoding_diff(subject_means, colors=["hotpink", "blueviolet"], ax=None):
    # set up fig
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("steps in future/past")
    ax.set_ylabel("decoding acc. \n (chance normalised)")
    # process
    diff = (subject_means.spatial_spikes - subject_means.spatial).unstack(level=0).T
    grand_mean = diff.mean()
    grand_sem = diff.sem()
    # plot
    for mode, color in zip(["past", "future"], colors):
        mean = grand_mean[mode].values
        sem = grand_sem[mode].values
        x_vals = grand_mean[mode].index.values
        if mode == "past":
            x_vals = -1 * x_vals
        ax.errorbar(
            x_vals,
            mean,
            yerr=sem,
            marker="o",
            linestyle=None,
            color=color,
            linewidth=2,
            elinewidth=2,
            capsize=0,
            markersize=6,
        )


def _plot_decoding_raw(subject_means, colors=[("hotpink", "mediumvioletred"), ("blueviolet", "indigo")], ax=None):
    # set up fig
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("steps in future/past")
    ax.set_ylabel("decoding acc.")
    # process
    grouped = subject_means.groupby(level=[1, 2])
    grand_mean = grouped.mean()
    grand_sem = grouped.sem()
    # plot
    for mode, mode_colors in zip(["past", "future"], colors):
        for fs, color in zip(["spatial_spikes", "spatial"], mode_colors):
            mean = grand_mean.loc[mode, fs].values
            sem = grand_sem.loc[mode, fs].values
            x_vals = grand_mean.loc[mode, fs].index.values
            if mode == "past":
                x_vals = -1 * x_vals
            ax.plot(x_vals, mean, color=color, label=f"{fs} ({mode})", lw=1.5)
            ax.fill_between(x_vals, mean - sem, mean + sem, color=color, alpha=0.2)
    ax.legend(fontsize=8)


def get_place_decoding_summary(
    offset=12,
    subjects="all",
    maze_names=["maze_1", "maze_2"],
    days_on_maze="late",
    save=False,
    verbose=False,
):
    """ """
    save_path = RESULTS_DIR / f"place_decoding_summary2.parquet"
    if not save and save_path.exists():
        if verbose:
            print(f"Loading existing results from {save_path}")
        return pd.read_parquet(save_path)
    if verbose:
        print("Loading sessions ...")
    sessions = gs.get_maze_sessions(
        subject_IDs=subjects,
        maze_names=maze_names,
        days_on_maze=days_on_maze,
        with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "trials_df"],
        must_have_data=True,
    )
    dfs, failed_sessions = [], []
    for session in sessions:
        if verbose:
            print(session.name)
        try:
            results_df = get_session_future_place_direction_decoding(session, offset=offset)  # defualt settings
            results_df[("subject_ID", "")] = session.subject_ID
            results_df[("maze_name", "")] = session.maze_name
            results_df[("day_on_maze", "")] = session.day_on_maze
            dfs.append(results_df)
        except Exception as e:
            print(f"Error processing session {session.name}: {e}")
            failed_sessions.append(session.name)
    summary_df = pd.concat(dfs, axis=0)
    if save:
        summary_df.to_parquet(save_path)
        if verbose:
            print(f"Saving results to {save_path}")
    return summary_df, failed_sessions


# %% Dev new core decoding function


def get_session_future_place_direction_decoding(
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
    alpha=1,
    normalise_X=True,
    spikes_reg_weight=0.1,
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
            delayed(_process_offset)(mode, off, **_process_kwargs) for mode in modes for off in range(1, offset + 1)
        )
    else:
        results = [_process_offset(mode, off, **_process_kwargs) for mode in modes for off in range(1, offset + 1)]
    _results = []
    for res in results:
        _results.extend(res)
    results_df = pd.concat(_results, axis=0, ignore_index=True)
    return results_df


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
        if verbose:
            print(f"  fold {fold}")
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
            feature_set2alpha = {label: alpha for label in feature_set2X.keys()}
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
                model = LogisticRegression(C=alpha, random_state=0, max_iter=10_000, class_weight="balanced")
                model.fit(X_val, y_val)
                results[i, j, k] = model.score(X_test, y_test)
    opt_alphas = reg_range[results.mean(0).argmax(1)]
    return {label: alpha for label, alpha in zip(feature_set2X.keys(), opt_alphas)}


# %%


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


def get_decision_points(
    simple_maze, mode="future", edges_only=False, node_only=False, return_as="strings", plot=False, ax=None
):
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
                if not edges_only:
                    decision_points.add(
                        (coord2label[other_node], dir_)
                    )  # going in this direction from node1 yields a decision point
                if not node_only:
                    try:
                        edge = coord2label[(node1, node2)]
                    except:
                        edge = coord2label[(node2, node1)]
                    decision_points.add((edge, dir_))
    if plot:
        dps = pd.Series(index=pd.MultiIndex.from_tuples(list(decision_points)), data=1)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        mp.plot_directed_heatmap(simple_maze, dps, colormap="Greys", colorbar=False, ax=ax)
    if return_as == "tuples":
        return decision_points
    elif return_as == "strings":
        return {f"{label}_{dir_}" for label, dir_ in decision_points}
    else:
        raise ValueError(f"Unknown return_as type: {return_as}. Must be 'tuples' or 'strings'.")
