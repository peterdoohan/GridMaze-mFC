"""
Refactoring goal_decoding.py / gd2.py to include a separate utils supporing lib as code bases was growing too large
"""

# %% Imports
import json
from turtle import update
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt


from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import convert
from GridMaze.maze import representations as mr
from scipy.spatial.distance import euclidean


# %% Global Variables

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "distance_to_goal"


with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

FRAME_RATE = 60

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]
GOAL_SETS = ["subset_1", "subset_2", "all"]

# %% results crunching functions


def _get_transformed_steps_to_goal(session, results_df, event, round_steps=False):
    """
    Translates between time aliged to cue or reward and steps to goal, based on that
    subjects average distance across all relevant maze and goal subset sessions.
    """
    assert "timepoint" in results_df.columns, "results_df does not contain timepoint column"
    window2step = get_step_time_transformation(session, event)
    transformed_steps_to_goal = results_df.timepoint.map(window2step).astype(int)
    if round_steps:
        transformed_steps_to_goal = transformed_steps_to_goal.round().astype(int)
    return transformed_steps_to_goal


def decoding_accuracy_df(results_df, decoding_type="goal", alignment="timepoint"):
    """
    alignment: "timepoint" or "steps_to_goal"
    """
    idx = results_df.groupby(["trial_unique_ID", alignment])[f"predicted_{decoding_type}_prob"].idxmax()
    df = results_df.loc[idx]
    df["test_acc"] = (df[f"true_{decoding_type}"] == df[f"predicted_{decoding_type}"]).astype(int)
    return df.reset_index(drop=True)


def get_expected_distance_error_df(results_df, simple_maze, decoding_type="goal", alignment="timepoint"):
    """
    use distance cals
    """
    # check decoding_type matches results df
    _check_decoding_type(results_df, decoding_type)
    # add colums for distance to goal (geo or euc) for every true and predicted place/goal pair
    results_df = _add_distance_cols(results_df, simple_maze, decoding_type, round_euc=False)
    # calc expected distance error
    results_df["geo_weight_prob"] = results_df[f"predicted_{decoding_type}_prob"] * results_df["geo_dist"]
    results_df["euc_weight_prob"] = results_df[f"predicted_{decoding_type}_prob"] * results_df["euc_dist"]
    # EDE (expected distance error)
    trial_EDE = results_df.groupby(["trial_unique_ID", alignment])[["geo_weight_prob", "euc_weight_prob"]].sum()
    av_EDE = trial_EDE.groupby(alignment).mean()
    av_EDE.columns = ["geodesic", "euclidean"]
    return av_EDE


def get_decoding_probability_mass_df(results_df, simple_maze, decoding_type="goal", return_trial_av=True):
    """
    decoding_type in ["goal", "place"]
    """
    # check decoding_type matches results df
    _check_decoding_type(results_df, decoding_type)
    # add colums for distance to goal (geo or euc) for every true and predicted place/goal pair
    results_df = _add_distance_cols(results_df, simple_maze, decoding_type, round_euc=True)
    # calc prob mass over distances fro predictions
    prob_mass_curves = []
    for dist in ["euc_dist", "geo_dist"]:
        prob_mass_trial = results_df.groupby(["trial_unique_ID", dist]).predicted_goal_prob.mean().unstack()
        if return_trial_av:
            prob_mass_trial.columns = pd.MultiIndex.from_product([[dist], prob_mass_trial.columns])
            prob_mass_curves.append(prob_mass_trial)
        else:
            prob_mass_curves.append(prob_mass_trial.mean())
    if return_trial_av:
        return pd.concat(prob_mass_curves, axis=1)
    else:
        dist_prob_mass = pd.concat(prob_mass_curves, axis=1)
        dist_prob_mass.columns = ["euc_dist", "geo_dist"]
        return dist_prob_mass


def _check_decoding_type(results_df, decoding_type):
    """ """
    if decoding_type == "goal":
        assert "predicted_goal" in results_df.columns, "results_df does not contain goal decoding"
    elif decoding_type == "place":
        assert "predicted_place" in results_df.columns, "results_df does not contain place decoding"
    else:
        raise ValueError(f"Unknown decoding type {decoding_type}")


def _add_distance_cols(results_df, simple_maze, decoding_type, round_euc=False):
    """ """
    # precompute step distances (tower-->edge = 1 step)
    extended_maze = mr.get_extended_simple_maze(simple_maze)
    geo_dist = dict(nx.all_pairs_dijkstra_path_length(extended_maze, weight="weight"))
    label2coord = mr.get_maze_label2coord(simple_maze)
    all_coords = list(label2coord.values())
    euc_dist = {}
    for c1 in all_coords:
        _euc_dist = {}
        for c2 in all_coords:  # adapt edge coods eg, ((6,5), (6,6), to (6, 5.5) by taking mean
            _c1 = tuple(np.array(c1).mean(axis=0)) if isinstance(c1[0], tuple) else c1
            _c2 = tuple(np.array(c2).mean(axis=0)) if isinstance(c2[0], tuple) else c2
            _euc_dist[c2] = euclidean(_c1, _c2) * 2
        euc_dist[c1] = _euc_dist
    # calc geo or euc distance from every goal to every possible predicted goal/place
    true_coords = results_df[f"true_{decoding_type}"].map(label2coord)
    pred_coords = results_df[f"predicted_{decoding_type}"].map(label2coord)
    results_df["geo_dist"] = [geo_dist[coord1][coord2] for coord1, coord2 in zip(true_coords, pred_coords)]
    results_df["euc_dist"] = [euc_dist[coord1][coord2] for coord1, coord2 in zip(true_coords, pred_coords)]
    if round_euc:
        results_df["euc_dist"] = results_df["euc_dist"].round().astype(int)  # round euclidean to nearest step
    return results_df


# %% get sessions


def get_sessions_for_analysis(subject_IDs, maze_names, goal_subsets):
    """ """
    days_on_maze = "late" if "all" in goal_subsets else "all"
    sessions = gs.get_maze_sessions(
        subject_IDs=subject_IDs,
        maze_names=maze_names,
        days_on_maze="all",
        goal_subsets=goal_subsets,
        with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "trials_df"],
        must_have_data=True,
    )
    sessions = [sessions] if isinstance(sessions, gs.MazeSession) else sessions
    # check sessions have at least 2 trials per goal
    keep_sessions = []
    for session in sessions:
        trials_df = session.trials_df
        if trials_df.groupby("goal").trial.count().ge(2).all():
            keep_sessions.append(session)
    if len(keep_sessions) == 0:
        raise FileNotFoundError(f"No valid session for: {subject_IDs}, {maze_names}, {goal_subsets}")
    return keep_sessions


# %% get distance aligned input data


def get_distance_aligned_input_data(
    session,
    resolution=0.5,
    include_multi_units=True,
    include_trial_phases=["navigation"],
    ignore_last_n=0,
    binning_method="uniform",
    n_bins=25,
    max_steps_to_goal=25,
    include_place_onehots=False,
):
    """
    Returns a dataframe with spike counts aligned to future path-distance to goal over all trials in a session.
    """
    # load data
    trials_df = session.trials_df
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
    ds_nav_info, ds_spike_counts_df = _downsample_data(navigation_df, spike_counts_df, resolution)
    # update distances in non_navigation trial periods
    if "ITI" in include_trial_phases or "reward_consumption" in include_trial_phases:
        ds_nav_info[("steps_to_goal", "future")] = _add_non_nav_distances(
            session, ds_nav_info, ignore_last_n=ignore_last_n
        )
    # add distance bins info
    bins = convert._get_distance_bins(
        binning_method=binning_method,
        n_distance_bins=n_bins,
        distance_metrics=("steps_to_goal", "future"),
        max_distance=max_steps_to_goal,
    )
    ds_nav_info[("steps_to_goal", "bin")] = pd.cut(ds_nav_info.steps_to_goal.future, bins=bins)
    ds_nav_info[("steps_to_goal", "bin_mid")] = ds_nav_info.steps_to_goal.bin.apply(lambda x: x.mid).astype(float)
    # add event aligned time info (for later cross-decoder comparisons)
    ds_nav_info[("event_aligned_time", "cue")] = _get_event_aligned_times(ds_nav_info, trials_df, "cue")
    ds_nav_info[("event_aligned_time", "reward")] = _get_event_aligned_times(ds_nav_info, trials_df, "reward")
    # combine and filter out points where distance is not defined
    ds_nav_rates_df = pd.concat([ds_nav_info, ds_spike_counts_df], axis=1)
    ds_nav_rates_df = ds_nav_rates_df[~ds_nav_rates_df.steps_to_goal.bin.isna()]
    ds_nav_rates_df = ds_nav_rates_df[ds_nav_rates_df.trial_phase.isin(include_trial_phases)]
    ds_nav_rates_df[("trial", "")] = ds_nav_rates_df[("trial", "")].astype(int)
    # include position inputs for decoder (optional)
    if include_place_onehots:
        place_onehots_df = _get_place_onehots_df(session, ds_nav_rates_df)
        ds_nav_rates_df = pd.concat([ds_nav_rates_df, place_onehots_df], axis=1)
    return ds_nav_rates_df


def _add_non_nav_distances(session, nav_info_df, ignore_last_n=2):
    """ """
    # load additional data
    trials_df = session.trials_df
    trials_df = trials_df.set_index(("trial", ""))
    last_trial = trials_df.index.max()
    next_goals = trials_df[("goal", "")].shift(-1)
    # precompute distances
    simple_maze = session.simple_maze()
    extended = mr.get_extended_simple_maze(simple_maze)
    raw_dist = dict(nx.all_pairs_dijkstra_path_length(extended, weight="weight"))
    label2coord = mr.get_maze_label2coord(simple_maze)
    # Build dense N×N matrix for fast look ups
    coords = list(label2coord.values())
    coord2idx = {coord: i for i, coord in enumerate(coords)}
    N = len(coords)
    dist_mat = np.zeros((N, N), float)
    for src_coord, targets in raw_dist.items():
        i = coord2idx[src_coord]
        for dst_coord, d in targets.items():
            dist_mat[i, coord2idx[dst_coord]] = d
    # rename cols for convience
    nav = nav_info_df.copy().sort_index(axis=1)
    nav["trial"] = nav[("trial", "")]
    nav["phase"] = nav[("trial_phase", "")]
    nav["future_dist"] = nav[("steps_to_goal", "future")]
    nav["pos_label"] = nav[("maze_position", "simple")]
    nav["next_goal"] = nav["trial"].map(next_goals)
    nav["window_idx"] = nav.groupby("trial").cumcount()
    nav["n_windows"] = nav.groupby("trial")["trial"].transform("count")
    # optionally ignore last n windows of each trial
    valid = (
        nav["future_dist"].isna()
        & nav["phase"].isin(["reward_consumption", "ITI"])
        & (nav["window_idx"] < nav["n_windows"] - ignore_last_n)
        & (nav["trial"] != last_trial)
    )
    # compute distances
    src_idxs = nav.loc[valid, "pos_label"].map(label2coord).map(coord2idx).to_numpy()
    dst_idxs = nav.loc[valid, "next_goal"].map(label2coord).map(coord2idx).to_numpy()
    computed = dist_mat[src_idxs, dst_idxs]
    # update distances and return
    out = nav_info_df[("steps_to_goal", "future")].copy()
    out.loc[valid.values] = computed
    return out


def _get_place_onehots_df(session, nav_rates_df):
    """ """
    simple_maze = session.simple_maze()
    positions = mr.get_maze_locations(simple_maze)
    # convert position labels to one-hot encoding
    place_onehots = convert.place2onehot(nav_rates_df.maze_position.simple.values, session.simple_maze())
    place_onehots_df = pd.DataFrame(
        data=place_onehots.astype(int),
        index=nav_rates_df.index,
        columns=pd.MultiIndex.from_product([["place_onehot"], positions]),
    )
    return place_onehots_df


def _get_event_aligned_times(nav_info, trials_df, event):
    """
    Returns a series of time relative to event on every trial.
    """
    trials2event_time = trials_df.set_index("trial")["time"][event]
    trial_ids = nav_info["trial"].astype("Int64")
    event_times = trial_ids.map(trials2event_time)
    return nav_info["time"] - event_times


# %%


def get_place_decoding_input_data(
    session,
    resolution=0.5,
    include_multi_units=True,
    window=(-10, 10),
):
    """
    Simpler version of other input_data functions which just returns downsampled navigation
    data but with info relevant for place decoding analyses
    """
    # load data
    session_info = session.session_info
    trials_df = session.trials_df
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
    nav_info, spike_counts_df = _downsample_data(navigation_df, spike_counts_df, resolution)
    # update trial definitions to start in previous ITI
    nav_info["trial"] = update_trial_ID(nav_info, trials_df)
    nav_info["trial_unique_ID"] = convert.trial2trial_unique_ID(session_info, nav_info["trial"])
    # add event aligned time
    nav_info[("event_aligned_time", "cue")] = _get_event_aligned_times(nav_info, trials_df, "cue")
    nav_info[("event_aligned_time", "reward")] = _get_event_aligned_times(nav_info, trials_df, "reward")
    # bin event aligned time and report bin_mids for convience
    bins = pd.IntervalIndex.from_breaks(np.arange(window[0], window[1] + resolution, resolution), closed="right")
    cue_aligned_bins = pd.cut(nav_info[("event_aligned_time", "cue")], bins=bins)
    reward_aligned_bins = pd.cut(nav_info[("event_aligned_time", "reward")], bins=bins)
    nav_info[("event_aligned_bin", "cue")] = cue_aligned_bins.apply(lambda x: x.mid).astype(float)
    nav_info[("event_aligned_bin", "reward")] = reward_aligned_bins.apply(lambda x: x.mid).astype(float)
    # combine and remove out of trial times
    nav_rates_df = pd.concat([nav_info, spike_counts_df], axis=1)
    nav_rates_df = nav_rates_df[~nav_rates_df.trial.isna()]
    return nav_rates_df


def update_trial_ID(
    nav_info,
    trials_df,
):
    """
    Vectorized assignment of ITI-period rows to the *next* trial ID,
    with all other rows keeping their own trial ID (and non-string
    phases becoming NaN). Returns a NumPy array of the updated IDs.
    """
    trial = nav_info.trial
    trial_phase = nav_info.trial_phase

    # compute some masks
    is_def = trial_phase.map(lambda x: isinstance(x, str))
    is_iti = is_def & (trial_phase == "ITI")
    next_trial = trial + 1
    within_bounds = next_trial <= trials_df.trial.max()

    updated = np.where(~is_def, np.nan, np.where(is_iti, np.where(within_bounds, next_trial, np.nan), trial))

    return pd.Series(updated)


# %% event aligned input data


def get_event_aligned_input_data(
    session,
    event="cue",
    resolution=0.5,
    window=(-10, 10),
    include_multi_units=True,
    binning_method="uniform",
    n_bins=25,
    max_steps_to_goal=25,
    include_place_onehots=False,
):
    """
    Returns a dataframe with spike counts aligned to event (cue & reward) times.
    """
    # load data
    simple_maze = session.simple_maze()
    session_info = session.session_info
    trials_df = session.trials_df
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
    # get rates aligned to event
    frames_before, frames_after = int(window[0] * FRAME_RATE), int(window[1] * FRAME_RATE)
    event_times = trials_df.set_index("trial").time[event]
    trial2goal = trials_df.set_index("trial").goal
    # precalculate distances to goal
    extended_simple_maze = mr.get_extended_simple_maze(simple_maze)
    path_distances = dict(nx.all_pairs_dijkstra_path_length(extended_simple_maze, weight="weight"))
    label2coord = mr.get_maze_label2coord(simple_maze)
    nav_info_dfs, spike_count_dfs = [], []
    for trial, event_time in event_times.items():
        event_frame = (navigation_df.time - event_time).abs().argmin()
        nav_aligned_df = navigation_df.iloc[event_frame + frames_before : event_frame + frames_after].reset_index(
            drop=True
        )
        spikes_aligned_df = spike_counts_df.iloc[event_frame + frames_before : event_frame + frames_after].reset_index(
            drop=True
        )
        # downsample to speficied resolution
        ds_nav_aligned_df, ds_spikes_aligned_df = _downsample_data(nav_aligned_df, spikes_aligned_df, resolution)
        # add event aligned time info
        timepoints = np.arange(window[0], window[1], resolution)
        if len(timepoints) > ds_nav_aligned_df.shape[0]:
            # can happen for last trial in session (no more frames)
            timepoints = timepoints[: ds_nav_aligned_df.shape[0]]
        ds_nav_aligned_df[("event_aligned_time", event)] = timepoints
        # update distnace outside navigation where they are not defined (use shortest path
        # upcoming goal (event=cue) or shortest path to just visted goal (event=reward))
        ds_nav_aligned_df[("goal", "")] = trial2goal[trial]
        outside_trial_mask = (ds_nav_aligned_df.trial != trial) | (ds_nav_aligned_df.trial_phase != "navigation")
        pos_coords = ds_nav_aligned_df.loc[outside_trial_mask, ("maze_position", "simple")].map(label2coord)
        goal_coords = ds_nav_aligned_df.loc[outside_trial_mask, ("goal", "")].map(label2coord)
        ds_nav_aligned_df.loc[outside_trial_mask, ("steps_to_goal", "future")] = [
            path_distances[src][dst] for src, dst in zip(pos_coords, goal_coords)
        ]
        # update trial info so it is consistent across all aligned times
        ds_nav_aligned_df[("trial", "")] = trial
        ds_nav_aligned_df[("trial_unique_ID", "")] = convert.trial2trial_unique_ID(session_info, trial)
        nav_info_dfs.append(ds_nav_aligned_df)
        spike_count_dfs.append(ds_spikes_aligned_df)
    # combine over trials
    nav_info_df = pd.concat(nav_info_dfs, axis=0).reset_index(drop=True)
    spike_count_df = pd.concat(spike_count_dfs, axis=0).reset_index(drop=True)
    # add distance bins info
    bins = convert._get_distance_bins(
        binning_method=binning_method,
        n_distance_bins=n_bins,
        distance_metrics=("steps_to_goal", "future"),
        max_distance=max_steps_to_goal,
    )
    nav_info_df[("steps_to_goal", "bin")] = pd.cut(nav_info_df.steps_to_goal.future, bins=bins)
    nav_info_df[("steps_to_goal", "bin_mid")] = nav_info_df.steps_to_goal.bin.apply(lambda x: x.mid).astype(float)
    # combine nav_info and spike counts
    event_aligned_nav_rates_df = pd.concat([nav_info_df, spike_count_df], axis=1)
    if include_place_onehots:
        place_onehots_df = _get_place_onehots_df(session, event_aligned_nav_rates_df)
        # combine with other data (nav_info and spike_counts)
        event_aligned_nav_rates_df = pd.concat([event_aligned_nav_rates_df, place_onehots_df], axis=1)
    return event_aligned_nav_rates_df


def _downsample_data(navigation_df, spike_counts_df, resolution=0.2):
    """ """
    # downsample spike counts by suming spikes within resolution window
    ds_frames = int(FRAME_RATE * resolution)
    ds_spike_counts_df = spike_counts_df.groupby(spike_counts_df.index // ds_frames).sum().reset_index(drop=True)
    # keep only relevant navigation info
    nav_info = navigation_df[
        [
            ("time", ""),
            ("trial_unique_ID", ""),
            ("trial", ""),
            ("goal", ""),
            ("trial_phase", ""),
            ("maze_position", "simple"),
            ("steps_to_goal", "future"),
        ]
    ]
    # downsample navigation info by taking values in mid window
    mid_window_inds = (spike_counts_df.index // ds_frames).unique() * ds_frames + (ds_frames // 2)
    mid_window_inds = mid_window_inds[mid_window_inds < len(nav_info)]
    nav_info = nav_info.iloc[mid_window_inds]
    # account for differences in ds methods
    nav_info.reset_index(drop=True, inplace=True)
    if nav_info.shape[0] < ds_spike_counts_df.shape[0]:
        ds_spike_counts_df = ds_spike_counts_df.iloc[:-1]
    return nav_info, ds_spike_counts_df


# %% Cross valdiation functions


def get_folds_df(session, goal_stratified=True, valid_trials=None, return_unique_IDs=True, n_test_trials=None):
    """ """
    if goal_stratified:
        folds_df = _get_folds_goal_stratified(session, valid_trials, return_unique_IDs)
    else:
        folds_df = _get_folds_non_stratified(session, valid_trials, n_test_trials, return_unique_IDs)
    return folds_df


def _get_folds_goal_stratified(session, valid_trials=None, return_unique_IDs=True):
    """ """
    goals_df = get_goals_df(session, valid_trials, return_unique_IDs)
    # check there are at least 2 trials per goal (needed for test/train split)
    valid_goals_df = goals_df[goals_df.count(axis=1).ge(2)]
    # shuffle
    valid_goals_df = valid_goals_df.apply(lambda x: np.random.choice(x, size=len(x), replace=False), axis=1).apply(
        pd.Series
    )
    # split into test and train folds
    cols = valid_goals_df.columns
    fold_dfs = []
    for i in cols:
        fold = f"fold_{i}"
        test_df = pd.DataFrame(valid_goals_df[cols[i]])
        test_df.columns = pd.MultiIndex.from_product([[fold], ["test"], test_df.columns])
        train_df = valid_goals_df.drop(columns=cols[i])
        train_df.columns = pd.MultiIndex.from_product([[fold], ["train"], train_df.columns])
        fold_dfs.append(pd.concat([test_df, train_df], axis=1))
    # return as df
    folds_df = pd.concat(fold_dfs, axis=1)
    return folds_df


def get_goals_df(session, valid_trials=None, return_unique_IDs=True):
    """
    returns df with goals in index and corresponding session trials in columns
    """
    trials_df = session.trials_df
    if valid_trials is not None:
        trials_df = trials_df[trials_df.trial.isin(valid_trials)].reset_index(drop=True)
    goal2trials = {}
    for goal in session.goals:
        goal2trials[goal] = trials_df[trials_df.goal == goal].trial.to_list()
    goals_df = pd.DataFrame.from_dict(goal2trials, orient="index")
    if return_unique_IDs:
        session_info = session.session_info
        goals_df = goals_df.apply(lambda x: convert.trial2trial_unique_ID(session_info, x))
    return goals_df


def _get_folds_non_stratified(
    session,
    valid_trials=None,
    n_test_trials=5,
    return_unique_IDs=True,
):
    """
    No goal stratified validation folds. Instead specify how many trials per fold.
    Can be useful for all goals sessions where you want more training data per fold.
    """
    assert n_test_trials is not None, "n_test_trials must be specified for non goal-stratified validation folds"
    session_info = session.session_info
    trials_df = session.trials_df
    trials = trials_df.trial.values if valid_trials is None else valid_trials
    # shuffle trials
    trials = np.random.choice(trials, size=len(trials), replace=False)
    fold_dfs = []
    for fold, i in enumerate(range(0, len(trials), n_test_trials)):
        test_trials = trials[i : i + n_test_trials]
        train_trials = np.concatenate([trials[:i], trials[i + n_test_trials :]])
        fold_df = pd.DataFrame(
            {
                "test": pd.Series(test_trials),
                "train": pd.Series(train_trials),
            }
        )
        fold_df.columns = pd.MultiIndex.from_product([[f"fold_{fold}"], fold_df.columns])
        fold_dfs.append(fold_df)
    folds_df = pd.concat(fold_dfs, axis=1)
    if return_unique_IDs:
        folds_df = folds_df.apply(lambda x: convert.trial2trial_unique_ID(session_info, x))
    return folds_df


# %% Transform between distance and event aligned time


def plot_steps_vs_time_curves(step_time_df, event="reward"):
    """ """
    f, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)
    for i, goal_subset in enumerate(GOAL_SETS):
        for j, maze_name in enumerate(MAZE_NAMES):
            ax = axes[i, j]
            df = step_time_df.query(f"goal_subset == '{goal_subset}' and maze == '{maze_name}' and event == '{event}'")
            grouped_df = df.groupby("event_aligned_time").steps_to_goal
            mean = grouped_df.mean()
            sem = grouped_df.sem()
            ax.plot(mean.index, mean)
            ax.fill_between(mean.index, mean - sem, mean + sem, alpha=0.2)
            ax.set_title(f"{goal_subset} {maze_name}")
            ax.set_xlabel("Steps to goal")
            ax.set_ylabel("Event-aligned time (s)")
    f.tight_layout()
    return


def get_step_time_transformation(session, event):
    """ """
    step_time_df = get_step_time_transformation_df()
    df = step_time_df.query(
        f"subject == '{session.subject_ID}' and goal_subset == '{session.goal_subset}' and maze == '{session.maze_name}' and event == '{event}'"
    )
    return df.set_index("event_aligned_time").steps_to_goal


def get_step_time_transformation_df(overwrite=False):
    """ """
    save_path = RESULTS_DIR / "step_time_transformation_df.csv"
    if save_path.exists() and not overwrite:
        return pd.read_csv(save_path, index_col=0)
    else:
        print("Generating step time transformation df")
        dfs = []
        for subject in SUBJECT_IDS:
            print(f"Processing subject {subject}")
            for maze in MAZE_NAMES:
                for goal_subset in GOAL_SETS:
                    for event in ["cue", "reward"]:
                        step_time_df = get_steps_vs_time_curve(subject, maze, goal_subset, event)
                        step_time_df["subject"] = subject
                        step_time_df["maze"] = maze
                        step_time_df["goal_subset"] = goal_subset
                        step_time_df["event"] = event
                        dfs.append(step_time_df)
        step_time_df = pd.concat(dfs).reset_index(drop=True)
        # save
        step_time_df.to_csv(save_path)
        return step_time_df


def get_steps_vs_time_curve(subject, maze, goal_subset, event, max_steps=30):
    sessions = get_sessions_for_analysis(subject_IDs=[subject], maze_names=[maze], goal_subsets=[goal_subset])
    dfs = []
    for session in sessions:
        df = get_event_aligned_input_data(session, event=event, resolution=0.5)
        df = df[[("event_aligned_time", event), ("steps_to_goal", "future")]]
        dfs.append(df)
    step_time_df = pd.concat(dfs).reset_index(drop=True).droplevel(1, axis=1)
    step_time_curve = step_time_df.groupby("event_aligned_time").steps_to_goal.mean()
    return step_time_curve[step_time_curve.index <= max_steps + 1].reset_index()
