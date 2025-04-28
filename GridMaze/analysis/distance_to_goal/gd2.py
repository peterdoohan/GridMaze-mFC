"""
Library for distance-to-goal alaigned goal decoding.
Eg, build separate decoders for neural activity 1, step from goal, 2 steps from goal, etc.
@peterdoohan
"""

# %% Imports
from curses import window
import json
from math import nan
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler


from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import convert
from GridMaze.maze import representations as mr


# %% Global Variables

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "distance_to_goal"


with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

FRAME_RATE = 60

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]
GOAL_SETS = ["subset_1", "subset_2", "all"]


# %% dev


# %% results plotting functions


def plot_distance_aligned_results(results_df, ax=None, color="rosybrown", sig_color="slategrey", ymax=0.45):
    """ """
    # set up plot
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(4, 3), clear=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Steps to goal")
    ax.set_ylabel("Decoding Acc. \n (chance subtracted)")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax.set_ylim(-0.02, ymax)
    # average chance subtracted decoding acc over steps_to_goal across subjects
    df = results_df.groupby(["steps_to_goal", "subject_ID"]).norm_acc.mean().unstack().T
    steps = df.columns.values
    mean = df.mean(axis=0)
    sem = df.sem(axis=0)
    # plot
    ax.plot(steps, mean, color=color, lw=2)
    ax.fill_between(steps, mean - sem, mean + sem, color=color, alpha=0.2)
    ax.set_xlim(0, steps.max())
    # run stats
    _plot_p_values(ax, df, ymax, sig_color)


def plot_event_aligned_results(
    results_df, event, ax=None, chance=1 / 12, color="darkorange", sig_color="sandybrown", ymax=0.55
):
    """ """
    # set up plot
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(4, 3), clear=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(f"{event} (s)")
    ax.set_ylabel("Decoding Acc. \n (chance subtracted)")
    ax.axhline(y=chance, color="k", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="k", linestyle="--", alpha=0.5)
    ax.set_ylim(0, ymax)
    # average chance subtracted decoding acc over steps_to_goal across subjects
    df = results_df.groupby(["timepoint", "subject_ID"]).test_acc.mean().unstack().T
    timepoints = df.columns.values
    mean = df.mean(axis=0)
    sem = df.sem(axis=0)
    # plot
    ax.plot(timepoints, mean, color=color, lw=2)
    ax.fill_between(timepoints, mean - sem, mean + sem, color=color, alpha=0.2)
    ax.set_xlim(timepoints.min(), timepoints.max())
    ax.set_xticks([-5, 0, 5])
    _plot_p_values(ax, df, ymax, sig_color, chance=chance)
    return


def _plot_p_values(ax, df, height, color, chance=0):
    """"""
    p_values = []
    x = df.columns
    for i in x:
        t_stat, p_val = ttest_1samp(df[i], popmean=chance, alternative="greater")
        p_values.append(p_val)
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
    # indicate significant timepoints with line
    sig_idx = np.where(reject)[0]
    runs = np.split(sig_idx, np.where(np.diff(sig_idx) != 1)[0] + 1)
    for run in runs:
        if run.size > 0:
            x_run = x[run]
            y_run = np.full_like(x_run, height - 0.04, dtype=float)
            ax.plot(x_run, y_run, color=color, linewidth=2)


def plot_event_aligned_decoding_heatmap_summary(cue_results_df, reward_results_df, axes=None, cmap="Oranges", vmax=0.6):
    """
    Split decoding reusults by maze and goal subset to plot decoding acc summary of conditions
    in a heatmap.
    """
    if axes is None:
        f, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    (
        cue_grouped_df,
        reward_grouped_df,
    ) = [
        df.groupby(["goal_subset", "maze_name", "subject_ID", "timepoint"]).test_acc.mean().unstack()
        for df in [cue_results_df, reward_results_df]
    ]
    cue_mean_df, reward_mean_df = [
        df.groupby(["goal_subset", "maze_name"]).mean() for df in [cue_grouped_df, reward_grouped_df]
    ]
    # get complementary dfs that are True when value is sig above chance
    cue_sig_df = pd.DataFrame(index=cue_mean_df.index, columns=cue_mean_df.columns)
    reward_sig_df = pd.DataFrame(index=reward_mean_df.index, columns=reward_mean_df.columns)

    for df, sig_df in zip([cue_grouped_df, reward_grouped_df], [cue_sig_df, reward_sig_df]):
        times = df.columns
        for maze in cue_grouped_df.index.get_level_values(1).unique():
            for goal_subset in cue_grouped_df.index.get_level_values(0).unique():
                chance = (1 / 24) if goal_subset == "all" else (1 / 12)
                _df = df.loc[(goal_subset, maze)]
                # get p-values for these trials
                p_values = []
                for t in times:
                    t_stat, p_val = ttest_1samp(_df[t], popmean=chance, alternative="greater")
                    p_values.append(p_val)
                reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
                sig_df.loc[(goal_subset, maze), times] = reject

    # reorder goalset index
    for df in [cue_mean_df, reward_mean_df, cue_sig_df, reward_sig_df]:
        df.index = pd.MultiIndex.from_product(
            [["subset_1", "subset_2", "all"], df.index.levels[1]], names=["goal_subset", "maze_name"]
        )
    for ax, mean_df, sig_df, event in zip(
        axes, [cue_mean_df, reward_mean_df], [cue_sig_df, reward_sig_df], ["cue", "reward"]
    ):
        cbar = True if event == "reward" else False
        sns.heatmap(
            mean_df[sig_df],
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            ax=ax,
            rasterized=True,
            cbar=cbar,
            cbar_kws={"label": "Decoding Acc."},
        )
        times = mean_df.columns.values.astype(float)
        zero_point = np.argmin(np.abs(times))
        ax.axvline(zero_point, color="k", ls="--", alpha=0.5)
        tick_labels = [-5, 0, 5]
        tick_positions = [np.argmin(np.abs(times - tick)) for tick in tick_labels]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0)
        ax.set_xlabel(f"{event} time (s)")
        if event == "reward":
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
            spine.set_linewidth(0.5)

    return


def plot_distance_aligned_decoding_heatmap_summary(results_df, ax=None, cmap="Reds", vmax=0.5):
    """ """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(4, 4), clear=True)
    # average decoding acc for each subject then average over subjects
    subject_mean_df = (
        results_df.groupby(["goal_subset", "maze_name", "subject_ID", "steps_to_goal"]).norm_acc.mean().unstack()
    )
    mean_df = subject_mean_df.groupby(["goal_subset", "maze_name"]).mean(0)
    # calc sig from cross subject variance
    sig_df = pd.DataFrame(index=mean_df.index, columns=mean_df.columns)
    steps = mean_df.columns
    for maze in subject_mean_df.index.get_level_values(1).unique():
        for goal_subset in subject_mean_df.index.get_level_values(0).unique():
            _df = subject_mean_df.loc[(goal_subset, maze)]
            # get p-values for these trials
            p_values = []
            for t in steps:
                t_stat, p_val = ttest_1samp(_df[t], popmean=0, alternative="greater", nan_policy="omit")
                p_values.append(p_val)
            reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
            sig_df.loc[(goal_subset, maze), steps] = reject
    for df in [mean_df, sig_df]:
        df.index = pd.MultiIndex.from_product(
            [["subset_1", "subset_2", "all"], df.index.levels[1]], names=["goal_subset", "maze_name"]
        )
    # plot
    sns.heatmap(
        mean_df[sig_df],
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        ax=ax,
        rasterized=True,
        cbar=True,
        cbar_kws={"label": "Decoding Acc."},
    )
    tick_labels = steps[::4]
    tick_positions = [np.argmin(np.abs(steps - tick)) for tick in tick_labels]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_xlabel(f"Steps to goal")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(0.5)
    ax.set_ylabel("")


# %% Single reference frame exp average decoding


def get_aligned_decoding(reference, maze_names="all", goal_sets="all", verbose=True):
    """ """
    maze_names = MAZE_NAMES if maze_names == "all" else maze_names
    goal_sets = GOAL_SETS if goal_sets == "all" else goal_sets
    # run separately for all sessions for each subject
    results_dfs = []
    for subject_ID in SUBJECT_IDS:
        if verbose:
            print(f"Loading {subject_ID} data...")
        sessions = get_sessions_for_analysis([subject_ID], maze_names, goal_sets)
        for session in sessions:
            if verbose:
                print(f"Decoding: {session.name}")
            if reference == "distance":
                results_df = get_session_distance_aligned_decoding(session)
            elif reference in ["cue", "reward"]:
                results_df = get_session_event_aligned_decoding(session, event=reference)
            else:
                NotImplementedError
            results_df["subject_ID"] = subject_ID
            results_df["maze_name"] = session.maze_name
            results_df["goal_subset"] = session.goal_subset
            results_df["days_on_maze"] = session.day_on_maze
            results_dfs.append(results_df)
    return pd.concat(results_dfs, axis=0)


def get_sessions_for_analysis(subject_IDs, maze_names, goal_subsets):
    """ """
    days_on_maze = "late" if "all" in goal_subsets else "all"
    sessions = gs.get_maze_sessions(
        subject_IDs=subject_IDs,
        maze_names=maze_names,
        days_on_maze=days_on_maze,
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


# %% Cross reference frame decoding


# %% distance aligned analyses


def get_sessions_distance_basis_decoding():
    """ """
    return


def get_session_distance_aligned_decoding(
    session,
    inputs=["spikes"],
    trial_phases=["navigation"],
    resolution=0.5,
    binning_method="uniform",
    max_steps_from_goal=20,
    n_bins=20,
    goal_stratified_validation=True,
    n_test_trials=None,
    include_multi_units=False,
    whiten_features=True,
):
    """ """
    input_data = get_distance_aligned_input_data(
        session,
        resolution,
        include_multi_units,
        trial_phases,
        max_steps_to_goal=max_steps_from_goal,
        n_bins=n_bins,
        binning_method=binning_method,
    )
    bin_mids = sorted(input_data.steps_to_goal.bin_mid.dropna().unique())
    results_df = []
    for steps in bin_mids:
        steps_df = input_data[input_data.steps_to_goal.bin_mid == steps]
        valid_trials = steps_df.trial.unique()
        folds_df = get_folds_df(
            session, goal_stratified_validation, valid_trials, return_unique_IDs=True, n_test_trials=n_test_trials
        )
        if folds_df.shape[0] < 2:
            continue  # only one valid goal, cannot run classifer
        folds = folds_df.columns.levels[0].unique()
        for fold in folds:
            # get test and train data
            fold_df = folds_df[fold]
            test_trials = [t for t in fold_df.test.values.flatten() if isinstance(t, str)]
            train_trials = [t for t in fold_df.train.values.flatten() if isinstance(t, str)]
            test_df = steps_df[steps_df.trial_unique_ID.isin(test_trials)]
            train_df = steps_df[steps_df.trial_unique_ID.isin(train_trials)]
            train_y, test_y = train_df.goal.values, test_df.goal.values
            train_X, test_X = [], []
            if "spikes" in inputs:
                train_X.append(train_df.spike_count.values)
                test_X.append(test_df.spike_count.values)
            if "place" in inputs:
                train_X.append(train_df.place_onehot.values)
                test_X.append(test_df.place_onehot.values)
            train_X, test_X = np.concatenate(train_X, axis=1), np.concatenate(test_X, axis=1)
            if whiten_features:  # zscore features
                scaler = StandardScaler()  # mean=0, std=1 per column
                scaler.fit(train_X)  # learn stats on train
                train_X = scaler.transform(train_X)
                test_X = scaler.transform(test_X)
            # fit model
            decoder = LogisticRegression(max_iter=10000, penalty=None, random_state=0, class_weight="balanced")
            decoder.fit(train_X, train_y)
            chance = 1 / len(decoder.classes_)
            # test decoder
            test_pred = decoder.predict(test_X)
            for y, yhat, trial in zip(test_y, test_pred, test_trials):
                results_df.append(
                    {
                        "steps_to_goal": steps,
                        "fold": fold,
                        "trial": trial,
                        "goal": y,
                        "predicted_goal": yhat,
                        "test_acc": int(y == yhat),
                        "chance": chance,
                    }
                )
    results_df = pd.DataFrame(results_df)
    results_df["norm_acc"] = results_df.test_acc - results_df.chance
    return results_df


def get_distance_aligned_input_data(
    session,
    resolution=0.5,
    include_multi_units=True,
    include_trial_phases=["navigation"],
    ignore_last_n=2,
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
    ds_nav_rates_df = ds_nav_rates_df[~ds_nav_rates_df.steps_to_goal.future.isna()]
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


# %% event aligned analyses


def get_session_event_aligned_decoding(
    session,
    event="cue",
    resolution=0.5,
    window=(-10, 10),
    goal_stratified_validation=True,
    n_test_trials=None,
    include_multi_units=True,
    add_distance_transformation=True,
):
    """ """
    input_data = get_event_aligned_input_data(session, event, resolution, window, include_multi_units)
    timepoints = sorted(input_data.event_aligned_time[event].unique())
    folds_df = get_folds_df(session, goal_stratified_validation, return_unique_IDs=True, n_test_trials=n_test_trials)
    results_df = []
    for fold in folds_df.columns.levels[0].unique():
        fold_df = folds_df[fold]
        test_trials = [t for t in fold_df.test.values.flatten() if isinstance(t, str)]
        train_trials = [t for t in fold_df.train.values.flatten() if isinstance(t, str)]
        train_df = input_data[input_data.trial_unique_ID.isin(train_trials)]
        test_df = input_data[input_data.trial_unique_ID.isin(test_trials)]
        decoder = LogisticRegression(penalty=None, max_iter=10000, random_state=0)
        for t in timepoints:
            _train_df = train_df[train_df.event_aligned_time[event] == t]
            _test_df = test_df[test_df.event_aligned_time[event] == t]
            if _train_df.empty or _test_df.empty:
                continue  # rare cases when no trials for that timepoint (eg, end of session trial)
            X_train, y_train = _train_df.spike_count.values, _train_df.goal.values
            X_test, y_test = _test_df.spike_count.values, _test_df.goal.values
            # fit model
            decoder.fit(X_train, y_train)
            chance = 1 / len(decoder.classes_)
            # test decoder
            test_pred = decoder.predict(X_test)
            for y, yhat, trial in zip(y_test, test_pred, _test_df.trial_unique_ID.values):
                results_df.append(
                    {
                        "event": event,
                        "timepoint": t,
                        "fold": fold,
                        "trial": trial,
                        "goal": y,
                        "predicted_goal": yhat,
                        "test_acc": int(y == yhat),
                        "chance": chance,
                    }
                )
    results_df = pd.DataFrame(results_df)
    results_df["norm_acc"] = results_df.test_acc - results_df.chance
    if add_distance_transformation:
        window2steps = get_step_time_transformation(session, event)
        results_df["transformed_steps_to_goal"] = results_df.timepoint.map(window2steps)
    return results_df


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
