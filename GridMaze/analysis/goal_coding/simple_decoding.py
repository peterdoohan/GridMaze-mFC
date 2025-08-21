"""
Can we decode the goal from mFC? Let's start simple, LogReg decoding on session level data
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.goal_coding import decoding_utils as du

# %% Global Variables
from GridMaze.paths import ANALYSIS_INFO_PATH, RESULTS_PATH

with open(ANALYSIS_INFO_PATH / "intra_trial_interval_times.json", "r") as f:
    INTRA_TRIAL_INTERVAL_TIMES = json.load(f)

RESULTS_DIR = RESULTS_PATH / "simple_decoding"

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]
GOAL_SUBSETS = ["subset_1", "subset_2", "all"]

# %%  plotting


def plot_event_aligned_decoding_summary(summary_df, axes=None):
    """ """
    if axes is None:
        f, axes = plt.subplots(3, 6, figsize=(6, 3), sharey=True, sharex=True)
    for i, maze in enumerate(MAZE_NAMES):
        for j, goal_subset in enumerate(GOAL_SUBSETS):
            chance = 1 / 24 if goal_subset == "all" else 1 / 12
            label = f"{maze} - {goal_subset}"
            these_axes = axes[i, 2 * j : 2 * j + 2]
            plot_event_aligned_decoding(
                summary_df,
                maze_names=[maze],
                goal_subsets=[goal_subset],
                chance=chance,
                axes=these_axes,
            )
            these_axes[0].set_title(label, fontsize=8)
    return


def plot_trial_aligned_decoding_summary(summary_df, axes=None):
    if axes is None:
        f, axes = plt.subplots(3, 3, figsize=(6, 3), sharex=True, sharey=True)
    for i, maze in enumerate(MAZE_NAMES):
        for j, goal_subset in enumerate(GOAL_SUBSETS):
            chance = 1 / 24 if goal_subset == "all" else 1 / 12
            label = f"{maze} - {goal_subset}"
            ax = axes[i, j]
            plot_trial_aligned_decoding(
                summary_df,
                maze_names=[maze],
                goal_subsets=[goal_subset],
                color="deepskyblue",
                chance=chance,
                ax=ax,
            )
            ax.set_title(label, fontsize=8)
    return


def plot_distance_aligned_decoding(
    summary_df,
    maze_names=["maze_1", "maze_2"],
    goal_subsets=["subset_1", "subset_2"],
    chance=1 / 12,
    color="deepskyblue",
    max_dist=20,
    y_max=0.5,
    axes=None,
):
    """ """
    # set up figure
    if axes is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 1.5), sharey=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(y=chance, color="k", ls=":", alpha=0.5)
    ax.set_xlabel("steps to goal")
    ax.set_ylabel("acc.")
    # process data
    df = summary_df[(summary_df.maze_name.isin(maze_names)) & (summary_df.goal_subset.isin(goal_subsets))]
    df = df[(df.event == "reward") & df.timepoint.le(0) & df.trial_phase.isin(["navigation"])]
    subject_means = df.groupby(["subject_ID", "steps_to_goal"]).accuracy.mean().unstack(level=1)
    grand_mean = subject_means.mean()
    distances = grand_mean.index.values
    grand_mean = grand_mean.values
    grand_sem = subject_means.sem().values
    # plot
    ax.plot(distances, grand_mean, color=color)
    ax.fill_between(distances, grand_mean - grand_sem, grand_mean + grand_sem, color=color, alpha=0.2)
    # do stats
    reject, p_vals = _timeseries_ttests(subject_means, chance=chance)
    plot_sig(reject, distances, ax, sig_pos=y_max, sig_color=color)
    ax.set_xlim(0, max_dist)


def plot_event_aligned_decoding(
    summary_df,
    maze_names=["maze_1", "maze_2"],
    goal_subsets=["subset_1", "subset_2"],
    chance=1 / 12,
    plot_smooth_SD=False,
    color="deepskyblue",
    y_max=0.65,
    axes=None,
):
    """ """
    # set up figure
    if axes is None:
        f, axes = plt.subplots(1, 2, figsize=(3, 1.5), sharey=True)
    for event, ax in zip(["cue", "reward"], axes):
        ax.axvline(x=0, color="k", ls="--", alpha=0.5)
        ax.axhline(y=chance, color="k", ls=":", alpha=0.5)
        ax.set_xlabel(event)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].spines[["top", "left", "right"]].set_visible(False)
    axes[0].set_ylabel("acc.")

    # process data
    df = summary_df[(summary_df.maze_name.isin(maze_names)) & (summary_df.goal_subset.isin(goal_subsets))]
    subject_mean_dfs = []
    for event, ax in zip(["cue", "reward"], axes):
        event_df = df[df.event == event]
        if event == "cue":
            # we don't want to non nav times after the cue (eg, consuming reward)
            event_df = event_df[~(event_df.timepoint.gt(0) & (event_df.trial_phase != "navigation"))]
        subject_means = event_df.groupby(["timepoint", "subject_ID"]).accuracy.mean().unstack(level=0)
        _subject_means_df = subject_means.copy()
        _subject_means_df.columns = pd.MultiIndex.from_product([[event], _subject_means_df.columns])
        subject_mean_dfs.append(_subject_means_df)
        # do stats before smoothing
        if plot_smooth_SD:
            subject_means = pd.DataFrame(
                gaussian_filter1d(subject_means.values, sigma=plot_smooth_SD, axis=1),
                index=subject_means.index,
                columns=subject_means.columns,
            )
        grand_mean = subject_means.mean().values
        grand_sem = subject_means.sem().values
        timepoints = np.sort(event_df.timepoint.unique())
        ax.plot(timepoints, grand_mean, color=color)
        ax.fill_between(timepoints, grand_mean - grand_sem, grand_mean + grand_sem, color=color, alpha=0.2)
    # do stats
    combined_subject_means = pd.concat(subject_mean_dfs, axis=1)
    reject, p_values_corr = _timeseries_ttests(combined_subject_means, chance=chance)
    n_timepoints = timepoints.shape[0]
    for ax, _reg in zip(axes, [reject[:n_timepoints], reject[n_timepoints:]]):
        plot_sig(_reg, timepoints, ax, sig_pos=y_max, sig_color=color)
    return combined_subject_means


def plot_trial_aligned_decoding(
    summary_df,
    maze_names=["maze_1", "maze_2", "rooms_maze"],
    goal_subsets=["subset_1", "subset_2"],
    color="deepskyblue",
    chance=1 / 12,
    plot_smooth_SD=False,
    y_max=0.32,
    ax=None,
):
    """ """
    # set up figure
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 1.5))
    ax.spines[["top", "right"]].set_visible(False)
    event_times = list(INTRA_TRIAL_INTERVAL_TIMES.values())[:-1]
    timepoints = np.sort(summary_df.timepoint.unique())
    event_inds = [np.argmin(np.abs(np.array(timepoints) - time)) for time in event_times]
    for ind in event_inds:
        ax.axvline(ind, color="k", ls="--", alpha=0.5)
    ax.set_xticks(event_inds)
    ax.set_xticklabels(["Cue", "Reward", "ITI"], rotation=0)
    ax.axhline(chance, color="k", ls=":", alpha=0.5)
    ax.set_ylabel("acc.")
    # process data
    df = summary_df[(summary_df.maze_name.isin(maze_names)) & (summary_df.goal_subset.isin(goal_subsets))]
    subject_means = df.groupby(["timepoint", "subject_ID"]).accuracy.mean().unstack(level=0)
    # do stats before smoothing
    reject_null = _timeseries_ttests(subject_means, chance)
    if plot_smooth_SD:
        subject_means = pd.DataFrame(
            gaussian_filter1d(subject_means.values, sigma=plot_smooth_SD, axis=1),
            index=subject_means.index,
            columns=subject_means.columns,
        )
    grand_mean = subject_means.mean().values
    grand_sem = subject_means.sem().values
    x_range = np.arange(len(timepoints))
    ax.plot(x_range, grand_mean, color=color)
    ax.fill_between(x_range, grand_mean - grand_sem, grand_mean + grand_sem, color=color, alpha=0.2)
    plot_sig(reject_null, x_range, ax, sig_pos=y_max, sig_color=color)


def _timeseries_ttests(subject_means, chance):
    p_values = []
    timepoints = subject_means.columns.values
    for t in timepoints:
        stat, p = ttest_1samp(subject_means[t], popmean=chance, alternative="greater")
        p_values.append(p)
    reject, p_values_corr, _, _ = multipletests(p_values, method="fdr_bh")
    return reject, p_values_corr


def plot_sig(reject, x_range, ax, sig_pos=1, sig_color="red"):
    if sum(reject) > 0:
        ax.scatter(
            x_range[reject],
            np.ones(len(x_range[reject])) * sig_pos,
            marker="s",
            color=sig_color,
            s=0.75,
        )


# %% pop level functions


def get_event_aligned_decoding_summary(
    resolution=0.5,
    window=(-10, 10),
    verbose=True,
    save=False,
):
    """ """
    # load cached results if already processed
    save_path = RESULTS_DIR / "event_aligned_decoding_summary.parquet"
    if save_path.exists() and not save:
        if verbose:
            print(f"Loading existing results from {save_path}")
        return pd.read_parquet(save_path)
    if verbose:
        print("Loading sessions ...")
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names="all",
        days_on_maze="late",
        with_data=[
            "navigation_df",
            "navigation_spike_counts_df",
            "cluster_metrics",
            "trials_df",
        ],
        must_have_data=True,
    )
    dfs = []
    for session in sessions:
        if verbose:
            print(session.name)
        for event in ["cue", "reward"]:
            results_df = get_event_aligned_goal_decoding(
                session,
                event=event,
                resolution=resolution,
                window=window,
            )
            dfs.append(results_df)
    summary_df = pd.concat(dfs, axis=0)
    summary_df.reset_index(drop=True, inplace=True)
    if save:
        if verbose:
            print(f"Saving results to {save_path}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_parquet(save_path)
    return summary_df


def get_trial_aligned_decoding_summary(
    resolution=0.25,
    verbose=True,
    save=False,
):
    # load cached results if already processed
    save_path = RESULTS_DIR / "trial_aligned_decoding_summary.parquet"
    if save_path.exists() and not save:
        if verbose:
            print(f"Loading existing results from {save_path}")
        return pd.read_parquet(save_path)
    if verbose:
        print("Loading sessions ...")
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names="all",
        days_on_maze="late",
        with_data=["trial_aligned_rates_df", "cluster_metrics", "trials_df"],
        must_have_data=True,
    )
    dfs = []
    failed_sessions = []
    for session in sessions:
        if verbose:
            print(session.name)
        try:
            results_df = get_trial_aligned_goal_decoding(session, resolution=resolution)
            dfs.append(results_df)
        except Exception as e:
            if verbose:
                print(f"Failed to process session {session.name}: {e}")
            failed_sessions.append(session.name)
            continue
    summary_df = pd.concat(dfs, axis=0)
    summary_df.reset_index(drop=True, inplace=True)
    if save:
        if verbose:
            print(f"Saving results to {save_path}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_parquet(save_path)
    if len(failed_sessions) > 0:
        print(f"Failed sessions: {', '.join(failed_sessions)}")
    return summary_df


# %% session level functions


def get_event_aligned_goal_decoding(
    session,
    event="cue",
    resolution=0.5,
    window=(-10, 10),
    goal_stratified_validation=True,
    n_test_trials=None,
    include_multi_units=True,
    zscore_spikes=True,
):
    """ """
    input_data = du.get_event_aligned_input_data(session, event, resolution, window, include_multi_units)
    timepoints = sorted(input_data.event_aligned_time[event].unique())
    folds_df = du.get_folds_df(session, goal_stratified_validation, return_unique_IDs=True, n_test_trials=n_test_trials)
    results_dfs = []
    for fold in folds_df.columns.levels[0].unique():
        fold_df = folds_df[fold]
        test_trials = [t for t in fold_df.test.values.flatten() if isinstance(t, str)]
        train_trials = [t for t in fold_df.train.values.flatten() if isinstance(t, str)]
        train_df = input_data[input_data.trial_unique_ID.isin(train_trials)]
        test_df = input_data[input_data.trial_unique_ID.isin(test_trials)]
        decoder = LogisticRegression(penalty=None, max_iter=10000, random_state=0, class_weight="balanced")
        for t in timepoints:
            _train_df = train_df[train_df.event_aligned_time[event] == t]
            _test_df = test_df[test_df.event_aligned_time[event] == t]
            if _train_df.empty or _test_df.empty:
                continue  # rare cases when no trials for that timepoint (eg, end of session trial)
            X_train, y_train = _train_df.spike_count.values, _train_df.goal.values
            X_test, y_test = _test_df.spike_count.values, _test_df.goal.values
            if zscore_spikes:  # zscore features
                scaler = StandardScaler()  # mean=0, std=1 per column
                scaler.fit(X_train)  # learn stats on train
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
            # fit model
            decoder.fit(X_train, y_train)
            # out_df
            y_pred = decoder.predict(X_test)
            correct = (y_pred == y_test).astype(int)
            n_test_samp = len(y_test)
            df = pd.DataFrame(
                {
                    "timepoint": np.repeat(t, n_test_samp),
                    "trial_unique_ID": _test_df.trial_unique_ID.values,
                    "steps_to_goal": _test_df.steps_to_goal.bin_mid,
                    "trial_phase": _test_df.trial_phase.values,
                    "true_goal": y_test,
                    "predicted_goal": y_pred,
                    "accuracy": correct,
                }
            )
            df["fold"] = fold
            df["event"] = event
            results_dfs.append(df)
    results_df = pd.concat(results_dfs, axis=0)
    results_df.reset_index(drop=True, inplace=True)
    # add session info
    results_df["subject_ID"] = session.subject_ID
    results_df["maze_name"] = session.maze_name
    results_df["day_on_maze"] = session.day_on_maze
    results_df["goal_subset"] = session.goal_subset
    return results_df


def get_trial_aligned_goal_decoding(
    session,
    resolution=0.25,
    goal_stratified_validation=True,
    n_test_trials=None,
    include_multi_units=True,
    zscore_spikes=True,
):
    """ """
    # load data
    trial_aligned_rates = session.trial_aligned_rates_df
    cluster_metrics = session.cluster_metrics
    session_info = session.session_info
    # filter clusters
    keep_clusters = gc.filter_clusters(
        cluster_metrics,
        session_info,
        return_unique_IDs=False,
        single_units=True,
        multi_units=include_multi_units,
    )
    trial_aligned_rates = trial_aligned_rates[trial_aligned_rates.cluster_ID.isin(keep_clusters)]
    # downsample resolution
    rates = trial_aligned_rates.firing_rate
    times = rates.columns.astype(float).values
    rates_T = rates.T
    rates_T.index = rates_T.index.astype(float)
    ds_rates = rates_T.groupby(rates_T.index // resolution).mean().T  # average over samples to desired resolution
    new_times = np.linspace(times[0], times[-1], ds_rates.shape[1])
    ds_rates.columns = pd.MultiIndex.from_product([["firing_rate"], new_times])
    input_data = pd.concat(
        [trial_aligned_rates.drop("firing_rate", level=0, axis=1), ds_rates],
        axis=1,
    )  # recombine
    timepoints = new_times
    folds_df = du.get_folds_df(
        session, goal_stratified_validation, return_unique_IDs=False, n_test_trials=n_test_trials
    )
    folds = np.unique(folds_df.columns.get_level_values(0))
    results_dfs = []
    for fold in folds:
        fold_df = folds_df[fold]
        try:
            test_trials = fold_df.test.stack().dropna().values
            train_trials = fold_df.train.stack().dropna().values
        except AttributeError:
            # non goal strat too few trials
            test_trials = fold_df.test.dropna().values
            train_trials = fold_df.train.dropna().values
        train_df = input_data[input_data.trial.isin(train_trials)]
        test_df = input_data[input_data.trial.isin(test_trials)]
        train_df.set_index(["cluster_ID", "trial"], inplace=True)
        test_df.set_index(["cluster_ID", "trial"], inplace=True)
        decoder = LogisticRegression(penalty=None, max_iter=10000, random_state=0, class_weight="balanced")
        y_train = train_df.goal.unstack(level=0)[0].values  # n train trials
        y_test = test_df.goal.unstack(level=0)[0].values  # n test trials
        for timepoint in timepoints:
            X_train = train_df.firing_rate[timepoint].unstack(level=0).values  # n_trials x n_clusters
            X_test = test_df.firing_rate[timepoint].unstack(level=0).values
            if zscore_spikes:  # zscore features
                scaler = StandardScaler()  # mean=0, std=1 per column
                scaler.fit(X_train)  # learn stats on train
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                # fit model
            decoder.fit(X_train, y_train)
            # out_df
            y_pred = decoder.predict(X_test)
            correct = (y_pred == y_test).astype(int)
            n_test_samp = len(y_test)
            df = pd.DataFrame(
                {
                    "timepoint": np.repeat(timepoint, n_test_samp),
                    "trial": test_df.firing_rate[new_times[0]].unstack(level=0).index.values,
                    "true_goal": y_test,
                    "predicted_goal": y_pred,
                    "accuracy": correct,
                }
            )
            df["fold"] = fold
            results_dfs.append(df)
    results_df = pd.concat(results_dfs, axis=0)
    results_df.reset_index(drop=True, inplace=True)
    # add session info
    results_df["subject_ID"] = session.subject_ID
    results_df["maze_name"] = session.maze_name
    results_df["day_on_maze"] = session.day_on_maze
    results_df["goal_subset"] = session.goal_subset
    return results_df
