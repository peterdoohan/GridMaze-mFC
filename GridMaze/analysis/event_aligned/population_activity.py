"""
Library of calculating and ploting mFC population activity aligned to cue, reward and other trial events
@ peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.event_aligned import delta_distance_to_goal as ddtg

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, ANALYSIS_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(ANALYSIS_INFO_PATH / "intra_trial_interval_times.json", "r") as input_file:
    INTRA_TRIAL_INTERVAL_TIMES = json.load(input_file)

FRAME_RATE = 60

DDTG_LOWER_THRES = -0.1  # rate of change of distance to goal thres for goal-directed behaviour at cue

DDTG_UPPER_THRES = 0.015  #  ... for non-goal-directed at cue

DDTG_WINDOW = (1, 3)  # seconds post cue to calculate ddtg

# %% New functions


# %% Population activity timeseries


def get_event_aligned_population_acitivity(
    plot=True, aligned_to="trial", normalise_clusters="max", normalise_sessions="zscore"
):
    """"""
    av_rates = []
    data_structure = aligned_to + "_aligned_rates_df"
    for subject in SUBJECT_IDS:
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject], maze_names="all", days_on_maze="late", with_data=[data_structure, "cluster_metrics"]
        )
        session_av_rates = []
        for session in sessions:
            aligned_rates_df = getattr(session, data_structure)
            # only include single units
            cluster_metrics = session.cluster_metrics
            single_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID.values
            aligned_rates_df = aligned_rates_df[aligned_rates_df.cluster_ID.isin(single_units)]
            # average neurons over trials
            trial_average_rates = (
                aligned_rates_df.set_index("cluster_unique_ID")
                .groupby("cluster_unique_ID")
                .firing_rate.mean()
                .firing_rate
            )
            if normalise_clusters == "max":
                if aligned_to == "event":
                    for event in ["cue_aligned", "reward_aligned"]:
                        trial_average_rates[event] = trial_average_rates[event].apply(lambda x: x / x.max(), axis=1)
                else:
                    trial_average_rates = trial_average_rates.apply(lambda x: x / x.max(), axis=1)
            population_average_rates = aligned_rates_df.firing_rate.mean(axis=0)
            session_av_rates.append(population_average_rates)
        subject_av_rates = pd.concat(session_av_rates, axis=1).T
        if normalise_sessions == "max":
            subject_av_rates = subject_av_rates.apply(lambda x: x / x.max(), axis=1)
        elif normalise_sessions == "zscore":
            subject_av_rates = subject_av_rates.apply(zscore, axis=1)
        subject_av_rates = subject_av_rates.mean()
        av_rates.append(subject_av_rates)
    population_average_rates = pd.concat(av_rates, axis=1).T
    population_average_rates.index = SUBJECT_IDS
    if plot:
        if aligned_to == "event":
            _plot_population_event_aligned_activity(population_average_rates)
        elif aligned_to == "trial":
            _plot_population_trial_aligned_activity(population_average_rates)
    return population_average_rates


def _plot_population_event_aligned_activity(population_average_rates, ax=None, color="black"):
    if ax is None:
        f, axes = plt.subplots(1, 2, figsize=(6, 3), clear=True, sharey=True)
    ax.spines[["right", "top"]].set_visible(False)
    for i, event in enumerate(["cue_aligned", "reward_aligned"]):
        event_aligned_activity = population_average_rates[event]
        time = event_aligned_activity.columns.to_numpy(dtype=float)
        y = event_aligned_activity.mean(axis=0).to_numpy()
        sem = event_aligned_activity.sem(axis=0).to_numpy()
        axes[i].plot(time, y, color=color)
        axes[i].fill_between(time, y - sem, y + sem, color=color, alpha=0.5)
        axes[i].axvline(0, color="k", linewidth=1, alpha=0.5, zorder=0)
        axes[i].set_xlabel(f"{event} time (s)")
        axes[i].spines["right"].set_visible(False)
        axes[i].spines["top"].set_visible(False)
        if i == 0:
            axes[i].set_ylabel("Pop. Rate (z-score)")
    return


def _plot_population_trial_aligned_activity(population_average_rates, color="black", ax=None, t_min=-2):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(6, 2), clear=True)
    ax.spines[["right", "top"]].set_visible(False)
    time = population_average_rates.columns.to_numpy(dtype=float)
    mask = time > t_min
    time = time[mask]
    y = population_average_rates.mean(axis=0).to_numpy()
    y = y[mask]
    sem = population_average_rates.sem(axis=0).to_numpy()
    sem = sem[mask]
    ax.plot(time, y, color=color)
    ax.fill_between(time, y - sem, y + sem, color=color, alpha=0.5)
    for x in INTRA_TRIAL_INTERVAL_TIMES.values():
        ax.axvline(x, color="k", linewidth=1, ls="--", alpha=0.5, zorder=0)
    ax.set_xlabel("Time (s)")
    ax.set_xticks([float(x) for x in INTRA_TRIAL_INTERVAL_TIMES.values()])
    ax.set_xticklabels(["Cue", "Reward", "ERC", "ITI"])
    ax.set_ylabel("Pop. Rate (z-score)")
    return


# %% Goal directed vs Non-goal directed cue bump


def get_cue_aligned_population_activity_residual(normalise_clusters=False, normalise_sessions=False, plot=True):
    """
    Fix ylims, best results without normalisation
    """
    dfs = []
    for subject in SUBJECT_IDS:
        print(subject)
        conditions = ([], [], [])  # goal-directed, non-goal-directed, not-moving
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject],
            maze_names="all",
            days_on_maze="late",
            with_data=["event_aligned_rates_df", "cluster_metrics", "navigation_df"],
            must_have_data=True,
        )
        for session in sessions:
            aligned_rates_df = session.event_aligned_rates_df
            # only include single units
            cluster_metrics = session.cluster_metrics
            single_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID.values
            aligned_rates_df = aligned_rates_df[aligned_rates_df.cluster_ID.isin(single_units)]
            # split trials into goal_direced, non_goal_directed and not_moving
            ddtg_df = ddtg.get_session_delta_dtg(session, DDTG_WINDOW)
            goal_directed_trials = ddtg_df[ddtg_df.cue_aligned_time.mean(1).le(DDTG_LOWER_THRES)].trial.values.astype(
                int
            )
            non_goal_directed_trials = ddtg_df[
                ddtg_df.cue_aligned_time.mean(1).ge(DDTG_UPPER_THRES)
            ].trial.values.astype(int)
            not_moving_trials = ddtg_df[ddtg_df.cue_aligned_time.mean(1) == 0].trial.values.astype(int)
            # get average cluster responses in each condition
            condition_dfs = [
                aligned_rates_df[aligned_rates_df.trial.isin(trials)]
                for trials in [goal_directed_trials, non_goal_directed_trials, not_moving_trials]
            ]
            # average neurons over trials
            for i, df in enumerate(condition_dfs):
                if not df.empty:
                    av_rates_df = df.groupby("cluster_unique_ID").firing_rate.mean().firing_rate.cue_aligned
                    if normalise_clusters == "max":
                        av_rates_df = av_rates_df.apply(lambda x: x / x.max(), axis=1)
                    population_rate = av_rates_df.mean()
                    conditions[i].append(population_rate)
        subject_dfs = [pd.concat(condition, axis=1).T for condition in conditions]
        # normalise across sessions
        if normalise_sessions == "max":
            subject_dfs = [condition.apply(lambda x: x / x.max(), axis=1) for condition in subject_dfs]
        elif normalise_sessions == "zscore":
            subject_dfs = [condition.apply(zscore, axis=1) for condition in subject_dfs]
        conditions_df = pd.concat([df.mean() for df in subject_dfs], axis=1).T
        conditions_df.columns = pd.MultiIndex.from_product([["time"], conditions_df.columns])
        conditions_df[("condition", "")] = ["goal-directed", "non-goal-directed", "not-moving"]
        conditions_df[("subject_ID", "")] = subject
        dfs.append(conditions_df)
    results_df = pd.concat(dfs, axis=0)
    if plot:
        plot_cue_aligned_residuals(results_df, conditions=["goal-directed", "non-goal-directed"])
    return results_df


def plot_cue_aligned_residuals(
    results_df, conditions=["goal-directed", "non-goal-directed"], window=(-0.5, 1), ax=None
):
    """ """
    # prepare axes
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3), clear=True)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xlabel("Cue-aligned Time (s)")
    ax.set_ylabel("Pop. Rate (z-score)")
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    # prepare and plot data
    time = results_df.time.columns.values.astype(float)
    if not conditions:  # plot conditions separately
        for condition, color in zip(results_df.condition.unique(), ["darkorange", "dodgerblue", "gray"]):
            condition_df = results_df[results_df.condition == condition].time
            mean = condition_df.mean()
            sem = condition_df.sem()
            ax.plot(time, mean, color=color, label=condition)
            ax.fill_between(time, mean - sem, mean + sem, color=color, alpha=0.2)
            # ax.set_ylim(-0.05, 2)
    else:  # plot residual between condtions
        residuals = np.zeros((len(SUBJECT_IDS), len(time)))
        for i, subject in enumerate(SUBJECT_IDS):
            subject_df = results_df[results_df.subject_ID == subject]
            residual = (
                subject_df[subject_df.condition == conditions[0]].time.values
                - subject_df[subject_df.condition == conditions[1]].time.values
            )
            residuals[i, :] = residual.reshape(-1)
        mean = residuals.mean(axis=0)
        sem = residuals.std(axis=0) / np.sqrt(len(SUBJECT_IDS))
        ax.plot(time, mean, color="purple", label=f"{conditions[0]} - {conditions[1]}")
        ax.fill_between(time, mean - sem, mean + sem, color="purple", alpha=0.2)
        ax.set_ylim(-0.1, 0.8)
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)

    ax.set_xlim(*window)
    # ax.legend(fontsize=8, loc="upper right")


# %%
