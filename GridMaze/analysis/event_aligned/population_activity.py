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
from GridMaze.analysis.core import convert
from GridMaze.analysis.event_aligned import delta_distance_to_goal as ddtg

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, ANALYSIS_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(ANALYSIS_INFO_PATH / "intra_trial_interval_times.json", "r") as input_file:
    INTRA_TRIAL_INTERVAL_TIMES = json.load(input_file)

FRAME_RATE = 60

DDTG_LOWER_THRES = -0.03  # -0.1 # rate of change of distance to goal thres for goal-directed behaviour at cue

DDTG_UPPER_THRES = 0.03  # 0.015 #  ... for non-goal-directed at cue

DDTG_WINDOW = (1, 3)  # seconds post cue to calculate ddtg

# %% New functions

# %%


# %% single main fn for all population activity analysis


def get_aligned_population_activity(aligned_to="trial", cue_moving_window=(-0.2, 0.5)):
    """ """
    data_structure = aligned_to + "_aligned_rates_df"
    _lvl = 1 if aligned_to == "trial" else 2
    dfs = []
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names="all",
        days_on_maze="late",
        with_data=[data_structure, "cluster_metrics", "navigation_df"],
    )
    for session in sessions:
        aligned_rates_df = getattr(session, data_structure)
        # only include single units
        cluster_metrics = session.cluster_metrics
        single_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID.values
        aligned_rates_df = aligned_rates_df[aligned_rates_df.cluster_ID.isin(single_units)]
        # z score normalise neurons across trials
        norm_aligned_rates_df = (
            aligned_rates_df.set_index(["cluster_unique_ID", "trial"])
            .firing_rate.reset_index("trial")
            .pivot(columns="trial")  # stack trials [neurons, timepoints x trials]
            .sort_index(axis=1, level=_lvl)
            .apply(zscore, axis=1)
        )
        pop_activity_df = norm_aligned_rates_df.stack(future_stack=True).groupby("trial").mean(0)
        if aligned_to == "trial":
            pop_activity_df.columns = pd.MultiIndex.from_product([["time"], pop_activity_df.columns])
        # split trials into goal_direced, non_goal_directed and not_moving
        ddtg_df = ddtg.get_session_delta_dtg(session, DDTG_WINDOW)
        trial2condition = {}
        window2ddtg = ddtg_df.set_index("trial").cue_aligned_time.mean(1)
        trial2condition = {}
        for trial, _ddtg in window2ddtg.items():
            trial = int(trial)
            if _ddtg <= DDTG_LOWER_THRES:
                trial2condition[trial] = "goal_directed"
            elif _ddtg >= DDTG_UPPER_THRES:
                trial2condition[trial] = "non_goal_directed"
            else:
                trial2condition[trial] = "not_moving"
        trial2cue_moving = _get_trial2cue_moving(session, window=cue_moving_window)
        pop_activity_df[("condition", "")] = pop_activity_df.index.map(trial2condition)
        pop_activity_df[("moving", "")] = pop_activity_df.index.map(trial2cue_moving)
        pop_activity_df[("subject_ID", "")] = session.subject_ID
        pop_activity_df[("trial_unique_ID", "")] = convert.trial2trial_unique_ID(
            session.session_info, pop_activity_df.index
        )
        pop_activity_df = pop_activity_df.reset_index(drop=True)
        dfs.append(pop_activity_df)
    return pd.concat(dfs, axis=0).reset_index(drop=True)


def _get_trial2cue_moving(session, window=(-0.5, 0.5)):
    """ """
    navigation_df = session.navigation_df
    trial2cue_moving = {}
    for trial in navigation_df.trial.dropna().unique():
        trial_df = navigation_df[navigation_df.trial == trial]
        cue_frame = trial_df.index[0]
        start_frame, end_frame = cue_frame + window[0] * FRAME_RATE, cue_frame + window[1] * FRAME_RATE
        speeds = trial_df.loc[start_frame:end_frame, "speed"].values
        moving = True if speeds.mean() > DDTG_UPPER_THRES else False
        trial2cue_moving[int(trial)] = moving
    return trial2cue_moving


def _plot_population_aligned_activity(results_df, smooth_SD=3, t_min=-2, color="black", ax=None):
    """ """
    # prepare axes
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(6, 2), clear=True)
    # process data
    df = results_df.groupby(["subject_ID"]).time.mean().time
    time = df.columns.to_numpy(dtype=float)
    if smooth_SD:
        df = df.apply(lambda x: gaussian_filter1d(x, smooth_SD), axis=1).apply(pd.Series)
    ax.spines[["right", "top"]].set_visible(False)
    mask = time > t_min
    time = time[mask]
    y = df.mean(axis=0).to_numpy()
    y = y[mask]
    sem = df.sem(axis=0).to_numpy()
    sem = sem[mask]
    # plot
    ax.plot(time, y, color=color)
    ax.fill_between(time, y - sem, y + sem, color=color, alpha=0.2)
    for x in INTRA_TRIAL_INTERVAL_TIMES.values():
        ax.axvline(x, color="k", ls="--", alpha=0.2, zorder=0)
    ax.set_xlabel("Time (s)")
    ax.set_xticks([float(x) for x in INTRA_TRIAL_INTERVAL_TIMES.values()])
    ax.set_xticklabels(["Cue", "Reward", "ERC", "ITI"])
    ax.set_ylabel("Pop. Rate (z-scored)")
    return


def _plot_cue_aligned_conditions(
    results_df,
    conditions=["goal_directed", "non_goal_directed"],
    aligned_to="trial",
    window=(-1, 1),
    stratify_moving=False,
    ax=None,
):
    """ """
    if aligned_to == "trial":
        df = results_df.groupby(["subject_ID", "condition"]).time.mean().time
    else:
        df = (
            results_df.groupby(
                [
                    "subject_ID",
                    "condition",
                ]
            )
            .cue_aligned.mean()
            .cue_aligned
        )
    # prepare axes
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3), clear=True)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xlabel("Cue (s)")
    ax.set_ylabel("Goal-dir. — non-goal-dir. \n Δ Pop. Rate (z-scored)")
    ax.axvline(0, color="k", linestyle="--", alpha=0.2)
    # prepare and plot data
    time = df.columns.values.astype(float)
    if not conditions:
        for condition, color in zip(np.unique(df.index.get_level_values(1)), ["darkorange", "dodgerblue", "gray"]):
            if stratify_moving and condition == "goal_directed":
                continue
            condition_df = df.xs(condition, level="condition")
            mean = condition_df.mean()
            sem = condition_df.sem()
            ax.plot(time, mean, color=color, label=condition)
            ax.fill_between(time, mean - sem, mean + sem, color=color, alpha=0.2)
        if stratify_moving:
            gd_df = results_df[results_df.condition == "goal_directed"]
            gd_df = gd_df.groupby(["subject_ID", "moving"]).time.mean().time
            for moving, ls, label in zip([True, False], ["-", "--"], ["moving", "not moving"]):
                condition_df = gd_df.xs(moving, level="moving")
                mean = condition_df.mean()
                sem = condition_df.sem()
                ax.plot(time, mean, color="darkorange", linestyle=ls, label=f"goal-directed ({label})")
                ax.fill_between(time, mean - sem, mean + sem, color="darkorange", alpha=0.2)
    else:
        residuals = np.zeros((len(SUBJECT_IDS), len(time)))
        for i, subject in enumerate(SUBJECT_IDS):
            residual = (df.loc[subject, conditions[0]] - df.loc[subject, conditions[1]]).values
            residuals[i, :] = residual.reshape(-1)
        mean = residuals.mean(axis=0)
        sem = residuals.std(axis=0) / np.sqrt(len(SUBJECT_IDS))
        ax.plot(time, mean, color="purple", label=f"{conditions[0]} - {conditions[1]}")
        ax.fill_between(time, mean - sem, mean + sem, color="purple", alpha=0.2)
        ax.set_ylim(-0.2, 0.4)
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlim(*window)
    ax.set_ylim(-0.01, 0.05)
    ax.legend()

    return


# %% Old


# def get_event_aligned_population_activity2():
#     """
#     same as below but with more z-scoring
#     """
#     subject_means = []
#     for subject in SUBJECT_IDS:
#         sessions = gs.get_maze_sessions(
#             subject_IDs=[subject],
#             maze_names="all",
#             days_on_maze="late",
#             with_data=["trial_aligned_rates_df", "cluster_metrics"],
#         )
#         pop_activity_dfs = []
#         for session in sessions:
#             aligned_rates_df = session.trial_aligned_rates_df
#             # only include single units
#             cluster_metrics = session.cluster_metrics
#             single_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID.values
#             aligned_rates_df = aligned_rates_df[aligned_rates_df.cluster_ID.isin(single_units)]
#             # z score normalise neurons across trials
#             norm_aligned_rates_df = (
#                 aligned_rates_df.set_index(["cluster_unique_ID", "trial"])
#                 .firing_rate.reset_index("trial")
#                 .pivot(columns="trial")  # stack trials [neurons, timepoints x trials]
#                 .sort_index(axis=1, level=1)
#                 .apply(zscore, axis=1)
#             )
#             # average neurons over trials to get population activity per trial
#             pop_activity_df = norm_aligned_rates_df.stack(future_stack=True).groupby("trial").mean(0)
#             pop_activity_dfs.append(pop_activity_df.reset_index(drop=True))
#         subject_means.append(pd.concat(pop_activity_dfs, axis=0).mean(0))
#     population_activity_df = pd.concat(subject_means, axis=1).T
#     population_activity_df.index = SUBJECT_IDS
#     return population_activity_df


# def get_event_aligned_population_acitivity(
#     aligned_to="trial",
#     normalise_clusters="max",
#     normalise_sessions="zscore",
#     plot=True,
# ):
#     """"""
#     av_rates = []
#     data_structure = aligned_to + "_aligned_rates_df"
#     for subject in SUBJECT_IDS:
#         sessions = gs.get_maze_sessions(
#             subject_IDs=[subject], maze_names="all", days_on_maze="late", with_data=[data_structure, "cluster_metrics"]
#         )
#         session_av_rates = []
#         for session in sessions:
#             aligned_rates_df = getattr(session, data_structure)
#             # only include single units
#             cluster_metrics = session.cluster_metrics
#             single_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID.values
#             aligned_rates_df = aligned_rates_df[aligned_rates_df.cluster_ID.isin(single_units)]
#             # average neurons over trials
#             trial_average_rates = (
#                 aligned_rates_df.set_index("cluster_unique_ID")
#                 .groupby("cluster_unique_ID")
#                 .firing_rate.mean()
#                 .firing_rate
#             )
#             if normalise_clusters == "max":
#                 if aligned_to == "event":
#                     for event in ["cue_aligned", "reward_aligned"]:
#                         trial_average_rates[event] = trial_average_rates[event].apply(lambda x: x / x.max(), axis=1)
#                 else:
#                     trial_average_rates = trial_average_rates.apply(lambda x: x / x.max(), axis=1)
#             population_average_rates = aligned_rates_df.firing_rate.mean(axis=0)
#             session_av_rates.append(population_average_rates)
#         subject_av_rates = pd.concat(session_av_rates, axis=1).T
#         if normalise_sessions == "max":
#             subject_av_rates = subject_av_rates.apply(lambda x: x / x.max(), axis=1)
#         elif normalise_sessions == "zscore":
#             subject_av_rates = subject_av_rates.apply(zscore, axis=1)
#         subject_av_rates = subject_av_rates.mean()
#         av_rates.append(subject_av_rates)
#     population_average_rates = pd.concat(av_rates, axis=1).T
#     population_average_rates.index = SUBJECT_IDS
#     if plot:
#         if aligned_to == "event":
#             _plot_population_event_aligned_activity(population_average_rates)
#         elif aligned_to == "trial":
#             _plot_population_trial_aligned_activity(population_average_rates)
#     return population_average_rates


# def _plot_population_event_aligned_activity(population_average_rates, smooth_SD=1, ax=None, color="black"):
#     if ax is None:
#         f, axes = plt.subplots(1, 2, figsize=(6, 3), clear=True, sharey=True)
#     ax.spines[["right", "top"]].set_visible(False)
#     for i, event in enumerate(["cue_aligned", "reward_aligned"]):
#         event_aligned_activity = population_average_rates[event]
#         time = event_aligned_activity.columns.to_numpy(dtype=float)
#         y = event_aligned_activity.mean(axis=0).to_numpy()
#         sem = event_aligned_activity.sem(axis=0).to_numpy()
#         if smooth_SD:
#             y = gaussian_filter1d(y, smooth_SD * FRAME_RATE)
#             sem = gaussian_filter1d(sem, smooth_SD * FRAME_RATE)
#         axes[i].plot(time, y, color=color)
#         axes[i].fill_between(time, y - sem, y + sem, color=color, alpha=0.5)
#         axes[i].axvline(0, color="k", alpha=0.2, zorder=0)
#         axes[i].set_xlabel(f"{event} time (s)")
#         axes[i].spines["right"].set_visible(False)
#         axes[i].spines["top"].set_visible(False)
#         if i == 0:
#             axes[i].set_ylabel("Pop. Rate (z-score)")
#     return


# def _plot_population_trial_aligned_activity(population_average_rates, smooth_SD=5, t_min=-2, color="black", ax=None):
#     if ax is None:
#         f, ax = plt.subplots(1, 1, figsize=(6, 2), clear=True)
#     time = population_average_rates.columns.to_numpy(dtype=float)
#     if smooth_SD:
#         population_average_rates = population_average_rates.apply(
#             lambda x: gaussian_filter1d(x, smooth_SD), axis=1
#         ).apply(pd.Series)
#     ax.spines[["right", "top"]].set_visible(False)
#     mask = time > t_min
#     time = time[mask]
#     y = population_average_rates.mean(axis=0).to_numpy()
#     y = y[mask]
#     sem = population_average_rates.sem(axis=0).to_numpy()
#     sem = sem[mask]
#     ax.plot(time, y, color=color)
#     ax.fill_between(time, y - sem, y + sem, color=color, alpha=0.2)
#     for x in INTRA_TRIAL_INTERVAL_TIMES.values():
#         ax.axvline(x, color="k", ls="--", alpha=0.2, zorder=0)
#     ax.set_xlabel("Time (s)")
#     ax.set_xticks([float(x) for x in INTRA_TRIAL_INTERVAL_TIMES.values()])
#     ax.set_xticklabels(["Cue", "Reward", "ERC", "ITI"])
#     ax.set_ylabel("Pop. Rate (z-score)")
#     return


# # %% Goal directed vs Non-goal directed cue bump


# def get_cue_aligned_population_activity_residual(normalise_clusters=False, normalise_sessions=False, plot=True):
#     """
#     Fix ylims, best results without normalisation
#     """
#     dfs = []
#     for subject in SUBJECT_IDS:
#         print(subject)
#         conditions = ([], [], [])  # goal-directed, non-goal-directed, not-moving
#         sessions = gs.get_maze_sessions(
#             subject_IDs=[subject],
#             maze_names="all",
#             days_on_maze="late",
#             with_data=["event_aligned_rates_df", "cluster_metrics", "navigation_df"],
#             must_have_data=True,
#         )
#         for session in sessions:
#             aligned_rates_df = session.event_aligned_rates_df
#             # only include single units
#             cluster_metrics = session.cluster_metrics
#             single_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID.values
#             aligned_rates_df = aligned_rates_df[aligned_rates_df.cluster_ID.isin(single_units)]
#             # split trials into goal_direced, non_goal_directed and not_moving
#             ddtg_df = ddtg.get_session_delta_dtg(session, DDTG_WINDOW)
#             goal_directed_trials = ddtg_df[ddtg_df.cue_aligned_time.mean(1).le(DDTG_LOWER_THRES)].trial.values.astype(
#                 int
#             )
#             non_goal_directed_trials = ddtg_df[
#                 ddtg_df.cue_aligned_time.mean(1).ge(DDTG_UPPER_THRES)
#             ].trial.values.astype(int)
#             not_moving_trials = ddtg_df[ddtg_df.cue_aligned_time.mean(1) == 0].trial.values.astype(int)
#             # get average cluster responses in each condition
#             condition_dfs = [
#                 aligned_rates_df[aligned_rates_df.trial.isin(trials)]
#                 for trials in [goal_directed_trials, non_goal_directed_trials, not_moving_trials]
#             ]
#             # average neurons over trials
#             for i, df in enumerate(condition_dfs):
#                 if not df.empty:
#                     av_rates_df = df.groupby("cluster_unique_ID").firing_rate.mean().firing_rate.cue_aligned
#                     if normalise_clusters == "max":
#                         av_rates_df = av_rates_df.apply(lambda x: x / x.max(), axis=1)
#                     population_rate = av_rates_df.mean()
#                     conditions[i].append(population_rate)
#         subject_dfs = [pd.concat(condition, axis=1).T for condition in conditions]
#         # normalise across sessions
#         if normalise_sessions == "max":
#             subject_dfs = [condition.apply(lambda x: x / x.max(), axis=1) for condition in subject_dfs]
#         elif normalise_sessions == "zscore":
#             subject_dfs = [condition.apply(zscore, axis=1) for condition in subject_dfs]
#         conditions_df = pd.concat([df.mean() for df in subject_dfs], axis=1).T
#         conditions_df.columns = pd.MultiIndex.from_product([["time"], conditions_df.columns])
#         conditions_df[("condition", "")] = ["goal-directed", "non-goal-directed", "not-moving"]
#         conditions_df[("subject_ID", "")] = subject
#         dfs.append(conditions_df)
#     results_df = pd.concat(dfs, axis=0)
#     if plot:
#         _plot_cue_aligned_residuals(results_df, conditions=["goal-directed", "non-goal-directed"])
#     return results_df


# def _plot_cue_aligned_residuals(results_df, conditions=["goal-directed", "non-goal-directed"], window=(-1, 1), ax=None):
#     """ """
#     # prepare axes
#     if ax is None:
#         f, ax = plt.subplots(1, 1, figsize=(3, 3), clear=True)
#     ax.spines[["right", "top"]].set_visible(False)
#     ax.set_xlabel("Cue (s)")
#     ax.set_ylabel("Goal-dir. — non-goal-dir. \n Δ Firing Rate (Hz)")
#     ax.axvline(0, color="k", linestyle="--", alpha=0.2)
#     # prepare and plot data
#     time = results_df.time.columns.values.astype(float)
#     if not conditions:  # plot conditions separately
#         for condition, color in zip(results_df.condition.unique(), ["darkorange", "dodgerblue", "gray"]):
#             condition_df = results_df[results_df.condition == condition].time
#             mean = condition_df.mean()
#             sem = condition_df.sem()
#             ax.plot(time, mean, color=color, label=condition)
#             ax.fill_between(time, mean - sem, mean + sem, color=color, alpha=0.2)
#             ax.set_ylim(0, 0.1)
#     else:  # plot residual between condtions
#         residuals = np.zeros((len(SUBJECT_IDS), len(time)))
#         for i, subject in enumerate(SUBJECT_IDS):
#             subject_df = results_df[results_df.subject_ID == subject]
#             residual = (
#                 subject_df[subject_df.condition == conditions[0]].time.values
#                 - subject_df[subject_df.condition == conditions[1]].time.values
#             )
#             residuals[i, :] = residual.reshape(-1)
#         mean = residuals.mean(axis=0)
#         sem = residuals.std(axis=0) / np.sqrt(len(SUBJECT_IDS))
#         ax.plot(time, mean, color="purple", label=f"{conditions[0]} - {conditions[1]}")
#         ax.fill_between(time, mean - sem, mean + sem, color="purple", alpha=0.2)
#         ax.set_ylim(-0.1, 0.4)
#         ax.axhline(0, color="k", linestyle="--", alpha=0.5)

#     ax.set_xlim(*window)
#     ax.legend()


# %%
