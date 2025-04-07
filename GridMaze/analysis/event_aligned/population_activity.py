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
