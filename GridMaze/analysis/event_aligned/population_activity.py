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

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, ANALYSIS_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(ANALYSIS_INFO_PATH / "intra_trial_interval_times.json", "r") as input_file:
    INTRA_TRIAL_INTERVAL_TIMES = json.load(input_file)

FRAME_RATE = 60

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
