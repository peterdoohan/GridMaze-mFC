# %% Imports
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore

from .. import get_sessions as gs


# %% Global variables
with open("../data/experiment_info.json") as input_file:
    EXP_INFO = json.load(input_file)

ANALYSIS_DATA_PATH = "../data/analysis_data"
with open(os.path.join(ANALYSIS_DATA_PATH, "analysis_info.json"), "r") as infile:
    ANALYSIS_INFO = json.load(infile)

ANALYSIS_INFO_PATH = Path("../data/analysis_data/analysis_info")
# %% Functions


def _plot_distance_split_population_cue_aligned_activity(population_activity_split_by_distance_df):
    f, ax = plt.subplots(1, 1, figsize=(6, 3), clear=True)
    time = population_activity_split_by_distance_df.population_firing_rate.columns.to_numpy(dtype=float)
    for distance_label, color in zip(["short", "medium", "long"], ["blue", "green", "red"]):
        population_average_rates = population_activity_split_by_distance_df[
            population_activity_split_by_distance_df.distance_label == distance_label
        ].population_firing_rate
        y = population_average_rates.mean(axis=0).to_numpy()
        sem = population_average_rates.sem(axis=0).to_numpy()
        ax.plot(time, y, color=color, label=distance_label)
        ax.fill_between(time, y - sem, y + sem, color=color, alpha=0.5)
    ax.axvline(0, color="k", linewidth=1, ls="--", alpha=0.5, zorder=0)
    ax.set_xlabel("Cue-aligned Time (s)")
    ax.set_xlim(-0.25, 1)
    ax.set_ylabel("Population average firing rate")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend()
    return


def get_population_average_activity_split_by_distance_to_goal(distance_metric="geodesic", cue_window=(-0.25, 1)):
    """"""
    distance_labels = ["short", "medium", "long"]
    distance2subject_av_rates = {distance_label: [] for distance_label in distance_labels}
    for subject in EXP_INFO["subject_IDs"]:
        print(f"Processing subject {subject}...")
        sessions = gs.get_sessions(
            subject_IDs=[subject],
            maze_number="all",
            day_on_maze="late",
            with_data=["event_aligned_rates_df", "navigation_df"],
        )
        distance2session_av_rates = {distance_label: [] for distance_label in distance_labels}
        for session in sessions:
            distance_at_cue_df = get_distance_to_goal_at_cue(session, distance_metric=distance_metric)
            event_aligned_rates_df = session.event_aligned_rates_df
            event_aligned_rates_df = event_aligned_rates_df[event_aligned_rates_df.cluster_type == "good"]
            for distance_label in distance_labels:
                applicable_trials = distance_at_cue_df[distance_at_cue_df.distance_label == distance_label].trial
                distance_split_event_aligned_rates_df = event_aligned_rates_df[
                    event_aligned_rates_df.trial.isin(applicable_trials)
                ]
                event_averaged_rates_df = (
                    distance_split_event_aligned_rates_df.set_index("cluster_unique_ID")
                    .groupby("cluster_unique_ID")
                    .mean()
                    .firing_rate.cue_aligned
                )
                event_averaged_rates_df = event_averaged_rates_df.apply(
                    lambda x: x / x.max(), axis=1
                )  # normalise by max firing rate
                population_average_rates = event_averaged_rates_df.mean(axis=0)
                population_average_rates.index = population_average_rates.index.astype(float)
                if cue_window:
                    population_average_rates = population_average_rates[
                        np.logical_and.reduce(
                            [
                                population_average_rates.index > cue_window[0],
                                population_average_rates.index < cue_window[1],
                            ]
                        )
                    ]
                distance2session_av_rates[distance_label].append(population_average_rates)

        for distance_label in distance_labels:
            subject_population_av_rates = pd.concat(distance2session_av_rates[distance_label], axis=1).T
            subject_population_av_rates = subject_population_av_rates.apply(
                zscore, axis=1
            )  # zscore normalise across sessions
            distance2subject_av_rates[distance_label].append(subject_population_av_rates.mean())
    population_activity_split_by_distance_df = []
    for distance_label in distance_labels:
        distance_population_activity_df = pd.concat(distance2subject_av_rates[distance_label], axis=1).T
        distance_population_activity_df.columns = pd.MultiIndex.from_product(
            [["population_firing_rate"], distance_population_activity_df.columns]
        )
        distance_population_activity_df[("subject_ID", "")] = EXP_INFO["subject_IDs"]
        distance_population_activity_df[("distance_label", "")] = distance_label
        population_activity_split_by_distance_df.append(distance_population_activity_df)
    return pd.concat(population_activity_split_by_distance_df)


def get_distance_to_goal_at_cue(session, distance_metric="geodesic", add_distance_category=True):
    """ """
    try:
        with open(ANALYSIS_INFO_PATH / f"distance_at_cue_{distance_metric}_quantiles.json", "r") as infile:
            distance_quantiles = json.load(infile)
            distance_quantiles = {float(k): v for k, v in distance_quantiles.items()}
    except FileNotFoundError:
        print(f"{distance_metric} distance quantiles not found, calculating now...")
        distance_at_cue_distribution = get_distance_at_cue_distribution(distance_metric, save_quantiles=[0.33, 0.67])
        distance_quantiles = distance_at_cue_distribution.quantile([0.33, 0.67]).to_dict()
    distance_cutoffs = [0] + list(distance_quantiles.values()) + [np.inf]
    distance_labels = ["short", "medium", "long"]
    cue_distances_to_goal = []
    navigation_df = session.navigation_df
    trials = navigation_df.trial.dropna().unique()
    for trial in trials:
        # cue presentation defines onset of navigation
        navigation_frames = navigation_df[
            np.logical_and.reduce([(navigation_df.trial == trial), (navigation_df.trial_phase == "navigation")])
        ]
        if len(navigation_frames) == 0:
            continue
        cue_distance_to_goal = navigation_frames.distance_to_goal[distance_metric].iloc[
            0
        ]  # first frame of navigation (cue presentation)
        cue_distances_to_goal.append({"trial": trial, "cue_distance_to_goal": cue_distance_to_goal})
        distance_at_cue_df = pd.DataFrame(cue_distances_to_goal)
    if not add_distance_category:
        return distance_at_cue_df
    else:
        distance_at_cue_df["distance_label"] = pd.cut(
            distance_at_cue_df["cue_distance_to_goal"], bins=distance_cutoffs, labels=distance_labels, right=False
        )
        return distance_at_cue_df


def get_distance_at_cue_distribution(distance_metric, late_sessions=True, save_quantiles=False):
    """Returns the distribution of distances to goal at cue over all session in a pd.Series."""
    days_on_maze = "late" if late_sessions else "all"
    sessions = gs.get_sessions(
        subject_IDs="all", maze_number="all", day_on_maze=days_on_maze, with_data=["navigation_df"]
    )
    distances_at_cue = []
    for session in sessions:
        distance_at_cue_df = get_distance_to_goal_at_cue(
            session, distance_metric=distance_metric, add_distance_category=False
        )
        distances_at_cue.append(distance_at_cue_df.cue_distance_to_goal)
    distance_at_cue_distribution = pd.concat(distances_at_cue).reset_index(drop=True)
    if save_quantiles:
        distance_quantiles = distance_at_cue_distribution.quantile(save_quantiles).to_dict()
        filename = f"distance_at_cue_{distance_metric}_quantiles.json"
        with open((ANALYSIS_INFO_PATH / filename), "w") as outfile:
            json.dump(distance_quantiles, outfile)
    return distances_at_cue


# %%


def get_population_average_aligned_activity(
    plot=True, aligned_to="event", normalise_clusters="max", normalise_sessions="max"
):
    """"""
    av_rates = []
    data_structure = aligned_to + "_aligned_rates_df"
    for subject in EXP_INFO["subject_IDs"]:
        sessions = gs.get_sessions(
            subject_IDs=[subject], maze_number="all", day_on_maze="late", with_data=[data_structure]
        )
        session_av_rates = []
        for session in sessions:
            aligned_rates_df = getattr(session, data_structure)
            aligned_rates_df = aligned_rates_df[aligned_rates_df.cluster_type == "good"]
            # average neurons over trials
            trial_average_rates = (
                aligned_rates_df.set_index("cluster_unique_ID").groupby("cluster_unique_ID").mean().firing_rate
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
    population_average_rates.index = EXP_INFO["subject_IDs"]
    if plot:
        if aligned_to == "event":
            _plot_population_event_aligned_activity(population_average_rates)
        elif aligned_to == "trial":
            _plot_population_trial_aligned_activity(population_average_rates)
    return population_average_rates


# %% Plotting
def _plot_population_event_aligned_activity(population_average_rates):
    f, axes = plt.subplots(1, 2, figsize=(6, 3), clear=True, sharey=True)
    for i, event in enumerate(["cue_aligned", "reward_aligned"]):
        event_aligned_activity = population_average_rates[event]
        time = event_aligned_activity.columns.to_numpy(dtype=float)
        y = event_aligned_activity.mean(axis=0).to_numpy()
        sem = event_aligned_activity.sem(axis=0).to_numpy()
        axes[i].plot(time, y, color="orange")
        axes[i].fill_between(time, y - sem, y + sem, color="orange", alpha=0.5)
        axes[i].axvline(0, color="k", linewidth=1, alpha=0.5, zorder=0)
        axes[i].set_xlabel(f"{event} time (s)")
        axes[i].spines["right"].set_visible(False)
        axes[i].spines["top"].set_visible(False)
        if i == 0:
            axes[i].set_ylabel("Population average firing rate")
    return


def _plot_population_trial_aligned_activity(population_average_rates):
    f, ax = plt.subplots(1, 1, figsize=(6, 3), clear=True)
    time = population_average_rates.columns.to_numpy(dtype=float)
    y = population_average_rates.mean(axis=0).to_numpy()
    sem = population_average_rates.sem(axis=0).to_numpy()
    ax.plot(time, y, color="orange")
    ax.fill_between(time, y - sem, y + sem, color="orange", alpha=0.5)
    for x in EXP_INFO["intra_trial_interval_times"]:
        ax.axvline(x, color="k", linewidth=1, ls="--", alpha=0.5, zorder=0)
    ax.set_xlabel("Time (s)")
    ax.set_xticks([float(x) for x in EXP_INFO["intra_trial_interval_times"]])
    ax.set_xticklabels(["Cue", "Reward", "ITI", "end"])
    ax.set_ylabel("Population average firing rate")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return
