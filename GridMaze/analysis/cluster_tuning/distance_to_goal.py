"""New library for plotting distance to goal aligned rates"""

# %% Imports
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from ..core import get_clusters as gc

from ...maze import plotting as mp
from scipy.ndimage import gaussian_filter1d

# %% Global Variables (TODO: Update to store in analysis info folder)
ANALYSIS_DATA_PATH = "../data/analysis_data"


# with open(os.path.join(ANALYSIS_DATA_PATH, "analysis_info.json"), "r") as infile:
#     ANALYSIS_INFO = json.load(infile)

# MAX_DISTANCES = ANALYSIS_INFO["trial_max_distance_85th_quantiles"]

# %% Functions


def plot_session_distance_to_goal_tuning(session, metrics=("distance_to_goal", "geodesic"), goal_stratified=False):
    navigation_rates_df = session.get_navigation_activity_df(type="rates", cluster_kwargs={"single_units": True})
    distance_info = navigation_rates_df[[("goal", ""), ("trial", ""), ("moving", ""), metrics]]
    distance_info = distance_info.droplevel(1, axis=1)
    cluster_unique_IDs = navigation_rates_df.firing_rate.columns.to_numpy()
    for cluster in cluster_unique_IDs[:1]:
        cluster_rates = navigation_rates_df.xs(cluster, level=1, axis=1)
        distance_rates_df = pd.concat([distance_info, cluster_rates], axis=1)
        distance_tuning_df = get_distance_to_goal_tuning_df(distance_rates_df, metrics=metrics)
        plot_distance_tuning(distance_tuning_df, metrics, goal_stratified=goal_stratified)


def get_distance_to_goal_tuning_df(
    distance_rates_df, metrics=("distance_to_goal", "geodesic"), bin_spacing=0.04, n_bins=40, moving_only=False
):
    """ """
    distance_rates_df = distance_rates_df.copy()
    trial2goal = distance_rates_df.set_index("trial").goal.dropna().to_dict()
    # deal with moving only
    if moving_only:
        distance_rates_df = distance_rates_df[distance_rates_df.moving]
    # remove frames where distance is above max (treat as outliers)
    if metrics[0] == "distance_to_goal":
        max_distance = MAX_DISTANCES[metrics[1]]
        n_bins = int(max_distance / bin_spacing)
        distance_rates_df = distance_rates_df[distance_rates_df[metrics[0]] < max_distance]
    # bin distances
    distance_rates_df["distance_bin"] = pd.cut(
        distance_rates_df[metrics[0]], bins=n_bins, include_lowest=True
    ).to_numpy()
    # average over frames in each bin over trials
    trial_av_rates = distance_rates_df.groupby(["trial", "distance_bin"], observed=True).firing_rate.mean().unstack()
    # organise tuning df
    tuning_df = pd.DataFrame()
    tuning_df[("trial", "")] = trial_av_rates.index
    tuning_df[("goal", "")] = trial_av_rates.index.map(trial2goal)
    tuning_df.columns = pd.MultiIndex.from_tuples(tuning_df.columns)
    trial_av_rates.columns = pd.MultiIndex.from_tuples([("distance", c.mid) for c in trial_av_rates.columns])
    tuning_df = pd.concat([tuning_df, trial_av_rates.reset_index(drop=True)], axis=1)
    return tuning_df


def plot_distance_tuning(distance_tuning_df, metrics, goal_stratified=False, smooth_SD=1, ax=None, color="black"):
    """"""
    # format axis
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(6, 3), clear=True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Firing Rate (Hz)")
    ax.set_xlabel(f"{metrics[0]}: {metrics[1]}")
    if metrics[0] == "distance_to_goal":
        ax.set_xlim(0, MAX_DISTANCES[metrics[1]])
    else:  # progress_to_goal
        ax.set_xlim(0, 1)
    # process data for plotting
    if not goal_stratified:
        mean_rates = distance_tuning_df.distance.mean(axis=0)
        sem_rates = distance_tuning_df.distance.sem(axis=0)
        distances = mean_rates.index.to_numpy().astype(float)
        mean = mean_rates.to_numpy()
        sem = sem_rates.to_numpy()
        if smooth_SD:
            mean = gaussian_filter1d(mean, smooth_SD)
            sem = gaussian_filter1d(sem, smooth_SD)
        _plot_distance_tuning(mean, sem, distances, ax, color)
    else:
        goal2color = mp.get_goal2standard_color()
        for goal in distance_tuning_df.goal.unique():
            mean_rates = distance_tuning_df[distance_tuning_df.goal == goal].distance.mean(axis=0)
            sem_rates = distance_tuning_df[distance_tuning_df.goal == goal].distance.sem(axis=0)
            distances = mean_rates.index.to_numpy().astype(float)
            mean = mean_rates.to_numpy()
            sem = sem_rates.to_numpy()
            if smooth_SD:
                mean = gaussian_filter1d(mean, smooth_SD)
                sem = gaussian_filter1d(sem, smooth_SD)
            _plot_distance_tuning(mean, sem, distances, ax, goal2color[goal])
    return


def _plot_distance_tuning(mean, sem, distances, ax, color):
    ax.plot(distances, mean, color=color)
    ax.fill_between(distances, mean - sem, mean + sem, color=color, alpha=0.2)
    return
