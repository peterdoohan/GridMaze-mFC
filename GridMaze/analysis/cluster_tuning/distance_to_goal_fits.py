"""This module visualises cluster tuning to euclidean and geodesic distance to goal."""
# %% Imports
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .. import get_sessions as gs
from ...maze import plotting as mp
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib.gridspec as gridspec

# %% Global variables
ANALYSIS_DATA_PATH = "../data/analysis_data"

with open(os.path.join(ANALYSIS_DATA_PATH, "analysis_info.json"), "r") as infile:
    ANALYSIS_INFO = json.load(infile)

MAX_DISTANCES = ANALYSIS_INFO["trial_max_distance_85th_quantiles"]


#%%






# %% Old plotting functions
def plot_session_all_distances_to_goal(session, colormap="gist_ncar", smoothed=True, smooth_SD=2):
    """ """
    cluster_unique_IDs, f_plotting_variables = prepare_plot_distance_to_goal_tuning(
        session, ("distance_to_goal", "future")
    )
    cluster_unique_IDs, g_plotting_variables = prepare_plot_distance_to_goal_tuning(
        session, ("distance_to_goal", "geodesic")
    )
    cluster_unique_IDs, e_plotting_variables = prepare_plot_distance_to_goal_tuning(
        session, ("distance_to_goal", "euclidean")
    )
    cluster_unique_IDs, m_plotting_variables = prepare_plot_distance_to_goal_tuning(
        session, ("distance_to_goal", "manhattan")
    )
    cluster_unique_IDs, p_plotting_variables = prepare_plot_distance_to_goal_tuning(
        session, ("progress_to_goal", "path_length")
    )
    cluster_unique_IDs, t_plotting_variables = prepare_plot_distance_to_goal_tuning(
        session, ("progress_to_goal", "time")
    )
    for cluster_unique_ID in cluster_unique_IDs:
        fig = plt.figure(
            figsize=(20, 20),
            clear=True,
        )
        gs = gridspec.GridSpec(3, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])  # future distance
        ax2 = fig.add_subplot(gs[0, 1])  # geodesic distance
        ax3 = fig.add_subplot(gs[1, 0])  # euclidean distance
        ax4 = fig.add_subplot(gs[1, 1])  # manhattan distance
        ax5 = fig.add_subplot(gs[2, 0])  # path length progress
        ax6 = fig.add_subplot(gs[2, 1])  # time progress
        plot_cluster_distance_to_goal_tuning(
            ax1, cluster_unique_ID, "Future Distance", colormap, *f_plotting_variables, smoothed, smooth_SD
        )
        plot_cluster_distance_to_goal_tuning(
            ax2, cluster_unique_ID, "Geodesic Distance", colormap, *g_plotting_variables, smoothed, smooth_SD
        )
        plot_cluster_distance_to_goal_tuning(
            ax3, cluster_unique_ID, "Euclidean Distance", colormap, *e_plotting_variables, smoothed, smooth_SD
        )
        plot_cluster_distance_to_goal_tuning(
            ax4, cluster_unique_ID, "Manhattan Distance", colormap, *m_plotting_variables, smoothed, smooth_SD
        )
        plot_cluster_distance_to_goal_tuning(
            ax5, cluster_unique_ID, "Path length Progress", colormap, *p_plotting_variables, smoothed, smooth_SD
        )
        plot_cluster_distance_to_goal_tuning(
            ax6, cluster_unique_ID, "Time Progress", colormap, *t_plotting_variables, smoothed, smooth_SD
        )
    return


def plot_session_distance_to_goal_tuning(session, metrics, colormap="gist_ncar", smoothed=True, smooth_SD=2):
    cluster_unique_IDs, plotting_variables = prepare_plot_distance_to_goal_tuning(session, metrics)
    for cluster_unique_ID in cluster_unique_IDs[:1]:
        f, ax = plt.subplots(figsize=(6, 4), clear=True)
        plot_cluster_distance_to_goal_tuning(
            ax, cluster_unique_ID, metrics[-1], colormap, *plotting_variables, smoothed, smooth_SD
        )


def prepare_plot_distance_to_goal_tuning(
    session, metrics, moving_only=False, distance_bin_spacing=0.05, n_progress_bins=40  # m
):
    data_needed = ["navigation_df", "navigation_spike_rates_df", "cluster_metrics"]
    if not gs.check_session_has_data(session, data_needed):
        pass
    # raises error if necessary session attributes not available
    navigation_rates_df = session.get_navigation_activity_df(activity_type="firing_rate", cluster_type="good")
    cluster_unique_IDs = list(navigation_rates_df.firing_rate.columns)
    navigation_rates_df = navigation_rates_df[navigation_rates_df.trial_phase == "navigation"]
    if moving_only:
        navigation_rates_df = navigation_rates_df[navigation_rates_df.moving]
    if metrics[0] == "distance_to_goal":
        d = "geodesic" if metrics[1] == "future" else metrics[1]  # plot geo and future with same max distance
        max_distance = MAX_DISTANCES[d]  # remove frames where distance is above max distance (~5% of data)
        navigation_rates_df = navigation_rates_df[navigation_rates_df[metrics] < max_distance]
        n_bins = int(max_distance / distance_bin_spacing)
    elif metrics[0] == "progress_to_goal":
        n_bins = n_progress_bins
    binned_column = ("binned_" + metrics[0], metrics[1])
    navigation_rates_df[binned_column] = pd.cut(navigation_rates_df[metrics], bins=n_bins, include_lowest=True)
    # first average over frames at each distance over trials
    trial2goal = navigation_rates_df.set_index("trial").goal.dropna().to_dict()
    distance_trial_grouped_nav_rates_df = (
        navigation_rates_df.groupby([binned_column, ("trial", "")]).mean().reset_index()
    )
    distance_trial_grouped_nav_rates_df["goal"] = distance_trial_grouped_nav_rates_df.trial.map(trial2goal)
    # next average rates across trials split by goal for each distance bin
    distance_goal_grouped_nav_rates = distance_trial_grouped_nav_rates_df.set_index(
        [binned_column, ("goal", "")]
    ).groupby([binned_column, ("goal", "")])
    av_distance_goal_grouped_rates = distance_goal_grouped_nav_rates.mean().firing_rate
    sem_distance_goal_grouped_rates = distance_goal_grouped_nav_rates.sem().firing_rate
    # and finally average rates across all trials for each distance bin (for inset plot)
    distance_grouped_nav_rates = distance_trial_grouped_nav_rates_df.set_index([binned_column]).groupby([binned_column])
    av_distance_grouped_rates = distance_grouped_nav_rates.mean().firing_rate
    sem_distance_grouped_rates = distance_grouped_nav_rates.sem().firing_rate
    for df in [av_distance_goal_grouped_rates, sem_distance_goal_grouped_rates]:
        df.reset_index(inplace=True)
        df.rename(columns={binned_column: "distance", ("goal", ""): "goal"}, inplace=True)
        df.distance = [i.mid for i in df.distance.to_list()]
    for df in [av_distance_grouped_rates, sem_distance_grouped_rates]:
        df.reset_index(inplace=True)
        df.rename(columns={binned_column: "distance"}, inplace=True)
        df.distance = [i.mid for i in df.distance.to_list()]
    plotting_variables = (
        session.simple_maze(),  # simple_maze
        np.sort(navigation_rates_df.goal.unique()),  # goals
        av_distance_goal_grouped_rates,
        sem_distance_goal_grouped_rates,
        av_distance_grouped_rates,
        sem_distance_grouped_rates,
    )
    return cluster_unique_IDs, plotting_variables


def plot_cluster_distance_to_goal_tuning(
    ax,
    cluster_unique_ID,
    distance,
    colormap,  # plot specific info
    simple_maze,
    goals,  # maze info
    av_distance_goal_grouped_rates,
    sem_distance_goal_grouped_rates,  # main plot info
    av_distance_grouped_rates,
    sem_distance_grouped_rates,  # inset plot info
    smoothed,
    smooth_SD,
):
    plot_cluster_distance_to_goal_tuning_by_goal(
        ax,
        cluster_unique_ID,
        distance,
        colormap,  # plot specific info
        goals,  # maze info
        av_distance_goal_grouped_rates,
        sem_distance_goal_grouped_rates,
        smoothed,
        smooth_SD,
    )  # main plot info
    plot_inset_cluster_av_distance_to_goal_tuning(
        ax,
        cluster_unique_ID,
        av_distance_grouped_rates,
        sem_distance_grouped_rates,
        smoothed,
        smooth_SD,
    )
    plot_inset_maze_legend(
        ax,
        simple_maze,
        colormap,
        goals,
    )


# %% Sub-functions


def plot_cluster_distance_to_goal_tuning_by_goal(
    ax,
    cluster_unique_ID,
    distance,
    colormap,  # plot specific info
    goals,  # maze info
    av_distance_goal_grouped_rates,
    sem_distance_goal_grouped_rates,
    smoothed,
    smooth_SD,
):  # main plot info
    distance_bin_midpoints = np.sort(av_distance_goal_grouped_rates.distance.unique())
    goal2color = mp.get_goal2standard_color(colormap)
    f_max = 0
    for goal in goals:
        color = goal2color[goal]
        goal_mask = av_distance_goal_grouped_rates.goal == goal
        av_cluster_distance_goal_grouped_rates = av_distance_goal_grouped_rates[goal_mask][cluster_unique_ID]
        sem_cluster_distance_goal_grouped_rates = sem_distance_goal_grouped_rates[goal_mask][cluster_unique_ID]
        if smoothed:
            av_cluster_distance_goal_grouped_rates = gaussian_filter1d(
                av_cluster_distance_goal_grouped_rates, sigma=smooth_SD
            )
            sem_cluster_distance_goal_grouped_rates = gaussian_filter1d(
                sem_cluster_distance_goal_grouped_rates, sigma=smooth_SD
            )
        goal_f_max = np.nanmax(av_cluster_distance_goal_grouped_rates) + np.nanmax(
            sem_cluster_distance_goal_grouped_rates
        )
        if goal_f_max > f_max:
            f_max = goal_f_max
        ax.plot(distance_bin_midpoints, av_cluster_distance_goal_grouped_rates, color=color, linewidth=0.85)
        ax.fill_between(
            distance_bin_midpoints,
            av_cluster_distance_goal_grouped_rates - sem_cluster_distance_goal_grouped_rates,
            av_cluster_distance_goal_grouped_rates + sem_cluster_distance_goal_grouped_rates,
            alpha=0.1,
            color=color,
        )
        # plot formatting
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel(distance + " to Goal (m)")
        ax.set_ylabel("Firing rate (Hz)")
        ax.set_xlim(0, np.max(distance_bin_midpoints))
        ax.set_ylim(0, 1.5 * f_max)
        ax.set_title(f"{cluster_unique_ID}", fontdict={"family": "Courier", "size": 10}, loc="left", pad=10, x=0)


def plot_inset_cluster_av_distance_to_goal_tuning(
    ax,
    cluster_unique_ID,
    av_distance_grouped_rates,
    sem_distance_grouped_rates,
    smoothed,
    smooth_SD,
):
    distance_bin_midpoints = np.sort(av_distance_grouped_rates.distance.unique())
    inset_axis = inset_axes(
        ax,
        width="30%",
        height="30%",
        bbox_to_anchor=(0.28, -0.015, 0.70, 1),
        bbox_transform=ax.transAxes,
        loc="upper center",
    )
    av_cluster_distance_grouped_rates = av_distance_grouped_rates[cluster_unique_ID]
    sem_cluster_distance_grouped_rates = sem_distance_grouped_rates[cluster_unique_ID]
    if smoothed:
        av_cluster_distance_grouped_rates = gaussian_filter1d(av_cluster_distance_grouped_rates, sigma=smooth_SD)
        sem_cluster_distance_grouped_rates = gaussian_filter1d(sem_cluster_distance_grouped_rates, sigma=smooth_SD)
    inset_axis.plot(distance_bin_midpoints, av_cluster_distance_grouped_rates, color="k")
    inset_axis.fill_between(
        distance_bin_midpoints,
        av_cluster_distance_grouped_rates - sem_cluster_distance_grouped_rates,
        av_cluster_distance_grouped_rates + sem_cluster_distance_grouped_rates,
        alpha=0.1,
        color="k",
    )
    # plot formatting
    inset_axis.tick_params(axis="both", which="major", labelsize=7)
    inset_axis.set_xlim(0, np.max(distance_bin_midpoints))
    inset_axis.set_ylim(
        0, np.nanmax(av_cluster_distance_grouped_rates) + 1.5 * np.nanmax(sem_cluster_distance_grouped_rates)
    )
    for spine in ["left", "right", "top", "bottom"]:
        inset_axis.spines[spine].set_linewidth(0.5)


def plot_inset_maze_legend(ax, simple_maze, colormap, goals):
    inset_axis = inset_axes(
        ax,
        width="35%",
        height="35%",
        bbox_to_anchor=(0.065, 0.01, 1, 1),
        bbox_transform=ax.transAxes,
        loc="upper right",
    )
    mp.plot_simple_maze_for_figure_legend(simple_maze, inset_axis, goals, colormap)
