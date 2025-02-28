"""This module generates plots for visualising trial and event aligned neural activity."""
# %% Imports
import json
import numpy as np
import matplotlib.pyplot as plt
from .. import get_sessions as gs
from ...maze import plotting as mp
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# %% Global variables
plt.rcParams["pdf.fonttype"] = 42
with open("../data/experiment_info.json") as input_file:
    EXP_INFO = json.load(input_file)

# %% Plot event aligned neural activity


def plot_session_event_aligned_rates(session, colormap="gist_ncar"):
    """
    This function plots event aligned neural activity for every good (KSLabel) cluster in a session.
    This function class the prepare_plot_event_aligned_rates function to gather, cluster grouped data (for inset plot) and cluster & goal grouped data
    (for the main plot) and outputs the list of clusters to plot as well as a tuple of the plotting variables. A plotting axis, individual cluster IDs,
    plot colormap, and these plotting variables are then passed into the plot_cluster_event_aligned_rates function to generate the plot for a single cluster.
    """
    cluster_unique_IDs, plotting_variables = prepare_plot_event_aligned_rates(session)
    for cluster_unique_ID in cluster_unique_IDs:
        f, ax = plt.subplots(figsize=(10, 6), clear=True)
        plot_cluster_event_aligned_rates(ax, cluster_unique_ID, colormap, *plotting_variables)  # plot specific info


def prepare_plot_event_aligned_rates(session, smoothed=True, smooth_SD=15, cluster_type="good"):
    # check relevant data is available in session object
    data_needed = ["event_aligned_rates_df"]
    if not gs.check_session_has_data(session, data_needed):
        pass
    simple_maze = session.simple_maze()
    event_aligned_rates_df = session.event_aligned_rates_df
    if cluster_type == "good":
        event_aligned_rates_df = event_aligned_rates_df[event_aligned_rates_df.cluster_type == "good"]
    cluster_unique_IDs = event_aligned_rates_df.cluster_unique_ID.unique()
    goals = np.sort(event_aligned_rates_df.goal.unique())
    if smoothed:
        event_aligned_rates = event_aligned_rates_df.copy()
        firing_rate_data = event_aligned_rates.loc[:, ("firing_rate", slice(None), slice(None))].to_numpy()
        event_aligned_rates.loc[:, ("firing_rate", slice(None), slice(None))] = gaussian_filter1d(
            firing_rate_data, sigma=smooth_SD
        )
        event_aligned_rates_df = event_aligned_rates
    # data grouped by cluster
    cluster_grouped_event_aligned_rates = event_aligned_rates_df.set_index("cluster_unique_ID").firing_rate.groupby(
        ["cluster_unique_ID"]
    )
    av_clusters_event_aligned_rates = cluster_grouped_event_aligned_rates.mean()
    sem_clusters_event_aligned_rates = cluster_grouped_event_aligned_rates.sem()
    # data group by cluster and goal
    cluster_goal_grouped_event_aligned_rates = event_aligned_rates.set_index(
        ["cluster_unique_ID", "goal"]
    ).firing_rate.groupby(["cluster_unique_ID", "goal"])
    av_clusters_goals_event_aligned_rates = cluster_goal_grouped_event_aligned_rates.mean()
    sem_clusters_goals_event_aligned_rates = cluster_goal_grouped_event_aligned_rates.sem()
    plotting_variables = (
        simple_maze,
        goals,  # maze info
        av_clusters_event_aligned_rates,
        sem_clusters_event_aligned_rates,  # cluster grouped data
        av_clusters_goals_event_aligned_rates,
        sem_clusters_goals_event_aligned_rates,
    )  # cluster & goal grouped data
    return cluster_unique_IDs, plotting_variables


def plot_cluster_event_aligned_rates(
    ax,
    cluster_unique_ID,
    colormap,  # plot specific info
    simple_maze,
    goals,  # maze info
    av_clusters_event_aligned_rates,
    sem_clusters_event_aligned_rates,  # cluster grouped data
    av_clusters_goals_event_aligned_rates,
    sem_clusters_goals_event_aligned_rates,
):  # cluster & goal grouped data
    ax.axis("off")
    plot_event_aligned_activity_by_goal(
        ax,
        cluster_unique_ID,
        colormap,
        goals,
        av_clusters_goals_event_aligned_rates,
        sem_clusters_goals_event_aligned_rates,
    )
    plot_inset_event_aligned_activity(
        ax, cluster_unique_ID, av_clusters_event_aligned_rates, sem_clusters_event_aligned_rates
    )
    plot_inset_maze_legend(ax, colormap, simple_maze, goals)
    return


def plot_event_aligned_activity_by_goal(
    ax,
    cluster_unique_ID,
    colormap,
    goals,
    av_clusters_goals_event_aligned_rates,
    sem_clusters_goals_event_aligned_rates,
):
    goal2color = mp.get_goal2standard_color(colormap)
    av_cluster_goals_cue_aligned_rates = av_clusters_goals_event_aligned_rates.loc[cluster_unique_ID].cue_aligned
    sem_cluster_goals_cue_aligned_rates = sem_clusters_goals_event_aligned_rates.loc[cluster_unique_ID].cue_aligned
    av_cluster_goals_reward_aligned_rates = av_clusters_goals_event_aligned_rates.loc[cluster_unique_ID].reward_aligned
    sem_cluster_goals_reward_aligned_rates = sem_clusters_goals_event_aligned_rates.loc[
        cluster_unique_ID
    ].reward_aligned
    f_max = np.max(
        [np.max(av_cluster_goals_cue_aligned_rates, axis=1), np.max(av_cluster_goals_reward_aligned_rates, axis=1)]
    ) + 1.5 * np.nanmax(
        [
            np.nanmax(sem_cluster_goals_cue_aligned_rates, axis=1),
            np.nanmax(sem_cluster_goals_reward_aligned_rates, axis=1),
        ]
    )
    cue_aligned_timepoints = av_cluster_goals_cue_aligned_rates.columns.to_numpy().astype(float)
    reward_aligned_timepoints = av_cluster_goals_reward_aligned_rates.columns.to_numpy().astype(float)
    # set up plotting
    axis_left = inset_axes(
        ax,
        width="50%",
        height="100%",
        bbox_to_anchor=(0.0, -0.025, 1, 1),
        bbox_transform=ax.transAxes,
        loc="lower left",
    )
    ax.set_title(f"{cluster_unique_ID}", fontdict={"family": "Courier", "size": 10}, loc="left", pad=10, x=0.02)
    axis_left.set_ylim(0, f_max)
    axis_left.set_xlim(cue_aligned_timepoints[0], cue_aligned_timepoints[-1])
    axis_left.set_xticks([-10, -5, -0, 5, 10])
    axis_left.set_xlabel("Cue-aligned Time (s)", size=12, labelpad=1)
    axis_left.set_ylabel("Firing Rate (Hz)", size=12, labelpad=5)
    axis_left.spines["right"].set_visible(False)
    axis_left.spines["top"].set_visible(False)
    axis_left.axvline(x=0, color="k", linestyle="--", linewidth=1)
    axis_right = inset_axes(
        ax,
        width="50%",
        height="100%",
        bbox_to_anchor=(0.04, -0.025, 1, 1),
        bbox_transform=ax.transAxes,
        loc="lower right",
    )
    axis_right.set_ylim(0, f_max)
    axis_right.set_xlim(reward_aligned_timepoints[0], reward_aligned_timepoints[-1])
    axis_right.set_xticks([-10, -5, -0, 5, 10])
    axis_right.set_xlabel("Reward-aligned Time (s)", size=12, labelpad=1)
    axis_right.set_yticks([])
    axis_right.set_yticklabels([])
    axis_right.spines["left"].set_visible(False)
    axis_right.spines["right"].set_visible(False)
    axis_right.spines["top"].set_visible(False)
    axis_right.axvline(x=0, color="k", linestyle="--", linewidth=1)
    for goal in goals:
        color = goal2color[goal]
        av_cluster_goal_cue_aligned_rates = av_cluster_goals_cue_aligned_rates.loc[goal].to_numpy()
        sem_cluster_goal_cue_aligned_rates = sem_cluster_goals_cue_aligned_rates.loc[goal].to_numpy()
        av_cluster_goal_reward_aligned_rates = av_cluster_goals_reward_aligned_rates.loc[goal].to_numpy()
        sem_cluster_goal_reward_aligned_rates = sem_cluster_goals_reward_aligned_rates.loc[goal].to_numpy()
        # plot cue aligned activity
        axis_left.plot(cue_aligned_timepoints, av_cluster_goal_cue_aligned_rates, color=color, linewidth=0.85)
        axis_left.fill_between(
            cue_aligned_timepoints,
            av_cluster_goal_cue_aligned_rates - sem_cluster_goal_cue_aligned_rates,
            av_cluster_goal_cue_aligned_rates + sem_cluster_goal_cue_aligned_rates,
            color=color,
            alpha=0.1,
        )
        axis_right.plot(reward_aligned_timepoints, av_cluster_goal_reward_aligned_rates, color=color, linewidth=0.85)
        axis_right.fill_between(
            reward_aligned_timepoints,
            av_cluster_goal_reward_aligned_rates - sem_cluster_goal_reward_aligned_rates,
            av_cluster_goal_reward_aligned_rates + sem_cluster_goal_reward_aligned_rates,
            color=color,
            alpha=0.1,
        )
    return


def plot_inset_event_aligned_activity(
    ax, cluster_unique_ID, av_clusters_event_aligned_rates, sem_clusters_event_aligned_rates, color="black"
):
    av_cluster_cue_aligned_rates = av_clusters_event_aligned_rates.loc[cluster_unique_ID].cue_aligned.to_numpy()
    sem_cluster_cue_aligned_rates = sem_clusters_event_aligned_rates.loc[cluster_unique_ID].cue_aligned.to_numpy()
    av_cluster_reward_aligned_rates = av_clusters_event_aligned_rates.loc[cluster_unique_ID].reward_aligned.to_numpy()
    sem_cluster_reward_aligned_rates = sem_clusters_event_aligned_rates.loc[cluster_unique_ID].reward_aligned.to_numpy()
    f_max = (
        np.max([np.max(av_cluster_cue_aligned_rates), np.max(av_cluster_reward_aligned_rates)])
        + np.max([np.max(sem_cluster_cue_aligned_rates), np.max(sem_cluster_reward_aligned_rates)]) / 2
    )
    inset_axis_left = inset_axes(
        ax, width="30%", height="30%", bbox_to_anchor=(0.21, 0, 0.5, 1), bbox_transform=ax.transAxes, loc="upper center"
    )
    inset_axis_right = inset_axes(
        ax, width="30%", height="30%", bbox_to_anchor=(0.32, 0, 0.5, 1), bbox_transform=ax.transAxes, loc="upper center"
    )
    # plot cue aligned activity
    inset_axis_left.axvline(x=len(av_cluster_cue_aligned_rates) / 2, color="grey", linestyle="--", linewidth=1)
    inset_axis_left.plot(av_cluster_cue_aligned_rates, color=color)
    inset_axis_left.fill_between(
        range(len(av_cluster_cue_aligned_rates)),
        av_cluster_cue_aligned_rates - sem_cluster_cue_aligned_rates,
        av_cluster_cue_aligned_rates + sem_cluster_cue_aligned_rates,
        color=color,
        alpha=0.2,
    )
    inset_axis_left.set_xticks([len(av_cluster_cue_aligned_rates) / 2])
    inset_axis_left.set_xticklabels([])
    inset_axis_left.tick_params(axis="both", which="major", labelsize=7)
    inset_axis_left.set_xlabel("Cue", size=7, labelpad=-1)
    inset_axis_left.set_ylim([0, f_max])
    # inset_axis_left.set_ylabel('Firing Rate (Hz)', size=7, labelpad=-1)
    inset_axis_left.spines["right"].set_visible(False)

    # plot reward aligned activity
    inset_axis_right.axvline(x=len(av_cluster_reward_aligned_rates) / 2, color="grey", linestyle="--", linewidth=1)
    inset_axis_right.plot(av_cluster_reward_aligned_rates, color="black")
    inset_axis_right.fill_between(
        range(len(av_cluster_reward_aligned_rates)),
        av_cluster_reward_aligned_rates - sem_cluster_reward_aligned_rates,
        av_cluster_reward_aligned_rates + sem_cluster_reward_aligned_rates,
        color=color,
        alpha=0.2,
    )
    inset_axis_right.set_xticks([len(av_cluster_reward_aligned_rates) / 2])
    inset_axis_right.set_xticklabels([])
    inset_axis_right.set_yticks([])
    inset_axis_right.set_yticklabels([])
    inset_axis_right.tick_params(axis="both", which="major", labelsize=7)
    inset_axis_right.set_xlabel("Reward", size=7, labelpad=-1)
    inset_axis_right.spines["left"].set_visible(False)
    return


# %% Plot trial aligned neural activity


def plot_session_trial_aligned_rates(session, colormap="gist_ncar"):
    """
    This function plots trial aligned neural activity for every good (KSLabel) cluster in a session.
    This function class the prepare_plot_trial_aligned_rates function to gather, cluster grouped data (for inset plot) and cluster & goal grouped data
    (for the main plot) and outputs the list of clusters to plot as well as a tuple of the plotting variables. A plotting axis, individual cluster IDs,
    plot colormap, and these plotting variables are then passed into the plot_cluster_trial_aligned_rates function to generate the plot for a single cluster.
    """
    cluster_unique_IDs, plotting_variables = prepare_plot_trial_aligned_rates(session)
    for cluster_unique_ID in cluster_unique_IDs:
        f, ax = plt.subplots(figsize=(10, 6), clear=True)
        plot_cluster_trial_aligned_rates(ax, cluster_unique_ID, colormap, *plotting_variables)  # plot specific info


def prepare_plot_trial_aligned_rates(session, smoothed=True, smooth_SD=15, cluster_type="good"):
    # check relevant data is available in session object
    data_needed = ["trial_aligned_rates_df"]
    if not gs.check_session_has_data(session, data_needed):
        pass
    simple_maze = session.simple_maze()
    trial_aligned_rates_df = session.trial_aligned_rates_df
    timepoints = trial_aligned_rates_df.firing_rate.columns.to_numpy()
    if cluster_type == "good":
        trial_aligned_rates_df = trial_aligned_rates_df[trial_aligned_rates_df.cluster_type == "good"]
    cluster_unique_IDs = trial_aligned_rates_df.cluster_unique_ID.unique()
    goals = np.sort(trial_aligned_rates_df.goal.unique())
    if smoothed:
        trial_aligned_rates = trial_aligned_rates_df.copy()
        firing_rate_data = trial_aligned_rates.loc[:, ("firing_rate", slice(None), slice(None))].to_numpy()
        trial_aligned_rates.loc[:, ("firing_rate", slice(None), slice(None))] = gaussian_filter1d(
            firing_rate_data, sigma=smooth_SD
        )
        trial_aligned_rates_df = trial_aligned_rates
    # data grouped by cluster
    cluster_grouped_trial_aligned_rates = trial_aligned_rates_df.set_index("cluster_unique_ID").firing_rate.groupby(
        ["cluster_unique_ID"]
    )
    av_clusters_trial_aligned_rates = cluster_grouped_trial_aligned_rates.mean()
    sem_clusters_trial_aligned_rates = cluster_grouped_trial_aligned_rates.sem()
    # data group by cluster and goal
    cluster_goal_grouped_trial_aligned_rates = trial_aligned_rates.set_index(
        ["cluster_unique_ID", "goal"]
    ).firing_rate.groupby(["cluster_unique_ID", "goal"])
    av_clusters_goals_trial_aligned_rates = cluster_goal_grouped_trial_aligned_rates.mean()
    sem_clusters_goals_trial_aligned_rates = cluster_goal_grouped_trial_aligned_rates.sem()
    plotting_variables = (
        simple_maze,
        goals,  # maze info
        timepoints,
        av_clusters_trial_aligned_rates,
        sem_clusters_trial_aligned_rates,  # cluster grouped data
        av_clusters_goals_trial_aligned_rates,
        sem_clusters_goals_trial_aligned_rates,
    )  # cluster & goal grouped data
    return cluster_unique_IDs, plotting_variables


def plot_cluster_trial_aligned_rates(
    ax,
    cluster_unique_ID,
    colormap,  # plot specific info
    simple_maze,
    goals,  # maze info
    timepoints,
    av_clusters_trial_aligned_rates,
    sem_clusters_trial_aligned_rates,  # cluster grouped data
    av_clusters_goals_trial_aligned_rates,
    sem_clusters_goals_trial_aligned_rates,
):  # cluster & goal grouped data
    plot_trial_aligned_activity_by_goal(
        ax,
        cluster_unique_ID,
        colormap,
        goals,
        timepoints,
        av_clusters_goals_trial_aligned_rates,
        sem_clusters_goals_trial_aligned_rates,
    )
    plot_inset_trial_aligned_activity(
        ax,
        cluster_unique_ID,
        timepoints,
        av_clusters_trial_aligned_rates,
        sem_clusters_trial_aligned_rates,
        color="black",
    )
    plot_inset_maze_legend(ax, colormap, simple_maze, goals)
    return


def plot_trial_aligned_activity_by_goal(
    ax,
    cluster_unique_ID,
    colormap,
    goals,
    timepoints,
    av_clusters_goals_trial_aligned_rates,
    sem_clusters_goals_trial_aligned_rates,
):
    goal2color = mp.get_goal2standard_color(colormap)
    av_cluster_goals_trial_aligned_rates = av_clusters_goals_trial_aligned_rates.loc[cluster_unique_ID]
    sem_cluster_goals_trial_aligned_rates = sem_clusters_goals_trial_aligned_rates.loc[cluster_unique_ID]
    f_max = np.nanmax(av_cluster_goals_trial_aligned_rates) + 1.5 * np.nanmax(sem_cluster_goals_trial_aligned_rates)
    for goal in goals:
        color = goal2color[goal]
        av_cluster_goal_trial_aligned_rates = av_cluster_goals_trial_aligned_rates.loc[goal].to_numpy()
        sem_cluster_goal_trial_aligned_rates = sem_cluster_goals_trial_aligned_rates.loc[goal].to_numpy()
        ax.plot(timepoints, av_cluster_goal_trial_aligned_rates, color=color, linewidth=0.85)
        ax.fill_between(
            timepoints.astype(float),
            av_cluster_goal_trial_aligned_rates - sem_cluster_goal_trial_aligned_rates,
            av_cluster_goal_trial_aligned_rates + sem_cluster_goal_trial_aligned_rates,
            color=color,
            alpha=0.1,
        )
        # plot formatting
    intra_trial_interval_times = EXP_INFO["intra_trial_interval_times"][:-1]
    for event in intra_trial_interval_times:
        ax.axvline(x=event, color="grey", linestyle="--", linewidth=1)
    ax.set_title(f"{cluster_unique_ID}", fontdict={"family": "Courier", "size": 10}, loc="left", pad=10, x=0)
    ax.set_xlim([-2, timepoints[-1]])
    ax.set_ylim([0, f_max])
    ax.set_xticks(intra_trial_interval_times)
    ax.set_xticklabels(
        [
            f"Cue:{intra_trial_interval_times[0]: .1f}",
            f"Reward:{intra_trial_interval_times[1]: .1f}",
            f"ITI start:{intra_trial_interval_times[2]: .1f}",
        ]
    )
    ax.set_xlabel("Trial-aligned Time (s)", size=12, labelpad=5)
    ax.set_ylabel("Firing Rate (Hz)", size=12)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return


def plot_inset_trial_aligned_activity(
    ax, cluster_unique_ID, timepoints, av_clusters_trial_aligned_rates, sem_clusters_trial_aligned_rates, color="black"
):
    inset_axis = inset_axes(
        ax, width="20%", height="20%", bbox_to_anchor=(0.12, -0.01, 1, 1), bbox_transform=ax.transAxes, loc="upper left"
    )
    av_cluster_trial_aligned_rates = av_clusters_trial_aligned_rates.loc[cluster_unique_ID].to_numpy()
    sem_cluster_trial_aligned_rates = sem_clusters_trial_aligned_rates.loc[cluster_unique_ID].to_numpy()
    f_max = np.nanmax(av_cluster_trial_aligned_rates) + 1.5 * np.nanmax(sem_cluster_trial_aligned_rates)
    inset_axis.plot(timepoints, av_cluster_trial_aligned_rates, color=color)
    inset_axis.fill_between(
        timepoints.astype(float),
        av_cluster_trial_aligned_rates - sem_cluster_trial_aligned_rates,
        av_cluster_trial_aligned_rates + sem_cluster_trial_aligned_rates,
        color=color,
        alpha=0.2,
    )
    inset_axis.set_xlim([-2, timepoints[-1]])
    inset_axis.set_ylim([0, f_max])
    intra_trial_interval_times = EXP_INFO["intra_trial_interval_times"][:-1]
    for event in intra_trial_interval_times:
        inset_axis.axvline(x=event, color="grey", linestyle="--", linewidth=1)
    inset_axis.set_xticks(intra_trial_interval_times)
    inset_axis.set_xticklabels(["C", "R", "ITI"])
    inset_axis.tick_params(axis="both", which="major", labelsize=7)
    return


# %% Common functions


def plot_inset_maze_legend(ax, colormap, simple_maze, goals, node_size=10, edge_size=2):
    """Adds right inset maze figure legend to a plot. Colors coloed by the specificed colormap."""
    inset_axis = inset_axes(
        ax,
        width="35%",
        height="35%",
        bbox_to_anchor=(0.065, 0.01, 1, 1),
        bbox_transform=ax.transAxes,
        loc="upper right",
    )
    mp.plot_simple_maze_for_figure_legend(
        simple_maze, inset_axis, goals, colormap, node_size=node_size, edge_size=edge_size
    )
