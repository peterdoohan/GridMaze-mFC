"""Library for plotting Angle to Goal Tuning (& Head Direction Tuning)"""

# %% Imports
import numpy as np
import pandas as pd
from GridMaze.analysis.core import filter as filt
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from GridMaze.analysis.cluster_tuning import head_direction as hd

from GridMaze.maze import plotting as mp

# %% Global Variables


# %% Functions


def plot_session_angle_to_goal_tuning(session, metric="egocentric", goal_stratified=False):
    # load data
    navigation_rates_df = session.get_navigation_activity_df(type="rates", cluster_kwargs={"single_units": True})
    tuning_df = _get_angle_tuning_df(navigation_rates_df, metric)
    cluster_unique_IDs = navigation_rates_df.firing_rate.columns.to_numpy()
    for cluster_unique_ID in cluster_unique_IDs:
        cluster_tuning = tuning_df[tuning_df.cluster_unique_ID == cluster_unique_ID]
        plot_angle_tuning(cluster_tuning, metric, goal_stratified=goal_stratified)
    return


def plot_angle_tuning(cluster_tuning, metric_key, goal_stratified=False, smooth_SD=2, ax=None):
    metric_tuning = metric_key + "_tuning"
    if ax is None:
        f = plt.figure(figsize=(3, 3), clear=True)
        ax = f.add_subplot(111, projection="polar")
        ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
        ax.set_xticklabels([int(i) for i in np.linspace(0, 360, 8, endpoint=False)])
        ax.set_title(metric_key)
    bins = cluster_tuning[metric_tuning].columns.to_numpy().astype(float)
    bins = np.concatenate([bins, [bins[0]]])  # wrap
    bins_rad = np.radians(bins)
    if not goal_stratified:
        tuning_mean = cluster_tuning[metric_tuning].mean(axis=0).to_numpy()
        tuning_sem = cluster_tuning[metric_tuning].sem(axis=0).to_numpy()
        if smooth_SD:
            tuning_mean = hd.smooth_polar(tuning_mean, smooth_SD)
            tuning_sem = hd.smooth_polar(tuning_sem, smooth_SD)
        # wrap
        tuning_mean = np.concatenate([tuning_mean, [tuning_mean[0]]])
        tuning_sem = np.concatenate([tuning_sem, [tuning_sem[0]]])
        # plot
        hd._plot_angle_aligned_rates(bins_rad, tuning_mean, tuning_sem, ax, color="black", label=metric_key)
    else:
        goal2color = mp.get_goal2standard_color()
        for goal in cluster_tuning.goal.unique():
            tuning_mean = cluster_tuning[cluster_tuning.goal == goal][metric_tuning].mean(axis=0).to_numpy()
            tuning_sem = cluster_tuning[cluster_tuning.goal == goal][metric_tuning].sem(axis=0).to_numpy()
            if smooth_SD:
                tuning_mean = hd.smooth_polar(tuning_mean, smooth_SD)
                tuning_sem = hd.smooth_polar(tuning_sem, smooth_SD)
            # wrap
            tuning_mean = np.concatenate([tuning_mean, [tuning_mean[0]]])
            tuning_sem = np.concatenate([tuning_sem, [tuning_sem[0]]])
            # plot
            hd._plot_angle_aligned_rates(bins_rad, tuning_mean, tuning_sem, ax, color=goal2color[goal], label=goal)
    return


# %%
def _get_angle_tuning_df(navigation_rates_df, metric="egocentric", n_bins=120):
    """
    Generates the angle_to_goal or head_direction tuning on all trials and all clusters from a session
    Args:
        processed_data_path (Path): Path to processed data directory
        analysis_data_path (Path): Path to analysis data directory
        metric (tuple): Tuple of (metric, value) to calculate tuning for
            head_direction tuning, metric=("head_direction", "value")
            allocentric_angle_to_goal tuning, metric=("angle_to_goal", "allocentric")
            egocentric_angle_to_goal tuning, metric=("angle_to_goal", "egocentric")
        n_bins (int): Number of bins to split the metric into
    """
    # filter data
    subject_ID = navigation_rates_df.subject_ID.unique()[0]
    maze_name = navigation_rates_df.maze_name.unique()[0]
    day_on_maze = navigation_rates_df.day_on_maze.unique()[0]
    trial2goal = navigation_rates_df[["trial", "goal"]].dropna().drop_duplicates().set_index("trial").goal.to_dict()
    navigation_rates_df = filt.filter_navigation_rates_df(navigation_rates_df, moving_only=False)
    # get angle to goal tuning trial by trial
    angle_bins = ("angle_to_goal", metric + "_bined")
    bins = pd.IntervalIndex.from_breaks(np.linspace(0, 360, num=n_bins + 1, endpoint=True))
    navigation_rates_df[angle_bins] = pd.cut(navigation_rates_df.angle_to_goal[metric], bins=bins)
    cluster_unique_IDs = navigation_rates_df.firing_rate.columns.to_numpy()
    trials = navigation_rates_df.trial.unique()
    angle_tuning_dfs = []
    for t in trials:
        # get trial info
        trial_info = pd.DataFrame(
            {
                "subject_ID": subject_ID,
                "maze_name": maze_name,
                "day_on_maze": day_on_maze,
                "trial": t,
                "goal": trial2goal[t],
                "cluster_unique_ID": cluster_unique_IDs,
            }
        )
        trial_info.columns = pd.MultiIndex.from_product([trial_info.columns, [""]])
        trial_df = navigation_rates_df[navigation_rates_df.trial == t]
        angle_grouped_rates = trial_df.groupby([angle_bins], observed=True).firing_rate.mean().firing_rate.T
        missing_bins_df = _get_missing_bins_df(bins, angle_grouped_rates)  # df with missing bins filled with NaN
        angle_grouped_rates = pd.concat([angle_grouped_rates, missing_bins_df], axis=1)
        angle_grouped_rates = angle_grouped_rates.sort_index(axis=1)
        tuning_name = f"{metric}_tuning"
        angle_grouped_rates.columns = pd.MultiIndex.from_product(
            [[tuning_name], [b.mid for b in angle_grouped_rates.columns]]
        )
        angle_grouped_rates.reset_index(names="cluster_unique_ID", inplace=True)
        angle_tuning_dfs.append(pd.merge(trial_info, angle_grouped_rates, on=[("cluster_unique_ID", "")]))
    return pd.concat(angle_tuning_dfs, axis=0)


def _get_missing_bins_df(all_bins, hd_grouped_rates):
    current_bins = hd_grouped_rates.columns
    missing_bins = np.setdiff1d(all_bins, current_bins)
    return pd.DataFrame(data=np.nan, index=hd_grouped_rates.index, columns=missing_bins)


# %% Plot angle summary with ego, allo atg and head direction


def plot_session_angles_summary(session, n_bins=120, smooth_SD=2):
    """
    Plot unit egocentric angle to goal, allocentric angle to goal and head direction tuning
    on a single polar axis.
    """
    navigation_rates_df = session.get_navigation_activity_df(type="rates", cluster_kwargs={"single_units": True})
    cluster_unique_IDs = navigation_rates_df.firing_rate.columns.to_numpy()
    ego_tuning_df = _get_angle_tuning_df(navigation_rates_df, "egocentric", n_bins)
    allo_tuning_df = _get_angle_tuning_df(navigation_rates_df, "allocentric", n_bins)
    hd_tuning_mean, hd_tuning_sem = hd._process_head_direction_tuning(navigation_rates_df, n_bins)
    for cluster_unique_ID in cluster_unique_IDs:
        ego_tuning = ego_tuning_df[ego_tuning_df.cluster_unique_ID == cluster_unique_ID]
        ego_mean, ego_sem = ego_tuning.egocentric_tuning.mean(axis=0), ego_tuning.egocentric_tuning.sem(axis=0)
        allo_tuning = allo_tuning_df[allo_tuning_df.cluster_unique_ID == cluster_unique_ID]
        allo_mean, allo_sem = allo_tuning.allocentric_tuning.mean(axis=0), allo_tuning.allocentric_tuning.sem(axis=0)
        hd_mean, hd_sem = hd_tuning_mean[cluster_unique_ID], hd_tuning_sem[cluster_unique_ID]
        _plot_angles_summary(
            ego_tuning=(ego_mean, ego_sem),
            allo_tuning=(allo_mean, allo_sem),
            hd_tuning=(hd_mean, hd_sem),
            smooth_SD=smooth_SD,
        )
    return


def _plot_angles_summary(ego_tuning, allo_tuning, hd_tuning, smooth_SD=2, ax=None):
    # set up axis
    if ax is None:
        f = plt.figure(figsize=(3, 3), clear=True)
        ax = f.add_subplot(111, projection="polar")
    ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    ax.set_xticklabels([int(i) for i in np.linspace(0, 360, 4, endpoint=False)])
    ax.spines["polar"].set_visible(False)

    # unpack inputs
    ego_mean, ego_sem = ego_tuning
    allo_mean, allo_sem = allo_tuning
    hd_mean, hd_sem = hd_tuning
    # get bins
    bins = ego_mean.index.to_numpy().astype(float)
    bins = np.concatenate([bins, [bins[0]]])  # wrap
    bins_rad = np.radians(bins)
    # smooth
    if smooth_SD:
        ego_mean, ego_sem, allo_mean, allo_sem, hd_mean, hd_sem = [
            hd.smooth_polar(x, smooth_SD) for x in (ego_mean, ego_sem, allo_mean, allo_sem, hd_mean, hd_sem)
        ]
    # wrap for plotting
    wrap = lambda x: np.concatenate([x, [x[0]]])
    ego_mean, ego_sem, allo_mean, allo_sem, hd_mean, hd_sem = [
        wrap(x) for x in (ego_mean, ego_sem, allo_mean, allo_sem, hd_mean, hd_sem)
    ]
    for mean, sem, label, color in zip(
        [ego_mean, allo_mean, hd_mean],
        [ego_sem, allo_sem, hd_sem],
        ["Ego", "Allo", "HD"],
        ["darkred", "royalblue", "black"],
    ):
        hd._plot_angle_aligned_rates(bins_rad, mean, sem, ax, label=label, color=color)
    # adjust axis
    # rmax = ax.get_rmax()
    # ax.plot([0, 0], [0, rmax], color="black", lw=1)  # positive x–axis (0°)
    # ax.plot([np.pi, np.pi], [0, rmax], color="black", lw=1)  # negative x–axis (180°)
    # ax.plot([np.pi / 2, np.pi / 2], [0, rmax], color="black", lw=1)  # positive y–axis (90°)
    # ax.plot([3 * np.pi / 2, 3 * np.pi / 2], [0, rmax], color="black", lw=1)  # negative y–axis (270°)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.2, 1.2))
