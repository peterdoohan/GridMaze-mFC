"""Library for plotting Angle to Goal Tuning (& Head Direction Tuning)"""

# %% Imports
import numpy as np
import pandas as pd
from GridMaze.analysis.core import filter as filt
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

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
            tuning_mean = smooth_angles(tuning_mean, smooth_SD)
            tuning_sem = smooth_angles(tuning_sem, smooth_SD)
        # wrap
        tuning_mean = np.concatenate([tuning_mean, [tuning_mean[0]]])
        tuning_sem = np.concatenate([tuning_sem, [tuning_sem[0]]])
        # plot
        _plot_angle_aligned_rates(bins_rad, tuning_mean, tuning_sem, ax, color="black", label=metric_key)
    else:
        goal2color = mp.get_goal2standard_color()
        for goal in cluster_tuning.goal.unique():
            tuning_mean = cluster_tuning[cluster_tuning.goal == goal][metric_tuning].mean(axis=0).to_numpy()
            tuning_sem = cluster_tuning[cluster_tuning.goal == goal][metric_tuning].sem(axis=0).to_numpy()
            if smooth_SD:
                tuning_mean = smooth_angles(tuning_mean, smooth_SD)
                tuning_sem = smooth_angles(tuning_sem, smooth_SD)
            # wrap
            tuning_mean = np.concatenate([tuning_mean, [tuning_mean[0]]])
            tuning_sem = np.concatenate([tuning_sem, [tuning_sem[0]]])
            # plot
            _plot_angle_aligned_rates(bins_rad, tuning_mean, tuning_sem, ax, color=goal2color[goal], label=goal)
    return


def _plot_angle_aligned_rates(bins_rad, tuning_mean, tuning_sem, ax, color="green", label=None):
    ax.plot(bins_rad, tuning_mean, color=color, label=label)
    ax.fill_between(bins_rad, tuning_mean - tuning_sem, tuning_mean + tuning_sem, color=color, alpha=0.3)
    return


def smooth_angles(angles, smooth_SD, wrap_pad=10):
    """
    Smooths bin averaged angles [n_bins, n_clusters] in polar coordinates before translating back to deg.
    Wraps the data to avoid bin edge discontinuities at 0/360deg.
    """
    angles_rad = np.deg2rad(angles)
    x = np.cos(angles_rad)
    y = np.sin(angles_rad)
    # Wrap the data
    x = np.concatenate((x[-wrap_pad:], x, x[:wrap_pad]), axis=0)
    y = np.concatenate((y[-wrap_pad:], y, y[:wrap_pad]), axis=0)
    x_smooth = gaussian_filter1d(x, sigma=smooth_SD, axis=0)
    y_smooth = gaussian_filter1d(y, sigma=smooth_SD, axis=0)
    # Unwrap the data
    x_smooth = x_smooth[wrap_pad:-wrap_pad]
    y_smooth = y_smooth[wrap_pad:-wrap_pad]
    angles_smooth = np.rad2deg(np.arctan2(y_smooth, x_smooth)) % 360
    return angles_smooth


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
    navigation_rates_df = filt.filter_navigation_rates_df(navigation_rates_df, moving_only=True)
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
