"""Library for plotting Angle to Goal Tuning (& Head Direction Tuning)"""

# %% Imports
import numpy as np
from ..core import get_clusters as gc
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

from ...maze import plotting as mp

# %% Global Variables


# %% Functions


def plot_session_angle_to_goal_tuning(session, metric="allocentric_angle_to_goal", goal_stratified=False):
    metric_key = metric + "_tuning"
    tuning_df = getattr(session, metric_key + "_df")
    keep_clusters = gc.filter_clusters(
        session.cluster_metrics,  # plot only single units
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
    )
    for cluster_unique_ID in keep_clusters[:1]:
        cluster_tuning = tuning_df[tuning_df.cluster_unique_ID == cluster_unique_ID]
        plot_angle_tuning(cluster_tuning, metric_key, goal_stratified=goal_stratified)
    return


def plot_angle_tuning(cluster_tuning, metric_key, goal_stratified=False, smooth_SD=2, ax=None):
    metric_tuning = metric_key + "_tuning"
    if ax is None:
        f = plt.figure(figsize=(3, 3), clear=True)
        ax = f.add_subplot(111, projection="polar")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
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
