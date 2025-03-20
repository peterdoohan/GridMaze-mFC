"""
Library for generating and plotting head direction tuning curves from analysis data
"""

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d

# %% Global Variables

# %% Functions


def plot_session_head_direction_tuning(session, smooth_SD=2, n_bins=180):
    # get data
    navigation_rates_df = session.get_navigation_activity_df(type="rates", cluster_kwargs={"single_units": True})
    clusters = navigation_rates_df.firing_rate.columns.values
    mean_hd_rates, sem_hd_rates = _process_head_direction_tuning(navigation_rates_df, n_bins)
    for cluster in clusters:
        mean_rates = mean_hd_rates[cluster]
        sem_rates = sem_hd_rates[cluster]
        plot_head_direction_tuning(mean_rates, sem_rates, smooth_SD)
    return


def _process_head_direction_tuning(navigation_rates_df, n_bins):
    navigation_rates_df = navigation_rates_df[
        navigation_rates_df.time.ge(0)
    ].copy()  # filter for times after pycontrol has started >0
    navigation_rates_df.loc[:, ("head_direction", "binned")] = pd.cut(
        navigation_rates_df.head_direction.value,
        bins=n_bins,
    )
    hd_grouped_rates = navigation_rates_df.groupby(("head_direction", "binned"), observed=True).firing_rate
    mean_hd_rates = hd_grouped_rates.mean().firing_rate
    sem_hd_rates = hd_grouped_rates.sem().firing_rate
    return mean_hd_rates, sem_hd_rates


def plot_head_direction_tuning(mean_rates, sem_rates, smooth_SD, ax=None):
    if ax is None:
        f = plt.figure(figsize=(3, 3), clear=True)
        ax = f.add_subplot(111, projection="polar")
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels([int(i) for i in np.linspace(0, 360, 8, endpoint=False)])
    ax.set_title("Head Direction Tuning")
    # plotting
    mid_bins = [b.mid for b in mean_rates.index]
    mid_bins = np.concatenate([mid_bins, [mid_bins[0]]])  # wrap
    bins_rad = np.radians(mid_bins)
    tuning_mean = mean_rates.values
    tuning_sem = sem_rates.values
    if smooth_SD:
        tuning_mean = smooth_polar(tuning_mean, smooth_SD)
        tuning_sem = smooth_polar(tuning_sem, smooth_SD)
    # wrap for plotting
    tuning_mean = np.concatenate([tuning_mean, [tuning_mean[0]]])
    tuning_sem = np.concatenate([tuning_sem, [tuning_sem[0]]])
    _plot_angle_aligned_rates(bins_rad, tuning_mean, tuning_sem, ax=ax, color="black")
    return


def smooth_polar(angles, smooth_SD, wrap_pad=10):
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


def _plot_angle_aligned_rates(bins_rad, tuning_mean, tuning_sem, ax, color="green", label=None):
    ax.plot(bins_rad, tuning_mean, color=color, label=label)
    ax.fill_between(bins_rad, tuning_mean - tuning_sem, tuning_mean + tuning_sem, color=color, alpha=0.1)
    return
