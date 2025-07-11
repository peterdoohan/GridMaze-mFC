"""
Is there a systematic shift in distance tuning curves across theta phases (peak vs trough)?
@peterdoohan
"""

# %% Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy.ndimage import gaussian_filter1d

from GridMaze.analysis.core import convert

from GridMaze.analysis.cluster_tuning import distance_to_goal as dtg

# %% Global Variables


# %% Function


def test(
    session,
    min_split_half_corr=0.6,
    metrics=("distance_to_goal", "geodesic"),
    theta_peak_ind=[4, 5, 6, 7],
    theta_trough_ind=[0, 1, 10, 11],
    bin_spacing=0.02,
    max_steps_to_goal=30,
    moving_only=True,
    smooth_SD=4,
    n_shift=4,
):
    """ """
    # load data
    navigation_df = session.navigation_df.copy()
    distance_info = navigation_df[[("goal", ""), ("trial", ""), ("moving", ""), ("steps_to_goal", "future"), metrics]]
    theta_spike_counts = session.navigation_theta_spike_counts_df.reset_index(drop=True)
    distance_tuning_metrics = session.cluster_distance_tuning_metrics
    # filter for sufficiently distance tuned clusters
    valid_units = distance_tuning_metrics[
        distance_tuning_metrics.single_unit & (distance_tuning_metrics.split_half_corr.value > min_split_half_corr)
    ].cluster_unique_ID.values

    # get theta phases
    phases = theta_spike_counts.columns.get_level_values(2).unique().astype(float)
    theta_peak_cols = phases[theta_peak_ind]
    theta_trough_cols = phases[theta_trough_ind]

    for cluster in valid_units:
        print(cluster)
        theta_spikes = theta_spike_counts.spike_count[cluster]
        theta_spikes = theta_spike_counts.spike_count[cluster]
        theta_peak_spikes = theta_spikes[theta_peak_cols].sum(axis=1)
        theta_trough_spikes = theta_spikes[theta_trough_cols].sum(axis=1)
        distance_spikes_df = distance_info.copy()
        distance_spikes_df.loc[:, ("theta", "peak")] = theta_peak_spikes
        distance_spikes_df.loc[:, ("theta", "trough")] = theta_trough_spikes
        distance_theta_tuning_df = dtg.get_theta_distance_to_goal_tuning(
            distance_spikes_df,
            metrics=metrics,
            bin_spacing=bin_spacing,
            max_steps_to_goal=max_steps_to_goal,
            moving_only=moving_only,
        )
        mean_tuning = distance_theta_tuning_df.groupby("distance_bin").mean()
        mean_tuning.index = [c.mid for c in mean_tuning.index]
        get_x_shift_err(mean_tuning, smooth_SD=smooth_SD, n_shift=n_shift, plot=True)


def get_x_shift_err(theta_tuning, smooth_SD=2, n_shift=2, plot=True):
    """ """
    peak = theta_tuning.theta.peak
    trough = theta_tuning.theta.trough
    if smooth_SD:
        peak, trough = [
            pd.Series(gaussian_filter1d(x.values, smooth_SD), index=x.index, name="peak") for x in (peak, trough)
        ]
    # only cal MSE over bins that are vald in all shifts
    valid_bins = trough.index[n_shift:-n_shift]
    # calculate MSE for all shifts
    shifts = np.arange(-n_shift, n_shift + 1, 1)
    MSEs = np.zeros(len(shifts))
    for i, s in enumerate(shifts):
        shift_trough = trough.shift(s)
        MSEs[i] = ((peak.loc[valid_bins] - shift_trough.loc[valid_bins]) ** 2).mean()
    min_shift = shifts[np.argmin(MSEs)]
    if plot:
        f, axes = plt.subplots(1, 2, figsize=(6, 3))
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        # plot MSE landscape
        axes[0].plot(shifts, MSEs)
        axes[0].set_xlabel("Shift (bins)")
        axes[0].set_ylabel("MSE")
        axes[0].scatter(min_shift, MSEs.min(), color="red")

        # also plot the best shifted trough
        best_shifted_trough = trough.shift(min_shift)
        peak.plot(ax=axes[1], label="peak")
        trough.plot(ax=axes[1], label="trough")
        best_shifted_trough.plot(ax=axes[1], label=f"trough shifted {min_shift} bins")
        axes[1].legend()
    # return shift with lowest MSE
    return min_shift
