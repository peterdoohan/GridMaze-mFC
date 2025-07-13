"""
Is there a systematic shift in distance tuning curves across theta phases (peak vs trough)?
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.cluster_tuning import distance_to_goal as dtg

# %% Global Variables

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "distance_to_goal" / "theta_mod_tuning"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)


# %% Function


def plot_theta_x_shift_hist(summary_df, ax=None, print_stats=True):
    """ """
    # set up fig
    if ax is None:
        f, ax = plt.subplots(figsize=(3, 3))
    # ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("opt. x-shift (cm)")
    ax.set_ylabel("prop. distance tuned neurons")

    # process data
    shift_counts = summary_df.groupby(level=0).value_counts().unstack()
    norm_shift_counts = shift_counts.div(shift_counts.sum(axis=1), axis=0)
    shifts = norm_shift_counts.columns.astype(float).values
    shifts_cm = shifts * 100  # convert to cm
    mean = norm_shift_counts.mean(axis=0).values
    sem = norm_shift_counts.sem(axis=0).values
    # plot
    ax.bar(shifts_cm, mean, yerr=sem, color="grey", width=1.8, alpha=0.7)
    ax.set_ylim(0, 0.25)
    ax.set_xticks(shifts_cm)
    ax.set_xticklabels([f"{s:.0f}" for s in shifts_cm])
    # stats
    if print_stats:
        p_value = get_stats(summary_df, plot=False)
        print(f"p-value for x-shift < 0: {p_value:.3f}")


def get_stats(summary_df, n_resamples=10_000, plot=False):
    """
    Get random effects p-value to see if distribution of cluster
    x-shifts is significantly less than 0. As hypothesised from
    theta-mod decoding analyses
    """
    mean_shift = np.zeros(n_resamples)
    for i in range(n_resamples):
        # randomly sample subjects
        sampled_subjects = np.random.choice(SUBJECT_IDS, size=len(SUBJECT_IDS), replace=True)
        shifts = pd.concat([summary_df.loc[s] for s in sampled_subjects], axis=0, ignore_index=True)
        mean_shift[i] = shifts.mean()
    # calculate p-value
    p_value = (mean_shift > 0).sum() / n_resamples
    if plot:
        f, ax = plt.subplots(figsize=(3, 3))
        ax.hist(mean_shift, bins=50, color="grey", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--")
        ax.set_xlabel("mean x-shift (m)")
        ax.set_ylabel("count")
    return p_value


def get_theta_x_shift_summary(verbose=True, save=False):
    """ """
    save_path = RESULTS_DIR / "theta_x_shift_summary.csv"
    if not save and save_path.exists():
        if verbose:
            print(f"Loading existing results from {save_path}")
        return pd.read_csv(save_path, index_col=[0, 1])
    all_results = []
    for subject_ID in SUBJECT_IDS:
        if verbose:
            print(subject_ID)
            print("loading sessions ...")
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names="all",
            days_on_maze="all",
            with_data=[
                "navigation_df",
                "navigation_theta_spike_counts_df",
                "cluster_distance_tuning_metrics",
            ],
        )
        for session in sessions:
            if verbose:
                print(session.name)
            session_results = get_session_theta_x_shift(session, plot=False)  # default params
            if session_results is None:
                continue  # no valid clusters in this session
            session_results.index = pd.MultiIndex.from_tuples([(subject_ID, c) for c in session_results.index])
            all_results.append(session_results)
    x_shift_summary = pd.concat(all_results)
    if save:
        if verbose:
            print(f"Saving results to {save_path}")
        x_shift_summary.to_csv(save_path)
    return x_shift_summary


def get_session_theta_x_shift(
    session,
    min_split_half_corr=0.75,
    metrics=("distance_to_goal", "geodesic"),
    theta_peak_ind=[4, 5, 6, 7],
    theta_trough_ind=[0, 1, 10, 11],
    bin_spacing=0.02,
    max_steps_to_goal=30,
    moving_only=True,
    smooth_SD=4,
    n_shift=4,
    plot=True,
):
    """
    Calculates distance tuning curves for clusters in a session that have good distnace tuning (min_split_half_corr > thres)
    at the peak and trough of theta phase, separately, then calculates the optimal x shift needed to get best alignement between
    the tuning curves. If systematic shift differece in rep of goal at theta peak vs trough, the distribution of optimal
    x shifts should be shifted from 0.
    """
    # load data
    navigation_df = session.navigation_df.copy()
    distance_info = navigation_df[[("goal", ""), ("trial", ""), ("moving", ""), ("steps_to_goal", "future"), metrics]]
    theta_spike_counts = session.navigation_theta_spike_counts_df.reset_index(drop=True)
    distance_tuning_metrics = session.cluster_distance_tuning_metrics
    # filter for sufficiently distance tuned clusters
    valid_units = distance_tuning_metrics[
        distance_tuning_metrics.single_unit & (distance_tuning_metrics.split_half_corr.value > min_split_half_corr)
    ].cluster_unique_ID.values
    if len(valid_units) == 0:
        # no clusters with sufficient distance tuning
        return None

    # get theta phases
    phases = theta_spike_counts.columns.get_level_values(2).unique().astype(float)
    theta_peak_cols = phases[theta_peak_ind]
    theta_trough_cols = phases[theta_trough_ind]
    results = pd.Series(index=valid_units, name="theta_shift")
    for cluster in valid_units:
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
        results.loc[cluster] = get_theta_x_shift(
            mean_tuning,
            bin_spacing,
            smooth_SD=smooth_SD,
            n_shift=n_shift,
            plot=plot,
        )
    return results


def get_theta_x_shift(theta_tuning, bin_spacing, smooth_SD=2, n_shift=2, demean=False, plot=True):
    """ """
    peak = theta_tuning.theta.peak
    trough = theta_tuning.theta.trough
    if smooth_SD:
        peak, trough = [
            pd.Series(gaussian_filter1d(x.values, smooth_SD), index=x.index, name="peak") for x in (peak, trough)
        ]
    # only cal MSE over bins that are vald in all shifts
    valid_bins = trough.index[n_shift:-n_shift]
    _peak = peak.loc[valid_bins]
    if demean:
        _peak = _peak - _peak.mean()
    # calculate MSE for all shifts
    shifts = np.arange(-n_shift, n_shift + 1, 1)
    MSEs = np.zeros(len(shifts))
    for i, s in enumerate(shifts):
        shift_trough = trough.shift(s)
        _shift_trough = shift_trough.loc[valid_bins]
        if demean:
            _shift_trough = _shift_trough - _shift_trough.mean()
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
    return min_shift * bin_spacing  # shift in m
