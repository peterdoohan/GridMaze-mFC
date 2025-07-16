"""
New set of analyses to compare distance and progress metrics for understanding neural tuning.
@peterdoohan
"""

# %% Imports
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.distance_to_goal import distributions as dd
from GridMaze.analysis.distance_to_goal import population_tuning as pt
from GridMaze.analysis.distance_to_goal import distance_metrics as dm
from GridMaze.analysis.distance_to_goal.theta_mod_tuning import _downsample_neurons

# %% Global Variables

MEDIAN_STEPS_TO_GOAL = int(dd.get_distance_percentile(distance_metric=("steps_to_goal", "future"), percentile=0.5))

# %% Functions


def plot_heatmap_slices(
    short_long_distance_tuning,
    metric="distance_to_goal",
    sign="pos",
    fit="gamma_4p",
    smooth_SD=2,
    normalisation_method="max",
    n_groups=8,
    ax=None,
):
    """ """
    # set up fig
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.spines[["top", "right"]].set_visible(False)
    # process data
    heatmap_df = _get_short_long_distance_heatmap(
        short_long_distance_tuning, metric, sign, fit, smooth_SD, normalisation_method
    )
    neuron_groups_df = _downsample_neurons(heatmap_df, n_bins=n_groups)
    # plot
    colors = sns.color_palette("tab10", n_colors=n_groups)
    for i in range(n_groups):
        g = neuron_groups_df.iloc[i]
        for label, ls in zip(["short", "long"], ["-", "--"]):
            g_tc = g.loc[f"{metric}_{label}"]
            x = g_tc.index.astype(float).values
            y = g_tc.values
            ax.plot(x, y, label=f"{i + 1}:{label}", color=colors[i], linestyle=ls)
    ax.set_xlim(0, 1.0)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1), fontsize=8)
    ax.set_xlabel("Distance to goal (m)")
    ax.set_ylabel("norm. firing rate")


def plot_short_long_trial_split_distance_tuning_heatmaps(
    short_long_distance_tuning,
    metric="distance_to_goal",
    sign="pos",
    fit="gamma_4p",
    smooth_SD=2,
    normalisation_method="zscore",
    cmap="coolwarm",
    v_range=(-1, 2),
    axes=None,
):
    """ """
    # set up fig
    if axes is None:
        f, axes = plt.subplots(1, 2, figsize=(6, 6), sharey=True)
    # process data
    heatmap_df = _get_short_long_distance_heatmap(
        short_long_distance_tuning, metric, sign, fit, smooth_SD, normalisation_method
    )

    for ax, label in zip(axes, ["short", "long"]):
        D = heatmap_df[f"{metric}_{label}"].values
        sns.heatmap(
            D,
            ax=ax,
            cmap=cmap,
            vmax=v_range[1],
            vmin=v_range[0],
            cbar_kws={"label": "Firing rate (Hz)"},
        )


def _get_short_long_distance_heatmap(
    short_long_distance_tuning,
    metric="distance_to_goal",
    sign="pos",
    fit="gamma_4p",
    smooth_SD=2,
    normalisation_method="zscore",
):
    """
    order, smooth and normalise distance tuning data for heatmap plotting
    """
    df = short_long_distance_tuning.copy()
    if sign == "pos":
        sign_mask = df[fit]["size"].gt(0)
    elif sign == "neg":
        sign_mask = df[fit]["size"].lt(0)
    else:
        raise ValueError(f"Unknown sign: {sign}")
    df = df[sign_mask].copy()
    x = df[f"{metric}_long"].columns.astype(float).values  # distance bins
    if sign == "pos":
        df[("idx_max", "")] = df.apply(lambda row: pt.get_idx_order(row, x, fit=fit, op="max"), axis=1)
        df = df.sort_values(by=[("idx_max", "")], ascending=True)
    elif sign == "neg":
        df[("idx_min", "")] = df.apply(lambda row: pt.get_idx_order(row, x, fit=fit, op="min"), axis=1)
        df = df.sort_values(by=[("idx_min", "")], ascending=True)
    # smooth
    if smooth_SD:
        for label in ["short", "long"]:
            df.loc[:, f"{metric}_{label}"] = nan_gaussian_filter1d(
                df.loc[:, f"{metric}_{label}"].values, smooth_SD, axis=1
            )
    if normalisation_method == "zscore":
        for label in ["short", "long"]:
            df.loc[:, f"{metric}_{label}"] = zscore(df.loc[:, f"{metric}_{label}"].values, axis=1, nan_policy="omit")
    elif normalisation_method == "max":
        short, long = df[f"{metric}_short"].values, df[f"{metric}_long"].values
        tmax = np.nanmax(np.hstack([short, long]), axis=1)[:, None]
        short, long = short / tmax, long / tmax
        for label, arr in zip(["short", "long"], [short, long]):
            df.loc[:, f"{metric}_{label}"] = arr
    return df


def nan_gaussian_filter1d(D, sigma, axis=1):
    """
    Apply a 1D Gaussian filter along `axis`, ignoring NaNs in the input array.
    """
    D = np.array(D, copy=False)  # ensure an ndarray
    nan_mask = np.isnan(D)
    if not np.any(nan_mask):
        return gaussian_filter1d(D, sigma, axis=axis)
    D_filled = np.where(nan_mask, 0, D)
    filtered_data = gaussian_filter1d(D_filled, sigma, axis=axis)
    filtered_mask = gaussian_filter1d((~nan_mask).astype(float), sigma, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = filtered_data / filtered_mask
    result[nan_mask] = np.nan
    return result


def get_population_short_long_distance_tuning(
    sessions,
    metrics=("distance_to_goal", "geodesic"),
    min_split_half_corr=0.5,
    mon_dec_trials=False,
    median_steps_to_goal=MEDIAN_STEPS_TO_GOAL,
    verbose=False,
):
    dfs = []
    for session in sessions:
        if verbose:
            print(session.name)
        try:
            short_tuning, long_tuning = get_session_short_long_trial_split_distance_tuning(
                session,
                metrics=metrics,
                mon_dec_trials=mon_dec_trials,
                median_steps_to_goal=median_steps_to_goal,
            )
        except ValueError as e:
            if verbose:
                print(f"Skipping session {session.name}: {e}")
            continue
        distance_metrics = session.cluster_distance_tuning_metrics
        distance_metrics = distance_metrics[distance_metrics.split_half_corr.value.gt(min_split_half_corr)]
        keep_clusters = distance_metrics.cluster_unique_ID.values
        short_tuning, long_tuning = short_tuning.loc[keep_clusters], long_tuning.loc[keep_clusters]
        # combine both tuning curves with distance fit info
        df = pd.concat([short_tuning, long_tuning, distance_metrics.set_index("cluster_unique_ID")], axis=1)
        dfs.append(df)
    short_long_distance_tuning = pd.concat(dfs, axis=0)
    return short_long_distance_tuning.sort_index(axis=1)


def get_session_short_long_trial_split_distance_tuning(
    session,
    metrics=("distance_to_goal", "geodesic"),
    bin_spacing=0.05,
    mon_dec_trials=False,
    mon_dec_tol=0.12,
    max_steps_to_goal=30,
    progress_bins=30,  # if metrics[0] == "progress_to_goal"
    median_steps_to_goal=MEDIAN_STEPS_TO_GOAL,
    moving_only=False,
):
    """ """
    short_trials, long_trials = get_short_long_trials(session, mon_dec_trials, mon_dec_tol, median_steps_to_goal)
    if len(short_trials) == 0 or len(long_trials) == 0:
        raise ValueError("No short or long trials found")
    trial_av_rates = pt._get_session_distance_tuning(
        session,
        metrics,
        bin_spacing,
        max_steps_to_goal,
        progress_bins,
        moving_only,
        return_as="trial_av_rates",
    )
    tuning_curve_dfs = []
    for label, trials in zip(["short", "long"], [short_trials, long_trials]):
        _trial_av_rates = trial_av_rates.loc[trials]
        distance_tuning_df = (
            _trial_av_rates.groupby(["distance_bin"]).mean().firing_rate.T
        )  # cluster x distance_bins (average over trials)
        distance_tuning_df.columns = pd.MultiIndex.from_product(
            [[f"{metrics[0]}_{label}"], [b.mid for b in distance_tuning_df.columns]]
        )
        tuning_curve_dfs.append(distance_tuning_df)
    short_tuning_df, long_tuning_df = tuning_curve_dfs
    return short_tuning_df, long_tuning_df


def get_short_long_trials(session, mon_dec_trials=False, mon_dec_tol=0.12, median_steps_to_goal=MEDIAN_STEPS_TO_GOAL):
    """ """
    navigation_df = session.navigation_df
    navigation_df = navigation_df[navigation_df.trial_phase == "navigation"]
    if mon_dec_trials:
        valid_trials = dm.get_monotonic_decreasing_trials(session, tol=mon_dec_tol)
        navigation_df = navigation_df[navigation_df.trial.isin(valid_trials)]
    trial_start_frames = navigation_df[navigation_df.trial != navigation_df.trial.shift(1)]
    trial_total_steps = trial_start_frames[[("trial", ""), ("steps_to_goal", "future")]].droplevel(1, axis=1)
    short_trials = trial_total_steps[trial_total_steps.steps_to_goal <= median_steps_to_goal].trial.values
    long_trials = trial_total_steps[trial_total_steps.steps_to_goal > median_steps_to_goal].trial.values
    return short_trials, long_trials


# %%


def get_analysis_sessions():
    """
    same as cpd danalysis
    """
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names=["maze_1", "maze_2"],
        days_on_maze="all",
        with_data=["navigation_df", "navigation_spike_rates_df", "cluster_metrics", "cluster_distance_tuning_metrics"],
    )
    return sessions


def get_mon_dec_trial_path_length_dist(tol=0.12, verbose=True):
    """ """
    if verbose:
        print("Loading sessions...")
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names="all",
        days_on_maze="late",
        with_data=["navigation_df"],
    )
    path_lengths = []
    for session in sessions:
        if verbose:
            print(session.name)
        valid_trials = dm.get_monotonic_decreasing_trials(session, tol=tol)
        navigation_df = session.navigation_df
        navigation_df = navigation_df[navigation_df.trial_phase == "navigation"]
        navigation_df = navigation_df[navigation_df.trial.isin(valid_trials)]
        trial_start_frames = navigation_df[navigation_df.trial != navigation_df.trial.shift(1)]
        path_lengths.append(trial_start_frames.steps_to_goal.future.values)
    return np.hstack(path_lengths)
