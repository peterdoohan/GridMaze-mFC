"""
Another idea for measuring theta mod place-direction tuning
@peterdoohan
"""

# %% Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import filter as filt

from GridMaze.analysis.cluster_tuning import spatial
from scipy.ndimage import gaussian_filter

# %% Global variables

NSEW = ["N", "S", "E", "W"]

_DIR_SHIFT = {"N": (1, 0), "S": (-1, 0), "E": (0, 1), "W": (0, -1)}


# %% Functions


def test(
    session,
    phases_1=[0, 1],
    phases_2=[7, 8],
):
    nav_spikes_df = get_theta_stratified_nav_spikes_df(session)
    cluster_unique_IDs = nav_spikes_df.spike_count.columns.get_level_values(0).unique()
    phases = nav_spikes_df.spike_count.columns.get_level_values(1).unique()
    # separate data by movement direction (with appropraite filtering)
    dir2df = {dir: nav_spikes_df[nav_spikes_df.cardinal_movement_direction == dir] for dir in NSEW}
    for cluster in cluster_unique_IDs:
        f, axes = plt.subplots(2, 4, figsize=(10, 5))
        for i, phase in enumerate([phases_1, phases_2]):
            hms = {}
            for _dir in NSEW:
                dir_df = dir2df[_dir]
                spikes = dir_df.spike_count[cluster][phases[phase]].mean(axis=1).values
                pos = dir_df.centroid_position.values
                rate_map, binx, biny = spatial.get_2D_ratemap(
                    spikes,
                    pos,
                    x_size=0.01,
                    y_size=0.01,
                    smooth_SD=0.05,
                    x_range=(0, 1.4),
                    y_range=(0, 1.4),
                    nan_unvisited=True,
                    min_occupancy=None,
                )
                hms[_dir] = rate_map
            # plot
            _max = np.nanmax(list(hms.values()))
            for j, _dir in enumerate(NSEW):
                ax = axes[i, j]
                ax.imshow(hms[_dir], vmin=0, vmax=_max, origin="lower", extent=(0, 1.4, 0, 1.4))
                ax.set_title(f"Phase {phase}, dir {_dir}")


# %%


def get_shifted_heatmap_MSE(T, R, all_shifts_valid=True):
    """ """
    n_clusters, n_dirs, n_phases, n_shifts, flat_hm_size = T.shape
    # Flatten all clusters, dirs, and heatmap bins into a single vector
    # T: (n_clusters, n_dirs, n_phases, n_shifts, flat_hm_size)
    #  -> t: (n_phases, n_shifts, n_clusters * n_dirs * flat_hm_size)
    t = T.transpose(2, 3, 0, 1, 4).reshape(n_phases, n_shifts, n_clusters * n_dirs * flat_hm_size)
    # R: (n_clusters, n_dirs, flat_hm_size)
    #  -> r: (n_clusters * n_dirs * flat_hm_size,)
    r = R.reshape(n_clusters * n_dirs * flat_hm_size)
    # bins valid (non-NaN) across all phases and shifts (should be same across phases)
    valid_mask = ~np.any(np.isnan(t), axis=(0, 1))  # (n_clusters * n_dirs * flat_hm_size,)
    if all_shifts_valid:
        r = r[valid_mask]
    MSE = np.zeros((n_phases, n_shifts))
    for i in range(n_phases):
        for j in range(n_shifts):
            _t = t[i, j]
            if all_shifts_valid:
                _t = _t[valid_mask]
            mse = np.nanmean((_t - r) ** 2)
            MSE[i, j] = mse
    return MSE


def get_session_theta_shifted_heatmaps(
    session,
    bin_size=0.04,
    upscale_bin_size=0.01,
    smooth_SD=0.06,
    range=(0, 1.4),
    shift_range=0.08,
):
    """
    For each cluster x direction, compute theta-phase-stratified 2D ratemaps shifted
    across ±shift_range in the movement direction, plus a phase-averaged reference ratemap.
    Returns T (n_clusters, n_dirs, n_phases, n_shifts, flat_hm_size), R (n_clusters, n_dirs, flat_hm_size),
    cluster_unique_IDs, phases, and shifts_cm (shift offsets in metres).
    """
    nav_spikes_df = get_theta_stratified_nav_spikes_df(session)
    cluster_unique_IDs = nav_spikes_df.spike_count.columns.get_level_values(0).unique()
    phases = nav_spikes_df.spike_count.columns.get_level_values(1).unique()
    scale_factor = int(bin_size / upscale_bin_size)
    n_shifts = int(shift_range / upscale_bin_size)
    shifts = np.arange(-n_shifts, n_shifts + 1)
    shifts_cm = shifts * upscale_bin_size
    ratemap_kwargs = {
        "x_size": bin_size,
        "y_size": bin_size,
        "smooth_SD": 0,  # smooth after upscaling
        "x_range": range,
        "y_range": range,
        "nan_unvisited": True,
        "min_occupancy": None,
    }
    dir2df = {_dir: nav_spikes_df[nav_spikes_df.cardinal_movement_direction == _dir] for _dir in NSEW}
    flat_hm_size = int(((range[1] - range[0]) / upscale_bin_size) ** 2)
    # T: theta-phase-stratified shifted heatmaps (n_clusters, n_dirs, n_phases, n_shifts, flat_hm_size)
    T = np.zeros((len(cluster_unique_IDs), len(NSEW), len(phases), len(shifts), flat_hm_size))
    # R: reference heatmaps averaged across theta phases (n_clusters, n_dirs, flat_hm_size)
    R = np.zeros((len(cluster_unique_IDs), len(NSEW), flat_hm_size))
    for i, cluster in enumerate(cluster_unique_IDs):
        for j, _dir in enumerate(NSEW):
            dir_df = dir2df[_dir]
            pos = dir_df.centroid_position.values
            for k, phase in enumerate(phases):
                spikes = dir_df.spike_count[cluster][phase].values
                upscaled = _get_upscaled_ratemap(spikes, pos, ratemap_kwargs, scale_factor, smooth_SD, upscale_bin_size)
                for l, shift in enumerate(shifts):
                    T[i, j, k, l] = _shift_ratemap(upscaled, _dir, shift).flatten()
            avg_spikes = dir_df.spike_count[cluster].mean(axis=1).values
            R[i, j] = _get_upscaled_ratemap(
                avg_spikes, pos, ratemap_kwargs, scale_factor, smooth_SD, upscale_bin_size
            ).flatten()
    return T, R, cluster_unique_IDs, phases, shifts_cm


def _get_upscaled_ratemap(spikes, pos, ratemap_kwargs, scale_factor, smooth_SD, upscale_bin_size):
    """Compute ratemap at coarse resolution, upscale nearest-neighbour, then smooth."""
    rate_map, _, _ = spatial.get_2D_ratemap(spikes, pos, **ratemap_kwargs)
    upscaled = np.repeat(np.repeat(rate_map, scale_factor, axis=0), scale_factor, axis=1)
    if smooth_SD:
        upscaled = _smooth_upscaled_ratemap(upscaled, smooth_SD, upscale_bin_size)
    return upscaled


def _shift_ratemap(rate_map, direction, shift_bins):
    """Shift a 2D ratemap by shift_bins in the given cardinal direction, filling new space with NaN."""
    dy, dx = _DIR_SHIFT[direction]
    dy *= shift_bins
    dx *= shift_bins
    shifted = np.full_like(rate_map, np.nan)
    rows, cols = rate_map.shape
    src_r = slice(max(0, -dy), min(rows, rows - dy))
    dst_r = slice(max(0, dy), min(rows, rows + dy))
    src_c = slice(max(0, -dx), min(cols, cols - dx))
    dst_c = slice(max(0, dx), min(cols, cols + dx))
    shifted[dst_r, dst_c] = rate_map[src_r, src_c]
    return shifted


def _smooth_upscaled_ratemap(rate_map, smooth_SD, bin_size):
    """apply smoothing corrected for unvisited (NaN) bins, same as get_2D_ratemap"""
    nan_mask = np.isnan(rate_map)
    weights = np.where(~nan_mask, 1.0, 0.0)
    h = np.nan_to_num(rate_map, nan=0.0)
    sigma = smooth_SD / bin_size
    h = gaussian_filter(h, sigma=sigma)
    weights = gaussian_filter(weights, sigma=sigma)
    valid = weights > 0
    h[valid] = h[valid] / weights[valid]
    h[~valid] = np.nan
    h[nan_mask] = np.nan  # restore original unvisited NaNs
    return h


def get_theta_stratified_nav_spikes_df(
    session,
    split_half_corr_thres=0.6,
    moving_thres=0.05,
    verbose=True,
):
    # load data
    place_dir_metrics = session.cluster_place_direction_tuning_metrics.copy()
    navigation_df = session.navigation_df.copy()
    spikes_df = session.navigation_theta_spike_counts_df.reset_index(drop=True)
    # filter for clusters with strong pd tuning
    consider_clusters = place_dir_metrics[
        place_dir_metrics.split_half_corr.value.gt(split_half_corr_thres)
    ].index.values
    if len(consider_clusters) == 0:
        if verbose:
            print(f"No place-dir. tuned cluster for session: {session.name}")
        return None
    spikes_df = spikes_df[spikes_df.columns[spikes_df.columns.get_level_values(1).isin(consider_clusters)]]
    # combine spikes and nav data
    navigation_df.columns = pd.MultiIndex.from_tuples([(*col, "") for col in navigation_df.columns])
    nav_spikes_df = pd.concat([navigation_df, spikes_df], axis=1).copy()
    # filter data same as normal place-direction heatmaps
    nav_spikes_df = filt.filter_navigation_rates_df(
        nav_spikes_df, navigation_only=True, moving_only=False, exclude_time_at_goal=True, max_steps_to_goal=30
    )
    # apply custom movement threshold to keep as much data as possbible
    nav_spikes_df = nav_spikes_df[nav_spikes_df.speed.gt(moving_thres)]
    nav_spikes_df = nav_spikes_df.reset_index(drop=True).sort_index(axis=0)
    return nav_spikes_df
