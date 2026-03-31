"""
Another idea for measuring theta mod place-direction tuning
@peterdoohan
"""

# %% Imports
import json
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import filter as filt

from GridMaze.analysis.cluster_tuning import spatial
from scipy.ndimage import gaussian_filter

# %% Global variables

from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

RESULTS_DIR = RESULTS_PATH / "theta_mod" / "place_direction_tuning"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)


NSEW = ["N", "S", "E", "W"]

_DIR_SHIFT = {"N": (1, 0), "S": (-1, 0), "E": (0, 1), "W": (0, -1)}


# %%


def get_SSE_df(
    split_half_corr_thres=0.5,
    bin_size=0.04,
    upscale_bin_size=0.001,
    smooth_SD=0.05,
    xy_range=(0, 1.4),
    shift_range=0.015,
    verbose=True,
    save=False,
    n_jobs=-1,
):
    """
    Compute per-cluster, per-phase, per-shift SSE across all late sessions.

    If save=False and the parquet already exists it is loaded directly.
    If save=True the result is (re-)computed and written to disk.

    Sessions are processed in parallel (n_jobs=-1 uses all available cores).
    """
    save_path = RESULTS_DIR / "SSE_df.parquet"
    if not save and save_path.exists():
        return pd.read_parquet(save_path)

    def _run(session):
        if verbose:
            print(session.name)
        try:
            return get_session_SSE_df(
                session,
                split_half_corr_thres=split_half_corr_thres,
                bin_size=bin_size,
                upscale_bin_size=upscale_bin_size,
                smooth_SD=smooth_SD,
                xy_range=xy_range,
                shift_range=shift_range,
                verbose=verbose,
            )
        except Exception as e:
            print(f"Error processing session {session.name}: {e}")
            return None

    dfs = []
    for subject in SUBJECT_IDS:
        if verbose:
            print(f"Processing subject {subject}...")
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject],
            maze_names="all",
            days_on_maze="late",
            with_data=[
                "cluster_place_direction_tuning_metrics",
                "navigation_df",
                "navigation_theta_spike_counts_df",
            ],
            must_have_data=True,
        )
        results = Parallel(n_jobs=n_jobs)(delayed(_run)(s) for s in sessions)
        dfs.extend(r for r in results if r is not None)
    if not dfs:
        raise RuntimeError("No sessions returned valid results.")
    SSE_df = pd.concat(dfs, ignore_index=True)
    if save:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        SSE_df.to_parquet(save_path)
    return SSE_df


# %%
def get_session_SSE_df(
    session,
    split_half_corr_thres=0.5,
    bin_size=0.04,
    upscale_bin_size=0.001,
    smooth_SD=0.05,
    xy_range=(0, 1.4),
    shift_range=0.015,
    verbose=True,
):
    """
    Compute per-cluster, per-phase, per-shift SSE relative to the mean-phase reference.

    Rather than storing full heatmap feature vectors, SSE and n_features are computed
    on the fly inside the cluster loop and accumulated across directions, keeping peak
    memory at O(n_valid_bins) per iteration instead of O(n_clusters*n_phases*n_shifts*n_bins).

    MSE for any subset of clusters can be recovered as sum(SSE) / sum(n_features),
    grouped by phase and shift_cm. n_features is constant across phase and shift for a
    given cluster (determined solely by the occupancy-based valid mask).

    Returns:
        df: DataFrame with columns [cluster_unique_ID, phase, shift_cm, SSE, n_features,
            split_half_corr, subject_ID, maze_name, day_on_maze, late_session]
    """
    metrics_df = session.cluster_place_direction_tuning_metrics
    nav_spikes_df = get_theta_stratified_nav_spikes_df(session, split_half_corr_thres, verbose=verbose)
    if nav_spikes_df is None:
        return None
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
        "x_range": xy_range,
        "y_range": xy_range,
        "nan_unvisited": True,
        "min_occupancy": None,
    }
    dir2df = {_dir: nav_spikes_df[nav_spikes_df.cardinal_movement_direction == _dir] for _dir in NSEW}
    n_coarse_bins = int((xy_range[1] - xy_range[0]) / bin_size)
    flat_hm_size = (n_coarse_bins * scale_factor) ** 2

    # Pre-compute per-direction valid masks from occupancy.
    dir_valid_masks = []
    n_valid = []
    for _dir in NSEW:
        mask = _get_direction_valid_mask(dir2df[_dir], _dir, ratemap_kwargs, scale_factor, shifts)
        dir_valid_masks.append(mask)
        n_valid.append(int(mask.sum()))

    if verbose:
        total_valid = sum(n_valid)
        pct_removed = 100 * (1 - total_valid / (len(NSEW) * flat_hm_size))
        print(f"Features removed by all-shifts validity mask: {pct_removed:.1f}%")

    records = []
    for cluster in cluster_unique_IDs:
        if verbose:
            print(cluster)
        # n_features for this cluster: total valid bins across all directions (constant across phase/shift)
        n_features_cluster = sum(n_valid)
        # Pre-compute ref_masked per direction — identical for all phases
        dir_ref_masked = {}
        for j, _dir in enumerate(NSEW):
            if n_valid[j] == 0:
                continue
            dir_df = dir2df[_dir]
            pos = dir_df.centroid_position.values
            ref_spikes = dir_df.spike_count[cluster].mean(axis=1).values
            dir_ref_masked[j] = _get_upscaled_ratemap(
                ref_spikes, pos, ratemap_kwargs, scale_factor, smooth_SD, upscale_bin_size
            ).flatten()[dir_valid_masks[j]]
        for k, phase in enumerate(phases):
            # Accumulate SSE across directions for each shift
            sse_per_shift = np.zeros(len(shifts))
            for j, _dir in enumerate(NSEW):
                if n_valid[j] == 0:
                    continue
                dir_df = dir2df[_dir]
                pos = dir_df.centroid_position.values
                shift_spikes = dir_df.spike_count[cluster][phase].values
                shift_up = _get_upscaled_ratemap(
                    shift_spikes, pos, ratemap_kwargs, scale_factor, smooth_SD, upscale_bin_size
                )
                for l, shift in enumerate(shifts):
                    shifted_masked = _shift_ratemap(shift_up, _dir, shift).flatten()[dir_valid_masks[j]]
                    sse_per_shift[l] += np.nansum((shifted_masked - dir_ref_masked[j]) ** 2)
            for l, shift_cm in enumerate(shifts_cm):
                records.append(
                    {
                        "cluster_unique_ID": cluster,
                        "phase": phase,
                        "shift_cm": shift_cm,
                        "SSE": sse_per_shift[l],
                        "n_features": n_features_cluster,
                        "split_half_corr": metrics_df.loc[cluster].split_half_corr.value,
                    }
                )

    df = pd.DataFrame(records)
    # add session info
    df["subject_ID"] = session.subject_ID
    df["maze_name"] = session.maze_name
    df["day_on_maze"] = session.day_on_maze
    df["late_session"] = session.late_session
    return df


def _get_direction_valid_mask(dir_df, direction, ratemap_kwargs, scale_factor, shifts):
    """
    Compute a boolean validity mask for a single movement direction.

    A bin is True if it is non-NaN in the upscaled occupancy map AND in every shifted
    version of that map. Uses pseudo-spikes (ones) so only visited/unvisited is determined,
    with no smoothing (smoothing is irrelevant for the binary validity decision).

    Args:
        dir_df: DataFrame rows for this direction's timepoints
        direction: one of NSEW
        ratemap_kwargs: dict passed to spatial.get_2D_ratemap (smooth_SD must be 0)
        scale_factor: int — bin_size / upscale_bin_size
        shifts: 1D int array of shift amounts, e.g. np.arange(-15, 16)

    Returns:
        bool array of shape (flat_hm_size,) — True where valid across all shifts
    """
    if len(dir_df) == 0:
        n_coarse = int((ratemap_kwargs["x_range"][1] - ratemap_kwargs["x_range"][0]) / ratemap_kwargs["x_size"])
        flat_hm_size = (n_coarse * scale_factor) ** 2
        return np.zeros(flat_hm_size, dtype=bool)
    pos = dir_df.centroid_position.values
    pseudo_spikes = np.ones(len(dir_df))
    occ_map, _, _ = spatial.get_2D_ratemap(pseudo_spikes, pos, **ratemap_kwargs)
    upscaled = np.repeat(np.repeat(occ_map, scale_factor, axis=0), scale_factor, axis=1)
    valid_2d = ~np.isnan(upscaled)
    for shift in shifts:
        valid_2d &= ~np.isnan(_shift_ratemap(upscaled, direction, shift))
    return valid_2d.flatten()


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
