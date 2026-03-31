"""
Another idea for measuring theta mod place-direction tuning
@peterdoohan
"""

# %% Imports
import json
import pickle
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


def load_phase_rel_heatmaps(subject_ID="m2", maze_name="all", plot=True):
    """
    Load and combine saved phase-relative heatmaps across sessions.

    Mirrors the session selection used in get_phase_rel_heatmaps so that only
    sessions for which results were computed are considered.

    Args:
        subject_ID: subject identifier, or "all" for all subjects
        maze_name:  maze name to filter sessions, or "all" for all mazes

    Returns:
        t:        (n_cluster_heatmaps, n_shifts) — shifted heatmaps concatenated across sessions
        v:        (n_cluster_heatmaps,)          — reference heatmaps concatenated across sessions
        shifts_cm:(n_shifts,)                    — shift values in metres (common across sessions)
    """
    save_dir = RESULTS_DIR / "shifted_heatmaps"
    t_list = []
    r_list = []
    shifts_cm = None
    # Accumulators for MSE: avoids allocating a full-size difference array at the end.
    # MSE = total_sse / n_valid is mathematically identical to nanmean over all features.
    sse = None  # (n_shifts,) running sum of squared errors
    n_valid = 0  # total number of valid (non-NaN) features across sessions

    for pkl_path in sorted(save_dir.glob("*.pkl")):
        # Filter by subject_ID from filename prefix before loading
        if subject_ID != "all" and pkl_path.name.split(".")[0] != subject_ID:
            continue
        with open(pkl_path, "rb") as f:
            res = pickle.load(f)
        # Filter by maze_name from loaded metadata
        if maze_name != "all" and res["maze_name"] != maze_name:
            continue
        t_s = res["t"]  # (n_shifts, n_features_session)
        r_s = res["r"]  # (n_features_session,)
        t_list.append(t_s)
        r_list.append(r_s)
        if shifts_cm is None:
            shifts_cm = res["shifts_cm"]
            sse = np.zeros(len(shifts_cm))
        # Accumulate squared errors session by session — O(n_shifts) overhead
        sse += np.nansum((t_s - r_s[np.newaxis, :]) ** 2, axis=-1)
        n_valid += int(np.sum(~np.isnan(r_s)))

    if not t_list:
        raise ValueError(f"No saved results found for subject_ID={subject_ID}, maze_name={maze_name}")

    # Concatenate across sessions along the feature dimension
    t = np.concatenate(t_list, axis=1)  # (n_shifts, n_cluster_heatmaps)
    r = np.concatenate(r_list, axis=0)  # (n_cluster_heatmaps,)

    if plot:
        MSE = sse / n_valid  # (n_shifts,) — no large temporary array needed
        best_shift_cm = shifts_cm[np.argmin(MSE)]

        f, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(shifts_cm * 100, MSE, marker="o", markersize=4)
        ax.axvline(best_shift_cm * 100, color="r", linestyle="--", label=f"best: {best_shift_cm*100:.1f} cm")
        ax.set_xlabel("Spatial shift (cm)")
        ax.set_ylabel("MSE")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

    return t, r, shifts_cm


def get_phase_rel_heatmaps(
    subject_ID="all",
    split_half_corr_thres=0.5,
    bin_size=0.04,
    upscale_bin_size=0.001,
    smooth_SD=0.05,
    xy_range=(0, 1.4),
    shift_range=0.015,
    verbose=True,
):
    """
    Compute phase-relative shifted heatmaps across all late sessions of a given subject.

    The reference heatmap R is the mean firing rate across all theta phases.
    T stores, for each phase and spatial shift, the shifted ratemap — allowing
    downstream MSE to find which shift best aligns each phase to the mean.
    Results are saved per session as .pkl files.
    """
    save_dir = RESULTS_DIR / "shifted_heatmaps"
    subject_IDs = [subject_ID] if subject_ID != "all" else SUBJECT_IDS
    for subject in subject_IDs:
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
        for session in sessions:
            if verbose:
                print(session.name)
            save_path = save_dir / f"{session.name}.pkl"
            if save_path.exists():
                if verbose:
                    print(f"Results already exist for {session.name}, skipping...")
                continue
            try:
                res = get_session_phase_rel_heatmaps(
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
                if verbose:
                    print(f"Error processing session {session.name}: {e}")
                continue
            if res is None:
                continue
            t, r, shifts_cm, cluster_unique_IDs = res
            results = {
                "t": t,
                "r": r,
                "shifts_cm": shifts_cm,
                "cluster_unique_IDs": cluster_unique_IDs,
                "subject_ID": subject_ID,
                "maze_name": session.maze_name,
                "day_on_maze": session.day_on_maze,
                "late_session": session.late_session,
            }
            if verbose:
                print(f"Saving results to {save_path}...")
            with open(save_path, "wb") as f:
                pickle.dump(results, f)
    # save params
    params = {
        "split_half_corr_thres": split_half_corr_thres,
        "bin_size": bin_size,
        "upscale_bin_size": upscale_bin_size,
        "smooth_SD": smooth_SD,
        "xy_range": xy_range,
        "shift_range": shift_range,
    }
    with open(save_dir / "params.json", "w") as f:
        json.dump(params, f)


def test_phase_rel_shift(session, **kwargs):
    """
    Quick test: for each theta phase compute the MSE between the spatially shifted
    phase ratemap and the mean-phase reference, then plot best shift per phase.

    Returns:
        MSE: (n_phases, n_shifts) — mean squared error at each phase × shift
        best_shifts_cm: (n_phases,) — shift in metres that minimises MSE per phase
        shifts_cm: (n_shifts,) — shift values in metres
        phases: phase labels
    """
    result = get_session_phase_rel_heatmaps(session, **kwargs)
    if result is None:
        print("No valid clusters for this session.")
        return None
    df, shifts_cm = result
    # MSE per (phase, shift) = sum(SSE) / sum(n_features)
    grouped = df.groupby(["phase", "shift_cm"]).agg(SSE=("SSE", "sum"), n_features=("n_features", "sum"))
    grouped["MSE"] = grouped["SSE"] / grouped["n_features"]
    MSE_df = grouped["MSE"].unstack("shift_cm")  # (n_phases, n_shifts)
    MSE = MSE_df.values
    phases = MSE_df.index.values
    best_shifts_cm = shifts_cm[np.argmin(MSE, axis=1)]  # (n_phases,)

    f, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(shifts_cm * 100, MSE.T)
    ax.set_xlabel("Spatial shift (cm)")
    ax.set_ylabel("MSE")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    f2, ax2 = plt.subplots(1, 1, figsize=(5, 3))
    ax2.plot(phases, best_shifts_cm * 100, marker="o")
    ax2.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Theta phase")
    ax2.set_ylabel("Best shift (cm)")
    ax2.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    print(f"Best shifts (cm): {best_shifts_cm * 100}")
    return MSE, best_shifts_cm, shifts_cm


def save_all_phase_rel_heatmaps(verbose=True):
    for subject_ID in SUBJECT_IDS:
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names="all",
            days_on_maze="late",
            with_data=[
                "cluster_place_direction_tuning_metrics",
                "navigation_df",
                "navigation_theta_spike_counts_df",
            ],
            must_have_data=True,
        )
        for session in sessions:
            if verbose:
                print(session.name)
            try:
                res = get_session_phase_rel_heatmaps(
                    session,
                    verbose=verbose,
                )
            except Exception as e:
                print(f"Error occurred while processing {session.name}: {e}")


def get_session_SEE_df(
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
        df: DataFrame with columns [cluster_unique_ID, phase, shift_cm, SSE, n_features]
        shifts_cm: (n_shifts,) array of shift values in metres
    """
    nav_spikes_df = get_theta_stratified_nav_spikes_df(session, split_half_corr_thres)
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
        for k, phase in enumerate(phases):
            # Accumulate SSE across directions for each shift
            sse_per_shift = np.zeros(len(shifts))
            for j, _dir in enumerate(NSEW):
                if n_valid[j] == 0:
                    continue
                mask = dir_valid_masks[j]
                dir_df = dir2df[_dir]
                pos = dir_df.centroid_position.values
                ref_spikes = dir_df.spike_count[cluster].mean(axis=1).values
                ref_masked = _get_upscaled_ratemap(
                    ref_spikes, pos, ratemap_kwargs, scale_factor, smooth_SD, upscale_bin_size
                ).flatten()[mask]
                shift_spikes = dir_df.spike_count[cluster][phase].values
                shift_up = _get_upscaled_ratemap(
                    shift_spikes, pos, ratemap_kwargs, scale_factor, smooth_SD, upscale_bin_size
                )
                for l, shift in enumerate(shifts):
                    shifted_masked = _shift_ratemap(shift_up, _dir, shift).flatten()[mask]
                    sse_per_shift[l] += np.nansum((shifted_masked - ref_masked) ** 2)
            for l, shift_cm in enumerate(shifts_cm):
                records.append(
                    {
                        "cluster_unique_ID": cluster,
                        "phase": phase,
                        "shift_cm": shift_cm,
                        "SSE": sse_per_shift[l],
                        "n_features": n_features_cluster,
                    }
                )

    df = pd.DataFrame(records)
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
