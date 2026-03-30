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

RESULTS_DIR = RESULTS_PATH / "place_direction_tuning"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)


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


def get_shifted_heatmap_MSE(t, r, phases, shifts_cm, plot=True):
    """
    Compute MSE between theta-phase-stratified shifted heatmaps and the reference.

    Args:
        t: (n_phases, n_shifts, n_features) — output of get_theta_shifted_heatmaps
        r: (n_features,) — reference feature vector

    Returns:
        MSE: (n_phases, n_shifts)
        best_shifts: (n_phases,) shift in metres that minimises MSE at each phase
    """
    n_phases, n_shifts, _ = t.shape
    MSE = np.nanmean((t - r[np.newaxis, np.newaxis, :]) ** 2, axis=-1)
    best_shifts = shifts_cm[np.argmin(MSE, axis=1)]  # (n_phases,)
    if plot:
        plt.plot(MSE.T)
        plt.show()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(phases, best_shifts, marker="o")
        ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Theta phase")
        ax.set_ylabel("Best shift (cm)")
        ax.set_title("MSE-minimising shift per theta phase")
        plt.tight_layout()
    return MSE, best_shifts


def get_theta_shifted_heatmaps(
    subject_ID="all",
    sessions=None,
    split_half_corr_thres=0.5,
    bin_size=0.015,
    upscale_bin_size=0.003,
    smooth_SD=0.03,
    range=(0, 1.4),
    shift_range=0.015,
    n_jobs=False,
    verbose=True,
):
    subject_ID = [subject_ID] if subject_ID != "all" else subject_ID
    if sessions is None:
        if verbose:
            print("Loading sessions...")
        sessions = gs.get_maze_sessions(
            subject_IDs=subject_ID,
            maze_names="all",
            days_on_maze="late",
            with_data=[
                "cluster_place_direction_tuning_metrics",
                "navigation_df",
                "navigation_theta_spike_counts_df",
            ],
            must_have_data=True,
        )

    def _process_session(session):
        if verbose:
            print(session.name)
        return get_session_theta_shifted_heatmaps(
            session,
            split_half_corr_thres,
            bin_size,
            upscale_bin_size,
            smooth_SD,
            range,
            shift_range,
            return_as="session_features",
            all_shifts_valid=True,
            verbose=verbose,
        )

    if n_jobs is False:
        results = [_process_session(s) for s in sessions]
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(_process_session)(s) for s in sessions)

    ts, rs = [], []
    phases, shifts_cm = None, None
    for session_output in results:
        if session_output is None:
            continue
        t, r, phases, shifts_cm = session_output
        ts.append(t)
        rs.append(r)
    # concatenate feature vectors across sessions along the feature axis
    # t: (n_phases, n_shifts, big_feature_vec), r: (big_feature_vec,)
    t = np.concatenate(ts, axis=-1)
    r = np.concatenate(rs, axis=0)

    return t, r, phases, shifts_cm


def get_all_phase_rel_shifted_heatmaps():
    """ """
    save_dir = RESULTS_DIR / "shifted_heatmaps"
    results = {}
    for subject_ID in SUBJECT_IDS:
        print(f"Processing subject {subject_ID}...")
        t, r, shifts_cm = get_phase_rel_heatmaps(subject_ID=subject_ID)
        res = {"t": t, "r": r, "shifts_cm": shifts_cm}
        # save to pickle
        save_path = save_dir / f"{subject_ID}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(res, f)
        results[subject_ID] = {"t": t, "r": r, "shifts_cm": shifts_cm}
    return results


def get_phase_rel_heatmaps(
    subject_ID="all",
    ref_phases=[0, 1],
    shift_phases=[7, 8],
    sessions=None,
    split_half_corr_thres=0.5,
    bin_size=0.04,
    upscale_bin_size=0.001,
    smooth_SD=0.05,
    range=(0, 1.4),
    shift_range=0.015,
    n_jobs=False,
    verbose=True,
):
    """
    Compute phase-relative shifted heatmaps across all late sessions of a given subject,
    concatenating feature vectors across sessions so the combined t & r can be used to
    find the optimal spatial shift aligning shift_phases to ref_phases.

    Returns:
        t: (n_shifts, total_features) — shifted heatmaps concatenated across sessions
        r: (total_features,) — reference heatmaps concatenated across sessions
        shifts_cm: (n_shifts,) — shift values in metres
    """
    subject_ID = [subject_ID] if subject_ID != "all" else subject_ID
    if sessions is None:
        if verbose:
            print("Loading sessions...")
        sessions = gs.get_maze_sessions(
            subject_IDs=subject_ID,
            maze_names="all",
            days_on_maze="late",
            with_data=[
                "cluster_place_direction_tuning_metrics",
                "navigation_df",
                "navigation_theta_spike_counts_df",
            ],
            must_have_data=True,
        )

    def _process_session(session):
        if verbose:
            print(session.name)
        return get_session_phase_rel_heatmaps(
            session,
            ref_phases=ref_phases,
            shift_phases=shift_phases,
            split_half_corr_thres=split_half_corr_thres,
            bin_size=bin_size,
            upscale_bin_size=upscale_bin_size,
            smooth_SD=smooth_SD,
            range=range,
            shift_range=shift_range,
            return_as="session_features",
            all_shifts_valid=True,
            verbose=verbose,
        )

    if n_jobs is False:
        results = [_process_session(s) for s in sessions]
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(_process_session)(s) for s in sessions)

    ts, rs = [], []
    shifts_cm = None
    for session_output in results:
        if session_output is None:
            continue
        t, r, shifts_cm = session_output
        ts.append(t)
        rs.append(r)
    # concatenate feature vectors across sessions along the feature axis
    # t: (n_shifts, total_features), r: (total_features,)
    t = np.concatenate(ts, axis=-1)
    r = np.concatenate(rs, axis=0)
    return t, r, shifts_cm


def test_phase_rel_shift(
    session,
    ref_phases=[0, 1],
    shift_phases=[7, 8],
    **kwargs,
):
    """
    Quick test: compute MSE between shifted heatmaps (shift_phases) and reference (ref_phases)
    across all spatial shifts, and plot to find the best-aligning shift.

    Returns:
        MSE: (n_shifts,) — mean squared error at each shift
        best_shift_cm: scalar — shift in metres that minimises MSE
        shifts_cm: (n_shifts,) — shift values in metres
    """
    result = get_session_phase_rel_heatmaps(session, ref_phases=ref_phases, shift_phases=shift_phases, **kwargs)
    if result is None:
        print("No valid clusters for this session.")
        return None
    t, r, shifts_cm = result
    # t: (n_shifts, n_features), r: (n_features,)
    MSE = np.nanmean((t - r[np.newaxis, :]) ** 2, axis=-1)  # (n_shifts,)
    best_shift_cm = shifts_cm[np.argmin(MSE)]

    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(shifts_cm * 100, MSE, marker="o", markersize=4)
    ax.axvline(best_shift_cm * 100, color="r", linestyle="--", label=f"best: {best_shift_cm*100:.1f} cm")
    ax.set_xlabel("Spatial shift (cm)")
    ax.set_ylabel("MSE")
    ax.set_title(f"ref phases {ref_phases} vs shift phases {shift_phases}")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    print(f"Best shift: {best_shift_cm*100:.2f} cm (index {np.argmin(MSE)})")
    return MSE, best_shift_cm, shifts_cm


def get_session_phase_rel_heatmaps(
    session,
    ref_phases=[0, 1],
    shift_phases=[7, 8],
    split_half_corr_thres=0.5,
    bin_size=0.04,
    upscale_bin_size=0.001,
    smooth_SD=0.05,
    range=(0, 1.4),
    shift_range=0.015,
    return_as="session_features",  # session_features or session_tensors
    all_shifts_valid=True,
    verbose=True,
):
    """
    Simplified version of get_session_theta_shifted_heatmaps.

    Instead of using all phases for the reference and each individual phase for shifted
    heatmaps, this function uses a specified subset of phases for each role:
        - ref_phases: phase indices used to build the reference heatmap R
        - shift_phases: phase indices whose summed activity is spatially shifted to build T

    This lets you later test which spatial shift best aligns T to R.

    Args:
        ref_phases: list of integer indices into the phase axis used for the reference heatmap.
        shift_phases: list of integer indices into the phase axis used for the shifted heatmaps.
        return_as: "session_features" returns (t, r, shifts_cm) where
            t: (n_shifts, n_valid_features), r: (n_valid_features,).
            "session_tensors" returns (T, R, shifts_cm) with shapes
            (n_clusters, n_dirs, n_shifts, flat_hm_size) and (n_clusters, n_dirs, flat_hm_size).
        all_shifts_valid: if True, restricts features to bins non-NaN across every shift.
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
        "x_range": range,
        "y_range": range,
        "nan_unvisited": True,
        "min_occupancy": None,
    }
    dir2df = {_dir: nav_spikes_df[nav_spikes_df.cardinal_movement_direction == _dir] for _dir in NSEW}
    n_coarse_bins = int((range[1] - range[0]) / bin_size)
    flat_hm_size = (n_coarse_bins * scale_factor) ** 2
    # T: shift_phases-based shifted heatmaps (n_clusters, n_dirs, n_shifts, flat_hm_size)
    T = np.zeros((len(cluster_unique_IDs), len(NSEW), len(shifts), flat_hm_size))
    # R: reference heatmaps from ref_phases (n_clusters, n_dirs, flat_hm_size)
    R = np.zeros((len(cluster_unique_IDs), len(NSEW), flat_hm_size))
    for i, cluster in enumerate(cluster_unique_IDs):
        if verbose:
            print(cluster)
        for j, _dir in enumerate(NSEW):
            dir_df = dir2df[_dir]
            pos = dir_df.centroid_position.values
            # reference: mean spikes over ref_phases
            ref_spikes = dir_df.spike_count[cluster][phases[ref_phases]].sum(axis=1).values
            R[i, j] = _get_upscaled_ratemap(
                ref_spikes, pos, ratemap_kwargs, scale_factor, smooth_SD, upscale_bin_size
            ).flatten()
            # shifted heatmaps: mean spikes over shift_phases, then apply spatial shifts
            shift_spikes = dir_df.spike_count[cluster][phases[shift_phases]].sum(axis=1).values
            upscaled = _get_upscaled_ratemap(
                shift_spikes, pos, ratemap_kwargs, scale_factor, smooth_SD, upscale_bin_size
            )
            for l, shift in enumerate(shifts):
                T[i, j, l] = _shift_ratemap(upscaled, _dir, shift).flatten()
    if return_as == "session_tensors":
        return T, R, shifts_cm
    elif return_as == "session_features":
        n_clusters, n_dirs, n_shifts, flat_hm_size = T.shape
        # Flatten all clusters, dirs, and heatmap bins into a single session feature vector
        #  -> t: (n_shifts, n_clusters * n_dirs * flat_hm_size)
        t = T.transpose(2, 0, 1, 3).reshape(n_shifts, n_clusters * n_dirs * flat_hm_size)
        #  -> r: (n_clusters * n_dirs * flat_hm_size,)
        r = R.reshape(n_clusters * n_dirs * flat_hm_size)
        if all_shifts_valid:
            n_feature = (~np.isnan(r)).sum() // n_clusters
            valid_mask = ~np.any(np.isnan(t), axis=0)  # (n_clusters * n_dirs * flat_hm_size,)
            n_valid_feature = valid_mask.sum() // n_clusters
            pct_removed = 100 * (1 - n_valid_feature / n_feature)
            if verbose:
                print(f"Features removed by all-shifts validity mask: {pct_removed:.1f}% ")
            r = r[valid_mask]
            t = t[:, valid_mask]
        return t, r, shifts_cm
    else:
        raise ValueError(f"Invalid return_as value: {return_as}")


def get_session_theta_shifted_heatmaps(
    session,
    split_half_corr_thres=0.7,
    bin_size=0.04,
    upscale_bin_size=0.004,
    smooth_SD=0.05,
    range=(0, 1.4),
    shift_range=0.04,
    return_as="session_features",  # session_features or session_tensors
    all_shifts_valid=True,
    verbose=True,
):
    """
    Build theta-phase-stratified, direction-stratified shifted ratemaps for a single session.

    Args:
        return_as: "session_features" (default) returns (t, r) as flat composite vectors
            ready for MSE computation — t: (n_phases, n_shifts, n_valid_features),
            r: (n_valid_features,), where features concatenate all clusters × dirs × bins.
            "session_tensors" returns raw (T, R) with shapes
            (n_clusters, n_dirs, n_phases, n_shifts, flat_hm_size) and
            (n_clusters, n_dirs, flat_hm_size).
        all_shifts_valid: if True (default), restricts features to bins that are non-NaN
            across every shift and phase. Substantially reduces memory when aggregating
            across sessions. Only applies in "session_features" mode.
        split_half_corr_thres: minimum split-half spatial correlation for a cluster to
            be included; sessions where no clusters pass are skipped (returns None).
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
        "x_range": range,
        "y_range": range,
        "nan_unvisited": True,
        "min_occupancy": None,
    }
    dir2df = {_dir: nav_spikes_df[nav_spikes_df.cardinal_movement_direction == _dir] for _dir in NSEW}
    n_coarse_bins = int((range[1] - range[0]) / bin_size)
    flat_hm_size = (n_coarse_bins * scale_factor) ** 2
    # T: theta-phase-stratified shifted heatmaps (n_clusters, n_dirs, n_phases, n_shifts, flat_hm_size)
    T = np.zeros((len(cluster_unique_IDs), len(NSEW), len(phases), len(shifts), flat_hm_size))
    # R: reference heatmaps averaged across theta phases (n_clusters, n_dirs, flat_hm_size)
    R = np.zeros((len(cluster_unique_IDs), len(NSEW), flat_hm_size))
    for i, cluster in enumerate(cluster_unique_IDs):
        if verbose:
            print(cluster)
        for j, _dir in enumerate(NSEW):
            dir_df = dir2df[_dir]
            pos = dir_df.centroid_position.values
            for k, phase in enumerate(phases):
                spikes = dir_df.spike_count[cluster][phase].values
                upscaled = _get_upscaled_ratemap(spikes, pos, ratemap_kwargs, scale_factor, smooth_SD, upscale_bin_size)
                for l, shift in enumerate(shifts):
                    T[i, j, k, l] = _shift_ratemap(upscaled, _dir, shift).flatten()
            avg_spikes = dir_df.spike_count[cluster].sum(axis=1).values / len(phases)
            R[i, j] = _get_upscaled_ratemap(
                avg_spikes, pos, ratemap_kwargs, scale_factor, smooth_SD, upscale_bin_size
            ).flatten()
    if return_as == "session_tensors":
        return T, R, phases, shifts_cm
    elif return_as == "session_features":
        n_clusters, n_dirs, n_phases, n_shifts, flat_hm_size = T.shape
        # Flatten all clusters, dirs, and heatmap bins into a single session feature vector
        #  -> t: (n_phases, n_shifts, n_clusters * n_dirs * flat_hm_size)
        t = T.transpose(2, 3, 0, 1, 4).reshape(n_phases, n_shifts, n_clusters * n_dirs * flat_hm_size)
        #  -> r: (n_clusters * n_dirs * flat_hm_size,)
        r = R.reshape(n_clusters * n_dirs * flat_hm_size)
        if all_shifts_valid:  # saves alot of memory
            n_feature = (~np.isnan(r)).sum() // n_clusters
            # bins valid (non-NaN) across all phases and shifts (should be same across phases)
            valid_mask = ~np.any(np.isnan(t), axis=(0, 1))  # (n_clusters * n_dirs * flat_hm_size,)
            n_valid_feature = valid_mask.sum() // n_clusters
            pct_removed = 100 * (1 - n_valid_feature / n_feature)
            if verbose:
                print(f"Features removed by all-shifts validity mask: {pct_removed:.1f}% ")
            r = r[valid_mask]
            t = t[:, :, valid_mask]
        return t, r, phases, shifts_cm
    else:
        raise ValueError(f"Invalid return_as value: {return_as}")


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


def _shift_all_ratemaps(rate_map, direction, shifts):
    """
    Compute all shifted versions of rate_map in one vectorized operation.

    Instead of calling _shift_ratemap in a loop, this pads the map once and
    uses a sliding window view to extract all shifts simultaneously.

    Args:
        rate_map: (H, W) 2D ratemap
        direction: one of NSEW
        shifts: 1D array of integer shift amounts (e.g. np.arange(-n, n+1))

    Returns:
        (len(shifts), H, W) array — same result as stacking _shift_ratemap calls
    """
    dy, dx = _DIR_SHIFT[direction]
    rows, cols = rate_map.shape
    n_pad = len(shifts) // 2  # == n_shifts
    if dy != 0:  # N or S: shift along rows
        padded = np.full((rows + 2 * n_pad, cols), np.nan)
        padded[n_pad : n_pad + rows, :] = rate_map
        offsets = n_pad - dy * shifts  # start row for each shift
        windows = np.lib.stride_tricks.sliding_window_view(padded, (rows, cols))
        # windows shape: (2*n_pad+1, 1, rows, cols)
        return windows[offsets, 0, :, :]
    else:  # E or W: shift along cols
        padded = np.full((rows, cols + 2 * n_pad), np.nan)
        padded[:, n_pad : n_pad + cols] = rate_map
        offsets = n_pad - dx * shifts  # start col for each shift
        windows = np.lib.stride_tricks.sliding_window_view(padded, (rows, cols))
        # windows shape: (1, 2*n_pad+1, rows, cols)
        return windows[0, offsets, :, :]


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
