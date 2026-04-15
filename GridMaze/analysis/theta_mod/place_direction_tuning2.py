"""
Another idea for measuring theta mod place-direction tuning
@peterdoohan
"""

# %% Imports
import json
import pandas as pd
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.theta_mod import theta_utils as tu
from GridMaze.analysis.neGLM import load_model_sets as lms
from GridMaze.analysis.neGLM import variance_explained as ve


from GridMaze.analysis.cluster_tuning import spatial
from scipy.ndimage import gaussian_filter, zoom

# %% Global variables

from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

RESULTS_DIR = RESULTS_PATH / "theta_mod" / "place_direction_tuning"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)


NSEW = ["N", "S", "E", "W"]

_DIR_SHIFT = {"N": (1, 0), "S": (-1, 0), "E": (0, 1), "W": (0, -1)}


# %% plotting


def plot_theta_mod_tuning_summary(
    SSE_df,
    maze_names=["maze_1", "maze_2", "rooms_maze"],
    place_tuned_only=False,
    min_split_half_corr=0.5,
    late_sessions_only=False,
    demean=True,
    color="grey",
    label=None,
    norm=False,
    print_stats=True,
    ax=None,
):
    # filter data
    df = SSE_df[SSE_df.maze_name.isin(maze_names)]
    if late_sessions_only:
        df = df[df.late_session]
    if place_tuned_only:
        place_tuned = get_place_tuned_clusters()
        df = df[df.cluster_unique_ID.isin(place_tuned)]
    if min_split_half_corr is not None:
        df = df[df.split_half_corr.gt(min_split_half_corr)]
    # per-subject × phase preferred shift: MSE summed across clusters/maze/day, argmin over shift_m
    grouped = df.groupby(["subject_ID", "phase", "shift_m"])[["SSE", "n_features"]].sum()
    grouped["MSE"] = grouped["SSE"] / grouped["n_features"]
    mse = grouped["MSE"].unstack("shift_m")  # (subject × phase, shift_m)
    best_shifts = mse.idxmin(axis=1) * 1000  # m → mm
    best_shifts = best_shifts.unstack("phase")  # (subject, phase)
    # demean across phases per subject
    if demean:
        best_shifts = best_shifts.sub(best_shifts.mean(axis=1), axis=0)
    # plot + sinusoid fit + theta modulation stats via theta_utils
    tu.plot_decoding_bias(
        best_shifts,
        color=color,
        label=label,
        ylabel="preferred shift (mm)",
        norm=norm,
        print_stats=print_stats,
        ax=ax,
    )
    return best_shifts


# %%


def get_place_tuned_clusters():
    """
    get cluster unique IDs for cells that have only sig ve by place-direction (place for short)
    """
    feature_tuned_df = ve.get_feature_tuned_df(
        lms.load_model_set_cv_scores("variance_explained_all_sessions"),
        reduced_models=["remove_distance_to_goal", "remove_place_direction"],
    )
    place_tuned = (
        feature_tuned_df[(feature_tuned_df.place_direction)]  # ~feature_tuned_df.distance_to_goal &
        .index.get_level_values(1)
        .values
    )
    return place_tuned


def get_preferred_shift(df, min_split_half_corr=0.7):
    """ """
    _df = df.copy()
    if min_split_half_corr is not None:
        _df = _df[_df.split_half_corr.gt(min_split_half_corr)]
    # grouped = _df.groupby(["subject_ID", "phase", "shift_m"])[["SSE", "n_features"]].sum()
    grouped = _df.groupby(["subject_ID", "cluster_unique_ID", "phase", "shift_m"])[["SSE", "n_features"]].sum()
    grouped["MSE"] = grouped["SSE"] / grouped["n_features"]
    mse = grouped["MSE"].unstack("shift_m")  # (n_phases, n_shifts)
    best_shifts = mse.idxmin(axis=1)  # Series: phase → best shift_cm
    return best_shifts.groupby(level=[0, 2]).mean().unstack(0)


# %%
def get_SSE_df(
    split_half_corr_thres=0.3,
    single_units_only=True,
    phase_halfwidth=0,
    bin_size=0.08,
    upscale_bin_size=0.002,
    smooth_SD=0.02,
    xy_range=(0, 1.4),
    shift_range=0.02,
    zscore=True,
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
                single_units_only=single_units_only,
                phase_halfwidth=phase_halfwidth,
                bin_size=bin_size,
                upscale_bin_size=upscale_bin_size,
                smooth_SD=smooth_SD,
                xy_range=xy_range,
                shift_range=shift_range,
                zscore=zscore,
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
            days_on_maze="all",
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
    split_half_corr_thres=0.3,
    single_units_only=True,
    phase_halfwidth=0,
    bin_size=0.08,
    upscale_bin_size=0.002,
    smooth_SD=0.02,
    xy_range=(0, 1.4),
    shift_range=0.02,
    zscore=True,
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
    nav_spikes_df = get_theta_stratified_nav_spikes_df(
        session, split_half_corr_thres, single_units_only=single_units_only, verbose=verbose
    )
    if nav_spikes_df is None:
        return None
    cluster_unique_IDs = nav_spikes_df.spike_count.columns.get_level_values(0).unique()
    phases = nav_spikes_df.spike_count.columns.get_level_values(1).unique()
    scale_factor = int(bin_size / upscale_bin_size)
    n_shifts = int(shift_range / upscale_bin_size)
    shifts = np.arange(-n_shifts, n_shifts + 1)
    shifts_m = shifts * upscale_bin_size
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

    phase_list = list(phases)
    n_phases = len(phase_list)
    window_len = 2 * phase_halfwidth + 1
    records = []
    for cluster in cluster_unique_IDs:
        if verbose:
            print(cluster)
        # n_features for this cluster: total valid bins across all directions (constant across phase/shift)
        n_features_cluster = sum(n_valid)
        # Pre-compute ref_masked per direction — identical for all phases.
        # Reference pools spikes across ALL theta phases (summed) for a denser, lower-variance
        # estimate of the place field, then rescales by window_len/n_phases so the reference
        # ratemap is on the same effective-rate scale as the phase-specific window-summed maps.
        # If zscore=True, divide by the ref std so each cluster/direction contributes
        # equally regardless of firing rate. The same std is applied to the shifted maps,
        # so the normalisation is consistent: (shifted_z - ref_z) = (shifted - ref) / ref_std.
        dir_ref_masked = {}
        dir_ref_mean = {}
        dir_ref_std = {}
        for j, _dir in enumerate(NSEW):
            if n_valid[j] == 0:
                continue
            dir_df = dir2df[_dir]
            pos = dir_df.centroid_position.values
            ref_spikes = dir_df.spike_count[cluster].sum(axis=1).values
            ref_up = _get_upscaled_ratemap(ref_spikes, pos, ratemap_kwargs, scale_factor, smooth_SD, upscale_bin_size)
            ref_up = ref_up * (window_len / n_phases)
            ref_masked = ref_up.flatten()[dir_valid_masks[j]]
            if zscore:
                dir_ref_mean[j] = np.nanmean(ref_masked)
                ref_std = np.nanstd(ref_masked)
                dir_ref_std[j] = ref_std if ref_std > 0 else 1.0
                dir_ref_masked[j] = (ref_masked - dir_ref_mean[j]) / dir_ref_std[j]
            else:
                dir_ref_masked[j] = ref_masked
        for phase_idx, phase in enumerate(phase_list):
            # Rolling sum of spike counts across ±phase_halfwidth adjacent phases (circular)
            window_indices = [(phase_idx + d) % n_phases for d in range(-phase_halfwidth, phase_halfwidth + 1)]
            window_phases = [phase_list[i] for i in window_indices]
            # Accumulate SSE across directions for each shift
            sse_per_shift = np.zeros(len(shifts))
            for j, _dir in enumerate(NSEW):
                if n_valid[j] == 0:
                    continue
                dir_df = dir2df[_dir]
                pos = dir_df.centroid_position.values
                shift_spikes = dir_df.spike_count[cluster][window_phases].sum(axis=1).values
                shift_up = _get_upscaled_ratemap(
                    shift_spikes, pos, ratemap_kwargs, scale_factor, smooth_SD, upscale_bin_size
                )
                for l, shift in enumerate(shifts):
                    shifted_masked = _shift_ratemap(shift_up, _dir, shift).flatten()[dir_valid_masks[j]]
                    if zscore:
                        shifted_masked = (shifted_masked - dir_ref_mean[j]) / dir_ref_std[j]
                    sse_per_shift[l] += np.sum((shifted_masked - dir_ref_masked[j]) ** 2)
            for l, shift_m in enumerate(shifts_m):
                records.append(
                    {
                        "cluster_unique_ID": cluster,
                        "phase": phase,
                        "shift_m": shift_m,
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
    # use nearest-neighbour for the binary validity mask (interpolation would blur boundaries)
    upscaled = np.repeat(np.repeat(occ_map, scale_factor, axis=0), scale_factor, axis=1)
    valid_2d = ~np.isnan(upscaled)
    for shift in shifts:
        valid_2d &= ~np.isnan(_shift_ratemap(upscaled, direction, shift))
    return valid_2d.flatten()


def _nan_aware_zoom(arr, scale_factor):
    """Upscale a 2D array with cubic interpolation, correctly handling NaN (unvisited) bins."""
    nan_mask = np.isnan(arr)
    weights = (~nan_mask).astype(float)
    data = np.nan_to_num(arr, nan=0.0)
    zoomed_data = zoom(data, scale_factor, order=3)
    zoomed_weights = zoom(weights, scale_factor, order=3)
    valid = zoomed_weights > 0.01  # small threshold to avoid dividing by near-zero
    result = np.full_like(zoomed_data, np.nan)
    result[valid] = zoomed_data[valid] / zoomed_weights[valid]
    return result


def _get_upscaled_ratemap(spikes, pos, ratemap_kwargs, scale_factor, smooth_SD, upscale_bin_size):
    """Compute ratemap at coarse resolution, upscale with NaN-aware cubic interpolation, then smooth."""
    rate_map, _, _ = spatial.get_2D_ratemap(spikes, pos, **ratemap_kwargs)
    upscaled = _nan_aware_zoom(rate_map, scale_factor)
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
    return h


def get_theta_stratified_nav_spikes_df(
    session,
    split_half_corr_thres=0.3,
    single_units_only=True,
    moving_thres=0.075,
    verbose=True,
):
    # load data
    place_dir_metrics = session.cluster_place_direction_tuning_metrics.copy()
    navigation_df = session.navigation_df.copy()
    spikes_df = session.navigation_theta_spike_counts_df.reset_index(drop=True)
    # filter for clusters with strong pd tuning
    mask = place_dir_metrics.split_half_corr.value.gt(split_half_corr_thres)
    if single_units_only:
        mask = mask & place_dir_metrics.single_unit.squeeze()
    consider_clusters = place_dir_metrics[mask].index.values
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


# %% Development / diagnostics


def test_session_SSE(subject_ID=None, maze_name=None, day_on_maze=None, verbose=True, **kwargs):
    """
    Run get_session_SSE_df on a single session and print diagnostics.
    Useful for rapid parameter tuning without running the full pipeline.

    Pass any get_session_SSE_df parameter via **kwargs to override defaults.
    """
    # load a single session directly (faster than loading multiple and indexing)
    _subject = subject_ID if subject_ID is not None else "m6"
    _maze = maze_name if maze_name is not None else "maze_1"
    _day = day_on_maze if day_on_maze is not None else 12
    session = gs.get_maze_sessions(
        subject_IDs=[_subject], maze_names=[_maze], days_on_maze=[_day], with_data="all", must_have_data=False
    )
    if not isinstance(session, list):
        session = [session]
    if len(session) == 0:
        print(f"No sessions found for {_subject} / {_maze} / day {_day}")
        return None
    session = session[0]
    print(f"Testing on session: {session.name}")
    # run analysis
    df = get_session_SSE_df(session, verbose=verbose, **kwargs)
    if df is None:
        print("No valid clusters found for this session.")
        return None
    # diagnostics
    clusters = df.cluster_unique_ID.unique()
    phases = df.phase.unique()
    shifts = df.shift_m.unique()
    print(f"\n--- Diagnostics ---")
    print(f"  Clusters:  {len(clusters)}")
    print(f"  Phases:    {len(phases)}")
    print(f"  Shifts:    {len(shifts)}  (range: {shifts.min():.4f} to {shifts.max():.4f} m)")
    print(f"  n_features per cluster: {df.groupby('cluster_unique_ID').n_features.first().describe()}")
    # SSE landscape: is it flat or does it vary across shifts?
    mse_by_shift = df.groupby(["phase", "shift_m"])[["SSE", "n_features"]].sum()
    mse_by_shift["MSE"] = mse_by_shift["SSE"] / mse_by_shift["n_features"]
    mse_pivot = mse_by_shift["MSE"].unstack("shift_m")
    shift_range_per_phase = mse_pivot.max(axis=1) - mse_pivot.min(axis=1)
    print(f"\n  MSE range across shifts (per phase):")
    print(f"    mean: {shift_range_per_phase.mean():.6f}")
    print(f"    min:  {shift_range_per_phase.min():.6f}")
    print(f"    max:  {shift_range_per_phase.max():.6f}")
    # preferred shifts
    best_shifts = mse_pivot.idxmin(axis=1)
    best_mse = mse_pivot.min(axis=1)
    print(f"\n  Preferred shift per phase (m):")
    for phase, shift in best_shifts.items():
        print(f"    phase {phase:+.2f}: {shift:+.4f} m  (MSE={best_mse.loc[phase]:.4f})")
    # Per-phase MSE curves: demean each phase to show relative shape
    mse_demeaned = mse_pivot.sub(mse_pivot.mean(axis=1), axis=0)
    shift_vals = mse_pivot.columns.values
    print(f"\n  Per-phase MSE vs shift (demeaned across shifts):")
    print(f"  {'phase':>8s}", end="")
    # print a subset of shift values for readability
    step = max(1, len(shift_vals) // 10)
    display_shifts = shift_vals[::step]
    for s in display_shifts:
        print(f"  {s:+.004f}", end="")
    print()
    for phase in mse_demeaned.index:
        print(f"  {phase:+8.2f}", end="")
        for s in display_shifts:
            val = mse_demeaned.loc[phase, s]
            print(f"  {val:+.4f}", end="")
        print()
    # check for NaN
    n_nan = df.SSE.isna().sum()
    if n_nan > 0:
        print(f"\n  WARNING: {n_nan} NaN values in SSE!")
    else:
        print(f"\n  No NaN values in SSE (good)")
    # plot MSE vs shift per phase
    import seaborn as sns

    f, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.spines[["top", "right"]].set_visible(False)
    colors = sns.color_palette("husl", len(mse_demeaned.index))
    for i, phase in enumerate(mse_demeaned.index):
        ax.plot(shift_vals * 1000, mse_demeaned.loc[phase].values, color=colors[i], label=f"{phase:+.1f}")
    ax.set_xlabel("shift (mm)")
    ax.set_ylabel("MSE (demeaned)")
    ax.set_title("MSE vs shift per theta phase")
    ax.legend(fontsize=6, ncol=2, title="theta phase")
    ax.axvline(0, color="k", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()
    return df
