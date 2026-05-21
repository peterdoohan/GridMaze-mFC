"""
Theta-cycle x phase-bin input data structure for theta modulation analyses.

Higher temporal resolution alternative to `session.navigation_theta_spike_counts_df`:
rows are (theta_cycle_idx, phase_bin) so each row spans the actual duration of one
phase bin within one cycle (variable width, driven by local theta period). Phase bins
match `bin_lfp_phase` (default 12) so they line up with the modulation-profile bins
used in `decoding_offsets.py`.

@peterdoohan
"""

# %% Imports
import numpy as np
import pandas as pd

from GridMaze.analysis.core import convert
from GridMaze.analysis.lfp import lfp_utils as lu
from GridMaze.analysis.processing import get_lfp_aligned_spike_counts as la

from GridMaze.analysis.neGLM import variance_explained_null as ve
from GridMaze.analysis.neGLM import load_model_sets as lms

# %% Global variables
THETA_RANGE = (7, 10)
FS_LFP = 1500

# %% Main entry point


def get_session_df(
    session,
    place_trough_bins=[1, 2, 3],
    distance_peak_bins=[4, 5, 6],
    distance_trough_bins=[10, 11, 0],
    place_tuned_cells=None,
    **input_data_kwargs,
):
    """Placeholder for the per-session analysis pipeline.

    Leave-one-trial-out CV over navigation trials: each iteration holds out one
    trial as `test_df`, with all remaining trials as `train_df`. Inside the loop
    a per-cycle place-tuned population "update" score (cosine similarity between
    cycle c and cycle c+1's firing-rate vectors in place_trough_bins, stored on
    cycle c) is added to both dataframes, respecting trial boundaries.

    `place_tuned_cells`: None → load via `get_place_direction_tuned_neurons()`;
    otherwise an array of cluster_unique_ID strings.
    """
    input_data = get_input_data(session, **input_data_kwargs)
    if place_tuned_cells is None:
        place_tuned_cells = get_place_direction_tuned_neurons()
    session_cells = input_data.spike_count.columns
    place_cells_in_session = [c for c in np.unique(place_tuned_cells) if c in session_cells]

    trials = input_data.trial.dropna().unique()
    for held_out_trial in trials:
        test_mask = input_data.trial == held_out_trial
        test_df = _add_place_update_column(input_data[test_mask], place_cells_in_session, place_trough_bins)
        train_df = _add_place_update_column(input_data[~test_mask], place_cells_in_session, place_trough_bins)
        # TODO: decoder fit on train_df, evaluate on test_df, append per-trial result
    return


def get_input_data(session, n_bins=12, shank=3, navigation_only=True, max_steps_to_goal=20, exclude_at_goal=True):
    """Build the theta-cycle x phase-bin input dataframe for one session.

    Returns a single dataframe with MultiIndex rows (cycle_idx, phase_bin) and 2-level
    MultiIndex columns. Top-level column groups: every group already in
    `session.navigation_df`, plus `phase_window`, `cycle_metrics`, `spike_count`.
    Cycle metrics are constant within a cycle (broadcast across the bin rows).

    Filters (applied to the assembled df, all aligned to phase-bin midpoints):
      navigation_only=True       keep only rows where trial_phase == "navigation"
      max_steps_to_goal=20       keep only rows where steps_to_goal.future < this
      exclude_at_goal=True       drop rows where goal == maze_position.simple
    Set any filter to None / False to skip it.
    """
    # --- LFP, theta phase, phase bins ---
    raw_lfp = lu.get_LFP(session, shank=shank)
    filt_osc, theta_phase = la.get_lfp_phase(raw_lfp, freq_range=THETA_RANGE, N=4, return_filtered=True)
    _, bin_indices = la.bin_lfp_phase(theta_phase, n_bins=n_bins)
    lfp_times = session.lfp_times

    # --- detect (cycle, phase_bin) windows ---
    start_samples, end_samples = _detect_cycle_phase_windows(bin_indices, n_bins=n_bins)
    n_cycles = start_samples.shape[0]

    start_times = lfp_times[start_samples]
    end_times = lfp_times[end_samples]
    midpoint_times = (start_times + end_times) / 2

    # --- spike counts per (cluster, cycle, bin) ---
    spike_times = np.asarray(session.spike_times).reshape(-1)
    spike_clusters = np.asarray(session.spike_clusters).reshape(-1)
    cluster_IDs = np.sort(np.unique(spike_clusters)).astype(np.float64)
    cluster_unique_IDs = convert.cluster_IDs2scluster_unique_IDs(session.session_info, cluster_IDs)

    start_flat = start_times.ravel()
    end_flat = end_times.ravel()
    spike_counts = np.zeros((len(cluster_IDs), n_cycles * n_bins), dtype=np.int32)
    for i, cluster_id in enumerate(cluster_IDs):
        cst = spike_times[spike_clusters == cluster_id]
        spike_counts[i] = np.searchsorted(cst, end_flat) - np.searchsorted(cst, start_flat)

    # --- per-cycle quality metrics (broadcast to all bins of that cycle) ---
    amplitudes, periods, mean_lfp_powers = _compute_cycle_metrics(
        filt_osc, raw_lfp, start_samples, end_samples, start_times, end_times
    )

    # --- assemble ---
    row_index = pd.MultiIndex.from_product([np.arange(n_cycles), np.arange(n_bins)], names=["cycle_idx", "phase_bin"])

    # navigation aligned to midpoint of each phase bin (nearest frame, with duplication)
    nav_block = _align_navigation(session.navigation_df, midpoint_times.ravel(), row_index)

    phase_window_block = pd.DataFrame(
        {
            ("phase_window", "start_time"): start_flat,
            ("phase_window", "end_time"): end_flat,
            ("phase_window", "midpoint_time"): midpoint_times.ravel(),
            ("phase_window", "duration"): end_flat - start_flat,
        },
        index=row_index,
    )
    phase_window_block.columns = pd.MultiIndex.from_tuples(phase_window_block.columns)

    cycle_metrics_block = pd.DataFrame(
        {
            ("cycle_metrics", "amplitude"): np.repeat(amplitudes, n_bins),
            ("cycle_metrics", "period"): np.repeat(periods, n_bins),
            ("cycle_metrics", "mean_lfp_power"): np.repeat(mean_lfp_powers, n_bins),
        },
        index=row_index,
    )
    cycle_metrics_block.columns = pd.MultiIndex.from_tuples(cycle_metrics_block.columns)

    spike_count_block = pd.DataFrame(
        spike_counts.T,
        index=row_index,
        columns=pd.MultiIndex.from_product([["spike_count"], cluster_unique_IDs]),
    )

    input_data = pd.concat([nav_block, phase_window_block, cycle_metrics_block, spike_count_block], axis=1)

    # --- row filters ---
    if navigation_only:
        input_data = input_data[input_data.trial_phase == "navigation"]
    if max_steps_to_goal is not None:
        input_data = input_data[input_data.steps_to_goal.future < max_steps_to_goal]
    if exclude_at_goal:
        input_data = input_data[input_data.goal != input_data.maze_position.simple]
    return input_data


# %% Per-trial place-tuned population update


def _add_place_update_column(df, place_cells, place_trough_bins):
    """Add ('place_update', 'cos_sim') and ('place_update', 'cycle_complete') columns.

    cos_sim: within-trial cosine similarity between the place-tuned population
        firing-rate vectors of cycle c and cycle c+1, computed over place_trough_bins
        (using whatever trough bins survived row-filters in each cycle), stored on
        cycle c. NaN at the last cycle of every trial (no successor) and where the
        raw-LFP adjacency check fails or either vector is all-zero.
    cycle_complete: True if the cycle still has all n_bins phase bins after row
        filters; False otherwise. Use this downstream to subset only-complete cycles.

    Raises ValueError loudly if:
      - any cycle in a trial has ALL of its place_trough_bins missing (can't form
        the rate vector), or
      - any non-last cycle of a trial has no immediate cycle_idx successor (c+1
        missing from the trial — implies non-consecutive cycle_idx, which the
        diagnostic shows never happens normally).
    """
    n_bins = int(df.index.get_level_values("phase_bin").max()) + 1
    update = pd.Series(np.nan, index=df.index, dtype=float)
    complete_flag = pd.Series(False, index=df.index, dtype=bool)

    for trial, trial_df in df.groupby(df.trial):
        if trial_df.empty or not place_cells:
            continue

        # completeness tracking (not a filter — just a flag downstream code can use)
        cycle_sizes = trial_df.groupby(level="cycle_idx").size()
        complete_cycles = set(cycle_sizes[cycle_sizes == n_bins].index)
        if complete_cycles:
            mask = trial_df.index.get_level_values("cycle_idx").isin(complete_cycles)
            complete_flag.loc[trial_df.index[mask]] = True

        # raw-LFP adjacency lookup (entries only present where that bin survived filtering)
        try:
            cycle_starts = trial_df.xs(0, level="phase_bin").phase_window.start_time
        except KeyError:
            cycle_starts = pd.Series(dtype=float)
        try:
            cycle_ends = trial_df.xs(n_bins - 1, level="phase_bin").phase_window.end_time
        except KeyError:
            cycle_ends = pd.Series(dtype=float)

        # per-cycle rates over place_trough_bins (NO completeness filter)
        trough = trial_df[trial_df.index.get_level_values("phase_bin").isin(place_trough_bins)]
        spikes_by_cycle = trough.spike_count[place_cells].groupby(level="cycle_idx").sum()
        dur_by_cycle = trough.phase_window.duration.groupby(level="cycle_idx").sum()
        rates_by_cycle = spikes_by_cycle.div(dur_by_cycle, axis=0)

        # loud fail #1: every cycle in the trial must have at least one trough bin present
        present = set(rates_by_cycle.index)
        all_cycles_in_trial = set(trial_df.index.get_level_values("cycle_idx").unique())
        missing_all_trough = all_cycles_in_trial - present
        if missing_all_trough:
            raise ValueError(
                f"trial {trial}: cycles {sorted(missing_all_trough)} have ALL "
                f"place_trough_bins {place_trough_bins} missing after row filters — "
                f"cannot form firing-rate vector."
            )

        trial_max_cycle = max(present)
        for c in sorted(present):
            if c == trial_max_cycle:
                continue  # last cycle of trial, no successor in this trial
            # loud fail #2: c+1 absent within the trial → non-consecutive cycle_idx
            if (c + 1) not in present:
                raise ValueError(
                    f"trial {trial}: cycle {c} has no successor cycle {c + 1} in trial "
                    f"(non-consecutive cycle_idx — should not happen)."
                )
            # raw-LFP adjacency check, only when both endpoint bins survived
            if c in cycle_ends.index and (c + 1) in cycle_starts.index:
                if not np.isclose(cycle_ends.loc[c], cycle_starts.loc[c + 1]):
                    continue
            v_curr = rates_by_cycle.loc[c].values
            v_next = rates_by_cycle.loc[c + 1].values
            n_curr = np.linalg.norm(v_curr)
            n_next = np.linalg.norm(v_next)
            if n_curr == 0 or n_next == 0:
                continue
            val = float(np.dot(v_curr, v_next) / (n_curr * n_next))
            in_cycle = trial_df.index.get_level_values("cycle_idx") == c
            update.loc[trial_df.index[in_cycle]] = val

    df = df.copy()
    df[("place_update", "cos_sim")] = update.values
    df[("place_update", "cycle_complete")] = complete_flag.values
    return df


# %% Cycle / phase-bin window detection


def _detect_cycle_phase_windows(bin_indices, n_bins):
    """Find first LFP sample of each phase bin within every complete theta cycle.

    A cycle is defined as the LFP samples between consecutive `bin_idx (n_bins-1) -> 0`
    transitions. Cycles missing any phase bin (rare, only on noisy theta) are dropped.

    Returns
    -------
    start_samples : (n_cycles, n_bins) int array
        First LFP sample where this phase bin starts within the cycle.
    end_samples : (n_cycles, n_bins) int array
        First LFP sample of the next phase bin (exclusive end).
    """
    wrap_idxs = np.flatnonzero((bin_indices[1:] == 0) & (bin_indices[:-1] == n_bins - 1)) + 1
    n_complete = len(wrap_idxs) - 1  # cycles bracketed by two wraps
    start_samples = np.zeros((n_complete, n_bins), dtype=np.int64)
    next_cycle_starts = wrap_idxs[1:].copy()
    valid = np.ones(n_complete, dtype=bool)
    for k in range(n_complete):
        cycle_start = wrap_idxs[k]
        cycle_end = wrap_idxs[k + 1]
        cycle_bins = bin_indices[cycle_start:cycle_end]
        for j in range(n_bins):
            firsts = np.flatnonzero(cycle_bins == j)
            if len(firsts) == 0:
                valid[k] = False
                break
            start_samples[k, j] = cycle_start + firsts[0]
    start_samples = start_samples[valid]
    next_cycle_starts = next_cycle_starts[valid]
    end_samples = np.empty_like(start_samples)
    end_samples[:, :-1] = start_samples[:, 1:]
    end_samples[:, -1] = next_cycle_starts
    return start_samples, end_samples


# %% Per-cycle metrics


def _compute_cycle_metrics(filt_osc, raw_lfp, start_samples, end_samples, start_times, end_times):
    """Amplitude (peak-to-peak filt), period (s), mean raw LFP power per cycle."""
    n_cycles = start_samples.shape[0]
    cycle_first = start_samples[:, 0]
    cycle_last_excl = end_samples[:, -1]
    amplitudes = np.zeros(n_cycles)
    mean_lfp_powers = np.zeros(n_cycles)
    for k in range(n_cycles):
        s, e = cycle_first[k], cycle_last_excl[k]
        seg = filt_osc[s:e]
        amplitudes[k] = seg.max() - seg.min()
        raw_seg = raw_lfp[s:e]
        mean_lfp_powers[k] = float(np.mean(raw_seg.astype(np.float64) ** 2))
    periods = end_times[:, -1] - start_times[:, 0]
    return amplitudes, periods, mean_lfp_powers


# %% Navigation alignment


def _align_navigation(navigation_df, midpoint_times_flat, row_index):
    """For each midpoint time, pick the nearest video-frame row from navigation_df."""
    frame_times = navigation_df.time.values.ravel()
    right = np.searchsorted(frame_times, midpoint_times_flat)
    right = np.clip(right, 0, len(frame_times) - 1)
    left = np.clip(right - 1, 0, len(frame_times) - 1)
    left_dist = np.abs(frame_times[left] - midpoint_times_flat)
    right_dist = np.abs(frame_times[right] - midpoint_times_flat)
    nearest = np.where(left_dist <= right_dist, left, right)
    aligned = navigation_df.iloc[nearest].copy()
    aligned.index = row_index
    return aligned


# %%


def get_place_direction_tuned_neurons():
    """
    Cluster IDs of neurons selectively tuned (via neGLM variance-explained) place-direction.
    """
    feature_tuned_df = ve.get_feature_tuned_df(
        lms.load_model_set_cv_scores("variance_explained_multiunit"),
        r2_thres=0.05,
    )
    return feature_tuned_df[feature_tuned_df.place_direction].index.get_level_values(-1).values
