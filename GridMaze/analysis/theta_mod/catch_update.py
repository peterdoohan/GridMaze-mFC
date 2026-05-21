"""
Cycle-resolved theta-modulation pipeline (catch-update analysis).

Public entry points (in pipeline order):

  1. `get_input_data(session)` — builds the per-(cycle, phase_bin) dataframe:
        LFP-derived cycle/bin windows → spike counts → per-cycle metrics →
        navigation alignment (nearest video frame) → row filters
        (navigation_only / max_steps_to_goal / exclude_at_goal) →
        distance-to-goal binning → place-tuned population update score.

  2. `get_session_df(session)` — wraps `get_input_data` in a leave-one-trial-out
        loop: each fold trains a distance-to-goal logistic-regression decoder on
        super-phase-averaged firing rates of the training trials, then applies
        it twice per test cycle (at `distance_peak_bins` and
        `distance_trough_bins`) to produce a per-cycle `distance_error =
        trough_pred - peak_pred`. Returns one row per test cycle.

  3. `get_session_corr(session_df)` — correlates the resulting
        `place_update.cos_sim` against `decoder.distance_error` with optional
        amplitude filtering and partialled-out covariates (speed,
        distance-to-goal, head-direction).

Higher temporal resolution alternative to `session.navigation_theta_spike_counts_df`:
rows are (cycle_idx, phase_bin), each spanning the actual duration of one phase
bin within one cycle. Phase bin definitions match `bin_lfp_phase` (default 12) so
they line up with the modulation-profile bins used in `decoding_offsets.py`.

@peterdoohan
"""

# %% Imports
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from GridMaze.analysis.core import convert
from GridMaze.analysis.core import get_clusters as gc
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
    distance_peak_bins=[3, 4, 5],
    distance_trough_bins=[9, 10, 11],
    n_training_phases=3,
    normalise_X=True,
    C=1.0,
    output="weighted",
    **input_data_kwargs,
):
    """Stage 2 — LOO distance-to-goal decoder with peak/trough theta-phase readout.

    For each held-out trial: train a logistic-regression distance decoder on
    super-phase-averaged firing rates from the training trials, then apply it to
    each test cycle at `distance_peak_bins` and `distance_trough_bins`. Returns
    one row per test cycle with `decoder.peak_pred`, `decoder.trough_pred`,
    `decoder.distance_error = trough_pred - peak_pred`, plus the session-level
    MAE constants and all the nav / `place_update` / `cycle_metrics` columns
    that `get_input_data` produced. `spike_count` columns are dropped.

    Non-obvious args:
      distance_peak_bins / distance_trough_bins: theta phase bins (early / late
        in cycle by default) whose average rates form the test-time peak and
        trough rate vectors.
      n_training_phases: bins per super-phase group used for training (must
        divide n_bins evenly; default 3 → 4 super-bins per cycle).
      C, normalise_X, output: LogisticRegression hyperparameters; `output` is
        either "weighted" (prob-weighted bin-mid readout) or "max" (argmax).
    """
    input_data = get_input_data(session, **input_data_kwargs)

    # one-time setup
    cluster_ids = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=True,
    )
    decoder_cells = [c for c in input_data.spike_count.columns if c in set(cluster_ids)]
    distance_bin_mids = np.array(sorted(input_data.distance_bin_mid.dropna().unique()))
    n_bins = int(input_data.index.get_level_values("phase_bin").max()) + 1
    assert n_bins % n_training_phases == 0, "n_training_phases must divide n_bins"
    if output not in ("weighted", "max"):
        raise ValueError(f"output must be 'weighted' or 'max', got {output!r}")

    per_cycle_results = []
    for held_out_trial in input_data.trial.dropna().unique():
        test_mask = input_data.trial == held_out_trial
        train_df = input_data[~test_mask]
        # restrict test to complete cycles so bin-0 lookup for cycle_meta is safe
        test_df = input_data[test_mask & input_data.place_update.cycle_complete]

        # --- training data: firing rates over super-phases, complete cycles only ---
        rates_train, y_train = _super_phase_rates(train_df, decoder_cells, n_training_phases)
        if rates_train.empty:
            continue
        X_train = rates_train.values
        scaler = StandardScaler().fit(X_train) if normalise_X else None
        if scaler is not None:
            X_train = scaler.transform(X_train)
        decoder = LogisticRegression(C=C, random_state=0, max_iter=10_000, class_weight="balanced").fit(
            X_train, y_train.values
        )
        # mids in the order of decoder.classes_ (some bin_ids may be absent in this fold)
        mids_for_classes = distance_bin_mids[decoder.classes_]

        # --- test data: one peak rate vec + one trough rate vec per cycle ---
        peak_rates = _cycle_subset_rates(test_df, decoder_cells, distance_peak_bins)
        trough_rates = _cycle_subset_rates(test_df, decoder_cells, distance_trough_bins)
        shared_cycles = peak_rates.index.intersection(trough_rates.index)
        if len(shared_cycles) == 0:
            continue
        peak_rates = peak_rates.loc[shared_cycles]
        trough_rates = trough_rates.loc[shared_cycles]
        X_peak = peak_rates.values
        X_trough = trough_rates.values
        if scaler is not None:
            X_peak = scaler.transform(X_peak)
            X_trough = scaler.transform(X_trough)

        if output == "weighted":
            peak_pred = decoder.predict_proba(X_peak) @ mids_for_classes
            trough_pred = decoder.predict_proba(X_trough) @ mids_for_classes
        else:
            peak_pred = distance_bin_mids[decoder.predict(X_peak)]
            trough_pred = distance_bin_mids[decoder.predict(X_trough)]
        distance_error = trough_pred - peak_pred

        # --- per-cycle output rows: take bin-0 of test_df for the nav + place_update vars ---
        cycle_meta = test_df.xs(0, level="phase_bin").loc[shared_cycles].copy()
        cycle_meta = cycle_meta.drop(columns=["spike_count"], level=0, errors="ignore")
        cycle_meta[("decoder", "peak_pred")] = peak_pred
        cycle_meta[("decoder", "trough_pred")] = trough_pred
        cycle_meta[("decoder", "distance_error")] = distance_error
        cycle_meta[("decoder", "held_out_trial")] = held_out_trial
        per_cycle_results.append(cycle_meta)

    if not per_cycle_results:
        return pd.DataFrame()
    session_df = pd.concat(per_cycle_results).sort_index()

    # --- session-level MAE between decoded distance and the cycle's true bin midpoint ---
    true_dist = session_df[("distance_bin_mid", "")].values.astype(float)
    peak_mae = float(np.mean(np.abs(session_df[("decoder", "peak_pred")].values - true_dist)))
    trough_mae = float(np.mean(np.abs(session_df[("decoder", "trough_pred")].values - true_dist)))
    session_df[("decoder", "peak_mae")] = peak_mae
    session_df[("decoder", "trough_mae")] = trough_mae
    print(f"{session.name}: decoder MAE — peak={peak_mae:.4f} m, trough={trough_mae:.4f} m")
    return session_df


def get_input_data(
    session,
    n_bins=12,
    shank=3,
    navigation_only=True,
    max_steps_to_goal=14,
    exclude_at_goal=True,
    max_distance=None,
    min_distance=None,
    bin_spacing=0.05,
    place_trough_bins=[1, 2, 3],
    place_tuned_cells=None,
):
    """Stage 1 — build the per-(cycle, phase_bin) input dataframe.

    MultiIndex rows = (cycle_idx, phase_bin); 2-level MultiIndex columns. Pipeline:
        cycle/bin windows from LFP → spike counts → per-cycle metrics →
        navigation aligned to bin midpoints → row filters →
        distance-to-goal binning → place-tuned population update score.

    Output columns: every navigation group + `phase_window` (start/end/midpoint/
    duration) + `cycle_metrics` (amplitude/period/mean_lfp_power, constant per
    cycle) + `spike_count` (per cluster) + `distance_bin`/`distance_bin_mid`/
    `distance_bin_id` + `place_update` (cos_sim / cycle_complete /
    trough_bins_complete).
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

    # --- distance-to-goal binning (over the post-filter range) ---
    dist_series = input_data[("distance_to_goal", "geodesic")]
    if max_distance is None:
        # nudge above the observed max so the topmost left-closed bin includes it.
        # pad must survive the freq-accumulation rounding inside pd.interval_range
        # (a single ulp is not enough — ~n_bins ulps can be lost).
        max_distance = float(dist_series.max()) + bin_spacing * 1e-3
    if min_distance is None:
        min_distance = float(dist_series.min())
    n_distance_bins = int((max_distance - min_distance) / bin_spacing)
    bins = convert._get_distance_bins(
        binning_method="uniform",
        n_distance_bins=n_distance_bins,
        distance_metrics=("distance_to_goal", "geodesic"),
        max_distance=max_distance,
        min_distance=min_distance,
    )
    intervals = pd.cut(dist_series, bins=bins, include_lowest=True)
    input_data.loc[:, ("distance_bin", "")] = intervals.to_numpy()
    input_data.loc[:, ("distance_bin_mid", "")] = [iv.mid if pd.notna(iv) else np.nan for iv in intervals]
    observed_mids = sorted(input_data[("distance_bin_mid", "")].dropna().unique())
    mid_to_id = {m: i for i, m in enumerate(observed_mids)}
    input_data.loc[:, ("distance_bin_id", "")] = input_data[("distance_bin_mid", "")].map(mid_to_id).astype("Int64")

    # --- place-tuned population update (per-trial cosine sim across cycles) ---
    if place_tuned_cells is None:
        place_tuned_cells = get_place_direction_tuned_neurons()
    session_cells = input_data.spike_count.columns
    place_cells_in_session = [c for c in np.unique(place_tuned_cells) if c in session_cells]
    input_data = _add_place_update_column(input_data, place_cells_in_session, place_trough_bins)
    return input_data


# %% Per-trial place-tuned population update


def _add_place_update_column(df, place_cells, place_trough_bins):
    """Append `place_update` columns to the per-(cycle, phase_bin) df.

    Per-trial scope so cosine similarity never crosses trial boundaries.
      cos_sim: within-trial cos-sim of place-tuned population rate vectors
        between cycle c and c+1 (averaged over place_trough_bins), stored on
        cycle c. NaN at trial-final cycles, when either cycle lost all trough
        bins, or when either vector is all-zero.
      cycle_complete: True if the cycle still has all n_bins phase bins.
      trough_bins_complete: True if every place_trough_bin survived row filters.

    Raises ValueError if cycle_idx values within a trial are non-consecutive.
    """
    n_bins = int(df.index.get_level_values("phase_bin").max()) + 1
    update = pd.Series(np.nan, index=df.index, dtype=float)
    complete_flag = pd.Series(False, index=df.index, dtype=bool)
    trough_complete_flag = pd.Series(False, index=df.index, dtype=bool)

    for trial, trial_df in df.groupby(df.trial):
        if trial_df.empty or not place_cells:
            continue

        # all-12-bins completeness tracking
        cycle_sizes = trial_df.groupby(level="cycle_idx").size()
        complete_cycles = set(cycle_sizes[cycle_sizes == n_bins].index)
        if complete_cycles:
            mask = trial_df.index.get_level_values("cycle_idx").isin(complete_cycles)
            complete_flag.loc[trial_df.index[mask]] = True

        # cycle_idx must be strictly consecutive within the trial (no skips)
        trial_cycles = np.sort(trial_df.index.get_level_values("cycle_idx").unique())
        if len(trial_cycles) > 1 and not np.all(np.diff(trial_cycles) == 1):
            raise ValueError(
                f"trial {trial}: cycle_idx values are not consecutive "
                f"(diffs found: {sorted(set(np.diff(trial_cycles).tolist()))}, expected all == 1)."
            )

        # per-cycle rates over place_trough_bins (no completeness filter)
        trough = trial_df[trial_df.index.get_level_values("phase_bin").isin(place_trough_bins)]
        spikes_by_cycle = trough.spike_count[place_cells].groupby(level="cycle_idx").sum()
        dur_by_cycle = trough.phase_window.duration.groupby(level="cycle_idx").sum()
        rates_by_cycle = spikes_by_cycle.div(dur_by_cycle, axis=0)

        # trough-bin completeness flag (True when every place_trough_bin survived)
        trough_bin_counts = trough.groupby(level="cycle_idx").size()
        trough_complete_cycles = set(trough_bin_counts[trough_bin_counts == len(place_trough_bins)].index)
        if trough_complete_cycles:
            mask = trial_df.index.get_level_values("cycle_idx").isin(trough_complete_cycles)
            trough_complete_flag.loc[trial_df.index[mask]] = True

        # cosine for (c, c+1) pairs; cycles with no trough bins → cos_sim NaN
        present = set(rates_by_cycle.index)
        with np.errstate(invalid="ignore"):
            for c in trial_cycles[:-1]:  # last cycle of trial has no successor
                if c not in present or (c + 1) not in present:
                    continue  # at least one cycle has zero surviving trough bins
                val = 1.0 - cosine(rates_by_cycle.loc[c].values, rates_by_cycle.loc[c + 1].values)
                in_cycle = trial_df.index.get_level_values("cycle_idx") == c
                update.loc[trial_df.index[in_cycle]] = val

    df = df.copy()
    df[("place_update", "cos_sim")] = update.values
    df[("place_update", "cycle_complete")] = complete_flag.values
    df[("place_update", "trough_bins_complete")] = trough_complete_flag.values
    return df


# %% Decoder feature builders


def _super_phase_rates(df, decoder_cells, n_training_phases):
    """Build the training feature matrix for one LOO fold.

    Collapses (cycle_idx, phase_bin) rows into (cycle_idx, super_phase) rows of
    firing rate per decoder cell, where super_phase = phase_bin //
    n_training_phases (e.g. n=3 → [0,1,2]→0, [3,4,5]→1, …). Only cycles with
    `place_update.cycle_complete == True` contribute. The training label is the
    most-common `distance_bin_id` in the super-phase. Returns (rates_df, bin_id).
    """
    df = df[df.place_update.cycle_complete]
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype="Int64")
    cycle_idx = df.index.get_level_values("cycle_idx")
    super_phase = pd.Index(df.index.get_level_values("phase_bin") // n_training_phases, name="super_phase")
    grouper = [cycle_idx, super_phase]
    spikes = df.spike_count[decoder_cells].groupby(grouper).sum()
    durations = df.phase_window.duration.groupby(grouper).sum()
    rates = spikes.div(durations, axis=0)

    def _most_common(s):
        vc = s.value_counts(dropna=True)
        return vc.index[0] if len(vc) > 0 else pd.NA

    bin_id = df.distance_bin_id.groupby(grouper).agg(_most_common)
    # drop super-phases with no usable label
    valid = bin_id.notna()
    return rates[valid], bin_id[valid].astype("int64")


def _cycle_subset_rates(df, decoder_cells, phase_bins):
    """Build the test feature matrix for one bin subset (peak or trough).

    One firing-rate row per cycle, averaged over `phase_bins`
    (rate = sum(spikes) / sum(durations)). Cycles missing any of the requested
    bins are dropped so peak and trough rate vectors stay comparable across
    cycles.
    """
    sub = df[df.index.get_level_values("phase_bin").isin(phase_bins)]
    counts = sub.groupby(level="cycle_idx").size()
    keep = counts[counts == len(phase_bins)].index
    sub = sub.loc[sub.index.get_level_values("cycle_idx").isin(keep)]
    spikes = sub.spike_count[decoder_cells].groupby(level="cycle_idx").sum()
    durations = sub.phase_window.duration.groupby(level="cycle_idx").sum()
    rates = spikes.div(durations, axis=0)
    return rates


# %% Cycle / phase-bin window detection


def _detect_cycle_phase_windows(bin_indices, n_bins):
    """Detect cycle and bin boundaries in the LFP phase signal.

    A cycle = LFP samples between consecutive `bin_idx (n_bins-1) → 0` wraps.
    Cycles that lose any phase bin (rare, noisy-theta only) are dropped.
    Returns (start_samples, end_samples), each shape (n_cycles, n_bins).
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
    """Per-cycle quality metrics for the cycle_metrics block: peak-to-peak
    amplitude on the band-passed signal, cycle period (s), and mean raw-LFP
    power. Each is broadcast across the n_bins rows of its cycle."""
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
    """Nearest-video-frame lookup: pick one navigation_df row per phase-bin
    midpoint time. Adjacent (cycle, bin) rows often land on the same frame, so
    the returned block has expected within-cycle duplication."""
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
    """Default population used by `place_update.cos_sim`: cluster_unique_IDs
    flagged as selectively place-direction-tuned by the neGLM
    variance-explained analysis (multi-unit model set, r² > 0.05)."""
    feature_tuned_df = ve.get_feature_tuned_df(
        lms.load_model_set_cv_scores("variance_explained_multiunit"),
        r2_thres=0.05,
    )
    return feature_tuned_df[feature_tuned_df.place_direction].index.get_level_values(-1).values


# %% Prelim correlation: place_update.cos_sim vs decoder.distance_error


def get_session_corr(
    session_df,
    use_abs=True,
    min_amplitude=None,
    max_amplitude=None,
    regress_out=None,
):
    """Stage 3 — Pearson correlation between `place_update.cos_sim` and
    `decoder.distance_error` on one `get_session_df` output.

    Returns dict with `n`, `r`, `p`, `regressed_out`.

    Non-obvious args:
      use_abs: correlate cos_sim against |distance_error| (magnitude of the
        within-cycle distance shift); set True to test the "big distance update
        → big place update" hypothesis (expected r < 0).
      min_amplitude / max_amplitude: optional bounds on `cycle_metrics.amplitude`
        (peak-to-peak uV) — drop low- or high-amp cycles before correlating.
      regress_out: list from {'speed', 'distance_to_goal', 'head_direction'};
        each is partialled out of both vars via OLS (separate fits, intercept
        included) and the correlation is taken on residuals. `head_direction`
        (degrees) is sin/cos-expanded so wrap-around at 0/360 is handled.
    """

    df = session_df

    if min_amplitude is not None:
        df = df[df[("cycle_metrics", "amplitude")] >= min_amplitude]
    if max_amplitude is not None:
        df = df[df[("cycle_metrics", "amplitude")] <= max_amplitude]

    cs = df[("place_update", "cos_sim")].values.astype(float)
    de = df[("decoder", "distance_error")].values.astype(float)
    if use_abs:
        de = np.abs(de)

    valid = ~(np.isnan(cs) | np.isnan(de))

    if regress_out:
        X = _build_regressors(df, regress_out)
        valid &= ~np.isnan(X).any(axis=1)
        cs, de, X = cs[valid], de[valid], X[valid]
        cs_resid = _ols_residuals(cs, X)
        de_resid = _ols_residuals(de, X)
        r, p = pearsonr(cs_resid, de_resid)
    else:
        cs, de = cs[valid], de[valid]
        r, p = pearsonr(cs, de)

    return {
        "n": int(valid.sum()),
        "r": float(r),
        "p": float(p),
        "regressed_out": list(regress_out) if regress_out else [],
    }


def _build_regressors(df, names):
    """Stack named covariate columns into a (n_rows, n_regressors) array;
    `head_direction` (degrees) is sin/cos-expanded for circular handling."""
    cols = []
    for name in names:
        if name == "speed":
            cols.append(df[("speed", "")].values.astype(float))
        elif name == "distance_to_goal":
            cols.append(df[("distance_to_goal", "geodesic")].values.astype(float))
        elif name == "head_direction":
            hd_rad = np.deg2rad(df[("head_direction", "value")].values.astype(float))
            cols.append(np.sin(hd_rad))
            cols.append(np.cos(hd_rad))
        else:
            raise ValueError(
                f"Unknown regressor {name!r}. Supported: " "'speed', 'distance_to_goal', 'head_direction'."
            )
    return np.column_stack(cols)


def _ols_residuals(y, X):
    """OLS residuals of y regressed on [1, X] (intercept-included least squares)."""
    Xi = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xi, y, rcond=None)
    return y - Xi @ beta
