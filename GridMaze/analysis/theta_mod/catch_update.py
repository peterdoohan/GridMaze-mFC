"""
Cycle-resolved theta-modulation pipeline (catch-update analysis).

Public entry points (in pipeline order):

  1. `get_input_data(session)` — builds the per-(cycle, phase_bin) dataframe:
        LFP-derived cycle/bin windows → spike counts → per-cycle metrics →
        navigation alignment (nearest video frame) → row filters
        (navigation_only / max_steps_to_goal / exclude_at_goal) →
        distance-to-goal binning → place-tuned population update score.

  2a. `get_session_sweep_update_df(session)` — wraps `get_input_data` in a
        leave-one-trial-out loop: each fold trains a distance-to-goal
        logistic-regression decoder on super-phase-averaged firing rates of
        the training trials, then applies it twice per test cycle (at
        `distance_peak_bins` and `distance_trough_bins`) to produce a per-cycle
        `distance_error = trough_pred - peak_pred`. Returns one row per test
        cycle.

  2b. `get_sweep_update_df(save=False)` — runs 2a across every subject × maze
        × day in parallel and concatenates the per-session results into one
        cross-session dataframe; cached to parquet under `RESULTS_PATH/theta_mod/`.

  3. `plot_update_corr(experiment_df)` — one Pearson r per subject between
        `place_update.cos_sim` and `decoder.distance_error` (cycles pooled
        across that subject's sessions), with optional amplitude / session-MAE
        / neuron-count / maze filters and within-subject partialling of
        speed / distance-to-goal / head-direction. Plots individual subject
        rs as grey dots + a cross-subject mean ± SEM marker (style mirrors
        `decoding_error_corr.plot_decoding_error_corr`) and runs a one-sided
        cross-subject t-test against 0 (default alt='less' — the "neg corr
        across subjects" hypothesis).

Higher temporal resolution alternative to `session.navigation_theta_spike_counts_df`:
rows are (cycle_idx, phase_bin), each spanning the actual duration of one phase
bin within one cycle. Phase bin definitions match `bin_lfp_phase` (default 12) so
they line up with the modulation-profile bins used in `decoding_offsets.py`.

@peterdoohan
"""

# %% Imports
import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, ttest_1samp

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from GridMaze.analysis.core import convert
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.lfp import lfp_utils as lu
from GridMaze.analysis.processing import get_lfp_aligned_spike_counts as la

from GridMaze.analysis.neGLM import variance_explained_null as ve
from GridMaze.analysis.neGLM import load_model_sets as lms

from GridMaze.analysis.place_direction import future_decoding as fd
from GridMaze.maze import representations as mr

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

# %% Global variables
THETA_RANGE = (7, 10)
FS_LFP = 1500

RESULTS_DIR = RESULTS_PATH / "theta_mod"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json") as _f:
    SUBJECT_IDS = json.load(_f)

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]

# %% Main entry point


def get_sweep_update_df(save=False, verbose=True, n_jobs=6, tag=None, **session_kwargs):
    """Stage 2 (population) — `get_session_sweep_update_df` across all
    subjects × mazes × days, concatenated and cached to parquet.

    Mirrors the pattern in `distance_to_goal_decoder.get_theta_mod_distance_error_df`:
    sessions are processed in parallel with joblib; per-session failures are
    swallowed (printed if `verbose`) so one bad session doesn't kill the run.

    Args:
      save: True to (re)run and overwrite the cache; False to load cached parquet
        if present.
      verbose: log session names as they're processed.
      n_jobs: joblib `Parallel` n_jobs (default 6). Pass 1 for
        sequential / debugging.
      tag: optional string suffix for the cache filename. None → canonical run
        at `RESULTS_DIR/sweep_update_df.parquet`. Set (e.g. tag="_C1e-2") to
        save hyperparameter-sweep runs under
        `RESULTS_DIR/sweep_update_tests/sweep_update_df{tag}.parquet`, so they
        don't overwrite the canonical cache.
      **session_kwargs: forwarded to `get_session_sweep_update_df` (decoder
        hyperparameters or `input_data_kwargs`).

    Returns:
      DataFrame: one row per test cycle across all sessions. Row index is a
      fresh RangeIndex; original `cycle_idx` is moved to a column. A
      `late_session` boolean column is added.
    """
    if tag is None:
        save_path = RESULTS_DIR / "sweep_update_df.parquet"
    else:
        save_path = RESULTS_DIR / "sweep_update_tests" / f"sweep_update_df{tag}.parquet"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists() and not save:
        if verbose:
            print(f"loading cached: {save_path}")
        return pd.read_parquet(save_path)

    def _process(session):
        if verbose:
            print(session.name)
        try:
            df = get_session_sweep_update_df(session, verbose=verbose, **session_kwargs)
            return df
        except Exception as e:
            if verbose:
                print(f"  error on {session.name}: {e}")
            return None

    all_dfs = []
    for subject in SUBJECT_IDS:
        for maze_name in MAZE_NAMES:
            sessions = gs.get_maze_sessions(
                subject_IDs=[subject],
                maze_names=[maze_name],
                days_on_maze="late",
                with_data=[
                    "lfp_times",
                    "lfp_signal",
                    "lfp_metrics",
                    "cluster_metrics",
                    "navigation_df",
                    "spike_times",
                    "spike_clusters",
                ],
                must_have_data=True,
            )
            session_dfs = Parallel(n_jobs=n_jobs)(delayed(_process)(s) for s in sessions)
            all_dfs.extend([d for d in session_dfs if d is not None])

    pop_df = pd.concat(all_dfs).reset_index()  # promote cycle_idx to column

    if save:
        pop_df.to_parquet(save_path)
        if verbose:
            print(f"saved: {save_path}")
    return pop_df


def get_session_sweep_update_df(
    session,
    distance_peak_bins=[3, 4, 5],
    distance_trough_bins=[9, 10, 11],
    n_training_phases=3,
    normalise_X=True,
    C=1e-1,
    output="weighted",
    place_tuned_cells=None,
    exclude_place_cells_from_decoder=False,
    verbose=False,
    **input_data_kwargs,
):
    """Stage 2 (per session) — LOO distance-to-goal decoder with peak/trough theta-phase readout.

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
      C, normalise_X, output: LogisticRegression hyperparameters. `C=None` →
        `penalty=None` (unregularised LR; C is ignored). `output` is either
        "weighted" (prob-weighted bin-mid readout) or "max" (argmax).
      place_tuned_cells: resolved once here and forwarded to `get_input_data`
        (so it's not re-loaded inside) and also used for the optional decoder
        exclusion below. None → load via `get_place_direction_tuned_neurons()`.
      exclude_place_cells_from_decoder: if True, drop `place_tuned_cells` from
        the decoder feature set so the two correlation axes
        (`place_update.cos_sim`, `decoder.distance_error`) are built from
        disjoint neural populations. Default False preserves existing behaviour.
    """
    if place_tuned_cells is None:
        place_tuned_cells = get_place_direction_tuned_neurons()

    if verbose:
        print("Loading input data...")
    input_data = get_input_data(session, place_tuned_cells=place_tuned_cells, **input_data_kwargs)
    cluster_ids = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=True,
    )
    decoder_cells = [c for c in input_data.spike_count.columns if c in set(cluster_ids)]

    if exclude_place_cells_from_decoder:
        place_set = set(np.unique(place_tuned_cells))
        n_before = len(decoder_cells)
        decoder_cells = [c for c in decoder_cells if c not in place_set]
        if verbose:
            print(
                f"excluded {n_before - len(decoder_cells)} place-tuned cells; "
                f"decoder uses {len(decoder_cells)} cells"
            )
    distance_bin_mids = np.array(sorted(input_data.distance_bin_mid.dropna().unique()))
    n_bins = int(input_data.index.get_level_values("phase_bin").max()) + 1
    assert n_bins % n_training_phases == 0, "n_training_phases must divide n_bins"
    if output not in ("weighted", "max"):
        raise ValueError(f"output must be 'weighted' or 'max', got {output!r}")

    per_cycle_results = []
    for held_out_trial in input_data.trial.dropna().unique():
        if verbose:
            print(f"  held-out trial: {held_out_trial}")

        train_df, test_df = _split_loo_fold(input_data, held_out_trial)

        # --- training data: firing rates over super-phases, complete cycles only ---
        rates_train, y_train = _super_phase_rates(train_df, decoder_cells, n_training_phases)
        if rates_train.empty:
            continue
        X_train = rates_train.values
        scaler = StandardScaler().fit(X_train) if normalise_X else None
        if scaler is not None:
            X_train = scaler.transform(X_train)
        lr_kwargs = {"random_state": 0, "max_iter": 2_000, "class_weight": "balanced"}
        if C is None:
            lr_kwargs["penalty"] = None
        else:
            lr_kwargs["C"] = C
        decoder = LogisticRegression(**lr_kwargs).fit(X_train, y_train.values)
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

    session_df = pd.concat(per_cycle_results).sort_index()

    # --- session-level MAE between decoded distance and the cycle's true bin midpoint ---
    true_dist = session_df[("distance_bin_mid", "")].values.astype(float)
    peak_mae = float(np.mean(np.abs(session_df[("decoder", "peak_pred")].values - true_dist)))
    trough_mae = float(np.mean(np.abs(session_df[("decoder", "trough_pred")].values - true_dist)))
    session_df[("decoder", "peak_mae")] = peak_mae
    session_df[("decoder", "trough_mae")] = trough_mae
    # session-constant count of cells the decoder actually used (after
    # single/multi filter and optional place-cell exclusion).
    session_df[("decoder", "n_decoder_neurons")] = len(decoder_cells)
    if verbose:
        print(f"{session.name}: decoder MAE — peak={peak_mae:.4f} m, trough={trough_mae:.4f} m")
    return session_df


def get_input_data(
    session,
    n_bins=12,
    shank=3,
    max_steps_to_goal=20,
    exclude_at_goal=True,
    max_distance=None,
    min_distance=None,
    bin_spacing=0.05,
    place_trough_bins=[0, 1, 2],
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
      n_place_neurons: int, constant across the session — # of place-tuned cells
        actually present in this session (i.e. `len(place_cells)`). (The
        complementary `decoder.n_decoder_neurons` is added by
        `get_session_sweep_update_df`.)

    Trials whose cycle_idx values are non-consecutive (a raw LFP cycle was
    dropped mid-trial) are **excluded** from the returned df with a warning
    print, rather than raising. This keeps the rest of the session usable.
    """
    n_bins = int(df.index.get_level_values("phase_bin").max()) + 1

    # drop trials with non-consecutive cycle_idx (rare: implies a dropped LFP cycle)
    bad_trials = []
    for trial, trial_df in df.groupby(df.trial):
        trial_cycles = np.sort(trial_df.index.get_level_values("cycle_idx").unique())
        if len(trial_cycles) > 1 and not np.all(np.diff(trial_cycles) == 1):
            bad_trials.append(trial)
    if bad_trials:
        print(
            f"_add_place_update_column: dropping {len(bad_trials)} trial(s) with "
            f"non-consecutive cycle_idx: {bad_trials}"
        )
        df = df[~df.trial.isin(bad_trials)]

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

        trial_cycles = np.sort(trial_df.index.get_level_values("cycle_idx").unique())

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
        trial_cycle_idxs = trial_df.index.get_level_values("cycle_idx").values
        with np.errstate(invalid="ignore"):
            for c in trial_cycles[:-1]:  # last cycle of trial has no successor
                if c not in present or (c + 1) not in present:
                    continue  # at least one cycle has zero surviving trough bins
                val = 1.0 - cosine(rates_by_cycle.loc[c].values, rates_by_cycle.loc[c + 1].values)
                update.loc[trial_df.index[trial_cycle_idxs == c]] = val

    df = df.copy()
    df[("place_update", "cos_sim")] = update.values
    df[("place_update", "cycle_complete")] = complete_flag.values
    df[("place_update", "trough_bins_complete")] = trough_complete_flag.values
    # session-constant count for downstream session filtering; the matching
    # decoder-population count is added by get_session_sweep_update_df.
    df[("place_update", "n_place_neurons")] = len(place_cells)
    return df


# %% Decoder feature builders


def _split_loo_fold(input_data, held_out_trial):
    """Slice `input_data` into (train_df, test_df) for one LOO fold. `test_df`
    is further restricted to complete cycles so the bin-0 lookup for cycle_meta
    is safe."""
    test_mask = input_data.trial == held_out_trial
    train_df = input_data[~test_mask]
    test_df = input_data[test_mask & input_data.place_update.cycle_complete]
    return train_df, test_df


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

    Wrap-aware: if `phase_bins` is ordered in ascending phase but contains a
    descending step (e.g. `[10, 11, 0]`), bins *after* the step are sampled
    from the SUCCESSOR cycle (i+1) rather than the current cycle (i). This lets
    the caller select a contiguous phase window that straddles the cycle
    boundary (here, the late-cycle bins 10, 11 plus the next cycle's bin 0).
    Cycles whose successor isn't in `df` (last cycle of a trial) are dropped
    from the returned rates.
    """
    bins = list(phase_bins)

    # Detect wrap: first descending step in the bin sequence.
    wrap_idx = None
    for i in range(len(bins) - 1):
        if bins[i + 1] < bins[i]:
            wrap_idx = i + 1
            break

    if wrap_idx is None:
        # Non-wrapping case — original semantics.
        sub = df[df.index.get_level_values("phase_bin").isin(bins)]
        counts = sub.groupby(level="cycle_idx").size()
        keep = counts[counts == len(bins)].index
        sub = sub.loc[sub.index.get_level_values("cycle_idx").isin(keep)]
        spikes = sub.spike_count[decoder_cells].groupby(level="cycle_idx").sum()
        durations = sub.phase_window.duration.groupby(level="cycle_idx").sum()
        return spikes.div(durations, axis=0)

    # Wrap-aware path: pre-wrap bins → cycle i, post-wrap bins → cycle i+1.
    pre_bins, post_bins = bins[:wrap_idx], bins[wrap_idx:]
    pre = df[df.index.get_level_values("phase_bin").isin(pre_bins)]
    post = df[df.index.get_level_values("phase_bin").isin(post_bins)]
    if not post.empty:
        # Attribute each post-wrap row to its predecessor cycle (i.e. shift -1).
        post = post.set_index(
            pd.MultiIndex.from_arrays(
                [
                    post.index.get_level_values("cycle_idx") - 1,
                    post.index.get_level_values("phase_bin"),
                ],
                names=["cycle_idx", "phase_bin"],
            )
        )
    combined = pd.concat([pre, post])
    counts = combined.groupby(level="cycle_idx").size()
    keep = counts[counts == len(bins)].index
    combined = combined.loc[combined.index.get_level_values("cycle_idx").isin(keep)]
    spikes = combined.spike_count[decoder_cells].groupby(level="cycle_idx").sum()
    durations = combined.phase_window.duration.groupby(level="cycle_idx").sum()
    return spikes.div(durations, axis=0)


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


# %% Stage 3 — experiment-level per-subject correlation


def plot_update_corr(
    experiment_df,
    use_abs=True,
    min_amplitude=None,
    max_amplitude=None,
    regress_out=None,
    maze_names=None,
    max_peak_mae=None,
    max_trough_mae=None,
    min_n_place_neurons=5,
    min_n_decoder_neurons=None,
    max_distance=None,
    decision_points=False,
    alternative="less",
    color="blue",
    ax=None,
    print_stats=True,
):
    """Stage 3 — one Pearson r per subject (cycles pooled across that
    subject's sessions), plotted as individual grey dots with a mean ± SEM
    pointplot, and a one-sided cross-subject t-test (default alt='less').

    Style mirrors `decoding_error_corr.plot_decoding_error_corr`.

    Args:
      use_abs: correlate cos_sim against |distance_error| (default — tests the
        "big |distance update| ↔ small cos_sim" hypothesis, expected r < 0).
      min_amplitude / max_amplitude: cycle-level bounds on `cycle_metrics.amplitude`.
      maze_names: keep only these mazes.
      max_peak_mae / max_trough_mae: drop sessions whose decoder MAE exceeds.
      min_n_place_neurons / min_n_decoder_neurons: drop sessions with too few cells.
      max_distance: drop cycles whose `distance_to_goal.geodesic` (m) exceeds
        this — cycle-level filter, complementary to the `max_steps_to_goal`
        sweep applied at parquet-build time.
      decision_points: False | "future" | "past" — restrict cycles to those whose
        `(maze_position.simple, cardinal_movement_direction)` is a decision point
        (mirrors `place_direction_decoding._filter_decision_points`). "future"
        uses edges_only=True, "past" uses node_only=True. Drops rooms_maze (no
        decision points defined for it).
      regress_out: subset of {'speed', 'distance_to_goal', 'head_direction'}
        partialled out within-subject before correlating; head_direction is
        sin/cos-expanded.
      alternative: ttest_1samp alternative ("less" tests r < 0).

    Returns (ax, {per_subject: DataFrame[n,r,p], cross_subject: dict}).
    """
    df = experiment_df.copy()

    if maze_names is not None:
        df = df[df[("maze_name", "")].isin(maze_names)]
    if min_amplitude is not None:
        df = df[df[("cycle_metrics", "amplitude")] >= min_amplitude]
    if max_amplitude is not None:
        df = df[df[("cycle_metrics", "amplitude")] <= max_amplitude]
    if max_peak_mae is not None:
        df = df[df[("decoder", "peak_mae")] <= max_peak_mae]
    if max_trough_mae is not None:
        df = df[df[("decoder", "trough_mae")] <= max_trough_mae]
    if min_n_place_neurons is not None:
        df = df[df[("place_update", "n_place_neurons")] >= min_n_place_neurons]
    if min_n_decoder_neurons is not None:
        df = df[df[("decoder", "n_decoder_neurons")] >= min_n_decoder_neurons]
    if max_distance is not None:
        df = df[df[("distance_to_goal", "geodesic")] <= max_distance]
    if decision_points:
        df = _filter_decision_points(df, decision_points=decision_points)

    per_subject_rows = []
    for subject_ID, sdf in df.groupby(df[("subject_ID", "")]):
        cs_vals = sdf[("place_update", "cos_sim")].values.astype(float)
        de_vals = sdf[("decoder", "distance_error")].values.astype(float)
        if use_abs:
            de_vals = np.abs(de_vals)

        valid = ~(np.isnan(cs_vals) | np.isnan(de_vals))

        if regress_out:
            X = _build_regressors(sdf, regress_out)
            valid &= ~np.isnan(X).any(axis=1)
            cs_vals, de_vals, X = cs_vals[valid], de_vals[valid], X[valid]
            if len(cs_vals) < 2:
                r, p = np.nan, np.nan
            else:
                cs_resid = _ols_residuals(cs_vals, X)
                de_resid = _ols_residuals(de_vals, X)
                r, p = pearsonr(cs_resid, de_resid)
        else:
            cs_vals, de_vals = cs_vals[valid], de_vals[valid]
            if len(cs_vals) < 2:
                r, p = np.nan, np.nan
            else:
                r, p = pearsonr(cs_vals, de_vals)

        per_subject_rows.append({"subject_ID": subject_ID, "n": int(valid.sum()), "r": float(r), "p": float(p)})

    if per_subject_rows:
        per_subject = pd.DataFrame.from_records(per_subject_rows).set_index("subject_ID")
    else:
        per_subject = pd.DataFrame(columns=["n", "r", "p"]).rename_axis("subject_ID")

    rs = per_subject["r"].dropna().values
    if len(rs) >= 2:
        t_stat, p_t = ttest_1samp(rs, 0, alternative=alternative)
        sem_r = float(rs.std(ddof=1) / np.sqrt(len(rs)))
    else:
        t_stat, p_t, sem_r = np.nan, np.nan, np.nan
    cross_subject = {
        "n_subjects": int(len(rs)),
        "mean_r": float(rs.mean()) if len(rs) else np.nan,
        "sem_r": sem_r,
        "t": float(t_stat) if not np.isnan(t_stat) else np.nan,
        "p": float(p_t) if not np.isnan(p_t) else np.nan,
        "alternative": alternative,
    }

    if print_stats:
        de_label = "|distance_error|" if use_abs else "distance_error"
        print(f"per-subject corr  cos_sim ↔ {de_label}" + (f"  (regressed out: {regress_out})" if regress_out else ""))
        for sub, row in per_subject.iterrows():
            print(f"  {sub}: n={int(row['n']):>6d}  r={row['r']:+.4f}  p={row['p']:.3g}")
        print(
            f"cross-subject ttest_1samp (alt={alternative!r}): "
            f"mean r={cross_subject['mean_r']:+.4f} ± {cross_subject['sem_r']:.4f}  "
            f"t={cross_subject['t']:+.3f}  p={cross_subject['p']:.4g}  (n={cross_subject['n_subjects']})"
        )

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(0.8, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", ls="--", alpha=0.5)
    # subject dots: jittered around x=-0.2 so they never overlap the mean marker
    dot_x = -0.2 + np.linspace(-0.07, 0.07, max(len(rs), 1))
    ax.scatter(
        dot_x,
        rs,
        color="grey",
        alpha=0.7,
        s=30,
        edgecolors="none",
        zorder=3,
    )
    # cross-subject mean ± SEM at x=+0.2 via seaborn pointplot (native_scale
    # keeps x numeric so it co-exists with the scatter on the same axis);
    # wide error bar, no caps.
    if len(rs) >= 2:
        sns.pointplot(
            x=np.full(len(rs), 0.2),
            y=rs,
            errorbar="se",
            color=color,
            marker="o",
            markersize=8,
            linestyle="none",
            err_kws={"linewidth": 2.5},
            capsize=0,
            native_scale=True,
            ax=ax,
            zorder=2,
        )
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_xlabel("")
    ylabel = "correlation\n(cos_sim ↔ |dist err|)" if use_abs else "correlation\n(cos_sim ↔ dist err)"
    ax.set_ylabel(ylabel)

    return ax, {"per_subject": per_subject, "cross_subject": cross_subject}


def plot_update_corr_sweep_grid(
    indep=False,
    C_values=("none", "1", "10", "100"),
    msg_values=(8, 10, 12),
    sweep_dir=None,
    figsize_per_cell=(0.9, 1.8),
    sharey=True,
    **plot_kwargs,
):
    """Quick C × msg grid of `plot_update_corr` panels for the C × msg × indep
    sweep parquets in `RESULTS_DIR/sweep_update_tests/`.

    Args:
      indep: select the `_indep` (place cells excluded from decoder) variant.
      C_values: rows. Strings matching the tag (e.g. "none", "1e-1", "1", "10",
        "100"). Recognised so `_C{val}_msg{m}[_indep]` resolves to a real parquet.
      msg_values: columns (max_steps_to_goal ints, e.g. (8, 10, 12)).
      sweep_dir: parquet directory. Defaults to
        `RESULTS_DIR/sweep_update_tests/`.
      figsize_per_cell: (w, h) per axis in inches.
      sharey: share y-axis limits across all cells so corr magnitudes are
        directly comparable.
      **plot_kwargs: forwarded to `plot_update_corr` for every cell. Use this
        to flip any of its parameters across the whole grid — e.g.
        `use_abs=False`, `regress_out=['distance_to_goal','speed']`,
        `maze_names=['maze_1']`, `min_n_place_neurons=20`,
        `max_distance=0.5`, `decision_points='future'`.

    Returns (fig, ax_grid, results) where `results[(C, msg)]` is the dict
    returned by `plot_update_corr` for that cell (or None if the parquet was
    missing).
    """
    if sweep_dir is None:
        sweep_dir = RESULTS_DIR / "sweep_update_tests"
    sweep_dir = Path(sweep_dir)

    n_rows, n_cols = len(C_values), len(msg_values)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        sharey=sharey,
        squeeze=False,
    )

    suffix = "_indep" if indep else ""
    results = {}
    for i, C in enumerate(C_values):
        for j, msg in enumerate(msg_values):
            ax = axes[i, j]
            tag = f"_C{C}_msg{msg}{suffix}"
            path = sweep_dir / f"sweep_update_df{tag}.parquet"
            df = None
            placeholder_label = None
            if not path.exists():
                placeholder_label = "no data"
            else:
                try:
                    df = pd.read_parquet(path)
                except Exception as e:
                    # parquet exists but unreadable — usually a write still in
                    # flight (pyarrow writes in-place). Treat as missing.
                    placeholder_label = f"read err\n({type(e).__name__})"
            if df is None:
                ax.text(
                    0.5,
                    0.5,
                    placeholder_label,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="grey",
                )
                # match data-cell xlim so the placeholder doesn't autoscale
                # away from the rest of the grid. don't touch yticks — with
                # sharey=True, clearing yticks here strips them everywhere.
                ax.set_xlim(-0.5, 0.5)
                ax.set_xticks([])
                for s in ax.spines.values():
                    s.set_alpha(0.3)
                results[(C, msg)] = None
            else:
                _, res = plot_update_corr(df, ax=ax, print_stats=False, **plot_kwargs)
                cs = res["cross_subject"]
                title = f"t({cs['n_subjects']})={cs['t']:+.2f},\n p={cs['p']:.3g}"
                ax.set_title(title, fontsize=7)
                results[(C, msg)] = res

            # clear per-cell ylabels (the row label goes outside, below)
            ax.set_ylabel("")
            # column labels (msg=…) above the top row — pushed further up so
            # they clear the per-cell `t(n)=…, p=…` title
            if i == 0:
                ax.annotate(
                    f"msg={msg}",
                    xy=(0.5, 1.45),
                    xycoords="axes fraction",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                )
            # row labels (C=…) to the left of the first column — pushed further
            # left so they clear the ytick labels
            if j == 0:
                ax.annotate(
                    f"C={C}",
                    xy=(-1.0, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    rotation=90,
                )

    fig.tight_layout()
    return fig, axes, results


def _filter_decision_points(df, decision_points="future"):
    """Restrict the experiment df to cycles at decision points (mirrors
    `place_direction_decoding._filter_decision_points`).

    A decision point is the `(maze_position.simple, cardinal_movement_direction)`
    pair from which a single step lands at (future) — or originates from (past) —
    a node with ≥3 neighbours. Per-maze decision-point sets come from
    `future_decoding.get_decision_points`. rooms_maze is dropped (decision
    points are only defined for maze_1 / maze_2).
    """
    kept = []
    for maze_name in ["maze_1", "maze_2"]:
        maze_df = df[df[("maze_name", "")] == maze_name]
        if maze_df.empty:
            continue
        simple_maze = mr.get_simple_maze(maze_name)
        if decision_points == "future":
            dp_set = fd.get_decision_points(
                simple_maze, mode="future", edges_only=True, node_only=False, return_as="strings", plot=False
            )
        elif decision_points == "past":
            dp_set = fd.get_decision_points(
                simple_maze, mode="past", edges_only=False, node_only=True, return_as="strings", plot=False
            )
        else:
            raise ValueError(f"decision_points must be False, 'future', or 'past'. Got {decision_points!r}.")
        pd_strings = (
            maze_df[("maze_position", "simple")].astype(str)
            + "_"
            + maze_df[("cardinal_movement_direction", "")].astype(str)
        )
        kept.append(maze_df[pd_strings.isin(dp_set)])
    if not kept:
        return df.iloc[0:0]
    return pd.concat(kept, axis=0)


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


# %%
