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
        cross-subject t-test against 0 (default alt='two-sided'; switch to
        'less' for the directional "neg corr across subjects" hypothesis).

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

from GridMaze.analysis.neGLM import load_model_sets as lms
from GridMaze.analysis.neGLM import variance_explained_null as ve

from GridMaze.maze import representations as mr

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

# %% Global variables
THETA_RANGE = (7, 11)

RESULTS_DIR = RESULTS_PATH / "theta_mod"

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
      fresh RangeIndex; original `cycle_idx` (single-level after the bin-0
      collapse in `get_session_sweep_update_df`) is promoted to a column.
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

    # Load the CPD score Series once; forward to every per-session call so the
    # model-set parquet isn't reloaded inside each joblib worker.
    cpd_scores = get_population_distance_score()

    def _process(session):
        if verbose:
            print(session.name)
        try:
            df = get_session_sweep_update_df(session, cpd_scores=cpd_scores, verbose=verbose, **session_kwargs)
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

    # Per-session dfs are indexed by cycle_idx alone (phase_bin was collapsed
    # via test_df.xs(0, level="phase_bin") inside get_session_sweep_update_df),
    # so reset_index here promotes that single cycle_idx level to a column.
    pop_df = pd.concat(all_dfs).reset_index()

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
    C=1,
    output="weighted",
    n_folds=10,
    cv_seed=0,
    cpd_scores=None,
    verbose=False,
    **input_data_kwargs,
):
    """Stage 2 (per session) — distance-to-goal decoder with peak/trough theta-phase readout.

    For each CV fold: train a logistic-regression distance decoder on
    super-phase-averaged firing rates from the training trials, then apply it to
    each test cycle at `distance_peak_bins` and `distance_trough_bins`. Returns
    one row per test cycle with `decoder.peak_pred`, `decoder.trough_pred`,
    `decoder.distance_error = trough_pred - peak_pred`, plus the session-level
    MAE constants and all the nav / `place_update` / `cycle_metrics` columns
    that `get_input_data` produced. `spike_count` columns are dropped.

    The decoder feature set is whatever `get_input_data` left in
    `spike_count.columns` (i.e. the distance pool from the CPD-driven split);
    no further filtering happens here.

    Non-obvious args:
      distance_peak_bins / distance_trough_bins: theta phase bins (early / late
        in cycle by default) whose average rates form the test-time peak and
        trough rate vectors.
      n_training_phases: bins per super-phase group used for training (must
        divide n_bins evenly; default 3 → 4 super-bins per cycle).
      C, normalise_X, output: LogisticRegression hyperparameters. `C=None` →
        `penalty=None` (unregularised LR; C is ignored). `output` is either
        "weighted" (prob-weighted bin-mid readout) or "max" (argmax).
      n_folds: -1 → leave-one-trial-out CV (one fold per trial). Any positive
        int → that many folds, trials randomly partitioned with `cv_seed`.
      cv_seed: RNG seed for the trial shuffle when `n_folds > 0`. Ignored for
        LOO. Default 0 → deterministic.
      cpd_scores: optional precomputed Series from `get_population_distance_score`;
        forwarded to `get_input_data` so the sweep can load it once.
    """
    if verbose:
        print("Loading input data...")
    input_data = get_input_data(session, cpd_scores=cpd_scores, verbose=verbose, **input_data_kwargs)
    decoder_cells = list(input_data.spike_count.columns)
    distance_bin_mids = np.array(sorted(input_data.distance_bin_mid.dropna().unique()))
    n_bins = int(input_data.index.get_level_values("phase_bin").max()) + 1
    if n_bins % n_training_phases != 0:
        raise ValueError(f"n_training_phases ({n_training_phases}) must divide n_bins ({n_bins}) evenly")
    if output not in ("weighted", "max"):
        raise ValueError(f"output must be 'weighted' or 'max', got {output!r}")

    trials = input_data.trial.dropna().unique()
    if n_folds == -1:
        folds = [np.array([t]) for t in trials]
    elif n_folds > 0:
        if n_folds > len(trials):
            raise ValueError(f"n_folds={n_folds} exceeds number of trials ({len(trials)}) for {session.name}")
        rng = np.random.default_rng(cv_seed)
        folds = np.array_split(rng.permutation(trials), n_folds)
    else:
        raise ValueError(f"n_folds must be -1 (LOO) or a positive integer, got {n_folds!r}")

    per_cycle_results = []
    for fold_idx, held_out_trials in enumerate(folds):
        if verbose:
            print(f"  fold {fold_idx} ({len(held_out_trials)} trial(s) held out)")

        train_df, test_df = _split_fold(input_data, held_out_trials)

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
        cycle_meta[("decoder", "fold")] = fold_idx
        per_cycle_results.append(cycle_meta)

    if not per_cycle_results:
        raise ValueError(f"{session.name}: no usable test cycles in any fold")
    session_df = pd.concat(per_cycle_results).sort_index()

    # --- session-level MAE between decoded distance and the cycle's true bin midpoint ---
    true_dist = session_df[("distance_bin_mid", "")].values.astype(float)
    peak_mae = float(np.mean(np.abs(session_df[("decoder", "peak_pred")].values - true_dist)))
    trough_mae = float(np.mean(np.abs(session_df[("decoder", "trough_pred")].values - true_dist)))
    session_df[("decoder", "peak_mae")] = peak_mae
    session_df[("decoder", "trough_mae")] = trough_mae
    # session-constant count of decoder cells (= distance pool after the
    # CPD median split + > 0 filter in get_input_data).
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
    moving_only=True,
    min_firing_rate=0.5,
    cpd_scores=None,
    verbose=False,
):
    """Stage 1 — build the per-(cycle, phase_bin) input dataframe with CPD-driven
    independent neuron pools.

    MultiIndex rows = (cycle_idx, phase_bin); 2-level MultiIndex columns. Pipeline:
        cycle/bin windows from LFP → spike counts → per-cycle metrics →
        navigation aligned to bin midpoints → row filters (incl. moving-only) →
        drop any cycle missing a phase bin → neuron split (unit-filter →
        FR ≥ `min_firing_rate` over surviving rows → CPD lookup → within-session
        median split → distance pool further filtered to CPD > 0) →
        distance-to-goal binning → place-tuned population update score (on the
        place pool) → restrict `spike_count` columns to the distance pool.

    The distance pool drives the downstream LR decoder; the place pool drives
    `place_update.cos_sim`. The two pools are disjoint by construction. The
    returned df's `spike_count` columns hold ONLY the distance pool; the place
    pool is consumed for cos_sim and then discarded.

    Args:
      moving_only: drop rows where the animal is stationary (uses
        `navigation_df.moving`, built per-subject from MOVEMENT_THRESHOLD in
        `get_navigation_df.py`). Default True — theta is movement-locked, so
        stationary rows mostly contribute noise.
      min_firing_rate: per-cluster FR floor (Hz) computed over the surviving
        rows of this session (spike_count.sum() / phase_window.duration.sum()).
      cpd_scores: optional precomputed Series from `get_population_distance_score`
        (indexed by cluster_unique_ID). None → load here. Pass the precomputed
        Series in the sweep loop to avoid reloading the model-set parquet per
        session.
      verbose: print a one-line per-session neuron-count diagnostic.

    Cycles that lose any phase bin to row filtering (e.g. animal stopped mid-
    cycle, briefly left the on-task window) are dropped wholesale BEFORE the
    neuron split — so every surviving cycle has all `n_bins` rows by
    construction. Trials may end up with non-consecutive `cycle_idx` (gaps);
    `_add_place_update_column` handles these by writing NaN cos_sim at the
    cycle just before each gap.

    Output columns: every navigation group + `phase_window` (start/end/midpoint/
    duration) + `cycle_metrics` (amplitude, constant per cycle) +
    `spike_count` (distance-pool clusters only) +
    `distance_bin`/`distance_bin_mid`/`distance_bin_id` + `place_update`
    (cos_sim / n_place_neurons).
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
    amplitudes = _compute_cycle_metrics(filt_osc, start_samples, end_samples)

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
        {("cycle_metrics", "amplitude"): np.repeat(amplitudes, n_bins)},
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
    if moving_only:
        input_data = input_data[input_data.moving]
    if max_steps_to_goal is not None:
        input_data = input_data[input_data.steps_to_goal.future < max_steps_to_goal]
    if exclude_at_goal:
        input_data = input_data[input_data.goal != input_data.maze_position.simple]

    # --- drop any cycle that lost a phase bin to the filters above ---
    # Surviving cycles all have exactly n_bins rows; trials may have gaps in
    # cycle_idx (handled by _add_place_update_column → NaN cos_sim at gaps).
    cycle_sizes = input_data.groupby(level="cycle_idx").size()
    complete_cycles = cycle_sizes[cycle_sizes == n_bins].index
    input_data = input_data.loc[input_data.index.get_level_values("cycle_idx").isin(complete_cycles)]

    # --- neuron split (unit filter → FR floor → CPD lookup → median split → CPD>0) ---
    unit_pool = set(
        gc.filter_clusters(
            session.cluster_metrics,
            session.session_info,
            return_unique_IDs=True,
            single_units=True,
            multi_units=True,
        )
    )
    session_cells = list(input_data.spike_count.columns)
    after_unit = [c for c in session_cells if c in unit_pool]

    nav_duration = float(input_data.phase_window.duration.sum())
    if nav_duration <= 0:
        raise ValueError(
            f"{session.name}: no rows survived row filters (moving_only={moving_only}, "
            f"max_steps_to_goal={max_steps_to_goal}, exclude_at_goal={exclude_at_goal}) "
            "— cannot compute firing rates."
        )
    spikes_per_cell = input_data.spike_count[after_unit].sum(axis=0)
    fr_per_cell = spikes_per_cell / nav_duration
    after_fr = [c for c in after_unit if fr_per_cell.get(c, 0.0) >= min_firing_rate]

    if cpd_scores is None:
        cpd_scores = get_population_distance_score()
    with_cpd = [c for c in after_fr if c in cpd_scores.index]
    if not with_cpd:
        raise ValueError(
            f"{session.name}: no clusters survive unit/FR/CPD filtering "
            f"(after_unit={len(after_unit)}, after_fr={len(after_fr)})"
        )

    scores = cpd_scores.loc[with_cpd]
    median_score = float(np.median(scores.values))
    distance_pool = [c for c in with_cpd if scores.loc[c] > median_score]
    place_pool = [c for c in with_cpd if scores.loc[c] <= median_score]
    distance_pool = [c for c in distance_pool if scores.loc[c] > 0]

    if verbose:
        print(
            f"{session.name}: n_unit={len(after_unit)} n_fr={len(after_fr)} "
            f"n_cpd={len(with_cpd)} n_dist_pool={len(distance_pool)} "
            f"n_place_pool={len(place_pool)} (median CPD={median_score:+.4f}%)"
        )
    if len(distance_pool) == 0:
        raise ValueError(
            f"{session.name}: empty distance pool after CPD>0 filter "
            f"(median CPD={median_score:+.4f}%, n_cpd={len(with_cpd)})"
        )

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
    input_data = _add_place_update_column(input_data, place_pool, place_trough_bins)

    # --- restrict `spike_count` columns to the distance pool (place pool already
    # consumed for cos_sim; downstream decoder uses only this subset) ---
    distance_set = set(distance_pool)
    keep_cols = [c for c in input_data.columns if c[0] != "spike_count" or c[1] in distance_set]
    input_data = input_data.loc[:, keep_cols]
    return input_data


# %% Per-trial place-tuned population update


def _add_place_update_column(df, place_cells, place_trough_bins):
    """Append `place_update` columns to the per-(cycle, phase_bin) df.

    Per-trial scope so cosine similarity never crosses trial boundaries.
      cos_sim: within-trial cos-sim of place-tuned population rate vectors
        between cycle c and c+1 (averaged over `place_trough_bins`), attributed
        to cycle c — so the distance-code update *within* cycle c is correlated
        with the place-code change from cycle c → c+1. NaN at trial-final
        cycles AND when cycle c+1 is missing (gap from upstream row filters);
        we only measure updates across immediately consecutive cycles.
      n_place_neurons: int, constant across the session — # of place-tuned cells
        actually present in this session (i.e. `len(place_cells)`). (The
        complementary `decoder.n_decoder_neurons` is added by
        `get_session_sweep_update_df`.)

    Incomplete cycles are removed upstream in `get_input_data`, so every
    cycle here has all `n_bins` phase bins; trials can still have gaps in
    `cycle_idx` (handled by the consecutive-cycles check below).
    """
    update = pd.Series(np.nan, index=df.index, dtype=float)

    for _, trial_df in df.groupby(df.trial):
        if trial_df.empty or not place_cells:
            continue

        trial_cycles = np.sort(trial_df.index.get_level_values("cycle_idx").unique())

        # per-cycle rates over place_trough_bins
        trough = trial_df[trial_df.index.get_level_values("phase_bin").isin(place_trough_bins)]
        spikes_by_cycle = trough.spike_count[place_cells].groupby(level="cycle_idx").sum()
        dur_by_cycle = trough.phase_window.duration.groupby(level="cycle_idx").sum()
        rates_by_cycle = spikes_by_cycle.div(dur_by_cycle, axis=0)

        # cos_sim only between immediately consecutive cycles (c, c+1).
        # `(c + 1) not in present` catches both trial-end and mid-trial gaps.
        # `update` is initialised NaN, so skipped cycles stay NaN — that's the
        # intended behaviour for "next cycle missing".
        present = set(rates_by_cycle.index)
        trial_cycle_idxs = trial_df.index.get_level_values("cycle_idx").values
        for c in trial_cycles[:-1]:
            if c not in present or (c + 1) not in present:
                continue
            v_c = rates_by_cycle.loc[c].values
            v_next = rates_by_cycle.loc[c + 1].values
            # cosine() divides by ||v|| and emits a RuntimeWarning on all-zero
            # vectors; explicitly leave cos_sim NaN in that case.
            if not v_c.any() or not v_next.any():
                continue
            val = 1.0 - cosine(v_c, v_next)
            # Attribute (c → c+1) sim to cycle c's rows.
            update.loc[trial_df.index[trial_cycle_idxs == c]] = val

    df = df.copy()
    df[("place_update", "cos_sim")] = update.values
    # session-constant count for downstream session filtering; the matching
    # decoder-population count is added by get_session_sweep_update_df.
    df[("place_update", "n_place_neurons")] = len(place_cells)
    return df


# %% Decoder feature builders


def _split_fold(input_data, held_out_trials):
    """Slice `input_data` into (train_df, test_df) for one CV fold. Accepts
    either a single trial (LOO) or any iterable of trial IDs (k-fold). Every
    surviving cycle already has all n_bins phase bins (incomplete cycles were
    dropped in `get_input_data`), so the bin-0 lookup for cycle_meta is safe."""
    test_mask = input_data.trial.isin(np.atleast_1d(held_out_trials))
    train_df = input_data[~test_mask]
    test_df = input_data[test_mask]
    return train_df, test_df


def _super_phase_rates(df, decoder_cells, n_training_phases):
    """Build the training feature matrix for one LOO fold.

    Collapses (cycle_idx, phase_bin) rows into (cycle_idx, super_phase) rows of
    firing rate per decoder cell, where super_phase = phase_bin //
    n_training_phases (e.g. n=3 → [0,1,2]→0, [3,4,5]→1, …). All cycles are
    complete by construction (see `get_input_data`). The training label is the
    most-common `distance_bin_id` in the super-phase. Returns (rates_df, bin_id).
    """
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

    Note: with non-consecutive `cycle_idx` (gaps from upstream row filtering,
    e.g. moving_only), the wrap path also drops every cycle whose immediate
    successor was filtered out — there's no bin 0 to borrow.
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


def _compute_cycle_metrics(filt_osc, start_samples, end_samples):
    """Per-cycle peak-to-peak amplitude on the band-passed signal. Broadcast
    across the n_bins rows of its cycle by the caller."""
    n_cycles = start_samples.shape[0]
    cycle_first = start_samples[:, 0]
    cycle_last_excl = end_samples[:, -1]
    amplitudes = np.zeros(n_cycles)
    for k in range(n_cycles):
        seg = filt_osc[cycle_first[k] : cycle_last_excl[k]]
        amplitudes[k] = seg.max() - seg.min()
    return amplitudes


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


# %% get neurons


def get_population_distance_score(r2_thres=0.05):
    """Per-cluster distance-to-goal CPD (coefficient of partial determination,
    %) from the neGLM `variance_explained_multiunit` model set — includes
    multi-units, unlike `interaction_validation`. Computed via
    `ve.get_cpd_df`, then averaged over (maze, day) appearances of each
    cluster. Returns a pandas Series indexed by `cluster_unique_ID`.

    Used by `get_input_data` to rank cells for the distance-decoder vs.
    place-cell median split: top half by CPD → distance pool (further filtered
    to CPD > 0), bottom half → place pool.

    Args:
      r2_thres: full-model R² floor passed to `ve.get_cpd_df`; cells with
        unreliable model fits are dropped before CPD is computed. Default
        0.05 — looser than `ve.get_cpd_df`'s own 0.075 to retain more cells.
    """
    cv_scores = lms.load_model_set_cv_scores("variance_explained_multiunit")
    cpd_df = ve.get_cpd_df(cv_scores, r2_thres=r2_thres)
    return cpd_df["distance_to_goal"].groupby("cluster_unique_ID").mean()


# %% Stage 3 — experiment-level per-subject correlation


def test_corr(session_df, use_abs=True):
    """Single-session sanity check: Pearson r between `place_update.cos_sim`
    and `|decoder.distance_error|` (or signed `distance_error` if
    `use_abs=False`). Rows with NaN in either signal are dropped. Returns
    `(r, p, n)`."""
    cs = session_df[("place_update", "cos_sim")].values.astype(float)
    de = session_df[("decoder", "distance_error")].values.astype(float)
    if use_abs:
        de = np.abs(de)
    valid = ~(np.isnan(cs) | np.isnan(de))
    cs, de = cs[valid], de[valid]
    if len(cs) < 2:
        return np.nan, np.nan, int(valid.sum())
    r, p = pearsonr(cs, de)
    label = "|distance_error|" if use_abs else "distance_error"
    print(f"cos_sim ↔ {label}: r={r:+.4f}  p={p:.3g}  n={len(cs)}")
    return float(r), float(p), len(cs)


def plot_update_corr(
    experiment_df,
    min_amplitude=None,
    max_amplitude=None,
    regress_out=None,
    maze_names=None,
    max_peak_mae=None,
    max_trough_mae=None,
    min_n_place_neurons=10,
    min_n_decoder_neurons=10,
    max_distance=0.8,
    decision_points=False,
    alternative="two-sided",
    color="blue",
    axes=None,
    print_stats=True,
):
    """Stage 3 — one Pearson r per subject (cycles pooled across that
    subject's sessions), plotted as individual grey dots with a mean ± SEM
    pointplot, and a cross-subject t-test (default alt='two-sided').

    Two columns side-by-side: left = `|distance_error|` (main hypothesis,
    expected r < 0), right = signed `distance_error`. Both share the y-axis
    when `axes` is None.

    Style mirrors `decoding_error_corr.plot_decoding_error_corr`.

    Args:
      min_amplitude / max_amplitude: cycle-level bounds on `cycle_metrics.amplitude`.
      maze_names: keep only these mazes.
      max_peak_mae / max_trough_mae: drop sessions whose decoder MAE exceeds.
      min_n_place_neurons / min_n_decoder_neurons: drop sessions with too few cells.
      max_distance: drop cycles whose `distance_to_goal.geodesic` (m) exceeds
        this — cycle-level filter, complementary to the `max_steps_to_goal`
        sweep applied at parquet-build time.
      decision_points: True → restrict cycles whose `maze_position.simple` is a
        branching tower (`simple_maze.degree(node) >= 3`); False → no filter.
        Drops rooms_maze (decision points only defined for maze_1 / maze_2).
      regress_out: subset of {'speed', 'distance_to_goal', 'head_direction'}
        partialled out within-subject before correlating; head_direction is
        sin/cos-expanded.
      alternative: ttest_1samp alternative. Default "two-sided"; pass "less"
        to test the directional r < 0 hypothesis.
      axes: pair `(ax_abs, ax_signed)` to plot into. None → create a 1×2 figure
        with shared y. `plot_sweep_grid` passes pre-allocated axes here.

    Returns (axes, {"abs": {per_subject, cross_subject},
                    "signed": {per_subject, cross_subject}}).
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
        df = _filter_decision_points(df)

    results = {
        "abs": _compute_per_subject_corr(df, use_abs=True, regress_out=regress_out, alternative=alternative),
        "signed": _compute_per_subject_corr(df, use_abs=False, regress_out=regress_out, alternative=alternative),
    }

    if print_stats:
        for label, key in [("|distance_error|", "abs"), ("distance_error", "signed")]:
            ps = results[key]["per_subject"]
            cs = results[key]["cross_subject"]
            print(f"per-subject corr  cos_sim ↔ {label}" + (f"  (regressed out: {regress_out})" if regress_out else ""))
            for sub, row in ps.iterrows():
                print(f"  {sub}: n={int(row['n']):>6d}  r={row['r']:+.4f}  p={row['p']:.3g}")
            print(
                f"cross-subject ttest_1samp (alt={alternative!r}): "
                f"mean r={cs['mean_r']:+.4f} ± {cs['sem_r']:.4f}  "
                f"t={cs['t']:+.3f}  p={cs['p']:.4g}  (n={cs['n_subjects']})"
            )

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(1.8, 2), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    if len(axes) != 2:
        raise ValueError(f"`axes` must be a pair (ax_abs, ax_signed); got {len(axes)}")

    for ax, key, ylabel in [
        (axes[0], "abs", "correlation\n(cos_sim ↔ |dist err|)"),
        (axes[1], "signed", "correlation\n(cos_sim ↔ dist err)"),
    ]:
        rs = results[key]["per_subject"]["r"].dropna().values
        _draw_subject_dots_and_mean(ax, rs, color=color)
        ax.set_ylabel(ylabel)

    return axes, results


def _compute_per_subject_corr(df, use_abs, regress_out, alternative):
    """Per-subject Pearson r between cos_sim and (|distance_error| if use_abs
    else signed distance_error), cycles pooled within subject. Returns
    {"per_subject": DataFrame[n,r,p], "cross_subject": dict}."""
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
    return {"per_subject": per_subject, "cross_subject": cross_subject}


def _draw_subject_dots_and_mean(ax, rs, color="blue"):
    """Single-axis subject-dots + cross-subject mean ± SEM marker."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", ls="--", alpha=0.5)
    dot_x = -0.2 + np.linspace(-0.07, 0.07, max(len(rs), 1))
    ax.scatter(dot_x, rs, color="grey", alpha=0.7, s=30, edgecolors="none", zorder=3)
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


def plot_sweep_grid(
    feat1="msg",
    feat2="C",
    feat1_values=None,
    feat2_values=None,
    C=1.0,
    max_steps_to_goal=12,
    moving_only=True,
    phase_suffix="",
    sweep_dir=None,
    figsize_per_cell=(1.6, 2.6),
    sharey=True,
    **plot_kwargs,
):
    """Multipanel sweep summary: each cell is a 2-column `plot_update_corr`
    panel (|dist err| on the left, signed dist err on the right) for one
    `(feat1, feat2)` combination of the catch-update sweep.

    Layout: rows = `feat1`, cols = `feat2`, each cell holds the abs/signed
    pair. With sharey across the whole grid the cross-condition shape is
    directly comparable.

    Args:
      feat1, feat2: choose two of {"msg", "C", "moving", "phase"}. The other
        two axes are pinned via the matching kwargs below.
      feat1_values, feat2_values: explicit value lists for the swept axes.
        None → defaults to the full sweep ranges from `jobs/sweep_update/submit.py`:
          msg ∈ [8, 12, 16, 20, 24, 28]
          C ∈ [0.1, 1.0, 10.0]
          moving ∈ [True, False]
          phase ∈ ["", "shifted"]
      C, max_steps_to_goal, moving_only, phase_suffix: pinned values for the
        axes NOT being swept. Match the `_build_param_sets` convention in
        `submit.py` so tag-building works.
      sweep_dir: directory holding the cached parquets. None →
        `RESULTS_DIR/sweep_update_tests/`.
      figsize_per_cell: (w, h) per `plot_update_corr` cell — the cell itself
        is two stacked subpanels so the figure ends up ~2*w wide per col.
      sharey: share y-axis across all panels in the grid.
      **plot_kwargs: forwarded to `plot_update_corr` (filters, regress_out,
        etc.).

    Returns (fig, results_df) where results_df has one row per (feat1, feat2)
    cell × {"abs", "signed"} with columns:
      feat1, feat2, abs_or_signed, n_subjects, mean_r, sem_r, t, p, path_exists.
    """
    if sweep_dir is None:
        sweep_dir = RESULTS_DIR / "sweep_update_tests"
    sweep_dir = Path(sweep_dir)

    defaults = {
        "msg": [8, 12, 16, 20, 24, 28],
        "C": [1e-1, 1.0, 10.0],
        "moving": [True, False],
        "phase": ["", "shifted"],
    }
    if feat1 not in defaults or feat2 not in defaults:
        raise ValueError(f"feat1/feat2 must be in {list(defaults)}; got {feat1!r}, {feat2!r}")
    if feat1 == feat2:
        raise ValueError("feat1 and feat2 must differ")

    f1_vals = feat1_values if feat1_values is not None else defaults[feat1]
    f2_vals = feat2_values if feat2_values is not None else defaults[feat2]

    pinned = {"msg": max_steps_to_goal, "C": C, "moving": moving_only, "phase": phase_suffix}

    n_rows, n_cols = len(f1_vals), len(f2_vals)
    fig, ax_grid = plt.subplots(
        n_rows,
        n_cols * 2,
        figsize=(figsize_per_cell[0] * n_cols * 2, figsize_per_cell[1] * n_rows),
        sharey=sharey,
        squeeze=False,
    )

    results_rows = []
    for r, v1 in enumerate(f1_vals):
        for c, v2 in enumerate(f2_vals):
            params = dict(pinned)
            params[feat1] = v1
            params[feat2] = v2
            tag = _build_sweep_tag(params["C"], params["msg"], params["moving"], params["phase"])
            path = sweep_dir / f"sweep_update_df{tag}.parquet"
            ax_abs = ax_grid[r, c * 2]
            ax_signed = ax_grid[r, c * 2 + 1]

            if not path.exists():
                for ax in (ax_abs, ax_signed):
                    ax.text(
                        0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes, fontsize=8, color="grey"
                    )
                    ax.set_xticks([])
                    for s in ax.spines.values():
                        s.set_alpha(0.3)
                results_rows.append(
                    {
                        feat1: v1,
                        feat2: v2,
                        "abs_or_signed": "abs",
                        "n_subjects": 0,
                        "mean_r": np.nan,
                        "sem_r": np.nan,
                        "t": np.nan,
                        "p": np.nan,
                        "path_exists": False,
                    }
                )
                results_rows.append(
                    {
                        feat1: v1,
                        feat2: v2,
                        "abs_or_signed": "signed",
                        "n_subjects": 0,
                        "mean_r": np.nan,
                        "sem_r": np.nan,
                        "t": np.nan,
                        "p": np.nan,
                        "path_exists": False,
                    }
                )
                continue

            df = pd.read_parquet(path)
            _, res = plot_update_corr(df, axes=(ax_abs, ax_signed), print_stats=False, **plot_kwargs)

            # per-cell titles: t / p for each panel
            for ax, key in [(ax_abs, "abs"), (ax_signed, "signed")]:
                cs = res[key]["cross_subject"]
                ax.set_title(f"t({cs['n_subjects']})={cs['t']:+.2f}\np={cs['p']:.3g}", fontsize=8)
                ax.set_ylabel("")  # row label handles this — see below

            for key in ("abs", "signed"):
                cs = res[key]["cross_subject"]
                results_rows.append(
                    {
                        feat1: v1,
                        feat2: v2,
                        "abs_or_signed": key,
                        "n_subjects": cs["n_subjects"],
                        "mean_r": cs["mean_r"],
                        "sem_r": cs["sem_r"],
                        "t": cs["t"],
                        "p": cs["p"],
                        "path_exists": True,
                    }
                )

            # mini-header labelling abs/signed columns inside each cell
            if r == 0:
                ax_abs.annotate(
                    "|err|", xy=(0.5, 1.55), xycoords="axes fraction", ha="center", fontsize=9, color="grey"
                )
                ax_signed.annotate(
                    "signed", xy=(0.5, 1.55), xycoords="axes fraction", ha="center", fontsize=9, color="grey"
                )

    # outer row labels (feat1) and col-pair labels (feat2)
    for r, v1 in enumerate(f1_vals):
        ax_grid[r, 0].annotate(
            f"{feat1}={_fmt_value(v1)}",
            xy=(-1.1, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            rotation=90,
        )
    for c, v2 in enumerate(f2_vals):
        ax_grid[0, c * 2].annotate(
            f"{feat2}={_fmt_value(v2)}",
            xy=(1.0, 2.0),
            xycoords="axes fraction",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    fig.tight_layout()

    results_df = pd.DataFrame.from_records(results_rows)
    return fig, results_df


def _build_sweep_tag(C, msg, moving, phase_suffix):
    """Mirror of `jobs.sweep_update.submit._build_param_sets` tag scheme."""
    if C == 1e-1:
        C_str = "C1e-1"
    elif C == 1.0:
        C_str = "C1"
    elif C == 10.0:
        C_str = "C10"
    else:
        raise ValueError(f"unrecognised C={C!r}")
    parts = [C_str, f"msg{msg}", "move" if moving else "nomove"]
    if phase_suffix:
        parts.append(phase_suffix)
    return "_" + "_".join(parts)


def _fmt_value(v):
    """Compact value formatter for grid labels (avoid 0.1 vs 1e-1 noise)."""
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, float):
        return f"{v:g}"
    if v == "":
        return "canon"
    return str(v)


def _filter_decision_points(df, min_degree=3):
    """Restrict the experiment df to cycles whose `maze_position.simple` is at
    or adjacent to a branching tower in the simple-maze graph.

    Two kinds of decision-point positions are kept:
      1. Towers with `simple_maze.degree(node) >= min_degree` (default 3 —
         i.e. ≥3 walkways meet at that tower).
      2. Edges (walkways) that touch at least one such tower.

    rooms_maze is dropped (decision points only defined for maze_1 / maze_2).
    """
    import networkx as nx

    kept = []
    for maze_name in ["maze_1", "maze_2"]:
        maze_df = df[df[("maze_name", "")] == maze_name]
        if maze_df.empty:
            continue
        simple_maze = mr.get_simple_maze(maze_name)
        node_labels = nx.get_node_attributes(simple_maze, "label")  # coord → label
        edge_labels = nx.get_edge_attributes(simple_maze, "label")  # (u, v) → label

        branching_nodes = {n for n in simple_maze.nodes if simple_maze.degree(n) >= min_degree}
        dp_labels = {node_labels[n] for n in branching_nodes}
        for (u, v), label in edge_labels.items():
            if u in branching_nodes or v in branching_nodes:
                dp_labels.add(label)

        kept.append(maze_df[maze_df[("maze_position", "simple")].astype(str).isin(dp_labels)])
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
