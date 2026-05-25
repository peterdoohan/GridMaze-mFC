"""
catch_update2 — minimal max-SNR test of theta-timescale place ↔ distance coupling.

Hypothesis: on a theta timescale the distance-to-goal code and the place-direction
code update together — cycles where the distance representation changes a lot are
followed by cycles where the place representation changes a lot.

This is the simplified pipeline distilled from the exploratory first pass: one
operation per code — **how much the decoded posterior changes** (Jensen–Shannon
divergence) — which carried essentially all the signal.

  * Phase: Hilbert, 7–11 Hz (`la.get_lfp_phase`).
  * Pools: disjoint AND each genuinely tuned — distance pool by distance-to-goal
    CPD, place pool by place-direction CPD ("argmax" split).
  * `distance_update` (within a cycle): JS divergence between the decoded
    distance-to-goal posterior at theta peak vs trough.
  * `place_update` (across cycles): JS divergence between the place-decoder
    posterior at cycle c vs c+1.
  * Confounds carried per cycle (per-pool spike counts, speed, theta amplitude,
    head direction, distance-to-goal) so the link can be shown to survive the
    per-cycle spike-count artefact.

Main result (n=6): per-subject correlation distance_update ↔ place_update →
t≈6.2, p≈0.0016; pooled standardised β≈+0.045. See `link_per_subject` /
`link_pooled` / `plot_link`.

Stage 5 (general): `get_sliding_update_df` / `link_matrix` / `plot_link_matrix` generalise
this single test into a full lag × lag scan — one fixed decoder per code trained on
width-`window_width` sliding theta-phase windows and read out at every window, the JS²
"update" taken at every SIGNED lag from one cycle before the anchor to one cycle after
(`span_cycles` cycles each way), then distance- vs place-update lag profiles correlated across
codes. The result above is one entry of that matrix; signed lags make it a lead/lag map
(distance-leads-place vs place-leads-distance), and which code is "within" vs "across" is just
which axis you read.

HARD RULE: this file is the only place new behaviour is written; everything
reusable is imported (notably the per-cycle backbone from `catch_update`) and
never modified.

@peterdoohan
"""

# %% Imports
import hashlib
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.spatial.distance import jensenshannon
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

# reuse the per-cycle backbone helpers + distance decoder (import, never modify)
from GridMaze.analysis.theta_mod import catch_update as cu

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

# %% Global variables
THETA_RANGE = (7, 11)

RESULTS_DIR = RESULTS_PATH / "theta_mod" / "catch_update2"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json") as _f:
    SUBJECT_IDS = json.load(_f)

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]

# per-cycle covariates partialled out / controlled in the link tests
CONFOUNDS = ["speed", "theta_amp", "n_sp_place", "n_sp_dist", "dist2goal", "hd_sin", "hd_cos"]


# %% Neuron pools (disjoint, each genuinely tuned)


def get_population_cpd_scores(r2_thres=0.05):
    """Per-cluster CPD (%) for BOTH distance-to-goal and place-direction from the
    neGLM `variance_explained_multiunit` model set, averaged over each cluster's
    (maze, day) appearances. DataFrame indexed by `cluster_unique_ID`."""
    cv_scores = lms.load_model_set_cv_scores("variance_explained_multiunit")
    cpd_df = ve.get_cpd_df(cv_scores, r2_thres=r2_thres)
    return cpd_df[["distance_to_goal", "place_direction"]].groupby(level="cluster_unique_ID").mean()


def _select_pools(cells, cpd_scores):
    """Assign each cluster to the pool of its LARGER CPD, provided that CPD is
    positive: distance pool = distance-CPD > 0 AND distance-CPD > place-CPD;
    place pool = place-CPD > 0 AND place-CPD > distance-CPD. A cluster tuned to
    both still goes to its dominant variable; dropped only if its larger CPD is
    ≤ 0 (untuned to both) or it ties exactly (distance-CPD == place-CPD). The
    two `>` comparisons are mutually exclusive, so the pools are disjoint by
    construction (asserted below). Assumes a unique `cluster_unique_ID` index so
    `d[c]`/`p[c]` are scalars (guaranteed by `get_population_cpd_scores`)."""
    with_cpd = [c for c in cells if c in cpd_scores.index]
    d = cpd_scores["distance_to_goal"].reindex(with_cpd)
    p = cpd_scores["place_direction"].reindex(with_cpd)
    distance_pool = [c for c in with_cpd if d[c] > 0 and d[c] > p[c]]
    place_pool = [c for c in with_cpd if p[c] > 0 and p[c] > d[c]]
    assert not (set(distance_pool) & set(place_pool)), "distance/place pools overlap"
    return distance_pool, place_pool


# %% Stage 1 — per-(cycle, phase_bin) input data


def _circular_shift_spike_times(spike_times, spike_clusters, t_lo, t_hi, rng, min_frac=0.05):
    """Circularly shift the whole spike train in time by one shared random offset,
    wrapping within [t_lo, t_hi). All clusters are shifted together, so each cluster's
    spike count and the population's co-firing are preserved while the alignment of
    spikes to behaviour and to LFP/theta phase is broken. The offset is drawn uniformly
    from [min_frac, 1-min_frac]·span so it is never trivially small. Re-sorts (times,
    clusters) jointly by the shifted time so the per-cluster `searchsorted` spike-count
    downstream stays valid (it assumes ascending times). Returns the shifted,
    time-sorted (spike_times, spike_clusters)."""
    span = t_hi - t_lo
    offset = rng.uniform(min_frac, 1.0 - min_frac) * span
    shifted = t_lo + np.mod(spike_times - t_lo + offset, span)
    order = np.argsort(shifted, kind="stable")
    return shifted[order], spike_clusters[order]


def get_input_data(
    session,
    n_bins=12,
    shank=3,
    max_steps_to_goal=20,
    exclude_at_goal=True,
    bin_spacing=0.05,
    moving_only=True,
    min_firing_rate=0.5,
    circular_shift_seed=None,
    circular_shift_min_frac=0.05,
    cpd_scores=None,
    verbose=False,
):
    """Build the per-(cycle, phase_bin) dataframe and the disjoint neuron pools.

    Returns (input_data, pools) with pools = {"distance": [...], "place": [...]}.
    Keeps `spike_count` columns for BOTH pools (the place pool drives the place
    update), plus
    navigation, `phase_window`, `cycle_metrics.amplitude`, distance binning,
    per-cycle pool spike-count confounds, and subject/maze/session metadata.
    Backbone (cycle/phase-window detection, navigation alignment) reused from
    `catch_update`.

    CONTROL — `circular_shift_seed` (None = off): when set, the whole spike train is
    circularly shifted in time by one shared random offset (wrapping within the LFP
    span) BEFORE spikes are counted into windows — see `_circular_shift_spike_times`.
    This decouples spikes from behaviour and from LFP/theta phase while preserving each
    cluster's spike count and the population's co-firing, so it tests whether the
    distance↔place coupling needs spikes aligned to behaviour at all (a complement to
    the within-cycle phase permutation, which keeps that alignment). The CPD-based pool
    assignment is unaffected (it reads the precomputed true-data model set), so the
    shifted run uses essentially the same neurons. `circular_shift_min_frac` bounds the
    offset to [min_frac, 1-min_frac]·span so the shift is never trivially small.
    """
    # --- LFP → Hilbert theta phase → phase bins ---
    raw_lfp = lu.get_LFP(session, shank=shank)
    filt_osc, theta_phase = la.get_lfp_phase(raw_lfp, freq_range=THETA_RANGE, N=4, return_filtered=True)
    _, bin_indices = la.bin_lfp_phase(theta_phase, n_bins=n_bins)
    lfp_times = session.lfp_times

    # --- detect (cycle, phase_bin) windows ---
    start_samples, end_samples = cu._detect_cycle_phase_windows(bin_indices, n_bins=n_bins)
    n_cycles = start_samples.shape[0]
    if n_cycles == 0:
        raise ValueError(f"{session.name}: no complete theta cycles")
    start_times = lfp_times[start_samples]
    end_times = lfp_times[end_samples]
    midpoint_times = (start_times + end_times) / 2

    # --- spike counts per (cluster, cycle, bin) ---
    spike_times = np.asarray(session.spike_times).reshape(-1)
    spike_clusters = np.asarray(session.spike_clusters).reshape(-1)
    if circular_shift_seed is not None:
        spike_times, spike_clusters = _circular_shift_spike_times(
            spike_times,
            spike_clusters,
            float(lfp_times[0]),
            float(lfp_times[-1]),
            np.random.default_rng(circular_shift_seed),
            circular_shift_min_frac,
        )
    cluster_IDs = np.sort(np.unique(spike_clusters)).astype(np.float64)
    cluster_unique_IDs = convert.cluster_IDs2scluster_unique_IDs(session.session_info, cluster_IDs)

    start_flat, end_flat = start_times.ravel(), end_times.ravel()
    spike_counts = np.zeros((len(cluster_IDs), n_cycles * n_bins), dtype=np.int32)
    for i, cluster_id in enumerate(cluster_IDs):
        cst = spike_times[spike_clusters == cluster_id]
        spike_counts[i] = np.searchsorted(cst, end_flat) - np.searchsorted(cst, start_flat)

    amplitudes = cu._compute_cycle_metrics(filt_osc, start_samples, end_samples)

    # --- assemble ---
    row_index = pd.MultiIndex.from_product([np.arange(n_cycles), np.arange(n_bins)], names=["cycle_idx", "phase_bin"])
    nav_block = cu._align_navigation(session.navigation_df, midpoint_times.ravel(), row_index)

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

    cycle_metrics_block = pd.DataFrame({("cycle_metrics", "amplitude"): np.repeat(amplitudes, n_bins)}, index=row_index)
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

    # --- drop any cycle that lost a phase bin to the filters ---
    cycle_sizes = input_data.groupby(level="cycle_idx").size()
    complete_cycles = cycle_sizes[cycle_sizes == n_bins].index
    input_data = input_data.loc[input_data.index.get_level_values("cycle_idx").isin(complete_cycles)]
    if input_data.empty:
        raise ValueError(f"{session.name}: no rows survived row filters")

    # --- neuron pools (unit filter → FR floor → CPD → tuned disjoint split) ---
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
    fr_per_cell = input_data.spike_count[after_unit].sum(axis=0) / nav_duration
    after_fr = [c for c in after_unit if fr_per_cell.get(c, 0.0) >= min_firing_rate]

    if cpd_scores is None:
        cpd_scores = get_population_cpd_scores()
    distance_pool, place_pool = _select_pools(after_fr, cpd_scores)
    if verbose:
        print(
            f"{session.name}: n_unit={len(after_unit)} n_fr={len(after_fr)} "
            f"n_dist={len(distance_pool)} n_place={len(place_pool)}"
        )
    if len(distance_pool) == 0 or len(place_pool) == 0:
        raise ValueError(f"{session.name}: empty pool (dist={len(distance_pool)}, place={len(place_pool)})")

    # --- distance-to-goal binning (over the post-filter range) ---
    dist_series = input_data[("distance_to_goal", "geodesic")]
    max_distance = float(dist_series.max()) + bin_spacing * 1e-3
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
    input_data.loc[:, ("distance_bin_mid", "")] = [iv.mid if pd.notna(iv) else np.nan for iv in intervals]
    observed_mids = sorted(input_data[("distance_bin_mid", "")].dropna().unique())
    mid_to_id = {m: i for i, m in enumerate(observed_mids)}
    input_data.loc[:, ("distance_bin_id", "")] = input_data[("distance_bin_mid", "")].map(mid_to_id).astype("Int64")

    # --- per-cycle pool spike-count confounds (summed over all bins of the cycle) ---
    dist_spikes = input_data.spike_count[distance_pool].groupby(level="cycle_idx").sum().sum(axis=1)
    place_spikes = input_data.spike_count[place_pool].groupby(level="cycle_idx").sum().sum(axis=1)
    cyc = input_data.index.get_level_values("cycle_idx")
    input_data.loc[:, ("n_spikes", "distance_pool")] = dist_spikes.reindex(cyc).values
    input_data.loc[:, ("n_spikes", "place_pool")] = place_spikes.reindex(cyc).values

    # --- metadata for the experiment-level df ---
    input_data.loc[:, ("subject_ID", "")] = session.subject_ID
    input_data.loc[:, ("maze_name", "")] = session.maze_name
    input_data.loc[:, ("session_ID", "")] = session.name

    return input_data, {"distance": distance_pool, "place": place_pool}


# %% Stage 2 — per-session metrics


def _js2(p, q):
    """Squared Jensen–Shannon divergence between two probability vectors. Returns
    0.0 for (numerically) identical distributions: scipy's `jensenshannon` takes
    a sqrt of a divergence that rounds slightly negative there, which both emits
    a RuntimeWarning and returns NaN. Identical posteriors == genuine zero change,
    so 0.0 is the correct value; the warning is suppressed for that sqrt only."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = jensenshannon(p, q)
    return 0.0 if np.isnan(d) else float(d) ** 2


def get_session_catch_update2_df(
    session,
    distance_peak_bins=[3, 4, 5],
    distance_trough_bins=[9, 10, 11],
    place_phase_bins=[0, 1, 2],
    n_training_phases=3,
    normalise_X=True,
    C=1.0,
    n_folds=10,
    cv_seed=0,
    n_permutations=1,
    permutation_seed=0,
    cpd_scores=None,
    verbose=False,
    **input_data_kwargs,
):
    """One row per test cycle with the distance-update and place-update metrics
    plus confounds + meta. Trial CV; per fold a distance decoder and a place
    decoder are trained on the training trials and applied to the held-out trials.

    ("distance","update") = JS(peak_posterior, trough_posterior)   within-cycle distance change
    ("place","update")    = JS(place_post(c), place_post(c+1))      across-cycle place change
    ("qc",…)              = decoder MAE, fold, per-pool neuron counts (diagnostics)

    Theta-phase permutation null (`n_permutations`):
      None (default) → the df above, unchanged.
      int N          → ONE df = the true run (("qc","permutation")==0) stacked above N
        phase-permuted runs (1..N). Each permutation independently scrambles which
        spike/timing sub-window sits at each phase bin *within* every cycle (the whole
        population vector at a bin moves as a unit), then RE-RUNS training + readout via
        `_permute_phases_within_cycles` + `_compute_session_metrics`. Behaviour, confounds
        and the per-cycle `n_spikes` totals are identical across permutations for a given
        cycle (the permutation only reorders within-cycle phase); only the decoded updates
        and decoder QC change. Partition by ("qc","permutation") to compare null vs true.
        Permutations are seeded per (permutation_seed, session, s) so they are uncorrelated
        across sessions yet reproducible.
    """
    input_data, pools = get_input_data(session, cpd_scores=cpd_scores, verbose=verbose, **input_data_kwargs)
    distance_cells, place_cells = pools["distance"], pools["place"]
    n_bins = int(input_data.index.get_level_values("phase_bin").max()) + 1
    params = dict(
        distance_peak_bins=distance_peak_bins,
        distance_trough_bins=distance_trough_bins,
        place_phase_bins=place_phase_bins,
        n_training_phases=n_training_phases,
        normalise_X=normalise_X,
        C=C,
        n_folds=n_folds,
        cv_seed=cv_seed,
    )

    true_df = _compute_session_metrics(input_data, distance_cells, place_cells, session.name, verbose=verbose, **params)
    if n_permutations is None:
        return true_df

    true_df[("qc", "permutation")] = 0  # 0 = true run; 1..N = phase-permuted nulls
    out = [true_df]
    sess_int = int.from_bytes(hashlib.blake2b(session.name.encode(), digest_size=8).digest(), "big")
    for s in range(1, n_permutations + 1):
        rng = np.random.default_rng(np.random.SeedSequence([permutation_seed, sess_int, s]))
        permuted = _permute_phases_within_cycles(input_data, n_bins, rng)
        null_df = _compute_session_metrics(permuted, distance_cells, place_cells, session.name, verbose=False, **params)
        null_df[("qc", "permutation")] = s
        out.append(null_df)
    return pd.concat(out)


def _permute_phases_within_cycles(input_data, n_bins, rng):
    """Return a copy of `input_data` with theta phases scrambled WITHIN each cycle.

    For every cycle the `spike_count` and `phase_window` rows (the physical sub-window
    content — spikes plus their integration timing) are reassigned to a random
    permutation of the `n_bins` phase-bin slots; navigation, `cycle_metrics`,
    `distance_bin_*`, `n_spikes` and metadata stay put. The whole population vector at a
    bin moves as a unit (preserves within-bin co-firing, destroys only the
    phase→population mapping). Permuting `phase_window` alongside `spike_count` keeps each
    slot's spikes matched to its own duration, so rate = sum(spikes)/sum(durations) stays
    physically correct inside the downstream helpers. The two groups are permuted
    separately (with the SAME per-cycle order) to preserve their dtypes.

    Relies on `input_data` being sorted by (cycle_idx, phase_bin) with exactly `n_bins`
    rows per surviving cycle (guaranteed by `get_input_data`), so a row-major reshape to
    (n_cycles, n_bins) groups each cycle's bins in order.
    """
    df = input_data.copy()
    order = np.arange(len(df)).reshape(-1, n_bins)  # (n_cycles, n_bins) positional, bin-sorted
    perm = np.argsort(rng.random(order.shape), axis=1)  # independent per-cycle permutation
    new_idx = np.take_along_axis(order, perm, axis=1).ravel()
    for group in ("spike_count", "phase_window"):
        gcols = [c for c in df.columns if c[0] == group]
        df[gcols] = df[gcols].values[new_idx]
    return df


def _compute_session_metrics(
    input_data,
    distance_cells,
    place_cells,
    session_name,
    distance_peak_bins,
    distance_trough_bins,
    place_phase_bins,
    n_training_phases,
    normalise_X,
    C,
    n_folds,
    cv_seed,
    verbose,
):
    """Core per-cycle metric computation for one (possibly phase-permuted) `input_data`.

    Trial CV; per fold a distance decoder and a place decoder are trained on the training
    trials and applied to the held-out test cycles. Returns the cycle_idx-indexed session
    df (distance/place updates + qc fold/decoder_mae/n_* diagnostics). Factored out of
    `get_session_catch_update2_df` so the permutation null can re-run the identical
    pipeline on scrambled inputs. `distance_bin_mids`/`n_bins` are recomputed here from
    the passed `input_data` (both invariant to the within-cycle phase permutation).
    """
    distance_bin_mids = np.array(sorted(input_data.distance_bin_mid.dropna().unique()))
    n_bins = int(input_data.index.get_level_values("phase_bin").max()) + 1
    if n_bins % n_training_phases != 0:
        raise ValueError(f"n_training_phases ({n_training_phases}) must divide n_bins ({n_bins})")

    trials = input_data.trial.dropna().unique()
    if n_folds == -1:
        folds = [np.array([t]) for t in trials]
    elif n_folds > 0:
        if n_folds > len(trials):
            raise ValueError(f"n_folds={n_folds} > n_trials={len(trials)} for {session_name}")
        rng = np.random.default_rng(cv_seed)
        folds = np.array_split(rng.permutation(trials), n_folds)
    else:
        raise ValueError(f"n_folds must be -1 or positive, got {n_folds!r}")

    per_cycle, abs_errs = [], []
    for fold_idx, held_out in enumerate(folds):
        train_df, test_df = cu._split_fold(input_data, held_out)

        # ---- distance decoder (super-phase training) ----
        rates_train, y_train = cu._super_phase_rates(train_df, distance_cells, n_training_phases)
        if rates_train.empty or y_train.nunique() < 2:
            continue
        scaler = StandardScaler().fit(rates_train.values) if normalise_X else None
        X_train = scaler.transform(rates_train.values) if scaler is not None else rates_train.values
        lr_kwargs = {"random_state": 0, "max_iter": 2000, "class_weight": "balanced"}
        if C is None:
            lr_kwargs["penalty"] = None
        else:
            lr_kwargs["C"] = C
        decoder = LogisticRegression(**lr_kwargs).fit(X_train, y_train.values)
        mids_for_classes = distance_bin_mids[decoder.classes_]

        # ---- distance update: JS between peak-phase and trough-phase posteriors ----
        peak_rates = cu._cycle_subset_rates(test_df, distance_cells, distance_peak_bins)
        trough_rates = cu._cycle_subset_rates(test_df, distance_cells, distance_trough_bins)
        shared = peak_rates.index.intersection(trough_rates.index)
        if len(shared) == 0:
            continue
        Xp = peak_rates.loc[shared].values
        Xt = trough_rates.loc[shared].values
        Xp_s = scaler.transform(Xp) if scaler is not None else Xp
        Xt_s = scaler.transform(Xt) if scaler is not None else Xt
        post_p = decoder.predict_proba(Xp_s)
        post_t = decoder.predict_proba(Xt_s)
        distance_update = np.array([_js2(a, b) for a, b in zip(post_p, post_t)])

        # ---- place update: JS between the place posteriors at cycle c and c+1 ----
        place_update = _decoded_place_update(train_df, test_df, place_cells, place_phase_bins, normalise_X)

        # ---- assemble per-test-cycle rows (bin-0 carries nav + confounds) ----
        meta = test_df.xs(0, level="phase_bin").loc[shared].copy()
        meta = meta.drop(columns=["spike_count"], level=0, errors="ignore")
        meta[("distance", "update")] = distance_update
        meta[("place", "update")] = place_update.reindex(shared).values
        meta[("qc", "fold")] = fold_idx
        per_cycle.append(meta)

        # decoder QC: |peak-phase decoded distance − true bin midpoint|
        true_d = meta[("distance_bin_mid", "")].values.astype(float)
        abs_errs.append(np.abs((post_p @ mids_for_classes) - true_d))

    if not per_cycle:
        raise ValueError(f"{session_name}: no usable test cycles")
    session_df = pd.concat(per_cycle).sort_index()
    session_df[("qc", "decoder_mae")] = float(np.mean(np.concatenate(abs_errs)))
    session_df[("qc", "n_distance_neurons")] = len(distance_cells)
    session_df[("qc", "n_place_neurons")] = len(place_cells)
    if verbose:
        print(f"{session_name}: decoder_mae={session_df[('qc','decoder_mae')].iloc[0]:.3f} m")
    return session_df


def _decoded_place_update(train_df, test_df, place_cells, place_phase_bins, normalise_X):
    """Place decoder trained on train trials, applied to held-out cycles; returns
    the place update = JS divergence between the place posterior at cycle c and
    c+1 (attributed to cycle c), as a Series indexed by cycle_idx."""
    out = pd.Series(dtype=float)
    if not place_cells:
        return out
    tr_rates = cu._cycle_subset_rates(train_df, place_cells, place_phase_bins)
    tr_pos = train_df.xs(place_phase_bins[0], level="phase_bin")[("maze_position", "simple")]
    tr_pos = tr_pos.reindex(tr_rates.index).astype(str)
    valid = tr_pos.notna() & (tr_pos != "nan")
    tr_rates, tr_pos = tr_rates[valid.values], tr_pos[valid.values]
    if tr_rates.empty or tr_pos.nunique() < 2:
        return out

    scaler = StandardScaler().fit(tr_rates.values) if normalise_X else None
    Xtr = scaler.transform(tr_rates.values) if scaler is not None else tr_rates.values
    place_dec = LogisticRegression(random_state=0, max_iter=2000, class_weight="balanced", C=1.0).fit(
        Xtr, tr_pos.values
    )
    te_rates = cu._cycle_subset_rates(test_df, place_cells, place_phase_bins)
    if te_rates.empty:
        return out
    Xte = scaler.transform(te_rates.values) if scaler is not None else te_rates.values
    post_df = pd.DataFrame(place_dec.predict_proba(Xte), index=te_rates.index)

    cycle_trial = test_df.trial.groupby(level="cycle_idx").first()
    rows = {}
    for _, cyc_idx in cycle_trial.groupby(cycle_trial).groups.items():
        present = set(int(c) for c in cyc_idx) & set(int(c) for c in te_rates.index)
        for c in sorted(present):
            if (c + 1) not in present:
                continue
            rows[c] = _js2(post_df.loc[c].values, post_df.loc[c + 1].values)
    return pd.Series(rows, dtype=float)


# %% Stage 3 — cross-session runner


def get_catch_update2_df(save=False, verbose=True, n_jobs=3, tag=None, **session_kwargs):
    """Run `get_session_catch_update2_df` across every subject × maze × late day
    and concatenate to one cross-session dataframe, cached to parquet under
    `RESULTS_DIR/runs/catch_update2_df{tag}.parquet`. Per-session failures are
    swallowed (printed if verbose).

    `**session_kwargs` are forwarded to `get_session_catch_update2_df`, so the
    theta-phase permutation null runs experiment-wide via e.g.
    `get_catch_update2_df(save=True, tag="_perm200", n_permutations=200)` — every
    session is permuted (same `permutation` labels across sessions, independently
    seeded), and the `("qc","permutation")` column flows through to the parquet."""
    save_path = RESULTS_DIR / "runs" / f"catch_update2_df{tag or ''}.parquet"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists() and not save:
        if verbose:
            print(f"loading cached: {save_path}")
        return pd.read_parquet(save_path)

    cpd_scores = get_population_cpd_scores()

    def _process(session):
        if verbose:
            print(session.name)
        try:
            return get_session_catch_update2_df(session, cpd_scores=cpd_scores, verbose=False, **session_kwargs)
        except Exception as e:
            if verbose:
                print(f"  error on {session.name}: {e}")
            return None

    all_dfs = []
    for subject in SUBJECT_IDS:
        for maze_name in MAZE_NAMES:
            try:
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
            except FileNotFoundError:
                if verbose:
                    print(f"  no sessions for {subject} / {maze_name}")
                continue
            if not isinstance(sessions, list):
                sessions = [sessions]
            session_dfs = Parallel(n_jobs=n_jobs)(delayed(_process)(s) for s in sessions)
            all_dfs.extend([d for d in session_dfs if d is not None])

    pop_df = pd.concat(all_dfs).reset_index()
    if save:
        pop_df.to_parquet(save_path)
        if verbose:
            print(f"saved: {save_path}  ({len(pop_df)} rows)")
    return pop_df


# %% Stage 4 — link test (distance-update ↔ place-update)


def build_tidy(df):
    """Flatten the experiment df to a tidy per-cycle frame: `dist`
    (distance_update), `place` (place_update), confounds, and subject/session/maze."""
    tidy = pd.DataFrame(
        {
            "subject": df[("subject_ID", "")].values,
            "session": df[("session_ID", "")].values,
            "maze": df[("maze_name", "")].values,
            "dist": df[("distance", "update")].astype(float).values,
            "place": df[("place", "update")].astype(float).values,
            "speed": df[("speed", "")].astype(float).values,
            "theta_amp": df[("cycle_metrics", "amplitude")].astype(float).values,
            "n_sp_place": df[("n_spikes", "place_pool")].astype(float).values,
            "n_sp_dist": df[("n_spikes", "distance_pool")].astype(float).values,
            "dist2goal": df[("distance_to_goal", "geodesic")].astype(float).values,
            "hd": df[("head_direction", "value")].astype(float).values,
        }
    )
    tidy["hd_sin"] = np.sin(np.deg2rad(tidy["hd"]))
    tidy["hd_cos"] = np.cos(np.deg2rad(tidy["hd"]))
    return tidy


def _apply_filters(
    df,
    maze_names=None,
    min_amplitude=None,
    max_decoder_mae=None,
    min_n_place_neurons=None,
    min_n_distance_neurons=None,
    max_distance=None,
):
    """Cycle/session-level filters on the experiment df."""
    out = df
    if maze_names is not None:
        out = out[out[("maze_name", "")].isin(maze_names)]
    if min_amplitude is not None:
        out = out[out[("cycle_metrics", "amplitude")] >= min_amplitude]
    if max_decoder_mae is not None:
        out = out[out[("qc", "decoder_mae")] <= max_decoder_mae]
    if min_n_place_neurons is not None:
        out = out[out[("qc", "n_place_neurons")] >= min_n_place_neurons]
    if min_n_distance_neurons is not None:
        out = out[out[("qc", "n_distance_neurons")] >= min_n_distance_neurons]
    if max_distance is not None:
        out = out[out[("distance_to_goal", "geodesic")] <= max_distance]
    return out


def _zscore(tidy, cols):
    out = tidy.copy()
    for c in cols:
        v = out[c].astype(float)
        sd = v.std()
        out[c] = (v - v.mean()) / sd if sd > 0 else 0.0
    return out


def link_per_subject(df, confounds=CONFOUNDS, alternative="two-sided", maze_names=None, **filters):
    """PRIMARY test: per-subject partial Pearson r between distance_update and
    place_update (confounds regressed out within subject), then a cross-subject
    one-sample t-test on the subject r's (n≈6). Returns dict(mean_r, sem_r, t, p,
    n_subjects, per_subject)."""
    tidy = build_tidy(_apply_filters(df, maze_names=maze_names, **filters))
    rows = []
    for subj, s in tidy.groupby("subject"):
        s = s.dropna(subset=["place", "dist"] + (confounds or []))
        if len(s) < 10:
            continue
        x, y = s["dist"].values, s["place"].values
        if confounds:
            X = np.column_stack([np.ones(len(s))] + [s[c].values for c in confounds])
            x = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
            y = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        r, p = pearsonr(x, y)
        rows.append({"subject": subj, "n": len(s), "r": r, "p": p})
    ps = pd.DataFrame(rows)
    rs = ps["r"].values if len(ps) else np.array([])
    if len(rs) >= 2:
        t, p = ttest_1samp(rs, 0, alternative=alternative)
        sem = rs.std(ddof=1) / np.sqrt(len(rs))
    else:
        t = p = sem = np.nan
    return {
        "mean_r": float(np.mean(rs)) if len(rs) else np.nan,
        "sem_r": float(sem) if len(rs) else np.nan,
        "t": float(t) if len(rs) >= 2 else np.nan,
        "p": float(p) if len(rs) >= 2 else np.nan,
        "n_subjects": len(rs),
        "per_subject": ps,
    }


def link_pooled(df, confounds=CONFOUNDS, n_boot=2000, maze_names=None, **filters):
    """SECONDARY test: pooled standardised effect via within-subject-demeaned OLS
    (`place ~ dist + confounds`, subject fixed effects) with a per-subject
    normal-equation cluster bootstrap. Returns dict(beta, ci_low, ci_high, p, n,
    n_subjects)."""
    confounds = confounds or []  # None → no-confound pooled model (mirrors link_per_subject)
    tidy = build_tidy(_apply_filters(df, maze_names=maze_names, **filters))
    tidy = tidy.dropna(subset=["place", "dist"] + confounds)
    if tidy["subject"].nunique() < 2 or len(tidy) < 50:
        return {"beta": np.nan, "p": np.nan, "n": len(tidy), "n_subjects": tidy["subject"].nunique()}
    tidy = _zscore(tidy, ["place", "dist"] + confounds)
    for c in ["place", "dist"] + confounds:
        tidy[c] = tidy[c] - tidy.groupby("subject")[c].transform("mean")
    preds = ["dist"] + confounds  # target "dist" at design col 1
    subs = tidy["subject"].unique()
    NE = {}
    for su in subs:
        s = tidy[tidy["subject"] == su]
        X = np.column_stack([np.ones(len(s))] + [s[p].values for p in preds])
        NE[su] = (X.T @ X, X.T @ s["place"].values)

    def _beta(counts):
        A = sum(c * NE[su][0] for su, c in counts.items())
        b = sum(c * NE[su][1] for su, c in counts.items())
        try:
            return float(np.linalg.solve(A, b)[1])
        except np.linalg.LinAlgError:
            return np.nan

    beta = _beta({su: 1 for su in subs})
    rng = np.random.default_rng(0)
    boot = np.array([_beta(Counter(rng.choice(subs, size=len(subs), replace=True))) for _ in range(n_boot)])
    boot = boot[~np.isnan(boot)]
    ci = np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan)
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean()) if len(boot) else np.nan
    return {
        "beta": beta,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p": float(p),
        "n": len(tidy),
        "n_subjects": len(subs),
    }


def plot_link(
    df, confounds=CONFOUNDS, alternative="two-sided", save_path=None, ax=None, color="C0", maze_names=None, **filters
):
    """Per-subject correlation dots + cross-subject mean ± SEM, with the t-test in
    the title. Returns (ax, result)."""
    res = link_per_subject(df, confounds=confounds, alternative=alternative, maze_names=maze_names, **filters)
    rs = res["per_subject"]["r"].dropna().values if len(res["per_subject"]) else np.array([])
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(2, 2.4))
    ax.axhline(0, color="k", ls="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    dot_x = -0.2 + np.linspace(-0.07, 0.07, max(len(rs), 1))
    ax.scatter(dot_x, rs, color="grey", alpha=0.7, s=30, edgecolors="none", zorder=3)
    if len(rs) >= 2:
        ax.errorbar(
            [0.2],
            [rs.mean()],
            yerr=[rs.std(ddof=1) / np.sqrt(len(rs))],
            fmt="o",
            color=color,
            markersize=8,
            capsize=0,
            elinewidth=2.5,
            zorder=2,
        )
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylabel("corr  distance_update ↔ place_update")
    ax.set_title(f"t({res['n_subjects']})={res['t']:+.2f}  p={res['p']:.3g}", fontsize=8)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved: {save_path}")
    return ax, res


def run_link_analysis(df, confounds=CONFOUNDS, alternative="two-sided", save_fig=None, verbose=True, **filters):
    """End-to-end Stage-4 driver: take the output of `get_catch_update2_df`
    (experiment-level, all sessions) and run it through the remaining pipeline —
    the per-subject partial-correlation t-test (primary) and the pooled
    cluster-bootstrap OLS (secondary), with an optional saved plot.

    `**filters` are forwarded to `_apply_filters` (maze_names, min_amplitude,
    max_decoder_mae, min_n_place_neurons, min_n_distance_neurons, max_distance).
    Auto-maps the older parquet's metric column names so it works on both the
    current schema and a pre-rename `summary_df`.

    Phase-permutation null: if `df` carries a `("qc","permutation")` column (from a
    `n_permutations` run), the true run is `permutation==0` and all reported true stats
    use only that slice. The per-subject link test is then re-run on every permutation,
    the per-subject r's are averaged across permutations into a per-subject null, and the
    true effect is compared to it with a paired across-subject t-test of (true − null) r
    plus a permutation p-value on the cross-subject mean r (see `_link_permutation_test`).

    Returns {"per_subject", "pooled", "permutation"} (the last is None when no
    permutations are present).
    """
    # back-compat: map old metric column names → current scheme
    ren = {}
    if ("distance", "update") not in df.columns and ("decoder", "D3_post_js") in df.columns:
        ren[("decoder", "D3_post_js")] = ("distance", "update")
    if ("place", "update") not in df.columns and ("place_update", "P4_post_js") in df.columns:
        ren[("place_update", "P4_post_js")] = ("place", "update")
    if ren:
        df = df.copy()
        df.columns = pd.MultiIndex.from_tuples([ren.get(c, c) for c in df.columns])

    # split off the phase-permutation null if present (true run = permutation 0)
    has_perm = ("qc", "permutation") in df.columns
    perm_col = df[("qc", "permutation")] if has_perm else None
    true_df = df[perm_col == 0] if has_perm else df
    null_perm_ids = sorted({int(p) for p in perm_col.unique() if p != 0}) if has_perm else []

    ps = link_per_subject(true_df, confounds=confounds, alternative=alternative, **filters)
    pooled = link_pooled(true_df, confounds=confounds, **filters)
    perm = (
        _link_permutation_test(df, perm_col, null_perm_ids, ps, confounds, alternative, **filters)
        if null_perm_ids
        else None
    )

    if verbose:
        print("=== link: distance_update ↔ place_update ===")
        if len(ps["per_subject"]):
            print(ps["per_subject"].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        print(
            f"\ncross-subject t-test (n={ps['n_subjects']}, alt={alternative}): "
            f"mean_r={ps['mean_r']:+.4f} ± {ps['sem_r']:.4f}  t={ps['t']:+.3f}  p={ps['p']:.4g}"
        )
        print(
            f"pooled OLS:  beta={pooled['beta']:+.4f}  "
            f"CI[{pooled['ci_low']:+.3f}, {pooled['ci_high']:+.3f}]  p={pooled['p']:.4g}  "
            f"(n={pooled['n']} cycles, {pooled['n_subjects']} subjects)"
        )
        if perm is not None:
            print(f"\n=== phase-permutation null ({perm['n_permutations']} permutation(s)) ===")
            print(perm["subject_r"].to_string(float_format=lambda x: f"{x:+.4f}"))
            print(
                f"cross-subject mean_r:  true={perm['true_mean_r']:+.4f}  "
                f"null={perm['null_mean_r']:+.4f}  (Δ={perm['true_mean_r'] - perm['null_mean_r']:+.4f})"
            )
            print(
                f"paired across-subject t-test (true − null, alt={alternative}): "
                f"t={perm['t']:+.3f}  p={perm['p']:.4g}  (n={perm['n_subjects']} subjects)"
            )
            if perm["perm_p"] is not None:
                print(
                    f"permutation p (true mean_r vs {perm['n_permutations']} null mean_r, one-sided): "
                    f"p={perm['perm_p']:.4g}"
                    + ("   [needs more permutations to be informative]" if perm["n_permutations"] < 20 else "")
                )

    if save_fig:
        plot_link(true_df, confounds=confounds, alternative=alternative, save_path=save_fig, **filters)
    return {"per_subject": ps, "pooled": pooled, "permutation": perm}


def _link_permutation_test(df, perm_col, null_perm_ids, true_ps, confounds, alternative, **filters):
    """Compare the true per-subject link correlations to a theta-phase-permutation null.

    For each permutation `s`, `link_per_subject` is re-run on that permutation's cycles
    to get per-subject r's; these are averaged per subject across permutations to form a
    per-subject null r. The true effect is then compared two ways:
      * `t`/`p`: paired across-subject one-sample t-test of (r_true − r_null) (respects
        the per-subject pairing; informative even for a single permutation).
      * `perm_p`: one-sided permutation p-value of the true cross-subject mean r against
        the distribution of per-permutation cross-subject mean r's,
        `(#{perm mean_r ≥ true mean_r} + 1) / (n_permutations + 1)` (only meaningful once
        many permutations exist).
    Returns a dict; `subject_r` is a per-subject DataFrame [r_true, r_null, diff].
    """
    null_r, perm_mean_r = {}, []
    for s in null_perm_ids:
        res_s = link_per_subject(df[perm_col == s], confounds=confounds, alternative=alternative, **filters)
        perm_mean_r.append(res_s["mean_r"])
        for _, row in res_s["per_subject"].iterrows():
            null_r.setdefault(row["subject"], []).append(row["r"])

    true_r = true_ps["per_subject"].set_index("subject")["r"] if len(true_ps["per_subject"]) else pd.Series(dtype=float)
    tbl = pd.DataFrame(
        {"r_true": true_r, "r_null": pd.Series({k: np.mean(v) for k, v in null_r.items()})}
    ).dropna()
    tbl["diff"] = tbl["r_true"] - tbl["r_null"]

    if len(tbl) >= 2:
        t, p = ttest_1samp(tbl["diff"].values, 0, alternative=alternative)
    else:
        t = p = np.nan

    perm_mean_r = np.asarray(perm_mean_r, dtype=float)
    perm_mean_r = perm_mean_r[~np.isnan(perm_mean_r)]
    true_mean_r = float(tbl["r_true"].mean()) if len(tbl) else np.nan
    perm_p = (np.sum(perm_mean_r >= true_mean_r) + 1) / (len(perm_mean_r) + 1) if len(perm_mean_r) else None

    return {
        "n_permutations": len(null_perm_ids),
        "n_subjects": len(tbl),
        "subject_r": tbl,
        "true_mean_r": true_mean_r,
        "null_mean_r": float(tbl["r_null"].mean()) if len(tbl) else np.nan,
        "t": float(t) if not np.isnan(t) else np.nan,
        "p": float(p) if not np.isnan(p) else np.nan,
        "perm_mean_r": perm_mean_r,
        "perm_p": float(perm_p) if perm_p is not None else None,
    }


# %% Single-session diagnostic helper


def test_session(session, cpd_scores=None, **kwargs):
    """Run one session and print a quick schema / sanity diagnostic."""
    df = get_session_catch_update2_df(session, cpd_scores=cpd_scores, verbose=True, **kwargs)
    print(f"\n{session.name}  rows={len(df)}")
    for col, m in [(("distance", "update"), "distance_update"), (("place", "update"), "place_update")]:
        v = df[col].astype(float)
        print(f"  {m:16s} mean={v.mean():+.4f}  std={v.std():.4f}  nan={v.isna().mean():.1%}")
    return df


# %% Stage 5 — general sliding-window cross-code update (lag × lag scan)
#
# Generalises the single-cell catch_update2 test (distance within-cycle peak→trough vs
# place across-cycle) into a full lag × lag scan. For each code ONE decoder is trained on
# width-`window_width` sliding theta-phase windows — every phase position × cycle of the
# training trials, pooled, labelled by the code value at the window — and read out at
# EVERY sliding window of the held-out cycles. Training and readout therefore use the same
# k-bin window features, and the decoder is held fixed across readout positions, so the
# JS² "update" stays a clean measure of how much the decoded posterior moved (not a
# decoder-vs-decoder artefact). The update at SIGNED lag L is JS²(posterior at anchor,
# posterior L bins away) in phase unrolled from one cycle before the anchor to one cycle after
# (`span_cycles` cycles each way); distance- and place-update lag profiles are then correlated
# across codes (per-subject partial correlation → cross-subject t-test) into a signed
# (lag_distance × lag_place) lead/lag coupling matrix. The original catch_update2 result is a
# single entry of that matrix. Reuses the catch_update backbone (`_cycle_subset_rates`,
# `_split_fold`) and `get_input_data` unchanged.

# code → (label column in input_data, key into the pools dict from get_input_data)
CODE_LABELS = {"distance": ("distance_bin_id", ""), "place": ("maze_position", "simple")}

# per-(cycle, start) covariates carried for the link tests (hd expanded to sin/cos in the matrix)
SLIDING_CONFOUNDS = ["speed", "theta_amp", "n_sp_place", "n_sp_dist", "dist2goal", "hd_sin", "hd_cos"]

# raw confound columns pulled from input_data at the anchor (cycle, start) row
_SLIDING_CONF_COLS = {
    "speed": ("speed", ""),
    "theta_amp": ("cycle_metrics", "amplitude"),
    "n_sp_place": ("n_spikes", "place_pool"),
    "n_sp_dist": ("n_spikes", "distance_pool"),
    "dist2goal": ("distance_to_goal", "geodesic"),
    "hd": ("head_direction", "value"),
}


def _sliding_window_bins(p, k, n_bins):
    """The k consecutive phase bins of the window starting at bin p, wrapping past
    `n_bins-1` into the next cycle (e.g. p=10, k=3, n_bins=12 → [10, 11, 0]). The wrap is
    exactly the descending-step form `_cycle_subset_rates` reads as 'borrow bin 0 of the
    successor cycle', so a late-cycle window correctly straddles the cycle boundary and is
    attributed to the starting cycle."""
    return [(p + j) % n_bins for j in range(k)]


def _js2_rows(A, B):
    """Row-wise squared Jensen–Shannon divergence (natural log), the vectorised twin of
    `_js2`: for posteriors from the SAME decoder, `_js2_rows(A, B)[i] == _js2(A[i], B[i])`.
    A, B are (m, n_classes) posterior arrays. Identical rows → genuine 0 (tiny negatives
    from floating point are clipped)."""
    M = 0.5 * (A + B)
    with np.errstate(divide="ignore", invalid="ignore"):
        ka = np.where(A > 0, A * np.log(A / M), 0.0).sum(axis=1)
        kb = np.where(B > 0, B * np.log(B / M), 0.0).sum(axis=1)
    return np.clip(0.5 * ka + 0.5 * kb, 0.0, None)


def _train_code_decoder(train_df, cells, label_col, n_bins, window_width, normalise_X, C):
    """Train ONE decoder for a code on the pooled width-`window_width` sliding windows of the
    training trials. Each (cycle, window-start p) contributes one firing-rate sample
    (sum spikes / sum durations over the window's bins, via `_cycle_subset_rates`) labelled
    by the code value at the window-start bin. Labels are stringified so distance bin ids and
    maze positions share one path; `nan` labels are dropped. Returns (decoder, scaler) or
    (None, None) when fewer than two classes survive."""
    X_parts, y_parts = [], []
    for p in range(n_bins):
        rates_p = cu._cycle_subset_rates(train_df, cells, _sliding_window_bins(p, window_width, n_bins))
        if rates_p.empty:
            continue
        raw = train_df.xs(p, level="phase_bin")[label_col].reindex(rates_p.index)
        valid = raw.notna().values  # filter NA on the pre-cast series (Int64 NA → "<NA>" once stringified)
        if valid.sum() == 0:
            continue
        X_parts.append(rates_p.values[valid])
        y_parts.append(raw.astype(str).values[valid])
    if not X_parts:
        return None, None
    X, y = np.vstack(X_parts), np.concatenate(y_parts)
    if len(np.unique(y)) < 2:
        return None, None
    scaler = StandardScaler().fit(X) if normalise_X else None
    Xs = scaler.transform(X) if scaler is not None else X
    lr_kwargs = {"random_state": 0, "max_iter": 2000, "class_weight": "balanced"}
    lr_kwargs["penalty" if C is None else "C"] = None if C is None else C
    return LogisticRegression(**lr_kwargs).fit(Xs, y), scaler


def _code_window_posteriors(test_df, cells, decoder, scaler, n_bins, window_width):
    """Decode the held-out cycles at every sliding window. Returns a dict
    p → DataFrame(predict_proba, index=cycle_idx) for p in 0..n_bins-1; columns are the
    decoder's classes (shared across p, so JS² between any two windows is well defined).
    Empty windows map to an empty DataFrame."""
    out = {}
    for p in range(n_bins):
        rates_p = cu._cycle_subset_rates(test_df, cells, _sliding_window_bins(p, window_width, n_bins))
        if rates_p.empty:
            out[p] = pd.DataFrame()
            continue
        Xs = scaler.transform(rates_p.values) if scaler is not None else rates_p.values
        out[p] = pd.DataFrame(decoder.predict_proba(Xs), index=rates_p.index)
    return out


def _sliding_update_table(P_code, s, lag_range, n_bins, trial_of, prefix):
    """For one code and one anchor start bin `s`, the per-cycle update at each lag.

    The update at lag L compares the window at (anchor cycle c, start s) to the window L bins
    later in unrolled phase — bin (s+L) % n_bins of cycle c + (s+L) // n_bins — keeping only
    pairs whose two cycles fall in the same trial. Returns a DataFrame indexed by anchor
    cycle with columns `{prefix}_L{L}` (JS² update); missing lags are simply absent columns.
    """
    base = P_code[s]
    cols = {}
    if not base.empty:
        for L in lag_range:
            q, dc = (s + L) % n_bins, (s + L) // n_bins
            tgt = P_code[q]
            if tgt.empty:
                continue
            common = base.index.intersection(tgt.index - dc).values  # anchor cycles c with c+dc decoded
            if common.size == 0:
                continue
            same_trial = trial_of.reindex(common).values == trial_of.reindex(common + dc).values
            keep = common[same_trial]
            if keep.size == 0:
                continue
            js2 = _js2_rows(base.loc[keep].values, tgt.loc[keep + dc].values)
            cols[f"{prefix}_L{L}"] = pd.Series(js2, index=keep)
    return pd.DataFrame(cols)


def get_session_sliding_update_df(
    session,
    window_width=3,
    span_cycles=1,
    normalise_X=True,
    C=1.0,
    n_folds=10,
    cv_seed=0,
    cpd_scores=None,
    verbose=False,
    **input_data_kwargs,
):
    """One row per (cycle, anchor start bin) with the distance- and place-update at every lag.

    Trial CV; per fold each code's fixed decoder is trained on sliding windows of the training
    trials (`_train_code_decoder`) and read out at every sliding window of the held-out cycles
    (`_code_window_posteriors`). For each anchor start bin s the per-cycle update at signed lags
    L = ±1..±span_cycles*n_bins is computed for both codes (`_sliding_update_table`) and merged
    on the shared anchor cycles, with confounds + metadata attached at (cycle, s). `span_cycles`
    is the number of theta cycles spanned in EACH direction, so lags run from one cycle before
    the anchor (L<0) to one cycle after (L>0) at the default span_cycles=1; this makes the
    `link_matrix` a lead/lag map rather than a forward-only one. Columns: `distance_L{L}` /
    `place_L{L}` (JS² updates, L signed), the `_SLIDING_CONF_COLS` confounds, and
    cycle / start / fold / subject / session / maze. Feed to `get_sliding_update_df` /
    `link_matrix`.
    """
    input_data, pools = get_input_data(session, cpd_scores=cpd_scores, verbose=verbose, **input_data_kwargs)
    n_bins = int(input_data.index.get_level_values("phase_bin").max()) + 1
    max_lag = span_cycles * n_bins
    lag_range = [L for L in range(-max_lag, max_lag + 1) if L != 0]  # signed, excludes the trivial L=0
    cells_by_code = {"distance": pools["distance"], "place": pools["place"]}

    trials = input_data.trial.dropna().unique()
    if n_folds == -1:
        folds = [np.array([t]) for t in trials]
    elif n_folds > 0:
        if n_folds > len(trials):
            raise ValueError(f"n_folds={n_folds} > n_trials={len(trials)} for {session.name}")
        rng = np.random.default_rng(cv_seed)
        folds = np.array_split(rng.permutation(trials), n_folds)
    else:
        raise ValueError(f"n_folds must be -1 or positive, got {n_folds!r}")

    out_rows = []
    for fold_idx, held_out in enumerate(folds):
        train_df, test_df = cu._split_fold(input_data, held_out)
        trial_of = test_df.trial.groupby(level="cycle_idx").first()

        P = {}
        usable = True
        for code, cells in cells_by_code.items():
            decoder, scaler = _train_code_decoder(
                train_df, cells, CODE_LABELS[code], n_bins, window_width, normalise_X, C
            )
            if decoder is None:
                usable = False
                break
            P[code] = _code_window_posteriors(test_df, cells, decoder, scaler, n_bins, window_width)
        if not usable:
            continue

        for s in range(n_bins):
            dist_tab = _sliding_update_table(P["distance"], s, lag_range, n_bins, trial_of, "distance")
            place_tab = _sliding_update_table(P["place"], s, lag_range, n_bins, trial_of, "place")
            if dist_tab.empty or place_tab.empty:
                continue
            common_cyc = dist_tab.index.intersection(place_tab.index)
            if len(common_cyc) == 0:
                continue
            block = pd.concat([dist_tab.loc[common_cyc], place_tab.loc[common_cyc]], axis=1)
            conf_s = test_df.xs(s, level="phase_bin")
            for name, col in _SLIDING_CONF_COLS.items():
                block[name] = conf_s[col].reindex(common_cyc).astype(float).values
            block["cycle"] = np.asarray(common_cyc)
            block["start"] = s
            block["fold"] = fold_idx
            out_rows.append(block.reset_index(drop=True))

    if not out_rows:
        raise ValueError(f"{session.name}: no usable sliding-update observations")
    df = pd.concat(out_rows, ignore_index=True)
    df["subject"], df["session"], df["maze"] = session.subject_ID, session.name, session.maze_name
    for code in cells_by_code:  # guarantee a full, contiguous lag grid even if some lags never filled
        for L in lag_range:
            if f"{code}_L{L}" not in df.columns:
                df[f"{code}_L{L}"] = np.nan
    if verbose:
        print(f"{session.name}: {len(df)} (cycle,start) rows, signed lags ±1..±{max_lag}")
    return df


# %% Stage 5 — cross-session runner


def get_sliding_update_df(save=False, verbose=True, n_jobs=3, tag=None, **session_kwargs):
    """Run `get_session_sliding_update_df` across every subject × maze × late day and
    concatenate to one cross-session dataframe, cached under
    `RESULTS_DIR/runs/sliding_update_df{tag}.parquet`. `**session_kwargs` (window_width,
    span_cycles, C, n_folds, …) are forwarded per session. Per-session failures are swallowed
    (printed if verbose). Mirrors `get_catch_update2_df`."""
    save_path = RESULTS_DIR / "runs" / f"sliding_update_df{tag or ''}.parquet"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists() and not save:
        if verbose:
            print(f"loading cached: {save_path}")
        return pd.read_parquet(save_path)

    cpd_scores = get_population_cpd_scores()

    def _process(session):
        if verbose:
            print(session.name)
        try:
            return get_session_sliding_update_df(session, cpd_scores=cpd_scores, verbose=False, **session_kwargs)
        except Exception as e:
            if verbose:
                print(f"  error on {session.name}: {e}")
            return None

    all_dfs = []
    for subject in SUBJECT_IDS:
        for maze_name in MAZE_NAMES:
            try:
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
            except FileNotFoundError:
                if verbose:
                    print(f"  no sessions for {subject} / {maze_name}")
                continue
            if not isinstance(sessions, list):
                sessions = [sessions]
            session_dfs = Parallel(n_jobs=n_jobs)(delayed(_process)(s) for s in sessions)
            all_dfs.extend([d for d in session_dfs if d is not None])

    pop_df = pd.concat(all_dfs, ignore_index=True)
    if save:
        pop_df.to_parquet(save_path)
        if verbose:
            print(f"saved: {save_path}  ({len(pop_df)} rows)")
    return pop_df


# %% Stage 5 — lag × lag link matrix


def _infer_lags(df):
    """Sorted list of lags L present as `distance_L{L}` columns."""
    return sorted(int(c.split("_L")[1]) for c in df.columns if c.startswith("distance_L"))


def link_matrix(df, confounds=SLIDING_CONFOUNDS, maze_names=None, max_distance=None, min_obs=10):
    """PRIMARY general test: the (lag_distance × lag_place) coupling matrix.

    For each (L_dist, L_place) pair: within each subject, partial Pearson r between
    `distance_L{L_dist}` and `place_L{L_place}` (confounds regressed out of both), then a
    cross-subject one-sample t-test on the subject r's — exactly the `link_per_subject`
    machinery, evaluated at every cell. The original catch_update2 result is one cell
    (distance ≈ half-cycle lag, place ≈ +1-cycle lag).

    Returns dict(lags, mean_r, t, p [each (nL, nL), row=distance lag, col=place lag],
    n_subjects, subjects, r_stack [n_subjects, nL, nL]).
    """
    df = df.copy()
    df["hd_sin"] = np.sin(np.deg2rad(df["hd"].astype(float)))
    df["hd_cos"] = np.cos(np.deg2rad(df["hd"].astype(float)))
    if maze_names is not None:
        df = df[df["maze"].isin(maze_names)]
    if max_distance is not None:
        df = df[df["dist2goal"] <= max_distance]

    lags = _infer_lags(df)
    nL = len(lags)
    subjects = list(df["subject"].unique())
    r_stack = np.full((len(subjects), nL, nL), np.nan)

    for si, subj in enumerate(subjects):
        s = df[df["subject"] == subj]
        conf = confounds or []
        for i, Ld in enumerate(lags):
            xcol = f"distance_L{Ld}"
            for j, Lp in enumerate(lags):
                ycol = f"place_L{Lp}"
                sub = s[[xcol, ycol] + conf].dropna()
                if len(sub) < min_obs:
                    continue
                x, y = sub[xcol].values.astype(float), sub[ycol].values.astype(float)
                if conf:
                    X = np.column_stack([np.ones(len(sub))] + [sub[c].values.astype(float) for c in conf])
                    x = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
                    y = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
                if np.std(x) == 0 or np.std(y) == 0:
                    continue
                r_stack[si, i, j] = pearsonr(x, y)[0]

    mean_r = np.full((nL, nL), np.nan)
    t_mat = np.full((nL, nL), np.nan)
    p_mat = np.full((nL, nL), np.nan)
    for i in range(nL):
        for j in range(nL):
            rs = r_stack[:, i, j]
            rs = rs[~np.isnan(rs)]
            if len(rs) >= 2:
                mean_r[i, j] = rs.mean()
                t_mat[i, j], p_mat[i, j] = ttest_1samp(rs, 0)
    return {
        "lags": lags,
        "mean_r": mean_r,
        "t": t_mat,
        "p": p_mat,
        "n_subjects": len(subjects),
        "subjects": subjects,
        "r_stack": r_stack,
    }


def _pi_label(m):
    """Axis label for an integer multiple m of π: 0 → '0', ±1 → '±π', else 'mπ'."""
    return {0: "0", 1: "π", -1: "−π"}.get(m, f"{m}π")


def _draw_lag_matrix(ax, R, lags, n_bins, diverging=True, cmap=None, cbar_label="partial r", mark_cell=None):
    """Render a (distance lag × place lag) coupling matrix on `ax` with theta-phase-radian axes
    (x = distance update lag, y = place update lag; `n_bins` bins = 2π).

    `R` is indexed (distance lag, place lag) on the sorted SIGNED `lags`. A blank row/col is
    inserted at lag 0 so the signed lags map to their true radian positions (and the blank cross
    marks the anchor); solid lines mark the anchor (0), dotted lines whole-cycle lags (±2π …).
    With negative lags the off-diagonal quadrants read as lead/lag: upper-left (distance lag < 0,
    place lag > 0) = distance leads place; lower-right = place leads distance. `mark_cell` is in
    BINS. Returns the image handle."""
    dphi = 2 * np.pi / n_bins
    full = list(range(lags[0], lags[-1] + 1))  # contiguous lags incl. 0
    idx = {L: i for i, L in enumerate(full)}
    D = np.full((len(full), len(full)), np.nan)  # D[place, distance]; lag-0 row/col stays NaN
    for i, Ld in enumerate(lags):
        for j, Lp in enumerate(lags):
            D[idx[Lp], idx[Ld]] = R[i, j]
    lo, hi = (full[0] - 0.5) * dphi, (full[-1] + 0.5) * dphi
    if diverging:
        vlim = np.nanmax(np.abs(D)) or 1.0
        im = ax.imshow(D, origin="lower", cmap=cmap or "RdBu_r", vmin=-vlim, vmax=vlim,
                       extent=[lo, hi, lo, hi], aspect="auto")
    else:
        im = ax.imshow(D, origin="lower", cmap=cmap or "viridis_r", extent=[lo, hi, lo, hi], aspect="auto")
    ax.set_xlabel("distance update lag (rad)")
    ax.set_ylabel("place update lag (rad)")
    ax.figure.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    n_pi = int(hi // np.pi)
    ms = list(range(-n_pi, n_pi + 1))
    ax.set_xticks([m * np.pi for m in ms])
    ax.set_xticklabels([_pi_label(m) for m in ms])
    ax.set_yticks([m * np.pi for m in ms])
    ax.set_yticklabels([_pi_label(m) for m in ms])
    ax.axvline(0, color="k", lw=0.8, alpha=0.6)  # anchor
    ax.axhline(0, color="k", lw=0.8, alpha=0.6)
    for k in range(1, lags[-1] // n_bins + 1):  # whole-cycle lags at ±k·2π
        for v in (k * 2 * np.pi, -k * 2 * np.pi):
            ax.axvline(v, color="k", ls=":", lw=0.8, alpha=0.6)
            ax.axhline(v, color="k", ls=":", lw=0.8, alpha=0.6)
    if mark_cell is not None:
        ax.scatter([mark_cell[0] * dphi], [mark_cell[1] * dphi], marker="*", s=110,
                   facecolor="none", edgecolor="k", lw=1.2, zorder=5)
    return im


def plot_link_matrix(df, value="mean_r", n_bins=12, mark_cell=None, save_path=None, ax=None, **matrix_kwargs):
    """Heatmap of the lag × lag coupling matrix (x = distance update lag, y = place update lag,
    in theta-phase radians). `value` ∈ {"mean_r", "t", "p"}. Solid lines mark the anchor (0),
    dotted lines whole-cycle lags (±2π); negative-lag quadrants read as lead/lag (see
    `_draw_lag_matrix`). Pass `mark_cell=(distance_lag, place_lag)` (BINS) to star a cell.
    Returns (ax, result-from-`link_matrix`)."""
    res = link_matrix(df, **matrix_kwargs)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4.7, 4))
    _draw_lag_matrix(ax, res[value], res["lags"], n_bins, diverging=(value != "p"),
                     cbar_label=value, mark_cell=mark_cell)
    ax.set_title(f"distance↔place update coupling ({value}, n={res['n_subjects']})", fontsize=8)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved: {save_path}")
    return ax, res


def plot_session_sliding_update(
    df, n_bins=12, confounds=SLIDING_CONFOUNDS, mark_cell=None, save_path=None, axes=None
):
    """Session-level intuition plot for one `get_session_sliding_update_df` output: the
    session's OWN lag × lag distance↔place partial-correlation matrix (x = distance update lag,
    y = place update lag, both in theta-phase radians; n_bins bins = 2π) — exactly the
    per-subject layer that `link_matrix` averages across subjects. Solid lines mark the anchor
    (0), dotted lines whole-cycle lags (±2π); the negative-lag quadrants read as lead/lag
    (distance-leads-place upper-left, place-leads-distance lower-right — see `_draw_lag_matrix`).

    Pass `mark_cell=(distance_lag, place_lag)` (lags in BINS) to star that cell and append a
    second panel: the raw per-(cycle, start) scatter behind it (distance vs place update),
    annotated with raw and partial r, so a heatmap entry maps to concrete points. Off by default
    so the summary shows no arbitrary cell.

    Note: a single session is one subject, so this is descriptive (no cross-subject t-test).
    Returns (axes, `link_matrix` result).
    """
    lags = _infer_lags(df)
    res = link_matrix(df, confounds=confounds)
    R = np.nanmean(res["r_stack"], axis=0)  # (lag_distance, lag_place); session df = one subject
    dphi = 2 * np.pi / n_bins

    n_panels = 2 if mark_cell is not None else 1
    if axes is None:
        _, axes = plt.subplots(1, n_panels, figsize=(4.7 * n_panels, 4))
    axes = np.atleast_1d(axes)
    axB = axes[0]
    axC = axes[1] if mark_cell is not None else None

    _draw_lag_matrix(axB, R, lags, n_bins, mark_cell=mark_cell)
    axB.set_title("lag × lag coupling (session)", fontsize=9)

    # optional — raw scatter behind the marked cell (update magnitudes, not lags)
    if mark_cell is not None:
        Ld, Lp = mark_cell
        xcol, ycol = f"distance_L{Ld}", f"place_L{Lp}"
        pts = df[[xcol, ycol]].dropna()
        axC.hexbin(pts[xcol], pts[ycol], gridsize=40, cmap="magma", mincnt=1, bins="log")
        r_raw = pearsonr(pts[xcol], pts[ycol])[0] if len(pts) > 2 else np.nan
        r_part = R[lags.index(Ld), lags.index(Lp)]
        axC.set_xlabel(f"distance update (lag {Ld * dphi / np.pi:.2g}π)")
        axC.set_ylabel(f"place update (lag {Lp * dphi / np.pi:.2g}π)")
        axC.set_title(f"marked cell: raw r={r_raw:+.3f}, partial r={r_part:+.3f}", fontsize=9)
        axC.spines[["top", "right"]].set_visible(False)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        axB.figure.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved: {save_path}")
    return axes, res


def test_sliding_session(session, cpd_scores=None, **kwargs):
    """Run one session through the sliding-update pipeline and print a quick diagnostic."""
    df = get_session_sliding_update_df(session, cpd_scores=cpd_scores, verbose=True, **kwargs)
    lags = _infer_lags(df)
    print(f"\n{session.name}  rows={len(df)}  lags={lags[0]}..{lags[-1]}")
    for code in ("distance", "place"):
        vals = df[[f"{code}_L{L}" for L in lags]]
        print(f"  {code:9s} update  mean={vals.values[~np.isnan(vals.values)].mean():+.4f}  "
              f"nan={np.isnan(vals.values).mean():.1%}")
    return df


# %%
