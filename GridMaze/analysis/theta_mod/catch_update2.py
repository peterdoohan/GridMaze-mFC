"""
catch_update2 — first-principles, max-SNR rebuild of the catch-update analysis.

Overarching hypothesis: on a theta timescale the distance-to-goal code updates
and *drives* a subsequent update of the place-direction code (distance → place).

This module keeps the good backbone of `catch_update.py` (per-theta-cycle
resolution, disjoint neuron pools, a built-in forward lag) and rebuilds the
weak parts, computing MULTIPLE metrics in parallel so they can be compared at
the summary stage:

  * Phase (Stage 0): two methods, `hilbert` (existing) and `waveform`
    (peak/trough-interpolated, robust to non-sinusoidal theta), selectable via
    `phase_method` and compared downstream.
  * Pools (Stage 1): disjoint AND each genuinely tuned — distance pool by
    distance-to-goal CPD, place pool by place-direction CPD (was: place pool
    defined by *exclusion* of distance cells).
  * Distance-code update within a cycle (D-metrics): D1 decoder trough−peak
    (signed), D2 ridge distance-axis projection shift (signed, continuous),
    D3 peak↔trough posterior Jensen–Shannon (unsigned magnitude).
  * Place-code update c→c+1 (P-metrics): P1 cosine, P2 mean-centred correlation,
    P3 decoded-position displacement, P4 place posterior JS, P5 Mahalanobis.
    ALL place metrics are "update magnitudes" (bigger = more place-code change).
  * Confounds: per-cycle pool spike counts, speed, theta amplitude, head
    direction, distance-to-goal — carried per row for control at the summary
    stage (the per-cycle spike-count confound can otherwise fake the result).

Stages 3 (cross-session runner) and 4 (hierarchical link test + cross-lagged
directionality summary) live lower in this file.

HARD RULE: this file is the *only* place new behaviour is written. Everything
reusable is imported from existing modules and never modified.

@peterdoohan (rebuild)
"""

# %% Imports
import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine, jensenshannon

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf

from GridMaze.analysis.core import convert
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.lfp import lfp_utils as lu
from GridMaze.analysis.processing import get_lfp_aligned_spike_counts as la

from GridMaze.analysis.neGLM import load_model_sets as lms
from GridMaze.analysis.neGLM import variance_explained_null as ve

# reuse the per-cycle backbone helpers + distance decoder (import, never modify)
from GridMaze.analysis.theta_mod import catch_update as cu

from GridMaze.maze import representations as mr

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

# %% Global variables
THETA_RANGE = (7, 11)

RESULTS_DIR = RESULTS_PATH / "theta_mod" / "catch_update2"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json") as _f:
    SUBJECT_IDS = json.load(_f)

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]


# %% Stage 0 — phase estimation (two methods)


def get_theta_phase(raw_lfp, phase_method="hilbert", freq_range=THETA_RANGE, n_bins=12):
    """Return (filt_osc, phase, bin_indices) for one of two phase conventions.

    Both methods band-pass with the same Butterworth filter (`la.get_lfp_phase`
    gives the filtered trace + Hilbert phase). The convention is shared so the
    rest of the pipeline (cycle detection, peak/trough bins) is phase-method
    agnostic:
      * phase = 0 at the LFP peak, ±π at the trough;
      * a "cycle" is trough→peak→trough (the +π→−π wrap, detected by
        `cu._detect_cycle_phase_windows`).

    phase_method:
      "hilbert"  — analytic-signal phase (existing pipeline). Biased by
        non-sinusoidal theta: phase advances unevenly within a cycle.
      "waveform" — peak/trough-interpolated phase (Belluscio-style): phase is
        linear in time between detected extrema, so it is robust to waveform
        asymmetry and is implicitly per-cycle frequency-normalised.
    """
    filt_osc, hilbert_phase = la.get_lfp_phase(raw_lfp, freq_range=freq_range, N=4, return_filtered=True)
    if phase_method == "hilbert":
        phase = hilbert_phase
    elif phase_method == "waveform":
        phase = get_waveform_phase(filt_osc)
    else:
        raise ValueError(f"phase_method must be 'hilbert' or 'waveform', got {phase_method!r}")
    _, bin_indices = la.bin_lfp_phase(phase, n_bins=n_bins)
    return filt_osc, phase, bin_indices


def get_waveform_phase(filt_osc):
    """Waveform-based theta phase by linear interpolation between extrema.

    Convention matches the Hilbert phase used elsewhere: peaks → 0, troughs →
    ±π, phase increasing peak→trough (0→π) and trough→peak (−π→0). Phase is
    linear *in time* within each half-cycle, which removes the within-cycle
    phase distortion that non-sinusoidal (sawtooth) theta imposes on Hilbert
    phase. Samples before the first / after the last extremum are clamped to the
    nearest landmark phase (those edge half-cycles are dropped downstream as
    incomplete cycles anyway).
    """
    x = np.asarray(filt_osc, dtype=float)
    peaks, _ = find_peaks(x)
    troughs, _ = find_peaks(-x)
    if len(peaks) == 0 or len(troughs) == 0:
        # degenerate (flat / no theta) — return zeros so cycles fail completeness
        return np.zeros_like(x)

    idx = np.concatenate([peaks, troughs])
    is_trough = np.concatenate([np.zeros(len(peaks), bool), np.ones(len(troughs), bool)])
    order = np.argsort(idx)
    idx, is_trough = idx[order], is_trough[order]

    # enforce strict peak/trough alternation: within a same-type run keep the extreme one
    keep_idx, keep_type = _enforce_alternation(idx, is_trough, x)

    phase = np.full(x.shape, np.nan)
    for i in range(len(keep_idx) - 1):
        s, e = keep_idx[i], keep_idx[i + 1]
        if e <= s:
            continue
        if keep_type[i] and not keep_type[i + 1]:  # trough → peak : −π → 0
            p0, p1 = -np.pi, 0.0
        elif (not keep_type[i]) and keep_type[i + 1]:  # peak → trough : 0 → +π
            p0, p1 = 0.0, np.pi
        else:
            continue
        phase[s:e] = np.linspace(p0, p1, e - s, endpoint=False)
    # clamp leading / trailing NaNs
    first, last = keep_idx[0], keep_idx[-1]
    phase[:first] = -np.pi if not keep_type[0] else 0.0
    phase[last:] = phase[last - 1] if last > 0 and not np.isnan(phase[last - 1]) else 0.0
    # any residual interior NaN (shouldn't happen after alternation) → 0
    phase[np.isnan(phase)] = 0.0
    return phase


def _enforce_alternation(idx, is_trough, x):
    """Collapse consecutive same-type extrema to the single most-extreme one, so
    the landmark sequence strictly alternates peak/trough."""
    keep_idx, keep_type = [idx[0]], [is_trough[0]]
    for j in range(1, len(idx)):
        if is_trough[j] == keep_type[-1]:
            # same type as previous kept landmark — keep the more extreme
            prev = keep_idx[-1]
            if is_trough[j]:  # troughs: more negative wins
                if x[idx[j]] < x[prev]:
                    keep_idx[-1] = idx[j]
            else:  # peaks: more positive wins
                if x[idx[j]] > x[prev]:
                    keep_idx[-1] = idx[j]
        else:
            keep_idx.append(idx[j])
            keep_type.append(is_trough[j])
    return np.asarray(keep_idx), np.asarray(keep_type, bool)


# %% Tuning scores (disjoint, each genuinely tuned)


def get_population_cpd_scores(r2_thres=0.05):
    """Per-cluster CPD (%) for BOTH distance-to-goal and place-direction from the
    neGLM `variance_explained_multiunit` model set, averaged over each cluster's
    (maze, day) appearances. Returns a DataFrame indexed by `cluster_unique_ID`
    with columns ['distance_to_goal', 'place_direction'].

    Mirrors `catch_update.get_population_distance_score` but returns both
    feature columns so pools can be selected by their own tuning.
    """
    cv_scores = lms.load_model_set_cv_scores("variance_explained_multiunit")
    cpd_df = ve.get_cpd_df(cv_scores, r2_thres=r2_thres)
    return cpd_df[["distance_to_goal", "place_direction"]].groupby(level="cluster_unique_ID").mean()


def _select_pools(cells, cpd_scores, pool_method="argmax"):
    """Split `cells` (cluster_unique_IDs) into disjoint distance / place pools.

    pool_method:
      "argmax" (default) — distance pool = cells with distance-CPD > 0 AND
        distance-CPD > place-CPD; place pool = place-CPD > 0 AND place-CPD >
        distance-CPD. Cells tuned to both equally / neither are dropped. Each
        pool is genuinely tuned to its own variable and the two are disjoint.
      "median" — legacy-style median split on distance-CPD (distance pool =
        top half ∩ CPD>0; place pool = bottom half). Kept for A/B comparison.
    """
    with_cpd = [c for c in cells if c in cpd_scores.index]
    d = cpd_scores["distance_to_goal"].reindex(with_cpd)
    p = cpd_scores["place_direction"].reindex(with_cpd)
    if pool_method == "argmax":
        distance_pool = [c for c in with_cpd if d[c] > 0 and d[c] > p[c]]
        place_pool = [c for c in with_cpd if p[c] > 0 and p[c] > d[c]]
    elif pool_method == "median":
        med = float(np.median(d.values))
        distance_pool = [c for c in with_cpd if d[c] > med and d[c] > 0]
        place_pool = [c for c in with_cpd if d[c] <= med]
    else:
        raise ValueError(f"pool_method must be 'argmax' or 'median', got {pool_method!r}")
    return distance_pool, place_pool


# %% Stage 1 — per-(cycle, phase_bin) input data


def get_input_data(
    session,
    phase_method="hilbert",
    n_bins=12,
    shank=3,
    max_steps_to_goal=20,
    exclude_at_goal=True,
    bin_spacing=0.05,
    moving_only=True,
    min_firing_rate=0.5,
    pool_method="argmax",
    cpd_scores=None,
    verbose=False,
):
    """Build the per-(cycle, phase_bin) dataframe and the disjoint neuron pools.

    Returns (input_data, pools) where pools = {"distance": [...], "place": [...]}.
    `input_data` keeps `spike_count` columns for BOTH pools (the place pool is
    needed for the place-update metrics), plus navigation, `phase_window`,
    `cycle_metrics` (amplitude), distance binning, per-cycle pool spike-count
    confounds, and subject/maze metadata.

    Backbone (cycle/phase-window detection, navigation alignment, row filters,
    complete-cycle drop, distance binning) is reused from `catch_update`.
    """
    # --- LFP, theta phase (chosen method), phase bins ---
    raw_lfp = lu.get_LFP(session, shank=shank)
    filt_osc, _phase, bin_indices = get_theta_phase(raw_lfp, phase_method=phase_method, n_bins=n_bins)
    lfp_times = session.lfp_times

    # --- detect (cycle, phase_bin) windows ---
    start_samples, end_samples = cu._detect_cycle_phase_windows(bin_indices, n_bins=n_bins)
    n_cycles = start_samples.shape[0]
    if n_cycles == 0:
        raise ValueError(f"{session.name}: no complete theta cycles ({phase_method})")
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

    amplitudes = cu._compute_cycle_metrics(filt_osc, start_samples, end_samples)

    # --- assemble ---
    row_index = pd.MultiIndex.from_product(
        [np.arange(n_cycles), np.arange(n_bins)], names=["cycle_idx", "phase_bin"]
    )
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

    cycle_metrics_block = pd.DataFrame(
        {("cycle_metrics", "amplitude"): np.repeat(amplitudes, n_bins)}, index=row_index
    )
    cycle_metrics_block.columns = pd.MultiIndex.from_tuples(cycle_metrics_block.columns)

    spike_count_block = pd.DataFrame(
        spike_counts.T,
        index=row_index,
        columns=pd.MultiIndex.from_product([["spike_count"], cluster_unique_IDs]),
    )

    input_data = pd.concat([nav_block, phase_window_block, cycle_metrics_block, spike_count_block], axis=1)

    # --- row filters (mirror catch_update) ---
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
        raise ValueError(f"{session.name}: no rows survived row filters ({phase_method})")

    # --- neuron pools (unit filter → FR floor → CPD → tuned disjoint split) ---
    unit_pool = set(
        gc.filter_clusters(
            session.cluster_metrics, session.session_info, return_unique_IDs=True,
            single_units=True, multi_units=True,
        )
    )
    session_cells = list(input_data.spike_count.columns)
    after_unit = [c for c in session_cells if c in unit_pool]
    nav_duration = float(input_data.phase_window.duration.sum())
    spikes_per_cell = input_data.spike_count[after_unit].sum(axis=0)
    fr_per_cell = spikes_per_cell / nav_duration
    after_fr = [c for c in after_unit if fr_per_cell.get(c, 0.0) >= min_firing_rate]

    if cpd_scores is None:
        cpd_scores = get_population_cpd_scores()
    distance_pool, place_pool = _select_pools(after_fr, cpd_scores, pool_method=pool_method)
    if verbose:
        print(
            f"{session.name} [{phase_method}]: n_unit={len(after_unit)} n_fr={len(after_fr)} "
            f"n_dist={len(distance_pool)} n_place={len(place_pool)}"
        )
    if len(distance_pool) == 0 or len(place_pool) == 0:
        raise ValueError(
            f"{session.name}: empty pool (dist={len(distance_pool)}, place={len(place_pool)})"
        )

    # --- distance-to-goal binning (over the post-filter range) ---
    dist_series = input_data[("distance_to_goal", "geodesic")]
    max_distance = float(dist_series.max()) + bin_spacing * 1e-3
    min_distance = float(dist_series.min())
    n_distance_bins = int((max_distance - min_distance) / bin_spacing)
    bins = convert._get_distance_bins(
        binning_method="uniform", n_distance_bins=n_distance_bins,
        distance_metrics=("distance_to_goal", "geodesic"),
        max_distance=max_distance, min_distance=min_distance,
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

    # --- metadata columns for the experiment-level df ---
    input_data.loc[:, ("subject_ID", "")] = session.subject_ID
    input_data.loc[:, ("maze_name", "")] = session.maze_name
    input_data.loc[:, ("session_ID", "")] = session.name
    input_data.loc[:, ("phase_method", "")] = phase_method

    pools = {"distance": distance_pool, "place": place_pool}
    return input_data, pools


# %% Stage 2 — per-session metrics (the heart)


def get_session_catch_update2_df(
    session,
    phase_method="hilbert",
    distance_peak_bins=[3, 4, 5],
    distance_trough_bins=[9, 10, 11],
    place_phase_bins=[0, 1, 2],
    n_training_phases=3,
    normalise_X=True,
    C=1.0,
    n_folds=10,
    cv_seed=0,
    cpd_scores=None,
    verbose=False,
    **input_data_kwargs,
):
    """One row per test cycle with all D-metrics (within-cycle distance update)
    and P-metrics (place update c→c+1) plus confounds + meta.

    CV is over whole trials (n_folds folds; -1 → leave-one-trial-out). For each
    fold a distance decoder + ridge distance-axis + place decoder are trained on
    the training trials and applied to the held-out trials.

    D1 decoder.distance_error  = trough_pred − peak_pred       (signed, weighted readout)
    D2 axis.distance_shift      = trough_proj − peak_proj       (signed, ridge axis)
    D3 decoder.post_js          = JS(peak_posterior, trough_posterior)  (unsigned)
    P1 place.cosine             = cosine distance(v_c, v_{c+1})  (raw rates)
    P2 place.corr               = 1 − corr(v_c, v_{c+1})        (mean-centred)
    P3 place.displacement       = ||decoded_pos(c) − decoded_pos(c+1)||  (metres)
    P4 place.post_js            = JS(place_post(c), place_post(c+1))
    P5 place.mahalanobis        = Mahalanobis(v_c, v_{c+1})     (whitened)
    """
    input_data, pools = get_input_data(
        session, phase_method=phase_method, cpd_scores=cpd_scores, verbose=verbose, **input_data_kwargs
    )
    distance_cells = pools["distance"]
    place_cells = pools["place"]
    distance_bin_mids = np.array(sorted(input_data.distance_bin_mid.dropna().unique()))
    n_bins = int(input_data.index.get_level_values("phase_bin").max()) + 1
    if n_bins % n_training_phases != 0:
        raise ValueError(f"n_training_phases ({n_training_phases}) must divide n_bins ({n_bins})")

    # training-free place metrics computed once on the whole session (P1, P2, P5)
    raw_place = _compute_raw_place_updates(input_data, place_cells, place_phase_bins)

    # maze label → (x,y) position for decoded-position displacement (P3)
    label2pos = _get_label2pos(session.maze_name)

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

    per_cycle = []
    for fold_idx, held_out in enumerate(folds):
        train_df, test_df = cu._split_fold(input_data, held_out)

        # ---- distance decoder (super-phase training) ----
        rates_train, y_train = cu._super_phase_rates(train_df, distance_cells, n_training_phases)
        if rates_train.empty or y_train.nunique() < 2:
            continue
        X_train = rates_train.values
        scaler = StandardScaler().fit(X_train) if normalise_X else None
        if scaler is not None:
            X_train = scaler.transform(X_train)
        lr_kwargs = {"random_state": 0, "max_iter": 2000, "class_weight": "balanced"}
        if C is None:
            lr_kwargs["penalty"] = None
        else:
            lr_kwargs["C"] = C
        decoder = LogisticRegression(**lr_kwargs).fit(X_train, y_train.values)
        mids_for_classes = distance_bin_mids[decoder.classes_]

        # ---- ridge distance axis (TDR) on the same training rates ----
        axis = _fit_distance_axis(rates_train.values, distance_bin_mids[y_train.values])

        # ---- distance test: peak + trough rate vectors per cycle ----
        peak_rates = cu._cycle_subset_rates(test_df, distance_cells, distance_peak_bins)
        trough_rates = cu._cycle_subset_rates(test_df, distance_cells, distance_trough_bins)
        shared = peak_rates.index.intersection(trough_rates.index)
        if len(shared) == 0:
            continue
        peak_rates, trough_rates = peak_rates.loc[shared], trough_rates.loc[shared]
        Xp, Xt = peak_rates.values, trough_rates.values
        Xp_s = scaler.transform(Xp) if scaler is not None else Xp
        Xt_s = scaler.transform(Xt) if scaler is not None else Xt

        post_p = decoder.predict_proba(Xp_s)
        post_t = decoder.predict_proba(Xt_s)
        peak_pred = post_p @ mids_for_classes
        trough_pred = post_t @ mids_for_classes
        d1_signed = trough_pred - peak_pred
        d3_post_js = np.array([jensenshannon(a, b) ** 2 for a, b in zip(post_p, post_t)])
        # D2: ridge axis projection shift (use raw rates; axis already centred)
        d2_signed = (Xt - axis["mean"]) @ axis["w"] - (Xp - axis["mean"]) @ axis["w"]

        # ---- place decoder + decoder-based place metrics (P3, P4) ----
        p3p4 = _compute_decoded_place_updates(
            train_df, test_df, place_cells, place_phase_bins, normalise_X, label2pos
        )

        # ---- assemble per-test-cycle rows (bin-0 carries nav + confounds) ----
        meta = test_df.xs(0, level="phase_bin").loc[shared].copy()
        meta = meta.drop(columns=["spike_count"], level=0, errors="ignore")
        meta[("decoder", "peak_pred")] = peak_pred
        meta[("decoder", "trough_pred")] = trough_pred
        meta[("decoder", "D1_signed")] = d1_signed
        meta[("decoder", "D2_axis_signed")] = d2_signed
        meta[("decoder", "D3_post_js")] = d3_post_js
        meta[("decoder", "fold")] = fold_idx
        # place metrics keyed by cycle_idx
        meta[("place_update", "P1_cosine")] = raw_place["P1_cosine"].reindex(shared).values
        meta[("place_update", "P2_corr")] = raw_place["P2_corr"].reindex(shared).values
        meta[("place_update", "P5_mahalanobis")] = raw_place["P5_mahalanobis"].reindex(shared).values
        meta[("place_update", "P3_displacement")] = p3p4["P3_displacement"].reindex(shared).values
        meta[("place_update", "P4_post_js")] = p3p4["P4_post_js"].reindex(shared).values
        per_cycle.append(meta)

    if not per_cycle:
        raise ValueError(f"{session.name}: no usable test cycles")
    session_df = pd.concat(per_cycle).sort_index()

    # session-level decoder MAE + pool sizes
    true_dist = session_df[("distance_bin_mid", "")].values.astype(float)
    session_df[("decoder", "peak_mae")] = float(np.mean(np.abs(session_df[("decoder", "peak_pred")].values - true_dist)))
    session_df[("decoder", "trough_mae")] = float(np.mean(np.abs(session_df[("decoder", "trough_pred")].values - true_dist)))
    session_df[("decoder", "n_distance_neurons")] = len(distance_cells)
    session_df[("place_update", "n_place_neurons")] = len(place_cells)
    if verbose:
        print(f"{session.name} [{phase_method}]: peak_mae={session_df[('decoder','peak_mae')].iloc[0]:.3f} m")
    return session_df


def _fit_distance_axis(rates, distances, alpha=1.0):
    """Ridge-regression distance axis (targeted dimensionality reduction).

    Fits distance ≈ (rates − mean) · w; the unit-normalised weight vector `w` is
    the population axis most predictive of distance-to-goal. Returns dict with
    centred mean and normalised axis so peak/trough projections are comparable.
    """
    mean = rates.mean(axis=0)
    reg = Ridge(alpha=alpha, fit_intercept=True).fit(rates - mean, distances)
    w = reg.coef_
    norm = np.linalg.norm(w)
    if norm > 0:
        w = w / norm
    return {"mean": mean, "w": w}


def _compute_raw_place_updates(input_data, place_cells, place_phase_bins):
    """Training-free place-update metrics over consecutive cycles within a trial.

    ALL place metrics are "update magnitudes": bigger = more place-code change
    between cycle c and c+1 (so they all point the same way at the summary stage).
      P1_cosine      = cosine DISTANCE        (1 − cosine similarity)
      P2_corr        = correlation DISTANCE   (1 − Pearson r, mean-centred)
      P5_mahalanobis = Mahalanobis distance   (session-global Ledoit–Wolf whitening)
    Each is attributed to cycle c (= update c→c+1); NaN at trial-final cycles /
    gaps. P5's global precision is a nuisance whitening estimate (a metric, not a
    predictor), so global pooling is acceptable.
    """
    cols = ["P1_cosine", "P2_corr", "P5_mahalanobis"]
    out = pd.DataFrame(np.nan, index=input_data.index.get_level_values("cycle_idx").unique(), columns=cols)
    if not place_cells:
        return out

    # per-cycle place-window rate vectors (one row per cycle), + trial map
    sub = input_data[input_data.index.get_level_values("phase_bin").isin(place_phase_bins)]
    spikes = sub.spike_count[place_cells].groupby(level="cycle_idx").sum()
    durations = sub.phase_window.duration.groupby(level="cycle_idx").sum()
    rates = spikes.div(durations, axis=0)
    cycle_trial = input_data.trial.groupby(level="cycle_idx").first()

    # global whitening precision for P5
    try:
        precision = LedoitWolf().fit(rates.values).precision_
    except Exception:
        precision = None

    for trial, cyc_idx in cycle_trial.groupby(cycle_trial).groups.items():
        present = set(int(c) for c in cyc_idx)
        for c in sorted(present):
            if (c + 1) not in present:
                continue
            v_c = rates.loc[c].values
            v_next = rates.loc[c + 1].values
            if not v_c.any() or not v_next.any():
                continue
            out.loc[c, "P1_cosine"] = float(cosine(v_c, v_next))  # cosine distance (1 − sim)
            # P2: mean-centred (across neurons) correlation distance
            a, b = v_c - v_c.mean(), v_next - v_next.mean()
            if a.any() and b.any():
                out.loc[c, "P2_corr"] = 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            if precision is not None:
                d = v_c - v_next
                out.loc[c, "P5_mahalanobis"] = float(np.sqrt(max(d @ precision @ d, 0.0)))
    return out


def _compute_decoded_place_updates(train_df, test_df, place_cells, place_phase_bins, normalise_X, label2pos):
    """Place decoder trained on train trials, applied to held-out cycles, giving
    decoded-position displacement (P3, metres) and place-posterior JS (P4)
    between consecutive cycles within each test trial. Returns DataFrame indexed
    by cycle_idx (attributed to cycle c)."""
    out = pd.DataFrame(columns=["P3_displacement", "P4_post_js"], dtype=float)
    if not place_cells:
        return out

    # training rate vectors + position labels (one per cycle)
    tr_rates = cu._cycle_subset_rates(train_df, place_cells, place_phase_bins)
    tr_pos = train_df.xs(place_phase_bins[0], level="phase_bin")[("maze_position", "simple")]
    tr_pos = tr_pos.reindex(tr_rates.index).astype(str)
    valid = tr_pos.notna() & (tr_pos != "nan")
    tr_rates, tr_pos = tr_rates[valid.values], tr_pos[valid.values]
    if tr_rates.empty or tr_pos.nunique() < 2:
        return out

    scaler = StandardScaler().fit(tr_rates.values) if normalise_X else None
    Xtr = scaler.transform(tr_rates.values) if scaler is not None else tr_rates.values
    place_dec = LogisticRegression(
        random_state=0, max_iter=2000, class_weight="balanced", C=1.0
    ).fit(Xtr, tr_pos.values)
    classes = place_dec.classes_
    pos_matrix = np.array([label2pos.get(c, (np.nan, np.nan)) for c in classes], dtype=float)

    te_rates = cu._cycle_subset_rates(test_df, place_cells, place_phase_bins)
    if te_rates.empty:
        return out
    Xte = scaler.transform(te_rates.values) if scaler is not None else te_rates.values
    post = place_dec.predict_proba(Xte)
    decoded_pos = post @ np.nan_to_num(pos_matrix)  # posterior-weighted (x, y)
    post_df = pd.DataFrame(post, index=te_rates.index)
    pos_df = pd.DataFrame(decoded_pos, index=te_rates.index, columns=["x", "y"])

    cycle_trial = test_df.trial.groupby(level="cycle_idx").first()
    rows = {}
    for trial, cyc_idx in cycle_trial.groupby(cycle_trial).groups.items():
        present = set(int(c) for c in cyc_idx) & set(int(c) for c in te_rates.index)
        for c in sorted(present):
            if (c + 1) not in present:
                continue
            disp = float(np.linalg.norm(pos_df.loc[c].values - pos_df.loc[c + 1].values))
            js = float(jensenshannon(post_df.loc[c].values, post_df.loc[c + 1].values) ** 2)
            rows[c] = {"P3_displacement": disp, "P4_post_js": js}
    return pd.DataFrame.from_dict(rows, orient="index")


def _get_label2pos(maze_name):
    """maze_position.simple label → cartesian (x, y), covering towers AND bridges
    (via the extended simple maze that carries edge positions)."""
    simple_maze = mr.get_simple_maze(maze_name)
    ext = mr.get_extended_simple_maze(simple_maze)
    coord2pos = nx.get_node_attributes(ext, "position")
    coord2label = nx.get_node_attributes(ext, "label")
    return {coord2label[c]: coord2pos[c] for c in ext.nodes if c in coord2label and c in coord2pos}


# %% Stage 3 — cross-session runner


def get_catch_update2_df(
    save=False, verbose=True, n_jobs=6, tag=None, phase_method="hilbert", **session_kwargs
):
    """Run `get_session_catch_update2_df` across every subject × maze × late day
    and concatenate to one cross-session dataframe, cached to parquet.

    Mirrors `catch_update.get_sweep_update_df`: joblib parallel over sessions,
    per-session failures swallowed (printed if verbose). Cache file encodes the
    phase method and optional tag so robustness conditions don't clobber.
    """
    from joblib import Parallel, delayed
    from GridMaze.analysis.core import get_sessions as gs

    suffix = f"_{phase_method}" + (tag or "")
    save_path = (RESULTS_DIR / "runs" / f"catch_update2_df{suffix}.parquet")
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
            return get_session_catch_update2_df(
                session, phase_method=phase_method, cpd_scores=cpd_scores, verbose=False, **session_kwargs
            )
        except Exception as e:
            if verbose:
                print(f"  error on {session.name}: {e}")
            return None

    all_dfs = []
    for subject in SUBJECT_IDS:
        for maze_name in MAZE_NAMES:
            try:
                sessions = gs.get_maze_sessions(
                    subject_IDs=[subject], maze_names=[maze_name], days_on_maze="late",
                    with_data=[
                        "lfp_times", "lfp_signal", "lfp_metrics", "cluster_metrics",
                        "navigation_df", "spike_times", "spike_clusters",
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


# %% Stage 4 — summary statistics (the comparison stage)

DISTANCE_METRICS = ["absD1", "D1", "absD2", "D2", "D3"]
PLACE_METRICS = ["P1_cosine", "P2_corr", "P3_displacement", "P4_post_js", "P5_mahalanobis"]
_CONFOUNDS = ["speed", "theta_amp", "n_sp_place", "n_sp_dist", "dist2goal", "hd_sin", "hd_cos"]


def build_tidy(df, distance_metric, place_metric):
    """Flatten the experiment df into a tidy per-cycle frame for modelling, with
    within-trial lagged columns for the cross-lagged directionality test.

    Columns: subject, session, maze, trial, cyc, dist (distance-update),
    place (place-update c→c+1), confounds, place_into (place-update (c-1)→c),
    dist_prev (distance-update at c-1). dist/place are signed or magnitude per
    `distance_metric` (absD1/D1/absD2/D2/D3) and `place_metric`.
    """
    dmap = {
        "absD1": df[("decoder", "D1_signed")].abs(),
        "D1": df[("decoder", "D1_signed")],
        "absD2": df[("decoder", "D2_axis_signed")].abs(),
        "D2": df[("decoder", "D2_axis_signed")],
        "D3": df[("decoder", "D3_post_js")],
    }
    dist = dmap[distance_metric].astype(float).values
    place = df[("place_update", place_metric)].astype(float).values
    if ("cycle_idx", "") in df.columns:
        cyc = df[("cycle_idx", "")].values
    elif "cycle_idx" in df.columns:
        cyc = df["cycle_idx"].values
    else:
        cyc = df.index.get_level_values("cycle_idx")
    tidy = pd.DataFrame(
        {
            "subject": df[("subject_ID", "")].values,
            "session": df[("session_ID", "")].values,
            "maze": df[("maze_name", "")].values,
            "trial": df[("trial", "")].values.astype(float),
            "cyc": np.asarray(cyc, dtype=float),
            "dist": dist,
            "place": place,
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
    # within-trial lag-1 (gated to strictly consecutive cycles)
    tidy = tidy.sort_values(["session", "trial", "cyc"]).reset_index(drop=True)
    g = tidy.groupby(["session", "trial"], sort=False)
    tidy["place_into"] = g["place"].shift(1)
    tidy["dist_prev"] = g["dist"].shift(1)
    cyc_prev = g["cyc"].shift(1)
    nonconsec = (tidy["cyc"] - cyc_prev) != 1
    tidy.loc[nonconsec, ["place_into", "dist_prev"]] = np.nan
    return tidy


def _zscore(tidy, cols):
    out = tidy.copy()
    for c in cols:
        v = out[c].astype(float)
        sd = v.std()
        out[c] = (v - v.mean()) / sd if sd > 0 else 0.0
    return out


def fit_link_mixedlm(df, distance_metric="absD1", place_metric="P1_cosine", confounds=_CONFOUNDS,
                     random_slope=True, maze_names=None, **filters):
    """Principled hierarchical link test: does distance-update predict place-update?

    `place ~ dist + confounds` with a per-subject random intercept (+ random
    slope of dist if `random_slope`), z-scored so `dist`'s coefficient is a
    standardised effect size. Slower than `fit_link_ols`; used as a cross-check.
    Returns dict(beta, ci_low, ci_high, p, n, n_subjects, converged).
    """
    import statsmodels.formula.api as smf

    tidy = build_tidy(_apply_filters(df, maze_names=maze_names, **filters), distance_metric, place_metric)
    use = ["place", "dist"] + confounds
    tidy = tidy.dropna(subset=use)
    if tidy["subject"].nunique() < 2 or len(tidy) < 50:
        return {"beta": np.nan, "p": np.nan, "n": len(tidy), "n_subjects": tidy["subject"].nunique(), "converged": False}
    tidy = _zscore(tidy, ["place", "dist"] + confounds)
    formula = "place ~ dist + " + " + ".join(confounds)
    re_formula = "~dist" if random_slope else None
    try:
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            res = smf.mixedlm(formula, tidy, groups=tidy["subject"], re_formula=re_formula).fit(reml=False, method="lbfgs")
        ci = res.conf_int().loc["dist"]
        return {
            "beta": float(res.params["dist"]), "ci_low": float(ci[0]), "ci_high": float(ci[1]),
            "p": float(res.pvalues["dist"]), "n": len(tidy), "n_subjects": tidy["subject"].nunique(),
            "converged": bool(res.converged),
        }
    except Exception as e:
        return {"beta": np.nan, "p": np.nan, "n": len(tidy), "n_subjects": tidy["subject"].nunique(),
                "converged": False, "error": str(e)}


def fit_link_ols(df, distance_metric="absD1", place_metric="P1_cosine", confounds=_CONFOUNDS,
                 n_boot=2000, maze_names=None, **filters):
    """Fast primary link test (grid default): within-subject-demeaned OLS of
    `place ~ dist + confounds` with a per-subject normal-equation cluster
    bootstrap for inference. This is a legitimate hierarchical-robust estimate
    (subject fixed effects + subject-cluster resampling) that scales to the full
    cycle-level data, where `fit_link_mixedlm` (kept as a principled cross-check)
    is slow. Returns dict(beta, ci_low, ci_high, p, n, n_subjects)."""
    tidy = build_tidy(_apply_filters(df, maze_names=maze_names, **filters), distance_metric, place_metric)
    use = ["place", "dist"] + confounds
    tidy = tidy.dropna(subset=use)
    if tidy["subject"].nunique() < 2 or len(tidy) < 50:
        return {"beta": np.nan, "p": np.nan, "n": len(tidy), "n_subjects": tidy["subject"].nunique()}
    tidy = _zscore(tidy, ["place", "dist"] + confounds)
    for c in ["place", "dist"] + confounds:
        tidy[c] = tidy[c] - tidy.groupby("subject")[c].transform("mean")
    preds = ["dist"] + confounds  # target = "dist" at design col 1
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

    from collections import Counter
    beta = _beta({su: 1 for su in subs})
    rng = np.random.default_rng(0)
    boot = np.array([_beta(Counter(rng.choice(subs, size=len(subs), replace=True))) for _ in range(n_boot)])
    boot = boot[~np.isnan(boot)]
    ci = np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan)
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean()) if len(boot) else np.nan
    return {"beta": beta, "ci_low": float(ci[0]), "ci_high": float(ci[1]), "p": float(p),
            "n": len(tidy), "n_subjects": len(subs)}


def per_subject_link(df, distance_metric="absD1", place_metric="P1_cosine", regress_out=_CONFOUNDS,
                     alternative="two-sided", maze_names=None, **filters):
    """Secondary, comparability test: per-subject partial Pearson r between dist
    and place (confounds regressed out within subject), then a cross-subject
    t-test on the subject r's (n≈6). Returns dict(mean_r, sem_r, t, p, per_subject)."""
    from scipy.stats import pearsonr, ttest_1samp

    tidy = build_tidy(_apply_filters(df, maze_names=maze_names, **filters), distance_metric, place_metric)
    rows = []
    for subj, s in tidy.groupby("subject"):
        cols = ["place", "dist"] + (regress_out or [])
        s = s.dropna(subset=cols)
        if len(s) < 10:
            continue
        x, y = s["dist"].values, s["place"].values
        if regress_out:
            X = np.column_stack([np.ones(len(s))] + [s[c].values for c in regress_out])
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
    return {"mean_r": float(np.mean(rs)) if len(rs) else np.nan, "sem_r": float(sem) if len(rs) else np.nan,
            "t": float(t) if len(rs) >= 2 else np.nan, "p": float(p) if len(rs) >= 2 else np.nan,
            "n_subjects": len(rs), "per_subject": ps}


def directionality_crosslag(df, distance_metric="absD1", place_metric="P1_cosine", confounds=_CONFOUNDS,
                            n_boot=2000, maze_names=None, **filters):
    """Cross-lagged directionality test (forward vs reverse), the core of the
    distance→place claim.

      forward:  place(c→c+1) ~ dist(c)        + place_into(c)  + confounds
      reverse:  dist(c)      ~ place_into(c)   + dist_prev(c)   + confounds

    Both fitted as within-subject-demeaned OLS (subject fixed effects) on
    z-scored data, so the standardised cross-lag coefficients β_fwd (dist→next
    place, beyond place autocorr) and β_rev (place→dist, beyond dist autocorr)
    are comparable. The directionality statistic is the asymmetry β_fwd − β_rev,
    with a subject-cluster bootstrap CI. β_fwd > β_rev ⇒ distance leads place.
    """
    tidy = build_tidy(_apply_filters(df, maze_names=maze_names, **filters), distance_metric, place_metric)
    need = ["place", "dist", "place_into", "dist_prev"] + confounds
    tidy = tidy.dropna(subset=need)
    if tidy["subject"].nunique() < 2 or len(tidy) < 50:
        return {"beta_fwd": np.nan, "beta_rev": np.nan, "asymmetry": np.nan, "p_boot": np.nan, "n": len(tidy)}
    tidy = _zscore(tidy, ["place", "dist", "place_into", "dist_prev"] + confounds)
    # within-subject demean (= subject fixed effects) so betas are within-subject
    cols = ["place", "dist", "place_into", "dist_prev"] + confounds
    for c in cols:
        tidy[c] = tidy[c] - tidy.groupby("subject")[c].transform("mean")

    fwd_pred = ["dist", "place_into"] + confounds  # target = "dist" at design col 1
    rev_pred = ["place_into", "dist_prev"] + confounds  # target = "place_into" at design col 1

    # precompute per-subject normal-equation components (XᵀX, Xᵀy) once. Because the
    # subject-cluster bootstrap resamples WHOLE subjects, each draw just sums the
    # selected subjects' components and solves a small (k×k) system — O(subjects),
    # not O(rows), per draw. target coefficient is design column 1 for both models.
    subs = tidy["subject"].unique()
    NE = {}
    for su in subs:
        s = tidy[tidy["subject"] == su]
        n = len(s)
        Xf = np.column_stack([np.ones(n)] + [s[p].values for p in fwd_pred])
        Xr = np.column_stack([np.ones(n)] + [s[p].values for p in rev_pred])
        NE[su] = (Xf.T @ Xf, Xf.T @ s["place"].values, Xr.T @ Xr, Xr.T @ s["dist"].values)

    def _betas(counts):
        Af = sum(c * NE[su][0] for su, c in counts.items())
        bf = sum(c * NE[su][1] for su, c in counts.items())
        Ar = sum(c * NE[su][2] for su, c in counts.items())
        br = sum(c * NE[su][3] for su, c in counts.items())
        try:
            return float(np.linalg.solve(Af, bf)[1]), float(np.linalg.solve(Ar, br)[1])
        except np.linalg.LinAlgError:
            return np.nan, np.nan

    from collections import Counter
    beta_fwd, beta_rev = _betas({su: 1 for su in subs})
    asym = beta_fwd - beta_rev

    rng = np.random.default_rng(0)
    boot = []
    for _ in range(n_boot):
        bf, br = _betas(Counter(rng.choice(subs, size=len(subs), replace=True)))
        if not (np.isnan(bf) or np.isnan(br)):
            boot.append(bf - br)
    boot = np.array(boot)
    if len(boot):
        ci = np.percentile(boot, [2.5, 97.5])
        p_boot = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    else:
        ci, p_boot = (np.nan, np.nan), np.nan
    return {"beta_fwd": beta_fwd, "beta_rev": beta_rev, "asymmetry": asym,
            "ci_low": ci[0], "ci_high": ci[1], "p_boot": p_boot, "n": len(tidy),
            "n_subjects": len(subs)}


def _apply_filters(df, maze_names=None, min_amplitude=None, max_peak_mae=None,
                   min_n_place_neurons=None, min_n_decoder_neurons=None, max_distance=None):
    """Row/session filters for the experiment df (cycle-level), mirroring the
    knobs in `catch_update.plot_update_corr`."""
    out = df
    if maze_names is not None:
        out = out[out[("maze_name", "")].isin(maze_names)]
    if min_amplitude is not None:
        out = out[out[("cycle_metrics", "amplitude")] >= min_amplitude]
    if max_peak_mae is not None:
        out = out[out[("decoder", "peak_mae")] <= max_peak_mae]
    if min_n_place_neurons is not None:
        out = out[out[("place_update", "n_place_neurons")] >= min_n_place_neurons]
    if min_n_decoder_neurons is not None:
        out = out[out[("decoder", "n_distance_neurons")] >= min_n_decoder_neurons]
    if max_distance is not None:
        out = out[out[("distance_to_goal", "geodesic")] <= max_distance]
    return out


def summarise(df, distance_metrics=("absD1", "absD2", "D3"), place_metrics=tuple(PLACE_METRICS),
              n_boot=1000, **filters):
    """Run the fast cluster-OLS link test + cross-lagged directionality for every
    (distance_metric × place_metric) pair and return a tidy results DataFrame for
    the comparison grid. `df` should be a single phase method (filter first).
    Uses `fit_link_ols` (scales to full data); `fit_link_mixedlm` is the slower
    principled cross-check, run separately on headline pairs."""
    rows = []
    for dm in distance_metrics:
        for pm in place_metrics:
            link = fit_link_ols(df, dm, pm, n_boot=n_boot, **filters)
            dirn = directionality_crosslag(df, dm, pm, n_boot=n_boot, **filters)
            rows.append({
                "distance_metric": dm, "place_metric": pm,
                "link_beta": link["beta"], "link_p": link["p"], "n": link["n"],
                "n_subjects": link.get("n_subjects", np.nan),
                "beta_fwd": dirn["beta_fwd"], "beta_rev": dirn["beta_rev"],
                "asymmetry": dirn["asymmetry"], "asym_p": dirn["p_boot"],
            })
    return pd.DataFrame(rows)


def plot_summary(results_by_phase, save_path=None):
    """Comparison grid across phase methods × metric pairs.

    `results_by_phase` is a dict {phase_method: summarise(df)}. Renders, per phase
    method, a heatmap of the standardised link-β (distance→place) annotated with
    significance, and a heatmap of the directionality asymmetry (β_fwd − β_rev).
    Saved as a PNG if `save_path` given.
    """
    import matplotlib.pyplot as plt

    phases = list(results_by_phase)
    fig, axes = plt.subplots(len(phases), 2, figsize=(10, 3.2 * len(phases)), squeeze=False)
    for i, ph in enumerate(phases):
        res = results_by_phase[ph]
        for j, (val_col, p_col, title) in enumerate(
            [("link_beta", "link_p", f"{ph}: link β (dist→place)"),
             ("asymmetry", "asym_p", f"{ph}: directionality β_fwd−β_rev")]
        ):
            piv = res.pivot(index="distance_metric", columns="place_metric", values=val_col)
            pp = res.pivot(index="distance_metric", columns="place_metric", values=p_col)
            ax = axes[i, j]
            vmax = np.nanmax(np.abs(piv.values)) or 1.0
            im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
            for r in range(piv.shape[0]):
                for c in range(piv.shape[1]):
                    v, p = piv.values[r, c], pp.values[r, c]
                    star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""
                    ax.text(c, r, f"{v:+.2f}\n{star}", ha="center", va="center", fontsize=7)
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved: {save_path}")
    return fig


# %% Single-session diagnostic helper


def test_session(session, phase_method="hilbert", cpd_scores=None, **kwargs):
    """Run one session and print a quick schema / sanity diagnostic."""
    df = get_session_catch_update2_df(session, phase_method=phase_method, cpd_scores=cpd_scores, verbose=True, **kwargs)
    print(f"\n{session.name} [{phase_method}]  rows={len(df)}")
    for m in ["D1_signed", "D2_axis_signed", "D3_post_js"]:
        v = df[("decoder", m)].astype(float)
        print(f"  decoder.{m:14s} mean={v.mean():+.4f}  std={v.std():.4f}  nan={v.isna().mean():.1%}")
    for m in ["P1_cosine", "P2_corr", "P3_displacement", "P4_post_js", "P5_mahalanobis"]:
        v = df[("place_update", m)].astype(float)
        print(f"  place.{m:16s} mean={v.mean():+.4f}  std={v.std():.4f}  nan={v.isna().mean():.1%}")
    return df


# %%
