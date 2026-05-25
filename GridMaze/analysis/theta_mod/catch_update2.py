"""
catch_update2 — theta-timescale coupling between the distance-to-goal and place-direction codes.

Hypothesis: both mEC codes are theta-modulated (one sinusoidal sweep per cycle), but the
MAGNITUDE of each code's within-cycle sweep varies cycle-to-cycle. We harness that variability:
if a big distance sweep tends to be followed by a big place sweep (more than the reverse), the
distance code leads the place code on a theta timescale.

Pipeline (one fixed decoder per code, cross-validated; the per-cycle backbone is imported from
`catch_update`):
  * Phase: Hilbert, 7–11 Hz (`la.get_lfp_phase`), n_bins=12 bins per cycle.
  * Pools: disjoint AND each genuinely tuned — distance pool by distance-to-goal CPD, place pool
    by place-direction CPD ("argmax" split). See `get_input_data` / `_select_pools`.
  * Sweep magnitude (per code, per cycle): JS² between the decoded posterior at the code's
    hardcoded PEAK and TROUGH phase windows. Windows are fixed because per-session phase
    estimates are too noisy: distance peak [3,4,5] / trough [9,10,11]; place peak [6,7,8] /
    trough [0,1,2] (n_bins=12; JS is symmetric, so peak/trough only choose which two to compare).
  * Coupling (`sweep_coupling_link`): per cycle offset k, partial correlation of
    distance_sweep[c] ↔ place_sweep[c+k] (confounds regressed out) per subject, then a
    cross-subject t-test. k=+1 = place after distance (distance→place); k=-1 = place before
    distance (place→distance). The paired (r[+1] − r[-1]) ASYMMETRY is the lead/lag readout.
  * Null (`sweep_coupling_null`): cycle-shuffle of place_sweep within (session, trial) isolates
    trial-to-trial coupling from the systematic phase geometry (the offset `decoding_offsets`
    measures). Runs at the link stage — no re-decoding needed.
  * Confounds carried per cycle (per-pool spike counts, speed, theta amplitude, head direction,
    distance-to-goal).

Entry points: `get_session_sweep_coupling_df` (one session) / `get_sweep_coupling_df`
(experiment, cached) / `sweep_coupling_link` / `sweep_coupling_null` / `plot_sweep_coupling`.

HARD RULE: the per-cycle backbone (cycle/phase-window detection, navigation alignment, fold
splitting, per-window rates) is imported from `catch_update` and never modified; new behaviour
lives only here.

@peterdoohan
"""

# %% Imports
import json
import warnings
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
    max_steps_to_goal=16,
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


# %% Jensen–Shannon divergence helper


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


# %% Shared decode infrastructure (one fixed decoder per code, posteriors at each phase window)

# code -> (label column in input_data, key into the pools dict from get_input_data)
CODE_LABELS = {"distance": ("distance_bin_id", ""), "place": ("maze_position", "simple")}

# raw confound columns pulled from input_data (per cycle, read at the representative phase bin)
_CONF_COLS = {
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


# %% Sweep-coupling analysis — per-cycle sweep magnitude + directional cycle-offset coupling


def _window_start(window, window_width, n_bins):
    """Start bin of a hardcoded phase window, with a guard that `window` really is the width-
    `window_width` contiguous (wrap-aware) window beginning at that bin, so a mismatched or
    wrong-width window fails loudly instead of silently reading the wrong posterior."""
    start = int(window[0])
    expected = _sliding_window_bins(start, window_width, n_bins)
    assert list(window) == expected, f"window {list(window)} != width-{window_width} window at bin {start} ({expected})"
    return start


def get_session_sweep_coupling_df(
    session,
    window_width=3,
    distance_peak=(3, 4, 5),
    distance_trough=(9, 10, 11),
    place_peak=(6, 7, 8),
    place_trough=(0, 1, 2),
    normalise_X=True,
    C=1.0,
    n_folds=10,
    cv_seed=0,
    cpd_scores=None,
    verbose=False,
    **input_data_kwargs,
):
    """One row per held-out theta cycle with each code's within-cycle SWEEP MAGNITUDE.

    For each code a single fixed decoder is trained on width-`window_width` sliding windows of the
    training trials (`_train_code_decoder`) and read out at the code's hardcoded peak and trough
    windows on the held-out cycles (`_code_window_posteriors`); the per-cycle sweep magnitude is
    `JS²(peak posterior, FOLLOWING-trough posterior)` (`_js2_rows`). "Following" matters: when a
    code's trough window precedes its peak within the cycle (place: peak [6,7,8] -> trough [0,1,2]),
    the trough is read from the NEXT cycle (within the same trial) so the peak is paired with the
    trough that genuinely follows it; the sweep is labelled by the peak's cycle. Distance's trough
    [9,10,11] already follows its peak [3,4,5] in the same cycle. Windows are fixed (per-session
    phase estimates are too noisy) but overridable; JS is symmetric, so peak/trough labels only
    choose which two windows are compared.

    Returns a flat-column DataFrame, one row per cycle: `distance_sweep`, `place_sweep`, the
    `_CONF_COLS` confounds (per cycle, read at phase bin 0), `trial`, `cycle`, `fold`, and
    `subject`/`session`/`maze`. Feed to `get_sweep_coupling_df` / `sweep_coupling_link`.
    """
    input_data, pools = get_input_data(session, cpd_scores=cpd_scores, verbose=verbose, **input_data_kwargs)
    n_bins = int(input_data.index.get_level_values("phase_bin").max()) + 1
    windows = {
        "distance": (
            _window_start(distance_peak, window_width, n_bins),
            _window_start(distance_trough, window_width, n_bins),
        ),
        "place": (_window_start(place_peak, window_width, n_bins), _window_start(place_trough, window_width, n_bins)),
    }
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

    rows = []
    for fold_idx, held_out in enumerate(folds):
        train_df, test_df = cu._split_fold(input_data, held_out)

        # per-cycle sweep magnitude = JS² between a code's peak window and its FOLLOWING trough.
        # delta=1 reads the trough from the next cycle for codes whose same-cycle trough precedes
        # the peak (place: peak [6,7,8] -> trough [0,1,2] of c+1), so the peak is always paired with
        # the trough that genuinely follows it; the cross-boundary pair is kept only within one trial.
        # delta=0 for distance (trough [9,10,11] already follows peak [3,4,5] in the same cycle).
        # Both sweeps are labelled by the peak's cycle.
        trial_of = test_df.trial.groupby(level="cycle_idx").first()
        sweeps, usable = {}, True
        for code, cells in cells_by_code.items():
            decoder, scaler = _train_code_decoder(
                train_df, cells, CODE_LABELS[code], n_bins, window_width, normalise_X, C
            )
            if decoder is None:
                usable = False
                break
            P = _code_window_posteriors(test_df, cells, decoder, scaler, n_bins, window_width)
            peak_start, trough_start = windows[code]
            peak, trough = P[peak_start], P[trough_start]
            delta = 1 if trough_start < peak_start else 0  # following trough is in the next cycle
            anchors = peak.index.intersection(trough.index - delta)  # peak@c with trough@(c+delta) present
            if delta:
                anchors = anchors[trial_of.reindex(anchors).values == trial_of.reindex(anchors + delta).values]
            if len(anchors) == 0:
                usable = False
                break
            sweeps[code] = pd.Series(
                _js2_rows(peak.loc[anchors].values, trough.loc[anchors + delta].values), index=anchors
            )
        if not usable:
            continue

        cyc = sweeps["distance"].index.intersection(sweeps["place"].index)
        if len(cyc) == 0:
            continue
        meta = test_df.xs(0, level="phase_bin").reindex(cyc)
        block = pd.DataFrame(
            {
                "distance_sweep": sweeps["distance"].reindex(cyc).values,
                "place_sweep": sweeps["place"].reindex(cyc).values,
                "trial": np.asarray(meta["trial"]).reshape(len(meta), -1)[:, 0],
                "cycle": np.asarray(cyc),
                "fold": fold_idx,
            }
        )
        for name, col in _CONF_COLS.items():
            block[name] = meta[col].astype(float).values
        rows.append(block)

    if not rows:
        raise ValueError(f"{session.name}: no usable test cycles")
    df = pd.concat(rows, ignore_index=True)
    df["subject"], df["session"], df["maze"] = session.subject_ID, session.name, session.maze_name
    if verbose:
        print(
            f"{session.name}: {len(df)} cycles  "
            f"distance_sweep mean={df.distance_sweep.mean():.4f}  place_sweep mean={df.place_sweep.mean():.4f}"
        )
    return df


def _offset_pairs(s, k, confounds):
    """Pair `distance_sweep[c]` with `place_sweep[c+k]` for one subject's rows, keeping each pair
    within the same (session, trial). Keyed by (session, trial, cycle) so cycles are unique across
    a subject's sessions and a partner whose `cycle+k` falls in another trial/session is dropped
    (its key is absent). Returns a DataFrame with `x` (distance@c), `y` (place@c+k), and the
    requested confounds taken at the anchor `c`; rows with any NaN are dropped."""
    key = ["session", "trial", "cycle"]
    d_by = s.set_index(key)["distance_sweep"]
    p_by = s.set_index(key)["place_sweep"]
    anchors = d_by.index
    partner = pd.MultiIndex.from_arrays(
        [anchors.get_level_values("session"), anchors.get_level_values("trial"), anchors.get_level_values("cycle") + k],
        names=key,
    )
    out = pd.DataFrame({"x": d_by.values, "y": p_by.reindex(partner).values})
    if confounds:
        conf_by = s.set_index(key)[list(confounds)].reindex(anchors)
        for c in confounds:
            out[c] = conf_by[c].values
    return out.dropna()


def _partial_r(sub, confounds):
    """Partial Pearson r between `sub.x` and `sub.y`, regressing `confounds` out of both via
    least squares. Returns nan if either residual is constant."""
    x, y = sub["x"].values.astype(float), sub["y"].values.astype(float)
    if confounds:
        X = np.column_stack([np.ones(len(sub))] + [sub[c].values.astype(float) for c in confounds])
        x = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
        y = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return pearsonr(x, y)[0]


def sweep_coupling_link(
    df,
    offsets=(-1, 0, 1),
    confounds=CONFOUNDS,
    maze_names=None,
    max_distance=None,
    min_obs=10,
    alternative="two-sided",
):
    """Directional coupling between the two codes' per-cycle sweep magnitudes.

    For each cycle offset k, correlate `distance_sweep[c]` with `place_sweep[c+k]` across cycles
    (within session/trial), per subject with `confounds` partialled out of both, then a
    cross-subject one-sample t-test on the subject r's. k=+1 → place sweep AFTER distance
    (distance-leads-place); k=-1 → place BEFORE distance (place-leads-distance). The `"asymmetry"`
    block is the paired cross-subject test of per-subject (r[+1] - r[-1]) — the lead/lag readout.

    Returns {k: {mean_r, sem_r, t, p, n_subjects, per_subject}} plus, when both ±1 are present,
    {"asymmetry": {mean_diff, t, p, n_subjects, per_subject}}.
    """
    df = df.copy()
    df["hd_sin"] = np.sin(np.deg2rad(df["hd"].astype(float)))
    df["hd_cos"] = np.cos(np.deg2rad(df["hd"].astype(float)))
    if maze_names is not None:
        df = df[df["maze"].isin(maze_names)]
    if max_distance is not None:
        df = df[df["dist2goal"] <= max_distance]
    conf = list(confounds or [])

    per_subject = {k: [] for k in offsets}
    for subj, s in df.groupby("subject"):
        for k in offsets:
            sub = _offset_pairs(s, k, conf)
            if len(sub) < min_obs:
                continue
            r = _partial_r(sub, conf)
            if not np.isnan(r):
                per_subject[k].append({"subject": subj, "n": len(sub), "r": r})

    result = {}
    for k in offsets:
        ps = pd.DataFrame(per_subject[k], columns=["subject", "n", "r"])
        rs = ps["r"].values
        if len(rs) >= 2:
            t, p = ttest_1samp(rs, 0, alternative=alternative)
            sem = rs.std(ddof=1) / np.sqrt(len(rs))
        else:
            t = p = sem = np.nan
        result[k] = {
            "mean_r": float(np.mean(rs)) if len(rs) else np.nan,
            "sem_r": float(sem) if len(rs) else np.nan,
            "t": float(t) if len(rs) >= 2 else np.nan,
            "p": float(p) if len(rs) >= 2 else np.nan,
            "n_subjects": len(rs),
            "per_subject": ps,
        }

    if 1 in offsets and -1 in offsets:
        rp = result[1]["per_subject"].set_index("subject")["r"]
        rm = result[-1]["per_subject"].set_index("subject")["r"]
        common = rp.index.intersection(rm.index)
        diffs = (rp.loc[common] - rm.loc[common]).values
        if len(diffs) >= 2:
            ta, pa = ttest_1samp(diffs, 0, alternative=alternative)
        else:
            ta = pa = np.nan
        result["asymmetry"] = {
            "mean_diff": float(np.mean(diffs)) if len(diffs) else np.nan,
            "t": float(ta) if len(diffs) >= 2 else np.nan,
            "p": float(pa) if len(diffs) >= 2 else np.nan,
            "n_subjects": len(diffs),
            "per_subject": pd.DataFrame(
                {
                    "subject": list(common),
                    "r_plus": rp.loc[common].values,
                    "r_minus": rm.loc[common].values,
                    "diff": diffs,
                }
            ),
        }
    return result


def sweep_coupling_null(
    df,
    offsets=(-1, 0, 1),
    confounds=CONFOUNDS,
    n_shuffles=1000,
    within_trial=True,
    seed=0,
    alternative="two-sided",
    **link_kwargs,
):
    """Cycle-shuffle null: permute the `place_sweep` column within (session[, trial]) and recompute
    the offset correlations `n_shuffles` times. This destroys the cycle-by-cycle pairing (isolating
    trial-to-trial coupling) while preserving each variable's marginal and the systematic phase
    geometry. Runs entirely at the link stage — no re-decoding. Returns {k: {true_mean_r,
    null_mean_r, perm_p}} and, when ±1 are present, an "asymmetry" entry; `perm_p` is the two-sided
    (1 + #|null| >= |true|) / (1 + n) permutation p-value."""
    true = sweep_coupling_link(df, offsets=offsets, confounds=confounds, alternative=alternative, **link_kwargs)
    group = ["session", "trial"] if within_trial else ["session"]
    null_mean = {k: [] for k in offsets}
    null_asym = []
    for i in range(n_shuffles):
        rng = np.random.default_rng([seed, i])
        shuf = df.copy()
        shuf["place_sweep"] = shuf.groupby(group)["place_sweep"].transform(lambda v: rng.permutation(v.values))
        r = sweep_coupling_link(shuf, offsets=offsets, confounds=confounds, alternative=alternative, **link_kwargs)
        for k in offsets:
            null_mean[k].append(r[k]["mean_r"])
        if "asymmetry" in r:
            null_asym.append(r["asymmetry"]["mean_diff"])

    def _perm_p(null_vals, true_val):
        nv = np.asarray([v for v in null_vals if not np.isnan(v)])
        if not len(nv) or np.isnan(true_val):
            return np.nan, np.nan
        return float(np.mean(nv)), float((1 + np.sum(np.abs(nv) >= abs(true_val))) / (1 + len(nv)))

    out = {}
    for k in offsets:
        nm, pp = _perm_p(null_mean[k], true[k]["mean_r"])
        out[k] = {"true_mean_r": true[k]["mean_r"], "null_mean_r": nm, "perm_p": pp}
    if "asymmetry" in true:
        nm, pp = _perm_p(null_asym, true["asymmetry"]["mean_diff"])
        out["asymmetry"] = {"true_mean_diff": true["asymmetry"]["mean_diff"], "null_mean_diff": nm, "perm_p": pp}
    return out


def get_sweep_coupling_df(save=False, verbose=True, n_jobs=3, tag=None, window_width=3, **session_kwargs):
    """Run `get_session_sweep_coupling_df` across every subject × maze × late day and concatenate
    to one cross-session dataframe, cached under `RESULTS_DIR/runs/`. Per-session failures are
    swallowed (printed if verbose).

    Cache key: the default filename folds in `window_width` (`sweep_coupling_df_w{window_width}`);
    an explicit `tag` overrides it. NOTE: the peak/trough windows, `C`, and `n_folds` are NOT in
    the default name, so changing those without a fresh `tag` (or `save=True`) loads a stale parquet.
    """
    save_path = RESULTS_DIR / "runs" / f"{tag or f'sweep_coupling_df_w{window_width}'}.parquet"
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
            return get_session_sweep_coupling_df(
                session, window_width=window_width, cpd_scores=cpd_scores, verbose=False, **session_kwargs
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


def plot_sweep_coupling(
    df,
    offsets=(-1, 0, 1),
    confounds=CONFOUNDS,
    scatter_offset=1,
    alternative="two-sided",
    save_path=None,
    axes=None,
    color="C0",
    **link_kwargs,
):
    """Panel A: per-offset cross-subject mean ± SEM partial r over cycle offset k, with faint
    per-subject traces; the title reports the +1 vs −1 asymmetry t/p. Panel B (optional): the raw
    pooled `distance_sweep[c]` vs `place_sweep[c+scatter_offset]` scatter. Returns (axes, result)."""
    res = sweep_coupling_link(df, offsets=offsets, confounds=confounds, alternative=alternative, **link_kwargs)
    offs = list(offsets)
    n_panels = 2 if scatter_offset is not None else 1
    if axes is None:
        _, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels + 0.8, 3.2))
    axes = np.atleast_1d(axes)
    ax = axes[0]

    subj_r = {k: res[k]["per_subject"].set_index("subject")["r"] for k in offs}
    subjects = sorted(set().union(*[set(subj_r[k].index) for k in offs])) if offs else []
    for subj in subjects:
        ax.plot(
            offs, [subj_r[k].get(subj, np.nan) for k in offs], color="grey", alpha=0.4, lw=1, marker="o", ms=3, zorder=2
        )
    ax.errorbar(
        offs,
        [res[k]["mean_r"] for k in offs],
        yerr=[res[k]["sem_r"] for k in offs],
        color=color,
        lw=2,
        marker="o",
        ms=7,
        capsize=3,
        zorder=3,
    )
    ax.axhline(0, color="k", ls="--", alpha=0.5)
    ax.set_xticks(offs)
    ax.set_xlabel("cycle offset k  (place sweep at c+k)")
    ax.set_ylabel("partial r  (distance_sweep ↔ place_sweep)")
    ax.spines[["top", "right"]].set_visible(False)
    asym = res.get("asymmetry")
    title = "distance→place (k>0)  vs  place→distance (k<0)"
    if asym is not None:
        title += f"\nasymmetry r[+1]−r[−1]: t={asym['t']:+.2f} p={asym['p']:.3g} (n={asym['n_subjects']})"
    ax.set_title(title, fontsize=8)

    if scatter_offset is not None and n_panels == 2:
        ax2 = axes[1]
        pts = pd.concat([_offset_pairs(s, scatter_offset, []) for _, s in df.groupby("subject")], ignore_index=True)
        if len(pts) > 2:
            ax2.hexbin(pts["x"], pts["y"], gridsize=40, cmap="magma", mincnt=1, bins="log")
            r_raw = pearsonr(pts["x"], pts["y"])[0]
            ax2.set_title(f"k={scatter_offset:+d}  raw r={r_raw:+.3f}", fontsize=8)
        ax2.set_xlabel("distance sweep (c)")
        ax2.set_ylabel(f"place sweep (c{scatter_offset:+d})")
        ax2.spines[["top", "right"]].set_visible(False)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        axes[0].figure.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved: {save_path}")
    return axes, res


def test_sweep_coupling_session(session, cpd_scores=None, offsets=(-1, 0, 1), **kwargs):
    """Run one session through the sweep-coupling pipeline and print a quick diagnostic. A single
    session is one subject, so the cross-subject t is undefined (NaN) — this is a schema / sanity /
    asymmetry-direction check."""
    df = get_session_sweep_coupling_df(session, cpd_scores=cpd_scores, verbose=True, **kwargs)
    print(f"\n{session.name}  cycles={len(df)}")
    for c in ("distance_sweep", "place_sweep"):
        v = df[c].astype(float)
        print(
            f"  {c:14s} mean={v.mean():+.4f}  std={v.std():.4f}  min={v.min():.4f}  max={v.max():.4f}  nan={v.isna().mean():.1%}"
        )
    res = sweep_coupling_link(df, offsets=offsets)
    for k in offsets:
        print(f"  k={k:+d}  mean_r={res[k]['mean_r']:+.4f}  (n_subjects={res[k]['n_subjects']})")
    if "asymmetry" in res:
        print(f"  asymmetry r[+1]-r[-1]: mean={res['asymmetry']['mean_diff']:+.4f}")
    return df, res
