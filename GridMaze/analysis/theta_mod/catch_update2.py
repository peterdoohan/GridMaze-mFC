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

HARD RULE: this file is the only place new behaviour is written; everything
reusable is imported (notably the per-cycle backbone from `catch_update`) and
never modified.

@peterdoohan
"""

# %% Imports
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


def get_input_data(
    session,
    n_bins=12,
    shank=3,
    max_steps_to_goal=20,
    exclude_at_goal=True,
    bin_spacing=0.05,
    moving_only=True,
    min_firing_rate=0.5,
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
    """
    input_data, pools = get_input_data(session, cpd_scores=cpd_scores, verbose=verbose, **input_data_kwargs)
    distance_cells, place_cells = pools["distance"], pools["place"]
    distance_bin_mids = np.array(sorted(input_data.distance_bin_mid.dropna().unique()))
    n_bins = int(input_data.index.get_level_values("phase_bin").max()) + 1
    if n_bins % n_training_phases != 0:
        raise ValueError(f"n_training_phases ({n_training_phases}) must divide n_bins ({n_bins})")

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
        raise ValueError(f"{session.name}: no usable test cycles")
    session_df = pd.concat(per_cycle).sort_index()
    session_df[("qc", "decoder_mae")] = float(np.mean(np.concatenate(abs_errs)))
    session_df[("qc", "n_distance_neurons")] = len(distance_cells)
    session_df[("qc", "n_place_neurons")] = len(place_cells)
    if verbose:
        print(f"{session.name}: decoder_mae={session_df[('qc','decoder_mae')].iloc[0]:.3f} m")
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


def get_catch_update2_df(save=False, verbose=True, n_jobs=6, tag=None, **session_kwargs):
    """Run `get_session_catch_update2_df` across every subject × maze × late day
    and concatenate to one cross-session dataframe, cached to parquet under
    `RESULTS_DIR/runs/catch_update2_df{tag}.parquet`. Per-session failures are
    swallowed (printed if verbose)."""
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
    current schema and a pre-rename `summary_df`. Returns {"per_subject", "pooled"}.
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

    ps = link_per_subject(df, confounds=confounds, alternative=alternative, **filters)
    pooled = link_pooled(df, confounds=confounds, **filters)
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
    if save_fig:
        plot_link(df, confounds=confounds, alternative=alternative, save_path=save_fig, **filters)
    return {"per_subject": ps, "pooled": pooled}


# %% Single-session diagnostic helper


def test_session(session, cpd_scores=None, **kwargs):
    """Run one session and print a quick schema / sanity diagnostic."""
    df = get_session_catch_update2_df(session, cpd_scores=cpd_scores, verbose=True, **kwargs)
    print(f"\n{session.name}  rows={len(df)}")
    for col, m in [(("distance", "update"), "distance_update"), (("place", "update"), "place_update")]:
        v = df[col].astype(float)
        print(f"  {m:16s} mean={v.mean():+.4f}  std={v.std():.4f}  nan={v.isna().mean():.1%}")
    return df


# %%
