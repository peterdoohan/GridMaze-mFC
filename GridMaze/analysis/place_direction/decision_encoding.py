"""
Decision-aligned Poisson encoding of place-direction and distance-to-goal in mFC.

Tests whether mFC encoding of place-direction (PD) and distance-to-goal (DTG) is
stronger at decision points where structure != habit and the animal chose the
structure action vs the habit action (with structure == habit decisions as a
baseline).

Only navigation-phase bins are trained on and scored (reward_consumption / ITI bins are
excluded but kept in the array as positional spacers so offset arithmetic stays
time-aligned). Per-cluster Poisson GLM is fit four times on training bins, with no nuisance
regressors (no speed, no trial_phase):
    null         : intercept only (mean firing rate)
    reduced_pd   : DTG bases
    reduced_dtg  : PD bases
    full         : PD bases + DTG bases

PD bases are the top `pd_n_bases` PCs of place-direction tuning learned from all OTHER
same-maze sessions; DTG bases are `dtg_n_bases` gamma functions over the geodesic distance
domain.
Per (cluster, category, offset), held-out test bins are pooled and unique D² is
computed by deviance ratio against the reduced models:
    d2_place_direction  = 1 - D(y, mu_full) / D(y, mu_reduced_pd)
    d2_distance_to_goal = 1 - D(y, mu_full) / D(y, mu_reduced_dtg)
    d2_combined         = 1 - D(y, mu_full) / D(y, mu_null)
combined is the full (PD + DTG) model's total deviance explained above the mean firing rate.

Decision events are arrival times at choice nodes (matches
theta_decisions.get_session_decision_times). Offset Δ < 0 = approach,
Δ ≈ 0 = at choice node, Δ > 0 = post-choice traversal.

@peterdoohan
"""

# %% Imports
import json
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from joblib import Parallel, delayed
from scipy.stats import ttest_rel, false_discovery_control
from sklearn.linear_model import PoissonRegressor

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import convert
from GridMaze.analysis.lfp import theta_decisions as tdec
from GridMaze.analysis.place_direction import bases as pdb
from GridMaze.analysis.distance_to_goal import bases as db
from GridMaze.analysis.distance_to_goal import distributions as dd
from GridMaze.maze import representations as mr

# %% Global Variables
from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

RESULTS_DIR = RESULTS_PATH / "place_direction"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

_MAX_ITER = 10_000  # PoissonRegressor max iterations
_MU_FLOOR = 1e-12  # floor on predicted rates before log in Poisson deviance
_META_COLS = ["subject_ID", "session_name", "maze_name", "day_on_maze"]  # per-cluster metadata


# %% Session-level decision-aligned encoding


def get_session_decision_aligned_encoding(
    session,
    resolution=0.2,
    window=(-3.0, 3.0),
    pd_n_bases=20,
    dtg_metric=("distance_to_goal", "geodesic"),
    dtg_n_bases=8,
    dtg_basis="gamma",
    dtg_max_distance_pct=90,
    n_folds=10,
    min_spikes=500,
    max_steps_to_goal=30,
    include_multi_units=False,
    alpha="opt",
    alpha_range=np.logspace(-3, 3, 10),
    n_inner_folds=4,
    categories=("agree", "chose_structure", "chose_habit"),
    random_state=0,
    verbose=False,
):
    """
    Fit per-cluster Poisson GLMs under trial-CV and score held-out bins at offsets ±
    `window` around each decision arrival time. Only navigation-phase bins are trained
    on and scored (reward_consumption / ITI bins remain in the array as positional
    spacers for offset alignment but are masked out of fitting and scoring). Four model
    variants are fit per cluster per fold (no nuisance regressors):
        null         : intercept only (mean firing rate)
        reduced_pd   : DTG bases
        reduced_dtg  : PD bases
        full         : PD bases + DTG bases

    Unique deviance explained per variable (PD, DTG) is derived from full vs the
    corresponding reduced model; combined (vs null) is the full model's total deviance
    explained above the mean rate.

    `alpha` can be a scalar (fixed regularisation) or the string `"opt"`, in which case
    each cluster's α is picked by inner k-fold CV (`n_inner_folds` folds over training
    trials) over `alpha_range`, independently per feature set. Scoring uses
    `PoissonRegressor.score` (pseudo-D²).

    Returns
    -------
    samples_df : sample-level, one row per (cluster_unique_ID, decision_id, offset) with
        `y`, `mu_null`, `mu_reduced_pd`, `mu_reduced_dtg`, `mu_full`, `distance_to_goal`,
        `steps_to_goal`, `category`, `trial`, `fold`, plus session metadata. Aggregate to
        per-variable D² (with optional DTG matching) via `aggregate_to_d2`.
    """
    # 1) PD bases: all OTHER same-maze sessions (cached, see bases.get_pd_heatmaps_df)
    bases_df = pdb.get_session_pd_bases(session, n_bases=pd_n_bases, dim_red="pca")

    # 2) Binned navigation + spike data (all phases, DTG plumbed through)
    navigation_spikes_df = _build_navigation_spikes_df(
        session,
        resolution=resolution,
        max_steps_to_goal=max_steps_to_goal,
        min_spikes=min_spikes,
        include_multi_units=include_multi_units,
    )
    simple_maze = session.simple_maze()

    # 3) Nested feature sets {name → (n_bins, n_features)} + spike-count matrix
    feature_sets, Y, cluster_unique_IDs = _build_feature_sets(
        navigation_spikes_df,
        simple_maze,
        bases_df,
        dtg_metric=dtg_metric,
        dtg_n_bases=dtg_n_bases,
        dtg_basis=dtg_basis,
        dtg_max_distance_pct=dtg_max_distance_pct,
    )

    # 4) Decision events with categories
    decision_df = tdec.get_session_decision_times(session, decision_points_only=True)
    if decision_df is None or decision_df.empty:
        return pd.DataFrame()
    decision_df = decision_df[decision_df.category.isin(list(categories))].reset_index(drop=True)
    # map each decision's session-clock arrival time to its bin index in navigation_spikes_df
    bin_times = navigation_spikes_df[("time", "")].values
    decision_df = _attach_bin_index(decision_df, bin_times)
    decision_df = decision_df.dropna(subset=["bin_idx"]).reset_index(drop=True)
    decision_df["bin_idx"] = decision_df["bin_idx"].astype(int)
    # attach goal per decision (constant within a trial; choice-node `location` already present)
    goal_by_trial = pd.Series(
        navigation_spikes_df[("goal", "")].values, index=navigation_spikes_df[("trial", "")].values
    )
    decision_df["goal"] = decision_df["trial"].map(goal_by_trial[~goal_by_trial.index.duplicated()])

    # 5) Random k-fold split by trial number
    all_trials = navigation_spikes_df.trial.dropna().unique()
    rng = np.random.default_rng(seed=random_state)
    test_trials_per_fold = np.array_split(rng.permutation(all_trials), n_folds)

    # 6) Run per-fold per-cluster fitting + scoring
    offsets_s = np.round(np.arange(window[0], window[1] + resolution / 2, resolution), 6)
    offsets_bins = np.round(offsets_s / resolution).astype(int)
    bin_trials = navigation_spikes_df.trial.values.ravel()
    dtg_per_bin = navigation_spikes_df[dtg_metric].values.ravel()  # raw, NaN preserved
    steps_per_bin = navigation_spikes_df.steps_to_goal.future.values.ravel()
    nav_per_bin = (navigation_spikes_df[("trial_phase", "")] == "navigation").values.ravel()

    fold_results = [
        _process_fold(
            fold=i,
            test_trials=test_trials_per_fold[i],
            all_trials=all_trials,
            bin_trials=bin_trials,
            decision_df=decision_df,
            feature_sets=feature_sets,
            Y=Y,
            dtg_per_bin=dtg_per_bin,
            steps_per_bin=steps_per_bin,
            nav_per_bin=nav_per_bin,
            cluster_unique_IDs=cluster_unique_IDs,
            offsets_s=offsets_s,
            offsets_bins=offsets_bins,
            alpha=alpha,
            alpha_range=alpha_range,
            n_inner_folds=n_inner_folds,
            random_state=random_state,
            verbose=verbose,
        )
        for i in range(n_folds)
    ]
    samples_df = (
        pd.concat([df for df in fold_results if df is not None and not df.empty], ignore_index=True)
        if fold_results
        else pd.DataFrame()
    )
    if samples_df.empty:
        return samples_df

    # 7) attach session metadata and return sample-level df
    samples_df["session_name"] = session.name  # per-session ID (enables per-session matching later)
    samples_df["subject_ID"] = session.subject_ID
    samples_df["maze_name"] = session.maze_name
    samples_df["day_on_maze"] = session.day_on_maze
    return samples_df


# %% Helpers


def _build_navigation_spikes_df(
    session,
    resolution,
    max_steps_to_goal,
    min_spikes,
    include_multi_units,
):
    """Bin + filter session data into a per-bin DataFrame with PD string, trial_phase,
    distance_to_goal, and steps_to_goal.

    Keeps all three trial_phases (navigation, reward_consumption, ITI) as a contiguous,
    uniformly-spaced series: only navigation bins are trained/scored, but the non-nav bins
    must stay as positional spacers so the offset arithmetic in `_process_fold` remains
    time-aligned. `max_steps_to_goal` is applied inline only to navigation rows
    (steps_to_goal is NaN in reward_consumption/ITI, which would otherwise drop them).
    """
    navigation_df = session.navigation_df
    spike_counts_df = session.navigation_spike_counts_df.reset_index(drop=True)

    keep_clusters = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=include_multi_units,
    )
    spike_counts_df = spike_counts_df[spike_counts_df.columns[spike_counts_df.spike_count.columns.isin(keep_clusters)]]
    ds_nav_df, ds_spikes_df = ds.downsample_nav_spikes_data(
        navigation_df,
        spike_counts_df,
        resolution=resolution,
        distance_metrics=[("steps_to_goal", "future"), ("distance_to_goal", "geodesic")],
    )
    df = pd.concat([ds_nav_df, ds_spikes_df], axis=1)
    df[("place_direction", "")] = df.maze_position.simple + "_" + df.cardinal_movement_direction
    # keep all phases + at-goal bins as positional spacers (only nav bins scored downstream)
    df = filt.filter_navigation_rates_df(
        df,
        navigation_only=False,
        moving_only=False,
        exclude_time_at_goal=False,
        max_steps_to_goal=None,
    )
    # apply max_steps_to_goal only to navigation rows (steps_to_goal is NaN elsewhere)
    if max_steps_to_goal is not None:
        is_nav = df[("trial_phase", "")] == "navigation"
        nav_far = is_nav & df.steps_to_goal.future.ge(max_steps_to_goal)
        df = df[~nav_far].reset_index(drop=True)
    if min_spikes is not None:
        spike_counts = df.spike_count
        low_spike_clusters = spike_counts.columns[spike_counts.sum(axis=0) < min_spikes].values
        df = df.drop(columns=low_spike_clusters, level=1, axis=1)
    return df.reset_index(drop=True)


def _build_feature_sets(
    navigation_spikes_df,
    simple_maze,
    bases_df,
    dtg_metric,
    dtg_n_bases,
    dtg_basis,
    dtg_max_distance_pct,
):
    """Build the four nested Poisson-GLM feature sets from the binned session df.

    Blocks: place-direction projected onto PD-PC bases, distance-to-goal gamma bases.
    No nuisance regressors (no speed, no trial_phase):
        null        : intercept only (a single zero column → intercept solves to log(mean rate))
        reduced_pd  : DTG bases only (no PD)
        reduced_dtg : PD bases only (no DTG)
        full        : PD bases + DTG bases

    Returns (feature_sets, Y, cluster_unique_IDs) where feature_sets maps
    {null, reduced_pd, reduced_dtg, full} → (n_bins, n_features) arrays.
    """
    # PD: project place-direction one-hot onto the PD-PC bases
    pd_strings = navigation_spikes_df.place_direction.values.astype(str)
    X_pd = _project_onehot_onto_bases(convert.place_direction2onehot(pd_strings, simple_maze), simple_maze, bases_df)
    # DTG: gamma bases over the geodesic-distance domain (max at the given percentile)
    dtg_max = dd.get_distance_percentile(dtg_metric, percentile=dtg_max_distance_pct)
    dtg_basis_fn = db.distance_basis_generator(
        n_bases=dtg_n_bases, basis=dtg_basis, btype="distance", max_distance=dtg_max
    )
    dtg_values = navigation_spikes_df[dtg_metric].values.astype(float)
    X_dtg = np.nan_to_num(dtg_basis_fn(dtg_values), nan=0.0)  # NaN DTG (inter-trial) → 0

    n_bins = X_pd.shape[0]
    feature_sets = {
        "null": np.zeros((n_bins, 1)),  # intercept-only → mu_null = mean rate
        "reduced_pd": X_dtg,  # DTG only (no PD)
        "reduced_dtg": X_pd,  # PD only (no DTG)
        "full": np.hstack([X_pd, X_dtg]),  # PD + DTG
    }
    Y = navigation_spikes_df.spike_count.values  # (n_bins, n_clusters)
    cluster_unique_IDs = list(navigation_spikes_df.spike_count.columns)
    return feature_sets, Y, cluster_unique_IDs


def _project_onehot_onto_bases(X_onehot, simple_maze, bases_df):
    """
    X_onehot: (n_bins, n_pd_pairs) one-hot in `place_direction2onehot` column order.
    bases_df: rows = MultiIndex of (pos, dir) tuples, cols = component IDs.
    Reindex bases to the canonical column order and matrix-multiply.
    PD pairs missing from the bases (e.g. dropped by occupancy filter) project to 0.
    """
    pd_pairs = list(mr.get_maze_place_direction_pairs(simple_maze))
    bases_ordered = bases_df.reindex(pd.MultiIndex.from_tuples(pd_pairs)).fillna(0).values  # (n_pd, n_bases)
    return X_onehot @ bases_ordered


def _attach_bin_index(decision_df, bin_times):
    """For each decision (session-clock time), find the nearest bin's index (NaN if none)."""
    times = decision_df.time.values
    idxs = np.searchsorted(bin_times, times)
    nearest = np.full(len(times), np.nan)
    for i, (t, k) in enumerate(zip(times, idxs)):
        candidates = [c for c in (k - 1, k) if 0 <= c < len(bin_times)]
        if candidates:  # snap to whichever neighbouring bin is closest in time
            nearest[i] = candidates[int(np.argmin([abs(bin_times[c] - t) for c in candidates]))]
    out = decision_df.copy()
    out["bin_idx"] = nearest
    return out


def _process_fold(
    fold,
    test_trials,
    all_trials,
    bin_trials,
    decision_df,
    feature_sets,
    Y,
    dtg_per_bin,
    steps_per_bin,
    nav_per_bin,
    cluster_unique_IDs,
    offsets_s,
    offsets_bins,
    alpha,
    alpha_range,
    n_inner_folds,
    random_state,
    verbose,
):
    """Fit Poisson GLMs per cluster per feature set on train bins; score held-out
    test bins aligned to each test decision × offset.

    `feature_sets` is a dict mapping name -> (n_bins, n_features) matrix. One model
    per (cluster, feature_set) is fit on the outer-fold training bins; predictions
    are stored per bin under `mu_{name}`.

    Train/test masks are built by trial number: `train_trials = all_trials - test_trials`.
    Only navigation bins (`nav_per_bin`) are trained on or scored; non-navigation bins
    (reward_consumption / ITI) stay in the array purely as positional spacers so the
    offset arithmetic remains time-aligned. Bins with NaN trial label also match no
    trial set and so are excluded.

    If `alpha == "opt"`, each cluster's α is selected by inner k-fold CV over training
    trials (`_search_alpha_per_cluster`) independently per feature set.
    """
    if verbose:
        print(f"  fold {fold}")
    train_trials = np.setdiff1d(all_trials, test_trials)
    train_mask = np.isin(bin_trials, train_trials) & nav_per_bin  # navigation bins only

    test_decisions = decision_df[decision_df.trial.isin(test_trials)].reset_index(drop=True)
    if test_decisions.empty:
        return None

    n_bins = len(bin_trials)
    centre_bins = test_decisions.bin_idx.values.astype(int)
    bin_grid = centre_bins[:, None] + offsets_bins[None, :]  # (n_decisions, n_offsets)
    in_range = (bin_grid >= 0) & (bin_grid < n_bins)
    # score bins whose host trial is in the held-out set (no train leakage) AND are navigation
    safe_grid = np.clip(bin_grid, 0, n_bins - 1)
    bin_trial_of_grid = np.where(in_range, bin_trials[safe_grid], np.nan)
    in_test_trial = np.isin(bin_trial_of_grid, test_trials)
    valid = in_range & in_test_trial & nav_per_bin[safe_grid]

    # per-feature-set per-cluster α: fixed scalar or inner-CV searched
    if alpha == "opt":
        if verbose:
            print(f"    inner-CV α search (fold {fold})")
        alphas_per_set = {
            name: _search_alpha_per_cluster(
                X,
                Y,
                train_trials,
                bin_trials,
                nav_per_bin,
                alpha_range,
                n_inner_folds,
                random_state + fold,
            )
            for name, X in feature_sets.items()
        }
    else:
        alphas_per_set = {name: np.full(len(cluster_unique_IDs), float(alpha)) for name in feature_sets}

    rows = []
    for c_idx, cuid in enumerate(cluster_unique_IDs):
        y_train = Y[train_mask, c_idx]
        # fit each feature set for this cluster, predict over all bins
        cluster_mu = {}
        for name, X in feature_sets.items():
            m = PoissonRegressor(alpha=alphas_per_set[name][c_idx], max_iter=_MAX_ITER)
            m.fit(X[train_mask], y_train)
            cluster_mu[name] = m.predict(X)
        # collect per-(decision, offset) records
        for d_i in range(bin_grid.shape[0]):
            cat = test_decisions.category.iloc[d_i]
            trial = test_decisions.trial.iloc[d_i]
            location = test_decisions.location.iloc[d_i]  # choice-node label
            goal = test_decisions.goal.iloc[d_i]  # trial goal (constant within trial)
            decision_id = f"trial{int(trial)}_b{int(centre_bins[d_i])}"
            for o_i, off_s in enumerate(offsets_s):
                if not valid[d_i, o_i]:
                    continue
                b = int(bin_grid[d_i, o_i])
                row = {
                    "cluster_unique_ID": cuid,
                    "decision_id": decision_id,
                    "trial": trial,
                    "category": cat,
                    "location": location,
                    "goal": goal,
                    "offset": float(off_s),
                    "y": float(Y[b, c_idx]),
                    "distance_to_goal": float(dtg_per_bin[b]),
                    "steps_to_goal": float(steps_per_bin[b]),
                    "fold": fold,
                }
                for name in feature_sets:
                    row[f"mu_{name}"] = float(cluster_mu[name][b])
                rows.append(row)
    return pd.DataFrame(rows)


def _search_alpha_per_cluster(
    X,
    Y,
    train_trials,
    bin_trials,
    nav_per_bin,
    alpha_range,
    n_inner_folds,
    random_state,
):
    """Inner k-fold CV search for the best α per cluster on the outer fold's
    (navigation-only) training bins.

    Splits `train_trials` into `n_inner_folds`, fits PoissonRegressor at each α in
    `alpha_range`, accumulates `PoissonRegressor.score` (pseudo-D²) across inner
    folds, and picks the α maximising mean inner-fold score per cluster.

    Returns: 1D array of optimal α values, length = Y.shape[1].
    """
    rng = np.random.default_rng(random_state)
    inner_test_per_fold = np.array_split(rng.permutation(train_trials), n_inner_folds)
    n_clusters = Y.shape[1]
    scores = np.zeros((n_clusters, len(alpha_range)))
    for inner_test_trials in inner_test_per_fold:
        inner_train_trials = np.setdiff1d(train_trials, inner_test_trials)
        inner_train_mask = np.isin(bin_trials, inner_train_trials) & nav_per_bin
        inner_test_mask = np.isin(bin_trials, inner_test_trials) & nav_per_bin
        for a_i, alpha in enumerate(alpha_range):
            for c in range(n_clusters):  # each (alpha, cluster) fit is independent
                m = PoissonRegressor(alpha=alpha, max_iter=_MAX_ITER)
                m.fit(X[inner_train_mask], Y[inner_train_mask, c])
                scores[c, a_i] += m.score(X[inner_test_mask], Y[inner_test_mask, c])
    return alpha_range[scores.argmax(axis=1)]


_D2_COLS = ["cluster_unique_ID", "category", "offset", "variable", "d2", "n_bins", "n_decisions"]


def _decision_match_keys(samples_df, match_var="distance_to_goal", window=(-1, 0)):
    """Per-decision matching key from `match_var`.

    `window`:
        (lo, hi)   : nanmean of `match_var` over offsets in [lo, hi] (pre-decision default).
        "decision" : the value at the decision arrival bin only (offset 0).
    Returns one row per decision [decision_id, category, key]; decisions with no finite
    value are dropped.
    """
    if isinstance(window, str):
        if window != "decision":
            raise ValueError(f"window string must be 'decision'; got {window!r}")
        win = samples_df[np.isclose(samples_df.offset, 0.0, atol=1e-6)]
    else:
        lo, hi = window
        win = samples_df[(samples_df.offset >= lo) & (samples_df.offset <= hi)]
    keys = (
        win.groupby(["decision_id", "category"])[match_var]
        .apply(lambda s: np.nanmean(s.values) if np.isfinite(s.values).any() else np.nan)
        .reset_index(name="key")
    )
    return keys.dropna(subset=["key"]).reset_index(drop=True)


def _match_decisions(
    keys_df,
    method,
    n_bins=10,
    n_repeats=20,
    random_state=0,
    min_per_bin=1,
    tol=1e-2,
    min_floor=2,
):
    """Return a list of decision_id sets (one per repeat) matched across all categories
    present in `keys_df`.

    method="distribution": quantile-bin the pooled key; per bin subsample every category
        to the per-bin min count (random → benefits from n_repeats).
    method="mean": equalise category means to their median via greedy extreme-trimming
        (deterministic → single set, n_repeats ignored).
    """
    cats = sorted(keys_df.category.unique())
    if len(cats) < 2:
        return [set(keys_df.decision_id)]

    if method == "distribution":
        keys = keys_df.key.values
        edges = np.unique(np.nanquantile(keys, np.linspace(0, 1, n_bins + 1)))
        bins = (
            np.zeros(len(keys), int) if len(edges) < 2 else np.clip(np.digitize(keys, edges[1:-1]), 0, len(edges) - 2)
        )
        binned = keys_df.assign(_bin=bins)
        sets = []
        for r in range(n_repeats):
            rng = np.random.default_rng(random_state + r)
            kept = []
            for _, bin_df in binned.groupby("_bin"):
                counts = bin_df.groupby("category").size()
                if not all(c in counts.index for c in cats):
                    continue  # category missing in this bin → unmatchable, skip
                target = int(counts.min())
                if target < min_per_bin:
                    continue
                for c in cats:
                    ids = bin_df.loc[bin_df.category == c, "decision_id"].values
                    kept.extend(rng.choice(ids, size=target, replace=False).tolist())
            sets.append(set(kept))
        return sets

    if method == "mean":
        target = float(np.median(keys_df.groupby("category").key.mean().values))
        kept_ids = []
        for c in cats:
            cdf = keys_df[keys_df.category == c]
            ks, ids = cdf.key.tolist(), cdf.decision_id.tolist()
            while len(ks) > min_floor:
                m = float(np.mean(ks))
                if abs(m - target) <= tol:
                    break
                j = int(np.argmax(ks)) if m > target else int(np.argmin(ks))
                new_ks = ks[:j] + ks[j + 1 :]
                if abs(float(np.mean(new_ks)) - target) >= abs(m - target):
                    break  # dropping no longer helps
                ks, ids = new_ks, ids[:j] + ids[j + 1 :]
            kept_ids.extend(ids)
        return [set(kept_ids)]

    raise ValueError(f"method must be 'distribution' or 'mean'; got {method!r}")


def _aggregate_core(samples_df):
    """Pool test bins within each (cluster, category, offset) → long-form D² per variable.

    Vectorised: Poisson deviance is a per-row quantity that sums over a group, so unit
    deviances are computed in one numpy pass and totalled with a C-level groupby.sum
    (no per-group Python apply). Within a (cluster, category, offset) group each decision
    contributes exactly one bin, so n_decisions == n_bins == group size.
    """
    y = samples_df.y.values.astype(float)
    grouped = pd.DataFrame(
        {
            "cluster_unique_ID": pd.Categorical(samples_df.cluster_unique_ID.values),
            "category": pd.Categorical(samples_df.category.values),
            "offset": samples_df.offset.values,
            "d_null": _unit_poisson_deviance(y, samples_df.mu_null.values),
            "d_reduced_pd": _unit_poisson_deviance(y, samples_df.mu_reduced_pd.values),
            "d_reduced_dtg": _unit_poisson_deviance(y, samples_df.mu_reduced_dtg.values),
            "d_full": _unit_poisson_deviance(y, samples_df.mu_full.values),
        }
    ).groupby(["cluster_unique_ID", "category", "offset"], observed=True, sort=False)
    dev_cols = ["d_null", "d_reduced_pd", "d_reduced_dtg", "d_full"]
    summed = grouped[dev_cols].sum()
    summed["n_bins"] = grouped.size()
    summed = summed.reset_index()
    # plain str keys so per-session concat in aggregate_to_d2 doesn't fight categoricals
    summed["cluster_unique_ID"] = summed.cluster_unique_ID.astype(str)
    summed["category"] = summed.category.astype(str)

    def _d2(denom):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denom > 0, 1.0 - summed.d_full.values / denom, np.nan)

    parts = []
    for variable, denom_col in (
        ("place_direction", "d_reduced_pd"),
        ("distance_to_goal", "d_reduced_dtg"),
        ("combined", "d_null"),
    ):
        part = summed[["cluster_unique_ID", "category", "offset", "n_bins"]].copy()
        part["variable"] = variable
        part["d2"] = _d2(summed[denom_col].values)
        part["n_decisions"] = part["n_bins"]  # one bin per decision within a group
        parts.append(part)
    return pd.concat(parts, ignore_index=True)[_D2_COLS]


def _aggregate_matched(
    group_df, match, match_categories, match_var, match_window, match_n_bins, match_repeats, random_state
):
    """Match decisions within a single group (where decision_id is unique), then
    aggregate to D². Distribution matching is averaged over `match_repeats` random
    subsamples; mean matching is deterministic (single pass). Returns None if no
    decisions survive matching."""
    keys_df = _decision_match_keys(group_df, match_var=match_var, window=match_window)
    if match_categories is not None:
        keys_df = keys_df[keys_df.category.isin(list(match_categories))]
    if keys_df.empty:
        return None
    n_rep = 1 if match == "mean" else match_repeats
    reps = []
    for r in range(n_rep):
        dset = _match_decisions(keys_df, method=match, n_bins=match_n_bins, n_repeats=1, random_state=random_state + r)[
            0
        ]
        sub = group_df[group_df.decision_id.isin(dset)]
        if not sub.empty:
            reps.append(_aggregate_core(sub))
    if not reps:
        return None
    # average d2 (and matched counts) across repeats per cell
    return (
        pd.concat(reps, ignore_index=True)
        .groupby(["cluster_unique_ID", "category", "offset", "variable"], as_index=False)[
            ["d2", "n_bins", "n_decisions"]
        ]
        .mean()
    )


def aggregate_to_d2(
    samples_df,
    match=None,
    match_categories=None,
    match_var="distance_to_goal",
    match_window=(-1, 0),
    match_n_bins=10,
    match_repeats=20,
    match_within="session_name",
    random_state=0,
):
    """Aggregate sample-level df → unique Poisson D² per variable, optionally after
    matching the pre-decision DTG distribution across categories.

    Per (cluster, category, offset), held-out bins are pooled and:
        place_direction  : 1 - D_full / D_reduced_pd
        distance_to_goal : 1 - D_full / D_reduced_dtg
        combined         : 1 - D_full / D_null

    `match`:
        None           : pool all decisions (no control).
        "distribution" : equalise per-category DTG histograms (averaged over repeats).
        "mean"         : equalise per-category mean DTG (deterministic).
    Matching uses a per-decision key from `match_var`: mean over offsets in `match_window`
    (a tuple), or — with `match_window="decision"` — the value at the decision arrival bin
    (offset 0) only. Applied jointly across `match_categories` (restrict to the pair you
    compare, e.g. ("chose_structure", "chose_habit"); decisions in other categories are dropped).

    `match_within` (default "session_name") is the scope matching operates over. Since a
    cell's D² is pooled only from its own session's decisions, matching WITHIN each session
    is what makes each cell's cross-category comparison DTG-fair — global matching can leave
    per-session (hence per-cell) imbalance. Set `match_within=None` to match across the whole
    df instead (decisions are disambiguated by session so ids can't collide).

    Output: long-form [cluster_unique_ID, category, offset, variable, d2, n_bins, n_decisions]
    plus per-cluster metadata (subject_ID, session_name, maze_name, day_on_maze) where present.
    """
    if samples_df.empty:
        return pd.DataFrame(columns=_D2_COLS)
    if match is None:
        result = _aggregate_core(samples_df)
    else:
        _args = (match, match_categories, match_var, match_window, match_n_bins, match_repeats, random_state)
        if match_within is None:
            # global scope: make decision_id globally unique so cross-session ids can't merge
            work = samples_df.assign(
                decision_id=samples_df.session_name.astype(str) + "||" + samples_df.decision_id.astype(str)
            )
            result = _aggregate_matched(work, *_args)
            result = result if result is not None else pd.DataFrame(columns=_D2_COLS)
        else:
            # per-group (default per-session): decision_id is unique within a group; concatenate
            # each group's cells (cluster_unique_ID is group-bound so no cross-group dups)
            parts = [
                part
                for _, gdf in samples_df.groupby(match_within, sort=False)
                if (part := _aggregate_matched(gdf, *_args)) is not None and not part.empty
            ]
            result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=_D2_COLS)
    return _attach_cluster_metadata(result, samples_df)


def _attach_cluster_metadata(result, samples_df):
    """Join per-cluster metadata (subject_ID, session_name, maze_name, day_on_maze) onto the
    aggregated D² df. Each cluster_unique_ID maps to exactly one session/subject, so the
    lookup is well-defined. Needed by the cross-subject plot/stats (which group by subject_ID)."""
    meta_cols = [c for c in _META_COLS if c in samples_df.columns]
    if result.empty or not meta_cols:
        return result
    meta = samples_df[["cluster_unique_ID", *meta_cols]].drop_duplicates("cluster_unique_ID")
    return result.merge(meta, on="cluster_unique_ID", how="left")


def _unit_poisson_deviance(y, mu):
    """Per-row Poisson deviance (summing over rows gives the group deviance).
    y·log(y/mu) → 0 at y=0; mu floored at _MU_FLOOR before the log."""
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), _MU_FLOOR, None)
    log_term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return 2.0 * (log_term - (y - mu))


# %% Sample-level encoding across sessions


def _safe_session_encoding(session, **session_kwargs):
    """Run get_session_decision_aligned_encoding on one session, returning None on error
    (so one bad session can't abort an overnight batch)."""
    try:
        return get_session_decision_aligned_encoding(session, **session_kwargs)
    except Exception as e:
        print(f"Error processing {session.name}: {e!r}")
        return None


def get_decision_encoding_summary(
    maze_names=("maze_1", "maze_2"),
    days_on_maze="late",
    n_jobs=-1,
    save=False,
    verbose=True,
    **session_kwargs,
):
    """Run the session-level encoding over all sessions and concatenate the sample-level
    dfs. No D² aggregation or matching here — each row stays a held-out sample tagged with
    `session_name`, so per-session DTG matching + aggregation can be applied later via
    `aggregate_to_d2`.

    Sessions are loaded then processed with optional joblib parallelism (`n_jobs`), mirroring
    the other cross-session drivers in the codebase. `session_kwargs` are forwarded to
    `get_session_decision_aligned_encoding`.

    Returns the concatenated sample-level df. Sessions that error out are skipped (logged).
    """
    save_path = RESULTS_DIR / "decision_encoding_samples.parquet"
    if not save and save_path.exists():
        if verbose:
            print(f"Loading from {save_path}")
        return pd.read_parquet(save_path)

    sessions = gs.get_maze_sessions(
        subject_IDs=SUBJECT_IDS,
        maze_names=list(maze_names),
        days_on_maze=days_on_maze,
        with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "trials_df"],
        must_have_data=True,
    )
    if verbose:
        print(f"Processing {len(sessions)} sessions with n_jobs={n_jobs} ...")

    if n_jobs in (None, 1):
        dfs = [_safe_session_encoding(s, verbose=verbose, **session_kwargs) for s in sessions]
    else:
        dfs = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_safe_session_encoding)(s, **session_kwargs) for s in sessions
        )

    dfs = [d for d in dfs if d is not None and not d.empty]
    samples_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if save:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        samples_df.to_parquet(save_path)
        if verbose:
            print(f"Saved {len(samples_df)} rows from {len(dfs)} sessions to {save_path}")
    return samples_df


# %% Plotting


def plot_session_decision_encoding(
    samples_df,
    variable="place_direction",
    categories=("chose_structure", "chose_habit"),
    colors=None,
    match=None,
    match_var="distance_to_goal",
    match_window=(-1, 0),
    match_n_bins=10,
    match_repeats=20,
    match_within="session_name",
    random_state=0,
    window=(-1.5, 1.5),
    ax=None,
):
    """Mean ± SEM D² across clusters in one session, split by category.

    Takes the sample-level output of `get_session_decision_aligned_encoding` and
    aggregates internally via `aggregate_to_d2`, so the DTG-matching control (`match`,
    `match_*`) is a plot-time switch: None | "mean" | "distribution".

    `variable ∈ {"place_direction", "distance_to_goal", "combined"}` selects which
    unique-variance metric to plot. `window` (xlim) restricts the displayed offset range.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 3))
    if colors is None:
        colors = {"agree": "grey", "chose_structure": "blueviolet", "chose_habit": "hotpink"}

    summary_df = aggregate_to_d2(
        samples_df,
        match=match,
        match_categories=categories,  # match across exactly the categories being compared
        match_var=match_var,
        match_window=match_window,
        match_n_bins=match_n_bins,
        match_repeats=match_repeats,
        match_within=match_within,
        random_state=random_state,
    )
    df_var = summary_df[summary_df.variable == variable]
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0, color="k", ls="--", alpha=0.5)
    ax.axhline(0, color="k", ls="--", alpha=0.5)
    ax.set_xlabel("offset from decision arrival (s)")
    ax.set_ylabel(f"{variable}\nunique variance explained")

    for cat in categories:
        df = df_var[df_var.category == cat]
        if df.empty:
            continue
        agg = df.groupby("offset").d2.agg(["mean", "sem"])
        ax.plot(agg.index, agg["mean"], color=colors.get(cat, "k"), label=cat, lw=2)
        ax.fill_between(
            agg.index,
            agg["mean"] - agg["sem"],
            agg["mean"] + agg["sem"],
            color=colors.get(cat, "k"),
            alpha=0.2,
        )
    if window is not None:
        ax.set_xlim(*window)
    ax.legend(fontsize=8, frameon=False)


def _pairwise_category_stats(summary_df, categories, variable="place_direction", alpha=0.05, fdr_across="offsets"):
    """Per-offset paired t-tests across subjects between every pair of `categories`,
    BH-FDR corrected. Operates on the rows of `summary_df` matching `variable`.

    Per-subject value at each (category, offset) is the mean d² across that subject's
    clusters at that combination.

    fdr_across:
        "offsets"            : BH across offsets within each pair (each pair tested
                               independently of the others).
        "pairs_and_offsets"  : BH jointly over the full set of (pair × offset) p-values.
        "none"               : raw p-values, no correction (p_fdr = p_raw).

    Returns long df with cols: pair_a, pair_b, offset, t, p_raw, p_fdr, n_subjects, significant.
    """
    df_var = summary_df[summary_df.variable == variable]
    per_subj = df_var.groupby(["category", "subject_ID", "offset"]).d2.mean()
    rows = []
    for a, b in combinations(categories, 2):
        try:
            a_df = per_subj.loc[a].unstack("offset")
            b_df = per_subj.loc[b].unstack("offset")
        except KeyError:
            continue
        common_subjects = a_df.index.intersection(b_df.index)
        if len(common_subjects) < 2:
            continue
        a_df, b_df = a_df.loc[common_subjects], b_df.loc[common_subjects]
        common_offsets = a_df.columns.intersection(b_df.columns)
        for o in common_offsets:
            paired = pd.DataFrame({"a": a_df[o], "b": b_df[o]}).dropna()
            if len(paired) < 2:
                t, p = np.nan, np.nan
            else:
                t, p = ttest_rel(paired["a"], paired["b"])
            rows.append(
                {
                    "pair_a": a,
                    "pair_b": b,
                    "offset": float(o),
                    "t": float(t) if np.isfinite(t) else np.nan,
                    "p_raw": float(p) if np.isfinite(p) else np.nan,
                    "n_subjects": int(len(common_subjects)),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["p_fdr"] = np.nan
    if fdr_across == "none":
        df["p_fdr"] = df["p_raw"]
    elif fdr_across == "offsets":
        for (a, b), sub in df.groupby(["pair_a", "pair_b"], sort=False):
            mask = sub.p_raw.notna()
            if mask.any():
                df.loc[sub.index[mask], "p_fdr"] = false_discovery_control(sub.loc[mask, "p_raw"].values)
    elif fdr_across == "pairs_and_offsets":
        mask = df.p_raw.notna()
        if mask.any():
            df.loc[df.index[mask], "p_fdr"] = false_discovery_control(df.loc[mask, "p_raw"].values)
    else:
        raise ValueError(f"fdr_across must be 'offsets', 'pairs_and_offsets', or 'none'; got {fdr_across!r}")

    df["significant"] = df.p_fdr.lt(alpha).fillna(False)
    return df


def plot_decision_encoding(
    summary_df,
    variable="place_direction",
    categories=("chose_structure", "chose_habit"),  # agree
    colors=None,
    weight_by="uniform",  # "uniform" | "n_bins"
    stats_alpha=0.05,
    fdr_across="offsets",  # "offsets" | "pairs_and_offsets" | "none"
    ax=None,
):
    """Mean ± SEM D² across subjects, split by category, with pairwise paired-t
    significance overlaid.

    `variable ∈ {"place_direction", "distance_to_goal", "combined"}` selects the
    unique-variance metric to plot.

    `fdr_across` controls multiple-comparison correction scope (see
    `_pairwise_category_stats`):
        "offsets"           : BH across offsets within each pair (default).
        "pairs_and_offsets" : BH jointly over the full pair × offset grid.
        "none"              : no correction.

    Returns stats_df from `_pairwise_category_stats` for inspection.
    """
    category_abbr = {"agree": "A", "chose_structure": "S", "chose_habit": "H", "chose_neither": "N"}
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 3))
    if colors is None:
        colors = {"agree": "grey", "chose_structure": "blueviolet", "chose_habit": "hotpink"}

    df_var = summary_df[summary_df.variable == variable]
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0, color="k", ls="--", alpha=0.5)
    ax.axhline(0, color="k", ls="--", alpha=0.5)
    ax.set_xlabel("offset from decision arrival (s)")
    ax.set_ylabel(f"{variable}\nunique variance explained")

    for cat in categories:
        df = df_var[df_var.category == cat]
        if df.empty:
            continue
        if weight_by == "uniform":
            per_subj = df.groupby(["subject_ID", "offset"]).d2.mean().unstack(level=0)
        elif weight_by == "n_bins":
            df = df.copy()
            df["d2_w"] = df.d2 * df.n_bins
            per_subj = (
                df.groupby(["subject_ID", "offset"])
                .apply(lambda g: g.d2_w.sum() / g.n_bins.sum() if g.n_bins.sum() > 0 else np.nan)
                .unstack(level=0)
            )
        else:
            raise ValueError(f"Unknown weight_by: {weight_by}")
        mean = per_subj.mean(axis=1)
        sem = per_subj.sem(axis=1)
        ax.plot(mean.index, mean.values, color=colors.get(cat, "k"), label=cat, lw=2)
        ax.fill_between(
            mean.index,
            mean.values - sem.values,
            mean.values + sem.values,
            color=colors.get(cat, "k"),
            alpha=0.2,
        )

    # pairwise significance overlay
    stats_df = _pairwise_category_stats(
        summary_df,
        categories=categories,
        variable=variable,
        alpha=stats_alpha,
        fdr_across=fdr_across,
    )
    if not stats_df.empty:
        ymin, ymax = ax.get_ylim()
        row_h = (ymax - ymin) * 0.05
        offsets_arr = np.sort(stats_df.offset.unique())
        seg_w = float(np.median(np.diff(offsets_arr))) if len(offsets_arr) > 1 else 0.1
        pairs = list(combinations(categories, 2))
        for i, (a, b) in enumerate(pairs):
            if a not in colors or b not in colors:
                continue
            pair_color = tuple(np.mean([mcolors.to_rgb(colors[a]), mcolors.to_rgb(colors[b])], axis=0))
            y = ymax + row_h * (i + 1)
            sig = stats_df[(stats_df.pair_a == a) & (stats_df.pair_b == b) & stats_df.significant]
            for _, row in sig.iterrows():
                ax.hlines(y, row.offset - seg_w / 2, row.offset + seg_w / 2, color=pair_color, lw=3)
            ax.text(
                offsets_arr.max() + (offsets_arr.max() - offsets_arr.min()) * 0.02,
                y,
                f"{category_abbr.get(a, a)}↔{category_abbr.get(b, b)}",
                va="center",
                fontsize=7,
                color=pair_color,
            )
        ax.set_ylim(ymin, ymax + row_h * (len(pairs) + 1))

    ax.legend(fontsize=8, frameon=False)
    return stats_df
