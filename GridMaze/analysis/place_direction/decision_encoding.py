"""
Decision-aligned Poisson encoding of place-direction and distance-to-goal in mFC.

Tests whether mFC encoding of place-direction (PD) and distance-to-goal (DTG) is
stronger at decision points where structure != habit and the animal chose the
structure action vs the habit action (with structure == habit decisions as a
baseline).

Per-cluster Poisson GLM is fit four times on training bins, with speed and
trial_phase always present as nuisance regressors:
    null         : speed + trial_phase
    reduced_pd   : speed + trial_phase + DTG bases
    reduced_dtg  : speed + trial_phase + PD bases
    full         : speed + trial_phase + PD bases + DTG bases

PD bases are the top 20 PCs of place-direction tuning learned from all OTHER same-maze
sessions; DTG bases are 8 gamma functions over the geodesic distance domain.
Per (cluster, category, offset), held-out test bins are pooled and unique D² is
computed by deviance ratio against the reduced models:
    d2_place_direction  = 1 - D(y, mu_full) / D(y, mu_reduced_pd)
    d2_distance_to_goal = 1 - D(y, mu_full) / D(y, mu_reduced_dtg)
    d2_combined         = 1 - D(y, mu_full) / D(y, mu_null)
The bare null adds a shared-variance diagnostic (combined ≥ pd_unique + dtg_unique
when PD and DTG share variance).

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


# %% Session-level decision-aligned encoding


def get_session_decision_aligned_pd_encoding(
    session,
    resolution=0.2,
    window=(-3.0, 3.0),
    n_bases=20,
    dtg_metric=("distance_to_goal", "geodesic"),
    dtg_n_bases=8,
    dtg_basis="gamma",
    dtg_max_distance_pct=90,
    n_folds=10,
    min_spikes=300,
    max_steps_to_goal=30,
    include_multi_units=False,
    alpha=1.0,
    alpha_range=np.logspace(-3, 3, 10),
    n_inner_folds=5,
    categories=("agree", "chose_structure", "chose_habit"),
    random_state=0,
    verbose=False,
):
    """
    Fit per-cluster Poisson GLMs under trial-CV and score held-out bins at offsets ±
    `window` around each decision arrival time. Four model variants are fit per
    cluster per fold:
        null         : speed + trial_phase
        reduced_pd   : speed + trial_phase + DTG bases
        reduced_dtg  : speed + trial_phase + PD bases
        full         : speed + trial_phase + PD bases + DTG bases

    Unique deviance explained per variable (PD, DTG) is derived from full vs the
    corresponding reduced model; the bare null gives a joint-encoding sanity check.

    `alpha` can be a scalar (fixed regularisation) or the string `"opt"`, in which case
    each cluster's α is picked by inner k-fold CV (`n_inner_folds` folds over training
    trials) over `alpha_range`, independently per feature set. Scoring uses
    `PoissonRegressor.score` (pseudo-D²).

    Returns
    -------
    summary_df : long-form, one row per (cluster_unique_ID, category, offset, variable)
        with `variable ∈ {"place_direction", "distance_to_goal", "combined"}`, `d2`,
        `n_bins`, `n_decisions`, plus session metadata.
    """
    # 1) PD bases: all OTHER same-maze sessions (cached, see bases.get_pd_heatmaps_df)
    bases_df = pdb.get_session_pd_bases(session, n_bases=n_bases, dim_red="pca")

    # 2) Binned navigation + spike data (all phases, DTG plumbed through)
    navigation_spikes_df = _build_navigation_spikes_df(
        session,
        resolution=resolution,
        max_steps_to_goal=max_steps_to_goal,
        min_spikes=min_spikes,
        include_multi_units=include_multi_units,
    )
    simple_maze = session.simple_maze()

    # 3) Feature blocks
    # speed
    X_speed = navigation_spikes_df[("speed", "")].fillna(0).values.reshape(-1, 1)
    # phase: 3-level one-hot, drop ITI (intercept absorbs it)
    phase_cat = pd.Categorical(
        navigation_spikes_df[("trial_phase", "")].values,
        categories=["ITI", "navigation", "reward_consumption"],
    )
    X_phase = pd.get_dummies(phase_cat, drop_first=True).values.astype(float)
    # PD: 20-dim projection of place-direction one-hot onto PD-PC bases
    pd_strings = navigation_spikes_df.place_direction.values.astype(str)
    X_pd_onehot = convert.place_direction2onehot(pd_strings, simple_maze)
    X_pd = _project_onehot_onto_bases(X_pd_onehot, simple_maze, bases_df)
    # DTG: gamma bases over the geodesic-distance domain (max at 90th pct)
    dtg_max = dd.get_distance_percentile(dtg_metric, percentile=dtg_max_distance_pct)
    dtg_basis_fn = db.distance_basis_generator(
        n_bases=dtg_n_bases, basis=dtg_basis, btype="distance", max_distance=dtg_max,
    )
    dtg_values = navigation_spikes_df[dtg_metric].values.astype(float)
    # NaN DTG (e.g. inter-trial bins) → project to 0
    X_dtg = np.nan_to_num(dtg_basis_fn(dtg_values), nan=0.0)

    # 4) Assemble feature_sets dict {name → matrix}
    nuisance = np.hstack([X_speed, X_phase])
    feature_sets = {
        "null":        nuisance,
        "reduced_pd":  np.hstack([nuisance, X_dtg]),
        "reduced_dtg": np.hstack([nuisance, X_pd]),
        "full":        np.hstack([nuisance, X_pd, X_dtg]),
    }
    Y = navigation_spikes_df.spike_count.values  # (n_bins, n_clusters)
    cluster_unique_IDs = list(navigation_spikes_df.spike_count.columns)

    # 5) Decision events with categories
    decision_df = tdec.get_session_decision_times(session, decision_points_only=True)
    if decision_df is None or decision_df.empty:
        return pd.DataFrame()
    decision_df = decision_df[decision_df.category.isin(list(categories))].reset_index(drop=True)
    # map each decision's session-clock arrival time to its bin index in navigation_spikes_df
    bin_times = navigation_spikes_df[("time", "")].values
    decision_df = _attach_bin_index(decision_df, bin_times)
    decision_df = decision_df.dropna(subset=["bin_idx"]).reset_index(drop=True)
    decision_df["bin_idx"] = decision_df["bin_idx"].astype(int)

    # 6) Random k-fold split by trial number
    all_trials = navigation_spikes_df.trial.dropna().unique()
    rng = np.random.default_rng(seed=random_state)
    test_trials_per_fold = np.array_split(rng.permutation(all_trials), n_folds)

    # 7) Run per-fold per-cluster fitting + scoring
    offsets_s = np.round(np.arange(window[0], window[1] + resolution / 2, resolution), 6)
    offsets_bins = np.round(offsets_s / resolution).astype(int)
    bin_trials = navigation_spikes_df.trial.values.ravel()

    fold_results = [
        _process_fold(
            fold=i,
            test_trials=test_trials_per_fold[i],
            all_trials=all_trials,
            bin_trials=bin_trials,
            decision_df=decision_df,
            feature_sets=feature_sets,
            Y=Y,
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

    # 8) Aggregate to D2 per (cluster, category, offset, variable)
    summary_df = _aggregate_to_d2(samples_df)
    summary_df["subject_ID"] = session.subject_ID
    summary_df["maze_name"] = session.maze_name
    summary_df["day_on_maze"] = session.day_on_maze
    return summary_df


# %% Helpers


def _build_navigation_spikes_df(
    session,
    resolution,
    max_steps_to_goal,
    min_spikes,
    include_multi_units,
):
    """Bin + filter session data into a per-bin DataFrame with PD string, speed,
    trial_phase, and distance_to_goal.

    Keeps all three trial_phases (navigation, reward_consumption, ITI) so post-decision
    bins near the goal aren't silently dropped — phase enters the model as a nuisance
    regressor instead. `max_steps_to_goal` is applied inline only to navigation rows
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
    # keep all phases, no at-goal exclusion — phase is a nuisance regressor downstream
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
        _sp = df.spike_count
        reject = _sp.columns[_sp.sum(axis=0) < min_spikes].values
        df = df.drop(columns=reject, level=1, axis=1)
    return df.reset_index(drop=True)


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
    """For each decision (session-clock time), find the nearest bin's index."""
    times = decision_df.time.values
    idxs = np.searchsorted(bin_times, times)
    # snap to nearest of {idx-1, idx} where in bounds
    nearest = np.full_like(idxs, fill_value=-1, dtype=float)
    for i, (t, k) in enumerate(zip(times, idxs)):
        candidates = [c for c in (k - 1, k) if 0 <= c < len(bin_times)]
        if not candidates:
            continue
        diffs = [abs(bin_times[c] - t) for c in candidates]
        nearest[i] = candidates[int(np.argmin(diffs))]
    out = decision_df.copy()
    out["bin_idx"] = np.where(nearest >= 0, nearest, np.nan)
    return out


def _process_fold(
    fold,
    test_trials,
    all_trials,
    bin_trials,
    decision_df,
    feature_sets,
    Y,
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
    Bins with NaN trial label (inter-trial filler) match neither set, so they're
    excluded from train and from scoring.

    If `alpha == "opt"`, each cluster's α is selected by inner k-fold CV over training
    trials (`_search_alpha_per_cluster`) independently per feature set.
    """
    if verbose:
        print(f"  fold {fold}")
    train_trials = np.setdiff1d(all_trials, test_trials)
    train_mask = np.isin(bin_trials, train_trials)

    test_decisions = decision_df[decision_df.trial.isin(test_trials)].reset_index(drop=True)
    if test_decisions.empty:
        return None

    n_bins = len(bin_trials)
    centre_bins = test_decisions.bin_idx.values.astype(int)
    bin_grid = centre_bins[:, None] + offsets_bins[None, :]  # (n_decisions, n_offsets)
    in_range = (bin_grid >= 0) & (bin_grid < n_bins)
    # only score bins whose host trial is in the held-out set (no train leakage)
    safe_grid = np.clip(bin_grid, 0, n_bins - 1)
    bin_trial_of_grid = np.where(in_range, bin_trials[safe_grid], np.nan)
    in_test_trial = np.isin(bin_trial_of_grid, test_trials)
    valid = in_range & in_test_trial

    # per-feature-set per-cluster α: fixed scalar or inner-CV searched
    if alpha == "opt":
        if verbose:
            print(f"    inner-CV α search (fold {fold})")
        alphas_per_set = {
            name: _search_alpha_per_cluster(
                X, Y, train_trials, bin_trials, alpha_range, n_inner_folds, random_state + fold,
            )
            for name, X in feature_sets.items()
        }
    else:
        alphas_per_set = {
            name: np.full(len(cluster_unique_IDs), float(alpha)) for name in feature_sets
        }

    rows = []
    for c_idx, cuid in enumerate(cluster_unique_IDs):
        y_train = Y[train_mask, c_idx]
        # fit each feature set for this cluster, predict over all bins
        cluster_mu = {}
        for name, X in feature_sets.items():
            m = PoissonRegressor(alpha=alphas_per_set[name][c_idx], max_iter=10_000)
            m.fit(X[train_mask], y_train)
            cluster_mu[name] = m.predict(X)
        # collect per-(decision, offset) records
        for d_i in range(bin_grid.shape[0]):
            cat = test_decisions.category.iloc[d_i]
            trial = test_decisions.trial.iloc[d_i]
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
                    "offset": float(off_s),
                    "y": float(Y[b, c_idx]),
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
    alpha_range,
    n_inner_folds,
    random_state,
):
    """Inner k-fold CV search for the best α per cluster on the outer fold's
    training bins.

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
        itr_mask = np.isin(bin_trials, inner_train_trials)
        ite_mask = np.isin(bin_trials, inner_test_trials)
        for a_i, alpha in enumerate(alpha_range):
            for c in range(n_clusters):
                m = PoissonRegressor(alpha=alpha, max_iter=10_000)
                m.fit(X[itr_mask], Y[itr_mask, c])
                scores[c, a_i] += m.score(X[ite_mask], Y[ite_mask, c])
    return alpha_range[scores.argmax(axis=1)]


def _aggregate_to_d2(samples_df):
    """Pool test bins within each (cluster, category, offset) and compute unique
    Poisson D² per variable (place_direction, distance_to_goal, combined).

    Output is long-form with one row per (cluster, category, offset, variable):
        place_direction  : 1 - D_full / D_reduced_pd
        distance_to_goal : 1 - D_full / D_reduced_dtg
        combined         : 1 - D_full / D_null
    """
    if samples_df.empty:
        return pd.DataFrame(
            columns=["cluster_unique_ID", "category", "offset", "variable", "d2", "n_bins", "n_decisions"]
        )
    return (
        samples_df.groupby(["cluster_unique_ID", "category", "offset"])
        .apply(_d2_rows_from_group, include_groups=False)
        .reset_index()
    )


def _d2_rows_from_group(g):
    y = g.y.values.astype(float)
    d_null = _poisson_deviance(y, np.clip(g.mu_null.values.astype(float), 1e-12, None))
    d_reduced_pd = _poisson_deviance(y, np.clip(g.mu_reduced_pd.values.astype(float), 1e-12, None))
    d_reduced_dtg = _poisson_deviance(y, np.clip(g.mu_reduced_dtg.values.astype(float), 1e-12, None))
    d_full = _poisson_deviance(y, np.clip(g.mu_full.values.astype(float), 1e-12, None))
    n_bins = int(len(y))
    n_decisions = int(g.decision_id.nunique())

    def _d2(denom):
        return 1.0 - d_full / denom if denom > 0 else np.nan

    return pd.DataFrame(
        {
            "d2": [_d2(d_reduced_pd), _d2(d_reduced_dtg), _d2(d_null)],
            "n_bins": [n_bins] * 3,
            "n_decisions": [n_decisions] * 3,
        },
        index=pd.Index(["place_direction", "distance_to_goal", "combined"], name="variable"),
    )


def _poisson_deviance(y, mu):
    """Sum of unit Poisson deviances. y log(y/mu) → 0 at y=0."""
    y_safe = np.where(y > 0, y, 1.0)
    return 2.0 * np.sum(np.where(y > 0, y * np.log(y_safe / mu), 0.0) - (y - mu))


# %% Summary across sessions


def get_decision_encoding_summary(
    maze_names=("maze_1", "maze_2"),
    days_on_maze="late",
    save=False,
    verbose=True,
    **session_kwargs,
):
    """Loop over subjects, loading and processing each subject's sessions in
    turn, and concatenate per-session summary_dfs at the end.

    Memory profile: at most one subject's sessions (~14 for late maze_1 + maze_2)
    are held in memory at a time, rather than all subjects' sessions upfront.

    Returns the cross-session summary_df. Failed sessions are printed at the end.
    """
    save_path = RESULTS_DIR / "decision_encoding_summary.parquet"
    if not save and save_path.exists():
        if verbose:
            print(f"Loading from {save_path}")
        return pd.read_parquet(save_path)

    summaries, failed = [], []
    for subject in SUBJECT_IDS:
        if verbose:
            print(f"=== subject {subject} ===")
        try:
            sessions = gs.get_maze_sessions(
                subject_IDs=[subject],
                maze_names=list(maze_names),
                days_on_maze=days_on_maze,
                with_data=[
                    "navigation_df",
                    "navigation_spike_counts_df",
                    "cluster_metrics",
                    "trials_df",
                ],
                must_have_data=True,
            )
        except FileNotFoundError:
            if verbose:
                print(f"  skipping {subject}: missing data")
            continue
        for s in sessions:
            if verbose:
                print(s.name)
            try:
                summary_df = get_session_decision_aligned_pd_encoding(s, verbose=True, **session_kwargs)
                if summary_df.empty:
                    continue
                summary_df["session_name"] = s.name
                summaries.append(summary_df)
            except Exception as e:
                print(f"Error processing {s.name}: {e}")
                failed.append(s.name)
    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    if save:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_parquet(save_path)
    if failed:
        print(f"Failed sessions ({len(failed)}): {failed}")
    return summary_df


# %% Plotting


_VARIABLE_YLABEL = {
    "place_direction": "PD deviance explained (D²)",
    "distance_to_goal": "DTG deviance explained (D²)",
    "combined": "PD+DTG deviance explained (D²)",
}


def plot_session_decision_encoding(
    summary_df,
    variable="place_direction",
    categories=("agree", "chose_structure", "chose_habit"),
    colors=None,
    ax=None,
):
    """Mean ± SEM D² across clusters in one session, split by category.

    `variable ∈ {"place_direction", "distance_to_goal", "combined"}` selects which
    unique-variance metric to plot. Expects the single-session output of
    `get_session_decision_aligned_pd_encoding`.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 3))
    if colors is None:
        colors = {"agree": "grey", "chose_structure": "blueviolet", "chose_habit": "hotpink"}

    df_var = summary_df[summary_df.variable == variable]
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0, color="k", ls="--", alpha=0.5)
    ax.axhline(0, color="k", ls="--", alpha=0.5)
    ax.set_xlabel("offset from decision arrival (s)")
    ax.set_ylabel(_VARIABLE_YLABEL.get(variable, "D²"))

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
    ax.set_ylabel(_VARIABLE_YLABEL.get(variable, "D²"))

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
        summary_df, categories=categories, variable=variable, alpha=stats_alpha, fdr_across=fdr_across,
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
