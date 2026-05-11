""" """

# %% Imports
import json
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from matplotlib_venn import venn2
from scipy import stats
from statsmodels.stats.multitest import multipletests

from GridMaze.analysis.neGLM import load_model_sets as lms

# %% Global Variables

from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

# %% Functions


def load_null_data():
    ridge_df, perm_df = lms.load_model_set_rotation_null(
        "variance_explained_null",
        maze_names=["maze_1", "maze_2", "rooms_maze"],
        all_completed=True,
    )
    return ridge_df, perm_df


def load_data():
    results_df = lms.load_model_set_cv_scores("variance_explained_all_sessions")
    return results_df


def get_cpd_df(df, r2_thres=0.075, outlier_thres=90):
    """Per-cell unique variance explained by each feature, averaged over folds (in %)."""
    _df = _filter_nan_score_cells(df)
    cell_idx = ["subject_ID", "maze_name", "day_on_maze", "cluster_unique_ID"]
    cpd_df = _cpd_per_fold(_df, r2_thres=r2_thres).groupby(level=cell_idx).mean()
    if outlier_thres is not None:
        cpd_df = cpd_df[cpd_df.lt(outlier_thres).all(axis=1)]
    return cpd_df


def _filter_nan_score_cells(df):
    _df = df.copy()
    nan_rows = _df[_df.cv_score.isna()]
    if len(nan_rows) > 0:
        reject_cells = nan_rows.cluster_unique_ID.unique()
        _df = _df[~_df.cluster_unique_ID.isin(reject_cells)]
        return _df
    else:
        return _df


def _cpd_per_fold(df, r2_thres=0.075):
    """Per-(cell, fold) CPD table: full_model - remove_<feature>, in %.

    If r2_thres is set, drops clusters whose full-model R² (averaged over folds) is
    below threshold before computing CPD.
    """
    cell_idx = ["subject_ID", "maze_name", "day_on_maze", "cluster_unique_ID"]
    _df = df.copy()
    if r2_thres is not None:
        full_r2 = _df[_df.model_name == "full_model"].groupby("cluster_unique_ID").cv_score.mean()
        keep = full_r2[full_r2 >= r2_thres].index
        _df = _df[_df.cluster_unique_ID.isin(keep)]

    wide = _df.set_index(cell_idx + ["fold", "model_name"])["cv_score"].unstack("model_name")
    cpd = pd.DataFrame(index=wide.index)
    for m in [c for c in wide.columns if c.startswith("remove_")]:
        feature = m.split("remove_", 1)[1]
        cpd[feature] = (wide["full_model"] - wide[m]).mul(100)
    return cpd


def get_feature_tuned_df(df, r2_thres=0.075, alpha=0.01, mc_method=None):
    """Per-cell boolean tuning to each feature via one-sided t-test of CPD across folds.

    For each (cell, feature) pair, tests H1: mean per-fold CPD > 0. Returns a bool
    DataFrame indexed by cell with one column per feature; True iff p < `alpha`.

    Cell selection follows `get_cpd_df` (full-model R² ≥ `r2_thres`); no further
    full-model significance filter is applied.

    If `mc_method` is set (e.g. "fdr_bh", "bonferroni"), p-values are corrected across
    cells within each feature column before thresholding.
    """
    cell_idx = ["subject_ID", "maze_name", "day_on_maze", "cluster_unique_ID"]
    cpd_per_fold = _cpd_per_fold(df, r2_thres=r2_thres)
    grouped = cpd_per_fold.groupby(level=cell_idx)
    means = grouped.mean()
    n_folds = grouped.size().iloc[0]
    sems = grouped.std(ddof=1) / np.sqrt(n_folds)
    t_stats = means / sems
    p_vals = pd.DataFrame(
        stats.t.sf(t_stats, df=n_folds - 1),
        index=means.index,
        columns=means.columns,
    )
    if mc_method is not None:
        for col in p_vals.columns:
            p_vals[col] = multipletests(p_vals[col].values, method=mc_method, alpha=alpha)[1]
    return p_vals.lt(alpha)


def get_null(
    perm_df,
    r2_thres=0.075,
    metric="mean_selectivity",
    n_jobs=-1,
):
    perm_groups = dict(tuple(perm_df.groupby("permutation")))
    n = perm_df.permutation.max() + 1
    null = Parallel(n_jobs=n_jobs)(delayed(_null_one_perm)(perm_groups[i], r2_thres, metric) for i in range(n))
    return np.asarray(null)


def _null_one_perm(_df, r2_thres, metric):
    cpd_df = get_cpd_df(_df, r2_thres=r2_thres)
    return _subject_mean_metric(cpd_df, metric).mean()


def _subject_mean_metric(cpd_df, metric, features=("distance_to_goal", "place_direction")):
    """Per-subject metric values (random-effects unit). Returns array of length n_subjects."""
    f1, f2 = features
    out = []
    for _, sub in cpd_df.groupby(level="subject_ID"):
        uv_a = sub[f1].values
        uv_b = sub[f2].values
        if metric == "mean_selectivity":
            out.append(mean_selectivity(uv_a, uv_b))
        elif metric == "correlation":
            out.append(correlation(uv_a, uv_b))
        else:
            raise ValueError(f"Invalid metric: {metric}")
    return np.asarray(out)


def correlation(uv_a, uv_b):
    """Pearson r between two per-neuron unique-variance vectors."""
    return np.corrcoef(uv_a, uv_b)[0, 1]


def mean_selectivity(uv_a, uv_b):
    """Mean across-neuron selectivity index.
    SI = 1 when a cell loads on one feature only, 0 when equal on both.
    """
    return np.mean(np.abs(uv_a - uv_b) / (np.abs(uv_a) + np.abs(uv_b) + 1e-12))


def plot_feature_venn(
    feature_tuned_df,
    features=("distance_to_goal", "place_direction"),
    colors=("crimson", "royalblue"),
    alpha=0.5,
    ax=None,
):
    """Venn diagram of cell counts tuned to each feature (and their overlap)."""
    f1, f2 = features
    a = feature_tuned_df[f1].to_numpy()
    b = feature_tuned_df[f2].to_numpy()
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(2, 2))
    venn2(
        subsets=(int((a & ~b).sum()), int((~a & b).sum()), int((a & b).sum())),
        set_labels=features,
        set_colors=colors,
        alpha=alpha,
        ax=ax,
    )


def plot_cpd_2d(
    cpd_df,
    features=("distance_to_goal", "place_direction"),
    n_bins=30,
    cmap="rocket_r",
    pthresh=0.02,
    vmax=None,
    scatter_color=".15",
    scatter_size=5,
    scatter_alpha=0.4,
    xlims=None,
    ylims=None,
    ax=None,
):
    """2D histogram + scatter of unique-variance for two features. All cells, no filtering."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(3, 2.5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", linestyle="--", alpha=0.4)
    ax.axvline(0, color="k", linestyle="--", alpha=0.4)
    f1, f2 = features
    ax.set_xlabel(f"{f1} (%)")
    ax.set_ylabel(f"{f2} (%)")

    x = cpd_df[f1]
    y = cpd_df[f2]
    sns.scatterplot(
        x=x,
        y=y,
        s=scatter_size,
        color=scatter_color,
        alpha=scatter_alpha,
        edgecolor="none",
        ax=ax,
    )
    sns.histplot(
        x=x,
        y=y,
        bins=n_bins,
        pthresh=pthresh,
        cmap=cmap,
        vmax=vmax,
        cbar=True,
        cbar_kws={"shrink": 0.5, "label": "neurons"},
        ax=ax,
    )
    cax = ax.figure.axes[-1]
    for spine in cax.spines.values():
        spine.set_visible(False)
    if xlims is not None:
        ax.set_xlim(*xlims)
    if ylims is not None:
        ax.set_ylim(*ylims)

    si_per_subj = _subject_mean_metric(cpd_df, "mean_selectivity", features)
    si_mean = si_per_subj.mean()
    si_sem = si_per_subj.std(ddof=1) / np.sqrt(len(si_per_subj))
    ax.text(
        0.97,
        0.97,
        f"SI = {si_mean:.2f} ± {si_sem:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )


def plot_variance_explained(
    cpd_df,
    features=("distance_to_goal", "place_direction"),
    print_stats=True,
    plot_single_subject=True,
    marker_color=("crimson", "royalblue"),
    subject_line_color="grey",
    subject_line_alpha=0.3,
    ax=None,
):
    """Per-subject mean unique variance explained for each feature."""
    if ax is None:
        _, ax = plt.subplots(figsize=(2, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("features")
    ax.set_ylabel("unique variance explained (%)")

    df = cpd_df.copy()
    if features != "all":
        df = df[list(features)]
        order = list(features)
    else:
        order = list(df.columns)
    long_df = df.stack().rename("score").reset_index()
    long_df = long_df.rename(columns={long_df.columns[-2]: "feature"})
    subject_av = long_df.groupby(["subject_ID", "feature"])["score"].mean().reset_index()
    if plot_single_subject:
        sns.lineplot(
            data=subject_av,
            x="feature",
            y="score",
            units="subject_ID",
            estimator=None,
            sort=False,
            color=subject_line_color,
            alpha=subject_line_alpha,
            linewidth=2,
            ax=ax,
        )
    sns.pointplot(
        data=subject_av,
        x="feature",
        y="score",
        hue="feature",
        order=order,
        hue_order=order,
        palette=dict(zip(order, marker_color)),
        errorbar="se",
        linestyle="none",
        legend=False,
        alpha=1,
        ax=ax,
    )
    ax.set_xlim(-0.3, len(order) - 0.7)
    ax.tick_params(axis="x", rotation=30)
    if print_stats:
        print(_variance_explained_stats(cpd_df))


def _variance_explained_stats(cpd_df):
    """Per-feature one-sided t-test (>0) on per-subject mean CPD, FDR-corrected across features."""
    _df = cpd_df.groupby(level="subject_ID").mean()
    results = []
    for feature in _df.columns:
        t_stat, p_val = stats.ttest_1samp(_df[feature], 0, alternative="greater")
        results.append({"feature": feature, "t_stat": t_stat, "p_val": p_val})
    stats_df = pd.DataFrame(results)
    _, stats_df["p_val_corr"], _, _ = multipletests(stats_df["p_val"], method="fdr_bh")
    return stats_df


def plot_rotation_null(
    cv_df,
    ridge_df,
    perm_df,
    metric="mean_selectivity",
    features=("distance_to_goal", "place_direction"),
    r2_thres=0.075,
    n_bins=40,
    poisson_color="C3",
    ridge_color="C0",
    null_color="k",
    print_stats=True,
    ax=None,
):
    """Histogram the rotation null vs true Poisson + Ridge values for a chosen metric.

    Random-effects: subject is the unit of replication. The metric is computed per
    subject and averaged across subjects, both for the true data (Poisson, Ridge) and
    for each rotation permutation. The null distribution is over subject-mean metrics.

    - cv_df:    Poisson D² on true held-out spikes (headline embedding score).
    - ridge_df: Ridge R² on true held-out spikes (matched baseline for the rotation null).
    - perm_df:  Ridge R² on Haar-rotated held-out spikes — null distribution.

    Two-sided p-values are computed against the null mean.
    """
    cv_cpd = get_cpd_df(cv_df, r2_thres=r2_thres)
    ridge_cpd = get_cpd_df(ridge_df, r2_thres=r2_thres)
    poisson_per_subj = _subject_mean_metric(cv_cpd, metric, features)
    ridge_per_subj = _subject_mean_metric(ridge_cpd, metric, features)
    poisson_true = poisson_per_subj.mean()
    ridge_true = ridge_per_subj.mean()
    poisson_sem = poisson_per_subj.std(ddof=1) / np.sqrt(len(poisson_per_subj))
    ridge_sem = ridge_per_subj.std(ddof=1) / np.sqrt(len(ridge_per_subj))

    null = get_null(perm_df, r2_thres=r2_thres, metric=metric)

    centre = null.mean()
    p_poisson = (np.abs(null - centre) >= np.abs(poisson_true - centre)).mean()
    p_ridge = (np.abs(null - centre) >= np.abs(ridge_true - centre)).mean()

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.spines[["top", "right"]].set_visible(False)
    counts, edges = np.histogram(null, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.fill_between(centers, counts, color=null_color, alpha=0.5, linewidth=0)
    ax.axvline(
        poisson_true,
        color=poisson_color,
        lw=2,
        label=f"Poisson={poisson_true:.2f}±{poisson_sem:.2f}",
    )
    ax.axvline(
        ridge_true,
        color=ridge_color,
        lw=2,
        label=f"Ridge={ridge_true:.2f}±{ridge_sem:.2f}",
    )
    ax.set_xlabel(f"axis alignment score")
    ax.set_ylabel("# permutations")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=False)

    if print_stats:
        print(f"metric: {metric}  (random-effects: n_subjects={len(poisson_per_subj)})")
        print(f"  null mean: {centre:.4f}  std: {null.std():.4f}  n_perms: {len(null)}")
        print(f"  Poisson: {poisson_true:.4f} ± {poisson_sem:.4f}  p={p_poisson:.4f}")
        print(f"  Ridge:   {ridge_true:.4f} ± {ridge_sem:.4f}  p={p_ridge:.4f}")
