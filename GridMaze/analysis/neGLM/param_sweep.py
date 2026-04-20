"""
Load, process and visualise hyperparameter sweep results for the neGLM model.
Results are produced by jobs/neGLM/param_sweep/submit.py
@peterdoohan
"""

# %% Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

from GridMaze.analysis.neGLM import load_model_sets as lms
from GridMaze.analysis.neGLM import model_comparisons as mc

# %% Globals

# baseline (default) value for each swept hyperparameter (see jobs/neGLM/utils.py)
DEFAULTS = {
    "resolution": 0.2,
    "Nhid": [100, 50],
    "Nlat": 15,
    "beta_act": 1e-1,
    "beta_weight": 1e-1,
}


# %% Loading


def load_sweep_results(subfolder="param_sweep", maze_names=["maze_1"], all_completed=True):
    """Load cv scores for the param sweep (maze_1 only by default)."""
    return lms.load_model_set_cv_scores(subfolder, maze_names=maze_names, all_completed=all_completed)


def load_sweep_training(subfolder="param_sweep", maze_names=["maze_1"], all_completed=True):
    """Load per-epoch training logs for the param sweep (maze_1 only by default)."""
    return lms.load_model_set_training(subfolder, maze_names=maze_names, all_completed=all_completed)


# %% Plotting


def plot_hyperparam_sweep(
    results_df,
    hyperparam="beta_weight",
    outlier_threshold=-0.6,
    baseline_color="mediumslateblue",
    variant_color="grey",
    plot_single_subjects=True,
    print_stats=True,
    annotate_baseline=True,
    ax=None,
):
    """
    Plot a single-hyperparameter sweep against the baseline (all other kwargs at defaults).
    X-axis is ordered by hyperparameter value, with the baseline highlighted at its default.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))
    ax = mc._init_fig(ax=ax)
    ax.set_xlabel(hyperparam)

    # filter to baseline + variants of this hyperparam
    mask = (results_df.model_name == "baseline") | results_df.model_name.str.startswith(f"{hyperparam}_")
    filt_df = results_df[mask]
    if filt_df.model_name.nunique() < 2:
        raise ValueError(f"No sweep variants found for hyperparam '{hyperparam}'")

    # average over folds per cell → (cell, model) matrix → long → per-subject mean
    cell_df = mc._average_over_folds(filt_df, outlier_threshold=outlier_threshold)
    long_df = cell_df.stack().reset_index(name="score")
    subj_avg = long_df.groupby(["subject_ID", "model_name"])["score"].mean().reset_index()

    # order by hyperparam value; baseline sits at its default value
    order, labels = _sweep_order(subj_avg.model_name.unique(), hyperparam, annotate_baseline=annotate_baseline)
    palette = {m: baseline_color if m == "baseline" else variant_color for m in order}

    if plot_single_subjects:
        sns.stripplot(
            data=subj_avg,
            x="model_name",
            y="score",
            hue="model_name",
            order=order,
            palette=palette,
            size=3,
            alpha=0.3,
            jitter=False,
            legend=False,
            ax=ax,
        )
    sns.pointplot(
        data=subj_avg,
        x="model_name",
        y="score",
        hue="model_name",
        order=order,
        palette=palette,
        errorbar="se",
        linestyle="none",
        legend=False,
        alpha=1,
        ax=ax,
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    if print_stats:
        stats_df = _sweep_ttests(subj_avg, order)
        print(stats_df)
    return


def plot_training_curves(
    training_df,
    model_name="baseline",
    train_color="crimson",
    test_color="royalblue",
    loss_color="black",
    plot_single_subjects=False,
    axes=None,
):
    """
    Two panels for a single model: train_loss (left) and train/test embedding performance (right).
    Per-subject curves averaged with SEM across subjects; pass plot_single_subjects=True to overlay rats.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(6, 3))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("epoch")

    df = training_df[training_df.model_name == model_name]
    if df.empty:
        raise ValueError(f"No training data found for model '{model_name}'")

    # panel 1: train loss
    _plot_curve(axes[0], df, "train_loss", color=loss_color, plot_single_subjects=plot_single_subjects)
    axes[0].set_ylabel("train loss")

    # panel 2: train vs test embedding perf
    _plot_curve(
        axes[1], df, "train_embedding_perf", color=train_color, plot_single_subjects=plot_single_subjects, label="train"
    )
    _plot_curve(
        axes[1], df, "test_embedding_perf", color=test_color, plot_single_subjects=plot_single_subjects, label="test"
    )
    axes[1].set_ylabel("performance")
    axes[1].legend(frameon=False, loc="best")
    axes[1].axhline(0, color="k", linestyle="--", alpha=0.5)

    axes[0].set_title(model_name)


def _plot_curve(ax, df, metric, color, plot_single_subjects=False, label=None):
    if plot_single_subjects:
        sns.lineplot(
            data=df,
            x="epoch",
            y=metric,
            units="subject_ID",
            estimator=None,
            color=color,
            alpha=0.3,
            linewidth=1,
            legend=False,
            ax=ax,
        )
    sns.lineplot(
        data=df,
        x="epoch",
        y=metric,
        color=color,
        errorbar="se",
        linewidth=2,
        label=label,
        ax=ax,
    )


def plot_hyperparam_sweep_summary(
    results_df,
    hyperparams=("resolution", "Nhid", "Nlat", "beta_act", "beta_weight"),
    outlier_threshold=-0.6,
    baseline_color="mediumslateblue",
    variant_color="grey",
    plot_single_subjects=True,
    print_stats=False,
    axes=None,
):
    """One panel per hyperparameter, y-axis shared so absolute CV performance is comparable."""
    if axes is None:
        _, axes = plt.subplots(1, len(hyperparams), figsize=(3 * len(hyperparams), 3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, hp in zip(axes, hyperparams):
        plot_hyperparam_sweep(
            results_df,
            hyperparam=hp,
            outlier_threshold=outlier_threshold,
            baseline_color=baseline_color,
            variant_color=variant_color,
            plot_single_subjects=plot_single_subjects,
            print_stats=print_stats,
            annotate_baseline=False,
            ax=ax,
        )
    return axes


# %% Utils


def _sweep_order(model_names, hyperparam, annotate_baseline=True):
    """Return (ordered model_names, display labels) sorted by hyperparam value; baseline at its default."""
    records = []
    for m in model_names:
        value = DEFAULTS[hyperparam] if m == "baseline" else _parse_value(m, hyperparam)
        records.append({"model_name": m, "value": value})
    df = pd.DataFrame(records)
    if hyperparam == "Nhid":
        df["sort_key"] = df.value.apply(lambda v: (sum(v), len(v), v[0]))
        df["label"] = df.value.apply(lambda v: "x".join(str(x) for x in v))
    else:
        df["sort_key"] = df.value.astype(float)
        df["label"] = df.value.apply(_format_numeric_label)
    df = df.sort_values("sort_key").reset_index(drop=True)
    if annotate_baseline:
        df.loc[df.model_name == "baseline", "label"] = df.loc[df.model_name == "baseline", "label"] + "\n(baseline)"
    return df.model_name.tolist(), df.label.tolist()


def _parse_value(model_name, hyperparam):
    """Parse a sweep variant model name (e.g. 'beta_weight_1e-3') into its hyperparam value."""
    suffix = model_name[len(hyperparam) + 1 :]
    if hyperparam == "Nhid":
        return [int(x) for x in suffix.split("_")]
    if hyperparam == "Nlat":
        return int(suffix)
    return float(suffix)


def _format_numeric_label(v):
    # log-spaced betas look cleanest in scientific; small decimals look cleanest as-is
    if isinstance(v, float) and (v != 0) and (abs(np.log10(abs(v))) >= 2):
        return f"{v:.0e}"
    return f"{v:g}"


def _sweep_ttests(subj_avg, order):
    """Per-subject paired t-test of each variant vs baseline, with FDR correction."""
    df = subj_avg.set_index(["subject_ID", "model_name"]).unstack().score
    baseline = df["baseline"]
    results = []
    for m in order:
        if m == "baseline":
            continue
        t_stat, p_val = ttest_rel(df[m], baseline)
        results.append(
            {
                "model": m,
                "mean_diff": (df[m] - baseline).mean(),
                "t_stat": t_stat,
                "p_val": p_val,
            }
        )
    stats_df = pd.DataFrame(results)
    stats_df["p_val_corr"] = multipletests(stats_df.p_val, method="fdr_bh")[1]
    return stats_df
