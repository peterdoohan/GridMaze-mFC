"""
Evaluated unique variance explained among behavioural variables in the datasets (main features:
distance_to_goal, place, direction, egocentric_action) using the nbeGLM model comparisons.
@peterdoohan
"""

# %% Imports
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

from scipy.stats import pearsonr, ttest_1samp
from statsmodels.stats.multitest import multipletests


from GridMaze.analysis.neGLM import model_comparisons as mc


# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

# %%


def plot_cpd_clusters(
    cpd_df,
    feature_tuned_df,
    features=["distance_to_goal", "place_direction"],
    remove_no_unique_variance_clusters=False,
    n_bins=30,
    cmap="cividis",
    pthresh=0.01,
    scatter_color=".15",
    scatter_size=5,
    scatter_alpha=0.4,
    xlims=(-15, 70),
    ylims=(-15, 70),
    print_stats=True,
    ax=None,
):
    """ """
    # set up fig
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 2.5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", linestyle="--", alpha=0.4)
    ax.axvline(0, color="k", linestyle="--", alpha=0.4)
    ax.set_xlabel(f"{features[0]} (%)")
    ax.set_ylabel(f"{features[1]} (%)")
    # process data
    if remove_no_unique_variance_clusters:
        _feature_tuned_df = feature_tuned_df[feature_tuned_df.any(axis=1)]
    else:
        _feature_tuned_df = feature_tuned_df.copy()
    tuned_clusters = _feature_tuned_df.index.get_level_values(1)
    filt_cpd_df = cpd_df.loc[cpd_df.index.get_level_values(0).isin(tuned_clusters)]
    x = filt_cpd_df[features[0]]
    y = filt_cpd_df[features[1]]
    # plot
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
        cbar=True,
        cbar_kws={"shrink": 0.5, "label": "neurons"},
        ax=ax,
    )
    cax = ax.figure.axes[-1]
    for spine in cax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    # per-subject correlation stats
    summary = _cpd_correlation_stats(filt_cpd_df, features)
    if print_stats:
        print(summary)


# %% Unique variance explained acoss cells


def get_feature_tuned_df(
    results_df,
    reduced_models=[
        "remove_distance_to_goal",
        "remove_place_direction",
    ],
    multiple_comparisons_corrected=False,
    filter_for_full_model_significance=True,
    alpha=0.01,
):
    # filter models
    df = results_df.copy()
    if reduced_models != "all":
        df = df[df.model_name.isin(reduced_models + ["full_model"])]
    df = df.set_index(["subject_ID", "cluster_unique_ID", "fold", "model_name"])["cv_score"].unstack(
        level=3
    )  # neurons x folds, models
    reduced_models = [c for c in df.columns if "remove" in c]
    _cpd_names = [m.split("_", 1)[1] for m in reduced_models]
    # calculate cpd (full model - reduced model) for each variable
    full_model = df["full_model"]
    full_model_pval = df.full_model.unstack().apply(
        lambda row: ttest_1samp(row, popmean=0, alternative="greater").pvalue, axis=1
    )  # test full model across folds to see if any variance is explained
    if multiple_comparisons_corrected:
        _pval_corr = multipletests(full_model_pval, method="fdr_bh", alpha=alpha)[1]
        full_model_pval = pd.Series(_pval_corr, index=full_model_pval.index)
    cpd_df = pd.DataFrame(index=df.index, columns=_cpd_names)
    for m, _name in zip(reduced_models, _cpd_names):
        cpd_df[_name] = (full_model - df[m]).mul(100)  # convert to percent
    # run t-test against 0 for every cell-feature across folds
    p_df = cpd_df.groupby(level=[0, 1]).apply(group_ttest)
    if multiple_comparisons_corrected:
        for _name in p_df.columns:
            p_df[_name] = multipletests(p_df[_name], method="fdr_bh", alpha=alpha)[1]
    # filter for only clusters with sig variance explained in the full model
    if filter_for_full_model_significance:
        sig_df = p_df.loc[full_model_pval.lt(alpha)]
    else:
        sig_df = p_df.copy()
    sig_df = sig_df.lt(alpha)  # convert to bool
    return sig_df


def plot_summary_pointplot(
    feature_tuned_df,
    models=["distance_to_goal", "place_direction"],
    plot_single_subject=True,
    marker_color=["crimson", "royalblue", "grey"],
    subject_line_color="grey",
    subject_line_alpha=0.3,
    ax=None,
):
    # set up fig
    if ax is None:
        f, ax = plt.subplots(figsize=(2, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("features")
    ax.set_ylabel("prop. neurons")

    df = feature_tuned_df.copy()
    # counts cells in each condition
    m1, m2 = models
    both_label = f"{m1}+{m2}"
    counts = []
    for subject in SUBJECT_IDS:
        _df = df.loc[subject]
        total_count = len(_df)
        counts.append(
            {
                m1: len(_df[(_df[m1]) & (~_df[m2])]) / total_count,
                m2: len(_df[(~_df[m1]) & (_df[m2])]) / total_count,
                both_label: len(_df[(_df[m1]) & (_df[m2])]) / total_count,
            }
        )
    counts_df = pd.DataFrame(counts)
    counts_df.index = SUBJECT_IDS
    long_df = (
        counts_df.stack()
        .reset_index(name="prop")
        .rename(
            columns={"level_0": "subject_ID", "level_1": "feature"},
        )
    )
    # plot
    order = [m1, m2, both_label]
    if plot_single_subject:
        sns.lineplot(
            data=long_df,
            x="feature",
            y="prop",
            units="subject_ID",
            estimator=None,
            sort=False,
            color=subject_line_color,
            alpha=subject_line_alpha,
            linewidth=2,
            ax=ax,
        )
    sns.pointplot(
        data=long_df,
        x="feature",
        y="prop",
        hue="feature",
        order=order,
        hue_order=order,
        palette=dict(zip(order, marker_color)),
        markersize=9,
        errorbar="se",
        err_kws={"linewidth": 3},
        linestyle="none",
        legend=False,
        alpha=1,
        ax=ax,
    )
    ax.set_xlim(-0.3, len(order) - 0.7)
    ax.tick_params(axis="x", rotation=30)


def plot_summary_venn_diagram(df, models, colors=("crimson", "royalblue"), alpha=0.5, ax=None):
    """ """
    m1, m2 = models
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    venn_counts = {
        "10": len(df[(df[m1]) & (~df[m2])]),
        "01": len(df[(~df[m1]) & (df[m2])]),
        "11": len(df[(df[m1]) & (df[m2])]),
    }

    # Create the Venn diagram for 'distance', 'place_direction', and 'trial_phase'
    venn = venn2(
        subsets=(
            venn_counts["10"],
            venn_counts["01"],
            venn_counts["11"],
        ),
        set_labels=(m1, m2),
        set_colors=colors,
        alpha=alpha,
        ax=ax,
    )


def group_ttest(g):
    # only test numeric columns
    numeric = g.select_dtypes(include="number")
    # for each column, run ttest_1samp against popmean=0
    return pd.Series({col: ttest_1samp(g[col], popmean=0, alternative="greater").pvalue for col in numeric})


# %% Unique variance explained across features


def plot_variance_explained(
    cpd_df,
    features=["distance_to_goal", "place_direction"],
    print_stats=True,
    plot_single_subject=True,
    marker_color=["crimson", "royalblue"],
    subject_line_color="grey",
    subject_line_alpha=0.3,
    ax=None,
):
    """ """
    # set up fig
    if ax is None:
        f, ax = plt.subplots(figsize=(2, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("features")
    ax.set_ylabel("unique variance explained (%)")

    # process data
    df = cpd_df.copy()
    if features != "all":
        df = df[features]
        order = features
    else:
        order = None
    long_df = df.stack().reset_index(name="score").rename(columns={"level_2": "feature"})
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
    return


def _cpd_correlation_stats(cpd_df, features):
    """Per-subject Pearson r between two CPD features + one-sided t-test against 0 (negative)."""
    f1, f2 = features
    records = []
    for subject, sub_df in cpd_df.groupby(level=1):
        r, _ = pearsonr(sub_df[f1], sub_df[f2])
        records.append({"subject_ID": subject, "r": r, "n_neurons": len(sub_df)})
    corr_df = pd.DataFrame(records)
    t_stat, p_val = ttest_1samp(corr_df["r"], 0, alternative="less")
    summary = pd.Series(
        {
            "t_stat": t_stat,
            "p_val": p_val,
            "mean_r": corr_df["r"].mean(),
            "sem_r": corr_df["r"].sem(),
        }
    )
    return summary


def _variance_explained_stats(cpd_df):
    """ """
    # average over neurons for each subject
    _df = cpd_df.groupby(level=1).mean()
    features = _df.columns
    results = []
    for feature in features:
        # t-test against 0
        t_stat, p_val = ttest_1samp(_df[feature], 0, alternative="greater")
        results.append(
            {
                "feature": feature,
                "t_stat": t_stat,
                "p_val": p_val,
            }
        )
    stats_df = pd.DataFrame(results)
    # correct for multiple comparisons
    _, stats_df["p_val_corr"], _, _ = multipletests(stats_df["p_val"], method="fdr_bh")
    return stats_df


def get_cpd_df(
    results_df,
    outlier_threshold=-0.6,
    full_model_sig=True,
    full_model_thres=False,
    alpha=0.05,
    reduced_models=[
        "remove_distance_to_goal",
        "remove_place_direction",
    ],
):
    """ """
    _df = results_df.copy()
    # test full model against 0 across folds (before averaging collapses folds)
    if full_model_sig:
        _fold_df = (
            _df[_df.model_name == "full_model"]
            .set_index(["subject_ID", "cluster_unique_ID", "fold"])["cv_score"]
            .unstack(level=2)
        )
        full_model_pval = _fold_df.apply(
            lambda row: ttest_1samp(row, popmean=0, alternative="greater").pvalue,
            axis=1,
        )
        keep_clusters = full_model_pval[full_model_pval.lt(alpha)].index.droplevel(0).values
        _df = _df[_df.cluster_unique_ID.isin(keep_clusters)]

    # average over folds & remove neurons with large negative scores
    df = mc._average_over_folds(_df, outlier_threshold=outlier_threshold)
    if full_model_thres:
        df = df[df["full_model"] > full_model_thres]
    # filter for reduced models
    if reduced_models != "all":
        df = df[df.columns[df.columns.isin(reduced_models + ["full_model"])]]
    reduced_models = [c for c in df.columns if "remove" in c]
    _cpd_names = [m.split("_", 1)[1] for m in reduced_models]
    # calculate cpd (full model - reduced model) for each variable
    full_model = df["full_model"]
    cpd_df = pd.DataFrame(index=df.index, columns=_cpd_names)
    for m, _name in zip(reduced_models, _cpd_names):
        cpd_df[_name] = (full_model - df[m]).mul(100)  # convert to percent
    return cpd_df
