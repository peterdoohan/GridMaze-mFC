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
from matplotlib_venn import venn3

from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


from GridMaze.analysis.nbeGLM import model_comparisons as mc


# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

# %%


def plot_cpd_clusters(cpd_df, feature_tuned_df, features=["distance_to_goal", "place_direction"], ax=None):
    """ """
    # set up fig
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 2.5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", linestyle="--", alpha=0.2)
    ax.axvline(0, color="k", linestyle="--", alpha=0.2)
    # process data
    tuned_clusters = feature_tuned_df.index.get_level_values(1)
    filt_cpd_df = cpd_df.loc[cpd_df.index.get_level_values(0).isin(tuned_clusters)]
    x = filt_cpd_df[features[0]]
    y = filt_cpd_df[features[1]]
    # plot
    sns.histplot(
        x=x,
        y=y,
        bins=40,
        ax=ax,
        cbar=True,
    )


# %% Unique variance explained acoss cells


def get_feature_tuned_df(
    results_df,
    reduced_models=[
        "remove_distance_to_goal",
        "remove_place_direction",
        "remove_egocentric_action_action",
    ],
    multiple_comparisons_corrected=False,
    alpha=0.05,
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
    cpd_df = pd.DataFrame(index=df.index, columns=_cpd_names)
    for m, _name in zip(reduced_models, _cpd_names):
        cpd_df[_name] = (full_model - df[m]).mul(100)  # convert to percent
    # run t-test against 0 for every cell-feature across folds
    p_df = cpd_df.groupby(level=[0, 1]).apply(group_ttest)
    if multiple_comparisons_corrected:
        for _name in p_df.columns:
            p_df[_name] = multipletests(p_df[_name], method="fdr_bh", alpha=alpha)[1]
    sig_df = p_df.lt(alpha)
    # consider only neurons with significant cpd for at least one feature
    sig_df = sig_df[sig_df.any(axis=1)]
    return sig_df


def plot_summary_pointplot(
    df,
    models,
    ax=None,
):
    # set up fig
    if ax is None:
        f, ax = plt.subplots(figsize=(2, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_ylabel("prop. neurons")

    # counts cells in each condition
    m1, m2, m3 = models
    counts = []
    for subject in SUBJECT_IDS:
        _df = df.loc[subject]
        total_count = len(_df)
        counts.append(
            {
                (m1): len(_df[(_df[m1]) & (~_df[m2]) & (~_df[m3])]) / total_count,
                (m2): len(_df[(~_df[m1]) & (_df[m2]) & (~_df[m3])]) / total_count,
                (m3): len(_df[(~_df[m1]) & (~_df[m2]) & (_df[m3])]) / total_count,
                (m1, m2): len(_df[(_df[m1]) & (_df[m2]) & (~_df[m3])]) / total_count,
                (m1, m3): len(_df[(_df[m1]) & (~_df[m2]) & (_df[m3])]) / total_count,
                (m2, m3): len(_df[(~_df[m1]) & (_df[m2]) & (_df[m3])]) / total_count,
                (m1, m2, m3): len(_df[(_df[m1]) & (_df[m2]) & (_df[m3])]) / total_count,
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
    order = [(m1), (m2), (m3), (m1, m2), (m1, m3), (m2, m3), (m1, m2, m3)]
    colors = sns.color_palette("hls", n_colors=len(SUBJECT_IDS))
    sns.pointplot(
        data=long_df,
        x="feature",
        y="prop",
        hue="subject_ID",
        order=order,
        palette=colors,
        markers="o",
        markersize=7,
        markeredgewidth=0,
        errorbar=None,
        dodge=0.1,
        linestyle="none",
        legend=False,
        alpha=0.5,
        ax=ax,
    )
    sns.pointplot(
        data=long_df,
        x="feature",
        y="prop",
        order=order,
        markers="_",
        color="k",
        markersize=15,
        markeredgewidth=3,
        errorbar="se",
        linestyle="none",
        legend=False,
        alpha=1,
        ax=ax,
    )


def plot_summary_venn_diagram(df, models, ax=None):
    """ """
    m1, m2, m3 = models
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    venn_counts = {
        "100": len(df[(df[m1]) & (~df[m2]) & (~df[m3])]),
        "010": len(df[(~df[m1]) & (df[m2]) & (~df[m3])]),
        "001": len(df[(~df[m1]) & (~df[m2]) & (df[m3])]),
        "110": len(df[(df[m1]) & (df[m2]) & (~df[m3])]),
        "101": len(df[(df[m1]) & (~df[m2]) & (df[m3])]),
        "011": len(df[(~df[m1]) & (df[m2]) & (df[m3])]),
        "111": len(df[(df[m1]) & (df[m2]) & (df[m3])]),
    }

    # Create the Venn diagram for 'distance', 'place_direction', and 'trial_phase'
    venn = venn3(
        subsets=(
            venn_counts["100"],
            venn_counts["010"],
            venn_counts["110"],
            venn_counts["001"],
            venn_counts["101"],
            venn_counts["011"],
            venn_counts["111"],
        ),
        set_labels=(m1, m2, m3),
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
    features=["distance_to_goal", "place_direction", "egocentric_action_action"],
    print_stats=True,
    plot_single_subject=False,
    orientation="vertical",
    ax=None,
):
    """ """
    # set up fig
    if ax is None:
        f, ax = plt.subplots(figsize=(2, 3))
    ax.spines[["top", "right"]].set_visible(False)
    if orientation == "vertical":
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        ax.set_xlabel("features")
        ax.set_ylabel("unique variance explained (%)")
    else:
        ax.axvline(0, color="k", linestyle="--", alpha=0.5)
        ax.set_ylabel("features")
        ax.set_xlabel("unique variance explained (%)")

    # process data
    df = cpd_df.copy()
    if features != "all":
        df = df[features]
        order = features
    else:
        order = None
    long_df = df.stack().reset_index(name="score").rename(columns={"level_2": "feature"})
    subject_av = long_df.groupby(["subject_ID", "feature"])["score"].mean().reset_index()
    colors = sns.color_palette("hls", n_colors=len(SUBJECT_IDS))
    if orientation == "vertical":
        x = "feature"
        y = "score"
        marker = "_"
    else:
        x = "score"
        y = "feature"
        marker = "|"
    if plot_single_subject:
        sns.pointplot(
            data=long_df,
            x=x,
            y=y,
            hue="subject_ID",
            order=order,
            palette=colors,
            markers="o",
            markersize=7,
            markeredgewidth=0,
            errorbar=None,
            dodge=0.1,
            linestyle="none",
            legend=False,
            alpha=0.5,
            ax=ax,
        )
    sns.pointplot(
        data=subject_av,
        x=x,
        y=y,
        marker=marker,
        order=order,
        color="k",
        markersize=15,
        markeredgewidth=3,
        errorbar="se",
        linestyle="none",
        alpha=1,
        ax=ax,
    )
    if print_stats:
        print(_variance_explained_stats(cpd_df))
    return


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
    outlier_threshold=-0.3,
    full_model_thres=0,
    reduced_models=[
        "remove_distance_to_goal",
        "remove_place_direction",
        "remove_egocentric_action_action",
    ],
):
    """ """
    # average over folds & remove neurons with large negative scores
    df = mc._average_over_folds(results_df, outlier_threshold=outlier_threshold)
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
