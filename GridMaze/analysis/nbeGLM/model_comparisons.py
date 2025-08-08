""" """

# %% Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


# %% Globs


# %% Performance validation


def plot_performance_validation(
    results_df,
    model_types=["baseline2", "baseline", "embedding"],
    input_features=[
        "place",
        "place_direction",
        "place_direction_distance_to_goal",
        "place_direction_distance_to_goal_egocentric_action",
    ],
    outlier_threshold=-0.3,
    print_stats=True,
    ax=None,
):
    """ """
    # set up figure
    f, ax = _init_fig(ax=ax)
    # process data
    df = _average_over_folds(results_df, outlier_threshold=outlier_threshold)
    # set multiindex column for baseline vs embedding comparison
    df.columns = pd.MultiIndex.from_tuples([tuple(c.split("_", 1)) for c in df.columns])
    # filter for input features and model types
    df = df[df.columns[df.columns.get_level_values(1).isin(input_features)]]
    df = df[df.columns[df.columns.get_level_values(0).isin(model_types)]]
    if print_stats:
        print("baseline vs embedding:")
        subj_mean = df.groupby("subject_ID").mean()
        print(_performance_validation_stats(subj_mean))

    # plot with seaborn
    df_long = (
        df.reset_index()
        .set_index(["cluster_unique_ID", "subject_ID"])
        .stack(level=[0, 1], future_stack=True)  # ← add future_stack=True
        .reset_index(name="score")
        .rename(columns={"level_2": "version", "level_3": "model_name"})
    )
    subj_avg = df_long.groupby(["subject_ID", "version", "model_name"])["score"].mean().reset_index()
    sns.pointplot(
        data=subj_avg,
        x="model_name",
        y="score",
        hue="version",
        hue_order=model_types,
        marker="_",
        markersize=10,
        markeredgewidth=3,
        errorbar="se",
        dodge=0.4,
        linestyle="none",
        alpha=1,
        palette={"baseline": "grey", "baseline2": "teal", "embedding": "mediumslateblue"},
        ax=ax,
    )


def _performance_validation_stats(subject_mean_df):
    """cross subejct t-test for baseline vs embedding performance"""
    feature_groups = subject_mean_df.columns.get_level_values(1).unique()
    model_types = subject_mean_df.columns.get_level_values(0).unique()
    assert "embedding" in model_types
    stats = []
    for feature_group in feature_groups:
        _df = subject_mean_df.xs(feature_group, level=1, axis=1)
        _mtypes = [m for m in _df.columns if m != "embedding"]
        for _model in _mtypes:
            t_stat, p_val = ttest_rel(_df["embedding"], _df[_model])
            stats.append({"feature_group": feature_group, "model_type": _model, "t_stat": t_stat, "p_val": p_val})
    stats_df = pd.DataFrame(stats)
    stats_df["p_val_corr"] = multipletests(stats_df.p_val, method="fdr_bh")[1]
    return stats_df


# %% Interaction validation


def plot_interaction_validation(
    results_df,
    outlier_threshold=-0.3,
    models=["place", "direction", "place_direction_linear", "place_direction_nonlinear"],
    ax=None,
):
    """ """
    # set up figure
    f, ax = _init_fig(ax=ax, figsize=(1, 3))
    # process data
    df = _average_over_folds(results_df, outlier_threshold=outlier_threshold)
    # filter for input features and model types
    df = df[df.columns[df.columns.isin(models)]]
    df_long = df.stack().reset_index(name="score")
    subj_avg = df_long.groupby(["subject_ID", "model_name"])["score"].mean().reset_index()
    # plot
    sns.pointplot(
        data=subj_avg,
        x="model_name",
        y="score",
        marker="_",
        markersize=10,
        markeredgewidth=3,
        errorbar="se",
        linestyle="none",
        alpha=1,
        ax=ax,
    )


# %% Utils


def _average_over_folds(results_df, outlier_threshold=-0.3):
    """ """
    # prcoess data
    df = results_df.copy()
    # average over folds
    df = df.groupby(["cluster_unique_ID", "model_name"]).cv_score.mean().unstack(level=1)  # n_neurons, n_models
    # add back subject_ID
    df["subject_ID"] = [i.split(".")[0] for i in df.index]
    df = df.set_index("subject_ID", append=True)
    # remove outlier score values (useually due to few spikes in Poisson GLM)
    df = df[df.gt(outlier_threshold).all(axis=1)]  # (n_neurons, n_models)
    return df


def _init_fig(ax=None, figsize=(4, 3)):
    # set up figure
    if ax is None:
        f, ax = plt.subplots(figsize=figsize)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("models")
    ax.set_ylabel("cv performance")
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    return f, ax
