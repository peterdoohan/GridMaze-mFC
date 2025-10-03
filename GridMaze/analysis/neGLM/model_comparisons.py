""" """

# %% Imports
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


# %% Globs
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)


# %% Performance validation


def plot_performance_validation(
    results_df,
    model_types=["baseline2", "baseline", "embedding"],
    input_features=[
        "place",
        "place_direction",
        "place_direction_distance_to_goal",
    ],
    outlier_threshold=-0.6,
    plot_single_subjects=False,
    print_stats=True,
    ax=None,
):
    """ """
    # set up figure
    if ax is None:
        f, ax = plt.subplots(figsize=(4, 3))
    ax = _init_fig(ax=ax)
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
    # plot
    if plot_single_subjects:
        sns.pointplot(
            data=df_long,
            x="model_name",
            y="score",
            hue="subject_ID",
            order=model_types,
            palette=sns.color_palette("hls", n_colors=len(SUBJECT_IDS)),
            markers="o",
            markersize=7,
            markeredgewidth=0,
            errorbar=None,
            dodge=0.3,
            linestyle="none",
            legend=False,
            alpha=0.8,
            ax=ax,
        )
    # plot
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
    outlier_threshold=-0.6,
    models=["place", "direction", "place_direction_factorised", "place_direction_nonlinear"],
    colors=["grey", "grey", "lightgreen", "mediumslateblue"],
    plot_single_subjects=False,
    print_stats=True,
    ax=None,
):
    """ """
    # set up figure
    if ax is None:
        f, ax = plt.subplots(figsize=(4, 3))
    ax = _init_fig(ax=ax)
    # process data
    df = _average_over_folds(results_df, outlier_threshold=outlier_threshold)
    # filter for input features and model types
    if models != "all":
        df = df[df.columns[df.columns.isin(models)]]
        order = models
    else:
        order = None

    df_long = df.stack().reset_index(name="score")
    subj_avg = df_long.groupby(["subject_ID", "model_name"])["score"].mean().reset_index()
    if plot_single_subjects:
        sns.pointplot(
            data=df_long,
            x="model_name",
            y="score",
            hue="subject_ID",
            order=order,
            palette=sns.color_palette("hls", n_colors=len(SUBJECT_IDS)),
            markers="o",
            markersize=7,
            markeredgewidth=0,
            errorbar=None,
            dodge=0.3,
            linestyle="none",
            legend=False,
            alpha=0.8,
            ax=ax,
        )
    sns.pointplot(
        data=subj_avg,
        x="model_name",
        y="score",
        hue="model_name",
        marker="_",
        markersize=10,
        markeredgewidth=3,
        errorbar="se",
        linestyle="none",
        palette=colors,
        alpha=1,
        ax=ax,
    )
    if print_stats:
        t_stat, p_val, models = _interaction_validation_stats(subj_avg)
        print(f"{models[0]} vs {models[1]}")
        print(f"t_stat: {t_stat}, p_val: {p_val}")


def _interaction_validation_stats(subj_avg):
    """
    compare linear and nonlinear interaction cases
    """
    _df = subj_avg.set_index(["subject_ID", "model_name"]).unstack(level=1).score
    model_names = _df.columns
    lin_model = [m for m in model_names if "factorised" in m][0]
    nonlin_model = [m for m in model_names if "nonlinear" in m][0]
    t_stat, p_val = ttest_rel(_df[lin_model], _df[nonlin_model])
    return t_stat, p_val, (lin_model, nonlin_model)


# %% plot other features


def plot_other_feature_results(
    results_df,
    models=[
        "place_direction",
        "place_direction.distance_to_goal",
        "place_direction.distance_to_goal.goal",
        "place_direction.distance_to_goal.goal.egocentric_action",
        "place_direction.distance_to_goal.goal.egocentric_action.velocity",
    ],
    outlier_threshold=-0.6,
    plot_single_subjects=True,
    print_stats=True,
    stats_comparisons=[
        ("place_direction", "place_direction.distance_to_goal"),
        ("place_direction.distance_to_goal", "place_direction.distance_to_goal.goal"),
        ("place_direction.distance_to_goal.goal", "place_direction.distance_to_goal.goal.egocentric_action"),
        (
            "place_direction.distance_to_goal.goal.egocentric_action",
            "place_direction.distance_to_goal.goal.egocentric_action.velocity",
        ),
    ],
    ax=None,
):
    # set up figure
    if ax is None:
        f, ax = plt.subplots(figsize=(4, 3))
    ax = _init_fig(ax=ax)
    # process data
    df = _average_over_folds(results_df, outlier_threshold=outlier_threshold)
    # filter for input features and model types
    if models != "all":
        df = df[df.columns[df.columns.isin(models)]]
        order = models
    else:
        order = None

    df_long = df.stack().reset_index(name="score")
    subj_avg = df_long.groupby(["subject_ID", "model_name"])["score"].mean().reset_index()
    if plot_single_subjects:
        sns.pointplot(
            data=df_long,
            x="model_name",
            y="score",
            hue="subject_ID",
            order=order,
            palette=sns.color_palette("hls", n_colors=len(SUBJECT_IDS)),
            markers="o",
            markersize=7,
            markeredgewidth=0,
            errorbar=None,
            dodge=0.3,
            linestyle="none",
            legend=False,
            alpha=0.8,
            ax=ax,
        )
    sns.pointplot(
        data=subj_avg,
        x="model_name",
        y="score",
        marker="_",
        markersize=10,
        markeredgewidth=3,
        errorbar="se",
        linestyle="none",
        color="black",
        alpha=1,
        ax=ax,
    )
    if print_stats:
        df = subj_avg.set_index(["subject_ID", "model_name"]).unstack().score
        stats_df = _get_other_feature_stats(df, stats_comparisons)
        print(stats_df)


def _get_other_feature_stats(df, stats_comparisons):
    results = []
    for model_1, model_2 in stats_comparisons:
        t_stat, p_val = ttest_rel(df[model_2], df[model_1], alternative="greater")
        results.append({"model_1": model_1, "model_2": model_2, "t_stat": t_stat, "p_val": p_val})
    stats_df = pd.DataFrame(results)
    stats_df["p_val_corr"] = multipletests(stats_df.p_val, method="fdr_bh")[1]
    return stats_df


# %% main variable interactions


def plot_main_feature_interactions(
    results_df,
    outlier_threshold=-0.6,
    models="all",
    colors=["lightgreen", "grey", "mediumslateblue"],
    plot_single_subjects=True,
    print_stats=True,
    ax=None,
):
    if ax is None:
        f, ax = plt.subplots(figsize=(1.5, 3))
    ax = _init_fig(ax=ax)
    # process data
    df = _average_over_folds(results_df, outlier_threshold=outlier_threshold)
    # filter for input features and model types
    if models != "all":
        df = df[df.columns[df.columns.isin(models)]]
        order = models
    else:
        order = None
    df_long = df.stack().reset_index(name="score")
    subj_avg = df_long.groupby(["subject_ID", "model_name"])["score"].mean().reset_index()
    # plot
    if plot_single_subjects:
        sns.pointplot(
            data=df_long,
            x="model_name",
            y="score",
            hue="subject_ID",
            order=order,
            palette=sns.color_palette("hls", n_colors=len(SUBJECT_IDS)),
            markers="o",
            markersize=7,
            markeredgewidth=0,
            errorbar=None,
            dodge=0.3,
            linestyle="none",
            legend=False,
            alpha=0.8,
            ax=ax,
        )
    # plot
    sns.pointplot(
        data=subj_avg,
        x="model_name",
        y="score",
        hue="model_name",
        marker="_",
        markersize=10,
        markeredgewidth=3,
        errorbar="se",
        palette=colors,
        linestyle="none",
        alpha=1,
        ax=ax,
    )
    if print_stats:
        stats_df = _main_feature_interaction_stats(subj_avg)
        print(stats_df)


def _main_feature_interaction_stats(subj_avg):
    """just do all pairwise model comparisons"""
    _models = subj_avg.model_name.unique()
    # get all unique pairs of models
    pairwise = [(m1, m2) for i, m1 in enumerate(_models) for m2 in _models[i + 1 :]]
    stats = []
    for m1, m2 in pairwise:
        t_stat, p_val = ttest_rel(subj_avg[subj_avg.model_name == m1].score, subj_avg[subj_avg.model_name == m2].score)
        stats.append({"model_1": m1, "model_2": m2, "t_stat": t_stat, "p_val": p_val})

    stats_df = pd.DataFrame(stats)
    # account for muliple comparisons
    stats_df["p_val_corr"] = multipletests(stats_df.p_val, method="fdr_bh")[1]
    return stats_df


# %%  full feature interactions


def plot_full_feature_interactions(
    results_df,
    outlier_threshold=-0.6,
    models=[
        "place-direction-velocity-distance_to_goal-egocentric_action",
        "place.direction-velocity-distance_to_goal-egocentric_action",
        "place.direction.velocity-distance_to_goal-egocentric_action",
        "place.direction.velocity.distance_to_goal.egocentric_action",
    ],
    colors=["lightgreen", "grey", "grey", "mediumslateblue"],
    plot_single_subjects=False,
    print_stats=True,
    ax=None,
):
    if ax is None:
        f, ax = plt.subplots(figsize=(1.5, 3))
    ax = _init_fig(ax=ax)
    # process data
    df = _average_over_folds(results_df, outlier_threshold=outlier_threshold)
    df_long = df.stack().reset_index(name="score")
    subj_avg = df_long.groupby(["subject_ID", "model_name"])["score"].mean().reset_index()
    if plot_single_subjects:
        sns.pointplot(
            data=df_long,
            x="model_name",
            y="score",
            hue="subject_ID",
            order=models,
            palette=sns.color_palette("hls", n_colors=len(SUBJECT_IDS)),
            markers="o",
            markersize=7,
            markeredgewidth=0,
            errorbar=None,
            dodge=0.3,
            linestyle="none",
            legend=False,
            alpha=0.8,
            ax=ax,
        )
    # plot
    sns.pointplot(
        data=subj_avg,
        x="model_name",
        y="score",
        hue="model_name",
        order=models,
        marker="_",
        markersize=10,
        markeredgewidth=3,
        errorbar="se",
        palette=colors,
        linestyle="none",
        alpha=1,
        ax=ax,
    )
    if print_stats:
        stats_comparisons = [
            (
                "place-direction-velocity-distance_to_goal-egocentric_action",
                "place.direction-velocity-distance_to_goal-egocentric_action",
            ),
            (
                "place-direction-velocity-distance_to_goal-egocentric_action",
                "place.direction.velocity-distance_to_goal-egocentric_action",
            ),
            (
                "place.direction.velocity.distance_to_goal.egocentric_action",
                "place.direction-velocity-distance_to_goal-egocentric_action",
            ),
            (
                "place.direction.velocity.distance_to_goal.egocentric_action",
                "place.direction.velocity-distance_to_goal-egocentric_action",
            ),
        ]
        df = subj_avg.set_index(["subject_ID", "model_name"]).unstack().score
        stats_df = _get_other_feature_stats(df, stats_comparisons)
        print(stats_df)

    return


# %% Utils


def _average_over_folds(results_df, outlier_threshold=-0.6):
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


def _init_fig(ax):
    # set up figure
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("models")
    ax.set_ylabel("cv performance")
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    return ax
