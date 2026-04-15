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
    model_types=["baseline", "baseline2", "embedding"],
    input_features=[
        "place",
        "place_direction",
        "place_direction_distance_to_goal",
    ],
    colors=["grey", "crimson", "royalblue"],
    outlier_threshold=-0.6,
    plot_single_subjects=False,
    print_stats=True,
    axes=None,
):
    """ """
    # set up figure
    if axes is None:
        f, axes = plt.subplots(2, 1, figsize=(4, 3.5), gridspec_kw={"height_ratios": [4, 1], "hspace": 0.25})
    axes[0] = _init_fig(ax=axes[0])
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
    # display rename for version labels
    VERSION_RENAME = {"baseline": "GLMx", "baseline2": "GLM+", "embedding": "neGLM"}
    df_long["version"] = df_long["version"].replace(VERSION_RENAME)
    subj_avg["version"] = subj_avg["version"].replace(VERSION_RENAME)
    # GLMx == GLM+ for the place-only model, collapse to GLM+ for cleaner layout
    for _df in (df_long, subj_avg):
        _df.loc[(_df["model_name"] == "place") & (_df["version"] == "GLMx"), "version"] = "GLM+"
    display_hue_order = [VERSION_RENAME[m] for m in model_types]
    display_palette = {VERSION_RENAME[m]: c for m, c in zip(model_types, colors)}
    # plot
    if plot_single_subjects:
        sns.stripplot(
            data=subj_avg,
            x="model_name",
            y="score",
            hue="version",
            hue_order=display_hue_order,
            order=input_features,
            palette=display_palette,
            dodge=0.1,
            size=3,
            alpha=0.3,
            jitter=False,
            legend=False,
            ax=axes[0],
        )
    # plot
    sns.pointplot(
        data=subj_avg,
        x="model_name",
        y="score",
        hue="version",
        hue_order=display_hue_order,
        order=input_features,
        errorbar="se",
        dodge=0.45,
        linestyle="none",
        alpha=1,
        palette=display_palette,
        ax=axes[0],
    )
    axes[0].set_xlabel("")
    plot_variable_table(
        axes[1],
        row_labels=["place", "direction", "distance_to_goal"],
        columns=input_features,
        presence={
            "place": ["place"],
            "place_direction": ["place", "direction"],
            "place_direction_distance_to_goal": ["place", "direction", "distance_to_goal"],
        },
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
    colors=["grey", "grey", "crimson", "royalblue"],
    plot_single_subjects=False,
    print_stats=True,
    axes=None,
):
    """ """
    # set up figure
    if axes is None:
        f, axes = plt.subplots(2, 1, figsize=(4, 3.5), gridspec_kw={"height_ratios": [4, 1], "hspace": 0.25})
    axes[0] = _init_fig(ax=axes[0])
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
        sns.stripplot(
            data=subj_avg,
            x="model_name",
            y="score",
            hue="model_name",
            order=order,
            palette=colors,
            size=3,
            alpha=0.3,
            jitter=False,
            legend=False,
            ax=axes[0],
        )
    sns.pointplot(
        data=subj_avg,
        x="model_name",
        y="score",
        hue="model_name",
        order=order,
        errorbar="se",
        linestyle="none",
        palette=colors,
        alpha=1,
        ax=axes[0],
    )
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", labelbottom=False)
    # auto-infer table contents from model names
    all_vars = ["place", "direction", "distance_to_goal"]
    presence = {m: [v for v in all_vars if v in m] for m in models}
    row_labels = [v for v in all_vars if any(v in vs for vs in presence.values())]
    connect_columns = [m for m in models if "nonlinear" in m]
    plot_variable_table(
        axes[1],
        row_labels=row_labels,
        columns=models,
        presence=presence,
        connect_columns=connect_columns,
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
    colors=None,
    plot_single_subjects=False,
    print_stats=True,
    axes=None,
):
    # set up figure
    if axes is None:
        f, axes = plt.subplots(2, 1, figsize=(4, 3.5), gridspec_kw={"height_ratios": [4, 1], "hspace": 0.25})
    axes[0] = _init_fig(ax=axes[0])
    if colors is None:
        colors = sns.color_palette("rocket_r", n_colors=len(models))
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
        sns.stripplot(
            data=subj_avg,
            x="model_name",
            y="score",
            hue="model_name",
            order=order,
            palette=colors,
            size=3,
            alpha=0.3,
            jitter=False,
            legend=False,
            ax=axes[0],
        )
    sns.pointplot(
        data=subj_avg,
        x="model_name",
        y="score",
        hue="model_name",
        order=order,
        errorbar="se",
        linestyle="none",
        palette=colors,
        alpha=1,
        ax=axes[0],
    )
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", labelbottom=False)
    # auto-infer table contents from dot-separated model names
    all_vars = list(dict.fromkeys(v for m in models for v in m.split(".")))
    presence = {m: m.split(".") for m in models}
    row_labels = all_vars
    plot_variable_table(
        axes[1],
        row_labels=row_labels,
        columns=models,
        presence=presence,
        connect_columns=models,
    )
    if print_stats:
        df = subj_avg.set_index(["subject_ID", "model_name"]).unstack().score
        stats_comparisons = list(zip(models[:-1], models[1:]))
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


def plot_prop_max_variance_explained(
    results_df,
    small_model="place_direction.distance_to_goal",
    big_model="place_direction.distance_to_goal.goal.egocentric_action.velocity",
    colors=("mediumslateblue", "lightgrey"),
    outlier_threshold=-0.6,
    print_stats=True,
    ax=None,
):
    """Pie chart: per-subject % of max variance explained captured by `small_model` vs `big_model`."""
    if ax is None:
        f, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.set_aspect("equal")
    # process data
    df = _average_over_folds(results_df, outlier_threshold=outlier_threshold)
    df = df[[small_model, big_model]]
    subj_avg = (
        df.stack()
        .reset_index(name="score")
        .groupby(["subject_ID", "model_name"])["score"]
        .mean()
        .unstack("model_name")
    )
    # per-subject % of max variance explained
    prop = subj_avg[small_model] / subj_avg[big_model] * 100
    mean_small, sem_small = prop.mean(), prop.sem()
    mean_other = 100 - mean_small
    # pie chart
    wedges, _ = ax.pie(
        [mean_small, mean_other],
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    # annotate each wedge centroid with mean ± sem
    labels = [
        f"{mean_small:.1f} ± {sem_small:.1f}%",
        f"{mean_other:.1f} ± {sem_small:.1f}%",
    ]
    for wedge, label in zip(wedges, labels):
        theta = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        ax.text(
            0.6 * np.cos(theta),
            0.6 * np.sin(theta),
            label,
            ha="center",
            va="center",
            fontsize=8,
        )
    if print_stats:
        print(prop)
        print(f"mean ± sem: {mean_small:.2f} ± {sem_small:.2f} %")


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


# %% cover all replacement model comp plotting fn


def plot_model_comparison(
    results_df,
    outlier_threshold=-0.6,
    models=[
        "place_direction.distance_to_goal",
        "place_direction.distance_to_goal.goal",
        "place_direction.distance_to_goal.head_direction",
        "place_direction-distance_to_goal-head_direction",
        "place_direction.distance_to_goal.head_direction",
        "place_direction.distance_to_goal.egocentric_angle_to_goal",
        "place_direction-distance_to_goal-egocentric_angle_to_goal",
    ],
    colors=None,
    plot_single_subjects=True,
    stats_comparisons=[
        (
            "place_direction.distance_to_goal.head_direction",
            "place_direction-distance_to_goal-head_direction",
        ),
        (
            "place_direction.distance_to_goal.head_direction",
            "place_direction.distance_to_goal.egocentric_angle_to_goal",
        ),
        (
            "place_direction.distance_to_goal.egocentric_angle_to_goal",
            "place_direction-distance_to_goal-egocentric_angle_to_goal",
        ),
    ],
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
        order=models,
        marker="_",
        markersize=10,
        markeredgewidth=3,
        errorbar="se",
        palette=colors,
        color="black" if colors is None else None,
        linestyle="none",
        alpha=1,
        ax=ax,
    )
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right")
    if stats_comparisons:
        stats_df = _model_comparison_ttests(subj_avg, stats_comparisons)
        return stats_df


def _model_comparison_ttests(subj_avg, model_comparisons):
    df = subj_avg.set_index(["subject_ID", "model_name"]).unstack().score
    results = []
    for model_1, model_2 in model_comparisons:
        t_stat, p_val = ttest_rel(df[model_2], df[model_1])
        results.append({"model_1": model_1, "model_2": model_2, "t_stat": t_stat, "p_val": p_val})
    stats_df = pd.DataFrame(results)
    stats_df["p_val_corr"] = multipletests(stats_df.p_val, method="fdr_bh")[1]
    return stats_df


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


def plot_variable_table(
    ax,
    row_labels,
    columns,
    presence,
    dot_size=40,
    dot_color="grey",
    dot_marker="s",
    dot_alpha=1,
    connect_columns=None,
):
    """Grid of dots indicating which variables (rows) are in each model (columns)."""
    n_rows, n_cols = len(row_labels), len(columns)
    row_idx = {r: i for i, r in enumerate(row_labels)}
    xs, ys = [], []
    for col_i, col in enumerate(columns):
        for var in presence.get(col, []):
            if var in row_idx:
                xs.append(col_i)
                ys.append(row_idx[var])
    if connect_columns:
        for col_i, col in enumerate(columns):
            if col not in connect_columns:
                continue
            ys_here = [row_idx[v] for v in presence.get(col, []) if v in row_idx]
            if len(ys_here) < 2:
                continue
            ax.plot(
                [col_i, col_i],
                [min(ys_here), max(ys_here)],
                color=dot_color,
                alpha=dot_alpha,
                linewidth=2,
                zorder=0,
            )
    ax.scatter(xs, ys, s=dot_size, color=dot_color, marker=dot_marker, alpha=dot_alpha, edgecolors="none")
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)
    return ax
