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

# %% shortcut vars for kep model comparisons

PERFORMANCE_VALIDATION = {
    "models": "all",
    "model_groups": ["baseline2", "baseline", "embedding"],
    "input_feature_groups": ["place", "place_direction", "place_direction_distance_to_goal"],
    "stats_comparisons": None,
}

INTERACTION_VALIDATION_1 = {
    "models": ["direction", "place", "place_direction_factorised", "place_direction_nonlinear"],
    "stats_comparisons": None,
}

INTERACTION_VALIDATION_2 = {
    "models": [
        "place_direction_conjunction",
        "place_direction_distance_to_goal_factorised",
        "place_direction_distance_to_goal_nonlinear",
    ],
    "stats_comparisons": None,
}

OTHER_FEATURES = {
    "models": [
        "place_direction",
        "place_direction.distance_to_goal",
        "place_direction.distance_to_goal.goal",
        "place_direction.distance_to_goal.goal.egocentric_action",
        "place_direction.distance_to_goal.goal.egocentric_action.velocity",
    ],
    "stats_comparisons": [
        ("place_direction", "place_direction.distance_to_goal"),
        ("place_direction.distance_to_goal", "place_direction.distance_to_goal.goal"),
        ("place_direction.distance_to_goal.goal", "place_direction.distance_to_goal.goal.egocentric_action"),
        (
            "place_direction.distance_to_goal.goal.egocentric_action",
            "place_direction.distance_to_goal.goal.egocentric_action.velocity",
        ),
    ],
}

FULL_FEATURE_INTERACTIONS = {
    "models": [
        "place-direction-velocity-distance_to_goal-egocentric_action",
        "place.direction-velocity-distance_to_goal-egocentric_action",
        "place.direction.velocity-distance_to_goal-egocentric_action",
        "place.direction.velocity.distance_to_goal.egocentric_action",
    ],
    "stats_comparisons": [
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
    ],
}

# %% cover all replacement model comp plotting fn


def plot_model_comparison(
    results_df,
    outlier_threshold=-0.6,
    models="all",
    model_groups=None,
    input_feature_groups=None,
    colors=None,
    plot_single_subjects=True,
    stats_comparisons=None,
    ax=None,
):
    if ax is None:
        f, ax = plt.subplots(figsize=(1.5, 3))
    ax = _init_fig(ax=ax)
    # process data
    df = _average_over_folds(results_df, outlier_threshold=outlier_threshold)
    # split models into groups based on naming convention is specified
    if model_groups is not None and input_feature_groups is not None:
        df.columns = pd.MultiIndex.from_tuples([tuple(c.split("_", 1)) for c in df.columns])
        # filter for input features and model types
        df = df[df.columns[df.columns.get_level_values(1).isin(input_feature_groups)]]
        df = df[df.columns[df.columns.get_level_values(0).isin(model_groups)]]
        df_long = (
            df.reset_index()
            .set_index(["cluster_unique_ID", "subject_ID"])
            .stack(level=[0, 1], future_stack=True)  # ← add future_stack=True
            .reset_index(name="score")
            .rename(columns={"level_2": "model_group", "level_3": "model_name"})
        )
        subj_avg = df_long.groupby(["subject_ID", "model_group", "model_name"])["score"].mean().reset_index()
    else:
        df_long = df.stack().reset_index(name="score")
        subj_avg = df_long.groupby(["subject_ID", "model_name"])["score"].mean().reset_index()
    if plot_single_subjects:
        sns.pointplot(
            data=df_long,
            x="model_name",
            y="score",
            hue="subject_ID",
            order=models if model_groups is None else None,
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
        hue=None if model_groups is None else "model_group",
        order=models if model_groups is None else None,
        marker="_",
        markersize=10,
        markeredgewidth=3,
        errorbar="se",
        palette="dark" if model_groups else None,
        color="black" if colors is None else None,
        linestyle="none",
        dodge=0.4 if model_groups else None,
        alpha=1,
        ax=ax,
    )
    x_tick_var = models if model_groups is None else input_feature_groups
    ax.set_xticks(range(len(x_tick_var)))
    ax.set_xticklabels(x_tick_var, rotation=45, ha="right")
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
