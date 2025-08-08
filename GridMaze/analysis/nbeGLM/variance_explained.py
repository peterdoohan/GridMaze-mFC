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

from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


from GridMaze.analysis.nbeGLM.load_model_sets import load_model_set_cv_scores
from GridMaze.analysis.nbeGLM import model_comparisons as mc


# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

# %% Functions


def plot_variance_explained(
    cpd_df, features=["distance_to_goal", "place_direction", "egocentric_action_action"], ax=None
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
    df = cpd_df[features]
    long_df = df.stack().reset_index(name="score").rename(columns={"level_2": "feature"})
    subject_av = long_df.groupby(["subject_ID", "feature"])["score"].mean().reset_index()
    colors = sns.color_palette("hls", n_colors=len(SUBJECT_IDS))
    sns.pointplot(
        data=long_df,
        x="feature",
        y="score",
        hue="subject_ID",
        palette=colors,
        markers="o",
        markersize=8,
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
        x="feature",
        y="score",
        marker="_",
        color="k",
        markersize=14,
        markeredgewidth=3,
        errorbar="se",
        linestyle="none",
        alpha=1,
        ax=ax,
    )
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
    reduced_models=[
        "remove_distance_to_goal",
        "remove_place_direction",
        "remove_egocentric_action_action",
    ],
):
    """ """
    # average over folds & remove neurons with large negative scores
    df = mc._average_over_folds(results_df, outlier_threshold=outlier_threshold)
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
