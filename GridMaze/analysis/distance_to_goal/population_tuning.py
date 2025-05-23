"""
Library for distance to goal tuning analyses: curve fits, headmaps etc.
"""

# %% Imports
from cProfile import label
import json
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from GridMaze.analysis.core import get_sessions as gs

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)


CURVE_FITS = ["gamma_4p", "gaussian_4p", "polynomial_4p"]


# %%


# %% Curve fit summary and plotting functions


def plot_curve_fit_distributions(summary_df, curve_fits=CURVE_FITS, ax=None):
    """ """
    df = summary_df[curve_fits]
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.set_xlabel("R2")
    ax.set_ylabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    for col in df.columns:
        sns.histplot(
            data=df[col],
            stat="count",
            kde=True,
            element="step",
            fill=False,
            ax=ax,
            label=col,
            alpha=0,
        )
    ax.legend(loc="upper left")


def plot_cross_subject_curve_fit_comparison(summary_df, curve_fits=CURVE_FITS, ax=None):
    """
    Make pretty later
    """
    # process data
    df = summary_df.groupby("subject_ID")[curve_fits].mean().unstack().reset_index()
    df.columns = ["fit", "subject_ID", "r2"]

    # plot
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("mean R2")
    ax.set_ylim(0.65, 0.85)
    sns.pointplot(
        data=df,
        x="fit",
        y="r2",
        hue="subject_ID",
        errorbar=None,
        dodge=False,
        markers="o",
        linestyles="-",
        legend=False,
        markersize=8,
        linewidth=4,
    )
    ax.tick_params(axis="x", which="both", top=False, bottom=True, labeltop=False, labelbottom=True, labelrotation=45)
    return


def get_tuning_fits_summary_df(late_sessions=True, min_split_half_corr=0.5):
    """
    Compare how well different distribution shapes (gamma, gaussian, polynomial) fit empirical
    distance to goal tuning curves.
    """
    days_on_maze = "late" if late_sessions else "all"
    subject_dfs = []
    for subject in SUBJECT_IDS:
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject],
            maze_names="all",
            days_on_maze=days_on_maze,
            with_data=["cluster_distance_tuning_metrics"],
        )
        r2_dfs = []
        for session in sessions:
            metrics_df = session.cluster_distance_tuning_metrics
            # select for "distance_tunned" clusters
            metrics_df = metrics_df[
                (metrics_df.distance_tuned) & (metrics_df.split_half_corr.value.gt(min_split_half_corr))
            ]
            # curve fit params and r2 values calculated in analysis/processing/get_distance_tuning_metrics_df.py
            r2_df = metrics_df[[(c, "r2") for c in CURVE_FITS]].droplevel(1, axis=1)
            r2_dfs.append(r2_df)
        subject_df = pd.concat(r2_dfs, axis=0).reset_index(drop=True)
        subject_df["subject_ID"] = subject
        subject_dfs.append(subject_df)
    # do cross subject states

    # combine all cells and plot distributions
    summary_df = pd.concat(subject_dfs, axis=0).reset_index(drop=True)

    return summary_df
