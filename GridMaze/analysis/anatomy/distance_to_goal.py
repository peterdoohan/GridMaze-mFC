"""
Look for anatomical gradients in distance to goal tuning
@peterdoohan
"""

# %% Imports
import re
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.distance_to_goal import population_tuning as pt


# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

# %% Functions


def plot_voxel_distance_tuning_heatmap(
    population_anatomy_df,
    ax=None,
):
    """ """
    # process data
    voxel_map = population_anatomy_df.groupby(["y", "x"]).tunned_distance.mean().unstack()
    # plot
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3))
    # plot heatmap
    sns.heatmap(
        voxel_map,
        cmap="viridis_r",
        cbar_kws={"label": "Tunned Distance", "shrink": 0.5},
        square=True,
        alpha=1,
        ax=ax,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("Anterior -> Posterior")
    ax.set_ylabel("Ventral -> Doral")


def plot_region_distance_tuning_distributions(population_anatomy_df, ignore_layers=True, min_cells=20, axes=None):
    """ """
    if ignore_layers:
        population_anatomy_df["region"] = population_anatomy_df.region.apply(
            lambda x: re.match(r"([A-Za-z]+)(.*)", x).groups()[0]
        )
    if min_cells is not None:
        cell_counts = population_anatomy_df.groupby("region").count().tunned_distance
        regions = cell_counts[cell_counts.gt(min_cells)].index.values
    else:
        regions = population_anatomy_df.region.unique()
    if axes is None:
        f, axes = plt.subplots(len(regions), 1, figsize=(3, len(regions)), sharex=True, sharey=True)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    for region, ax in zip(regions, axes.flatten()):
        region_df = population_anatomy_df[population_anatomy_df.region == region]
        sns.histplot(region_df, x="tunned_distance", stat="proportion", element="step", alpha=0.2, ax=ax, color="black")
        ax.set_ylabel(region)

    return


def get_population_anatomy_df(subject_IDs="all", late_sessions=True, sign="pos", verbose=False):
    """"""
    days_on_maze = "late" if late_sessions else "all"
    subject_IDs = SUBJECT_IDS if subject_IDs == "all" else subject_IDs
    if verbose:
        print("Loading sessions...")
    sessions = gs.get_maze_sessions(
        subject_IDs=subject_IDs,
        maze_names="all",
        days_on_maze=days_on_maze,
        with_data=["navigation_df", "navigation_spike_rates_df", "cluster_metrics", "cluster_distance_tuning_metrics"],
    )
    anat_dfs = []
    for session in sessions:
        if verbose:
            print(session.name)
        anat_df = _get_session_anatomical_distance_tuning(session, sign=sign)
        anat_dfs.append(anat_df)
    results_df = pd.concat(anat_dfs, axis=0).reset_index(drop=True)
    return results_df


def _get_session_anatomical_distance_tuning(session, sign="pos", fit="gamma_4p"):
    """
    returns df with x,y voxel coordinates and distance tuning peak for each cluster
    """
    distance_tuning_df = pt._get_session_distance_tuning(session)
    cluster_distance_tuning_metrics_df = session.cluster_distance_tuning_metrics
    cluster_metrics_df = session.cluster_metrics
    # filter for distance tunned clusters
    distance_tuned_mask = (
        cluster_metrics_df.single_unit
        & cluster_distance_tuning_metrics_df.distance_tuned
        & cluster_distance_tuning_metrics_df.gamma_4p_cv.sig
    )
    cluster_distance_tuning_metrics_df = cluster_distance_tuning_metrics_df[distance_tuned_mask]
    cluster_metrics_df = cluster_metrics_df[distance_tuned_mask]
    # filter for pos/neg fit tuned
    if sign == "pos":
        sign_mask = cluster_distance_tuning_metrics_df[fit]["size"].gt(0)
    else:  # neg
        sign_mask = cluster_distance_tuning_metrics_df[fit]["size"].lt(0)
    cluster_distance_tuning_metrics_df = cluster_distance_tuning_metrics_df[sign_mask]
    cluster_metrics_df = cluster_metrics_df[sign_mask]
    distance_tuning_df = distance_tuning_df.loc[cluster_distance_tuning_metrics_df.cluster_unique_ID.values]
    df = pd.concat(
        [
            distance_tuning_df.reset_index(drop=True),
            cluster_distance_tuning_metrics_df.reset_index(drop=True),
            cluster_metrics_df.reset_index(drop=True),
        ],
        axis=1,
    )
    # get distance tuning peak
    x = df.distance_to_goal.columns.values.astype(float)
    anat_df = df.voxel[["x", "y"]]
    anat_df["tunned_distance"] = df.apply(lambda row: pt.get_idx_order(row, x, fit=fit, op="max"), axis=1)
    anat_df["region"] = df.region.acronym
    return anat_df
