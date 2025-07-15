"""
Look for anatomical gradients in distance to goal tuning
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.stats import gamma

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.distance_to_goal import population_tuning as pt


# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

# %% Functions


def plot_anatomical_distance_tuning(anat_df, f=None, axes=None, jitter=2, colormap="cool"):
    """ """
    dist_tuned_cells = anat_df[anat_df.distance_tuned]
    other_cells = anat_df[~anat_df.distance_tuned]
    if axes is None or f is None:
        f, axes = plt.subplots(1, 2, figsize=(5, 3), width_ratios=[2, 1])
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, right=False, bottom=False, top=False, labelleft=False, labelbottom=False)
    axes[0].set_xlabel("P -> A")
    axes[0].set_ylabel("V -> D")
    axes[1].set_xlabel("L -> M")
    axes[1].set_ylabel("V -> D")
    # plot non-distance tunned cells
    for ax, (x, y) in zip(axes, [("x", "y"), ("z", "y")]):
        _x = other_cells["voxel"][x].values.astype(float)
        _y = other_cells["voxel"][y].values.astype(float)
        if jitter:
            _x += np.random.uniform(-jitter, jitter, size=_x.shape)
            _y += np.random.uniform(-jitter, jitter, size=_y.shape)
        ax.scatter(_x, _y, color="grey", s=0.5, alpha=0.1)
    # plot distance tuned cells
    for ax, (x, y) in zip(axes, [("x", "y"), ("z", "y")]):
        _x = dist_tuned_cells["voxel"][x].values.astype(float)
        _y = dist_tuned_cells["voxel"][y].values.astype(float)
        if jitter:
            _x += np.random.uniform(-jitter, jitter, size=_x.shape)
            _y += np.random.uniform(-jitter, jitter, size=_y.shape)
        # color cell by distance tuning
        dist_p50 = dist_tuned_cells.distance_p50.values
        vmin, vmax = dist_p50.min(), 1.5
        cmap = cm.get_cmap(colormap)
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        # colors = cmap(norm(dist_p50))
        if x == "z":
            sc = ax.scatter(_x, _y, c=dist_p50, cmap=cmap, norm=norm, s=0.5, alpha=0.3)
        else:
            ax.scatter(_x, _y, c=dist_p50, cmap=cmap, norm=norm, s=0.5, alpha=0.3)
    for ax in axes:
        ax.invert_yaxis()
        ax.invert_xaxis()
    cbar = f.colorbar(sc, ax=axes[1], label="close --> far \n distance tunned")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(left=False, labelleft=False, right=False, labelright=False)


def get_population_anatomy_df(subject_IDs="all", verbose=False):
    """"""
    subject_IDs = SUBJECT_IDS if subject_IDs == "all" else subject_IDs
    if verbose:
        print("Loading sessions...")
    sessions = gs.get_maze_sessions(
        subject_IDs=subject_IDs,
        maze_names="all",
        days_on_maze="all",
        with_data=["cluster_metrics", "cluster_distance_tuning_metrics"],
    )
    anat_dfs = []
    for session in sessions:
        if verbose:
            print(session.name)
        anat_df = get_session_anatomical_distance_tuning(session)
        anat_dfs.append(anat_df)
    results_df = pd.concat(anat_dfs, axis=0)
    return results_df


def get_session_anatomical_distance_tuning(session, min_split_half_corr=0.5):
    """
    min_split_half_corr defines if neuron is distance tuned or not
    """
    # load data
    distance_tuning_metrics = session.cluster_distance_tuning_metrics
    distance_tuning_metrics = distance_tuning_metrics[distance_tuning_metrics.single_unit]
    cluster_metrics = session.cluster_metrics  # with anatomy data
    cluster_metrics = cluster_metrics[cluster_metrics.single_unit].reset_index(drop=True)
    # get distance tuning info
    output_df = pd.DataFrame(index=distance_tuning_metrics.cluster_unique_ID)
    params_df = distance_tuning_metrics.gamma_4p
    output_df[("distance_p50", "")] = gamma.ppf(
        0.5, loc=0, a=params_df["shape"].values, scale=params_df["scale"].values
    )
    split_half_corrs = distance_tuning_metrics.split_half_corr.value.values
    output_df[("split_half_corr", "")] = split_half_corrs
    output_df[("distance_tuned", "")] = split_half_corrs > min_split_half_corr
    # get anatomy info
    _output_df = pd.concat(
        [
            output_df.reset_index(drop=True),
            cluster_metrics.xs("voxel", axis=1, level=0, drop_level=False).reset_index(),
            cluster_metrics.xs("region", axis=1, level=0, drop_level=False).reset_index(),
        ],
        axis=1,
    )
    _output_df.index = output_df.index
    _output_df.columns = pd.MultiIndex.from_tuples(_output_df.columns)
    return _output_df
