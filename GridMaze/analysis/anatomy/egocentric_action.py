"""
Looks for anaomtical gradients in egocentric action tuning
"""

# %% imports
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from GridMaze.analysis.core import get_sessions as gs

# %% global variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)
# %% plotting


def plot_anatomical_egocentric_action_tuning(
    anat_df,
    actions=["turn_left", "turn_right"],
    min_split_half_corr=0.3,
    min_pref_action_factor=2,
    min_pref_action_frac=0.5,
    cmap="spring",
    jitter=2,
    axes=None,
    f=None,
):
    """ """
    # set up fig
    if axes is None or f is None:
        f, axes = plt.subplots(2, 4, figsize=(6, 4), width_ratios=[0.25, 2, 1, 0.1], height_ratios=[1, 0.15])
    for ax in [axes[0, 1], axes[0, 2], axes[1, 0], axes[0, 3], axes[1, 3]]:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, right=False, bottom=False, top=False, labelleft=False, labelbottom=False)
    for ax in [axes[0, 0], axes[1, 1], axes[1, 2]]:
        ax.spines[["right", "top"]].set_visible(False)
    axes[0, 0].set_ylabel("V -> D")
    axes[0, 0].set_yticks([])
    axes[1, 1].set_xlabel("P -> A")
    axes[1, 1].set_xticks([])
    axes[1, 2].set_xlabel("L -> M")
    axes[1, 2].set_xticks([])

    # filter for action tuned and non-action tuned cells
    action_tuned_mask = [anat_df.pref_action.name.isin(actions)]
    if min_split_half_corr is not None:
        action_tuned_mask.append(anat_df.split_half_corr.value.gt(min_split_half_corr))
    if min_pref_action_factor is not None:
        action_tuned_mask.append(anat_df.pref_action.factor.gt(min_pref_action_factor))
    if min_pref_action_frac is not None:
        action_tuned_mask.append(anat_df.pref_action.frac.gt(min_pref_action_frac))
    action_tuned_mask = np.logical_and.reduce(action_tuned_mask)
    tuned_df = anat_df[action_tuned_mask]
    non_tuned_df = anat_df[~action_tuned_mask]

    # plot non-tuned cells
    for ax, (x, y) in zip([axes[0, 1], axes[0, 2]], [("x", "y"), ("z", "y")]):
        _x = non_tuned_df["voxel"][x].values.astype(float)
        _y = non_tuned_df["voxel"][y].values.astype(float)
        if jitter:
            _x += np.random.uniform(-jitter, jitter, size=_x.shape)
            _y += np.random.uniform(-jitter, jitter, size=_y.shape)
        ax.scatter(_x, _y, color="grey", s=0.5, alpha=0.1)

    # plot tuned cells
    for ax, (x, y) in zip([axes[0, 1], axes[0, 2]], [("x", "y"), ("z", "y")]):
        _x = tuned_df["voxel"][x].values.astype(float)
        _y = tuned_df["voxel"][y].values.astype(float)
        if jitter:
            _x += np.random.uniform(-jitter, jitter, size=_x.shape)
            _y += np.random.uniform(-jitter, jitter, size=_y.shape)
        # color cell t_max
        t_maxs = tuned_df.pref_action.t_max.values
        vmin, vmax = -1, 1
        cmap = cm.get_cmap(cmap)
        norm = Normalize(vmin=vmin, vmax=vmax)
        if x == "z":
            sc = ax.scatter(_x, _y, c=t_maxs, cmap=cmap, norm=norm, s=0.5, alpha=0.6)
        else:
            ax.scatter(_x, _y, c=t_maxs, cmap=cmap, norm=norm, s=0.5, alpha=0.6)
    for ax in [axes[0, 1], axes[0, 2]]:
        ax.invert_yaxis()
        ax.invert_xaxis()
    cbar = f.colorbar(sc, ax=axes[0, 3], label="early --> late \n action tunned")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(left=False, labelleft=False, right=False, labelright=False)

    # plot marginals of on each axis of prop action tuned cells
    plot_anatomical_axis_egocentric_action_tuning(
        anat_df, axis="y", ax=axes[0, 0], n_axis_bins=10, plot_xy=False, label="V -> D"
    )
    plot_anatomical_axis_egocentric_action_tuning(
        anat_df, axis="x", actions=actions, ax=axes[1, 1], n_axis_bins=6, label="P -> A"
    )
    plot_anatomical_axis_egocentric_action_tuning(
        anat_df, axis="z", actions=actions, ax=axes[1, 2], n_axis_bins=6, label="L -> M"
    )


def plot_anatomical_axis_egocentric_action_tuning(
    anat_df,
    axis="y",
    actions=["turn_left", "turn_right"],
    min_split_half_corr=0.3,
    min_pref_action_factor=2,
    min_pref_action_frac=0.5,
    min_cells_per_bin=None,
    n_axis_bins=10,
    plot_xy=True,
    label=None,
    ax=None,
):
    """ """
    # prep figure
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(6, 2), sharey=True)
    ax.spines[["top", "right"]].set_visible(False)
    # add binary egoaction tunned label
    ego_action_tunned = np.logical_and.reduce(
        [
            anat_df.pref_action.name.isin(actions),
            anat_df.split_half_corr.value.gt(min_split_half_corr),
            anat_df.pref_action.factor.gt(min_pref_action_factor),
            anat_df.pref_action.frac.gt(min_pref_action_frac),
        ]
    )
    anat_df[("is_tunned", "")] = ego_action_tunned
    # plot prop action tunned along each axis
    axis_grouped = anat_df.groupby([("voxel", axis)]).is_tunned
    total_cells = axis_grouped.count()
    sum_tunned = axis_grouped.sum()
    # downsample voxels
    n_voxels = len(total_cells)
    if n_voxels > n_axis_bins:
        group_n_voxels = n_voxels // n_axis_bins
        _n_groups = np.minimum(np.arange(n_voxels) // group_n_voxels, n_axis_bins - 1)
        total_cells = total_cells.groupby(_n_groups).sum()
        sum_tunned = sum_tunned.groupby(_n_groups).sum()
    if min_cells_per_bin is not None:
        mask = total_cells.gt(min_cells_per_bin)
        prop_tunned = sum_tunned[mask] / total_cells[mask]
    else:
        prop_tunned = sum_tunned / total_cells
    x = prop_tunned.index.values.astype(float)
    y = prop_tunned.values.reshape(-1)
    if plot_xy:
        ax.plot(x, y, linewidth=1.5, color="k")
        ax.set_xlabel(label)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_ylim(0.08, 0.3)
    else:
        ax.plot(y, x, linewidth=1.5, color="k")
        ax.set_ylabel(label)
        ax.set_yticks([])
        ax.set_xlim(0.08, 0.3)

    if axis in ["z", "y"]:
        if plot_xy:
            ax.invert_xaxis()
        else:
            ax.invert_yaxis()


# %% compile anat data


def get_population_anatomy_df(subject_IDs="all", verbose=True):
    """ """
    subject_IDs = SUBJECT_IDS if subject_IDs == "all" else subject_IDs
    if verbose:
        print("Loading sessions...")
    sessions = gs.get_maze_sessions(
        subject_IDs=subject_IDs,
        maze_names="all",
        days_on_maze="all",
        with_data=["cluster_metrics", "cluster_egocentric_action_tuning_metrics"],
    )
    anat_dfs = []
    for session in sessions:
        if verbose:
            print(session.name)
        anat_df = get_session_egoaction_anat_df(session)
        anat_dfs.append(anat_df)
    results_df = pd.concat(anat_dfs, axis=0)
    return results_df


def get_session_egoaction_anat_df(session):
    """ """
    # load data
    ego_metrics_df = session.cluster_egocentric_action_tuning_metrics
    ego_metrics_df = ego_metrics_df[ego_metrics_df.single_unit]
    cluster_metrics = session.cluster_metrics
    cluster_metrics = cluster_metrics[cluster_metrics.single_unit]
    # combine ant and tuning info
    _output_df = pd.concat(
        [
            ego_metrics_df.xs("all_action", axis=1, level=1).reset_index(drop=True),
            cluster_metrics.xs("voxel", axis=1, level=0, drop_level=False).reset_index(drop=True),
            cluster_metrics.xs("region", axis=1, level=0, drop_level=False).reset_index(drop=True),
        ],
        axis=1,
    )
    _output_df[("subject_ID", "")] = session.subject_ID
    return _output_df
