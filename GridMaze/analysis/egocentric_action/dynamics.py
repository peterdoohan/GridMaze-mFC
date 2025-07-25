"""
Visualise population dynamics around egocentric actions
@peterdoohan
"""

# %% Imports
from cv2 import subtract
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from GridMaze.analysis.egocentric_action import population_tuning as pt

# %% Global Variables

# %% Functions


def get_population_tuning_df(
    subject_IDs="all",
    maze_names=["maze_1", "maze_2"],  # frew forced choice on rooms_maze
    late_sessions=False,
    sessions=None,
    actions=["turn_left", "turn_right", "go_forward"],
    window=(-3, 3),
    min_split_half_corr=False,
    smooth_SD=16,
    normalise="zscore",
    fill_nan="mean",
    verbose=False,
):
    """ """
    tuning_df, metrics_df = pt.get_population_egocentric_action_tuning(
        subject_IDs,
        maze_names,
        late_sessions,
        sessions=sessions,
        actions=actions,
        include_action_type=True,
        window=window,
        min_split_half_corr=min_split_half_corr,
        max_jobs=10,
        with_metrics=True,
        verbose=verbose,
    )
    # smooth tuning curves
    if smooth_SD:
        tcs = tuning_df.values
        tuning_df = pd.DataFrame(
            data=gaussian_filter1d(tcs, smooth_SD, axis=1), index=tuning_df.index, columns=tuning_df.columns
        )
    wide_df = (
        tuning_df.unstack(level=[2, 1])
        .action_aligned_rates.swaplevel(0, 1, axis=1)
        .swaplevel(1, 2, axis=1)
        .sort_index(axis=1)
    )  # n_neurons, free-or-forced x action x time
    # normalise tuning curves before PCA
    if normalise == "zscore":
        tcs = wide_df.values
        wide_df = pd.DataFrame(data=zscore(tcs, axis=1), index=wide_df.index, columns=wide_df.columns)
    else:
        raise NotImplementedError
    # deal with NaNs ()
    if fill_nan == "zero":
        wide_df = wide_df.fillna(0)
    elif fill_nan == "mean":
        wide_df = wide_df.fillna(wide_df.mean(axis=1), axis=0)
    else:
        raise NotImplementedError

    return wide_df, metrics_df


def PC_plot(
    wide_df,
    metrics_df,
    actions=["turn_left", "turn_right", "go_forward"],
    colors=["darkviolet", "royalblue", "grey"],
    min_spit_half_corr=False,
    min_pref_action_factor=False,
    min_pref_action_frac=False,
    crop_window=False,
    subtract_action_mean=False,
    PCs=(0, 1, 2),
    f=None,
    ax=None,
):
    # set up fig
    if ax is None or f is None:
        if len(PCs) == 3:
            f, ax = _init_3D_plot(PCs)
        else:
            f, ax = _init_2D_plot(PCs)

    # filter clusters going into PCA
    keep_clusters = _filter_clusters(
        metrics_df,
        min_spit_half_corr=min_spit_half_corr,
        min_pref_action_factor=min_pref_action_factor,
        min_pref_action_frac=min_pref_action_frac,
    )
    df = wide_df.loc[keep_clusters]

    # clip window around action to going into PCA
    if crop_window:
        timepoints = df.columns.get_level_values(2).astype(float)
        crop_mask = (timepoints >= crop_window[0]) & (timepoints <= crop_window[1])
        df = df.loc[:, crop_mask]

    # subtract the mean across action types
    if subtract_action_mean:
        tuning_df = df.stack(level=[0, 1], future_stack=True)
        action_type_mean_df = tuning_df.groupby(level=0).mean()
        norm_tuning_df = tuning_df - action_type_mean_df
        df = norm_tuning_df.unstack(level=[2, 1]).swaplevel(0, 2, axis=1).sort_index(axis=1)

    # do PCA
    X = df.values  # n_neurons, n_tuning curves (2 (free/forced) x n_actions x timepoints)
    pca = PCA(random_state=0, whiten=True)
    pca.fit(X.T)
    pc_components = pca.components_

    # plot
    for choice_type, ls in zip(["free", "forced"], ["-", "--"]):
        choice_type_activity = df[choice_type]
        for action, color in zip(actions, colors):
            action_activity = choice_type_activity[action]
            A = action_activity.values  # n_neurons, timepoints
            if len(PCs) == 3:
                _3d_plot(A, pc_components, PCs, ax, color, ls, choice_type, action)
            elif len(PCs) == 2:
                _2d_plot(A, pc_components, PCs, ax, color, ls, choice_type, action)
            else:
                raise NotImplementedError(f"PCs must be 2 or 3, got {len(PCs)}")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))


def _3d_plot(A, pc_components, PCs, ax, color, ls, choice_type, action):
    traj_x = A.T @ pc_components[PCs[0], :]
    traj_y = A.T @ pc_components[PCs[1], :]
    traj_z = A.T @ pc_components[PCs[2], :]
    ax.plot(traj_x, traj_y, traj_z, color=color, linestyle=ls, label=f"{choice_type} {action}", lw=2)
    # add markers for the start and end of each trajectory
    facecolor = "none" if choice_type == "forced" else color
    ax.scatter(traj_x[0], traj_y[0], traj_z[0], color=color, marker="o", facecolor=facecolor, s=50)
    ax.scatter(traj_x[-1], traj_y[-1], traj_z[-1], color=color, marker="*", facecolor=facecolor, s=50)


def _2d_plot(A, pc_components, PCs, ax, color, ls, choice_type, action):
    traj_x = A.T @ pc_components[PCs[0], :]
    traj_y = A.T @ pc_components[PCs[1], :]
    ax.plot(traj_x, traj_y, color=color, linestyle=ls, label=f"{choice_type} {action}", lw=2)
    # add markers for the start and end of each trajectory
    facecolor = "none" if choice_type == "forced" else color
    ax.scatter(traj_x[0], traj_y[0], color=color, marker="o", facecolor=facecolor, s=50)
    ax.scatter(traj_x[-1], traj_y[-1], color=color, marker="*", facecolor=facecolor, s=50)


def _filter_clusters(
    metrics_df,
    min_spit_half_corr=0.25,
    min_pref_action_factor=False,
    min_pref_action_frac=False,
):
    """ """
    masks = [metrics_df.single_unit]
    if min_spit_half_corr:
        masks.append(metrics_df.split_half_corr.all_action.value.gt(min_spit_half_corr))
    if min_pref_action_factor:
        masks.append(metrics_df.pref_action.all_action.factor.gt(min_pref_action_factor))
    if min_pref_action_frac:
        masks.append(metrics_df.pref_action.all_action.frac.gt(min_pref_action_frac))
    combined_mask = pd.concat(masks, axis=1).all(axis=1)
    keep_clusters = metrics_df[combined_mask].index.values
    return keep_clusters


def _init_3D_plot(PCs, figsize=(5, 5)):
    f = plt.figure(figsize=figsize)
    ax = f.add_subplot(111, projection="3d")
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel(f"PC{PCs[0]}")
    ax.set_ylabel(f"PC{PCs[1]}")
    ax.set_zlabel(f"PC{PCs[2]}")
    return f, ax


def _init_2D_plot(PCs, figsize=(2, 2)):
    f, ax = plt.subplots(figsize=figsize)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(f"PC{PCs[0]}")
    ax.set_ylabel(f"PC{PCs[1]}")
    ax.set_xticks([])
    ax.set_yticks([])
    return f, ax
