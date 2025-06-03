"""
Library for population dynamics analysis on GridMaze data
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from dPCA import dPCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm


from GridMaze.analysis.core import get_sessions as gs
from GridMaze.maze import plotting as mp

# %% Global Variables
from GridMaze.paths import ANALYSIS_INFO_PATH

with open(ANALYSIS_INFO_PATH / "intra_trial_interval_times.json", "r") as input_file:
    INTRA_TRIAL_INTERVAL_TIMES = json.load(input_file)

# %% New


def test2(condition_aligned_rates):
    """ """
    df = condition_aligned_rates.firing_rate
    goals = df.columns.get_level_values(1).unique().values
    n_neurons = df.shape[0]
    n_goals = len(goals)
    n_timepoints = len(df.columns.get_level_values(0).unique())
    X = np.zeros((n_neurons, n_goals, n_timepoints))
    for i, goal in enumerate(goals):
        X[:, i, :] = df.xs(goal, level=1, axis=1).values
    # demean
    X = X - np.mean(X, axis=2, keepdims=True)  # [n_neurons x n_goals x n_timepoints]
    # do dPCA
    dpca = dPCA.dPCA(
        labels="gt",
        n_components=3,
        regularizer=None,
    )
    Z = dpca.fit_transform(X)
    # plotting
    f = plt.figure(figsize=(8, 6))
    ax = f.add_subplot(111, projection="3d")
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    # plot
    for i in range(n_goals):
        x_traj = Z["t"][0, i, :]
        y_traj = Z["t"][1, i, :]
        z_traj = Z["gt"][0, i, :]
        ax.plot(x_traj, y_traj, z_traj, label=goals[i], alpha=1)

    return


def test(
    maze_name="maze_1",
    goal_subset="subset_2",
    late_sessions=False,
    PCs=(0, 1, 2),
    single_units=True,
    smooth_SD=20,
    ax=None,
):
    """ """

    # load data
    days_on_maze = "late" if late_sessions else "all"
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names=[maze_name],
        days_on_maze=days_on_maze,
        goal_subsets=[goal_subset],
        with_data=["trial_aligned_rates_df", "cluster_metrics"],
        must_have_data=True,
    )
    # combine trial-aligned (warped) rates across all neurons and average across trials for each goal
    dfs = []
    for session in sessions:
        df = session.trial_aligned_rates_df
        if not len(set(session.goals) - set(df.goal.unique())) == 0:
            continue
        if single_units:
            cluster_metrics = session.cluster_metrics
            keep_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID
            df = df[df.cluster_ID.isin(keep_units)]
        dfs.append(df)
    trial_aligned_rates = pd.concat(dfs, ignore_index=True)
    # smooth rates
    if smooth_SD:
        rates = trial_aligned_rates.firing_rate.values
        smoothed_rates = gaussian_filter1d(rates, sigma=smooth_SD, axis=1)
        trial_aligned_rates.loc[:, "firing_rate"] = smoothed_rates
    trial_x_goal_aligned_rates = trial_aligned_rates.groupby(
        ["cluster_unique_ID", "goal"]
    ).firing_rate.mean()  # [n_clusters x n_goals x timepoints]
    condition_aligned_rates = trial_x_goal_aligned_rates.unstack().sort_index(
        axis=1, level=[0, 2]
    )  # clusters x [timepoints x goals]
    return condition_aligned_rates
    PC_plot(condition_aligned_rates, PCs=PCs)

    return


def PC_plot(condition_aligned_rates, PCs=(0, 1, 2), pre_cue=0, post_ERC=4, ax=None):
    """ """
    # set up figure
    if ax is None:
        f = plt.figure(figsize=(8, 6))
        ax = f.add_subplot(111, projection="3d")
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.set_xlabel(f"PC{PCs[0]}")
    ax.set_ylabel(f"PC{PCs[1]}")
    ax.set_zlabel(f"PC{PCs[2]}")
    # remove times
    pre_cue_mask = condition_aligned_rates.columns.get_level_values(1) > -pre_cue
    post_ERC_mask = (
        condition_aligned_rates.columns.get_level_values(1)
        < INTRA_TRIAL_INTERVAL_TIMES["end_reward_consumption"] + post_ERC
    )
    condition_aligned_rates = condition_aligned_rates[condition_aligned_rates.columns[pre_cue_mask & post_ERC_mask]]
    # get event timpoints
    timepoints = condition_aligned_rates.columns.get_level_values(1).unique().values
    event2t_ind = {}
    for event, time in INTRA_TRIAL_INTERVAL_TIMES.items():
        event2t_ind[event] = np.argmin(abs(timepoints - time).astype(float))
    mask = timepoints < INTRA_TRIAL_INTERVAL_TIMES["reward"]
    # do PCA
    X = condition_aligned_rates.values
    pca = PCA()
    pca.fit(X.T)
    pc_componets = pca.components_
    # plot condition (goal) trajectories in PCA space
    goals = condition_aligned_rates.columns.get_level_values(2).unique().values
    colors = cm.get_cmap("jet", len(goals))  # use tab10 colormap
    for i, goal in enumerate(goals):
        condition_activity = condition_aligned_rates.xs(goal, level=2, axis=1)
        C = condition_activity.values  # [n_clusters x timepoints]
        traj_x = C.T @ pc_componets[PCs[0], :]
        traj_y = C.T @ pc_componets[PCs[1], :]
        traj_z = C.T @ pc_componets[PCs[2], :]
        ax.plot(
            traj_x[~mask],
            traj_y[~mask],
            traj_z[~mask],
            label=goal,
            alpha=1,
            color=colors(i),
        )
        # plot markers fo
        for key, t_ind in zip(["$R$", "$E$"], [event2t_ind["reward"], event2t_ind["ITI_end"]]):
            ax.scatter(
                traj_x[t_ind],
                traj_y[t_ind],
                traj_z[t_ind],
                marker=key,
                color="k",
                s=50,
                alpha=0.5,
            )

    return


# %% Functions
