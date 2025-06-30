"""
Script for introductory distance to goal figure showing neurons spiking at different distances to goal
along trajectories in very different parts of the maze.
@peterdoohan
"""

# %% Imports
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.behaviour import trajectory_plotting as tp
from GridMaze.analysis.distance_to_goal import population_tuning as pt

# %% Global Variables


# %% Functions


def test(
    session,
    cluster_unique_IDs=[
        "m3.2022-07-14.maze_cluster48",
        "m3.2022-07-14.maze_cluster85",
        "m3.2022-07-14.maze_cluster75",
    ],
    trials=[33, 50, 38],
    axes=None,
):
    """ """
    # set up figure
    if axes is None:
        f, axes = plt.subplots(1, 1 + len(trials), figsize=(12, 3))
    axes[0].spines[["top", "right"]].set_visible(False)
    colors = plt.get_cmap("cool", len(cluster_unique_IDs))
    # get tuning curves
    Clusters = [gc.get_cluster(c) for c in cluster_unique_IDs]
    # plot tuning curves
    for i, Clust in enumerate(Clusters):
        Clust.plot_tuning(feature="distance_to_goal", feature_kwargs={"color": colors(i)}, ax=axes[0])

    navigation_spikes_df = session.get_navigation_activity_df(type="spikes")

    for ax, trial in zip(axes[1:], trials):
        tp.plot_trial_trajectories(session, trials=[trial], traj_colors=["grey"], ax=ax, linewidth=6)
        trial_df = navigation_spikes_df[
            (navigation_spikes_df.trial == trial) & (navigation_spikes_df.trial_phase == "navigation")
        ]
        x = trial_df.centroid_position.x.values
        y = trial_df.centroid_position.y.values
        for i, cuID in enumerate(cluster_unique_IDs):
            # add jitter to x, y
            _x = x + np.random.uniform(-0.02, 0.02, size=x.shape)
            _y = y + np.random.uniform(-0.02, 0.02, size=y.shape)
            spikes = trial_df.spike_count[cuID].values
            ax.scatter(_x[spikes > 0], _y[spikes > 0], color=colors(i), s=5, alpha=1, zorder=5)
            ax.set_title(trial)

    return
