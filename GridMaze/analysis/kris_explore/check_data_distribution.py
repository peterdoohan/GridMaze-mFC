""" """

# %% Imports
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from ..core import get_sessions as gs
from ..core import filter as filt
from ...maze import representations as mr

from GridMaze.maze import representations as mr
from GridMaze.maze import plotting as mp
from matplotlib.colors import LogNorm

from . import fit_experimental_data as fed

# %% Global Variables
FRAME_RATE = 60

# %% Functions


def plot_data_distribution_matrix(all_input_data, ax=None):
    """ """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(1, 10))
    feature_counts = (
        all_input_data.groupby(list(all_input_data.columns), observed=True).size().reset_index(name="counts")
    )
    feature_counts_matrix = feature_counts.pivot(
        index=["place", "direction"], columns="distance_to_goal", values="counts"
    ).values
    img = ax.imshow(
        feature_counts_matrix, cmap="Greens", norm=LogNorm(vmin=1e-2, vmax=np.nanmax(feature_counts_matrix))
    )
    cbar = fig.colorbar(img, ax=ax, label="Count (Log)")
    cbar.outline.set_visible(False)
    ax.set_xlabel("distance to goal")
    ax.set_ylabel("place-direction")
    ax.set_xticks([0, feature_counts_matrix.shape[1] - 1])
    ax.set_xticklabels(["goal", "start"])
    ax.set_yticks([])
    ax.set_yticklabels([])
    return


def plot_place_direction_distance_dist(all_input_data, maze_name="maze_1", log_scale=False, axes=None):
    distances = all_input_data.distance_to_goal.unique()
    if axes is None:
        fig, axes = plt.subplots(4, len(distances) // 4, figsize=(20, 20))
        axes = axes.flatten()
    simple_maze = mr.get_simple_maze(maze_name)
    max_obs = all_input_data.groupby(list(all_input_data.columns), observed=True).size().max()
    for i, dist in enumerate(distances):
        colorbar = True if i == len(distances) - 1 else False
        data_at_dist = all_input_data[all_input_data.distance_to_goal == dist]
        data_at_dist = data_at_dist.groupby(["place", "direction"]).count().distance_to_goal
        if log_scale:
            data_at_dist = data_at_dist.apply(lambda x: np.log10(x + 1))
            max_obs = np.log10(max_obs + 1)
        mp.plot_directed_heatmap(
            simple_maze,
            data_at_dist,
            ax=axes[i],
            fixed_vmax=max_obs,
            colorbar=colorbar,
            title=f"distance to goal: {dist:.2f}",
            silhouette_node_size=300,
            silhouette_edge_size=6,
            colormap="Greens",
        )


def check_total_data_dist(
    subject="m2",
    maze_name="maze_1",
):
    subject = [subject] if not subject == "all" else "all"
    sessions = gs.get_maze_sessions(
        subject_IDs=subject,
        maze_names=[maze_name],
        days_on_maze="late",
        with_data=["navigation_df"],
        must_have_data=True,
    )
    all_input_data = pd.concat([load_session_input_data(s) for s in sessions], axis=0)
    return all_input_data.reset_index(drop=True)


def load_session_input_data(
    session,
    distance_metrics=("distance_to_goal", "geodesic"),
    navigation_only=True,
    moving_only=True,
    resolution=0.5,  # s
    max_distance=1.8,  # m
    n_distance_bins=20,
):
    """ """
    navigation_df = session.navigation_df
    navigation_df = _downsample_navigation_df(navigation_df, resolution)
    navigation_df = filt.filter_navigation_rates_df(
        navigation_df,
        navigation_only=navigation_only,
        moving_only=moving_only,
        exclude_time_at_goal=False,
    )
    navigation_df = fed._remove_invalid_frames(navigation_df)
    # bin distance to goal
    bins = pd.interval_range(start=0, end=max_distance, freq=max_distance / n_distance_bins, closed="left")
    distance_bins = pd.cut(
        navigation_df[(distance_metrics[0], distance_metrics[1])],
        bins=bins,
    )
    distance_bin_mids = distance_bins.apply(lambda x: x.mid)
    # combine into input df
    input_data_df = pd.concat(
        [navigation_df.maze_position.simple, navigation_df.cardinal_movement_direction, distance_bin_mids], axis=1
    )
    input_data_df.columns = ["place", "direction", "distance_to_goal"]
    return input_data_df.dropna()


def _downsample_navigation_df(navigation_df, window_length):
    """ """
    combine_n_frames = int(FRAME_RATE * window_length)
    mid_window_indicies = (navigation_df.index // combine_n_frames).unique() * combine_n_frames + (
        combine_n_frames // 2
    )
    mid_window_indicies = mid_window_indicies[:-1]  # last index out of range
    return navigation_df.iloc[mid_window_indicies]
