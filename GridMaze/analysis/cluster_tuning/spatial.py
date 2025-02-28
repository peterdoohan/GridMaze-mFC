"""Library to plotting place direction heatmaps"""

# %% Imports
import numpy as np
import networkx as nx
from ...maze import plotting as mp
from ...maze import representations as mr
from ..core import get_clusters as gc
from ..core import filter

from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# %% Global Variables
FRAME_RATE = 60  # Hz

# %% Place-direction Tuning


def plot_session_place_direction_tuning(session):
    """ """
    # load processed place direction tuning (see analysis.processing.get_cluster_heatmap_dfs)
    place_direction_tuning_df = session.place_direction_tuning_df
    simple_maze = session.simple_maze()
    # only plot single units
    cluster_metrics = session.cluster_metrics
    session_info = session.session_info
    filtered_clusters = gc.filter_clusters(cluster_metrics, session_info, return_unique_IDs=True, single_units=True)
    for cluster_unique_ID in filtered_clusters:
        place_direction_tuning = place_direction_tuning_df.loc[cluster_unique_ID]
        plot_place_direction_tuning(place_direction_tuning, simple_maze)
    return


def plot_place_direction_tuning(simple_maze, place_direction_tuning_df, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(6, 6), clear=True)
    mp.plot_directed_heatmap(
        simple_maze,
        place_direction_tuning_df,
        colormap="heat",
        # title=place_direction_tuning_df.name,
        value_label="Firing Rate (Hz)",
        ax=ax,
    )
    return


# %% Place Tuning


def plot_session_place_tuning(session):
    """ """
    # load processed place tuning (see analysis.processing.get_cluster_heatmap_dfs)
    place_tuning_df = session.place_tuning_df
    simple_maze = session.simple_maze()
    goals = session.goals
    # only plot single units
    cluster_metrics = session.cluster_metrics
    session_info = session.session_info
    filtered_clusters = gc.filter_clusters(cluster_metrics, session_info, return_unique_IDs=True, single_units=True)
    for cluster_unique_ID in filtered_clusters[:10]:
        place_tuning = place_tuning_df.loc[cluster_unique_ID]
        plot_place_tuning(simple_maze, place_tuning, goals=goals)
    return


def plot_place_tuning(simple_maze, place_tuning_df, goals=None, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3), clear=True)
    mp.plot_simple_heatmap(
        simple_maze,
        place_tuning_df,
        colormap="heat",
        title=place_tuning_df.name,
        value_label="Firing Rate (Hz)",
        highlight_nodes=goals,
    )
    return


# %% 2D Space


def plot_session_spatial_tuning(session, navigation_only=False, moving_only=False, exclude_time_at_goal=False):
    """ """
    # load data
    simple_maze = session.simple_maze()
    goals = session.goals
    navigation_activity_df = session.get_navigation_activity_df(type="spikes", cluster_kwargs={"single_units": True})
    navigation_activity_df = filter.filter_navigation_rates_df(
        navigation_activity_df, navigation_only, moving_only, exclude_time_at_goal
    )
    pos = navigation_activity_df.centroid_position.to_numpy()
    cluster_unique_ID = navigation_activity_df.spike_count.columns.to_numpy()
    for cluster in cluster_unique_ID[:10]:
        spikes = navigation_activity_df.xs(cluster, level=1, axis=1).to_numpy().reshape(-1)
        plot_spatial_heatmap(simple_maze, pos, spikes, goals=goals)
    return


def plot_spatial_heatmap(pos, spike_counts, simple_maze, goals=False, bin_size=0.02, smooth_SD=0.04, ax=None):
    """Plots a spatial heatmap of neural firnig rate in 2D space"""
    rate_map, binx, biny = get_2D_ratemap(spike_counts, pos, x_size=bin_size, y_size=bin_size, smooth_SD=smooth_SD)
    if ax is None:
        fig, ax = plt.subplots()
    mp.plot_simple_maze_silhouette(
        simple_maze,
        ax=ax,
        color="gainsboro",
        highlight_nodes=goals,
    )
    ax.imshow(
        rate_map,
        extent=[binx[0], binx[-1], biny[0], biny[-1]],
        origin="lower",
        cmap="jet",  # mp.CUSTOM_COLORMAPS["heat"],
        zorder=2,
    )
    # add colorbar
    cbar = plt.colorbar(ax.images[0], ax=ax)
    cbar.set_label("Firing Rate (Hz)")
    # remove boarder from colorbar
    cbar.outline.set_visible(False)
    return


def get_2D_ratemap(
    spikes: np.ndarray,
    pos: np.ndarray,
    x_size: float = 0.02,  # Bin size in meters
    y_size: float = 0.02,  # Bin size in meters
    smooth_SD: float = 0.04,  # Smoothing window (standard deviation) in meters
):
    """
    Parameters
    ----------
    spikes: ndarray (n,)
        Number of spikes that occurred at each time step.
    pos: ndarray (n, 2)
        x, y coordinates representing the position of the animal when spikes occurred.
    x_size: float
        Bin size in meters for the x dimension.
    y_size: float
        Bin size in meters for the y dimension.
    smooth_SD: float
        Standard deviation of the Gaussian filter in meters. If 0, no smoothing will be applied.

    Returns
    -------
    h: ndarray (nybins, nxbins)
        Firing rate (Hz) falling on each bin through the recorded session. nybins is the number of bins in the y axis,
        nxbins is the number of bins in the x axis.
    binx: ndarray (nxbins +1,)
        Bin limits of the ratemap on the x axis.
    biny: ndarray (nybins +1,)
        Bin limits of the ratemap on the y axis.
    """
    x, y = pos[:, 0], pos[:, 1]  # Extract x and y coordinates

    # Determine the number of bins based on the range of x, y data and desired bin size
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # Calculate the number of bins based on the data range and the desired bin size
    nxbins = int((x_max - x_min) / x_size)
    nybins = int((y_max - y_min) / y_size)

    # Create a 2D histogram of the number of spikes (weighted histogram)
    spike_hist, binx, biny = np.histogram2d(
        x, y, bins=[nxbins, nybins], weights=spikes, range=[[x_min, x_max], [y_min, y_max]]
    )

    # Calculate the time spent in each bin (occupancy)
    # Time per frame is 1 / FRAME_RATE
    time_per_frame = 1 / FRAME_RATE
    # Create a 2D histogram of the number of frames spent in each bin
    occupancy, _, _ = np.histogram2d(x, y, bins=[nxbins, nybins], range=[[x_min, x_max], [y_min, y_max]])
    occupancy *= time_per_frame  # Convert frame count to time spent in each bin (seconds)

    # Compute the firing rate (spikes per second) for each bin
    # Avoid division by zero by only dividing where occupancy > 0
    h = np.zeros_like(spike_hist)
    valid_bins = occupancy > 0
    h[valid_bins] = spike_hist[valid_bins] / occupancy[valid_bins]

    # Apply Gaussian smoothing if smooth_SD > 0
    if smooth_SD > 0:
        # Convert smoothing window (SD) from meters to bin units
        sigma_x = smooth_SD / x_size
        sigma_y = smooth_SD / y_size
        h = gaussian_filter(h, sigma=[sigma_x, sigma_y])

    # Set bins to np.nan if they were not visited
    h[occupancy == 0] = np.nan

    # Transpose to change row-column coordinates to positions
    return h.T, binx, biny
