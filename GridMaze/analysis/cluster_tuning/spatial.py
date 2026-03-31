"""Library to plotting place direction heatmaps"""

# %% Imports
import numpy as np
import pandas as pd
from GridMaze.maze import plotting as mp
from GridMaze.maze import representations as mr
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import filter as filt


from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# %% Global Variables
FRAME_RATE = 60  # Hz

# %% Place-direction Tuning


def plot_session_place_direction_tuning(session):
    """ """
    # load data
    simple_maze = session.simple_maze()
    navigation_rates_df = session.get_navigation_activity_df(type="rates", cluster_kwargs={"single_units": True})
    # filter and average across place-direction
    place_direction_tuning_df = _get_place_direction_df(
        simple_maze,
        navigation_rates_df,
        navigation_only=True,
        moving_only=True,
        exclude_time_at_goal=True,
        minimum_occupancy=0.5,
        max_steps_from_goal=30,
    )
    cluster_unique_IDs = navigation_rates_df.firing_rate.columns.values
    for cluster_unique_ID in cluster_unique_IDs:
        place_direction_tuning = place_direction_tuning_df.loc[cluster_unique_ID]
        plot_place_direction_tuning(simple_maze, place_direction_tuning)
    return


def plot_place_direction_tuning(
    simple_maze,
    place_direction_tuning_df,
    colormap="heat",
    fixed_vmin=False,
    silhouette_node_size=650,
    silhouette_edge_size=13,
    star_base_length=0.05,
    max_point_length=0.026,
    ax=None,
):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(5, 5), clear=True)
    mp.plot_directed_heatmap(
        simple_maze,
        place_direction_tuning_df,
        colormap=colormap,
        fixed_vmin=fixed_vmin,
        silhouette_node_size=silhouette_node_size,
        silhouette_edge_size=silhouette_edge_size,
        star_base_length=star_base_length,
        max_point_length=max_point_length,
        value_label="Firing Rate (Hz)",
        ax=ax,
    )
    return


def _get_place_direction_df(
    simple_maze,
    navigation_rates_df,
    navigation_only,
    moving_only,
    exclude_time_at_goal,
    minimum_occupancy,
    max_steps_from_goal=30,
):
    navigation_rates_df = filt.filter_navigation_rates_df(
        navigation_rates_df,
        navigation_only,
        moving_only,
        exclude_time_at_goal,
        max_steps_from_goal,
    )
    cluster_unique_IDs = navigation_rates_df.firing_rate.columns.to_numpy()
    place_direction_cols = [("maze_position", "simple"), ("cardinal_movement_direction", "")]
    place_direction_grouped_df = navigation_rates_df.set_index(place_direction_cols).groupby(place_direction_cols)
    place_direction_av_rates_df = place_direction_grouped_df.firing_rate.mean().firing_rate
    # set low occupancy cdirs to nan for all clusters
    place_direction_av_rates_df[place_direction_grouped_df.count().time < minimum_occupancy * FRAME_RATE] = np.nan
    ###
    visited_place_directions = place_direction_av_rates_df.index.to_numpy()
    all_place_directions = mr.get_maze_place_direction_pairs(simple_maze)
    unvistied_place_directions = list(set(all_place_directions) - set(visited_place_directions))
    if len(unvistied_place_directions) > 0:
        unvisited_place_direction_nan_rates = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(unvistied_place_directions), columns=cluster_unique_IDs, data=np.nan
        )
        place_direction_av_rates_df = pd.concat(
            [place_direction_av_rates_df, unvisited_place_direction_nan_rates], axis=0
        )
    place_direction_av_rates_df = place_direction_av_rates_df.reindex(sorted(place_direction_av_rates_df.index), axis=0)
    place_direction_df = place_direction_av_rates_df.T  # [cluster_unique_IDs, location_cdirs]
    place_direction_df.columns.names = ["maze_position", "direction"]
    place_direction_df.sort_index(axis=1, inplace=True)
    return place_direction_df


# %% Place Tuning


def plot_session_place_tuning(session):
    """ """
    # load data
    navigation_rates_df = session.get_navigation_activity_df(type="rates", cluster_kwargs={"single_units": True})
    simple_maze = session.simple_maze()
    place_tuning_df = _get_place_df(simple_maze, navigation_rates_df)
    goals = session.goals
    cluster_unique_IDs = navigation_rates_df.firing_rate.columns.values
    for cluster_unique_ID in cluster_unique_IDs:
        place_tuning = place_tuning_df.loc[cluster_unique_ID]
        plot_place_tuning(simple_maze, place_tuning, goals=goals)
    return


def plot_place_tuning(simple_maze, place_tuning_df, goals=None, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(4, 4), clear=True)
    mp.plot_simple_heatmap(
        simple_maze,
        place_tuning_df,
        colormap="heat",
        title=place_tuning_df.name,
        value_label="Firing Rate (Hz)",
        highlight_nodes=goals,
        ax=ax,
    )
    return


def _get_place_df(
    simple_maze,
    navigation_rates_df,
    navigation_only=True,
    moving_only=True,
    exclude_time_at_goal=True,
    minimum_occupancy=1,
    max_steps_from_goal=30,
):
    """ """
    navigation_rates_df = filt.filter_navigation_rates_df(
        navigation_rates_df,
        navigation_only,
        moving_only,
        exclude_time_at_goal,
        max_steps_from_goal,
    )
    place_direction_grouped_df = navigation_rates_df.groupby([("maze_position", "simple")])
    cluster_unique_IDs = navigation_rates_df.firing_rate.columns.to_numpy()
    place_averaged_rates_df = place_direction_grouped_df.firing_rate.mean().firing_rate
    place_averaged_rates_df[place_direction_grouped_df.count().time < minimum_occupancy * FRAME_RATE] = np.nan
    all_places = mr.get_maze_locations(simple_maze)
    unvisited_places = list(set(all_places) - set(place_averaged_rates_df.index))
    place_averaged_rates_df = pd.concat(
        [place_averaged_rates_df, pd.DataFrame(index=unvisited_places, columns=cluster_unique_IDs, data=np.nan)]
    )
    place_averaged_rates_df = place_averaged_rates_df.reindex(sorted(place_averaged_rates_df.index))
    return place_averaged_rates_df.T


# %% 2D Space


def plot_session_spatial_tuning(session, navigation_only=False, moving_only=False, exclude_time_at_goal=False):
    """ """
    # load data
    simple_maze = session.simple_maze()
    goals = session.goals
    navigation_activity_df = session.get_navigation_activity_df(type="spikes", cluster_kwargs={"single_units": True})
    navigation_activity_df = filt.filter_navigation_rates_df(
        navigation_activity_df, navigation_only, moving_only, exclude_time_at_goal
    )
    pos = navigation_activity_df.centroid_position.to_numpy()
    cluster_unique_ID = navigation_activity_df.spike_count.columns.to_numpy()
    for cluster in cluster_unique_ID[:10]:
        spikes = navigation_activity_df.xs(cluster, level=1, axis=1).to_numpy().reshape(-1)
        plot_spatial_heatmap(simple_maze, pos, spikes, goals=goals)
    return


def plot_spatial_heatmap(
    pos, spike_counts, simple_maze, goals=False, bin_size=0.02, smooth_SD=0.04, maze_silhouette=True, cbar=True, ax=None
):
    """Plots a spatial heatmap of neural firnig rate in 2D space"""
    rate_map, binx, biny = get_2D_ratemap(spike_counts, pos, x_size=bin_size, y_size=bin_size, smooth_SD=smooth_SD)
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_axis_off()
    if maze_silhouette:
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
    if cbar:
        cbar = plt.colorbar(ax.images[0], ax=ax)
        cbar.set_label("Firing Rate (Hz)")
        # remove boarder from colorbar
        cbar.outline.set_visible(False)
    return


def get_2D_ratemap(
    spikes,
    pos,
    x_size=0.02,
    y_size=0.02,
    smooth_SD=0.04,
    x_range=None,
    y_range=None,
    nan_unvisited=True,
    min_occupancy=0,
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
    if x_range is None:
        x_min, x_max = np.min(x), np.max(x)
    else:
        x_min, x_max = x_range
    if y_range is None:
        y_min, y_max = np.min(y), np.max(y)
    else:
        y_min, y_max = y_range

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
        # smoothing needs to be corrected for smoothing over unvisited bins:
        weights = h.copy()
        weights[occupancy != 0] = 1  ## make constant in unvisited bins
        # Convert smoothing window (SD) from meters to bin units
        sigma_x = smooth_SD / x_size
        sigma_y = smooth_SD / y_size
        h = gaussian_filter(h, sigma=[sigma_x, sigma_y])
        weights = gaussian_filter(weights, sigma=[sigma_x, sigma_y])
        valid = weights > 0
        h[valid] = h[valid] / weights[valid]
        h[~valid] = np.nan

    if nan_unvisited:
        # Set bins to np.nan if they were not visited
        if min_occupancy is not None:
            h[occupancy <= min_occupancy] = np.nan
        else:
            h[occupancy == 0] = np.nan

    # Transpose to change row-column coordinates to positions
    return h.T, binx, biny
