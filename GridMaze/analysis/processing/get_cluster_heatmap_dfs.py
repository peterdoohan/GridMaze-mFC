"""This script is for generating dataframes with cluster_unique_IDs as rows and then either place,
place-direction as the columns, with average firing rates during navigation (moving) as the values
in the df. These will then be saved the analysis data folders"""

# %% Imports
import numpy as np
import pandas as pd

from ..core import load_data
from ..core import filter

from ...maze import representations as mr

# %% Global variables
FRAME_RATE = 60

# %% Functions


def get_place_df(
    processed_data_path,
    analysis_data_path,
    minimum_occupancy=1,
    navigation_only=True,
    moving_only=True,
    exclude_time_at_goal=True,
):
    """ """
    # load data
    try:
        session_info = load_data.load(processed_data_path / "session_info.json")
        navigation_df = load_data.load(analysis_data_path / "frames.navigation.parquet")
        navigation_spike_rates_df = load_data.load(analysis_data_path / "frames.spikeRates.parquet")
    except FileNotFoundError:
        print("Missing requisit processed/analysis data to run get_place_df. Returning None")
        return None
    navigation_rates_df = pd.concat((navigation_df, navigation_spike_rates_df.reset_index(drop=True)), axis=1)
    simple_maze = mr.simple_maze(session_info["maze_structure"])
    place_rates_df = _get_place_df(
        simple_maze,
        navigation_rates_df,
        navigation_only,
        moving_only,
        exclude_time_at_goal,
        minimum_occupancy,
    )
    return place_rates_df


def _get_place_df(
    simple_maze,
    navigation_rates_df,
    navigation_only=True,
    moving_only=True,
    exclude_time_at_goal=True,
    minimum_occupancy=1,
):
    """ """
    navigation_rates_df = filter.filter_navigation_rates_df(
        navigation_rates_df, navigation_only, moving_only, exclude_time_at_goal
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


def get_place_direction_df(
    processed_data_path,
    analysis_data_path,
    minimum_occupancy=0.5,  # seconds
    navigation_only=True,
    moving_only=True,
    exclude_time_at_goal=True,
):
    """ """
    # load_data
    try:
        session_info = load_data.load(processed_data_path / "session_info.json")
        navigation_df = load_data.load(analysis_data_path / "frames.navigation.parquet")
        navigation_spike_rates_df = load_data.load(analysis_data_path / "frames.spikeRates.parquet")
    except FileNotFoundError:
        print("Missing requisit processed/analysis data to run get_place_direction_df. Returning None")
        return None
    navigation_rates_df = pd.concat((navigation_df, navigation_spike_rates_df.reset_index(drop=True)), axis=1)
    simple_maze = mr.simple_maze(session_info["maze_structure"])
    place_direction_df = _get_place_direction_df(
        simple_maze,
        navigation_rates_df,
        navigation_only,
        moving_only,
        exclude_time_at_goal,
        minimum_occupancy,
    )
    return place_direction_df


def _get_place_direction_df(
    simple_maze,
    navigation_rates_df,
    navigation_only,
    moving_only,
    exclude_time_at_goal,
    minimum_occupancy,
    max_steps_from_goal=30,
):
    navigation_rates_df = filter.filter_navigation_rates_df(
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


# %%
