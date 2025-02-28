"""This script if for populating permuted datasets"""
# %% Imports
import os
import math
import numpy as np
import pandas as pd
from .. import get_sessions as gs
from ...maze import representations as mr
from ..processing.filter_navigation_rates_df import get_filtered_navigation_rates_df_from_session

# %% Global variables
FRAME_RATE = 60
SAVE_PATH = "../data/permuted_data/"


# %% Saving functions
def save_permuted_datasets(fn, foldername, maze_number, n_permuted):
    sessions = get_analysis_sessions(maze_number, subject="all", late=True)
    total_permutations = 0  # init
    n_batches = math.ceil(n_permuted / 100)

    for batch in range(n_batches):
        batch_size = 100 if (n_permuted - total_permutations) >= 100 else (n_permuted - total_permutations)
        permuted_neural_place_direction_dfs = get_multisession_permuted_dfs(sessions, fn, n_permuted=batch_size)
        for i, df in enumerate(permuted_neural_place_direction_dfs):
            p = total_permutations + i
            filename = f"p{p}_maze{maze_number}.parquet"
            save_path = os.path.join(SAVE_PATH, foldername, filename)
            df.to_parquet(save_path, compression="gzip")

        total_permutations += batch_size


def get_analysis_sessions(maze, subject, late=True):
    subject = [subject] if not subject == "all" else subject
    days_on_maze = "late" if late == True else "all"
    sessions = gs.get_sessions(
        subject_IDs=subject,
        maze_number=[maze],
        day_on_maze=days_on_maze,
        with_data=["navigation_df", "navigation_spike_rates_df", "cluster_metrics"],
    )
    return sessions


def get_multisession_permuted_dfs(
    sessions,
    fn,
    n_permuted=False,
):
    """If n_permuted is false, the unpermuted df is returned. Input function is either get_permuted_cluster_place_dfs
    or get_permuted_cluster_place_direction_dfs"""
    if not n_permuted:
        multisession_df = pd.concat(  # combine session place direction dfs
            [fn(session, n_permuted=False) for session in sessions],
            axis=0,
        )
        return multisession_df  # df
    else:
        permuted_multisession_df_lists = [fn(session, n_permuted=n_permuted) for session in sessions]
        permuted_mutlisession_dfs = [  # concatenate the nth permutation of each session
            pd.concat([inner_list[n] for inner_list in permuted_multisession_df_lists], axis=0)
            for n in range(n_permuted)
        ]
        return permuted_mutlisession_dfs  # list of dfs


# %% Place heatmaps


def get_permuted_cluster_place_dfs(
    session,
    n_permuted=False,
):
    navigation_rates_df = get_filtered_navigation_rates_df_from_session(session)
    simple_maze = session.simple_maze()
    if not n_permuted:
        place_averaged_rates_df = _get_cluster_place_df(navigation_rates_df, simple_maze)
        return place_averaged_rates_df
    else:
        permuted_cluster_place_dfs = []
        for _ in range(n_permuted):
            n_shift = np.random.randint(0, len(navigation_rates_df) - 1)
            column_to_shift = [("maze_position", "simple")]
            navigation_rates_df.loc[:, column_to_shift] = navigation_rates_df.loc[:, column_to_shift].apply(
                lambda x: np.roll(x, n_shift)
            )
            cluser_place_df = _get_cluster_place_df(navigation_rates_df, simple_maze)
            permuted_cluster_place_dfs.append(cluser_place_df)
        return permuted_cluster_place_dfs


def _get_cluster_place_df(navigation_rates_df, simple_maze, minimum_occupancy=1):
    place_direction_grouped_df = navigation_rates_df.set_index([("maze_position", "simple")]).groupby(
        [("maze_position", "simple")]
    )
    cluster_unique_IDs = navigation_rates_df.firing_rate.columns.to_numpy()
    place_averaged_rates_df = place_direction_grouped_df.mean().firing_rate
    place_averaged_rates_df[place_direction_grouped_df.count().time < minimum_occupancy * FRAME_RATE] = np.nan
    all_places = mr.get_maze_locations(simple_maze)
    unvisited_places = list(set(all_places) - set(place_averaged_rates_df.index))
    place_averaged_rates_df = pd.concat(
        [place_averaged_rates_df, pd.DataFrame(index=unvisited_places, columns=cluster_unique_IDs, data=np.nan)]
    )
    place_averaged_rates_df = place_averaged_rates_df.reindex(sorted(place_averaged_rates_df.index))
    return place_averaged_rates_df.T


# %% Place-direction heatmaps


def get_permuted_cluster_place_direction_dfs(
    session,
    n_permuted=False,
):
    """If n_permuted is False, returns the true place-direction df for comparison to permuted versions.
    No normalisation is applied to the permuted place-direction dfs to be saved out"""
    navigation_rates_df = get_filtered_navigation_rates_df_from_session(session)
    simple_maze = session.simple_maze()
    if not n_permuted:
        place_direction_df = _get_cluster_place_direction_df(navigation_rates_df, simple_maze)
        return place_direction_df
    else:
        permuted_place_direction_dfs = []
        for _ in range(n_permuted):
            n_shift = np.random.randint(0, len(navigation_rates_df) - 1)
            columns_to_shift = [("maze_position", "simple"), ("cardinal_movement_direction", "")]
            navigation_rates_df.loc[:, columns_to_shift] = navigation_rates_df.loc[:, columns_to_shift].apply(
                lambda x: np.roll(x, n_shift)
            )
            place_direction_df = _get_cluster_place_direction_df(navigation_rates_df, simple_maze)
            permuted_place_direction_dfs.append(place_direction_df)
        return permuted_place_direction_dfs


def _get_cluster_place_direction_df(
    navigation_rates_df,
    simple_maze,
    minimum_occupancy=0.5,  # seconds
):
    """ """
    cluster_unique_IDs = navigation_rates_df.firing_rate.columns.to_numpy()
    place_direction_cols = [("maze_position", "simple"), ("cardinal_movement_direction", "")]
    place_direction_grouped_df = navigation_rates_df.set_index(place_direction_cols).groupby(place_direction_cols)
    place_direction_av_rates_df = place_direction_grouped_df.mean().firing_rate
    # set low occupancy cdirs to nan for all clusters
    place_direction_av_rates_df[place_direction_grouped_df.count().time < minimum_occupancy * FRAME_RATE] = np.nan
    # add nan values for unvisited locations (for all clusters)
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


# %% Supporting functions


# def get_filtered_navigation_rates_df(
#     session,
#     minimum_firing_rate=0.25,
#     moving_navigation_only=True,
#     exclude_time_at_goal=True,
# ):
#     data_needed = ["navigation_df", "navigation_spike_rates_df", "cluster_metrics"]
#     if not gs.check_session_has_data(session, data_needed):
#         pass

#     navigation_rates_df = session.get_navigation_activity_df(activity_type="firing_rate", cluster_type="good")

#     if minimum_firing_rate:
#         invalid_clusters = navigation_rates_df.firing_rate.mean(axis=0).lt(minimum_firing_rate)
#         navigation_rates_df.drop(columns=invalid_clusters[invalid_clusters].index, level=1, inplace=True)

#     if moving_navigation_only or exclude_time_at_goal:
#         conditions = []
#         if moving_navigation_only:
#             conditions.append(
#                 np.logical_and(navigation_rates_df.trial_phase == "navigation", navigation_rates_df.moving)
#             )
#         if exclude_time_at_goal:
#             conditions.append(navigation_rates_df.goal != navigation_rates_df.maze_position.simple)
#         combined_condition = np.logical_and.reduce(conditions)
#         navigation_rates_df = navigation_rates_df[combined_condition]
#     return navigation_rates_df[:-1]  # exclude last row incase action is not defined
