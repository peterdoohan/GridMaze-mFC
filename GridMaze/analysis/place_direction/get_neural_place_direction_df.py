"""This module generates vector representations of single neuron place direction tuning heatmaps and combes these vectors
across neurons and sessions to make place-direction neural data matrices that feed into other analyses."""

# %% Imports
import os
import math
import numpy as np
import pandas as pd
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.cluster_tuning import spatial
from GridMaze.maze import representations as mr


# %% Global variables
FRAME_RATE = 60
SAVE_PATH = "../results/place_direction_permutations/new"

# %% Function for saving out permuted neural place_direction_dfs


def save_permuted_neural_place_direction_dfs(maze_number, n_permuted):
    sessions = get_analysis_sessions(maze_number, subject="all", late=True)
    permuted_neural_place_direction_dfs = get_multisession_neural_place_direction_df(
        sessions, normalisation_method="length", n_permuted=n_permuted
    )
    for p, df in enumerate(permuted_neural_place_direction_dfs):
        filename = f"p{p}_maze{maze_number}_permuted_neural_place_direction_df.parquet"
        save_path = os.path.join(SAVE_PATH, filename)
        df.to_parquet(save_path, compression="gzip")
    return


def save_permuted_neural_place_direction_dfs2(maze_number, n_permuted):
    sessions = get_analysis_sessions(maze_number, subject="all", late=True)
    total_permutations = 0  # init
    n_batches = math.ceil(n_permuted / 100)

    for batch in range(n_batches):
        batch_size = 100 if (n_permuted - total_permutations) >= 100 else (n_permuted - total_permutations)
        permuted_neural_place_direction_dfs = get_multisession_neural_place_direction_df(
            sessions, normalisation_method="length", n_permuted=batch_size
        )
        for i, df in enumerate(permuted_neural_place_direction_dfs):
            p = total_permutations + i
            filename = f"p{p}_maze{maze_number}_permuted_neural_place_direction_df.parquet"
            save_path = os.path.join(SAVE_PATH, filename)
            df.to_parquet(save_path, compression="gzip")

        total_permutations += batch_size


def load_permuted_neural_place_direction_dfs(maze_number, n_permuted=100):
    filenames = os.listdir(SAVE_PATH)
    filenames = [
        f for f in filenames if int(f.split("_")[1][-1]) == maze_number
    ]  # use filenames to select data from relevant maze
    if len(filenames) < n_permuted:
        raise ValueError(f"Only {len(filenames)} files found, but {n_permuted} requested")
    elif n_permuted == "all":
        n_permuted = len(filenames)
    filenames = filenames[:n_permuted]
    permuted_neural_place_direction_dfs = [pd.read_parquet(os.path.join(SAVE_PATH, f)) for f in filenames]
    return permuted_neural_place_direction_dfs


# %% Functions


def get_multisession_neural_place_direction_df(
    sessions,
    n_permuted=False,
    normalisation_method="length",
):
    if not n_permuted:
        maze_place_direction_df = pd.concat(  # combine session place direction dfs
            [get_place_direction_df(session, normalisation_method=normalisation_method) for session in sessions],
            axis=0,
        )
        return maze_place_direction_df  # df
    else:
        permuted_maze_place_direction_lists = [
            get_place_direction_df(session, normalisation_method=normalisation_method, n_permuted=n_permuted)
            for session in sessions
        ]
        permuted_maze_place_direction_dfs = [  # concatenate the nth permutation of each session
            pd.concat([inner_list[n] for inner_list in permuted_maze_place_direction_lists], axis=0)
            for n in range(n_permuted)
        ]
        return permuted_maze_place_direction_dfs  # list of dfs


def get_place_direction_df(
    session,
    n_permuted=False,
    normalisation_method="length",
):
    navigation_rates_df = get_filtered_navigation_rates_df(session, navigation_tuned_only=True)
    simple_maze = session.simple_maze()
    if not n_permuted:
        place_direction_df = _get_place_direction_df(
            navigation_rates_df, simple_maze, normalisation_method=normalisation_method
        )
        return place_direction_df
    else:
        permuted_place_direction_dfs = []
        for _ in range(n_permuted):
            n_shift = np.random.randint(0, len(navigation_rates_df) - 1)
            columns_to_shift = [("maze_position", "simple"), ("cardinal_movement_direction", "")]
            navigation_rates_df.loc[:, columns_to_shift] = navigation_rates_df.loc[:, columns_to_shift].apply(
                lambda x: np.roll(x, n_shift)
            )
            place_direction_df = _get_place_direction_df(
                navigation_rates_df, simple_maze, normalisation_method=normalisation_method
            )
            permuted_place_direction_dfs.append(place_direction_df)
        return permuted_place_direction_dfs


def get_filtered_navigation_rates_df(
    session,
    minimum_firing_rate=0.25,
    navigation_tuned_only=True,
    moving_navigation_only=True,
    exclude_time_at_goal=True,
):
    data_needed = ["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "cluster_analysis_metrics_df"]
    if not gs.check_session_has_data(session, data_needed):
        pass  # consider raising an exception or returning early
    navigation_rates_df = session.get_navigation_activity_df(activity_type="firing_rate", cluster_type="good")
    cluster_analysis_metrics_df = session.cluster_analysis_metrics_df
    cluster_analysis_metrics_df = cluster_analysis_metrics_df[cluster_analysis_metrics_df.KS_label == "good"]
    drop_clusters = np.array([])
    if minimum_firing_rate:
        drop_clusters = np.append(
            drop_clusters,
            cluster_analysis_metrics_df[
                cluster_analysis_metrics_df.average_firing_rate < minimum_firing_rate
            ].index.to_numpy(),
        )
    if navigation_tuned_only:
        drop_clusters = np.append(
            drop_clusters,
            cluster_analysis_metrics_df[~cluster_analysis_metrics_df.trial_phase_tuning.navigation].index.to_numpy(),
        )
    all_clusters = navigation_rates_df.firing_rate.columns.to_numpy()
    drop_clusters = np.intersect1d(all_clusters, drop_clusters)
    if len(drop_clusters) == len(all_clusters):
        return None  # no valid clusters
    navigation_rates_df.drop(columns=drop_clusters, level=1, inplace=True)
    if moving_navigation_only or exclude_time_at_goal:
        conditions = []
        if moving_navigation_only:
            conditions.append(
                np.logical_and(navigation_rates_df.trial_phase == "navigation", navigation_rates_df.moving)
            )
        if exclude_time_at_goal:
            conditions.append(navigation_rates_df.goal != navigation_rates_df.maze_position.simple)
        combined_condition = np.logical_and.reduce(conditions)
        navigation_rates_df = navigation_rates_df[combined_condition]
    return navigation_rates_df[:-1]  # exclude last row incase action is not defined


def _get_place_direction_df(
    navigation_rates_df,
    simple_maze,
    minimum_occupancy=1,  # seconds
    exclude_low_occupancy_place_directions=True,
    normalise_single_place_directions=False,
    fill_nans="mean",
    normalisation_method="length",
):
    """ """
    if navigation_rates_df is None:  # no valid clusters
        return None
    cluster_unique_IDs = navigation_rates_df.firing_rate.columns.to_numpy()
    place_direction_cols = [("maze_position", "simple"), ("cardinal_movement_direction", "")]
    place_direction_grouped_df = navigation_rates_df.set_index(place_direction_cols).groupby(place_direction_cols)
    place_direction_av_rates_df = place_direction_grouped_df.mean().firing_rate
    if exclude_low_occupancy_place_directions:  # set low occupancy cdirs to nan for all clusters
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
    if normalise_single_place_directions:  # normalise firing rates per location to sum to 1
        for name, group in place_direction_av_rates_df.groupby(level=0):
            if group.sum().sum() != 0:  # if firing rate is 0 at a location, set all cdirs to 0
                place_direction_av_rates_df.loc[name] = np.nan_to_num(group / group.sum(), nan=0)
            else:  # if all NaNs, keep as such
                place_direction_av_rates_df.loc[name] = group
    place_direction_df = place_direction_av_rates_df.T  # [cluster_unique_IDs, location_cdirs]
    if fill_nans:
        if fill_nans == "mean":
            place_direction_df.T.fillna(place_direction_df.mean(axis=1), inplace=True)  # replace nans with the mean
        elif fill_nans == "zero":
            place_direction_df.fillna(0, inplace=True)
    if normalisation_method is not None:
        normalisation_methods = {
            "mean": place_direction_df.mean(axis=1),
            "length": place_direction_df.pow(2).sum(axis=1).pow(0.5),
            "max": place_direction_df.max(axis=1),
        }
        normaliser = normalisation_methods.get(normalisation_method)
        if normaliser is not None:
            place_direction_df = place_direction_df.div(normaliser, axis=0)
    place_direction_df.columns.names = ["maze_position", "direction"]
    place_direction_df.sort_index(axis=1, inplace=True)
    return place_direction_df


# %% Supporting functions


def get_analysis_sessions(maze, subject, late=True):
    subject = [subject] if not subject == "all" else subject
    days_on_maze = "late" if late == True else "all"
    sessions = gs.get_sessions(
        subject_IDs=subject,
        maze_number=[maze],
        day_on_maze=days_on_maze,
        with_data=[
            "navigation_df",
            "navigation_spike_counts_df",
            "navigation_spike_rates_df",
            "cluster_metrics",
            "cluster_analysis_metrics_df",
        ],
    )
    return sessions
