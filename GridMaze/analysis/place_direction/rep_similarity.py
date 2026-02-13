"""
Characterise how representational similarity in place tuning changes as a function
of distance to distance and choice-points between locations
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import networkx as nx

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.maze import representations as mr
from GridMaze.analysis.cluster_tuning import spatial
from GridMaze.analysis.place_direction import dimensionality_reduction as dr

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

# %% Functions


def test(maze_names=["maze_1", "maze_2"], verbose=True):
    dfs = []
    for maze_name in maze_names:
        pairs_df = get_pairs_df(maze_name)
        for subject_ID in SUBJECT_IDS:
            if verbose:
                print(subject_ID)
            heatmap_df = get_population_place_tuning(
                subject_IDs=[subject_ID],
                maze_name=maze_name,
                late_sessions=True,
            )
            res_df = pairs_df.copy()
            res_df["rep_sim"] = np.nan
            res_df["subject_ID"] = subject_ID
            res_df["maze_name"] = maze_name
            for loc_1, loc_2 in pairs_df.index:
                res_df.loc[(loc_1, loc_2), "rep_sim"] = heatmap_df[loc_1].corr(heatmap_df[loc_2])
            dfs.append(res_df.reset_index())
    return pd.concat(dfs, axis=0)


def get_pairs_df(maze_name, edges_only=True):
    """ """
    simple_maze = mr.get_simple_maze(maze_name)
    coord2label = mr.get_maze_coord2label(simple_maze)
    extended_maze = mr.get_extended_simple_maze(simple_maze)
    all_pairs_paths = dict(nx.all_pairs_shortest_path(extended_maze))
    res = []
    for start_coord, paths in all_pairs_paths.items():
        start_label = coord2label[start_coord]
        if edges_only and "-" not in start_label:
            continue
        for end_coord, path in paths.items():
            end_label = coord2label[end_coord]
            if edges_only and "-" not in end_label:
                continue
            if len(path) == 1:
                continue
            if isinstance(path[0], list):
                path_choices = np.array([_get_path_n_choices(p, extended_maze) for p in path])
                n_choice = path_choices[:, 0].mean()
                n_deg_3 = path_choices[:, 1].mean()
                n_deg_4 = path_choices[:, 2].mean()
                path_length = len(path[0]) - 1
            else:
                n_choice, n_deg_3, n_deg_4 = _get_path_n_choices(path, extended_maze)
                path_length = len(path) - 1
            res.append(
                {
                    "loc_1": start_label,
                    "loc_2": end_label,
                    "path_length": path_length,
                    "n_choice": n_choice,
                    "n_deg_3": n_deg_3,
                    "n_deg_4": n_deg_4,
                }
            )
    df = pd.DataFrame(res)
    df["path_length"] = df["path_length"].div(2).astype(int)
    return df.set_index(["loc_1", "loc_2"])


def _get_path_n_choices(path, extended_maze):
    """ """
    n_choice = 0
    n_deg_3 = 0
    n_deg_4 = 0
    for coord in path:
        if extended_maze.degree(coord) > 2:
            n_choice += 1
            if extended_maze.degree(coord) == 3:
                n_deg_3 += 1
            elif extended_maze.degree(coord) == 4:
                n_deg_4 += 1
    return n_choice, n_deg_3, n_deg_4


# %%


def get_population_place_tuning(
    subject_IDs="all",
    maze_name="maze_1",
    late_sessions=True,
    sessions=None,
    include_multi_unit=False,
    fill_nans=False,
    normalisation="length",
    max_steps_to_goal=30,
    min_split_corr=0.5,
    verbose=False,
):
    """ """
    # if session objects are not input, generate them from input filters
    if sessions is None:
        days_on_maze = "late" if late_sessions else "all"
        if verbose:
            print("Loading sessions ...")
        sessions = gs.get_maze_sessions(
            subject_IDs=subject_IDs,
            maze_names=[maze_name],
            days_on_maze=days_on_maze,
            with_data=[
                "navigation_df",
                "navigation_spike_rates_df",
                "cluster_metrics",
                "cluster_place_direction_tuning_metrics",
            ],
            must_have_data=True,
        )
    dfs = []
    for session in sessions:
        if verbose:
            print(session.name)
        df = get_session_place_tuning(
            session,
            include_multi_unit=include_multi_unit,
            fill_nans=fill_nans,
            normalisation=normalisation,
            max_steps_from_goal=max_steps_to_goal,
            min_split_corr=min_split_corr,
            verbose=verbose,
        )
        if df is None:
            continue
        dfs.append(df)
    pop_tuning_df = pd.concat(dfs, axis=0)
    return pop_tuning_df


def get_session_place_tuning(
    session,
    navigation_rates_df=None,
    min_split_corr=0.5,
    include_multi_unit=False,
    fill_nans="mean",
    normalisation="length",
    navigation_only=True,
    moving_only=True,
    exclude_time_at_goal=True,
    minimum_occupancy=1.0,
    max_steps_from_goal=30,
    verbose=False,
):
    """
    Returns place-direction tuning for all place-direction tuned clusters in a session.
    w/ options for filtering clusters going in, data going into heatmap calculation, then
    further value filling and normalisation of the heatmaps
    """
    # load data
    simple_maze = session.simple_maze()
    navigation_rates_df = session.get_navigation_activity_df(
        type="rates", cluster_kwargs={"single_units": True, "multi_units": include_multi_unit}
    )
    pd_tuning_metrics = session.cluster_place_direction_tuning_metrics
    # filter for place-direction tuned clusters
    cluster_filters = []
    if min_split_corr is not None:
        cluster_filters.append(pd_tuning_metrics.split_half_corr.value.ge(min_split_corr))
    if len(cluster_filters) > 0:
        keep_clusters = pd_tuning_metrics[np.logical_and.reduce(cluster_filters)].index
        if len(keep_clusters) == 0:
            if verbose:
                print(f"No place-direction cluster found with split_half_corr >= {min_split_corr}")
            return None
        reject_clusters = [c for c in navigation_rates_df.firing_rate.columns.values if c not in keep_clusters]
        navigation_rates_df = navigation_rates_df.drop([("firing_rate", c) for c in reject_clusters], axis=1)
    # get average place direction tuning
    place_df = spatial._get_place_df(
        simple_maze,
        navigation_rates_df,
        navigation_only,
        moving_only,
        exclude_time_at_goal,
        minimum_occupancy,
        max_steps_from_goal,
    )
    # fill nan values of unvisited place-directions
    if fill_nans:
        if fill_nans == "mean":
            place_df.T.fillna(place_df.mean(axis=1), inplace=True)  # replace nans with the mean
        elif fill_nans == "zero":
            place_df.fillna(0, inplace=True)
        else:
            raise ValueError(f"Unknown fill_nans method: {fill_nans}")
    # normalise over clusters
    if normalisation:
        if normalisation == "mean":
            place_df = place_df.div(place_df.mean(axis=1), axis=0)
        elif normalisation == "length":
            place_df = place_df.div(place_df.pow(2).sum(axis=1).pow(0.5), axis=0)
        elif normalisation == "max":
            place_df = place_df.div(place_df.max(axis=1), axis=0)
        else:
            raise ValueError(f"Unknown normalisation method: {normalisation}")
    # return df
    place_df.sort_index(axis=1, inplace=True)
    return place_df
