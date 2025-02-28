"""This script is to calculate the median occumpacy of each maze location across session"""
# %% Imports
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from ..core import get_sessions as gs
from collections import Counter


from ...maze import representations as mr
from ...maze import plotting as mp

# %% Global variables


# %% Functions


def get_mean_occupancy(
    maze_name,
    occupancy_type="place",
    late_sessions=True,
    navigation_only=True,
    moving_only=True,
    exclude_time_at_goal=True,
    plot=False,
):
    grouping_cols = (
        [("maze_position", "simple")]
        if occupancy_type == "place"
        else [("maze_position", "simple"), ("cardinal_movement_direction", "")]
    )
    days_on_maze = "late" if late_sessions else "all"
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names=[maze_name],
        days_on_maze=days_on_maze,
        with_data=["navigation_df"],
    )

    occupancy_counts = []
    for session in sessions:
        navigation_df = session.navigation_df
        navigation_df = _filter_navigation_df(navigation_df, navigation_only, moving_only, exclude_time_at_goal)
        grouped_nav_df = navigation_df.set_index(grouping_cols).groupby(grouping_cols)
        nav_df_occupancy_counts = grouped_nav_df.count().time
        simple_maze = mr.get_simple_maze(maze_name)
        all_locations = (
            mr.get_maze_locations(simple_maze)
            if occupancy_type == "place"
            else mr.get_maze_place_direction_pairs(simple_maze)
        )
        unvistied_locations = list(set(all_locations) - set(nav_df_occupancy_counts.index.to_numpy()))
        if len(unvistied_locations) > 0:
            unvisited_locations_df = pd.Series(index=unvistied_locations, name="time", data=0)
            nav_df_occupancy_counts = pd.concat([nav_df_occupancy_counts, unvisited_locations_df], axis=0)
            nav_df_occupancy_counts = nav_df_occupancy_counts.reindex(sorted(nav_df_occupancy_counts.index))
        occupancy_counts.append(nav_df_occupancy_counts)
    maze_occupancy_counts = pd.concat(occupancy_counts, axis=1).sum(axis=1)  # normalise
    maze_occupancy_counts = maze_occupancy_counts.div(maze_occupancy_counts.sum())
    if plot:
        f, ax = plt.subplots(1, 1, figsize=(5, 5), clear=True)
        mp.plot_simple_heatmap(
            simple_maze,
            maze_occupancy_counts.to_dict(),
            ax=ax,
            title="Mean Place Occupancy",
            value_label="Occupancy (%)",
            colormap="PuBuGn",
        )
    return maze_occupancy_counts.to_dict()


# %% Supporinng functions


def _filter_navigation_df(navigation_df, navigation_only, moving_only, exclude_time_at_goal):
    """ """
    if navigation_only or moving_only or exclude_time_at_goal:
        conditions = []
        if navigation_only:
            conditions.append(navigation_df.trial_phase == "navigation")
        if moving_only:
            conditions.append(navigation_df.moving)
        if exclude_time_at_goal:
            conditions.append(navigation_df.goal != navigation_df.maze_position.simple)
        frames_mask = np.logical_and.reduce(conditions)
        navigation_df = navigation_df[frames_mask]
    return navigation_df


# %%


def get_mean_transitions(
    maze_name,
    late_sessions=True,
    navigation_only=True,
    plot=False,
):
    days_on_maze = "late" if late_sessions else "all"
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names=[maze_name],
        days_on_maze=days_on_maze,
        with_data=["trajectory_decisions_df"],
    )
    simple_maze = mr.get_simple_maze(maze_name)
    extended_simple_maze = mr.get_extended_simple_maze(simple_maze)
    transition_count_dfs = []
    for session in sessions:
        trajectory_decisions_df = session.trajectory_decisions_df
        if navigation_only:
            trajectory_decisions_df = trajectory_decisions_df[trajectory_decisions_df.trial_phase == "navigation"]
        trials = trajectory_decisions_df.trial.unique()
        all_edge_transitions = []
        for trial in trials:
            trial_df = trajectory_decisions_df[trajectory_decisions_df.trial == trial]
            transitions = trial_df.maze_position
            edge_transitions = list(zip(transitions, transitions.shift(1)))[1:]
            edge_transitions = [(a, b) if len(a) <= len(b) else (b, a) for a, b in edge_transitions]
            all_edge_transitions.extend(edge_transitions)
        edge2transition_count = pd.Series(Counter(all_edge_transitions), name="count")
        all_edges = list(nx.get_edge_attributes(extended_simple_maze, "label").values())
        unvisited_edges = list(set(all_edges) - set(edge2transition_count.index.to_numpy()))
        if len(unvisited_edges) > 0:
            unvisited_edges_df = pd.Series(index=unvisited_edges, name="count", data=0)
            edge2transition_count = pd.concat([edge2transition_count, unvisited_edges_df], axis=0)
            edge2transition_count = edge2transition_count.reindex(sorted(edge2transition_count.index))
        transition_count_dfs.append(edge2transition_count)
    maze_transition_counts = pd.DataFrame(transition_count_dfs).T.sum(axis=1)
    maze_transition_counts = maze_transition_counts.div(maze_transition_counts.max())  # normalise to sum to 1
    maze_transition_counts = maze_transition_counts.to_dict()
    if plot:
        f, ax = plt.subplots(1, 1, figsize=(5, 5), clear=True)
        plot_weighted_transitions_maze(maze_transition_counts, extended_simple_maze, ax)
    return maze_transition_counts


def plot_weighted_transitions_maze(edge2transition_value, extended_simple_maze, ax):
    """"""
    positions = nx.get_node_attributes(extended_simple_maze, "position")
    edge_label2edge_coord = {v: k for k, v in nx.get_edge_attributes(extended_simple_maze, "label").items()}
    edge2transition_value = {edge_label2edge_coord[k]: v for k, v in edge2transition_value.items()}
    nx.draw_networkx(
        extended_simple_maze,
        with_labels=False,
        pos=positions,
        ax=ax,
        node_size=50,
        node_color="silver",
        edge_color="tan",
        edgelist=list(edge2transition_value.keys()),
        width=5 * np.array(list(edge2transition_value.values())),
    )
    ax.set_facecolor("none")
    ax.set_aspect("equal")
    ax.axis("off")
    return
