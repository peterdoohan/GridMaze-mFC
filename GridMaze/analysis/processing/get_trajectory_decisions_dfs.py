"""This module creates a dataframe the tracks navigation decisions through the entire trajectory of a session"""

# %% Imports
import json
import random
import numpy as np
import pandas as pd
import networkx as nx
from ..core import load_data
from ..core import convert

from ...maze import representations as mr
from ...preprocessing import get_maze_trajectories as gmt
from .get_navigation_df import get_cardinal_movement_direction, get_basic_actions

# %% Global variables
from ...paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "maze_configs.json", "r") as input_file:
    MAZE_CONFIGS = json.load(input_file)

# %% Main Functions


def get_trajectory_decisions_df(
    processed_data_path,
    analysis_dat_path,
    with_edges=True,
    correct_backtracking=True,
    run_trajectory_qc=True,
    remove_egocentric_actions_at_goal=True,
):
    """
    Creates a dataframe that tracks navigation decisions through the entire trajectory of a session.

    Parameters:
    -----------
    subject_session_path : str
        The path to the session processed data from instead the processed data folder eg, "m2/2021-01-01_12-00-00".
    with_edges : bool, optional
        Whether to include edges between adjacent node visits in the trajectory decisions dataframe. Default is True.
    correct_backtracking : bool, optional
        Whether to correct backtracking in the trajectory decisions dataframe. Default is True.
    run_trajectory_qc : bool, optional
        Whether to run trajectory quality control on the trajectory decisions dataframe. Default is True.

    Returns:
    --------
    trajectory_decisions_df : pandas.DataFrame
        The trajectory decisions dataframe, with columns for subject ID, maze number, day on maze, time, trial, trial phase,
        goal, maze position, and action.
    """
    # load relevant processed data
    try:
        session_info = load_data.load(processed_data_path / "session_info.json")
        trajectories_df = load_data.load(processed_data_path / "frames.trajectories.htsv")
        frames_trial_info_df = load_data.load(processed_data_path / "frames.trialInfo.htsv")
    except FileNotFoundError:
        print("Missing requisit processed data to run get_trajectory_decisions_df. Returning None")
        return None
    edges = session_info["maze_structure"]
    simple_maze = mr.simple_maze(edges)
    skeleton_maze = mr.skeleton_maze(edges)
    # get preliminary df (with data frome every frame)
    trajectory_decisions_df = pd.concat(
        [
            trajectories_df[[("time", ""), ("maze_position", "simple")]],
            frames_trial_info_df,
        ],
        axis=1,
    )
    trajectory_decisions_df.columns = [c[0] if isinstance(c, tuple) else c for c in trajectory_decisions_df.columns]
    trajectory_decisions_df["subject_ID"] = session_info["subject_ID"]
    trajectory_decisions_df["maze_name"] = session_info["maze_name"]
    trajectory_decisions_df["day_on_maze"] = session_info["day_on_maze"]
    trajectory_decisions_df["trial_unique_ID"] = trajectory_decisions_df.apply(
        lambda row: convert.trial2trial_unique_ID(session_info, row.trial), axis=1
    )
    trajectory_decisions_df = trajectory_decisions_df.reindex(
        columns=[
            "subject_ID",
            "maze_name",
            "day_on_maze",
            "time",
            "trial",
            "trial_unique_ID",
            "trial_phase",
            "goal",
            "maze_position",
        ]
    )
    # distill trajectory decisions df to only one frame from each sequental node visit
    trajectory_decisions_df["maze_position_shifted"] = trajectory_decisions_df.maze_position.shift(1)
    trajectory_decisions_df["maze_position_change"] = (
        trajectory_decisions_df.maze_position != trajectory_decisions_df.maze_position_shifted
    )
    trajectory_decisions_df = trajectory_decisions_df[trajectory_decisions_df.maze_position_change]
    trajectory_decisions_df = trajectory_decisions_df.drop(columns=["maze_position_shifted", "maze_position_change"])
    trajectory_decisions_df.reset_index(drop=True, inplace=True)
    if not correct_backtracking:
        actions = get_trajectory_actions(trajectory_decisions_df.maze_position, simple_maze)
    else:
        # correct backtracking by first reducing traj to single nodes transitions and add back edges if required
        node_mask = trajectory_decisions_df.maze_position.apply(lambda x: len(x.split("-")) == 1)
        trajectory_decisions_df = trajectory_decisions_df[node_mask]
        node_trajectory = trajectory_decisions_df.maze_position
        trajectory_decisions_df = trajectory_decisions_df[~(node_trajectory == node_trajectory.shift(1))]
        if not with_edges:
            node_traj = trajectory_decisions_df.maze_position
            actions = get_trajectory_actions(node_traj, simple_maze)
        else:
            # artificially add edges between adjacent node visits
            trajectory_decisions_df = add_interpolated_edges_to_trajectory_decisions_df(
                trajectory_decisions_df, simple_maze
            )
            actions = get_node_edges_trajectory_actions(trajectory_decisions_df, simple_maze)
    trajectory_decisions_df.reset_index(drop=True, inplace=True)
    trajectory_decisions_df["action"] = actions
    # add dadd shortest path distance to goal column
    trajectory_decisions_df["geodesic_distance_to_goal"] = get_path_distances_to_goalf(
        trajectory_decisions_df, simple_maze, skeleton_maze
    )
    trajectory_decisions_df["steps_to_goal"] = get_n_steps_to_goal(trajectory_decisions_df)
    ego_action, choice_degree = get_basic_actions(trajectory_decisions_df.maze_position, simple_maze)
    trajectory_decisions_df["egocentric_action"] = ego_action
    trajectory_decisions_df["choice_degree"] = choice_degree
    if remove_egocentric_actions_at_goal:
        at_goal_mask = trajectory_decisions_df.maze_position == trajectory_decisions_df.goal
        trajectory_decisions_df.loc[at_goal_mask, ["egocentric_action", "choice_degree"]] = np.nan
    if run_trajectory_qc:
        if with_edges:
            assert trajectory_qc(trajectory_decisions_df.maze_position, simple_maze), "trajectory failed QC"
        else:
            assert node_trajectory_qc(trajectory_decisions_df.maze_position, simple_maze), "trajectory failed QC"
    return trajectory_decisions_df[:-1].reset_index(drop=True)  # last action not defined


def get_optimal_trajectory_decisions_df(
    maze_name,
    goal_subset="all",
    n_trials=100,
    method="random_optimal_path",
    with_edges=True,
    max_steps=None,
):
    """
    Generates a dataframe that tracks navigation decisions through a series of optimal trajectories on a maze.

    Parameters:
    -----------
    maze_name : int
        The name of the maze to use (e.g. maze_1).
    goal_subset : str or list of str, optional
        The subset of goals to use in the maze. Can "all", "subset_1" or "subset_2".
    n_trials : int, optional
        The number of trials to simulate for each goal. Default is 100.
    method : str, optional
        The method to use for generating the optimal trajectory.
        Must be one of "random_optimal_path", "random_optimal_choice", or "modified_random_walk". Default is "random_optimal_path".
    with_edges : bool, optional
        Whether to include interpolated edges in the resulting dataframe. Default is True.
    max_steps : int, optional
        The maximum number of steps to include in the resulting dataframe. Default is None (include all steps).

    Returns:
    --------
    optimal_trajectory_df : pandas.DataFrame
        A dataframe that tracks navigation decisions through an optimal trajectory in the maze.
        Structured similary to the trajectory decisions dataframe for real beahviour (with step
        instead of time column)
    """
    max_transitions = max_steps // 2 if max_steps is not None else None
    simple_maze = mr.get_simple_maze(maze_name)
    node_coord2label = nx.get_node_attributes(simple_maze, "label")
    node_label2coord = {v: k for k, v in node_coord2label.items()}
    goals = MAZE_CONFIGS[maze_name]["goal_sets"][goal_subset]
    goal_coords = [node_label2coord[goal] for goal in goals]
    start_location = random.choice(list(simple_maze.nodes()))
    goal_sampler = sample_without_replacement(goal_coords)
    if method not in [
        "random_optimal_path",
        "random_optimal_choice",
        "modified_random_walk",
    ]:
        raise ValueError("method must be either 'random_optimal_path', 'random_optimal_choice', 'modified_random_walk")
    else:
        if method == "random_optimal_path":
            optimal_trajectory_df = get_random_optimal_path_df(
                start_location,
                n_trials,
                goal_sampler,
                simple_maze,
                node_coord2label,
                max_transitions,
            )
        elif method == "random_optimal_choice":
            optimal_trajectory_df = get_random_optimal_choice_df(
                start_location,
                n_trials,
                goal_sampler,
                simple_maze,
                node_coord2label,
                max_transitions,
            )
        elif method == "modified_random_walk":
            optimal_trajectory_df = get_modified_random_walk_df(
                start_location,
                n_trials,
                goal_sampler,
                simple_maze,
                node_coord2label,
                max_transitions,
            )
    optimal_trajectory_df["step"] = optimal_trajectory_df.index + 1
    optimal_trajectory_df = optimal_trajectory_df.reindex(columns=["step"] + list(optimal_trajectory_df.columns[:-1]))
    optimal_trajectory_df["trial_unique_ID"] = "sim_" + optimal_trajectory_df.trial.astype("string")
    optimal_trajectory_df["trial_phase"] = "navigation"
    if with_edges:
        optimal_trajectory_df = add_interpolated_edges_to_trajectory_decisions_df(
            optimal_trajectory_df, simple_maze, type="optimal_behaviour"
        )
        actions = get_node_edges_trajectory_actions(optimal_trajectory_df, simple_maze)
    else:
        actions = get_trajectory_actions(optimal_trajectory_df.maze_position, simple_maze)
    optimal_trajectory_df["action"] = actions
    optimal_trajectory_df = optimal_trajectory_df[:-1]  # last action not defined
    return optimal_trajectory_df


# %% Real behaviour trajectory df supporting functions


def get_node_edges_trajectory_actions(trajectory_decisions_df, simple_maze):
    traj = trajectory_decisions_df.maze_position
    label2coord = get_maze_label2coord(simple_maze)
    traj = traj.map(label2coord).to_numpy()
    actions = [get_cardinal_movement_direction(traj[i + 1], traj[i]) for i in range(len(traj) - 1)]
    actions.append(np.nan)  # cannot define last action
    return actions


def add_interpolated_edges_to_trajectory_decisions_df(trajectory_decisions_df, simple_maze, type="subject_behaviour"):
    edges_df = trajectory_decisions_df.copy()
    edges = []
    for i in range(len(edges_df) - 1):
        label2coord = get_maze_label2coord(simple_maze)
        coord2label = {v: k for k, v in label2coord.items()}
        node1 = label2coord[edges_df.maze_position.iloc[i]]
        node2 = label2coord[edges_df.maze_position.iloc[i + 1]]
        edge = (node1, node2)
        edges.append(edge)
    edges = gmt.correct_edge_order(edges)
    edges = [coord2label[e] for e in edges]
    edges.append(np.nan)
    edges_df.maze_position = edges
    if type == "subject_behaviour":
        edges_df.time = (edges_df.time + edges_df.time.shift(-1)) / 2
        for col in ["trial", "trial_phase", "goal"]:
            edges_df[col] = edges_df[col].shift(-1)
    elif type == "optimal_behaviour":
        for col in ["trial", "goal"]:
            edges_df[col] = edges_df[col].shift(-1)
    trajectory_decisions_df.reset_index(drop=True, inplace=True)
    edges_df.reset_index(drop=True, inplace=True)
    trajectory_decisions_df = pd.concat([trajectory_decisions_df, edges_df], axis=0).sort_index(kind="merge")[:-1]
    if type == "optimal_behaviour":
        trajectory_decisions_df.reset_index(drop=True, inplace=True)
        trajectory_decisions_df.step = trajectory_decisions_df.index + 1
    return trajectory_decisions_df


def get_trajectory_actions(node_traj, simple_maze):
    label2coord = {v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()}
    node_traj = node_traj.map(label2coord).to_numpy()
    actions = []
    for i in range(len(node_traj) - 1):
        current_position = node_traj[i]
        next_position = node_traj[i + 1]
        dx = next_position[0] - current_position[0]
        dy = next_position[1] - current_position[1]
        if dx == 0:
            if dy == 1:
                action = "N"
            elif dy == -1:
                action = "S"
        else:
            if dx == 1:
                action = "E"
            elif dx == -1:
                action = "W"
        actions.append(action)
    actions.append(np.nan)
    return np.array(actions)


def node_trajectory_qc(node_traj, simple_maze):
    """ """
    label2coord = get_maze_label2coord(simple_maze)
    nodes = node_traj.map(label2coord).to_numpy()
    for i in range(len(nodes) - 1):
        if nodes[i + 1] not in simple_maze[nodes[i]]:
            return False
    return True


def trajectory_qc(traj, simple_maze):
    """ """
    label2coord = get_maze_label2coord(simple_maze)
    traj_coords = traj.map(label2coord)
    return gmt.trajectory_qc(traj_coords, simple_maze)


def get_maze_label2coord(simple_maze):
    label2coord = {
        **{v: k for k, v in nx.get_edge_attributes(simple_maze, "label").items()},
        **{v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()},
    }
    return label2coord


# %% Artifical trajectories supporting functions


def get_random_optimal_path_df(
    start_location,
    n_trials,
    goal_sampler,
    simple_maze,
    node_coord2label,
    max_transitions,
):
    """
    Generates sequences of optimal navigation behaviour by calculating all shortest paths between a
    start location and a goal location and randomly selecting one of them
    """
    optimal_trajectories_dfs = []
    total_transitions = 0
    for i in range(n_trials):
        goal = next(goal_sampler)
        optimal_path = random.choice(list(nx.all_shortest_paths(simple_maze, start_location, goal, method="dijkstra")))
        optimal_path = optimal_path[1:]  # remove start location (previous goal)
        n_steps = len(optimal_path)
        total_transitions += n_steps
        if max_transitions is not None and total_transitions >= max_transitions:
            break
        start_location = goal  # start next trial a previous goal
        optimal_trajectory_df = pd.DataFrame(
            {
                "trial": [i + 1] * n_steps,
                "goal": [node_coord2label[goal]] * n_steps,
                "maze_position": [node_coord2label[node] for node in optimal_path],
            }
        )
        optimal_trajectories_dfs.append(optimal_trajectory_df)
    optimal_trajectory_df = pd.concat(optimal_trajectories_dfs, axis=0).reset_index(drop=True)
    return optimal_trajectory_df


def get_random_optimal_choice_df(
    start_location,
    n_trials,
    goal_sampler,
    simple_maze,
    node_coord2label,
    max_transitions,
):
    """
    Generates sequences of optimal navigation behaviour by randomly selecting the next node from the
    set of optimal choices (which node(s) decreases my shortest-path distance to goal the most at every
    decision point
    """
    location = start_location
    goal = next(goal_sampler)
    trajectory_dicts = []
    total_transitions = 0
    for i in range(n_trials):
        goal = next(goal_sampler)
        while location != goal:
            neighbours = list(simple_maze.neighbors(location))
            path_lengths = [nx.shortest_path_length(simple_maze, n, goal, method="dijkstra") for n in neighbours]
            min_indices = np.argwhere(path_lengths == np.min(path_lengths)).flatten()
            chosen_neighbour = neighbours[np.random.choice(min_indices)]
            total_transitions += 1
            if max_transitions is not None and total_transitions >= max_transitions:
                return pd.DataFrame(trajectory_dicts)
            trajectory_dicts.append(
                {
                    "trial": i + 1,
                    "goal": node_coord2label[goal],
                    "maze_position": node_coord2label[chosen_neighbour],
                }
            )
            location = chosen_neighbour
    optimal_trajectory_df = pd.DataFrame(trajectory_dicts)
    return optimal_trajectory_df


def get_modified_random_walk_df(
    start_location,
    n_trials,
    goal_sampler,
    simple_maze,
    node_coord2label,
    max_transitions,
):
    """
    Generates sequences of optimal beahviour if the subject was following a random walk policy where they
    only backtrack (visit the node they just came from) if they are at a dead end
    """
    location = start_location
    previous_location = None
    trajectory_dicts = []
    total_transitions = 0
    for i in range(n_trials):
        goal = next(goal_sampler)
        while location != goal:
            # random walk wihout backtracking unless dead end
            neighbours = list(simple_maze.neighbors(location))
            if len(neighbours) > 1:
                neighbours = [n for n in neighbours if n != previous_location]
            chosen_neighbour = random.choice(neighbours)
            total_transitions += 1
            if max_transitions is not None and total_transitions >= max_transitions:
                return pd.DataFrame(trajectory_dicts)
            trajectory_dicts.append(
                {
                    "trial": i + 1,
                    "goal": node_coord2label[goal],
                    "maze_position": node_coord2label[chosen_neighbour],
                }
            )
            previous_location = location
            location = chosen_neighbour
    optimal_trajectory_df = pd.DataFrame(trajectory_dicts)
    return optimal_trajectory_df


def sample_without_replacement(original_list):
    """
    Generator that yields a random sample from the given list without replacement, infinitely.
    Example:
        >>> lst = [1, 2, 3, 4, 5]
        >>> gen = sample_without_replacement(lst)
        >>> next(gen)
        3
    """
    list_to_sample = original_list.copy()
    while True:
        if not list_to_sample:
            list_to_sample = original_list.copy()
        sample = random.sample(list_to_sample, 1)[0]
        list_to_sample.remove(sample)
        yield sample


# %% Distance/steps to goal calculation functions


def get_path_distances_to_goalf(trajectory_decisions_df, simple_maze, skeleton_maze):
    extended_simple_maze = mr.get_extended_simple_maze(simple_maze)
    skeleton_path_distances = dict(nx.all_pairs_dijkstra_path_length(skeleton_maze, weight="weight"))
    label2coord = {
        v: k + (0,) for k, v in nx.get_node_attributes(extended_simple_maze, "label").items()
    }  # add (0,) to make compatible with skeleton maze labels

    def get_path_distance(node1, node2, trial_phase):
        if trial_phase != "navigation":
            return np.nan
        else:
            return skeleton_path_distances[node1][node2]

    distances = pd.concat(
        [  # get maze position and goal as coordinates
            trajectory_decisions_df.maze_position.map(label2coord),
            trajectory_decisions_df.goal.map(label2coord),
            trajectory_decisions_df.trial_phase,
        ],
        axis=1,  # then find the distance betwen them (row by row)
    ).apply(
        lambda row: get_path_distance(row.maze_position, row.goal, row.trial_phase),
        axis=1,
    )
    return distances


def get_n_steps_to_goal(trajectory_decisions_df):
    """ """
    trials = trajectory_decisions_df.trial.dropna().unique()
    n_steps2goal = pd.Series(np.nan, index=trajectory_decisions_df.index)
    for t in trials:
        trial_df = trajectory_decisions_df[
            (trajectory_decisions_df.trial == t) & (trajectory_decisions_df.trial_phase == "navigation")
        ]
        steps_to_goal = np.arange(len(trial_df) - 1, -1, -1)
        n_steps2goal.loc[trial_df.index] = steps_to_goal
    return n_steps2goal
