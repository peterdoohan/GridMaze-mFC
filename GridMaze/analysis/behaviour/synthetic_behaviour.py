"""
Script for generating control behaviour for variance explained analysis.
Eg., diffusion w/or wo/ backtracking penalty, optimal behaviour etc.
"""

# %% Imports
import numpy as np
import pandas as pd
import random
import networkx as nx
from GridMaze.maze import representations as mr
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.processing import get_trajectory_decisions_dfs as tdf
from GridMaze.analysis.processing.get_navigation_df import get_cardinal_movement_direction

# %% Global Variablves

# %% Functions


def get_synthetic_maze_behavioural_sequences_df(
    policy="random_diffusion",
    subject_IDs="all",
    maze_name="maze_1",
    sessions=None,
    late_sessions=True,
    normalisation=False,
    max_steps=30,
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
            with_data=["navigation_df"],
            must_have_data=True,
        )
    dfs = []
    for session in sessions:
        if verbose:
            print(session.name)
        dfs.append(get_session_synthetic_behavioural_sequences(session, policy, normalisation, max_steps))
    output_df = pd.concat(dfs, axis=0, ignore_index=True)
    return output_df.sort_index(axis=1)


def get_session_synthetic_behavioural_sequences(
    session,
    policy="random_diffusion",
    normalisation=False,
    max_steps=30,
):
    """ """
    place_direction2idx = {_pd: i for i, _pd in enumerate(mr.get_maze_place_direction_pairs(session.simple_maze()))}
    trajectories_df = get_synthetic_behaviour(session, policy, max_steps)
    trajectories_df = trajectories_df[(trajectories_df.trial_phase == "navigation")]
    trajectories_df = trajectories_df[(trajectories_df.maze_position.notnull())]
    trajectories_df = trajectories_df[(trajectories_df.action.notnull())]
    trials = trajectories_df.trial.unique()
    session_sequences = np.zeros((len(trials), len(place_direction2idx)), dtype=int)
    for i, trial in enumerate(trials):
        trial_df = trajectories_df[trajectories_df.trial == trial]
        place_direction_sequence = list(zip(trial_df.maze_position, trial_df.action))
        for j in place_direction_sequence:
            session_sequences[i, place_direction2idx[j]] += 1
    behaviour_df = pd.DataFrame(data=session_sequences, columns=pd.MultiIndex.from_tuples(place_direction2idx.keys()))
    behaviour_df = behaviour_df.sort_index(axis=1)
    if normalisation:
        if normalisation == "mean":
            behaviour_df = behaviour_df.div(behaviour_df.mean(axis=1), axis=0)
        elif normalisation == "length":
            behaviour_df = behaviour_df.div(behaviour_df.pow(2).sum(axis=1).pow(0.5), axis=0)
        elif normalisation == "max":
            behaviour_df = behaviour_df.div(behaviour_df.max(axis=1), axis=0)
        else:
            raise ValueError(f"Unknown normalisation method: {normalisation}")
    return behaviour_df


# %% Core function for generating trajectory_decisions_df-like data structures for synthetic behaviour
def get_synthetic_behaviour(session, policy="random_diffusion", max_steps=30):
    """ """
    simple_maze = session.simple_maze()
    extended_simple_maze = mr.get_extended_simple_maze(simple_maze)
    skeleton_maze = session.skeleton_maze()
    coord2label = nx.get_node_attributes(extended_simple_maze, "label")
    label2coord = {v: k for k, v in coord2label.items()}
    coord2pos = nx.get_node_attributes(extended_simple_maze, "position")
    all_path_lengths = dict(nx.all_pairs_dijkstra_path_length(extended_simple_maze, weight="weight"))
    navigation_df = session.navigation_df
    navigation_df = navigation_df[navigation_df.trial_phase == "navigation"]
    trials = navigation_df.trial.dropna().unique()
    trial_trajectories = []
    for trial in trials:
        trial_df = navigation_df[navigation_df.trial == trial]
        if max_steps is None:
            max_steps = trial_df.steps_to_goal.future.max()
        true_nav_start = trial_df.iloc[0]
        start_loc = true_nav_start[("maze_position", "simple")]
        if len(start_loc.split("-")) == 2:  # edge
            start_loc = np.random.choice(start_loc.split("-"))  # force to start on node
        goal = true_nav_start[("goal", "")]
        if start_loc == goal:
            continue
        if policy == "random_diffusion":
            trajectory_df = random_walk(
                start_loc,
                goal,
                trial,
                max_steps,
                simple_maze,
                extended_simple_maze,
                label2coord,
                coord2label,
                backtracking_penalty=False,
            )
        elif policy == "forward_diffusion":
            trajectory_df = random_walk(
                start_loc,
                goal,
                trial,
                max_steps,
                simple_maze,
                extended_simple_maze,
                label2coord,
                coord2label,
                backtracking_penalty=True,
            )
        elif policy == "vector":
            trajectory_df = vector(
                start_loc,
                goal,
                trial,
                max_steps,
                simple_maze,
                extended_simple_maze,
                label2coord,
                coord2label,
                coord2pos,
            )
        elif policy == "optimal":
            trajectory_df = optimal(
                start_loc,
                goal,
                trial,
                simple_maze,
                extended_simple_maze,
                coord2label,
                label2coord,
                all_path_lengths,
                max_steps,
            )
        else:
            raise NotImplementedError
        trial_trajectories.append(trajectory_df)
    # add distance to goal at the end (more efficient)
    synthetic_behaviour_df = pd.concat(trial_trajectories, axis=0).reset_index(drop=True)
    synthetic_behaviour_df["geodeisc_distance_to_goal"] = get_path_distances_to_goal(
        synthetic_behaviour_df, extended_simple_maze, skeleton_maze
    )
    synthetic_behaviour_df["trial_phase"] = "navigation"  # add to match real data
    return synthetic_behaviour_df


# %% Policy functions


def random_walk(
    start_loc,
    goal,
    trial,
    max_steps,
    simple_maze,
    extended_simple_maze,
    label2coord,
    coord2label,
    backtracking_penalty=False,
):
    """ """
    goal_coord = label2coord[goal]
    location = label2coord[start_loc]
    previous_location = None
    trajectory_dicts = [{"trial": trial, "goal": goal, "maze_position": start_loc}]
    n_steps = 0
    while location != goal_coord:
        # random walk wihout backtracking unless dead end
        if n_steps >= max_steps:
            break
        neighbours = list(extended_simple_maze.neighbors(location))
        if len(neighbours) > 1 and backtracking_penalty:
            neighbours = [n for n in neighbours if n != previous_location]
        chosen_neighbour = random.choice(neighbours)
        trajectory_dicts.append(
            {
                "trial": trial,
                "goal": goal,
                "maze_position": coord2label[chosen_neighbour],
            }
        )
        previous_location = location
        location = chosen_neighbour
        n_steps += 1
        if n_steps >= max_steps:
            break
    trajectory_df = pd.DataFrame(trajectory_dicts)
    trajectory_df["action"] = tdf.get_node_edges_trajectory_actions(trajectory_df, simple_maze)
    return trajectory_df.reset_index(drop=True)


def optimal(
    start_loc,
    goal,
    trial,
    simple_maze,
    extended_simple_maze,
    coord2label,
    label2coord,
    all_path_length,
    max_steps,
):
    """
    Generates sequences of optimal navigation behaviour by randomly selecting the next node from the
    set of optimal choices (which node(s) decreases my shortest-path distance to goal the most at every
    decision point
    """
    goal_coord = label2coord[goal]
    location = label2coord[start_loc]
    trajectory_dicts = [{"trial": trial, "goal": goal, "maze_position": start_loc}]
    n_steps = 0
    while location != goal_coord:
        if n_steps >= max_steps:
            break
        neighbours = list(extended_simple_maze.neighbors(location))
        path_lengths = np.array([all_path_length[n][goal_coord] for n in neighbours])
        min_indices = np.argwhere(path_lengths == np.min(path_lengths)).flatten()
        chosen_neighbour = neighbours[np.random.choice(min_indices)]
        trajectory_dicts.append(
            {
                "trial": trial,
                "goal": goal,
                "maze_position": coord2label[chosen_neighbour],
            }
        )
        location = chosen_neighbour
        n_steps += 1
    trajectory_df = pd.DataFrame(trajectory_dicts)
    trajectory_df["action"] = tdf.get_node_edges_trajectory_actions(trajectory_df, simple_maze)
    return trajectory_df.reset_index(drop=True)


def vector(
    start_loc,
    goal,
    trial,
    max_steps,
    simple_maze,
    extended_simple_maze,
    label2coord,
    coord2label,
    coord2pos,
):
    """ """
    goal_coord = label2coord[goal]
    goal_pos = coord2pos[goal_coord]
    location = label2coord[start_loc]
    previous_location = None
    trajectory_dicts = [{"trial": trial, "goal": goal, "maze_position": start_loc}]
    n_steps = 0
    while location != goal_coord:
        if n_steps >= max_steps:
            break
        neighbours = list(extended_simple_maze.neighbors(location))
        if len(neighbours) > 1:  # no backtracking
            neighbours = [n for n in neighbours if n != previous_location]
        cosines = [cosine_to_goal(goal_pos, coord2pos[n], n, location) for n in neighbours]
        max_indices = np.argwhere(cosines == np.max(cosines)).flatten()
        chosen_neighbour = neighbours[np.random.choice(max_indices)]
        trajectory_dicts.append(
            {
                "trial": trial,
                "goal": goal,
                "maze_position": coord2label[chosen_neighbour],
            }
        )
        previous_location = location
        location = chosen_neighbour
        n_steps += 1
    trajectory_df = pd.DataFrame(trajectory_dicts)
    trajectory_df["action"] = tdf.get_node_edges_trajectory_actions(trajectory_df, simple_maze)
    return trajectory_df.reset_index(drop=True)


def cosine_to_goal(goal_pos, neighbour_pos, neighbour, current):
    goal_angle = np.arctan2(
        goal_pos[0] - neighbour_pos[0], goal_pos[1] - neighbour_pos[1]
    )  # in radians from pos y = true north
    if goal_angle < 0:
        goal_angle = 2 * np.pi + goal_angle
    cdir = get_cardinal_movement_direction(neighbour, current)
    if cdir == "N":
        return np.cos(goal_angle)
    elif cdir == "S":
        return np.cos(goal_angle + np.pi)
    elif cdir == "E":
        return np.cos(goal_angle - np.pi / 2)
    elif cdir == "W":
        return np.cos(goal_angle + np.pi / 2)


# %% Supporting functions


def get_path_distances_to_goal(trajectory_decisions_df, extended_simple_maze, skeleton_maze):
    skeleton_path_distances = dict(nx.all_pairs_dijkstra_path_length(skeleton_maze, weight="weight"))
    label2coord = {
        v: k + (0,) for k, v in nx.get_node_attributes(extended_simple_maze, "label").items()
    }  # add (0,) to make compatible with skeleton maze labels

    distances = pd.concat(
        [  # get maze position and goal as coordinates
            trajectory_decisions_df.maze_position.map(label2coord),
            trajectory_decisions_df.goal.map(label2coord),
        ],
        axis=1,  # then find the distance betwen them (row by row)
    ).apply(
        lambda row: skeleton_path_distances[row.maze_position][row.goal],
        axis=1,
    )
    return distances
