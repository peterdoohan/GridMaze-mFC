"""
This module generates a navigation stratagies df that can be used to model maze navigation behaviour as a function of vector navigation and
structure navigation (model-based) stragegies. These dataframes are to be populated in the analysis data folder and loaded as session object attributes
"""
# %% imports
import numpy as np
import pandas as pd
import networkx as nx
from ..core import load_data
from ...maze import representations as mr

# %% Global variables


# %% Main function


def get_navigation_strategies_df(
    processed_data_path, analysis_data_path, remove_edge_backtracks=True
):
    """
    Returns a pandas DataFrame containing decision values for each navigation choice made by a mouse in a session.

    Parameters
    ----------
    subject_session_path : str
        e.g., 'm2/m2_2020-11-23_12-00-00'
    remove_edge_backtracks : bool, optional
        Whether to remove backtracks on edges or not. If True, backtracks on edges are removed to give clean trajectory transitions and choices. If False, all transitions are included. Default is True.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing decision values for each navigation choice made by the mouse in the session. The DataFrame has the following columns:
        - subject_ID: the ID of the mouse
        - maze_number: the number of the maze
        - day_on_maze: the day on which the maze was run
        - trial: the trial number
        - goal: the goal location
        - current_location: the current location of the mouse
        - nth_visit: the nth visit to the current location
        - vector_navigation_value_N/S/E/W: the vector navigation value for each next choice (direction)
        - structure_navigation_value_N/S/E/W: the structure navigation value for each next choice (direction)
        - penalty_value_N/S/E/W: the penalty value for each next choice, penalties are (currently) only applied to the last visited location/direction
        - choice_value_N/S/E/W: the actual choice value for each direction (1 for choice value, others = 0)
        - available_N/S/E/W: whether each direction is available from the current location (defined by maze structure)

    """
    # load data
    try:
        navigation_df = load_data.load(analysis_data_path / "frames.navigation.parquet")
        session_info = load_data.load(processed_data_path / "session_info.json")
    except FileNotFoundError:
        print(
            "Missing requisit processed/analysis data to run get_navigation_strategies_df. Returning None"
        )
        return None
    trials = [t for t in navigation_df.trial.unique() if not np.isnan(t)]
    trial2goal = navigation_df.set_index("trial").goal.dropna().to_dict()
    simple_maze = mr.simple_maze(session_info["maze_structure"])
    # process data
    node2NSEW_available = get_node2NSEW_available(simple_maze)
    navigation_df[
        ("maze_position", "simple_shifted")
    ] = navigation_df.maze_position.simple.shift(1)
    navigation_df[("maze_position", "simple_change")] = (
        navigation_df.maze_position.simple != navigation_df.maze_position.simple_shifted
    )
    navigation_df = navigation_df[navigation_df.trial_phase == "navigation"]
    trial_decision_values_dfs = []
    # set up decision df columns
    columns = pd.MultiIndex.from_tuples(
        [
            ("subject_ID", ""),
            ("maze_name", ""),
            ("day_on_maze", ""),
            ("trial", ""),
            ("goal", ""),
            ("current_location", ""),
            ("nth_visit", ""),
            *pd.MultiIndex.from_product(
                [
                    [
                        "vector_navigation_value",
                        "structure_navigation_value",
                        "penalty_value",
                        "choice_value",
                        "optimal_choice_value",
                        "available",
                    ],
                    ["N", "S", "E", "W"],
                ]
            ).tolist(),
        ]
    )
    for trial in trials:
        navigation_trial_df = navigation_df[navigation_df.trial == trial]
        goal = trial2goal[trial]
        if (
            len(navigation_trial_df) == 0
        ):  # if no navigation data for trial (same reward location activated during rewared consumption)
            trial_decision_values_df = _get_empty_decision_values_df(
                trial, goal, session_info, columns
            )
            trial_decision_values_dfs.append(trial_decision_values_df)
            continue
        trial_transitions_df = navigation_trial_df[
            navigation_trial_df.maze_position.simple_change
        ]
        # some transitions oscilated between adjacent locations over sequential frames (cardinal direction = None/NaN), exclude these:
        valid_transitions_mask = (
            trial_transitions_df.cardinal_movement_direction.notnull()
        )
        trial_transitions_df = trial_transitions_df[valid_transitions_mask]
        trial_trajectory = trial_transitions_df.maze_position.simple
        trial_choices = trial_transitions_df.cardinal_movement_direction
        nodes_mask = trial_trajectory.apply(lambda x: len(x.split("-")) == 1)
        trial_node_trajectory = trial_trajectory[nodes_mask]
        trial_node_choices = trial_choices[nodes_mask]
        if remove_edge_backtracks:
            # node transitions can be double counted if a mouse backtacks on an edge or if mouse position oscillates between
            # an adjacent edge and node, if desired, remove these backtrack to give very clean trajectory transitions and choices
            node_duplication_mask = (
                trial_node_trajectory == trial_node_trajectory.shift(1)
            )
            trial_node_trajectory = trial_node_trajectory[~node_duplication_mask]
            trial_node_choices = pd.Series(
                get_trajectory_actions(trial_node_trajectory, simple_maze)
            )
        trial_node_trajectory = trial_node_trajectory[:-1].reset_index(
            drop=True
        )  # last node is at reward
        trial_node_choices = trial_node_choices[:-1].reset_index(
            drop=True
        )  # no choice to be made at reward
        nth_visit = trial_node_trajectory.to_frame().groupby("simple").cumcount()
        if (
            len(trial_node_trajectory) == 0
        ):  # trial started close to reward (no navigation)
            trial_decision_values_df = _get_empty_decision_values_df(
                trial, goal, session_info, columns
            )
            trial_decision_values_dfs.append(trial_decision_values_df)
            continue
        trial_decision_values_df = pd.DataFrame(
            columns=columns,
            data=np.full((len(trial_node_trajectory), len(columns)), np.nan),
        )
        trial_decision_values_df[("subject_ID", "")] = session_info["subject_ID"]
        trial_decision_values_df[("maze_name", "")] = session_info["maze_name"]
        trial_decision_values_df[("day_on_maze", "")] = session_info["day_on_maze"]
        trial_decision_values_df[("trial", "")] = trial
        trial_decision_values_df[("goal", "")] = goal
        trial_decision_values_df[("current_location", "")] = trial_node_trajectory
        trial_decision_values_df[("nth_visit", "")] = nth_visit
        for i, current_location in enumerate(trial_node_trajectory):
            choice = {
                cdir: 0 if cdir != trial_node_choices[i] else 1
                for cdir in ["N", "S", "E", "W"]
            }
            optimal_choice = get_location2optimal_choice_value(
                current_location, goal, simple_maze
            )
            vector_nav_option_values = get_location2option_vector_navigation_values(
                current_location, goal, simple_maze
            )
            structure_nav_option_values = (
                get_location2option_structure_navigation_values(
                    current_location, goal, simple_maze
                )
            )
            if i == 0:
                penalty_values = {
                    "N": 0,
                    "S": 0,
                    "E": 0,
                    "W": 0,
                }  # avoid penalising first move, in future could keep track of choice before navigation to avoid this
            else:
                previous_choice = trial_node_choices[i - 1]
                penalty_values = {
                    cdir: 1 if cdir != previous_choice else -1
                    for cdir in ["N", "S", "E", "W"]
                }
            available_values = node2NSEW_available[current_location]
            for option in ["N", "S", "E", "W"]:
                trial_decision_values_df.loc[
                    i, ("vector_navigation_value", option)
                ] = vector_nav_option_values[option]
                trial_decision_values_df.loc[
                    i, ("structure_navigation_value", option)
                ] = structure_nav_option_values[option]
                trial_decision_values_df.loc[
                    i, ("penalty_value", option)
                ] = penalty_values[option]
                trial_decision_values_df.loc[i, ("choice_value", option)] = choice[
                    option
                ]
                trial_decision_values_df.loc[
                    i, ("optimal_choice_value", option)
                ] = optimal_choice[option]
                trial_decision_values_df.loc[
                    i, ("available", option)
                ] = int(available_values[option])
        trial_decision_values_dfs.append(trial_decision_values_df)
        navigation_strategies_df = pd.concat(trial_decision_values_dfs, ignore_index=True)
        # convert binary valeus in available to bool (conserving nans)
        for d in ["N", "S", "E", "W"]:
            navigation_strategies_df[("available", d)] = navigation_strategies_df[("available", d)].map(_to_bool)
    return navigation_strategies_df

def _to_bool(x):
    """Convers binary to bool while retainings nans"""
    if pd.isna(x):
        return x
    return bool(x)

def get_trajectory_actions(node_traj, simple_maze):
    label2coord = {
        v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()
    }
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


def _get_empty_decision_values_df(trial, goal, session_info, columns):
    """
    Returns an empty trial_decision_values_df for trials where mouse started close to reward (no navigation).
    Still contains session and trial info for consistency
    """
    trial_decision_values_df = pd.DataFrame(
        columns=columns, data=np.full((1, len(columns)), np.nan)
    )
    trial_decision_values_df[("subject_ID", "")] = session_info["subject_ID"]
    trial_decision_values_df[("maze_name", "")] = session_info["maze_name"]
    trial_decision_values_df[("day_on_maze", "")] = session_info["day_on_maze"]
    trial_decision_values_df[("trial", "")] = trial
    trial_decision_values_df[("goal", "")] = goal
    return trial_decision_values_df


# %% Supporting functions


def get_location2option_vector_navigation_values(current_location, goal, simple_maze):
    """
    Returns a dictionary of vector navigation values for each available direction from the current location to the goal.
    Navigation values are calcuated as the cosine of the allocentric angle to goal (close to 1 if travleing in direction of goal,
    close to -1 if traveling away from goal, 0 if traveling perpendicular to goal).

    Note: this function ignores whether a direction is available or not, it just returns the vector navigation value for each direction.
    This is accounted for later by specifiying which options are available in the navigation strategies dataframe.

    Parameters
    ----------
    current_location : str
        The label of the current location in the maze.
    goal : str
        The label of the goal location in the maze.
    simple_maze : networkx.Graph
        The maze as a networkx graph.

    Returns
    -------
    dict
        A dictionary containing the vector navigation value for each available direction from the current location to the goal.
        The keys are the available directions ('N', 'S', 'E', 'W') and the values are the corresponding vector navigation values.

    """
    location_label2coord = {
        **{v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()},
        **{v: k for k, v in nx.get_edge_attributes(simple_maze, "label").items()},
    }

    location_coord2position = {
        **nx.get_node_attributes(simple_maze, "position"),
        **nx.get_edge_attributes(simple_maze, "position"),
    }

    current_coord = location_label2coord[current_location]
    current_position = location_coord2position[current_coord]
    goal_position = location_coord2position[location_label2coord[goal]]
    goal_angle = np.arctan2(
        goal_position[0] - current_position[0], goal_position[1] - current_position[1]
    )  # in radians from pos y = true north
    if goal_angle < 0:
        goal_angle = 2 * np.pi + goal_angle
    NSEW2navigation_values = {}
    for neighbour_cdir in ["N", "S", "E", "W"]:
        if neighbour_cdir == "N":
            NSEW2navigation_values[neighbour_cdir] = np.cos(goal_angle)
        elif neighbour_cdir == "S":
            NSEW2navigation_values[neighbour_cdir] = np.cos(goal_angle + np.pi)
        elif neighbour_cdir == "E":
            NSEW2navigation_values[neighbour_cdir] = np.cos(goal_angle - np.pi / 2)
        elif neighbour_cdir == "W":
            NSEW2navigation_values[neighbour_cdir] = np.cos(goal_angle + np.pi / 2)
    return NSEW2navigation_values


def get_location2option_structure_navigation_values(
    current_location, goal, simple_maze
):
    """
    Returns a dictionary of structure navigation values for each available direction from the current location to the goal.
    Options are assigned avalue of 1 if they decrease the shortest path distance to the goal, -1 if they increase the shortest
    path distance to the goal, and 0 if shortest-path distances remains unchanged.

    Note this function isgnores whether options/directions are available at each location, however shortest-path distance calucaltions
    do repsect maze structure. This is accounted for later by specifiying which options are available in the navigation strategies dataframe.

    Parameters
    ----------
    current_location : str
        The label of the current location in the maze.
    goal : str
        The label of the goal location in the maze.
    simple_maze : networkx.Graph
        The maze as a networkx graph.

    Returns
    -------
    dict
        A dictionary containing the structure navigation value for each available direction from the current location to the goal.
        The keys are the available directions ('N', 'S', 'E', 'W') and the values are the corresponding structure navigation values.
    """
    location_label2coord = {
        **{v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()},
        **{v: k for k, v in nx.get_edge_attributes(simple_maze, "label").items()},
    }
    current_coord = location_label2coord[current_location]
    current_x, current_y = current_coord
    goal_coord = location_label2coord[goal]
    current_geodesic_distance_to_goal = nx.shortest_path_length(
        simple_maze, current_coord, goal_coord, weight="weight"
    )
    # neighbors = list(simple_maze.neighbors(current_coord))
    neighbours = [
        (current_x, current_y + 1),
        (current_x, current_y - 1),
        (current_x + 1, current_y),
        (current_x - 1, current_y),
    ]
    neighbours = [
        node for node in neighbours if node in simple_maze.nodes()
    ]  # remove nodes that are not in the maze
    NSEW2navigation_values = {}
    for neigbour in neighbours:
        neigbour_cdir = get_neighbor_cdir(current_coord, neigbour)
        neigbour_geodesic_distance_to_goal = nx.shortest_path_length(
            simple_maze, neigbour, goal_coord, weight="weight"
        )
        if current_geodesic_distance_to_goal > neigbour_geodesic_distance_to_goal:
            NSEW2navigation_values[neigbour_cdir] = 1
        elif current_geodesic_distance_to_goal < neigbour_geodesic_distance_to_goal:
            NSEW2navigation_values[neigbour_cdir] = -1
        else:
            NSEW2navigation_values[neigbour_cdir] = 0
    invalid_cdirs = list(set(["N", "S", "E", "W"]) - set(NSEW2navigation_values.keys()))
    for (
        invalid_cdir
    ) in invalid_cdirs:  # going off the maze is always long shortest path
        NSEW2navigation_values[invalid_cdir] = -1
    return NSEW2navigation_values


def get_location2optimal_choice_value(current_location, goal, simple_maze):
    """
    Optimal choice defined as the option that decreases the shortest path distance to the goal the most.
    Note if multiple choice uptons have the same minimum shortest path distance to goal, all are considered optimal choices.
    """
    location_label2coord = {
        **{v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()},
        **{v: k for k, v in nx.get_edge_attributes(simple_maze, "label").items()},
    }
    current_coord = location_label2coord[current_location]
    goal_coord = location_label2coord[goal]
    neighbors = list(simple_maze.neighbors(current_coord))
    neighbor2geodesic_distance = {
        neighbor: nx.shortest_path_length(
            simple_maze, neighbor, goal_coord, weight="weight"
        )
        for neighbor in neighbors
    }
    min_distance = min(neighbor2geodesic_distance.values())
    optimal_neighbors = [
        neighbor
        for neighbor, distance in neighbor2geodesic_distance.items()
        if distance == min_distance
    ]
    optimal_choices = [
        get_neighbor_cdir(current_coord, optimal_neighbor)
        for optimal_neighbor in optimal_neighbors
    ]
    return {cdir: 1 if cdir in optimal_choices else 0 for cdir in ["N", "S", "E", "W"]}


def get_location2positions(simple_maze):
    """Returns a dict of the positions (x,y in meters) of each location in the maze."""
    node_label2coord = nx.get_node_attributes(simple_maze, "label")
    node_label2position = nx.get_node_attributes(simple_maze, "position")
    node_coord2position = {
        node_label2coord[node]: node_label2position[node] for node in node_label2coord
    }
    edge_label2coord = nx.get_edge_attributes(simple_maze, "label")
    edge_label2position = nx.get_edge_attributes(simple_maze, "position")
    edge_coord2position = {
        edge_label2coord[edge]: edge_label2position[edge] for edge in edge_label2coord
    }
    return {**node_coord2position, **edge_coord2position}


def get_node2NSEW_available(simple_maze):
    """
    Returns a dict of the available directions at each node in the maze.
    The keys are the available directions ('N', 'S', 'E', 'W') and the
    values are True if the direction is available and False otherwise.
    """
    node_coord2label = nx.get_node_attributes(simple_maze, "label")
    node2NSEW_available = {}
    for node in simple_maze.nodes:
        neighbors = list(simple_maze.neighbors(node))
        node_NSEW2available = {"N": False, "S": False, "E": False, "W": False}
        for neighbor in neighbors:
            if neighbor[0] == node[0] + 1:
                node_NSEW2available["E"] = True
            if neighbor[0] == node[0] - 1:
                node_NSEW2available["W"] = True
            if neighbor[1] == node[1] + 1:
                node_NSEW2available["N"] = True
            if neighbor[1] == node[1] - 1:
                node_NSEW2available["S"] = True
        node2NSEW_available[node_coord2label[node]] = node_NSEW2available
    return node2NSEW_available


def get_neighbor_cdir(location_coord, neigbour_coord):
    """Returns the cardinal direction of the neighbour relative to the current location."""
    if location_coord[0] == neigbour_coord[0] + 1:
        return "W"
    elif location_coord[0] == neigbour_coord[0] - 1:
        return "E"
    elif location_coord[1] == neigbour_coord[1] + 1:
        return "S"
    elif location_coord[1] == neigbour_coord[1] - 1:
        return "N"
