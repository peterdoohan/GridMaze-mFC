"""This module..."""

# %% Imports
import json
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import euclidean, cityblock
from ...maze import representations as mr
from . import get_movement_threshold as gmt

from ..core import load_data
from ..core import get_sessions as gs

# %% Global variables
from ...paths import ANALYSIS_INFO_PATH

with open(ANALYSIS_INFO_PATH / "movement_threshold.json", "r") as infile:
    MOVEMENT_THRESHOLD = json.load(infile)

FRAME_RATE = 60

# %% Main functions


def get_navigation_df(processed_data_path, analysis_data_path):
    """
    Returns a embelished version of trajectories processed data structure that includes information about:
    - trial info needed to identify, session, subject, trials, trial_phases, etc.
    - maze_positions
    - head_direction
    - speed, velocity
    - chosen actions: go_forward, turn_left, turn_right, go_back
    - cardinal_movement_direction: N, S, E, W
    - distance_to_goal: geodesic, eu, future, manhattan
    - progress_to_goal: path_length, time
    - angle_to_goal: allocentric, egocentric
    """
    # load_data
    session_info = load_data.load(processed_data_path / "session_info.json")
    trajectories_df = load_data.load(processed_data_path / "frames.trajectories.htsv")
    frames_trial_info_df = load_data.load(processed_data_path / "frames.trialInfo.htsv")
    maze_structure = session_info["maze_structure"]
    skeleton_maze = mr.skeleton_maze(maze_structure)
    simple_maze = mr.simple_maze(maze_structure)
    extended_simple_maze = mr.get_extended_simple_maze(simple_maze)
    frame_times = trajectories_df.time.to_numpy()
    # add movement information to trajectories_df
    velocities = gmt.get_velocities(trajectories_df)
    speeds = gmt.get_speeds(velocities)
    moving = [True if speed > MOVEMENT_THRESHOLD else False for speed in speeds]
    basic_actions, choice_degree = get_basic_actions(trajectories_df.maze_position.simple, simple_maze)
    cardinal_movement_direction = get_trajectory_cardinal_movement_directions(trajectories_df, simple_maze)
    trajectories_df[("velocity", "x")] = velocities.T[0]
    trajectories_df[("velocity", "y")] = velocities.T[1]
    trajectories_df[("speed", "")] = speeds
    trajectories_df[("moving", "")] = moving
    trajectories_df[("action", "basic")] = basic_actions
    trajectories_df[("action", "choice_degree")] = choice_degree
    trajectories_df[("cardinal_movement_direction", "")] = cardinal_movement_direction
    trajectories_df = trajectories_df.drop(
        columns=[("head_direction", "interpolated"), ("centroid_position", "interpolated"), ("time", "")]
    )  # remove uncessary info
    # add trial info to new df
    session_ID = gs.get_session_name(session_info)
    trial_unique_IDs = np.array(
        [f"{session_ID}_trial{int(t)}" if not np.isnan(t) else 0 for t in frames_trial_info_df.trial]
    ).astype(object)
    trial_unique_IDs[trial_unique_IDs == "0"] = np.nan
    session_info_df = pd.DataFrame(
        {
            ("time"): frame_times,
            ("subject_ID"): session_info["subject_ID"],
            ("maze_name"): session_info["maze_name"],
            ("day_on_maze"): session_info["day_on_maze"],
            ("trial_unique_ID"): trial_unique_IDs,
        }
    )
    session_info_df.columns = pd.MultiIndex.from_tuples(
        [(col, "") if isinstance(col, str) else col for col in session_info_df.columns],
    )
    navigation_df = pd.concat((session_info_df, frames_trial_info_df, trajectories_df), axis=1)
    navigation_df.columns = pd.MultiIndex.from_tuples(
        [i if isinstance(i, tuple) else (i, "") for i in navigation_df.columns]
    )
    # add distance to goal information
    distance_and_progress_to_goal_df = get_distance_and_progress_to_goal(
        navigation_df, extended_simple_maze, skeleton_maze
    )
    angles_to_goal_df = get_angles_to_goal(navigation_df, skeleton_maze)
    navigation_df = pd.concat((navigation_df, distance_and_progress_to_goal_df, angles_to_goal_df), axis=1)
    return navigation_df


# %% Subfunctions


def get_trajectory_cardinal_movement_directions(trajectories_df, simple_maze):
    """Loops through simple maze coordinate trajectories to determine the cardinal direction of movement, through
    a session. Cardinal direction of movement defined by the position the animal moved to on the maze after leaving its
    current position. NaN values for the first and last visited postiions."""
    simple_location2simple_coord = {
        **{v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()},
        **{v: k for k, v in nx.get_edge_attributes(simple_maze, "label").items()},
    }
    simple_trajectory = trajectories_df.maze_position.simple.to_numpy()
    simple_trajectory = [simple_location2simple_coord[pos] for pos in simple_trajectory]
    cardinal_movement = np.full(len(simple_trajectory), np.nan, dtype=object)
    last_position = None
    i_last = None
    for i, position in enumerate(simple_trajectory):
        if position == last_position:
            continue
        else:  # move into new location
            if last_position is None:
                last_position = position
                i_last = i
                continue
            cardinal_movement[(i_last):i] = get_cardinal_movement_direction(position, last_position)
            last_position = position
            i_last = i
    return cardinal_movement


def get_cardinal_movement_direction(position, last_position):
    """Uses current postiion and previous position (simple maze coordinates) to determine the cardinal direction of movement,
    returns a string of the cardinal direction"""
    position_type = "node" if isinstance(position[0], int) else "edge"
    last_position_type = "node" if isinstance(last_position[0], int) else "edge"
    if position_type == last_position_type:
        raise ValueError("position and last_position must be different types")
    if position_type == "node" and last_position_type == "edge":
        dx = (position[0] - last_position[0][0]) + (position[0] - last_position[1][0])
        dy = (position[1] - last_position[0][1]) + (position[1] - last_position[1][1])
    elif position_type == "edge" and last_position_type == "node":
        dx = (position[0][0] - last_position[0]) + (position[1][0] - last_position[0])
        dy = (position[0][1] - last_position[1]) + (position[1][1] - last_position[1])
    if dx > 0:
        cardinal_direction = "E"
    elif dx < 0:
        cardinal_direction = "W"
    elif dy > 0:
        cardinal_direction = "N"
    elif dy < 0:
        cardinal_direction = "S"
    return cardinal_direction


def get_session_ID(session_info):
    session_date_time = session_info["session_date"].split(" ")
    session_ID = (
        session_info["subject_ID"] + "_" + session_date_time[0] + "_" + "".join(session_date_time[-1].split(":"))
    )
    return session_ID


# %%
def get_angles_to_goal(navigation_df, skeleton_maze):
    """ """
    angle_to_goal_df = pd.DataFrame(
        columns=pd.MultiIndex.from_tuples([("angle_to_goal", "allocentric"), ("angle_to_goal", "egocentric")]),
        index=navigation_df.index,
        data=np.nan,
    )
    sk_label2sk_coord = {v: k for k, v in nx.get_node_attributes(skeleton_maze, "label").items()}
    sk_coord2pos = nx.get_node_attributes(skeleton_maze, "position")
    trials = trials = navigation_df.trial.dropna().unique()
    for trial in trials:
        trial_nav_df = navigation_df[
            np.logical_and(navigation_df.trial == trial, navigation_df.trial_phase == "navigation")
        ]
        if len(trial_nav_df) == 0:
            continue
        goal = trial_nav_df.goal.unique()[0]
        sk_goal = str(goal) + "_C"
        sk_goal_coord = sk_label2sk_coord[sk_goal]
        sk_goal_pos = sk_coord2pos[sk_goal_coord]
        trial_ego_angles = []
        trial_allo_angles = []
        for _, row in trial_nav_df.iterrows():
            current_position = (row.centroid_position.x, row.centroid_position.y)
            dy = current_position[1] - sk_goal_pos[1]  # change in y relative to goal location
            dx = current_position[0] - sk_goal_pos[0]  # change in x relative to goal location
            allocentric_angle_to_goal = np.arctan2(dy, dx)  # coordinates in  flipped order for arctan2
            allocentric_angle_to_goal = np.rad2deg(allocentric_angle_to_goal)
            allocentric_angle_to_goal = (allocentric_angle_to_goal) % 360
            head_direction = row.head_direction.values[0]
            if allocentric_angle_to_goal > head_direction:
                egocentric_angles_to_goal = allocentric_angle_to_goal - head_direction
            elif allocentric_angle_to_goal < head_direction:
                egocentric_angles_to_goal = 360 - head_direction + allocentric_angle_to_goal
            trial_allo_angles.append(allocentric_angle_to_goal)
            trial_ego_angles.append(egocentric_angles_to_goal)
        ego_angles = pd.Series(trial_ego_angles, index=trial_nav_df.index)
        allo_angles = pd.Series(trial_allo_angles, index=trial_nav_df.index)
        angle_to_goal_df.loc[trial_nav_df.index, ("angle_to_goal", "allocentric")] = allo_angles
        angle_to_goal_df.loc[trial_nav_df.index, ("angle_to_goal", "egocentric")] = ego_angles
    return angle_to_goal_df


# %% functions to get left & right turn or go straight actions


def get_basic_actions(maze_positions, simple_maze):
    """
    This function returns a 1D array of the basic actions performated over trajectory frames
    Basic actions include: go_forward, turn_left and turn_right (defined egocentrically)
    """
    simple_location2simple_coord = {
        **{v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()},
        **{v: k for k, v in nx.get_edge_attributes(simple_maze, "label").items()},
    }
    simple_positions = maze_positions.map(simple_location2simple_coord).to_numpy()
    basic_actions = np.full(len(maze_positions), np.nan, dtype=object)
    choice_degrees = np.full(len(maze_positions), np.nan, dtype=object)
    current_pos = np.nan
    current_i = np.nan
    last_pos = np.nan
    last_i = np.nan
    second_last_pos = np.nan
    for i, pos in enumerate(simple_positions):
        if not isinstance(pos[0], int):  # pos is edge
            continue
        else:  # pos is node
            if pos == current_pos:
                continue
            else:  # pos is new
                second_last_pos = last_pos
                last_pos = current_pos
                last_i = current_i
                current_pos = pos
                current_i = i
                if np.any(np.isnan(second_last_pos)) or np.any(np.isnan(last_pos)):
                    continue  # continue until variables are loaded
                else:
                    choice_degree = simple_maze.degree(last_pos)  # how many choices did they have
                    movement_direction = get_movement_direction(last_pos, second_last_pos)
                    x, y = current_pos
                    x_last, y_last = last_pos
                    if current_pos == second_last_pos:
                        basic_actions[last_i] = "go_back"
                        choice_degrees[last_i] = choice_degree
                    else:
                        if movement_direction == "N":
                            if y == (y_last + 1):
                                basic_actions[last_i] = "go_forward"
                                choice_degrees[last_i] = choice_degree
                            elif y == y_last:
                                if x == (x_last - 1):
                                    basic_actions[last_i] = "turn_left"
                                    choice_degrees[last_i] = choice_degree
                                elif x == (x_last + 1):
                                    basic_actions[last_i] = "turn_right"
                                    choice_degrees[last_i] = choice_degree
                        elif movement_direction == "S":
                            if y == (y_last - 1):
                                basic_actions[last_i] = "go_forward"
                                choice_degrees[last_i] = choice_degree
                            elif y == y_last:
                                if x == (x_last - 1):
                                    basic_actions[last_i] = "turn_right"
                                    choice_degrees[last_i] = choice_degree
                                elif x == (x_last + 1):
                                    basic_actions[last_i] = "turn_left"
                                    choice_degrees[last_i] = choice_degree
                        elif movement_direction == "E":
                            if x == (x_last + 1):
                                basic_actions[last_i] = "go_forward"
                                choice_degrees[last_i] = choice_degree
                            elif x == x_last:
                                if y == (y_last + 1):
                                    basic_actions[last_i] = "turn_left"
                                    choice_degrees[last_i] = choice_degree
                                elif y == (y_last - 1):
                                    basic_actions[last_i] = "turn_right"
                                    choice_degrees[last_i] = choice_degree
                        elif movement_direction == "W":
                            if x == (x_last - 1):
                                basic_actions[last_i] = "go_forward"
                                choice_degrees[last_i] = choice_degree
                            elif x == x_last:
                                if y == (y_last + 1):
                                    basic_actions[last_i] = "turn_right"
                                    choice_degrees[last_i] = choice_degree
                                elif y == (y_last - 1):
                                    basic_actions[last_i] = "turn_left"
                                    choice_degrees[last_i] = choice_degree
    return basic_actions, choice_degrees


def get_movement_direction(last_pos, second_last_pos):
    """
    Returns the direction of movement from the before turning
    - OUTPUTS:
        - direction: 'N', 'S', 'E', 'W'
    """
    x_last, y_last = last_pos
    x_second_last, y_second_last = second_last_pos
    if (x_second_last == x_last) and (y_second_last == (y_last - 1)):
        return "N"
    elif (x_second_last == x_last) and (y_second_last == (y_last + 1)):
        return "S"
    elif (x_second_last == (x_last - 1)) and (y_second_last == y_last):
        return "E"
    elif (x_second_last == (x_last + 1)) and (y_second_last == y_last):
        return "W"


# %% test


def get_distance_and_progress_to_goal(navigation_df, extended_simple_maze, skeleton_maze, remove_backtracking=True):
    # initialise df
    columns = [
        ("distance_to_goal", "geodesic"),
        ("distance_to_goal", "euclidean"),
        ("distance_to_goal", "future"),
        ("distance_to_goal", "manhattan"),
        ("progress_to_goal", "path_length"),
        ("progress_to_goal", "time"),
        ("steps_to_goal", "geodesic"),
        ("steps_to_goal", "future"),
    ]
    distance_and_progress_to_goal_df = pd.DataFrame(
        columns=pd.MultiIndex.from_tuples(columns), index=navigation_df.index, data=np.nan
    )
    # fine-grain distances with skeleton maze for distance_to_goal and progress_to_goal metrics
    sk_label2sk_coord = {v: k for k, v in nx.get_node_attributes(skeleton_maze, "label").items()}
    sk_coord2pos = nx.get_node_attributes(skeleton_maze, "position")
    shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(skeleton_maze, weight="weight"))
    # less fine-grain distances with simple maze for steps_to_goal metric
    es_label2coord = {v: k for k, v in nx.get_node_attributes(extended_simple_maze, "label").items()}
    # es_coord2pos = nx.get_node_attributes(extended_simple_maze, "position")
    shortest_step_lengths = dict(nx.all_pairs_dijkstra_path_length(extended_simple_maze, weight=None))
    trials = navigation_df.trial.dropna().unique()
    for trial in trials:
        trial_nav_df = navigation_df[
            np.logical_and(navigation_df.trial == trial, navigation_df.trial_phase == "navigation")
        ]
        if len(trial_nav_df) == 0:
            continue

        goal = trial_nav_df.goal.unique()[0]
        sk_goal = str(goal) + "_C"
        sk_goal_coord = sk_label2sk_coord[sk_goal]
        sk_goal_pos = sk_coord2pos[sk_goal_coord]

        # Calculate distances frame by frame
        euclidean_distances, geodesic_distances, manhattan_distances = [], [], []
        for _, row in trial_nav_df.iterrows():
            current_sk_pos = row.maze_position.skeleton  # bad naming
            current_sk_coord = sk_label2sk_coord[current_sk_pos]
            current_sk_pos = sk_coord2pos[current_sk_coord]
            geodesic_distances.append(shortest_path_lengths[current_sk_coord][sk_goal_coord])
            euclidean_distances.append(euclidean(current_sk_pos, sk_goal_pos))
            manhattan_distances.append(cityblock(current_sk_pos, sk_goal_pos))

        euclidean_distance_to_goal = pd.Series(euclidean_distances, index=trial_nav_df.index)
        geodesic_distance_to_goal = pd.Series(geodesic_distances, index=trial_nav_df.index)
        manhattan_distances_to_goal = pd.Series(manhattan_distances, index=trial_nav_df.index)

        # Calculate geodesic steps to goal frame by frame
        es_goal = str(goal)
        es_goal_coord = es_label2coord[es_goal]
        steps_to_goal = []
        for _, row in trial_nav_df.iterrows():
            current_es_pos = row.maze_position.simple
            current_es_coord = es_label2coord[current_es_pos]
            steps_to_goal.append(shortest_step_lengths[current_es_coord][es_goal_coord])

        steps_to_goal = pd.Series(steps_to_goal, index=trial_nav_df.index)

        # Calculate future path distances and path progress to goal
        sk_pos = trial_nav_df.maze_position.skeleton
        sk_traj = sk_pos[sk_pos.ne(sk_pos.shift(-1))]
        if remove_backtracking:  # CHECK THIS IS WORKING AS INTENDED
            mask = pd.Series(True, index=sk_traj.index)
            for i in range(len(sk_traj) - 2):
                if sk_traj.iloc[i] == sk_traj.iloc[i + 2]:
                    mask.iloc[i + 1] = mask.iloc[i + 2] = False
                elif sk_traj.iloc[i] == sk_traj.iloc[i + 1]:
                    mask.iloc[i + 1] = False
            sk_traj = sk_traj[mask]

        traj = sk_traj.map(sk_label2sk_coord)
        np_traj = traj.to_numpy()
        step_distances = [shortest_path_lengths[np_traj[i]][np_traj[i + 1]] for i in range(len(np_traj) - 1)]
        step_distances.append(0)
        future_distances = pd.Series(np.array(step_distances)[::-1].cumsum()[::-1], index=sk_traj.index)
        path_progress_to_goal = future_distances / future_distances.max()

        filled_future_distances = pd.Series(np.nan, index=sk_pos.index)
        filled_future_distances.update(future_distances)
        filled_future_distances = filled_future_distances.bfill()
        filled_future_distances.fillna(0, inplace=True)
        filled_path_progress_to_goal = pd.Series(np.nan, index=sk_pos.index)
        filled_path_progress_to_goal.update(path_progress_to_goal)
        filled_path_progress_to_goal = filled_path_progress_to_goal.bfill()
        filled_path_progress_to_goal.fillna(0, inplace=True)  # replace nans at the end of trajectories to 0

        # Calculate time progress to goal
        time_progress_to_goal = pd.Series(np.linspace(0, 1, len(sk_pos))[::-1], index=trial_nav_df.index)

        # calculate future steps to goal (int steps, unlike future distance in meters)
        ex_pos = trial_nav_df.maze_position.simple
        ex_traj = ex_pos[ex_pos.ne(ex_pos.shift(-1))]
        if remove_backtracking:
            mask = pd.Series(True, index=ex_traj.index)
            for i in range(len(ex_traj) - 2):
                if mask.iloc[i] == mask.iloc[i + 1] == False:
                    continue
                if ex_traj.iloc[i] == ex_traj.iloc[i + 2]:
                    mask.iloc[i + 1] = mask.iloc[i + 2] = False
            ex_traj = ex_traj[mask]

        ex_traj_steps_to_goal = pd.Series(np.arange(len(ex_traj) - 1, -0.1, -1, dtype=int), index=ex_traj.index)
        filled_future_steps = pd.Series(np.nan, index=ex_pos.index)
        filled_future_steps.update(ex_traj_steps_to_goal)
        filled_future_steps = filled_future_steps.bfill()
        filled_future_steps.fillna(0, inplace=True)

        # Add to df
        distance_and_progress_to_goal_df.loc[trial_nav_df.index, ("distance_to_goal", "geodesic")] = (
            geodesic_distance_to_goal
        )
        distance_and_progress_to_goal_df.loc[trial_nav_df.index, ("distance_to_goal", "euclidean")] = (
            euclidean_distance_to_goal
        )
        distance_and_progress_to_goal_df.loc[trial_nav_df.index, ("distance_to_goal", "future")] = (
            filled_future_distances
        )
        distance_and_progress_to_goal_df.loc[trial_nav_df.index, ("distance_to_goal", "manhattan")] = (
            manhattan_distances_to_goal
        )
        distance_and_progress_to_goal_df.loc[trial_nav_df.index, ("progress_to_goal", "path_length")] = (
            filled_path_progress_to_goal
        )
        distance_and_progress_to_goal_df.loc[trial_nav_df.index, ("progress_to_goal", "time")] = time_progress_to_goal
        distance_and_progress_to_goal_df.loc[trial_nav_df.index, ("steps_to_goal", "geodesic")] = steps_to_goal
        distance_and_progress_to_goal_df.loc[trial_nav_df.index, ("steps_to_goal", "future")] = filled_future_steps

    return distance_and_progress_to_goal_df
