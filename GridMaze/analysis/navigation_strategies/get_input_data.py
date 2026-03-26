"""
This module generates a navigation stratagies df that can be used to model maze navigation behaviour as a function of vector navigation and
structure navigation (model-based) stragegies. These dataframes are to be populated in the analysis data folder and loaded as session object attributes
"""

# %% imports
import json
import numpy as np
import pandas as pd
import networkx as nx

# from scipy.stats import zscore
from joblib import Parallel, delayed

from GridMaze.maze import representations as mr
from GridMaze.analysis.behaviour import trajectory_plotting as tp
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.navigation_strategies import habits as sh

# %% Global variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "navigation_strategies"

with (EXPERIMENT_INFO_PATH / "subject_IDs.json").open("r") as infile:
    SUBJECT_IDS = json.load(infile)

NSEW = ["N", "S", "E", "W"]  # TODO: should use this global varibale instead of defining locally in functions

NAV_STRATEGIES = [
    "vector",
    "vector_close",
    "vector_far",
    "structure",
    "structure_close",
    "structure_far",
    "habit",
    "backtracking_penalty",
    "forward_bias",
]

# %% get exp level navigation_strategies_df


def get_navigation_strategies_df(
    strategies=NAV_STRATEGIES,
    n_history=1,
    sessions=None,
    verbose=False,
    n_jobs=-1,
    save=False,
    close_far_cutoff=4,
):
    """"""
    # get strating navigation strategies df if save=False, load from disk
    exclusion_strings = ["_X_", "_ORTH_", "_close", "_far"]
    starting_strats = [s for s in strategies if all([excl not in s for excl in exclusion_strings])]
    df = _get_navigation_strategies_df(
        strategies=starting_strats,
        n_history=n_history,
        sessions=sessions,
        verbose=verbose,
        n_jobs=n_jobs,
        save=save,
    )

    # add derivative strategies (interactions, orthogonalised, close/far etc.)

    extra_dfs = []
    extra_strats = [s for s in strategies if s not in starting_strats]
    for s in extra_strats:
        # add close/far regressors
        if "_close" in s:
            base_strat = s.split("_close")[0]
            keep_mask = df.steps_to_goal.le(close_far_cutoff)
            strat_df = df.xs(base_strat, axis=1, level=0).copy()
            strat_df.loc[~keep_mask, :] = 0
            strat_df.columns = pd.MultiIndex.from_product([[s], strat_df.columns])
        elif "_far" in s:
            base_strat = s.split("_far")[0]
            keep_mask = df.steps_to_goal.gt(close_far_cutoff)
            strat_df = df.xs(base_strat, axis=1, level=0).copy()
            strat_df.loc[~keep_mask, :] = 0
            strat_df.columns = pd.MultiIndex.from_product([[s], strat_df.columns])

        else:
            raise ValueError(f"Unknown strategy: {s}")

        extra_dfs.append(strat_df)

    if len(extra_dfs) > 0:
        df = pd.concat([df] + extra_dfs, axis=1)
    return df


def _get_navigation_strategies_df(
    strategies=[
        "structure",
        "vector",
        "habit",
        "backtracking_penalty",
        "forward_bias",
    ],
    n_history=1,
    sessions=None,
    verbose=False,
    n_jobs=-1,
    save=False,
):
    """
    generate navigation strategies df from all expert stim days across subejcts
    """
    save_path = RESULTS_DIR / f"navigation_strategies_nhistory{n_history}.parquet"
    if not save and save_path.exists():
        if verbose:
            print(f"Loading existing navigation strategies df from: {save_path}")
        navigation_strategies_df = pd.read_parquet(save_path)
        return navigation_strategies_df
    if sessions is None:
        if verbose:
            print("Loading sessions...")
        sessions = gs.get_maze_sessions(
            subject_IDs="all",
            with_data=["navigation_df", "trials_df"],
            must_have_data=True,
        )

    if n_jobs:
        dfs = Parallel(n_jobs=n_jobs)(
            delayed(get_session_navigation_strategies_df)(s, strategies, n_history) for s in sessions
        )
    else:
        dfs = []
        for s in sessions:
            if verbose:
                print(f"Processing session: {s.name}")
            df = get_session_navigation_strategies_df(s, strategies, n_history)
            dfs.append(df)
    navigation_strategies_df = pd.concat(dfs, ignore_index=True)
    if save:
        if verbose:
            print(f"Saving navigation strategies df to: {save_path}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        navigation_strategies_df.to_parquet(save_path)
    return navigation_strategies_df


# %% session level navigation strategies df


def get_session_navigation_strategies_df(
    session,
    strategies=[
        "structure",
        "vector",
        "habit",
        "backtracking_penalty",
        "forward_bias",
    ],
    n_history=1,
    remove_edge_backtracks=True,
    ignore_final_step=True,
):
    # get initalised sd
    init_df = get_init_df(
        session,
        remove_edge_backtracks=remove_edge_backtracks,
        ignore_final_step=ignore_final_step,
        n_history=max(1, n_history),
    )
    # get further variables
    simple_maze = session.simple_maze()
    node2action_available = sh.get_node2action_available(simple_maze)
    label2coord = {v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()}
    coord2pos = nx.get_node_attributes(simple_maze, "position")
    all_shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(simple_maze))
    opp_actions = {"N": "S", "S": "N", "E": "W", "W": "E"}
    if "habit" in strategies:
        if n_history > 0:
            habit_values_df = sh.get_habit_values_df(
                session,
                n_history=n_history,
            )
        else:
            habit_values_df = sh.get_habit_values_no_history(session)

    # define general value mapping function
    def _get_values(row, strategy):
        if strategy == "subject_choice":
            return get_subject_choices(row[("action", "")])
        elif strategy == "optimal_action":
            return get_optimal_actions(
                row[("location", "")],
                row[("goal", "")],
                label2coord=label2coord,
                simple_maze=simple_maze,
                all_shortest_path_lengths=all_shortest_path_lengths,
            )
        elif strategy == "available":
            return get_available(
                row[("location", "")],
                node2action_available=node2action_available,
            )
        elif strategy == "vector":
            return get_vector_values(
                row[("location", "")],
                row[("goal", "")],
                label2coord=label2coord,
                coord2pos=coord2pos,
            )

        elif strategy == "structure":
            return get_structure_values(
                row[("location", "")],
                row[("goal", "")],
                label2coord=label2coord,
                simple_maze=simple_maze,
                all_shortest_path_lengths=all_shortest_path_lengths,
            )

        elif strategy == "backtracking_penalty":
            return get_backtracking_penalty_values(
                row[("previous_action", "")],
                opp_actions=opp_actions,
            )
        elif strategy == "forward_bias":
            return get_forward_bias_values(row[("previous_action", "")])
        elif strategy == "habit":
            if n_history > 0:
                histories = [row[(f"history", i)] for i in range(1, n_history + 1)]
                histories = tuple(histories[::-1])
            else:
                histories = None
            return get_habit_values(
                histories,  # reverse to get correct order
                row[("location", "")],
                habit_values_df,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    value_dfs = []
    # get values subject_choice, optimal_choice and availability + requested strats for all choices
    for v in ["subject_choice", "optimal_action", "available"] + strategies:
        df = pd.DataFrame(init_df.apply(_get_values, axis=1, strategy=v).to_list())
        df.columns = pd.MultiIndex.from_product([[v], df.columns])
        value_dfs.append(df)
    # combine all into single df
    nav_strats_df = pd.concat([init_df] + value_dfs, axis=1)
    return nav_strats_df


def get_init_df(
    session,
    remove_edge_backtracks=True,
    ignore_final_step=True,
    n_history=2,
):
    """
    initalise navigation_strategies_df with node transitions defined trial by trial
    with goal and other info specified too, that can use used to define strategies
    for modelling subject's choices

    note currently histories are defined with trial, we could fix this later to give
    inital decisions within a trial an appropriate history in the future...
    """
    # load data
    navigation_df = session.navigation_df.copy()
    session_info = session.session_info
    simple_maze = session.simple_maze()
    all_shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(simple_maze))
    label2coord = mr.get_maze_label2coord(simple_maze)
    trials_df = session.trials_df.copy()
    trials_df.set_index("trial", inplace=True)
    # further process navigation_df
    label2coord = {v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()}
    navigation_df[("maze_position", "simple_shifted")] = navigation_df.maze_position.simple.shift(1)
    navigation_df[("maze_position", "simple_change")] = (
        navigation_df.maze_position.simple != navigation_df.maze_position.simple_shifted
    )
    navigation_df = navigation_df[navigation_df.trial_phase == "navigation"]
    trials = navigation_df.trial.unique()
    # loop over trials
    dfs = []
    for t in trials:
        trial_df = navigation_df[navigation_df.trial == t]
        if trial_df.empty:
            continue
        start_time = trial_df.iloc[0].time.values[0]
        # filter transitions between nodes
        transitions_df = trial_df[trial_df.maze_position.simple_change]
        if transitions_df.empty:
            continue
        transitions_df = transitions_df[transitions_df.maze_position.simple.apply(lambda x: len(x.split("-")) == 1)]
        transitions_df = transitions_df[transitions_df.cardinal_movement_direction.notnull()].reset_index(drop=True)
        locs = transitions_df.maze_position.simple
        actions = transitions_df.cardinal_movement_direction
        histories = [locs.shift(i) for i in range(1, n_history + 1)]
        prev_action = actions.shift(1)
        times = transitions_df.time
        if remove_edge_backtracks:
            # node transitions can be double counted if a mouse backtacks on an edge or if mouse position oscillates between
            # an adjacent edge and node, if desired, remove these backtrack to give very clean trajectory transitions and choices
            backtrack_mask = locs == locs.shift(1)
            locs = locs[~backtrack_mask]
            histories = [locs.shift(i) for i in range(1, n_history + 1)]
            actions = pd.Series(get_trajectory_actions(locs, label2coord=label2coord))
            prev_action = actions.shift(1)
            times = times[~backtrack_mask]
        if ignore_final_step:
            locs, actions, prev_action, times = locs[:-1], actions[:-1], prev_action[:-1], times[:-1]
            histories = [h[:-1] for h in histories]
            if locs.empty:
                continue
        goal = trials_df.loc[t, ("goal", "")]
        steps_to_goal = [all_shortest_path_lengths[label2coord[loc]][label2coord[goal]] for loc in locs]
        # build df
        _df = pd.DataFrame(index=locs.index)
        _df[("trial", "")] = t
        _df[("trial_unique_ID", "")] = gs.get_session_name(session_info) + f"_trial{t}"
        _df[("time_in_trial", "")] = times.sub(start_time)
        _df[("goal", "")] = trials_df.loc[t, ("goal", "")]
        _df[("location", "")] = locs.values
        _df[("action", "")] = actions.values
        _df[("previous_action", "")] = prev_action.values
        _df[("nth_visit", "")] = locs.to_frame().groupby("simple").cumcount()
        _df[("steps_to_goal", "")] = steps_to_goal
        _df[("node_degree", "")] = [simple_maze.degree[label2coord[loc]] for loc in locs]
        for i in range(1, n_history + 1):
            _df[(f"history", i)] = histories[i - 1].values
        dfs.append(_df)
    # combine with session level info
    init_df = pd.concat(dfs, ignore_index=True)
    info_df = pd.DataFrame(
        {
            ("subject_ID", ""): session.subject_ID,
            ("maze_name", ""): session.maze_name,
            ("day_on_maze", ""): session.day_on_maze,
            ("late_session", ""): session.late_session,
        },
        index=init_df.index,
    )
    return pd.concat([info_df, init_df], axis=1)


# %% strategy functions (all return dict of option values for N/S/E/W)


def get_available(loc, node2action_available=None, simple_maze=None):
    if node2action_available is None:
        assert simple_maze is not None
        node2action_available = sh.get_node2action_available(simple_maze)
    return node2action_available[loc]


def get_subject_choices(action):
    return {cdir: 0 if cdir != action else 1 for cdir in ["N", "S", "E", "W"]}


def get_optimal_actions(loc, goal, label2coord=None, simple_maze=None, all_shortest_path_lengths=None):
    """
    Optimal choice defined as the option that decreases the shortest path distance to the goal the most.
    Note if multiple choice uptons have the same minimum shortest path distance to goal, all are considered optimal choices.
    """
    # check inputs
    if label2coord is None:
        assert simple_maze is not None, "Either simple_maze or label2coord must be provided"
        label2coord = {v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()}
    if all_shortest_path_lengths is None:
        assert simple_maze is not None, "Either simple_maze or all_shortest_path_lengths must be provided"
        all_shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(simple_maze))

    loc_coord = label2coord[loc]
    goal_coord = label2coord[goal]
    neighbors = list(simple_maze.neighbors(loc_coord))
    neighbor2geodesic_distance = {neighbor: all_shortest_path_lengths[neighbor][goal_coord] for neighbor in neighbors}
    min_distance = min(neighbor2geodesic_distance.values())
    optimal_neighbors = [
        neighbor for neighbor, distance in neighbor2geodesic_distance.items() if distance == min_distance
    ]
    optimal_choices = [get_neighbor_cdir(loc_coord, optimal_neighbor) for optimal_neighbor in optimal_neighbors]
    return {cdir: 1 if cdir in optimal_choices else 0 for cdir in ["N", "S", "E", "W"]}


def get_vector_values(loc, goal, label2coord=None, coord2pos=None, simple_maze=None):
    """
    Returns a dictionary of vector navigation values for each available direction from the current location to the goal.
    Navigation values are calcuated as the cosine of the allocentric angle to goal (close to 1 if travleing in direction of goal,
    close to -1 if traveling away from goal, 0 if traveling perpendicular to goal).

    Note: this function ignores whether a direction is available or not, it just returns the vector navigation value for each direction.
    This is accounted for later by specifiying which options are available in the navigation strategies dataframe.
    """
    if label2coord is None:
        assert simple_maze is not None, "Either simple_maze or label2coord must be provided"
        label2coord = {v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()}
    if coord2pos is None:
        assert simple_maze is not None, "Either simple_maze or coord2pos must be provided"
        coord2pos = nx.get_node_attributes(simple_maze, "position")

    coord = label2coord[loc]
    pos = coord2pos[coord]
    goal_pos = coord2pos[label2coord[goal]]
    goal_angle = np.arctan2(goal_pos[0] - pos[0], goal_pos[1] - pos[1])  # in radians from pos y = true north
    if goal_angle < 0:
        goal_angle = 2 * np.pi + goal_angle
    return {
        "N": np.cos(goal_angle),
        "S": np.cos(goal_angle + np.pi),
        "E": np.cos(goal_angle - np.pi / 2),
        "W": np.cos(goal_angle + np.pi / 2),
    }


def get_structure_values(loc, goal, label2coord=None, simple_maze=None, all_shortest_path_lengths=None):
    """
    Returns a dictionary of structure navigation values for each available direction from the current location to the goal.
    Options are assigned avalue of 1 if they decrease the shortest path distance to the goal, -1 if they increase the shortest
    path distance to the goal, and 0 if shortest-path distances remains unchanged.

    Note this function isgnores whether options/directions are available at each location, however shortest-path distance calucaltions
    do repsect maze structure. This is accounted for later by specifiying which options are available in the navigation strategies dataframe.
    """
    if label2coord is None:
        assert simple_maze is not None, "Either simple_maze or label2coord must be provided"
        label2coord = {v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()}
    if all_shortest_path_lengths is None:
        assert simple_maze is not None, "Either simple_maze or all_shortest_path_lengths must be provided"
        all_shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(simple_maze))

    coord = label2coord[loc]
    x, y = coord
    goal_coord = label2coord[goal]
    current_distance_to_goal = all_shortest_path_lengths[coord][goal_coord]
    neighbours = [  # need to be irrespective of maze structure
        (x, y + 1),
        (x, y - 1),
        (x + 1, y),
        (x - 1, y),
    ]
    neighbours = [node for node in neighbours if node in simple_maze.nodes()]  # remove nodes that off the maze
    values = {}
    for neigbour in neighbours:
        neigbour_cdir = get_neighbor_cdir(coord, neigbour)
        neigbour_distance = all_shortest_path_lengths[neigbour][goal_coord]
        if current_distance_to_goal > neigbour_distance:
            values[neigbour_cdir] = 1
        elif current_distance_to_goal < neigbour_distance:
            values[neigbour_cdir] = -1
        else:
            values[neigbour_cdir] = 0
    invalid_actions = list(set(["N", "S", "E", "W"]) - set(values.keys()))
    for invalid_cdir in invalid_actions:  # going off the maze is always long shortest path
        values[invalid_cdir] = -1
    # hacky reordering
    return {d: values[d] for d in ["N", "S", "E", "W"]}


def get_backtracking_penalty_values(prev_action, opp_actions=None):
    if prev_action is None:
        return {cdir: 0 for cdir in ["N", "S", "E", "W"]}
    if opp_actions is None:
        opp_actions = {"N": "S", "S": "N", "E": "W", "W": "E"}
    return {cdir: -1 if cdir == opp_actions.get(prev_action) else 0 for cdir in ["N", "S", "E", "W"]}


def get_forward_bias_values(prev_action):
    if prev_action is None:
        return {cdir: 0 for cdir in ["N", "S", "E", "W"]}
    return {cdir: 1 if cdir == prev_action else 0 for cdir in ["N", "S", "E", "W"]}


def get_habit_values(histories, loc, habit_values_df):
    """ """
    values = {}
    for action in ["N", "S", "E", "W"]:
        if histories is None:
            indx = (loc, action)
        else:
            indx = (*histories, loc, action)
        if indx not in habit_values_df.index:
            values[action] = 0
        else:
            values[action] = float(habit_values_df.loc[indx, "habit_value"])
    return values


# %% strategy utility functions


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


# %% other utility functions


def get_trajectory_actions(node_traj, simple_maze=None, label2coord=None):
    if label2coord is None:
        assert simple_maze is not None, "Either simple_maze or label2coord must be provided"
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
