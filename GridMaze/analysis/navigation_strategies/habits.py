""" """

# %% Imports
import json
import numpy as np
import pandas as pd
import networkx as nx
from collections import deque
from joblib import Parallel, delayed

from GridMaze.maze import representations as mr
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.processing import get_trajectory_decisions_dfs as td

# %% Global Variables
from GridMaze.paths import RESULTS2_PATH, EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

# %% Functions


def get_habit_values_no_history(
    session,
    subject_df=None,
    stim_day_range=None,
):
    if subject_df is None:
        subject_df = get_subject_decisions_df(
            session.subject_ID,
            n_history=1,
            save=False,
        )
    # filter subject df
    data_df = subject_df.copy()
    if stim_day_range is not None:
        data_df = subject_df[subject_df.total_stim_days.between(*stim_day_range)]
    data_df = data_df[data_df.maze_name == session.maze_name]
    # exclude current session
    data_df = data_df[data_df.session_name != session.name]
    simple_maze = session.simple_maze()
    all_state_actions = mr.get_maze_place_direction_pairs(simple_maze, edges=False)
    state_action_counts = data_df.groupby(["maze_position", "action"]).size()
    missing_state_actions = list(set(all_state_actions) - set(state_action_counts.index.to_list()))
    if len(missing_state_actions) > 0:
        state_action_counts = pd.concat(
            [state_action_counts, pd.Series(index=pd.MultiIndex.from_tuples(missing_state_actions), data=0)],
        )
    state_action_counts = state_action_counts.reset_index(name="count")
    group_sum = state_action_counts.groupby(["maze_position"])["count"].transform("sum")
    prob = state_action_counts["count"].div(group_sum).fillna(0)
    habit_values_df = state_action_counts.copy().drop(columns=["count"])
    habit_values_df["habit_value"] = prob
    habit_values_df.set_index(["maze_position", "action"], inplace=True)
    return habit_values_df


def get_habit_values_df(
    session,
    subject_df=None,
    n_history=2,
):
    """ """
    assert n_history > 0
    if subject_df is None:
        subject_df = get_subject_decisions_df(
            session.subject_ID,
            n_history=n_history,
            save=False,
        )
    assert f"history_{n_history}" in subject_df.columns
    # filter subject df
    data_df = subject_df.copy()
    data_df = data_df[data_df.maze_name == session.maze_name]
    # exclude current session
    data_df = data_df[data_df.session_name != session.name]
    # exclude decisions where history is not defined
    data_df = data_df.dropna()
    # tally state-action counds including missing ones
    simple_maze = session.simple_maze()
    habit_values_df = _get_habit_values(data_df, simple_maze, n_history)
    return habit_values_df


def _get_habit_values(data_df, simple_maze, n_history):
    node2available_actions = get_node2action_available(simple_maze, key_type="list")
    all_states = get_all_valid_histories(simple_maze, n=n_history + 1)
    all_state_actions = [(*state, a) for state in all_states for a in node2available_actions[state[-1]]]
    history_cols = [f"history_{i}" for i in range(1, n_history + 1)]
    state_action_cols = history_cols[::-1] + ["maze_position", "action"]
    state_action_counts = data_df.groupby(state_action_cols).size()
    missing_state_actions = list(set(all_state_actions) - set(state_action_counts.index.to_list()))
    state_action_counts = (
        pd.concat(
            [state_action_counts, pd.Series(index=pd.MultiIndex.from_tuples(missing_state_actions), data=0)],
        )
        .sort_index()
        .reset_index()
    )
    state_action_counts.columns = state_action_cols + ["count"]
    group_sum = state_action_counts.groupby(state_action_cols[:-1])["count"].transform("sum")
    prob = state_action_counts["count"].div(group_sum).fillna(0)
    habit_values_df = state_action_counts.copy().drop(columns=["count"])
    habit_values_df["habit_value"] = prob
    habit_values_df.set_index(state_action_cols, inplace=True)
    return habit_values_df


def get_subject_decisions_df(
    subject_ID,
    navigation_only=True,
    n_history=5,
    n_jobs=-1,
    save=False,
):
    """ """
    save_path = RESULTS2_PATH / "navigation_strategies" / "subject_decisions_dfs" / f"{subject_ID}.parquet"
    if not save and save_path.exists():
        return pd.read_parquet(save_path)

    sessions = gs.get_maze_sessions(
        subject_IDs=[subject_ID],
        with_data=["trajectories_df", "trial_info_df", "trials_df"],
        must_have_data=True,
        verbose=False,
    )
    if n_jobs:
        dfs = Parallel(n_jobs=n_jobs)(delayed(get_decisions_df)(s, navigation_only, n_history) for s in sessions)
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.concat(
            [get_decisions_df(s, navigation_only, n_history) for s in sessions],
            ignore_index=True,
        )

    if save:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path)

    return df


def _save_all_subject_decision_dfs(
    navigation_only=True,
    n_history=5,
    n_jobs=-1,
    verbose=True,
):
    for subject_ID in SUBJECT_IDS:
        if verbose:
            print(f"Processing subject {subject_ID}...")
        get_subject_decisions_df(
            subject_ID,
            navigation_only,
            n_history,
            n_jobs,
            save=True,
        )
    if verbose:
        print("All subject decision dfs saved.")


# %%


def get_decisions_df(
    session,
    navigation_only=True,
    n_history=10,
):
    """
    coppied from analysis.processing.get_trajectory_decision_dfs for convience
    """
    # load relevant processed data
    trials_df = session.trials_df
    simple_maze = session.simple_maze()
    trajectories_df = session.trajectories_df
    frames_trial_info_df = session.trial_info_df
    # get preliminary df (with data frome every frame)
    decisions_df = pd.concat(
        [
            trajectories_df[[("time", ""), ("maze_position", "simple")]],
            frames_trial_info_df,
        ],
        axis=1,
    )
    decisions_df.columns = [c[0] if isinstance(c, tuple) else c for c in decisions_df.columns]
    decisions_df["session_name"] = session.name
    decisions_df["maze_name"] = session.maze_name
    trial_unique_ID = session.name + "_trial" + decisions_df["trial"].astype(str)
    trial_unique_ID[trial_unique_ID.apply(lambda x: "nan" in x)] = np.nan
    decisions_df["trial_unique_ID"] = trial_unique_ID
    # distill trajectory decisions df to only one frame from each sequental node visit
    decisions_df["maze_position_shifted"] = decisions_df.maze_position.shift(1)
    decisions_df["maze_position_change"] = decisions_df.maze_position != decisions_df.maze_position_shifted
    decisions_df = decisions_df[decisions_df.maze_position_change]
    decisions_df = decisions_df.drop(columns=["maze_position_shifted", "maze_position_change"])
    decisions_df.reset_index(drop=True, inplace=True)
    # correct backtracking by first reducing traj to single nodes transitions and add back edges if required
    node_mask = decisions_df.maze_position.apply(lambda x: len(x.split("-")) == 1)
    decisions_df = decisions_df[node_mask]
    node_trajectory = decisions_df.maze_position
    decisions_df = decisions_df[~(node_trajectory == node_trajectory.shift(1))]
    node_traj = decisions_df.maze_position
    actions = td.get_trajectory_actions(node_traj, simple_maze)
    decisions_df["action"] = actions
    decisions_df = decisions_df[:-1].reset_index(drop=True)  # last action not defined
    # filter further based on input
    if navigation_only:
        decisions_df = decisions_df[decisions_df.trial_phase == "navigation"]

    # add position histories
    if n_history is not None:
        for i in range(1, n_history + 1):
            decisions_df[f"history_{i}"] = decisions_df.groupby("trial_unique_ID")["maze_position"].shift(i)

    return decisions_df


# %% a bit of graph theory


def get_all_valid_histories(simple_maze, n=3):
    coord2label = nx.get_node_attributes(simple_maze, "label")
    valid_walks = []
    for w in walks_of_length_n(simple_maze, n):
        # translate to label
        walk = tuple([coord2label[n] for n in w])
        valid_walks.append(walk)
    return valid_walks


def walks_of_length_n(G, n, sources=None):
    """
    Yield all node sequences (tuples) of length n that are valid walks in G.
    Walks may revisit nodes.
    If sources is None, start from every node in G.
    """
    if n <= 0:
        return
    if sources is None:
        sources = G.nodes()
    # sequences are tuples of nodes length current_len
    frontier = deque(((s,)) for s in sources)  # deque of tuple-of-tuples for memory locality
    while frontier:
        seq = frontier.popleft()
        if len(seq) == n:
            yield seq
            continue
        last = seq[-1]
        for nbr in G.neighbors(last):
            # extend
            frontier.append(seq + (nbr,))


# %% More untility fns


def get_node2action_available(simple_maze, key_type="dict"):
    """
    Returns a dict of the available directions at each node in the maze.
    The keys are the available directions ('N', 'S', 'E', 'W') and the
    values are True if the direction is available and False otherwise.
    """
    assert key_type in ["dict", "list"], "key_type must be either 'dict' or 'list'"
    node_coord2label = nx.get_node_attributes(simple_maze, "label")
    node2NSEW_available = {}
    for node in simple_maze.nodes:
        neighbors = list(simple_maze.neighbors(node))
        actions = []
        node_NSEW2available = {"N": False, "S": False, "E": False, "W": False}
        for neighbor in neighbors:
            if neighbor[0] == node[0] + 1:
                node_NSEW2available["E"] = True
                actions.append("E")
            if neighbor[0] == node[0] - 1:
                node_NSEW2available["W"] = True
                actions.append("W")
            if neighbor[1] == node[1] + 1:
                node_NSEW2available["N"] = True
                actions.append("N")
            if neighbor[1] == node[1] - 1:
                node_NSEW2available["S"] = True
                actions.append("S")
        if key_type == "dict":
            node2NSEW_available[node_coord2label[node]] = node_NSEW2available
        elif key_type == "list":
            node2NSEW_available[node_coord2label[node]] = actions
    return node2NSEW_available


# %% define anithabit metric


def get_goal_habit_scores(maze_name, subject_ID="all", stim_day_range=None):
    """ """
    all_pairs_antihabit = get_all_pairs_habit_score(
        maze_name,
        subject_ID,
        stim_day_range,
    )
    scores = {}
    for goal in all_pairs_antihabit.keys():
        _scores = []
        for start in all_pairs_antihabit[goal].keys():
            if start != goal:
                _scores.append(all_pairs_antihabit[goal][start])
        scores[goal] = np.nanmean(_scores)

    return scores


def get_all_pairs_habit_score(maze_name, subject_ID="all", stim_day_range=None):
    """
    habit score = sum(habit values along shortest path) / (path length - 2)
    np.nan if path is too short and history needed to get habit values is undefined

    note dict is directional
    dict[goal][start] = habit score for path from start to goal
    """
    # load decisions data
    if not subject_ID == "all":
        df = get_subject_decisions_df(subject_ID)
    else:
        df = pd.concat(
            [get_subject_decisions_df(s) for s in SUBJECT_IDS],
            ignore_index=True,
        )

    # filter data
    if stim_day_range is not None:
        df = df[df.total_stim_days.between(*stim_day_range)]
    df = df[df.maze_name == maze_name]

    # load maze
    simple_maze = mr.get_simple_maze(maze_name)
    coord2label = mr.get_maze_coord2label(simple_maze)
    all_shortest_paths = dict(nx.all_pairs_shortest_path(simple_maze))

    # get habit values from data
    habit_values_df = _get_habit_values(df, simple_maze, n_history=1)

    # calculate average habit score across shortest-paths between all pairs of nodes
    habit_scores = {}
    for start in all_shortest_paths.keys():
        slabel = coord2label[start]
        scores = {}
        for goal in all_shortest_paths[start].keys():
            glabel = coord2label[goal]
            paths = all_shortest_paths[start][goal]
            paths = [paths] if not isinstance(paths[0], list) else paths
            Hvs = []
            for path in paths:
                Hv = 0
                path_legth = len(path)
                if path_legth <= 2:
                    scores[glabel] = np.nan
                    continue
                for i in range(1, path_legth - 1):
                    prev_coord = path[i - 1]
                    current_coord = path[i]
                    next_coord = path[i + 1]
                    action = get_action_from_positions(current_coord, next_coord)
                    idx = (coord2label[prev_coord], coord2label[current_coord], action)
                    Hv += habit_values_df.loc[idx, "habit_value"]
                Hv = Hv / (path_legth - 2)  # average over path
                Hvs.append(Hv)
            if len(Hvs) == 0:
                scores[glabel] = np.nan
            else:
                scores[glabel] = np.nanmean(Hvs)
        habit_scores[slabel] = scores
    return habit_scores


def get_action_from_positions(current_coord, next_coord):
    """ """
    dx = next_coord[0] - current_coord[0]
    dy = next_coord[1] - current_coord[1]
    if dx == 0:
        if dy == 1:
            return "N"
        elif dy == -1:
            return "S"
    else:
        if dx == 1:
            return "E"
        elif dx == -1:
            return "W"
