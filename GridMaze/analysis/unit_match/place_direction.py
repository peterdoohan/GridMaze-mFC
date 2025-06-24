"""
Library for anlysing place-direction single cell tuning patterns across mazes.
"""

# %% Imports
import json
import random
import numpy as np
import pandas as pd
from collections import Counter
from copy import copy

from GridMaze.maze import representations as mr
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import convert
from GridMaze.analysis.place_direction import dimensionality_reduction as pdr
from GridMaze.analysis.unit_match import get_across_maze_matches as mm

# %% Global Variables
from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

RESULTS_DIR = RESULTS_PATH / "unit_match" / "place_direction"

MAZE_PAIR2VALID_DAYS = {
    "maze_1.maze_2": {"maze_1": [10, 11, 12, 13], "maze_2": [1, 2, 3, 4, 5, 6, 7]},
    "maze_2.rooms_maze": {"maze_2": [9, 10, 11], "rooms_maze": [1, 2, 3, 4, 5, 6, 7]},
}

# %% Load data functions


def get_place_direction_metrics(subject_ID="m2", maze_pair=("maze_1", "maze_2"), verbose=False):
    """ """
    _maze_pair = f"{maze_pair[0]}.{maze_pair[1]}"
    dfs = []
    for maze in maze_pair:
        if verbose:
            print(f"Loading sessions for {subject_ID} on {maze} maze")
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names=[maze],
            days_on_maze=MAZE_PAIR2VALID_DAYS[_maze_pair][maze],
            with_data=["cluster_place_direction_tuning_metrics"],
            must_have_data=True,
        )
        for session in sessions:
            df = session.cluster_place_direction_tuning_metrics
            dfs.append(df[df.single_unit])
    return pd.concat(dfs, axis=0)


def get_heatmaps(subject_ID="m2", maze_pair=("maze_1", "maze_2"), verbose=False):
    """
    returns all possible relevant place-direction heatmaps for a subject and pair of mazes
    as a df, with filtered place-direction paris common to both mazes
    """
    # get place directions that are common to both mazes
    simple_maze_A, simple_maze_B = [mr.get_simple_maze(m) for m in maze_pair]
    common_place_directions = get_common_maze_place_directions(simple_maze_A, simple_maze_B)
    # for each maze in the pair, generate the place-direction heatamps for comparison
    _maze_pair = f"{maze_pair[0]}.{maze_pair[1]}"
    dfs = []
    for maze in maze_pair:
        if verbose:
            print(f"Loading sessions for {subject_ID} on {maze} maze")
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names=[maze],
            days_on_maze=MAZE_PAIR2VALID_DAYS[_maze_pair][maze],
            with_data=[
                "navigation_df",
                "navigation_spike_rates_df",
                "cluster_metrics",
                "cluster_place_direction_tuning_metrics",
            ],
            must_have_data=True,
        )
        if verbose:
            print(f"generating place-direction heatmaps")
        maze_heatmaps = pdr.get_population_place_direction_tuning(
            sessions=sessions,
            fill_nans=False,
            normalisation=False,
            min_split_corr=None,
            max_steps_to_goal=30,
            place_direction_tuned=False,
            verbose=verbose,
        )
        # filter for the common place-directions
        maze_heatmaps = maze_heatmaps[common_place_directions]
        dfs.append(maze_heatmaps)
    # combine heatmaps from both mazes and return
    all_heatmaps = pd.concat(dfs, axis=0)
    all_heatmaps = all_heatmaps.droplevel(1, axis=0).sort_index(axis=1)
    return all_heatmaps


# %% Msic


def get_permuted_cluster_matches(subject_ID="m2", maze_pair=("maze_1", "maze_2"), n_permutations=1000):
    """ """
    # get number of matches for each session pair in true data (match for each permutation)
    session_pair2count = _session_pair2n_matches(subject_ID, maze_pair)
    # get all availble in a session for matching
    session_name2single_units = _session_name2single_units(subject_ID, maze_pair)
    # get permuted matches
    permuted_matches = []
    for _ in range(n_permutations):
        pseudo_matches = []
        for session_pair, n_matches in session_pair2count.items():
            A_units, B_units = [copy(session_name2single_units[s]) for s in session_pair]
            random.shuffle(A_units),
            random.shuffle(B_units)
            random_matches = list(zip(A_units[:n_matches], B_units[:n_matches]))
            pseudo_matches.extend(random_matches)
        permuted_matches.append(pseudo_matches)
    return permuted_matches


def _session_pair2n_matches(subject_ID, maze_pair=("maze_1", "maze_2")):
    """ """
    true_matches = mm.get_cross_maze_matches(subject_ID, maze_pair[0], maze_pair[1])
    match_session_names = [[c.split("_")[0] for c in m] for m in true_matches]
    session_pair_counts = Counter(tuple(pair) for pair in match_session_names)
    return dict(session_pair_counts)


def _session_name2single_units(subject_ID="m2", maze_pair=("maze_1", "maze_2")):
    """
    Note unit-match only considers single units when doing matching
    """
    _maze_pair = f"{maze_pair[0]}.{maze_pair[1]}"
    session_name2single_units = {}
    for maze in maze_pair:
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names=[maze],
            days_on_maze=MAZE_PAIR2VALID_DAYS[_maze_pair][maze],
            with_data=["cluster_metrics"],
            must_have_data=True,
        )
        for session in sessions:
            df = session.cluster_metrics
            session_info = session.session_info
            single_units = df[df.single_unit].cluster_ID
            session_name2single_units[session.name] = list(
                convert.cluster_IDs2scluster_unique_IDs(session_info, single_units)
            )
    return session_name2single_units


def get_common_maze_place_directions(maze_A, maze_B):
    """
    input twos simple maze nx objects
    returns place-direction pairs that are common to both mazes
    """
    A_pd = mr.get_maze_place_direction_pairs(maze_A)
    B_pd = mr.get_maze_place_direction_pairs(maze_B)
    common_pds = set(A_pd).intersection(set(B_pd))
    return list(common_pds)
