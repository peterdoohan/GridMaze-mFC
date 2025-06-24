"""
Library for anlysing place-direction single cell tuning patterns across mazes.
"""

# %% Imports
from importlib import simple
import json
import numpy as np
import pandas as pd

from GridMaze.maze import representations as mr
from GridMaze.analysis.core import get_sessions as gs
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

# %% Functions


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
            with_data=[
                "navigation_df",
                "navigation_spike_rates_df",
                "cluster_metrics",
                "cluster_place_direction_tuning_metrics",
            ],
            must_have_data=True,
        )
    return


def get_heatmaps(subject_ID="m2", maze_pair=("maze_1", "maze_2"), verbose=True):
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
            verbose=verbose,
        )
        # filter for the common place-directions
        maze_heatmaps = maze_heatmaps[common_place_directions]
        dfs.append(maze_heatmaps)
    # combine heatmaps from both mazes and return
    all_heatmaps = pd.concat(dfs, axis=0)
    all_heatmaps = all_heatmaps.droplevel(1, axis=0).sort_index(axis=1)
    return all_heatmaps


def get_common_maze_place_directions(maze_A, maze_B):
    """
    input twos simple maze nx objects
    returns place-direction pairs that are common to both mazes
    """
    A_pd = mr.get_maze_place_direction_pairs(maze_A)
    B_pd = mr.get_maze_place_direction_pairs(maze_B)
    common_pds = set(A_pd).intersection(set(B_pd))
    return list(common_pds)
