"""
Lib for generating permuted place-direction heatmaps for control analyses
"""

# %% Imports
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import permute
from GridMaze.analysis.place_direction import dimensionality_reduction as pdr

# %% Global Variables
from GridMaze.paths import RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "place_direction" / "permuted_heatmaps"

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]
# %% Functions


def populate_permuted_place_direction_heatmaps():
    """ """

    output_df = get_permuted_population_place_direction_tuning()
    output_df.to_csv(RESULTS_DIR / "permuted_place_direction_heatmaps.csv", index=True)
    return output_df


def get_permuted_population_place_direction_tuning(
    maze_name="maze_1",
    n_permutations=10,
    late_sessions=True,
    verbose=True,
    max_jobs=20,
):
    """ """
    # if session objects are not input, generate them from input filters
    days_on_maze = "late" if late_sessions else "all"
    if verbose:
        print("Loading sessions ...")
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
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

    permuted_dfs = Parallel(n_jobs=max_jobs)(
        delayed(_process_permutation)(sessions, i, verbose) for i in range(n_permutations)
    )
    output_df = pd.concat(permuted_dfs, axis=0, ignore_index=True)
    return output_df


def _process_permutation(sessions, i, verbose):
    """ """
    if verbose:
        print(f"permutation {i}")
    dfs = []
    for session in sessions:
        if verbose:
            print(session.name)
        df = get_session_permuted_place_direction_tuning(session)
        if df is None:
            continue  # not pd tuned clusters
        df.index.name = "cluster_unique_ID"
        df[("subject_ID", "")] = session.subject_ID
        df[("permutation", "")] = i
        df.set_index(["subject_ID", "permutation"], append=True, inplace=True)
        dfs.append(df)
    pop_pd_tuning_df = pd.concat(dfs, axis=0, ignore_index=True)
    return pop_pd_tuning_df


def get_session_permuted_place_direction_tuning(session):
    # load data
    navigation_df = session.navigation_df
    rates_df = session.navigation_spike_rates_df
    # circularly permute spikes relative to behaviour
    _rates_df = permute.random_circular_shift(rates_df)
    # get place_direction heatmaps from permuted data
    navigation_rates_df = pd.concat([navigation_df, _rates_df.reset_index(drop=True)], axis=1)
    place_direction_tuning = pdr.get_session_place_direction_tuning(
        session, navigation_rates_df, fill_nans="mean", normalisation="length"
    )
    return place_direction_tuning
