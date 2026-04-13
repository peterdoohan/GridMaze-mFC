"""
Look at decoding performance of distance-to-goal aligned to navigational errors
Does the internal representation move with subject's internal estimate of distance even
when it is wrong?
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from matplotlib import pyplot as plt

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.distance_to_goal import logreg_decoder as lr

# %% Global Variables

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "distance_to_goal" / "errors"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)


# %%


# %%


def get_distance_to_goal_decoding_df(sessions=None, resolution=0.2, verbose=True, save=False, n_jobs=-1):
    """
    slighly different params than logreg decoder
    """
    save_path = RESULTS_DIR / f"distance_to_goal_decoding_probs.parquet"
    if not save and save_path.exists():
        if verbose:
            print(f"Loading existing decoding df from {save_path}")
        return pd.read_parquet(save_path)

    if sessions is None:
        if verbose:
            print("Loading sessions...")
        sessions = gs.get_maze_sessions(
            subject_IDs="all",
            maze_names="all",
            days_on_maze="late",
            with_data=[
                "navigation_df",
                "navigation_spike_counts_df",
                "trajectory_decisions_df",
                "cluster_metrics",
                "trials_df",
                "events_df",
            ],
            must_have_data=True,
        )

    if n_jobs:
        results_dfs = Parallel(n_jobs=n_jobs)(
            delayed(lr.decode_session_distance_to_goal)(
                session,
                resolution=resolution,
                verbose=verbose,
            )
            for session in sessions
        )
    else:
        results_dfs = [
            lr.decode_session_distance_to_goal(
                session,
                resolution=resolution,
                verbose=verbose,
            )
            for session in sessions
        ]
    distance_to_goal_decoding_df = pd.concat(results_dfs, ignore_index=True)
    # save
    if save:
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
        distance_to_goal_decoding_df.to_parquet(save_path)
        if verbose:
            print(f"Saved decoding df to {save_path}")
    return distance_to_goal_decoding_df
