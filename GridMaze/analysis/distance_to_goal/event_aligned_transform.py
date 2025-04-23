"""
Library for using empirical distributions to convert between time relative to an event (e.g. reward)
and distance (or steps) to goal, on a per subject basis. Allows for conversion between both reference
frames.
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from GridMaze.analysis.core import get_sessions as gs
from . import goal_decoding as gd


# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "distance_to_goal"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]
GOAL_SETS = ["subset_1", "subset_2", "all"]

# %% Functions


def get_step_time_transformation_df(overwrite=False):
    """ """
    save_path = RESULTS_DIR / "step_time_transformation_df.csv"
    if save_path.exists() and not overwrite:
        return pd.read_csv(save_path, index_col=0)
    else:
        print("Generating step time transformation df")
        dfs = []
        for subject in SUBJECT_IDS:
            print(f"Processing subject {subject}")
            for maze in MAZE_NAMES:
                for goal_subset in GOAL_SETS:
                    for event in ["cue", "reward"]:
                        step_time_df = get_steps_vs_time_curve(subject, maze, goal_subset, event)
                        step_time_df["subject"] = subject
                        step_time_df["maze"] = maze
                        step_time_df["goal_subset"] = goal_subset
                        step_time_df["event"] = event
                        dfs.append(step_time_df)
        step_time_df = pd.concat(dfs).reset_index(drop=True)
        # save
        step_time_df.to_csv(save_path)
        return step_time_df


def get_steps_vs_time_curve(subject, maze, goal_subset, event, max_steps=30):
    sessions = gd.get_sessions_for_analysis(subject_IDs=[subject], maze_names=[maze], goal_subsets=[goal_subset])
    dfs = []
    for session in sessions:
        df = gd.get_event_aligned_input_data(session, event=event, resolution=0.5)
        df = df[[("event_aligned_time", event), ("steps_to_goal", "future")]]
        dfs.append(df)
    step_time_df = pd.concat(dfs).reset_index(drop=True).droplevel(1, axis=1)
    step_time_curve = step_time_df.groupby("event_aligned_time").steps_to_goal.mean()
    return step_time_curve[step_time_curve.index <= max_steps + 1].reset_index()
