"""This module get session information from pycontrol files using details manually specified in experiment_info.json"""

# %% Imports
import json
import pandas as pd
from datetime import date
from GridMaze.preprocessing import pycontrol_data_import as di

# %% Global variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "maze_configs.json", "r") as infile:
    MAZE_CONFIGS = json.load(infile)

with open(EXPERIMENT_INFO_PATH / "reward_size2dur.json", "r") as infile:
    REWARD_SIZE2DUR = json.load(infile)

with open(EXPERIMENT_INFO_PATH / "maze_day2goals.json", "r") as infile:
    MAZE_DAY2GOALS = json.load(infile)

EXPERIMENT_START_DATE = date.fromisoformat(MAZE_CONFIGS["maze_1"]["start"])

PROBE_DEPTHS_DF = pd.read_csv(EXPERIMENT_INFO_PATH / "probe_depths.htsv", sep="\t")
PROBE_DEPTHS_DF.loc[:, "date"] = PROBE_DEPTHS_DF.date.apply(date.fromisoformat)

# %% Main functions


def get_session_info(session_dir):
    """
    Processes maze or open field session info (dictorionary w/ subject_ID, date, maze etc.)
    """
    if session_dir.session_type == "rest":
        return get_rest_session_info(session_dir)
    elif session_dir.session_type == "maze":
        return get_maze_session_info(session_dir)


def get_rest_session_info(session_dir):
    """
    Returns a dictionary containing rest session information
    """
    probe_depth, tissue_sample = _get_probe_info(session_dir)
    session_info = {
        "subject_ID": session_dir.subject_ID,
        "session_type": "rest",
        "session_date": session_dir.date.isoformat(),
        "experimental_day": (session_dir.date - EXPERIMENT_START_DATE).days + 1,
        "maze_name": session_dir.maze_name,
        "day_on_maze": _get_day_on_maze(session_dir.maze_name, session_dir.date),
        "probe_depth": float(probe_depth),
        "tissue_sample": tissue_sample,
    }
    return session_info


def get_maze_session_info(session_dir):
    """
    Returns a dictionary containing maze session information
    """
    pycontrol_path = session_dir.pycontrol_path
    session = di.Session(pycontrol_path)
    session_date = session_dir.date
    experimental_day = (session_date - EXPERIMENT_START_DATE).days + 1
    maze_name = session_dir.maze_name
    day_on_maze = _get_day_on_maze(maze_name, session_date)
    goal_subset = MAZE_DAY2GOALS[maze_name][str(day_on_maze)]
    goals = MAZE_CONFIGS[maze_name]["goal_sets"][goal_subset]
    probe_depth, tissue_sample = _get_probe_info(session_dir)
    session_info = {
        "subject_ID": session_dir.subject_ID,
        "session_type": "maze",
        "session_date": session_date.isoformat(),
        "experimental_day": experimental_day,
        "maze_name": maze_name,
        "maze_structure": MAZE_CONFIGS[maze_name]["structure"],
        "day_on_maze": day_on_maze,
        "goal_subset": goal_subset,
        "goals": goals,
        "reward_size": _get_reward_size(session),
        "probe_depth": float(probe_depth),
        "tissue_sample": tissue_sample,
    }
    return session_info


# %% sub functions


def _get_day_on_maze(maze_name, session_date):
    maze_start_day = date.fromisoformat(MAZE_CONFIGS[maze_name]["start"])
    return (session_date - maze_start_day).days + 1


def _get_reward_size(session):
    """Retrieves the reward size used during the session
    - Note: only compatiable in sessions where reward size is fixed"""
    reward_dur2size = {v: k for k, v in REWARD_SIZE2DUR.items()}  # Flips dictionary
    reward_duration = int(session.task_variables[0].split()[-1])
    return reward_dur2size[reward_duration] if reward_duration in reward_dur2size.keys() else f"{reward_duration}ms"


def _get_probe_info(session_dir):
    """
    Returns the most recent probe depth measurement for a given subject on or before a given date.
    date should be datetime object
    """
    # Filter for the subject and for measurements on or before the provided date
    subject_df = PROBE_DEPTHS_DF[
        (PROBE_DEPTHS_DF["subject"] == session_dir.subject_ID) & (PROBE_DEPTHS_DF["date"] <= session_dir.date)
    ]
    # Get the row with the latest date (i.e. the most recent measurement)
    latest_row = subject_df.sort_values("date", ascending=False).iloc[0]
    return latest_row["probe_depth"], latest_row["tissue_sample"]
