""" Creates .json file with experiment info """
# %% Imports
import json
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

# %% Set experiment info path
EXPERIMENT_INFO_PATH = Path("../data/experiment_info")
if not EXPERIMENT_INFO_PATH.exists():
    EXPERIMENT_INFO_PATH.mkdir()

#%% Define experiment info

SUBJECT_IDS = ["m2", "m3", "m4", "m6", "m7", "m8"]

REWARD_SIZE2DUR = {"50uL": 210, "30uL": 132, "17uL": 82, "15uL": 74, "10uL": 55, "9uL": 51, "8uL": 47}


MAZE_1_STRUCTURE = [
    "A1-A2",
    "A3-A4",
    "A4-A5",
    "A5-A6",
    "A6-A7",
    "A2-B2",
    "A3-B3",
    "A5-B5",
    "A7-B7",
    "B4-B5",
    "B6-B7",
    "B1-C1",
    "B2-C2",
    "B3-C3",
    "B6-C6",
    "C1-C2",
    "C2-C3",
    "C3-C4",
    "C4-C5",
    "C5-C6",
    "C6-C7",
    "C2-D2",
    "C5-D5",
    "C7-D7",
    "D1-D2",
    "D3-D4",
    "D4-D5",
    "D6-D7",
    "D1-E1",
    "D2-E2",
    "D3-E3",
    "D4-E4",
    "D5-E5",
    "D6-E6",
    "E2-F2",
    "E3-F3",
    "E5-F5",
    "E6-F6",
    "E7-F7",
    "F1-F2",
    "F2-F3",
    "F4-F5",
    "F6-F7",
    "F2-G2",
    "F5-G5",
    "F6-G6",
    "G1-G2",
    "G2-G3",
    "G3-G4",
    "G4-G5",
    "G5-G6",
    "G6-G7",
]

MAZE_2_STRUCTURE = [
    "A1-A2",
    "A2-A3",
    "A3-A4",
    "A4-A5",
    "A5-A6",
    "A6-A7",
    "A1-B1",
    "A3-B3",
    "A5-B5",
    "A6-B6",
    "B1-B2",
    "B6-B7",
    "B1-C1",
    "B2-C2",
    "B4-C4",
    "B5-C5",
    "B7-C7",
    "C2-C3",
    "C3-C4",
    "C4-C5",
    "C5-C6",
    "C6-C7",
    "C1-D1",
    "C2-D2",
    "C5-D5",
    "D3-D4",
    "D4-D5",
    "D5-D6",
    "D6-D7",
    "D1-E1",
    "D3-E3",
    "D6-E6",
    "E1-E2",
    "E2-E3",
    "E4-E5",
    "E6-E7",
    "E3-F3",
    "E5-F5",
    "E7-F7",
    "F1-F2",
    "F2-F3",
    "F3-F4",
    "F4-F5",
    "F5-F6",
    "F6-F7",
    "F3-G3",
    "F5-G5",
    "F7-G7",
    "G1-G2",
    "G2-G3",
    "G4-G5",
    "G5-G6",
]

ROOMS_MAZE_STRUCTURE = [
    "A1-A2",
    "A2-A3",
    "A3-A4",
    "A4-A5",
    "A5-A6",
    "A6-A7",
    "A1-B1",
    "A2-B2",
    "A3-B3",
    "A4-B4",
    "A5-B5",
    "A6-B6",
    "A7-B7",
    "B1-B2",
    "B2-B3",
    "B3-B4",
    "B5-B6",
    "B6-B7",
    "B1-C1",
    "B2-C2",
    "B3-C3",
    "B4-C4",
    "B5-C5",
    "B6-C6",
    "B7-C7",
    "C1-C2",
    "C2-C3",
    "C3-C4",
    "C5-C6",
    "C6-C7",
    "C2-D2",
    "C3-D3",
    "C4-D4",
    "C5-D5",
    "C6-D6",
    "C7-D7",
    "D1-D2",
    "D3-D4",
    "D5-D6",
    "D6-D7",
    "D1-E1",
    "D2-E2",
    "D5-E5",
    "E1-E2",
    "E2-E3",
    "E4-E5",
    "E5-E6",
    "E6-E7",
    "E1-F1",
    "E2-F2",
    "E3-F3",
    "E4-F4",
    "E5-F5",
    "E6-F6",
    "E7-F7",
    "F1-F2",
    "F2-F3",
    "F4-F5",
    "F5-F6",
    "F6-F7",
    "F1-G1",
    "F2-G2",
    "F3-G3",
    "F4-G4",
    "F5-G5",
    "F6-G6",
    "F7-G7",
    "G1-G2",
    "G2-G3",
    "G3-G4",
    "G4-G5",
    "G5-G6",
    "G6-G7",
]

GOAL_SETS = {
    "all": [
        "A2",
        "A3",
        "A6",
        "B4",
        "B5",
        "B6",
        "C1",
        "C3",
        "C5",
        "C7",
        "D2",
        "D3",
        "D6",
        "E1",
        "E3",
        "E4",
        "E5",
        "E6",
        "F2",
        "F4",
        "F7",
        "G1",
        "G4",
        "G7",
    ],
    "subset_1": ["A2", "A6", "B4", "C1", "C3", "C7", "D3", "E3", "E5", "F2", "F7", "G4"],
    "subset_2": ["A3", "B5", "B6", "C5", "D2", "D6", "E1", "E4", "E6", "F4", "G1", "G7"],
}

MAZE_DAY2GOALS = {
    "maze_1": {
        1: "all",
        2: "all",
        3: "all",
        4: "all",
        5: "all",
        6: "all",
        7: "all",
        8: "all",
        9: "all",
        10: "all",
        11: "all",
        12: "subset_1",
        13: "subset_2",
    },
    "maze_2": {
        1: "all",
        2: "all",
        3: "all",
        4: "all",
        5: "all",
        6: "subset_2",
        7: "subset_1",
        8: "all",
        9: "all",
        10: "subset_1",
        11: "subset_2",
    },
    "rooms_maze": {
        1: "all",
        2: "all",
        3: "all",
        4: "all",
        5: "all",
        6: "subset_2",
        7: "subset_1",
        8: "all",
        9: "all",
        10: "subset_1",
        11: "subset_2",
    },
}

def get_maze_day2date():
    exp_dates = {}
    for maze_name in MAZE_CONFIGS.keys():
        start_date = date.fromisoformat(MAZE_CONFIGS[maze_name]["start"])
        end_date = date.fromisoformat(MAZE_CONFIGS[maze_name]["end"])
        day2date = {}
        d = start_date
        i = 1
        while d <= end_date:
            day2date[i] = d.isoformat()
            d += timedelta(days=1)
            i += 1
        exp_dates[maze_name] = day2date
    return exp_dates

MAZE_CONFIGS = {
    "maze_1": {
        "start": "2022-06-23", #isoformat
        "end": "2022-07-05",
        "structure": MAZE_1_STRUCTURE,
        "goal_sets": GOAL_SETS,
    },
    "maze_2": {
        "start": "2022-07-07",
        "end": "2022-07-17", 
        "structure": MAZE_2_STRUCTURE,
        "goal_sets": GOAL_SETS,
    },
    "rooms_maze": {
        "start": "2022-07-19",
        "end": "2022-07-29",
        "structure": ROOMS_MAZE_STRUCTURE,
        "goal_sets": GOAL_SETS,
    },
}


MAZE_MEASUREMENTS = { # physical detials of the maze apparatus
    "maze_node_dimensions": (7, 7),
    "lower_left_node_cartesian_center": (0.15, 0.15),  # meters
    "distance_between_node_centers": 0.18,  # meters
    "tower_width": 0.11,
}  # meters

ROOM2GOALS = {
        "top_left": ["A6", "B5", "B6", "C5", "C7", "D7"],
        "top_right": ["E4", "E5", "E6", "F4", "F7", "G4", "G7"],
        "bottom_right": ["D2", "E1", "E3", "F2", "G1"],
        "bottom_left": ["A2", "A3", "B4", "C1", "C3", "D3"],
    }

IGNORE_SESSIONS = pd.DataFrame([
    {"subject": "m3","datetime": "2022-07-28T11:45:50", "session_type": "maze", "reason": "restarted pycontrol"},
    {"subject": "m3","datetime": "2022-06-25T13:12:58", "session_type": "maze", "reason": "restarted pycontrol"},
    {"subject": "m2","datetime": "2022-07-15T11:05:01", "session_type": "maze", "reason": "reran session in afternoon"},
    {"subject": "m2","datetime": "2022-07-03T10:23:20", "session_type": "maze", "reason": "reran session in afternoon"},
    {"subject": "m2","datetime": "2022-07-20T10:57:03", "session_type": "maze", "reason": "reran session in afternoon"},
    {"subject": "m2","datetime": "2022-07-26T10:43:06", "session_type": "maze", "reason": "restarted pycontrol"},
    {"subject": "m2","datetime": "2022-06-25T11:55:43", "session_type": "maze", "reason": "reran session in afternoon"},
    {"subject": "m2","datetime": "2022-06-30T10:25:27", "session_type": "maze", "reason": "reran session in afternoon"},
    {"subject": "m7","datetime": "2022-07-11T13:50:29", "session_type": "maze", "reason": "reran session in afternoon"},
])

# %% Main Function


def save_exp_info():
    """
    Saves out experiment info described above (see get_experiment_info.py)
    """
    # process .json data structures
    filename2json_structure = {
        "subject_IDs": SUBJECT_IDS,
        "reward_size2dur": REWARD_SIZE2DUR,
        "maze_day2goals": MAZE_DAY2GOALS,
        "maze_day2date": get_maze_day2date(),
        "maze_configs": MAZE_CONFIGS,
        "maze_measurements": MAZE_MEASUREMENTS,
        "room2goals": ROOM2GOALS,
    }
    for filename, data_structure in filename2json_structure.items():
        with open(EXPERIMENT_INFO_PATH / (filename + ".json"), "w") as outfile:
            outfile.write(json.dumps(data_structure, indent=4))
    # process .htsv data structures
    IGNORE_SESSIONS.to_csv(EXPERIMENT_INFO_PATH / "ignore_sessions.htsv", sep="\t", index=False)
    return


#