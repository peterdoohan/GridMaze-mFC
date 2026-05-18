"""This script generates a json file with useful analysis information saved to disk"""

# %% Imports
import json
from GridMaze.analysis.processing import get_time_aligned_rates_dfs as tar
from GridMaze.analysis.processing import get_movement_threshold as gmt
from GridMaze.analysis.processing import get_mean_occupancy as gmo

# %% Path Variables
from GridMaze.paths import ANALYSIS_INFO_PATH

if not ANALYSIS_INFO_PATH.exists():
    ANALYSIS_INFO_PATH.mkdir(parents=True)

# %% ANALYSIS INFO VARIABLES

INTRA_TRIAL_INTERVAL_TIMES = tar.get_av_intra_trial_times()

MOVEMENT_THRESHOLD = gmt.get_movement_threshold(plot=False)

try:  # relies on navigation_dfs being populated
    MAZE_NAME2MEAN_OCCUPANCY = {
        "maze_1": gmo.get_mean_occupancy("maze_1", plot=False),
        "maze_2": gmo.get_mean_occupancy("maze_2", plot=False),
        "rooms_maze": gmo.get_mean_occupancy("rooms_maze", plot=False),
    }
except FileNotFoundError:
    MAZE_NAME2MEAN_OCCUPANCY = None

try:  # relies on trajectory decisions dfs being populated
    MAZE_NAME2EDGE_TRANSITION_COUNTS = {  # need to use eval() when loading to convery keys back to tuples
        "maze_1": {str(k): v for k, v in gmo.get_mean_transitions("maze_1", plot=False).items()},
        "maze_2": {str(k): v for k, v in gmo.get_mean_transitions("maze_2", plot=False).items()},
        "rooms_maze": {str(k): v for k, v in gmo.get_mean_transitions("rooms_maze", plot=False).items()},
    }
except AttributeError:
    MAZE_NAME2EDGE_TRANSITION_COUNTS = None


# %% Save Function


def save_analysis_info():
    """Saves analysis information to json files in the analysis_info directory"""
    filename2json_structure = {
        "intra_trial_interval_times": INTRA_TRIAL_INTERVAL_TIMES,
        "movement_threshold": MOVEMENT_THRESHOLD,
        "maze_name2mean_occupancy": MAZE_NAME2MEAN_OCCUPANCY,
        "maze_name2edge_transition_counts": MAZE_NAME2EDGE_TRANSITION_COUNTS,
    }
    # If values are None, do not save (variables are None if analysis data needed to generate them does not exist)
    # Warning will be displayed to user
    for filename, data_structure in filename2json_structure.items():
        if data_structure is None:
            print(f"Warning: {filename} not saved. Data structure is None.")
        else:
            with open(ANALYSIS_INFO_PATH / (filename + ".json"), "w") as outfile:
                outfile.write(json.dumps(data_structure, indent=4))
