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


# %% Builders for each analysis_info data structure


def _build_intra_trial_interval_times():
    return tar.get_av_intra_trial_times()


def _build_movement_threshold():
    return gmt.get_movement_threshold(plot=False)


def _build_maze_name2mean_occupancy():
    # relies on frames.navigation.parquet being populated
    return {
        "maze_1": gmo.get_mean_occupancy("maze_1", plot=False),
        "maze_2": gmo.get_mean_occupancy("maze_2", plot=False),
        "rooms_maze": gmo.get_mean_occupancy("rooms_maze", plot=False),
    }


def _build_maze_name2edge_transition_counts():
    # relies on trajectory_decisions.parquet being populated
    # need to use eval() when loading to convert keys back to tuples
    return {
        "maze_1": {str(k): v for k, v in gmo.get_mean_transitions("maze_1", plot=False).items()},
        "maze_2": {str(k): v for k, v in gmo.get_mean_transitions("maze_2", plot=False).items()},
        "rooms_maze": {str(k): v for k, v in gmo.get_mean_transitions("rooms_maze", plot=False).items()},
    }


BUILDERS = {
    "intra_trial_interval_times": _build_intra_trial_interval_times,
    "movement_threshold": _build_movement_threshold,
    "maze_name2mean_occupancy": _build_maze_name2mean_occupancy,
    "maze_name2edge_transition_counts": _build_maze_name2edge_transition_counts,
}


# %% Save Function


def save_analysis_info(data_structures="all"):
    """Build and save analysis_info json files to ANALYSIS_INFO_PATH.

    Parameters
    ----------
    data_structures : "all" or list of str, optional
        Which analysis_info files to (re)compute and save. Defaults to "all".
        Valid names (each saved as <name>.json):
            - "intra_trial_interval_times"
            - "movement_threshold"
            - "maze_name2mean_occupancy"          (needs frames.navigation.parquet)
            - "maze_name2edge_transition_counts"  (needs trajectory_decisions.parquet)
    """
    if data_structures == "all":
        data_structures = list(BUILDERS.keys())
    unknown = [n for n in data_structures if n not in BUILDERS]
    if unknown:
        raise ValueError(
            f"Unknown analysis_info structure(s): {unknown}. Valid names: {list(BUILDERS.keys())}"
        )
    for name in data_structures:
        try:
            data = BUILDERS[name]()
        except FileNotFoundError:
            print(f"Warning: {name} not saved. Missing prerequisite analysis data.")
            continue
        if data is None:
            print(f"Warning: {name} not saved. Data structure is None.")
            continue
        with open(ANALYSIS_INFO_PATH / (name + ".json"), "w") as outfile:
            outfile.write(json.dumps(data, indent=4))
