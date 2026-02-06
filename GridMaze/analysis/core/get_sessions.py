""" """

# %% Imports
import json
import pandas as pd
from pathlib import Path
from datetime import date

from GridMaze.analysis.core import load_data
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.maze import representations as mr

# %% Global variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, PROCESSED_DATA_PATH, ANALYSIS_DATA_PATH


MAZE_PROCESSED_DATA_STRUCTURE2FILENAME = {
    "events_df": "events.htsv",
    "trials_df": "trials.htsv",
    "spike_times": "spikes.times.npy",
    "spike_clusters": "spikes.clusters.npy",
    "cluster_metrics": "clusters.metrics.htsv",
    "tracking_df": "frames.tracking.htsv",
    "trajectories_df": "frames.trajectories.htsv",
    "trial_info_df": "frames.trialInfo.htsv",
    "lfp_times": "lfp.times.npy",
    "lfp_signal": "lfp.signal.npy",
    "lfp_metrics": "lfp.metrics.htsv",
}

MAZE_ANALYSIS_DATA_STRUCTURE2FILENAME = {
    "navigation_df": "frames.navigation.parquet",
    "navigation_spike_rates_df": "frames.spikeRates.parquet",
    "navigation_spike_counts_df": "frames.spikeCounts.parquet",
    "trial_aligned_rates_df": "trial_aligned_rates.parquet",
    "event_aligned_rates_df": "event_aligned_rates.parquet",
    "navigation_strategies_df": "navigation_strategies.parquet",
    "trajectory_decisions_df": "trajectory_decisions.parquet",
    "cluster_distance_tuning_metrics": "clusters.distanceTuningMetrics.parquet",
    "cluster_place_direction_tuning_metrics": "clusters.placeDirectionTuningMetrics.parquet",
    "cluster_egocentric_action_tuning_metrics": "clusters.egocentricActionTuningMetrics.parquet",
    "cluster_movement_metrics": "clusters.movementMetrics.parquet",
    "navigation_theta_spike_counts_df": "frames.thetaSpikeCounts.parquet",
    "navigation_4Hz_spike_counts_df": "frames.4HzSpikeCounts.parquet",
    "cluster_theta_modulation_metrics": "clusters.thetaModulationMetrics.parquet",
}

ALL_MAZE_DATA_STRUCTURES2FILENAME = {**MAZE_PROCESSED_DATA_STRUCTURE2FILENAME, **MAZE_ANALYSIS_DATA_STRUCTURE2FILENAME}

with open(Path(EXPERIMENT_INFO_PATH) / "maze_configs.json", "r") as input_file:
    MAZE_CONFIGS = json.load(input_file)

with open(Path(EXPERIMENT_INFO_PATH) / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(Path(EXPERIMENT_INFO_PATH) / "maze_day2date.json", "r") as input_file:
    MAZE_DAY2DATE = json.load(input_file)


# %% Updated functions


def get_maze_sessions(
    subject_IDs="all",
    maze_names="all",
    days_on_maze="all",
    goal_subsets="all",
    with_data="all",
    must_have_data=True,
    verbose=False,
):
    """ """
    return _get_sessions(
        "maze",
        ALL_MAZE_DATA_STRUCTURES2FILENAME,
        subject_IDs,
        maze_names,
        days_on_maze,
        goal_subsets,
        with_data,
        must_have_data,
        verbose,
    )


def _get_sessions(
    session_type,
    data2filename,
    subject_IDs,
    maze_names,
    days_on_maze,
    goal_subsets,
    with_data,
    must_have_data,
    verbose,
):
    """ """
    session_type2session_class = {
        "maze": MazeSession,
        "rest": RestSession,
    }
    with_data = list(data2filename.keys()) if with_data == "all" else with_data
    subject_IDs = SUBJECT_IDS if subject_IDs == "all" else subject_IDs
    maze_names = list(MAZE_CONFIGS.keys()) if maze_names == "all" else maze_names
    _check_request_inputs(subject_IDs, maze_names, days_on_maze, goal_subsets)
    goal_subsets = ["all", "subset_1", "subset_2"] if goal_subsets == "all" else goal_subsets
    requested_sessions = []
    for subject in subject_IDs:
        for maze in maze_names:
            all_days = [int(d) for d in MAZE_DAY2DATE[maze].keys()]
            if days_on_maze == "all":
                days = all_days
            elif days_on_maze == "late":
                days = all_days[-7:]  # last 7 days
            else:
                days = days_on_maze
            for day_on_maze in days:
                # check day_on_maze is valid
                if str(day_on_maze) not in MAZE_DAY2DATE[maze].keys():
                    continue
                session_date = MAZE_DAY2DATE[maze][str(day_on_maze)]
                session_name = f"{session_date}.{session_type}"
                processed_data_path = PROCESSED_DATA_PATH / subject / session_name
                # check goal subset is
                session_info = load_data.load(processed_data_path / "session_info.json")
                if session_type == "maze":
                    if not session_info["goal_subset"] in goal_subsets:
                        continue
                SessionClass = session_type2session_class[session_type]
                requested_sessions.append(SessionClass(subject, session_name, with_data=with_data, verbose=verbose))
    # ensure all data is available
    if must_have_data:
        requested_sessions = [s for s in requested_sessions if all(data in s.has_data for data in with_data)]
    # outpute session objects sensibly
    if len(requested_sessions) == 0:
        raise FileNotFoundError("No sessions found with the specified criteria.")
    elif len(requested_sessions) == 1:
        return requested_sessions[0]
    else:
        return requested_sessions


def _check_request_inputs(subject_IDs, maze_names, days_on_maze, goal_subsets):
    """Checks if the inputs for get_maze_sessions are valid. If not raises useful error messages."""
    for input in [subject_IDs, maze_names, days_on_maze, goal_subsets]:
        if (not isinstance(input, list)) and (input not in ["all", "late"]):
            raise ValueError(f"{input} must be a list")
    for subject in subject_IDs:
        if subject not in SUBJECT_IDS:
            raise ValueError(f"{subject} not in SUBJECT_IDS")
    for maze in maze_names:
        if maze not in MAZE_CONFIGS.keys():
            raise ValueError(f"{maze} not in MAZE_CONFIGS")
    return


# %% Session calss


class MazeSession:
    """ """

    def __init__(self, subject, session_name, with_data, verbose):
        """ """
        self.has_data = []
        processed_data_path = PROCESSED_DATA_PATH / subject / session_name
        analysis_data_path = ANALYSIS_DATA_PATH / subject / session_name
        # Load session info
        session_info = load_data.load(processed_data_path / "session_info.json")
        self.session_info = session_info
        self.name = get_session_name(session_info)
        self.date = date.fromisoformat(session_info["session_date"])
        for attr_name in [k for k in session_info.keys() if k != "date"]:
            setattr(self, attr_name, session_info[attr_name])
        # Load processed data
        for attr_name in with_data:
            file_name = ALL_MAZE_DATA_STRUCTURES2FILENAME[attr_name]
            if attr_name in with_data:
                if attr_name in MAZE_PROCESSED_DATA_STRUCTURE2FILENAME.keys():
                    file_path = processed_data_path / file_name
                elif attr_name in MAZE_ANALYSIS_DATA_STRUCTURE2FILENAME.keys():
                    file_path = analysis_data_path / file_name
                try:
                    data = load_data.load(file_path)
                    if data is not None:
                        self.has_data.append(attr_name)
                except FileNotFoundError:
                    if verbose:
                        print(f"{file_name} not found for {self.name}")
                    data = None
            else:
                data = None
            setattr(self, attr_name, data)
        return

    def __repr__(self):
        """Return a nicely formatted string representation of the MazeSession object."""
        # Define a total width for the left side (including spaces before the maze starts)
        total_width = 50  # You can adjust this value if needed for wider text

        return (
            f"\n-MazeSession{'-' * (total_width)}\n"
            f"  Subject ID     : {self.subject_ID:<{total_width - 26}}\n"
            f"  Maze Name      : {self.maze_name:<{total_width - 26}}\n"
            f"  Day on Maze    : {self.day_on_maze:<{total_width - 26}}\n"
            f"  Goal Subset    : {self.goal_subset:<{total_width - 26}}\n"
            f"  Date           : {self.date.isoformat():<{total_width - 26}}\n"
            f"{'-' * (total_width+13)}\n"
        )

    def get_clusters(self, single_units=True):
        """Returns GridMaze Cluster Objects from session"""
        return gc.get_maze_clusters(
            subject_IDs=[self.subject_ID],
            maze_names=[self.maze_name],
            days_on_maze=[self.day_on_maze],
            single_units=single_units,
        )

    def simple_maze(self):
        return mr.simple_maze(self.maze_structure)

    def skeleton_maze(self):
        return mr.skeleton_maze(self.maze_structure)

    def get_navigation_activity_df(self, type="rates", with_routes=False, cluster_kwargs={}):
        """
        Combines navigation_df (containing: trial information, distance_to_goal etc.) with neural activity aligned to
        video frames.
        """
        activity_df = self._get_activity_df(type, cluster_kwargs)
        navigation_activity_df = pd.concat([self.navigation_df, activity_df], axis=1)
        if with_routes:
            navigation_routes_df = self.navigation_routes_df.reset_index(drop=True)
            navigation_activity_df = pd.concat([navigation_activity_df, navigation_routes_df], axis=1)
        return navigation_activity_df

    def _get_activity_df(self, type, cluster_kwargs):
        """ """
        if type == "rates":
            activity_df = self.navigation_spike_rates_df
        elif type == "spikes":
            activity_df = self.navigation_spike_counts_df
        else:
            raise ValueError(f"Invalid type: {type}, must be one of ['rates', 'spikes']")
        # filter clusters, note default filter_clusters kwargs keep only single units
        keep_clusters = gc.filter_clusters(
            self.cluster_metrics, self.session_info, return_unique_IDs=True, **cluster_kwargs
        )
        keep_columns = activity_df.columns[activity_df.columns.get_level_values(1).isin(keep_clusters)]
        activity_df = activity_df[keep_columns]
        activity_df = activity_df.reset_index(drop=True)
        return activity_df


# %% MazeSession supporting functions


def get_session_name(session_info):
    return f"{session_info['subject_ID']}.{session_info['session_date']}.{session_info['session_type']}"


# %%


def get_rest_sessions(subject_IDs="all", after_maze="all", days_on_maze="all", with_data="all"):
    """ """
    subject_IDs = SUBJECT_IDS if subject_IDs == "all" else subject_IDs
    requested_sessions = []
    for subject in subject_IDs:
        session_folders = [s for s in (Path(PROCESSED_DATA_PATH) / subject).iterdir() if not s.name.startswith(".")]
        for session_folder in session_folders:
            with open(session_folder / "session_info.json", "r") as input_file:
                session_info = json.load(input_file)
            if (
                session_info["session_type"] != "rest"
                or session_info["after_maze"] not in after_maze
                or session_info["day_on_maze"] not in days_on_maze
            ):
                continue
            requested_sessions.append(
                RestSession(
                    subject,
                    session_folder.name,
                    with_data=with_data,
                )
            )
    if len(requested_sessions) == 0:
        print("No sessions found with the specified criteria.")
    elif len(requested_sessions) == 1:
        return requested_sessions[0]
    else:
        return requested_sessions


class RestSession:
    """ """

    def __init__(self, subject, session_name, with_data="all"):
        """ """
        self.has_data = with_data
        processed_data_path = PROCESSED_DATA_PATH / subject / session_name
        # Load session info
        with open(processed_data_path / "session_info.json", "r") as input_file:
            session_info = json.load(input_file)
        self.name = get_session_name(session_info)
        self.date = date.fromisoformat(session_info["session_date"])
        for attr_name in [k for k in session_info.keys() if k != "date"]:
            setattr(self, attr_name, session_info[attr_name])
        # Once ephys data has been processed for these session it could be loaded here
        return

    def __repr__(self):
        """Return a nicely formatted string representation of the ResteSession object."""
        # Define a total width for the entire row including borders
        total_width = 50

        return (
            f"\n-RestSession{'-' * (total_width)}\n"
            f"  Subject ID     : {self.subject_ID:<{total_width - 23}}💤  \n"
            f"  After Maze     : {self.after_maze:<{total_width - 24}}💤  \n"
            f"  Day on Maze    : {self.day_on_maze:<{total_width - 25}}💤  \n"
            f"  Date           : {self.date.isoformat():<{total_width - 26}}🐭  \n"
            f"{'-' * (total_width+13)}\n"  # Bottom border
        )
