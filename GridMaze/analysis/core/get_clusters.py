""" """

# %% Imports
import json
import re
import numpy as np
import pandas as pd
from datetime import date

from . import convert
from . import load_data
from . import filter as filt
from . import get_sessions as gs

from ..cluster_tuning import actions, angle_to_goal, distance_to_goal, events, spatial
from ..processing import get_cluster_heatmap_dfs as chm
from ...maze import representations as mr

# %% Global Variables

from ...paths import PROCESSED_DATA_PATH, ANALYSIS_DATA_PATH, EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "maze_configs.json", "r") as input_file:
    MAZE_CONFIGS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "maze_day2date.json", "r") as input_file:
    MAZE_DAY2DATE = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "maze_day2goals.json", "r") as input_file:
    MAZE_DAY2GOALS = json.load(input_file)

FRAME_RATE = 60


# %%
def get_cluster(cluster_unique_ID):
    """
    Returns a MazeCluster object specified by a given cluster_unique_ID
    """
    # extract subject, session_name and cluster_ID from cluster_unique_ID
    subject_date_maze, cluster = cluster_unique_ID.split("_")
    subject, session_name = subject_date_maze.split(".", 1)
    cluster_ID = int(re.search(r"cluster(\d+)", cluster).group(1))
    # instantiate MazeCluster object
    cluster = MazeCluster(subject, session_name, cluster_ID)
    return cluster


def get_clusters(
    subject_IDs="all",
    maze_names="all",
    days_on_maze="all",
    goal_subsets="all",
    cluster_IDs="all",
    single_units=True,
    multi_units=False,
    noise_units=False,
):
    """
    Returns a list of MazeCluster objects that meet the specified criteria.
    """
    subject_IDs = SUBJECT_IDS if subject_IDs == "all" else subject_IDs
    maze_names = list(MAZE_CONFIGS.keys()) if maze_names == "all" else maze_names
    gs._check_request_inputs(subject_IDs, maze_names, days_on_maze, goal_subsets)
    goal_subsets = ["all", "subset_1", "subset_2"] if goal_subsets == "all" else goal_subsets
    if days_on_maze == "all":
        days_on_maze = list(range(1, 15))
    elif days_on_maze == "late":
        days_on_maze = list(range(5, 15))
    requested_clusters = []
    for subject in subject_IDs:
        for maze_name in maze_names:
            for day_on_maze in days_on_maze:
                goal_set = MAZE_DAY2GOALS[maze_name][str(day_on_maze)]
                if goal_set not in goal_subsets:
                    continue
                else:
                    session_date = MAZE_DAY2DATE[maze_name][str(day_on_maze)]
                    session_name = f"{session_date}.maze"
                    processed_data_path = PROCESSED_DATA_PATH / subject / session_name
                    # now search over clusters
                    cluster_metrics = load_data.load(processed_data_path / "clusters.metrics.htsv")
                    cluster_IDs = cluster_metrics.cluster_ID.to_numpy() if cluster_IDs == "all" else cluster_IDs
                    cluster_IDs = filter_clusters(
                        cluster_metrics,
                        session_info=None,
                        return_unique_IDs=False,
                        single_units=single_units,
                        multi_units=multi_units,
                        noise_units=noise_units,
                    )
                    for cluster_ID in cluster_IDs:
                        requested_clusters.append(MazeCluster(subject, session_name, cluster_ID))
    # check requested clusters have required data

    if len(requested_clusters) == 0:
        print("No clusters found matching the specified criteria")
    else:
        return requested_clusters


def filter_clusters(
    cluster_metrics,
    session_info=None,
    return_unique_IDs=False,
    single_units=True,
    multi_units=False,
    noise_units=False,
    min_average_firing_rate=0.5,
):
    """
    Filters clusters from a session based on their metrics. Currently in this mFC dataset we have limited metrics.
    Further experiments will use spikeinterface to compute more detailed quality metrics. But for now:
        single_units: defined as clusters with firing rate > min_firing_rate (default 0.5 Hz) AND Kilosort label == "good"
        multi_units: defined as clusters with firing rate > min_firing_rate (default 0.5 Hz) AND Kilosort label == "mua"
        noise_units: defined as clusters with firing rate < min_firing_rate (default 0.5 Hz) OR Kilosort label == "noise"
    """
    filtered_clusters = []
    if single_units:
        single_units = cluster_metrics[
            (cluster_metrics.KSLabel == "good") & (cluster_metrics.average_firing_rate > min_average_firing_rate)
        ]
        filtered_clusters.extend(single_units.cluster_ID.to_numpy())
    if multi_units:
        multi_units = cluster_metrics[
            (cluster_metrics.KSLabel == "mua") & (cluster_metrics.average_firing_rate > min_average_firing_rate)
        ]
        filtered_clusters.extend(multi_units.cluster_ID.to_numpy())
    if noise_units:
        noise_units = cluster_metrics[
            (cluster_metrics.KSLabel == "noise") | (cluster_metrics.average_firing_rate < min_average_firing_rate)
        ]
        filtered_clusters.extend(noise_units.cluster_ID.to_numpy())
    if return_unique_IDs:
        if session_info is None:
            raise ValueError("session_info must not be None to convert cluster_IDs to cluster_unique_IDs")
        return convert.cluster_IDs2scluster_unique_IDs(session_info, filtered_clusters)
    else:
        return filtered_clusters


# %% Main Cluster Class


class Cluster:
    """ """

    def __init__(self, subject, session_name, cluster_ID):
        """ """
        self.cluster_ID = cluster_ID
        self.processed_data_path = PROCESSED_DATA_PATH / subject / session_name
        self.analysis_data_path = ANALYSIS_DATA_PATH / subject / session_name
        session_info = load_data.load(self.processed_data_path / "session_info.json")
        self.cluster_unique_ID = convert.cluster_IDs2scluster_unique_IDs(session_info, cluster_ID)
        self.name = gs.get_session_name(session_info)
        self.date = date.fromisoformat(session_info["session_date"])
        for attr_name in [k for k in session_info.keys() if k != "date"]:
            setattr(self, attr_name, session_info[attr_name])

    def _get_tuning_feature_kwargs(self, feature, input_feature_kwargs):
        """
        Set and update defualt kwargs for loading and plotting cluster feature tuning.
        Step 1. get default kwargs
        Step 2. check input_feature_kwargs are valid
        Step 3. Update them based on input_feature_kwargs
        """
        # Step 1
        if feature == "actions":
            default_kwargs = {"smooth_SD": 5}

        elif feature == "angle_to_goal":
            default_kwargs = {
                "angle_metric": "allocentric_angle_to_goal",
                "goal_stratified": False,
                "smooth_SD": 2,
            }

        elif feature == "distance_to_goal":
            default_kwargs = {
                "metrics": ("distance_to_goal", "geodesic"),
                "bin_spacing": 0.04,
                "n_bins": 40,
                "moving_only": True,
                "exclude_time_at_goal": False,
                "goal_stratified": False,
                "smooth_SD": 1,
                "color": "black",
            }
        elif feature == "trial_events":
            default_kwargs = {"smooth_SD": 10, "color": "black", "goal_stratified": False}

        elif feature == "spatial":
            default_kwargs = {
                "navigation_only": False,
                "moving_only": True,
                "exclude_time_at_goal": False,
                "bin_size": 0.025,
                "smooth_SD": 0.025,
            }

        elif feature == "place":
            default_kwargs = {
                "navigation_only": True,
                "moving_only": True,
                "exclude_time_at_goal": False,
                "minimum_occupancy": 1,
            }

        elif feature == "place_direction":
            default_kwargs = {
                "minimum_occupancy": 0.5,
                "navigation_only": True,
                "moving_only": True,
                "exclude_time_at_goal": False,
            }

        elif feature == "head_direction":
            default_kwargs = {
                "goal_stratified": False,
                "smooth_SD": 2,
            }
        elif feature == "routes":
            default_kwargs = {
                "sequence": "future",
                "smooth_SD": 1,
                "route_max": 3,
                "route_min": 2,
                "optimal": True,
                "moving": True,
                "distance_to_goal_decreasing": True,
                "min_time_per_estimate": 0.5,
                "min_trials_per_route_shift": 2,
            }
        elif feature == "route_aligned_rates":
            default_kwargs = {
                "remove_cue_events": False,
                "optimal_only": True,
                "max_routes": 2,
                "min_routes": 2,
                "smooth_SD": 1,
                "stretch_max": 5,
                "stretch_min": 0,
            }
        else:
            raise ValueError(f"Tuning feature: {feature} not recognised")
        # Step 3: check input kwargs are valid
        for i in input_feature_kwargs.keys():
            if i not in default_kwargs.keys():
                raise ValueError(f"{i} is not a valid feature_kwarg")
        # Step 2:
        for k, v in input_feature_kwargs.items():
            default_kwargs[k] = v
        feature_kwargs = default_kwargs  # assign feature_kwargs to updated defualt kwargs
        return feature_kwargs

    def get_default_feature_kwargs(self, feature):
        """
        Returns defualt kwargs for load and plotting data associated cluster tuning to input feature
        """
        return self._get_tuning_feature_kwargs(feature, input_feature_kwargs={})

    def _print_missing_data_error(self, feature):
        return print(f"Missing analysis data to load {feature} tuning for cluster {self.cluster_unique_ID}")

    def load_tuning_data(self, feature, feature_kwargs={}):
        """ """
        # update defualt feature kwargs based on input
        feature_kwargs = self._get_tuning_feature_kwargs(feature, feature_kwargs)
        if feature == "actions":
            try:  # load data
                action_aligned_rates_df = load_data.load(self.analysis_data_path / "action_aligned_rates.parquet")
            except FileNotFoundError:
                self._print_missing_data_error(self, feature)
                return None
            # filter for specified cluster
            tuning_data = action_aligned_rates_df[
                action_aligned_rates_df.cluster_unique_ID == self.cluster_unique_ID
            ].reset_index(drop=True)
            return tuning_data

        elif feature == "angle_to_goal":
            angle_metric = feature_kwargs["angle_metric"]
            try:  # load data
                tuning_df = load_data.load(self.analysis_data_path / (angle_metric + "_tuning.parquet"))
            except FileNotFoundError:
                self._print_missing_data_error(self, feature)
                return None
            # filter for specified cluster
            tuning_df = tuning_df[tuning_df.cluster_unique_ID == self.cluster_unique_ID]
            return (tuning_df, angle_metric)

        elif feature == "distance_to_goal":
            # load_data
            metrics = feature_kwargs["metrics"]
            try:
                navigation_df = load_data.load(self.analysis_data_path / "frames.navigation.parquet")
                navigation_rates_df = load_data.load(self.analysis_data_path / "frames.spikeRates.parquet")
            except FileNotFoundError:
                print(f"Missing analysis data to load distance to goal tuning for cluster {self.cluster_unique_ID}")
                return None
            # filter data
            navigation_rates_df = navigation_rates_df.xs(self.cluster_unique_ID, level=1, axis=1).reset_index(drop=True)
            distance_info = navigation_df[[("goal", ""), ("trial", ""), ("moving", ""), metrics]].droplevel(1, axis=1)
            distance_rates_df = pd.concat([distance_info, navigation_rates_df], axis=1)
            distance_tuning_df = distance_to_goal.get_distance_to_goal_tuning_df(distance_rates_df, metrics)
            return distance_tuning_df, metrics

        elif feature == "trial_events":
            try:  # load session data
                analysis_data_structure = "trial_aligned_rates.parquet"
                trial_aligned_rates_df = load_data.load(self.analysis_data_path / analysis_data_structure)
            except FileNotFoundError:
                self._print_missing_data_error(self, feature)
                return None
            # filter for cluster
            cluster_tuning_data = trial_aligned_rates_df[
                trial_aligned_rates_df.cluster_unique_ID == self.cluster_unique_ID
            ].reset_index(drop=True)
            return cluster_tuning_data

        elif feature == "spatial":
            # load data
            try:
                session_info = load_data.load(self.processed_data_path / "session_info.json")
                simple_maze = mr.simple_maze(session_info["maze_structure"])
                navigation_df = load_data.load(self.analysis_data_path / "frames.navigation.parquet")
                navigation_spike_counts_df = load_data.load(self.analysis_data_path / "frames.spikeCounts.parquet")
                navigation_spike_counts_df = navigation_spike_counts_df.xs(
                    self.cluster_unique_ID, level=1, axis=1, drop_level=False
                ).reset_index(drop=True)
            except FileNotFoundError:
                self._print_missing_data_error(self, feature)
                return None
            navigation_activity_df = pd.concat([navigation_df, navigation_spike_counts_df], axis=1)
            # return different tuning data for plotting based on session type
            navigation_activity_df = filt.filter_navigation_rates_df(
                navigation_activity_df,
                feature_kwargs["navigation_only"],
                feature_kwargs["moving_only"],
                feature_kwargs["exclude_time_at_goal"],
            )
            # get outputs for plotting
            pos = navigation_activity_df.centroid_position.to_numpy()
            spikes = navigation_activity_df.spike_count[self.cluster_unique_ID].to_numpy().reshape(-1)
            return (pos, spikes, simple_maze)

        elif feature == "place":
            try:  # load_data
                session_info = load_data.load(self.processed_data_path / "session_info.json")
                simple_maze = mr.simple_maze(session_info["maze_structure"])
                navigation_df = load_data.load(self.analysis_data_path / "frames.navigation.parquet")
                navigation_spike_rates_df = load_data.load(self.analysis_data_path / "frames.spikeRates.parquet")
                navigation_spike_rates_df = navigation_spike_rates_df.xs(
                    self.cluster_unique_ID, level=1, axis=1, drop_level=False
                ).reset_index(drop=True)
            except FileExistsError:
                self._print_missing_data_error(self, feature)
                return None
            # process data
            navigation_rates_df = pd.concat((navigation_df, navigation_spike_rates_df.reset_index(drop=True)), axis=1)
            place_tuning_df = chm._get_place_df(simple_maze, navigation_rates_df, **feature_kwargs)
            return (simple_maze, place_tuning_df.loc[self.cluster_unique_ID])

        elif feature == "place_direction":
            try:  # load_data
                session_info = load_data.load(self.processed_data_path / "session_info.json")
                simple_maze = mr.simple_maze(session_info["maze_structure"])
                navigation_df = load_data.load(self.analysis_data_path / "frames.navigation.parquet")
                navigation_spike_rates_df = load_data.load(self.analysis_data_path / "frames.spikeRates.parquet")
                navigation_spike_rates_df = navigation_spike_rates_df.xs(
                    self.cluster_unique_ID, level=1, axis=1, drop_level=False
                ).reset_index(drop=True)
            except FileExistsError:
                self._print_missing_data_error(self, feature)
                return None
            # process data
            navigation_rates_df = pd.concat((navigation_df, navigation_spike_rates_df.reset_index(drop=True)), axis=1)
            place_direction_df = chm._get_place_direction_df(simple_maze, navigation_rates_df, **feature_kwargs)
            return (simple_maze, place_direction_df.loc[self.cluster_unique_ID])

        elif feature == "head_direction":
            try:  # load data
                head_direction_tuning_df = load_data.load(self.analysis_data_path / "head_direction_tuning.parquet")
            except FileNotFoundError:
                self._print_missing_data_error(self, feature)
                # process data
            tuning_df = head_direction_tuning_df[head_direction_tuning_df.cluster_unique_ID == self.cluster_unique_ID]
            return tuning_df

        elif feature == "routes":
            try:  # load data
                navigation_df = load_data.load(self.analysis_data_path / "frames.navigation.parquet")
                navigation_spike_rates_df = load_data.load(self.analysis_data_path / "frames.spikeRates.parquet")
                navigation_routes_df = load_data.load(self.analysis_data_path / "frames.routes.parquet")
            except FileNotFoundError:
                print(f"Missing analysis data to load route tuning for cluster {self.cluster_unique_ID}")
            # filter data
            navigation_rates_df = pd.concat([navigation_df, navigation_routes_df.reset_index(drop=True)], axis=1)
            cluster_rates = navigation_spike_rates_df.xs(self.cluster_unique_ID, level=1, axis=1).firing_rate.to_numpy()
            navigation_rates_df.loc[:, ("firing_rate", "")] = cluster_rates
            route_tuning_df = routes.get_route_sequence_tuning_df(
                navigation_rates_df,
                sequence=feature_kwargs["sequence"],
                route_max=feature_kwargs["route_max"],
                route_min=feature_kwargs["route_min"],
                optimal=feature_kwargs["optimal"],
                moving=feature_kwargs["moving"],
                distance_to_goal_decreasing=feature_kwargs["distance_to_goal_decreasing"],
                min_time_per_estimate=feature_kwargs["min_time_per_estimate"],
                min_trials_per_route_shift=feature_kwargs["min_trials_per_route_shift"],
            )
            return route_tuning_df

        elif feature == "route_aligned_rates":
            try:  # load data
                route_aligned_rates_df = load_data.load(self.analysis_data_path / "route_aligned_rates.parquet")
            except FileNotFoundError:
                print(f"Missing analysis data to load route aligned rates for cluster {self.cluster_unique_ID}")
                return None
            return route_aligned_rates_df[
                route_aligned_rates_df.cluster_unique_ID == self.cluster_unique_ID
            ].reset_index(drop=True)

        else:
            raise ValueError(f"Tuning feature: {feature} not recognised")

    def plot_tuning(self, feature, feature_kwargs={}, ax=None):
        """ """
        # get data to plot
        feature_kwargs = self._get_tuning_feature_kwargs(feature, feature_kwargs)
        tuning_data = self.load_tuning_data(feature, feature_kwargs)
        if tuning_data is None:
            raise FileNotFoundError(
                f"Cannot plot {feature} tuning. Missing processed/analysis data for cluster {self.cluster_unique_ID}"
            )
        # plot tuning feature
        if feature == "actions":
            actions.plot_action_tuning(
                tuning_data,
                axes=ax,
                smooth_SD=feature_kwargs["smooth_SD"],
            )
        elif feature == "angle_to_goal":
            angle_to_goal.plot_angle_tuning(
                *tuning_data,
                goal_stratified=feature_kwargs["goal_stratified"],
                smooth_SD=feature_kwargs["smooth_SD"],
                ax=ax,
            )
        elif feature == "distance_to_goal":
            distance_to_goal.plot_distance_tuning(
                *tuning_data,
                goal_stratified=feature_kwargs["goal_stratified"],
                smooth_SD=feature_kwargs["smooth_SD"],
                color=feature_kwargs["color"],
                ax=ax,
            )
        elif feature == "trial_events":
            events.plot_trial_aligned_rates(
                tuning_data,
                smooth_SD=feature_kwargs["smooth_SD"],
                goal_stratified=feature_kwargs["goal_stratified"],
                ax=ax,
                color=feature_kwargs["color"],
            )
        elif feature == "spatial":
            # plot spatial heatmaps dependent on session types
            spatial.plot_spatial_heatmap(
                *tuning_data,
                bin_size=feature_kwargs["bin_size"],
                smooth_SD=feature_kwargs["smooth_SD"],
                ax=ax,
            )
        elif feature == "place":
            spatial.plot_place_tuning(*tuning_data, ax=ax)
        elif feature == "place_direction":
            spatial.plot_place_direction_tuning(*tuning_data, ax=ax)
        elif feature == "head_direction":
            angle_to_goal.plot_angle_tuning(
                tuning_data,
                "head_direction",
                goal_stratified=feature_kwargs["goal_stratified"],
                smooth_SD=feature_kwargs["smooth_SD"],
                ax=ax,
            )
        elif feature == "routes":
            routes.plot_routes_tuning(tuning_data, title=feature_kwargs["sequence"], axes=ax)
        elif feature == "route_aligned_rates":
            routes.plot_route_aligned_tuning(
                tuning_data,
                remove_cue_events=feature_kwargs["remove_cue_events"],
                optimal_only=feature_kwargs["optimal_only"],
                max_routes=feature_kwargs["max_routes"],
                min_routes=feature_kwargs["min_routes"],
                smooth_SD=feature_kwargs["smooth_SD"],
                stretch_max=feature_kwargs["stretch_max"],
                stretch_min=feature_kwargs["stretch_min"],
                ax=ax,
            )
        else:
            raise ValueError(f"Tuning feature: {feature} not recognised")
        return


# %%
class MazeCluster(Cluster):
    """ """

    def __init__(self, subject, session_name, cluster_ID):
        """ """
        super().__init__(subject, session_name, cluster_ID)
        self.tuning_features = [
            "actions",
            "angle_to_goal",
            "distance_to_goal",
            "trial_events",
            "spatial",
            "place",
            "place_direction",
            "head_direction",
        ]

    def __repr__(self):
        """ """
        return f"-MazeCluster- Unique ID: {self.cluster_unique_ID}"


class RestCluster(Cluster):
    """ """

    def __init__(self, subject, session_name, cluster_ID):
        """ """
        super().__init__(subject, session_name, cluster_ID)

    def __repr__(self):
        """ """
        return f"-RestCluster- Unique ID: {self.cluster_unique_ID}"
