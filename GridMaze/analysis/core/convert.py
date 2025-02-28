"""File for converting data representations"""

# %% Imports
import json
import pandas as pd
import regex as re
import numpy as np
import itertools

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.maze import representations as mr
from GridMaze.analysis.distance_to_goal import distributions as dd
from sklearn.preprocessing import OneHotEncoder
from collections.abc import Iterable

# %% Global Variables
from ...paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "maze_configs.json", "r") as input_file:
    MAZE_CONFIGS = json.load(input_file)

ALL_GOALS = MAZE_CONFIGS["maze_1"]["goal_sets"][
    "all"
]  # we used a 24/49 towers as goal locations in this exp (same across mazes)

# %% Convert unique IDs


def cluster_unique_IDs2cluster_IDs(cluster_unique_IDs):
    if not isinstance(cluster_unique_IDs, Iterable):
        return _reverse_cluster_unique(cluster_unique_IDs)
    else:
        return np.array([_reverse_cluster_unique(c) for c in cluster_unique_IDs])


def _reverse_cluster_unique(c):
    return int(re.search(r"cluster(\d+)$", c).group(1))


def cluster_IDs2scluster_unique_IDs(session_info, cluster_IDs):
    session_name = gs.get_session_name(session_info)
    if not isinstance(cluster_IDs, Iterable):
        return f"{session_name}_cluster{int(cluster_IDs)}"
    else:
        return [f"{session_name}_cluster{int(cID)}" for cID in cluster_IDs]


def trial2trial_unique_ID(session_info, trials):
    session_name = gs.get_session_name(session_info)
    if not isinstance(trials, Iterable):
        if np.isnan(trials):
            return trials
        else:
            return f"{session_name}_trial{int(trials)}"
    else:
        return [f"{session_name}_trial{int(t)}" if not np.isnan(t) else t for t in trials]


# %% Convert to OneHots


def trial_phase2onehot(tp, trial_phases=["navigation", "ITI", "reward_consumption"]):
    enc = OneHotEncoder(categories=[trial_phases], sparse_output=False, handle_unknown="ignore")
    onehot = enc.fit_transform(tp.reshape(-1, 1))
    return onehot


def dist_bin2onehot(bins_by_frame, max_distance, n_distance_bins, distance_metrics, binning_method):
    bins = _get_distance_bins(binning_method, n_distance_bins, distance_metrics, max_distance)
    enc = OneHotEncoder(categories=[bins], sparse_output=False, handle_unknown="ignore")
    onehot = enc.fit_transform(bins_by_frame.reshape(-1, 1))
    return onehot


def _get_distance_bins(binning_method, n_distance_bins, distance_metrics, max_distance):
    if max_distance is None:
        max_distance = dd.get_distance_percentile(distance_metrics, 1)
    if binning_method == "uniform":
        bins = pd.interval_range(start=0, end=max_distance, freq=max_distance / n_distance_bins, closed="left")
    elif binning_method == "non-uniform":
        bins = dd.bin_distribution_evenly(distance_metrics, n_distance_bins, max_distance=max_distance)
        bins = pd.IntervalIndex.from_breaks(bins, closed="left")
    return bins


def place_direction2onehot(pd_by_frame, simple_maze):
    all_place_direction_pairs = mr.get_maze_place_direction_pairs(simple_maze)
    all_place_direction_pairs = np.array(
        [x[0] + "_" + x[1] for x in all_place_direction_pairs], dtype=object
    )  # transform to unique string from tuples
    enc = OneHotEncoder(categories=[all_place_direction_pairs], sparse_output=False, handle_unknown="ignore")
    onehot = enc.fit_transform(pd_by_frame.reshape(-1, 1))
    return onehot


def place_direction_distance2onehot(
    pdd_by_frame, simple_maze, max_distance, n_distance_bins, distance_metrics, binning_method
):
    all_place_direction_pairs = mr.get_maze_place_direction_pairs(simple_maze)
    all_place_direction_pairs = np.array(
        [x[0] + "_" + x[1] for x in all_place_direction_pairs], dtype=object
    )  # transform to unique string from tuples
    bins = _get_distance_bins(binning_method, n_distance_bins, distance_metrics, max_distance)
    pdd = [f"{p}_{b.mid:.2f}" for p in all_place_direction_pairs for b in bins]
    enc = OneHotEncoder(categories=[pdd], sparse_output=False, handle_unknown="ignore")
    onehot = enc.fit_transform(pdd_by_frame.reshape(-1, 1))
    return onehot


def route_id2onehot(r, n_routes=10):
    """ """
    route_ids = ["non_route"] + [f"route_{i}" for i in range(n_routes)]
    enc = OneHotEncoder(categories=[route_ids], sparse_output=False, handle_unknown="ignore")
    onehot = enc.fit_transform(np.array(r).reshape(-1, 1))
    return onehot


def place2onehot(p, simple_maze):
    all_places = mr.get_maze_locations(simple_maze)
    enc = OneHotEncoder(categories=[all_places], sparse_output=False, handle_unknown="ignore")
    onehot = enc.fit_transform(p.reshape(-1, 1))
    return onehot


def direction2onehot(d, directions=["N", "E", "S", "W"]):
    enc = OneHotEncoder(categories=[directions], sparse_output=False, handle_unknown="ignore")
    onehot = enc.fit_transform(d.reshape(-1, 1))
    return onehot


def place2tower_bridge_onehot(p):
    tower_bridge = p.apply(lambda x: len(x.split("-")))
    tower_bridge = tower_bridge.map({1: "tower", 2: "bridge"}).values
    enc = OneHotEncoder(categories=[["tower", "bridge"]], sparse_output=False, handle_unknown="ignore")
    onehot = enc.fit_transform(tower_bridge.reshape(-1, 1))
    return onehot


def goal2onehot(g, goals=ALL_GOALS):
    enc = OneHotEncoder(categories=[goals], sparse_output=False, handle_unknown="ignore")
    onehot = enc.fit_transform(g.reshape(-1, 1))
    return onehot


def egocentric_action2onehot(ea, include_egocentric_choice_degree):
    ego_actions = ["go_forward", "go_back", "turn_left", "turn_right"]
    if include_egocentric_choice_degree:  # unique string for each action eg. 'go_forward_2.0'
        categories = [("go_back" + "_" + str(1.0))] + [
            x[0] + "_" + str(x[1]) for x in itertools.product(ego_actions, [2.0, 3.0, 4.0])
        ]
    else:
        categories = ego_actions
    enc = OneHotEncoder(categories=[categories], sparse_output=False, handle_unknown="ignore")
    onehot = enc.fit_transform(ea.reshape(-1, 1))
    return onehot
