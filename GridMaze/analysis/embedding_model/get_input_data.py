"""
Library for extracting input data for embedding model from GridMaze processed & analysis data
"""

# %% Imports
import json
import torch
import numpy as np
import pandas as pd

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import convert
from scipy.ndimage import gaussian_filter1d

from GridMaze.maze import representations as mr
from GridMaze.analysis.distance_to_goal import distributions as dd


# %% Global Variables
from ...paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

FRAME_RATE = 60  # Hz


# %% Main Function


def get_input_data(
    subject_IDs=["m2"],
    maze_name="maze_1",
    days_on_maze="late",
    input_features=["distance", "place_direction"],
    distance_metrics=("distance_to_goal", "geodesic"),
    include_multi_unit=False,
    navigation_only=True,
    moving_only=True,
    resolution=0.1,  # s
    max_distance=None,  # m
    max_steps_to_goal=30,
    n_distance_bins=20,
    distance_bin_method="uniform",
    min_spike_count=0,
    start_ind=0,
    end_ind=None,
):
    """"""
    # update inputs
    subject_IDs = SUBJECT_IDS if subject_IDs == "all" else subject_IDs
    # load data from disk
    sessions = gs.get_maze_sessions(
        subject_IDs=subject_IDs,
        maze_names=[maze_name],
        days_on_maze=days_on_maze,
        with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "navigation_routes_df"],
        must_have_data=True,
    )
    if end_ind is None:
        end_ind = len(sessions)
    sessions = sessions[start_ind:end_ind]
    # process each sessions data to format expected by embedding model
    sessions_input_data = []
    ind = 0
    for session in sessions:
        session_input_data = get_session_input_data(
            session,
            input_features=input_features,
            distance_metrics=distance_metrics,
            include_multi_unit=include_multi_unit,
            navigation_only=navigation_only,
            moving_only=moving_only,
            resolution=resolution,
            max_distance=max_distance,
            max_steps_to_goal=max_steps_to_goal,
            n_distance_bins=n_distance_bins,
            distance_bin_method=distance_bin_method,
            min_spike_count=min_spike_count,
        )
        if session_input_data is None:
            continue
        # index neurons
        n_clusters = session_input_data["spikes"].shape[0]
        session_input_data["cluster_inds"] = torch.from_numpy(np.arange(ind, ind + n_clusters)).to(torch.int32)
        ind += n_clusters
        # append data
        sessions_input_data.append(session_input_data)

    return sessions_input_data  # list of input_data dicts


# %% Process session level data


def get_session_input_data(
    session,
    input_features=["distance", "place_direction"],
    distance_metrics=("distance_to_goal", "geodesic"),
    include_multi_unit=False,
    navigation_only=True,
    moving_only=True,
    resolution=0.1,  # s
    max_distance=None,  # m
    max_steps_to_goal=30,
    n_distance_bins=20,
    distance_bin_method="uniform",
    min_spike_count=0,
):
    """
    Extracts input data for embedding model from session object.
    """
    # define input kwargs dict to return later
    input_kwargs = {
        "subject": session.subject_ID,
        "session_name": session.name,
        "maze_name": session.maze_name,
        "day_on_maze": session.day_on_maze,
        "input_types": input_features,
        "distance_metrics": distance_metrics,
        "include_multi_unit": include_multi_unit,
        "navigation_only": navigation_only,
        "moving_only": moving_only,
        "resolution": resolution,
        "max_distance": max_distance,
        "max_steps_to_goal": max_steps_to_goal,
        "n_distance_bins": n_distance_bins,
        "distance_bin_method": distance_bin_method,
        "min_spike_count": min_spike_count,
    }
    # updates for special input features
    with_routes = True if "current_route" in input_features or "next_route" in input_features else False
    navigation_only = False if "trial_phase" in input_features else navigation_only
    # load data from session obj
    navigation_spikes_df = session.get_navigation_activity_df(
        type="spikes",
        cluster_kwargs={"single_units": True, "multi_units": include_multi_unit},
        with_routes=with_routes,
    )
    # if left-right turns are included, do the convolution pre-downsample
    if "left_right_turns" in input_features or "free_forced_choice" in input_features:
        navigation_spikes_df = add_egocentric_regressors(navigation_spikes_df)
    # downsample data from frame-by-frame to specified resolution
    if resolution:
        navigation_spikes_df = _downsample_navigation_spike_counts(navigation_spikes_df, resolution)
    # filter data based on input kwargs
    navigation_spikes_df = filt.filter_navigation_rates_df(
        navigation_spikes_df,
        navigation_only=navigation_only,
        moving_only=moving_only,
        exclude_time_at_goal=False,
        max_steps_to_goal=max_steps_to_goal,
    )
    # filter based on number of spikes (after all the other filters)
    spike_count = navigation_spikes_df["spike_count"]
    too_few_spikes = (
        np.where(spike_count.sum(0) < min_spike_count)[0] + navigation_spikes_df.shape[1] - spike_count.shape[1]
    )
    navigation_spikes_df.drop(navigation_spikes_df.columns[too_few_spikes], axis=1, inplace=True)
    # if no neurons left, return None
    if "spike_count" not in navigation_spikes_df.columns.get_level_values(0):
        return None
    # build concatenated input matrix
    X, X_type_inds, ind = [], [], 0
    for _input in input_features:
        # note: input_feature can be onehot or scalar
        input_feature = _get_input_feature(_input, input_kwargs, navigation_spikes_df)
        Nin = input_feature.shape[1]  # number of new input dimensions
        X.append(input_feature)  # add to input data
        X_type_inds.append(torch.arange(ind, ind + Nin))
        ind += Nin  # update indices

    # get neural data
    spike_counts = navigation_spikes_df.spike_count.values
    cluster_unique_IDs = navigation_spikes_df.spike_count.columns.values
    trial_ids = navigation_spikes_df.trial.values
    return {  # organise output into dict expected by model
        "X": torch.from_numpy(np.concatenate(X, axis=1).T).to(torch.float32),
        "spikes": torch.from_numpy(spike_counts.T).to(torch.float32),
        "X_type_inds": [x.tolist() for x in X_type_inds],  # convert from torch to list (save out in .json eventually)
        "input_kwargs": input_kwargs,
        "input_feature_names": input_features,
        "cluster_unique_IDs": cluster_unique_IDs,
        "subject_ID": session.subject_ID,  # useful metadata
        "session_name": session.name,
        "maze_name": session.maze_name,
        "day_on_maze": session.day_on_maze,
        "trial_ids": trial_ids,
    }


# %% Supporting functions


def _get_input_feature(input_feature, input_kwargs, navigation_spikes_df):
    """ """
    if input_feature == "trial_phase":
        return convert.trial_phase2onehot(navigation_spikes_df.trial_phase.values)

    elif input_feature == "distance":  # distance to goal
        # load feature kwargs
        navigation_spikes_df, distance_bins_col, max_distance, n_distance_bins, distance_metrics, binning_method = (
            _add_distance_bins(navigation_spikes_df, input_kwargs)
        )
        return convert.dist_bin2onehot(
            navigation_spikes_df[distance_bins_col].values,
            max_distance,
            n_distance_bins,
            distance_metrics,
            binning_method,
        )

    elif input_feature == "place_direction":
        simple_maze = mr.get_simple_maze(input_kwargs["maze_name"])
        pd_by_frame = (  # convert to unique string eg, "A1_N", expected by onehot encoder
            navigation_spikes_df[("maze_position", "simple")]
            + "_"
            + navigation_spikes_df[("cardinal_movement_direction", "")]
        )
        return convert.place_direction2onehot(pd_by_frame.values, simple_maze)

    elif input_feature == "place_direction_distance":
        navigation_spikes_df, distance_bins_col, max_distance, n_distance_bins, distance_metrics, binning_method = (
            _add_distance_bins(navigation_spikes_df, input_kwargs)
        )
        simple_maze = mr.get_simple_maze(input_kwargs["maze_name"])
        pdd_by_frame = (  # convert to unique string eg, "A1_N_0.00", expected by onehot encoder
            navigation_spikes_df[("maze_position", "simple")]
            + "_"
            + navigation_spikes_df[("cardinal_movement_direction", "")]
            + "_"
            + navigation_spikes_df[distance_bins_col].apply(lambda x: f"{x.mid:.2f}").astype(str)
        )  # convert to unique string eg, "A1_N", expected by onehot encoder
        return convert.place_direction_distance2onehot(
            pdd_by_frame.values, simple_maze, max_distance, n_distance_bins, distance_metrics, binning_method
        )

    elif input_feature == "current_route":
        return convert.route_id2onehot(navigation_spikes_df.route["r"])

    elif input_feature == "next_route":
        return convert.route_id2onehot(navigation_spikes_df.route["r+1"])

    elif input_feature == "place":
        simple_maze = mr.get_simple_maze(input_kwargs["maze_name"])
        return convert.place2onehot(navigation_spikes_df.maze_position.simple.values, simple_maze)

    elif input_feature == "direction":
        return convert.direction2onehot(navigation_spikes_df.cardinal_movement_direction.values)

    elif input_feature == "tower_bridge":  # onehot ["tower", "bridge"]
        return convert.place2tower_bridge_onehot(navigation_spikes_df.maze_position.simple)

    elif input_feature == "speed":  # continous
        return navigation_spikes_df.speed.values.reshape(-1, 1)

    elif input_feature == "acceleration":  # continous
        v = navigation_spikes_df.velocity.values
        return np.diff(v, axis=0, prepend=[v[0]]) * (
            1 / input_kwargs["resolution"]
        )  # prepend first value to maintain shape

    elif input_feature == "head_direction":  # continous (degrees)
        hd = navigation_spikes_df.head_direction.value.values
        return hd.reshape(-1, 1)

    elif input_feature == "goal":
        return convert.goal2onehot(navigation_spikes_df.goal.values)
    elif input_feature == "trial":
        return navigation_spikes_df.trial.values.reshape(-1, 1)
    elif input_feature == "left_right_turns":
        return navigation_spikes_df.egocentric_regressors[["turn_left", "turn_right"]].values  # [n_frames, 2]
    elif input_feature == "free_forced_choice":
        return navigation_spikes_df.egocentric_regressors[["free", "forced"]].values  # [n_frames, 2]
    elif input_feature == "egocentric_angle_to_goal":
        return navigation_spikes_df.angle_to_goal.egocentric.apply(np.deg2rad).values.reshape(
            -1, 1
        )  # rads [n_frames, 1]
    elif input_feature == "allocentric_angle_to_goal":
        return navigation_spikes_df.angle_to_goal.allocentric.apply(np.deg2rad).values.reshape(
            -1, 1
        )  # rads [n_frames, 1]
    else:
        print(f"Input type: {input_feature} not recognised.")
        raise NotImplementedError(
            "Input type must be in ['distance', 'place', 'direction', 'place_direction', 'current_route', 'next_route', 'tower_bridge ', 'speed', 'acceleration', 'trial_phase', 'head_direction']"
        )


# %% subfunctions


def add_egocentric_regressors(navigation_spikes_df):
    """
    Returns regressors for left-right turns defined only over towers
    (except for actions at defined goal).
    """
    # fill forward each action to the next bridge (note actions are defined as first frame entry to a new tower)
    navigation_spikes_df = _conditional_ffill(
        navigation_spikes_df, [("action", "basic"), ("action", "choice_degree")], ("maze_position", "simple")
    )
    # tranform left-right turns to onehot
    action_labels = ["turn_left", "turn_right"]
    actions_array = np.array([(navigation_spikes_df.action.basic.values == e).astype(float) for e in action_labels]).T
    LR_df = pd.DataFrame(actions_array, columns=pd.MultiIndex.from_product([["egocentric_regressors"], action_labels]))
    # transform forced/free choices to onehot
    choice_labels = ["free", "forced"]
    choice = navigation_spikes_df.action.choice_degree.map(
        {1: "forced", 2: "forced", 3: "free", 4: "free"}, na_action="ignore"
    )
    choice_array = np.array([(choice == e).astype(float) for e in choice_labels]).T
    choice_df = pd.DataFrame(
        choice_array, columns=pd.MultiIndex.from_product([["egocentric_regressors"], choice_labels])
    )
    regressors_df = pd.concat([LR_df, choice_df], axis=1)
    # map choices at goal locations times at goal to 0
    at_goal_mask = navigation_spikes_df.goal == navigation_spikes_df.maze_position.simple
    regressors_df.loc[at_goal_mask] = 0
    return pd.concat([navigation_spikes_df, regressors_df], axis=1)


def _conditional_ffill(df, column_to_fill, condition_column):
    _df = df.copy()
    fill_column = _df[column_to_fill]
    group = (_df[condition_column] != _df[condition_column].shift()).cumsum()
    _df[column_to_fill] = fill_column.groupby(group).ffill()
    return _df


def _add_distance_bins(navigation_spikes_df, input_kwargs):
    distance_metrics = input_kwargs["distance_metrics"]
    max_distance = input_kwargs["max_distance"]
    n_distance_bins = input_kwargs["n_distance_bins"]
    binning_method = input_kwargs["distance_bin_method"]
    # bin continous distance to goal info and turn into onehot
    bins = convert._get_distance_bins(binning_method, n_distance_bins, distance_metrics, max_distance)
    distance_bins_col = (distance_metrics[0], distance_metrics[1] + "_binned")

    navigation_spikes_df[distance_bins_col] = pd.cut(
        navigation_spikes_df[(distance_metrics[0], distance_metrics[1])],
        bins=bins,
    )
    return navigation_spikes_df, distance_bins_col, max_distance, n_distance_bins, distance_metrics, binning_method


def _downsample_navigation_spike_counts(navigation_spike_counts_df, window_length):
    """
    Reduces resolution of navigation_spike_counts_df, that natively expresses spike counts per frame, to spike counts
    per window (defined in seconds). Note the resultant dataframe counts spike in non-overlapping blocks and for convience
    takes the place, direction, distance_to_goal (and other variables) for the mid window index of the original data structure.
    This could introduce errors where eg, positions change within a window (don't worry about this for now).
    """
    combine_n_frames = int(FRAME_RATE * window_length)
    nav_info_df = navigation_spike_counts_df.drop(columns="spike_count", level=0, axis=0)
    spike_counts_df = navigation_spike_counts_df.xs("spike_count", level=0, axis=1, drop_level=False)
    ds_spike_counts_df = spike_counts_df.groupby(spike_counts_df.index // combine_n_frames).sum()
    ds_spike_counts_df.reset_index(drop=True, inplace=True)
    mid_window_indicies = (spike_counts_df.index // combine_n_frames).unique() * combine_n_frames + (
        combine_n_frames // 2
    )
    mid_window_indicies = mid_window_indicies[mid_window_indicies < len(nav_info_df)]
    ds_nav_info_df = nav_info_df.iloc[mid_window_indicies]
    ds_nav_info_df.reset_index(drop=True, inplace=True)
    if ds_nav_info_df.shape[1] < ds_spike_counts_df.shape[1]:
        ds_spike_counts_df = ds_spike_counts_df.iloc[:-1]
    return pd.concat([ds_nav_info_df, ds_spike_counts_df], axis=1)
