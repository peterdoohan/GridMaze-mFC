"""
Map from GridMaze session data to input for embedding model.
"""

# %% Imports
import json
import torch
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import OneHotEncoder

from GridMaze.maze import representations as mr

from GridMaze.analysis.core import convert
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import get_sessions as gs

from GridMaze.analysis.distance_to_goal import bases as db
from GridMaze.analysis.distance_to_goal import distributions as dd

from GridMaze.analysis.lfp import extract_lfp_phase as elp


# %% Global Variables

from ...paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

FRAME_RATE = 60  # Hz

# %% main func


def get_input_data(
    subject_IDs=["m2"],
    maze_name="maze_1",
    days_on_maze="late",
    sessions=None,
    resolution=0.1,
    max_steps_to_goal=30,
    moving_only=False,
    min_spike_count=300,
    input_groups=["distance_to_goal", "place_direction", "egocentric_action"],
    input_group_kwargs={},
    verbose=False,
):
    # load session objects
    if sessions is None:
        if verbose:
            print("Loading session objects ...")
        with_data = ["navigation_df", "navigation_spike_counts_df", "cluster_metrics"]
        if "theta_phase" in input_groups:  # note if loading lfp data for all sessions >128GB RAM req
            with_data.extend(["lfp_signal", "lfp_times", "lfp_metrics"])
        sessions = gs.get_maze_sessions(
            subject_IDs=subject_IDs,
            maze_names=[maze_name],
            days_on_maze=days_on_maze,
            with_data=with_data,
            must_have_data=True,
        )
    input_data = []
    # gen input data for each session
    for session in sessions:
        if verbose:
            print(session.name)
        session_data = get_session_input_data(
            session,
            resolution,
            max_steps_to_goal,
            moving_only,
            min_spike_count,
            input_groups,
            input_group_kwargs,
            verbose,
        )
        if session_data is not None:
            input_data.append(session_data)
    return input_data


# %% session level funcs


def get_session_input_data(
    session,
    resolution=0.2,
    max_steps_to_goal=30,
    moving_only=False,
    min_spike_count=300,
    input_groups=["distance_to_goal", "place_direction", "egocentric_action", "acceleration"],
    input_group_kwargs={"distance_to_goal": None, "place_direction": None, "egocentric_action": None},
    verbose=False,
):
    """
    only loads navigation data
    """
    input_kwargs = locals()
    input_kwargs.pop("session", None)
    # load data and update navigation variables
    df = init_navigation_spikes_df(session, input_groups)
    # downsample data
    df = ds.downsample_navigation_activity_df(df, resolution=resolution)
    # filter navigation data
    df = filt.filter_navigation_rates_df(
        df,
        navigation_only=True,
        moving_only=moving_only,
        exclude_time_at_goal=False,
        max_steps_to_goal=max_steps_to_goal,
    )
    # filter out neurons with too few spikes (non-navigation tuned)
    reject_clusters = df.spike_count.columns[df.spike_count.sum().lt(min_spike_count)]
    if len(reject_clusters) > 0:
        df = df.drop(columns=reject_clusters, level=1, axis=1)
    if "spike_count" not in df.columns.get_level_values(0):
        if verbose:
            print("No neurons meet min spike count threshold")
        return None
    # gather feature data
    X, input_group_inds, ind = [], [], 0
    for _input in input_groups:
        kwargs = input_group_kwargs[_input] if _input in input_group_kwargs.keys() else None
        x = get_input_features(df, _input, kwargs)
        X.append(x)
        Nin = x.shape[1]
        input_group_inds.append(list(range(ind, ind + Nin)))
        ind += Nin
    X = np.concatenate(X, axis=1).T
    # gather spike data
    spikes = df.spike_count.values.T
    # collect kwarg data
    session_info = {k: getattr(session, k) for k in ["name", "subject_ID", "maze_name", "day_on_maze"]}
    return {
        "X": X,  # input features
        "spikes": spikes,  # spike data
        "input_group_indices": input_group_inds,  # indices of input groups in X
        "input_group_names": input_groups,  # names of input groups
        "trial_ids": df.trial.values,  # trial ID for each sample
        "input_kwargs": input_kwargs,  # kwargs used to generate input data
        "session_info": session_info,
        "cluster_unique_IDs": df.spike_count.columns.to_list(),
    }


def init_navigation_spikes_df(session, input_features):
    # load data (single units only)
    df = session.get_navigation_activity_df(type="spikes", cluster_kwargs={"single_units": True, "multi_units": False})
    # get cleaned up speed and acceleration
    if "speed" in input_features or "acceleration" in input_features:
        _speed, _acceleration = _get_smoothed_speed_and_acceleration(df, position_smoothing_ms=1000 * 1 / FRAME_RATE)
        df[("speed", "")] = _speed
        df[("acceleration", "")] = _acceleration
        df[("moving", "")] = df.speed.ge(ds.MOVEMENT_THRESHOLD)

    # update ego-action definitions of high-frame rate data
    if "egocentric_action" in input_features:
        df = _conditional_ffill(df, [("action", "basic"), ("action", "choice_degree")], ("maze_position", "simple"))
    # add theta phase
    if "theta_phase" in input_features:
        df[("theta_phase", "")] = elp.get_nearest_theta_phase(
            session, df.time.values, signal_type="LFP", return_binned=False
        )
    return df


def _conditional_ffill(df, column_to_fill, condition_column):
    """
    Forward fill values in a specified column conditionally based on another column.
    This function performs a forward fill operation on the specified column, but resets
    the fill operation whenever there's a change in the condition column. This creates
    groups based on consecutive identical values in the condition column, and only fills
    values within these groups.
    """
    _df = df.copy()
    fill_column = _df[column_to_fill]
    group = (_df[condition_column] != _df[condition_column].shift()).cumsum()
    _df[column_to_fill] = fill_column.groupby(group).ffill()
    return _df


def _get_smoothed_speed_and_acceleration(df, position_smoothing_ms=1000 * 1 / FRAME_RATE):
    _df = df.copy()
    positions = _df.centroid_position.values
    smoothed_positions = gaussian_filter1d(positions, position_smoothing_ms / 1000 * FRAME_RATE, axis=0)
    velocities = np.gradient(smoothed_positions, axis=0) * FRAME_RATE
    accelerations = np.gradient(velocities, axis=0) * FRAME_RATE
    speeds = np.linalg.norm(velocities, axis=1)
    ## Calculate tangential acceleration via finite differences on smoothed speed.
    # Compute tangential acceleration
    vel_minus_acc = velocities - accelerations
    angles = np.arctan2(vel_minus_acc[:, 1], vel_minus_acc[:, 0])
    tangential_acc = np.sin(np.pi / 2 - angles) * np.linalg.norm(accelerations, axis=1)
    return speeds, tangential_acc


# %% regressor level funcs


def get_input_features(df, input_feature, input_kwargs):
    # main variables of interest
    if input_feature == "distance_to_goal":
        x = _get_distance_to_goal_regressors(df, **(input_kwargs or {}))
    elif input_feature == "place_direction":
        x = _get_place_direction_regressors(df, regressor="place_direction")
    elif input_feature == "egocentric_action":
        x = _get_egocentric_action_regressors(df, **(input_kwargs or {}))

    # derivative variables
    elif input_feature == "place_direction_distance_to_goal_egocentric_action":
        x = _get_place_direction_distance_to_goal_egocentric_action_regressors(df, **(input_kwargs or {}))
    elif input_feature == "place":
        x = _get_place_direction_regressors(df, regressor="place")
    elif input_feature == "direction":
        x = _get_place_direction_regressors(df, regressor="direction")

    # other cognitive variables
    elif input_feature == "goal":
        x = _get_goal_regressors(df)
    elif input_feature == "egocentric_angle_to_goal":
        x = _get_angle_to_goal_regressors(df, metric="egocentric")
    elif input_feature == "allocentric_angle_to_goal":
        x = _get_angle_to_goal_regressors(df, metric="allocentric")

    # low level variables
    elif input_feature == "speed":
        x = _get_speed_regressors(df)
    elif input_feature == "acceleration":
        x = _get_acceleration_regressor(df)
    elif input_feature == "head_direction":
        x = _get_head_direction_regressors(df)
    elif input_feature == "theta_phase":
        x = _get_theta_regressors(df)
    else:
        raise ValueError(f"Unknown input feature: {input_feature}")
    return x


def _get_distance_to_goal_regressors(
    df,
    method="onehot",
    metric=("distance_to_goal", "geodesic"),
    bin_method="uniform",
    bin_spacing=0.05,
    max_distance=None,
    n_log_bins=30,
    n_bases=10,
    basis_type="gamma",
):
    """ """
    _df = df.copy()
    if max_distance is None:
        _max = dd.get_distance_percentile(metric, 0.9)
    if method == "basis_functions":
        # see analysis.distance_to_goal.bases for more info
        if metric[0] == "distance_to_goal":
            basis_fn = db.distance_basis_generator(
                n_bases=n_bases,
                basis=basis_type,
                btype="distance",
                max_distance=_max,
            )
        elif metric[0] == "progress_to_goal":
            basis_fn = db.distance_basis_generator(
                n_bases=n_bases,
                basis=basis_type,
                btype="progress",
            )
        regressors = basis_fn(_df[metric].values)  # n_samples x n_bases
    elif method == "onehot":
        if metric[0] == "distance_to_goal":
            if bin_method == "uniform":
                n_bins = int(_max / bin_spacing)
            elif bin_method == "log":
                n_bins = n_log_bins
        bins = convert._get_distance_bins(
            binning_method=bin_method,
            n_distance_bins=n_bins,
            distance_metrics=metric,
            max_distance=_max,
        )
        binned_distances = pd.cut(_df[metric], bins=bins, include_lowest=True).to_numpy()
        # convert to one-hot encoding
        regressors = convert.dist_bin2onehot(binned_distances, _max, n_bins, metric, "uniform")  # n_samples x n_bins
    else:
        raise ValueError(f"Unknown method {method} for building distance to goal regressors")
    return regressors


def _get_place_direction_regressors(df, regressor="place_direction"):
    """ """
    _df = df.copy()
    maze_name = df.maze_name.unique()[0]
    simple_maze = mr.get_simple_maze(maze_name)
    if regressor == "place_direction":
        pd_by_frame = (  # convert to unique string eg, "A1_N", expected by onehot encoder
            _df[("maze_position", "simple")] + "_" + _df[("cardinal_movement_direction", "")]
        )
        regressors = convert.place_direction2onehot(pd_by_frame.values, simple_maze)  # n_samples x n_place_directions
    elif regressor == "place":
        regressors = convert.place2onehot(_df.maze_position.simple.values, simple_maze)
    elif regressor == "direction":
        regressors = convert.direction2onehot(_df.cardinal_movement_direction.values)
    else:
        raise ValueError(f"Unknown regressor {regressor} for building place_/_direction regressors")
    return regressors


def _get_egocentric_action_regressors(
    df,
    components=["action", "free_forced", "tower_bridge"],
    actions=["turn_left", "turn_right", "go_forward", "go_back"],
):
    """ """
    _df = df.copy()
    X = []
    if "action" in components:
        a = _df[("action", "basic")].values
        X.append(convert.action2onehot(a, actions=actions))  # n_samples x n_actions
    if "free_forced" in components:
        ff = _df[("action", "choice_degree")].map({1: "forced", 2: "forced", 3: "free", 4: "free"}).values
        X.append(convert.free_forced2onehot(ff))  # n_samples x 2
    if "tower_bridge" in components:
        tb = _df.maze_position.simple
        X.append(convert.place2tower_bridge_onehot(tb))  # n_samples x 2
    if len(components) > 1:
        return np.hstack(X)
    else:
        return X[0]


def _get_goal_regressors(df):
    _df = df.copy()
    return convert.goal2onehot(_df.goal.values)


def _get_angle_to_goal_regressors(df, metric="egocentric"):
    angles_deg = df.angle_to_goal[metric].values  # angles in degrees
    angles_rad = np.deg2rad(angles_deg)  # convert to radians
    regressors = np.column_stack([np.sin(angles_rad), np.cos(angles_rad)])  # n_samples x 2
    return regressors


def _get_speed_regressors(df):
    return df.speed.values.reshape(-1, 1)


def _get_acceleration_regressor(df):
    _df = df.copy()
    return _df.acceleration.values.reshape(-1, 1)  # n_samples x 1


def _get_head_direction_regressors(df):
    angle_deg = df.head_direction.value
    angle_rad = np.deg2rad(angle_deg)  # convert to radians
    regressors = np.column_stack([np.sin(angle_rad), np.cos(angle_rad)])
    return regressors


def _get_theta_regressors(df):
    angle_rad = df.theta_phase.values  # angles in radians
    regressors = np.column_stack([np.sin(angle_rad), np.cos(angle_rad)])
    return regressors


def _get_place_direction_distance_to_goal_egocentric_action_regressors(
    df,
    actions=["turn_left", "turn_right", "go_forward", "go_back"],
    distance_metric=("distance_to_goal", "geodesic"),
    max_distance=None,
    bin_method="uniform",
    bin_spacing=0.05,
    n_log_bins=30,
    keep_only_visited=False,
):
    """onehot encoding over place_direction, distance_to_goal, and egocentric_action"""
    maze_name = df.maze_name.unique()[0]
    _df = df.copy()
    _df[("free_forced", "")] = (
        _df[("action", "choice_degree")].map({1: "forced", 2: "forced", 3: "free", 4: "free"}).values
    )
    if max_distance is None:
        _max = dd.get_distance_percentile(distance_metric, 0.9)
    if distance_metric[0] == "distance_to_goal":
        if bin_method == "uniform":
            n_bins = int(_max / bin_spacing)
        elif bin_method == "log":
            n_bins = n_log_bins
        bins = convert._get_distance_bins(
            binning_method=bin_method,
            n_distance_bins=n_bins,
            distance_metrics=distance_metric,
            max_distance=_max,
        )
        bin2bin_id = {v: k for k, v in enumerate(bins)}
        bin2bin_id[np.nan] = 0
        binned_distances = pd.cut(_df[distance_metric], bins=bins, include_lowest=True)
        bin_ids = binned_distances.map(bin2bin_id).values.astype(int)  # n_samples x 1

    labels = (
        _df.maze_position.simple
        + "."
        + _df.cardinal_movement_direction
        + "."
        + bin_ids.astype(str)
        + "."
        + _df.action.basic.astype(str)
        + "."
        + _df.free_forced.astype(str)
    )
    if keep_only_visited:
        cats = labels.unique()
    else:
        simple_maze = mr.get_simple_maze(maze_name)
        all_place_direction_pairs = mr.get_maze_place_direction_pairs(simple_maze)
        cats = []
        for _pd in all_place_direction_pairs:
            for distance_bin_id in bin2bin_id.values():
                for action in actions:
                    for ff in ["forced", "free"]:
                        cats.append(f"{_pd[0]}.{_pd[1]}.{distance_bin_id}.{action}.{ff}")
                cats.append(f"{_pd[0]}.{_pd[1]}.{distance_bin_id}.nan.nan")
        cats = list(np.unique(cats))  # ensure unique categories
    # convert to one-hot encoding
    enc = OneHotEncoder(categories=[cats], sparse_output=False, handle_unknown="ignore")

    onehot = enc.fit_transform(labels.values.reshape(-1, 1))
    return onehot  # n_samples x n_place_directions * n_distance_bins * n_actions * n_free_forced
