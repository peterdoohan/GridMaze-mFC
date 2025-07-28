"""
Map from GridMaze session data to input for embedding model.
"""

# %% Imports
import json
import torch
import numpy as np
import pandas as pd

from GridMaze.maze import representations as mr

from GridMaze.analysis.core import convert
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import get_sessions as gs

from GridMaze.analysis.distance_to_goal import bases as db
from GridMaze.analysis.distance_to_goal import distributions as dd


# %% Global Variables

from ...paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

FRAME_RATE = 60  # Hz

# %% main func


def get_input_data(
    subject_IDs=["m2"],
    maze_name="maze_1",
    days_on_maze="all",
    sessions=None,
    resolution=0.1,
    max_steps_to_goal=30,
    moving_only=False,
    min_spike_count=300,
    input_features=["distance_to_goal", "place_direction", "egocentric_action"],
    input_feature_kwargs={"distance_to_goal": None, "place_direction": None, "egocentric_action": None},
    verbose=False,
):
    # load session objects
    if sessions is None:
        if verbose:
            print("Loading session objects ...")
        sessions = gs.get_maze_sessions(
            subject_IDs=subject_IDs,
            maze_names=[maze_name],
            days_on_maze=days_on_maze,
            with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics"],
            must_have_data=True,
        )

    # convert to embedding model input format
    session_ind = 0
    for session in sessions:
        pass
    return


# %% session level funcs


def get_session_input_data(
    session,
    resolution=0.2,
    max_steps_to_goal=30,
    moving_only=False,
    min_spike_count=300,
    input_features=["distance_to_goal", "place_direction", "egocentric_action"],
    input_feature_kwargs={"distance_to_goal": None, "place_direction": None, "egocentric_action": None},
    verbose=False,
):
    """
    only loads navigation data
    """
    # load data (single units only)
    df = session.get_navigation_activity_df(type="spikes", cluster_kwargs={"single_units": True, "multi_units": False})
    # update ego-action definitions of high-frame rate data
    df = _conditional_ffill(df, [("action", "basic"), ("action", "choice_degree")], ("maze_position", "simple"))
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
    X, X_type_inds, ind = [], [], 0
    for _input in input_features:
        kwargs = input_feature_kwargs[_input] if _input in input_feature_kwargs.keys() else None
        x = get_input_features(df, _input, kwargs)
        X.append(x)
        Nin = x.shape[1]
        X_type_inds.append(list(np.arange(ind, ind + Nin)))
        ind += Nin
    X = torch.from_numpy(np.concatenate(X, axis=1).T).to(torch.float32)
    # gather spike data
    spikes = torch.from_numpy(df.spike_counts.values.T).to(torch.float32)
    # collect kwarg data
    input_kwargs = locals()
    input_kwargs.pop("session", None)
    session_info = {k: getattr(session, k) for k in ["name", "subject_ID", "maze_name", "days_on_maze"]}
    session_info["cluster_unique_IDs"] = df.spike_counts.columns.get_level_values(1).to_list()
    return {
        "X": X,  # input features
        "spikes": spikes,  # spike data
        "X_type_inds": X_type_inds,  # indices of input features in X
        "input_kwargs": input_kwargs,
        "session_info": session_info,
    }


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


# %% regressor level funcs


def get_input_features(df, input_feature, input_kwargs):
    # main variables of interest
    if input_feature == "distance_to_goal":
        raise NotImplementedError
    elif input_feature == "place_direction":
        raise NotImplementedError
    elif input_feature == "egocentric_action":
        raise NotImplementedError

    # derivate variables
    elif input_feature == "place_direction_distance_to_goal_egocentric_action":
        raise NotImplementedError
    elif input_feature == "place":
        raise NotImplementedError
    elif input_feature == "direction":
        raise NotImplementedError

    # other cognitive varaibles
    elif input_feature == "goal":
        raise NotImplementedError
    elif input_feature == "egocentric_angle_to_goal":
        raise NotImplementedError
    elif input_feature == "allocentric_angle_to_goal":
        raise NotImplementedError

    # low level variables
    elif input_feature == "speed":
        raise NotImplementedError
    elif input_feature == "acceleration":
        raise NotImplementedError
    elif input_feature == "head_direction":
        raise NotImplementedError
    elif input_feature == "theta":
        raise NotImplementedError

    return


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
        basis_activations = basis_fn(_df[metric].values)  # n_samples x n_bases
        return basis_activations
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
        onehot = convert.dist_bin2onehot(binned_distances, _max, n_bins, metric, "uniform")
        return onehot  # n_samples x n_bins
    else:
        raise ValueError(f"Unknown method {method} for building distance to goal regressors")


def _get_place_direction_regressors():
    return


def _get_egocentric_action_regressors():
    return


def _get_goal_regressors():
    return


def _get_angle_to_goal_regressors():
    return


def _get_speed_regressors():
    return


def _get_acceleration_regressors():
    return


def _get_head_direction_regressors():
    return


def _get_theta_regressors():
    return
