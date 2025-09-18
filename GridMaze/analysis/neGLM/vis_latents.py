"""
Lib for visualising tuning of latent units in nbeGLM models
@peterdoohan @krisjensen
"""

# %% Imports
import inspect
import torch
import pandas as pd
from matplotlib import pyplot as plt

from GridMaze.maze import representations as mr
from GridMaze.maze import plotting as mp
from GridMaze.analysis.nbeGLM import load_model_sets as lms
from GridMaze.analysis.nbeGLM import get_input_data as gid
from GridMaze.analysis.distance_to_goal import distributions as dd
from GridMaze.analysis.core import convert

# %% Global Variables


# %% plotting


def test(model_name, maze_name):
    latent_tuning_dfs = get_latent_tuning_dfs(model_name=model_name, maze_name=maze_name)
    simple_maze = mr.get_simple_maze(maze_name)
    plot_latent_place_direction_tuning(latent_tuning_dfs["place_direction"], simple_maze)


def plot_latent_place_direction_tuning(latent_tuning_df, simple_maze, axes=None):
    """"""
    _df = latent_tuning_df.T
    n = _df.shape[1]
    # set up fig
    if axes is None:
        f, axes = plt.subplots(2, n // 2, figsize=(6 * n // 2, 12))
    for i, ax in enumerate(axes.flat):
        mp.plot_directed_heatmap(
            simple_maze,
            _df[i],
            colormap="PuOr_r",
            allow_negative=True,
            ax=ax,
        )


def plot_latent_distance_to_goal_tuning(latent_tuning_df, axes=None):
    """ """
    _df = latent_tuning_df.T
    distances = _df.index.values.astype(float)
    n = _df.shape[1]
    # set up fig
    if axes is None:
        f, axes = plt.subplots(2, n // 2, figsize=(n / 1.5, 3), sharex=True)
    for ax in axes.flat:
        ax.spines[["right", "top"]].set_visible(False)
    axes[-1, 0].set_xlabel("Distance to goal (m)")
    axes[-1, 0].set_ylabel("activation")
    # plot tuning
    for i, ax in enumerate(axes.flat):
        ax.plot(distances, _df[i], color="purple", lw=2)


# %% load latent unit tuning from saved model


def get_latent_tuning_dfs(model_set="interpretable_models", model_name="standard", maze_name="maze_1"):
    """ """
    model, params = lms.load_model(
        model_set=model_set,
        model_name=model_name,
        maze_name=maze_name,
        with_model_params=True,
    )
    _Nlat = model.Nlat

    # map out tuning of latent units
    tuning = get_latent_tuning(model)
    # get input group labels
    input_data_kwargs = params["input_data_kwargs"]
    place_direction_labels = get_place_direction_labels(input_data_kwargs)
    distances_to_goal_labels = get_distance_to_goal_labels(input_data_kwargs)
    egocentric_action_labels = get_egocentric_action_labels(input_data_kwargs)
    # check input shapes line up
    pd_inputs, dist_inputs, ego_inputs = model.input_group_indices
    assert len(place_direction_labels) == len(pd_inputs)
    assert len(distances_to_goal_labels) == len(dist_inputs)
    assert len(egocentric_action_labels) == len(ego_inputs)
    # output as dataframe
    pd_df = pd.DataFrame(
        columns=pd.MultiIndex.from_tuples(place_direction_labels),
        data=tuning["place_direction"],
    )
    dist_df = pd.DataFrame(columns=distances_to_goal_labels, data=tuning["distance_to_goal"])
    ego_df = pd.DataFrame(columns=egocentric_action_labels, data=tuning["egocentric_action"])

    return {"place_direction": pd_df, "distance_to_goal": dist_df, "egocentric_action": ego_df}


def get_latent_tuning(model):
    """
    Vectorized version: for each input group, build an X of shape [Nin, k]
    where each column is one one-hot vector, then encode in one go.
    Returns a dict mapping group names → (Nlat × k) NumPy arrays.
    """
    device = next(model.parameters()).device
    model.to(device)
    _input_group_names = model.input_group_names
    _input_group_indices = model.input_group_indices
    try:
        _latent_split_inds = model.latent_split_inds
    except AttributeError:
        _latent_split_inds = None
    _latent_split_inds = (
        [None for _ in range(len(_input_group_names))] if _latent_split_inds is None else _latent_split_inds
    )

    Nin, Nlat = model.Nin, model.Nlat
    tuning = {}

    with torch.no_grad():
        for name, inds, latent_inds in zip(_input_group_names, _input_group_indices, _latent_split_inds):
            # Build set of onehot inputs
            k = len(inds)
            X = torch.zeros(Nin, k, device=device, dtype=torch.float32)
            cols = torch.arange(k, device=device)
            X[inds, cols] = 1.0
            # encode
            Z = model.encode(X).cpu().numpy()
            if latent_inds is None:
                tuning[name] = Z  # shape [Nlat, k]
            else:
                tuning[name] = Z[latent_inds.astype(int)]  # shape [Nlat, n_features]

    return tuning


# %% get feature group labels (should add these as attributes to the model & when making input data)


def get_egocentric_action_labels(params):

    ego_input_params = _reference_default_params(params, "egocentric_action", gid._get_egocentric_action_regressors)
    labels = []
    if "action" in ego_input_params["components"]:
        labels.extend(ego_input_params["actions"])
    if "free_forced" in ego_input_params["components"]:
        labels.extend(["free", "forced"])
    if "tower_bridge" in ego_input_params["components"]:
        labels.extend(["tower", "bridge"])
    return labels


def get_place_direction_labels(params):
    simple_maze = mr.get_simple_maze(params["maze_name"])
    place_directions = mr.get_maze_place_direction_pairs(simple_maze)
    return place_directions


def get_distance_to_goal_labels(params):
    """get distance bins same as input data for the model"""
    distance_input_params = _reference_default_params(params, "distance_to_goal", gid._get_distance_to_goal_regressors)
    assert distance_input_params["method"] == "onehot"
    metric = distance_input_params["metric"]
    if distance_input_params["max_distance"] is None:
        _max = dd.get_distance_percentile(metric, 0.9)
    if metric[0] == "distance_to_goal":
        bin_method = distance_input_params["bin_method"]
        if bin_method == "uniform":
            n_bins = int(_max / distance_input_params["bin_spacing"])
        elif bin_method == "log":
            n_bins = distance_input_params["n_log_bins"]
    bins = convert._get_distance_bins(
        binning_method=bin_method,
        n_distance_bins=n_bins,
        distance_metrics=metric,
        max_distance=_max,
    )
    return [b.mid for b in bins]


def _reference_default_params(params, input_group_name, input_data_func):
    """
    update input group kwargs with defualts if not given in model params
    """
    sig = inspect.signature(input_data_func)
    default_params = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}
    input_group_kwargs = params["input_group_kwargs"]
    if input_group_name not in input_group_kwargs.keys():
        return default_params
    else:
        _params = input_group_kwargs[input_group_name]
        if not all([x in _params.keys() for x in default_params.keys()]):
            return default_params
        else:
            return _params
