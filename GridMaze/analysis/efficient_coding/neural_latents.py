"""
Script for generating neural latents from tranined embedding models (loaded from disk)
"""

# %% Imports
import torch
import numpy as np
import pandas as pd

from GridMaze.analysis.core import convert
from GridMaze.maze import representations as mr
from GridMaze.analysis.embedding_model import load_experiment as le

# %% Global Variables

EXP_SET = "example_models"

# %% Functions


def get_latent_state_action_distance_tuning(maze_name, n_latents):
    """
    Note hardcoded exp_dir & EXP_SET names
    """
    exp_name = f"{maze_name}_state_action_distance_{n_latents}_latents"
    Encoder = le.load_encoder(exp_name, exp_set=EXP_SET)
    kwargs = le.load_kwargs(exp_name, exp_set=EXP_SET)
    latent_tuning_df = _get_latent_state_action_distance_tuning(Encoder, kwargs)
    return latent_tuning_df


def _get_latent_state_action_distance_tuning(Encoder, kwargs, return_as="df"):
    """ """
    input_kwargs = kwargs["input"]
    # get distance bins
    assert "distance" in kwargs["input"]["input_features"]
    distance_bins = convert._get_distance_bins(
        input_kwargs["distance_bin_method"],
        input_kwargs["n_distance_bins"],
        input_kwargs["distance_metrics"],
        input_kwargs["max_distance"],
    )
    # get place-directions
    assert "place_direction" in kwargs["input"]["input_features"]
    simple_maze = mr.get_simple_maze(kwargs["input"]["maze_name"])
    place_directions = mr.get_maze_place_direction_pairs(simple_maze)
    # get latent place-direction-distance (state-action-distance) tuning
    n_pd = len(place_directions)
    n_d = len(distance_bins)
    PD = torch.arange(n_pd)  # place-direction indices
    D = torch.arange(n_d)  # distance indices
    X = torch.zeros(n_pd + n_d, n_pd * n_d)  # init all paired inputs to their prod-space position
    for _pd in PD:
        for d in D:
            ind = _pd * n_d + d
            X[_pd, ind] = 1.0
            X[n_pd + d, ind] = 1.0
    Z = Encoder.encode(X.to(Encoder.Wout.device)).detach().cpu().numpy()  # [n_latents, n_pd * n_d]
    if return_as == "tensor":
        return Z.reshape(Encoder.Nlat, n_pd, n_d)  # [n_latents, n_pd, n_d]
    elif return_as == "df":
        return pd.DataFrame(
            Z,
            columns=pd.MultiIndex.from_product(
                [
                    place_directions,
                    [d.mid for d in distance_bins],
                ],
                names=[
                    "place_direction",
                    "distance_to_goal",
                ],
            ),
        ).T
