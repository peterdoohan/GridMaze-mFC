"""
Run nbeGLM (neural-behavioural embedding GLM) encoding analyses
Use-cases:
- run cross-validated nbeGLM, save out cv scores for all neurons
- run cross-validated GLM (without neural-behavioural embedding), save out cv scores for all neurons (control)
- train nbeGLM on all input data, save out latents and model weights
@peterdoohan @krisjensen
"""

# %% Imports
import sys
import json
import copy
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from nbeGLM.model_utils import Encoder, train_model


# %% Global Variables
from GridMaze.paths import RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "nbeGLM"

DEFAULT_INPUT_DATA_KWARGS = {
    "subject_IDs": "all",
    "maze_name": "maze_1",
    "days_on_maze": "all",
    "input_features": ["place_direction", "distance_to_goal", "egocentric_action"],
    "input_feature_kwargs": None,
    "resolution": 0.1,
    "max_steps_to_goal": 30,
    "min_spike_count": 300,
    "moving_only": False,
}

DEFAULT_MODEL_INIT_KWARGS = {
    "latent_inputs": None,
    "latent_nonlin": None,
    "partition": None,
    "Nhid": [100, 50],
    "Nlat": 10,
    "beta_act": 1e-1,
    "beta_weight": 1e-1,
    "inv_link": "exp",
    "noise_function": "Poisson",
    "sqrt_counts": False,
    "combine_frs": False,
}

DEFAULT_MODEL_TRAIN_KWARGS = {
    "lr": 5e-4,
    "nepochs": 101,
    "test_freq": 100,
}

DEFAULT_MODEL_EVAL_KWARGS = {
    "crossval_folds": 5,
    "crossval_alpha": 1e-3,
    "crossval_train_sessions": False,
}

# %% Imports


def run_nbeGLM_set():
    """ """
    return


def run_cv_nbeGLM(
    input_data_kwargs=DEFAULT_INPUT_DATA_KWARGS,
    model_init_kwargs=DEFAULT_MODEL_INIT_KWARGS,
    model_train_kwargs=DEFAULT_MODEL_TRAIN_KWARGS,
    model_eval_kwargs=DEFAULT_MODEL_EVAL_KWARGS,
    save_path=None,
    verbose=True,
    seed=0,
):
    """ """
    model_params = locals()
    # set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # get input data
    input_data = get_input_data(**input_data_kwargs)

    return


def run_cv_GLM(
    input_data_kwargs=DEFAULT_MODEL_INIT_KWARGS,
    model_init_kwargs={
        "beta_weight": 1e-1,
        "inv_link": "exp",
        "noise_function": "Poisson",
        "sqrt_counts": False,
        "combine_frs": False,
    },
):
    """run regular GLM without learned neural-behavioural embedding"""
    return


def train_nbeGLM(
    input_data_kwargs=DEFAULT_INPUT_DATA_KWARGS,
    model_init_kwargs=DEFAULT_MODEL_INIT_KWARGS,
    model_train_kwargs=DEFAULT_MODEL_TRAIN_KWARGS,
):
    """
    non-cv training embedding model on all input data. Useful for interigating latents
    """
    return
