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

from GridMaze.analysis.nbeGLM.get_input_data import get_input_data


# %% Global Variables
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # run on GPU if possible


from GridMaze.paths import RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "nbeGLM"

DEFAULT_INPUT_DATA_KWARGS = {
    "subject_IDs": "all",
    "maze_name": "maze_1",
    "days_on_maze": "all",
    "sessions": None,
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
    model_name=None,
    model_set_name=None,
    save_path=None,
    verbose=True,
    seed=0,
):
    """ """
    model_params = copy.deepcopy(locals())
    # set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # get input data
    if verbose:
        print("Loading input data ...")
    input_data = get_input_data(**input_data_kwargs)

    # update model params
    model_params["model_init_kwargs"]["input_streams"] = input_data[0]["X_type_inds"]
    model_params["model_init_kwargs"]["input_stream_names"] = input_data[0]["input_features"]
    model_params["model_init_kwargs"]["Nout"] = sum(s["spikes"].shape[0] for s in input_data)

    # save model params
    if save_path is not None:
        with open(save_path / "model_params.json", "w") as f:
            json.dump(model_params, f, indent=4)

    # get cv var explained by input features for all neurons
    training_cv_scores, cluster_cv_scores = [], []
    n_sessions = len(input_data)
    for i in range(n_sessions):
        if verbose:
            print(f"Running cross-validation for session {i + 1}/{n_sessions} ...")
        nbeGLM = Encoder(**model_init_kwargs)
        test_data = input_data[i]  # single session
        train_data = input_data[:i] + input_data[i + 1 :]  # all other sessions
        model, train_losses, test_perfs, train_perfs = train_model(
            nbeGLM,
            train_data,
            test_data,
            DEVICE,
            eval_alpha=model_eval_kwargs["crossval_alpha"],
            **model_train_kwargs,
        )
        # training_cv_scores.append(
        #     _get_learning_curve_df(test_data, train_losses, test_perfs, train_perfs, model_params)

        test_perf, valid_cluster_mask = nbeGLM.eval_representation(
            test_data["X"].to(DEVICE),
            test_data["spikes"].to(DEVICE),
            cv=model_eval_kwargs["crossval_folds"],
            alpha=model_eval_kwargs["crossval_alpha"],
            embed=True,
            return_keep=True,
            trials=test_data["trial_ids"],
        )
        # cluster_crossval_perfs.append(
        #     _get_cluster_cross_val_df(test_perf, test_data, session, exp_kwargs, valid_cluster_mask)
        # )

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


# %%
def _get_learning_curve_df(test_session_input, train_losses, test_perfs, train_perfs, exp_kwargs):
    """ """
    nepochs = exp_kwargs["model_train"]["nepochs"]
    test_freq = exp_kwargs["model_train"]["test_freq"]
    test_epochs = np.arange(0, nepochs, test_freq)
    return pd.DataFrame(
        {
            "subject_ID": test_session_input["subject_ID"],
            "maze_name": test_session_input["maze_name"],
            "day_on_maze": test_session_input["day_on_maze"],
            "epoch": test_epochs,
            "train_loss": train_losses,
            "train_embedding_perf": train_perfs,
            "test_embedding_perf": test_perfs,
        }
    )


def _get_cluster_cross_val_df(test_perf, test_session_input, eval_session_input, exp_kwargs, valid_clusters):
    """ """
    # if test session is eval session, not in training data
    if (
        test_session_input["subject_ID"] == eval_session_input["subject_ID"]
        and test_session_input["session_name"] == eval_session_input["session_name"]
    ):
        in_training_data = False
    else:
        in_training_data = True
    dfs = []
    for fold in range(exp_kwargs["model_eval"]["crossval_folds"]):
        dfs.append(
            pd.DataFrame(
                {
                    "subject_ID": test_session_input["subject_ID"],
                    "maze_name": test_session_input["maze_name"],
                    "day_on_maze": test_session_input["day_on_maze"],
                    "in_training_data": in_training_data,
                    "cluster_unique_ID": test_session_input["cluster_unique_IDs"][
                        valid_clusters
                    ],  # incase invalid folds bc no spikes and no model eval
                    "fold": fold,
                    "cv_performance": test_perf[:, fold],
                }
            )
        )
    return pd.concat(dfs, axis=0)
