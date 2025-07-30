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

from nbeGLM.models import Encoder

from GridMaze.analysis.nbeGLM.get_input_data import get_input_data


# %% Global Variables
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # run on GPU if possible

from GridMaze.paths import RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "nbeGLM"

from jobs.nbeGLM.utils import (
    DEFAULT_INPUT_DATA_KWARGS,
    DEFAULT_MODEL_INIT_KWARGS,
    DEFAULT_MODEL_TRAIN_KWARGS,
    DEFAULT_MODEL_EVAL_KWARGS,
)


# %% Imports


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
    model_params = copy.deepcopy(locals())
    # set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # get input data
    if verbose:
        print("Loading input data ...")
    input_data = get_input_data(**input_data_kwargs)

    # get cv var explained by input features for all neurons
    learning_curve_dfs, cluster_cv_scores = [], []
    n_sessions = len(input_data)
    for i in range(n_sessions):
        if verbose:
            print(f"Running cross-validation for session {i + 1}/{n_sessions} ...")
        nbeGLM = Encoder(**model_params["model_init_kwargs"])
        test_data = input_data[i]  # single session
        train_data = input_data[:i] + input_data[i + 1 :]  # all other sessions

        if verbose:
            print("     learning embedding ...")
        assert model_init_kwargs["with_embedding"], "nbeGLM requires with_embedding=True"

        # train sklearn style
        nbeGLM.train(train_data, test_data, device=DEVICE, **model_train_kwargs)
        # output learning curves for each model training
        learning_curve_dfs.append(
            _get_learning_curve_df(
                nbeGLM.train_losses,
                nbeGLM.test_perfs,
                nbeGLM.train_perfs,
                model_params,
                test_session_info=test_data["session_info"],
            )
        )
        # test on held-out session
        if verbose:
            print("     testing performance on held-out session ...")
        test_perf = nbeGLM.score(
            x=test_data["X"],
            y=test_data["spikes"],
            # need more kwargs
        )  # neurons, folds
        cluster_cv_scores.append(
            _get_cluster_cross_val_df(
                test_perf, test_data["session_info"], test_data["cluster_unique_IDs"], model_params, valid_cluster_mask
            )
        )

    training_df = pd.concat(learning_curve_dfs, axis=0).reset_index(drop=True)
    cv_scores_df = pd.concat(cluster_cv_scores, axis=0).reset_index(drop=True)

    if save_path is not None:
        if verbose:
            print(f"Saving outputs to: {save_path}")
        # save model params
        with open(save_path / "model_params.json", "w") as f:
            json.dump(model_params, f, indent=4)
        # save model training data
        training_df.to_csv(save_path / "training.csv", index=False)
        cv_scores_df.to_csv(save_path / "cv_scores.csv", index=False)

    return cv_scores_df


def run_cv_GLM(
    input_data_kwargs=DEFAULT_MODEL_INIT_KWARGS,
):
    """run regular GLM *without* learned neural-behavioural embedding"""
    # TODO: build separate class for running GLM on single sessions without embedding
    return


def train_nbeGLM(
    input_data_kwargs=DEFAULT_INPUT_DATA_KWARGS,
    model_init_kwargs=DEFAULT_MODEL_INIT_KWARGS,
    model_train_kwargs=DEFAULT_MODEL_TRAIN_KWARGS,
    save_path=None,
    seed=0,
    verbose=True,
):
    """
    non-cv training embedding model on all input data. Useful for looking at latents
    """
    # set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # def model params
    model_params = copy.deepcopy(locals())

    # get input data
    if verbose:
        print("Loading input data ...")
    input_data = get_input_data(**input_data_kwargs)

    # update model params
    model_params["model_init_kwargs"]["input_streams"] = input_data[0]["X_type_inds"]
    model_params["model_init_kwargs"]["input_stream_names"] = input_data_kwargs["input_features"]
    model_params["model_init_kwargs"]["Nout"] = sum(s["spikes"].shape[0] for s in input_data)

    # fit model
    nbeGLM = Encoder(**model_init_kwargs)
    if verbose:
        print("     training model on all input data ...")
    nbeGLM, train_losses, test_perfs, train_perfs = train_model(
        nbeGLM,
        train_sessions=input_data,
        test_session=None,
        device=DEVICE,
        **model_train_kwargs,
    )

    # save outputs
    if save_path is not None:
        if verbose:
            print(f"     saving outputs to: {save_path}")
        # save model params
        with open(save_path / "model_params.json", "w") as f:
            json.dump(model_params, f, indent=4)
        # save model training data
        learning_curve_df = _get_learning_curve_df(train_losses, test_perfs, train_perfs, model_params)
        learning_curve_df.to_csv(save_path / "training.csv", index=False)
        # save model w/ weights
        with open(save_path / "model.pkl", "wb") as f:
            pickle.dump(nbeGLM, f)

    return nbeGLM


# %% helper functions


def _get_learning_curve_df(train_losses, test_perfs, train_perfs, model_params, test_session_info=None):
    """ """
    nepochs = model_params["model_train_kwargs"]["nepochs"]
    test_freq = model_params["model_train_kwargs"]["test_freq"]
    test_epochs = np.arange(0, nepochs, test_freq)
    results = {
        "epoch": test_epochs,
        "train_loss": train_losses,
        "train_embedding_perf": train_perfs,
        "test_embedding_perf": test_perfs,
    }
    if test_session_info is not None:
        results.update(
            {
                "subject_ID": test_session_info["subject_ID"],
                "maze_name": test_session_info["maze_name"],
                "day_on_maze": test_session_info["day_on_maze"],
            }
        )
    return pd.DataFrame(results)


def _get_cluster_cross_val_df(test_perf, test_session_info, cluster_unique_IDs, model_params, valid_cluster_mask):
    """ """
    # if test session is eval session, not in training data
    dfs = []
    n_folds = model_params["model_eval_kwargs"]["crossval_folds"]
    for fold in range(n_folds):
        dfs.append(
            pd.DataFrame(
                {
                    "subject_ID": test_session_info["subject_ID"],
                    "maze_name": test_session_info["maze_name"],
                    "day_on_maze": test_session_info["day_on_maze"],
                    "cluster_unique_ID": np.array(cluster_unique_IDs)[valid_cluster_mask],
                    "fold": fold,
                    "cv_score": test_perf[:, fold],
                }
            )
        )
    return pd.concat(dfs, axis=0)
