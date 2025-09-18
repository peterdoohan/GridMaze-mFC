"""
Run neGLM (neural-behavioural embedding GLM) encoding analyses
Use-cases:
- run cross-validated neGLM, save out cv scores for all neurons
- run cross-validated GLM (without neural-behavioural embedding), save out cv scores for all neurons (control)
- train neGLM on all input data, save out latents and model weights
@peterdoohan @krisjensen
"""

# %% Imports
import json
import copy
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from neGLM import models

from GridMaze.analysis.neGLM.get_input_data import get_input_data


# %% Global Variables
from GridMaze.paths import RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "neGLM"

from jobs.neGLM.utils import (
    DEFAULT_INPUT_DATA_KWARGS,
    DEFAULT_MODEL_INIT_KWARGS,
    DEFAULT_MODEL_TRAIN_KWARGS,
    DEFAULT_SCORE_KWARGS,
)


# %% Imports


def run_cv_neGLM(
    input_data_kwargs=DEFAULT_INPUT_DATA_KWARGS,
    model_init_kwargs=DEFAULT_MODEL_INIT_KWARGS,
    model_train_kwargs=DEFAULT_MODEL_TRAIN_KWARGS,
    score_kwargs=DEFAULT_SCORE_KWARGS,
    seed=0,
    verbose=True,
    save_path=None,
    overwrite=False,
):
    """ """
    save_path = Path(save_path) if save_path is not None else None
    if _outputs_exist(save_path, overwrite, verbose):
        return
    # remember model params
    model_params = copy.deepcopy(locals())
    model_params["fn"] = "run_cv_neGLM"
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
            print(f"Running cross-validatied neGLM for session {i + 1}/{n_sessions} ...")

        test_session = input_data[i]  # single session
        train_sessions = input_data[:i] + input_data[i + 1 :]  # all other sessions

        # init model
        model = models.neGLM(**model_init_kwargs)

        if verbose:
            print("     learning embedding ...")
        model.train(train_sessions, test_session, **model_train_kwargs)
        # output learning curves for each model training
        learning_curve_dfs.append(
            _get_learning_curve_df(
                model.train_losses,
                model.test_perfs,
                model.train_perfs,
                model_train_kwargs,
                test_session_info=test_session["session_info"],
            )
        )

        # test on held-out session
        if verbose:
            print("     testing performance on held-out session ...")
        test_perf = model.score(
            x=test_session["X"],
            y=test_session["spikes"],
            trials=test_session["trial_ids"],
            **score_kwargs,
        )
        cluster_cv_scores.append(
            _get_cluster_cross_val_df(
                test_perf,
                test_session["session_info"],
                test_session["cluster_unique_IDs"],
            )
        )

    training_df = pd.concat(learning_curve_dfs, axis=0).reset_index(drop=True)
    cv_scores_df = pd.concat(cluster_cv_scores, axis=0).reset_index(drop=True)

    if save_path is not None:
        _save_outputs(
            save_path,
            model_params=model_params,
            training_df=training_df,
            cv_scores_df=cv_scores_df,
            verbose=verbose,
        )

    return cv_scores_df


def run_cv_baselineGLM(
    input_data_kwargs=DEFAULT_INPUT_DATA_KWARGS,
    score_kwargs=DEFAULT_SCORE_KWARGS,
    seed=0,
    verbose=True,
    save_path=None,
    overwrite=False,
):
    """run regular GLM *without* learned neural-behavioural embedding"""
    save_path = Path(save_path) if save_path is not None else None
    if _outputs_exist(save_path, overwrite, verbose):
        return
    # remember model params
    model_params = copy.deepcopy(locals())
    model_params["fn"] = "run_cv_baselineGLM"
    # set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # get input data
    if verbose:
        print("Loading input data ...")
    input_data = get_input_data(**input_data_kwargs)

    # get cv var explained by input features for all neurons
    cluster_cv_scores = []
    n_sessions = len(input_data)
    for i in range(n_sessions):
        if verbose:
            print(f"Running cross-validatied baseline Poisson GLM for session {i + 1}/{n_sessions} ...")

        test_session = input_data[i]  # single session

        # init model
        model = models.baselineGLM()
        test_scores = model.score(
            x=test_session["X"],
            y=test_session["spikes"],
            trials=test_session["trial_ids"],
            **score_kwargs,
        )
        cluster_cv_scores.append(
            _get_cluster_cross_val_df(
                test_scores,
                test_session["session_info"],
                test_session["cluster_unique_IDs"],
            )
        )
    cv_scores_df = pd.concat(cluster_cv_scores, axis=0).reset_index(drop=True)
    if save_path is not None:
        _save_outputs(
            save_path,
            model_params=model_params,
            cv_scores_df=cv_scores_df,
            verbose=verbose,
        )

    return cv_scores_df


def train_neGLM(
    input_data_kwargs=DEFAULT_INPUT_DATA_KWARGS,
    model_init_kwargs=DEFAULT_MODEL_INIT_KWARGS,
    model_train_kwargs=DEFAULT_MODEL_TRAIN_KWARGS,
    save_path=None,
    seed=0,
    verbose=True,
    overwrite=False,
):
    """
    non-cv training embedding model on all input data. Useful for looking at latents
    """
    save_path = Path(save_path) if save_path is not None else None
    if _outputs_exist(save_path, overwrite, verbose):
        return

    # remember model params
    model_params = copy.deepcopy(locals())
    model_params["fn"] = "train_neGLM"
    # set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # get input data
    if verbose:
        print("Loading input data ...")
    input_data = get_input_data(**input_data_kwargs)

    # fit model
    model = models.neGLM(**model_init_kwargs)
    if verbose:
        print("Learning embedding from all input data ...")
    model.train(input_data, test_session=None, **model_train_kwargs)

    # save outputs
    if save_path is not None:
        training_df = _get_learning_curve_df(
            model.train_losses,
            model.test_perfs,
            model.train_perfs,
            model_train_kwargs,
            test_session_info=None,
        )
        _save_outputs(
            save_path,
            model_params=model_params,
            training_df=training_df,
            model=model,
            verbose=verbose,
        )

    return model


# %% helper functions


def _outputs_exist(save_path, overwrite, verbose):
    """
    also returns False is save_path is None
    """
    if save_path is not None:
        if not overwrite and (save_path / "DONE.txt").exists():
            if verbose:
                print(f"model output already populated, set overwrite=True to overwrite existing results")
            return True
    else:
        return False


def _save_outputs(save_path, model_params=None, training_df=None, cv_scores_df=None, model=None, verbose=False):
    """ """
    if save_path is None:
        return
    else:
        # ensure save path exists
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Saving outputs to: {save_path}")
        # save model params
        if model_params is not None:
            model_params["save_path"] = str(save_path) if save_path is not None else None  # make .json serializable
            with open(save_path / "model_params.json", "w") as f:
                json.dump(model_params, f, indent=4)
        # save model training data
        if training_df is not None:
            training_df.to_csv(save_path / "training.csv", index=False)
        # save cv neuron scores
        if cv_scores_df is not None:
            cv_scores_df.to_csv(save_path / "cv_scores.csv", index=False)
        # save model w/ weights
        if model is not None:
            with open(save_path / "model.pkl", "wb") as f:
                pickle.dump(model, f)
        # save DONE.txt file to indicate that the job is done
        with open(save_path / "DONE.txt", "w") as f:
            f.write("DONE")
    return


def _get_learning_curve_df(train_losses, test_perfs, train_perfs, model_train_kwargs, test_session_info=None):
    """ """
    nepochs = model_train_kwargs["nepochs"]
    test_freq = model_train_kwargs["test_freq"]
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


def _get_cluster_cross_val_df(test_perf, test_session_info, cluster_unique_IDs):
    """ """
    # if test session is eval session, not in training data
    n_folds = test_perf.shape[1]
    dfs = []
    for fold in range(n_folds):
        dfs.append(
            pd.DataFrame(
                {
                    "subject_ID": test_session_info["subject_ID"],
                    "maze_name": test_session_info["maze_name"],
                    "day_on_maze": test_session_info["day_on_maze"],
                    "cluster_unique_ID": cluster_unique_IDs,
                    "fold": fold,
                    "cv_score": test_perf[:, fold],
                }
            )
        )
    return pd.concat(dfs, axis=0)
