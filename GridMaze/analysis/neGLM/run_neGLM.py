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
    n_permutations=0,
    save_fold_models=False,
    seed=0,
    verbose=True,
    save_path=None,
    overwrite=False,
):
    """
    Leave-one-session-out cross-validated neGLM.

    n_permutations > 0 additionally scores the trained embedding against rotationally-
    permuted held-out spikes (Haar-uniform rotation across neurons, see _rotate_spikes)
    to test whether encoded variables are factorised across neurons. The aggregated null
    is saved to perm_cv_scores.csv and returned alongside the non-permuted cv_scores_df.

    save_fold_models=True pickles each per-fold model + held-out test_session under
    save_path/models/, so permutations can be rerun from disk without retraining.
    """
    setup = _setup_run("run_cv_neGLM", locals())
    if setup is None:
        return
    input_data, model_params, save_path = setup

    learning_curve_dfs, cluster_cv_scores, perm_cluster_cv_scores = [], [], []
    n_sessions = len(input_data)
    for i in range(n_sessions):
        if verbose:
            print(f"Running cross-validatied neGLM for session {i + 1}/{n_sessions} ...")

        test_session = input_data[i]  # single session
        train_sessions = input_data[:i] + input_data[i + 1 :]  # all other sessions

        # init + train
        model = models.neGLM(**model_init_kwargs)
        if verbose:
            print("     learning embedding ...")
        model.train(train_sessions, test_session, **model_train_kwargs)

        # optionally pickle model + held-out test session so permutations can be rerun from disk
        if save_fold_models:
            _save_outputs(
                save_path,
                model=model,
                test_session=test_session,
                fold_idx=i,
                subdir="models",
                write_done=False,
            )

        learning_curve_dfs.append(
            _get_learning_curve_df(
                model.train_losses,
                model.test_perfs,
                model.train_perfs,
                model_train_kwargs,
                test_session_info=test_session["session_info"],
            )
        )

        test_X = test_session["X"]
        test_spikes = test_session["spikes"]
        test_trial_ids = test_session["trial_ids"]

        # non-permuted reference: score the trained embedding on the true held-out spikes
        if verbose:
            print("     testing performance on held-out session ...")
        test_perf = model.score(
            x=test_X,
            y=test_spikes,
            trials=test_trial_ids,
            **score_kwargs,
        )
        cluster_cv_scores.append(
            _get_cluster_cross_val_df(
                test_perf,
                test_session["session_info"],
                test_session["cluster_unique_IDs"],
            )
        )

        # optional permutation null
        if n_permutations > 0:
            if verbose:
                print("     testing performance on rotationally permuted held-out session ...")
            perm_dfs = []
            for perm in range(n_permutations):
                rotated_test_spikes = _rotate_spikes(test_spikes)
                test_perf = model.score(
                    x=test_X,
                    y=rotated_test_spikes,
                    trials=test_trial_ids,
                    **score_kwargs,
                )
                perm_df = _get_cluster_cross_val_df(
                    test_perf,
                    test_session["session_info"],
                    test_session["cluster_unique_IDs"],
                )
                perm_df["permutation"] = perm
                perm_dfs.append(perm_df)
            perm_cluster_cv_scores.append(pd.concat(perm_dfs, axis=0))

    training_df = pd.concat(learning_curve_dfs, axis=0).reset_index(drop=True)
    cv_scores_df = pd.concat(cluster_cv_scores, axis=0).reset_index(drop=True)
    perm_cv_scores_df = pd.concat(perm_cluster_cv_scores, axis=0).reset_index(drop=True) if n_permutations > 0 else None

    if save_path is not None:
        _save_outputs(
            save_path,
            model_params=model_params,
            training_df=training_df,
            cv_scores_df=cv_scores_df,
            perm_cv_scores_df=perm_cv_scores_df,
            verbose=verbose,
        )

    if n_permutations > 0:
        return cv_scores_df, perm_cv_scores_df
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
    setup = _setup_run("run_cv_baselineGLM", locals())
    if setup is None:
        return
    input_data, model_params, save_path = setup

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
    setup = _setup_run("train_neGLM", locals())
    if setup is None:
        return
    input_data, model_params, save_path = setup

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


def _rotate_spikes(spikes):
    """
    Apply a Haar-uniform random rotation across neurons to test whether information
    about encoded variables is factorised across neurons or mixed.

    spikes: array of shape (N_neurons, T_time)
    Returns: Q @ spikes, where Q is sampled uniformly from O(N_neurons).

    Sampling follows Mezzadri (2007) / https://mathstoshare.com/2024/03/09/uniformly-sampling-orthogonal-matrices/:
    QR-decompose a standard Gaussian matrix, then multiply each column of Q by the
    sign of the corresponding diagonal of R so Q is Haar-uniform on O(N).
    """
    n_neurons = spikes.shape[0]
    Z = np.random.randn(n_neurons, n_neurons)
    Q, R = np.linalg.qr(Z)
    Q = Q * np.sign(np.diag(R))  # sign correction → Haar-uniform on O(n_neurons)
    return Q @ spikes


def _setup_run(fn_name, fn_locals):
    """
    Shared setup for run_*/train_* entry points: validates save_path, returns early if
    outputs already exist, captures model params, sets seeds, and loads input data.

    Returns (input_data, model_params, save_path), or None if the run should be skipped.
    """
    save_path = Path(fn_locals["save_path"]) if fn_locals["save_path"] is not None else None
    if _outputs_exist(save_path, fn_locals["overwrite"], fn_locals["verbose"]):
        return None

    model_params = copy.deepcopy(fn_locals)
    model_params["fn"] = fn_name

    np.random.seed(fn_locals["seed"])
    torch.manual_seed(fn_locals["seed"])

    if fn_locals["verbose"]:
        print("Loading input data ...")
    input_data = get_input_data(**fn_locals["input_data_kwargs"])

    return input_data, model_params, save_path


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


def _save_outputs(
    save_path,
    model_params=None,
    training_df=None,
    cv_scores_df=None,
    perm_cv_scores_df=None,
    model=None,
    test_session=None,
    fold_idx=None,
    subdir=None,
    write_done=True,
    verbose=False,
):
    """
    Pickles for model/test_session land in `save_path/subdir/` (or `save_path/` if
    subdir is None). When fold_idx is given, filenames are suffixed (e.g. model_3.pkl),
    so this can be reused for per-fold saves inside CV loops. write_done=False skips the
    DONE.txt sentinel — useful for incremental writes that aren't the final save.
    """
    if save_path is None:
        return
    # ensure save path exists
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Saving outputs to: {save_path}")
    # save model params
    if model_params is not None:
        model_params["save_path"] = str(save_path)  # make .json serializable
        with open(save_path / "model_params.json", "w") as f:
            json.dump(model_params, f, indent=4)
    # save model training data
    if training_df is not None:
        training_df.to_csv(save_path / "training.csv", index=False)
    # save cv neuron scores
    if cv_scores_df is not None:
        cv_scores_df.to_csv(save_path / "cv_scores.csv", index=False)
    # save permutation cv neuron scores
    if perm_cv_scores_df is not None:
        perm_cv_scores_df.to_csv(save_path / "perm_cv_scores.csv", index=False)
    # save model and/or test_session pickle (optional subdir + per-fold index suffix)
    if model is not None or test_session is not None:
        pickle_dir = save_path / subdir if subdir is not None else save_path
        if not pickle_dir.exists():
            pickle_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{fold_idx}" if fold_idx is not None else ""
        if model is not None:
            with open(pickle_dir / f"model{suffix}.pkl", "wb") as f:
                pickle.dump(model, f)
        if test_session is not None:
            with open(pickle_dir / f"test_session{suffix}.pkl", "wb") as f:
                pickle.dump(test_session, f)
    # save DONE.txt file to indicate that the job is done
    if write_done:
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
