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

# %% functions


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

    Always: scores the trained embedding on held-out spikes with Poisson regression
    → cv_scores.csv (headline embedding-quality, Poisson D²).

    n_permutations > 0: additionally scores against rotationally-permuted held-out
    spikes (Haar-uniform rotation across neurons, see _rotate_spikes) to test whether
    encoded variables are factorised across neurons. Because rotated spikes are
    real-valued, the rotation null uses Ridge regression → perm_cv_scores.csv (R²).
    A matched Ridge-on-true-spikes baseline (ridge_cv_scores.csv, R²) is saved
    alongside; that — not cv_scores.csv — is the apples-to-apples reference for the
    rotation null.

    save_fold_models=True pickles each per-fold model + held-out test_session under
    save_path/models/, so permutations can be rerun from disk without retraining.
    """
    assert not (
        save_fold_models and save_path is None
    ), "save_fold_models=True requires save_path; otherwise per-fold pickles silently go nowhere"
    setup = _setup_run(
        "run_cv_neGLM",
        save_path=save_path,
        overwrite=overwrite,
        verbose=verbose,
        seed=seed,
        input_data_kwargs=input_data_kwargs,
        params=locals(),
    )
    if setup is None:
        return
    input_data, model_params, save_path = setup

    learning_curve_dfs, cluster_cv_score_dfs, ridge_cv_score_dfs, perm_cv_score_dfs = [], [], [], []
    n_sessions = len(input_data)
    for i in range(n_sessions):
        if verbose:
            print(f"Running cross-validatied neGLM for session {i + 1}/{n_sessions} ...")

        test_session = input_data[i]  # single session
        train_sessions = input_data[:i] + input_data[i + 1 :]  # all other sessions

        learning_curve_df, cv_score_df, ridge_cv_score_df, perm_cv_score_df = _run_single_fold(
            train_sessions=train_sessions,
            test_session=test_session,
            model_init_kwargs=model_init_kwargs,
            model_train_kwargs=model_train_kwargs,
            score_kwargs=score_kwargs,
            n_permutations=n_permutations,
            fold_idx=i,
            save_path=save_path,
            save_fold_models=save_fold_models,
            verbose=verbose,
        )
        learning_curve_dfs.append(learning_curve_df)
        cluster_cv_score_dfs.append(cv_score_df)
        if ridge_cv_score_df is not None:
            ridge_cv_score_dfs.append(ridge_cv_score_df)
        if perm_cv_score_df is not None:
            perm_cv_score_dfs.append(perm_cv_score_df)

    training_df = pd.concat(learning_curve_dfs, axis=0).reset_index(drop=True)
    cv_scores_df = pd.concat(cluster_cv_score_dfs, axis=0).reset_index(drop=True)
    ridge_cv_scores_df = (
        pd.concat(ridge_cv_score_dfs, axis=0).reset_index(drop=True) if n_permutations > 0 else None
    )
    perm_cv_scores_df = pd.concat(perm_cv_score_dfs, axis=0).reset_index(drop=True) if n_permutations > 0 else None

    if save_path is not None:
        _save_outputs(
            save_path,
            model_params=model_params,
            training_df=training_df,
            cv_scores_df=cv_scores_df,
            ridge_cv_scores_df=ridge_cv_scores_df,
            perm_cv_scores_df=perm_cv_scores_df,
            verbose=verbose,
        )

    if n_permutations > 0:
        return cv_scores_df, ridge_cv_scores_df, perm_cv_scores_df
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
    setup = _setup_run(
        "run_cv_baselineGLM",
        save_path=save_path,
        overwrite=overwrite,
        verbose=verbose,
        seed=seed,
        input_data_kwargs=input_data_kwargs,
        params=locals(),
    )
    if setup is None:
        return
    input_data, model_params, save_path = setup

    # get cv var explained by input features for all neurons
    cluster_cv_score_dfs = []
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
        cluster_cv_score_dfs.append(
            _get_cluster_cross_val_df(
                test_scores,
                test_session["session_info"],
                test_session["cluster_unique_IDs"],
            )
        )
    cv_scores_df = pd.concat(cluster_cv_score_dfs, axis=0).reset_index(drop=True)
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
    setup = _setup_run(
        "train_neGLM",
        save_path=save_path,
        overwrite=overwrite,
        verbose=verbose,
        seed=seed,
        input_data_kwargs=input_data_kwargs,
        params=locals(),
    )
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


def _run_single_fold(
    train_sessions,
    test_session,
    model_init_kwargs,
    model_train_kwargs,
    score_kwargs,
    n_permutations=0,
    fold_idx=None,
    save_path=None,
    save_fold_models=False,
    verbose=True,
):
    """Train one neGLM embedding on `train_sessions`, score on `test_session`, optionally
    score against rotationally-permuted held-out spikes. Returns
    (learning_curve_df, cv_score_df, ridge_cv_score_df_or_None, perm_cv_score_df_or_None).

    cv_score_df is the Poisson D² headline score; ridge_cv_score_df is the Ridge R²
    matched control for perm_cv_score_df (both Ridge, both on the same z, only
    difference is the rotation). The Ridge baseline + permutation null only run when
    n_permutations > 0.

    fold_idx + save_path + save_fold_models control optional pickling of the trained
    model + held-out test_session under save_path/models/."""
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
            fold_idx=fold_idx,
            subdir="models",
            write_done=False,
        )

    learning_curve_df = _get_learning_curve_df(
        model.train_losses,
        model.test_perfs,
        model.train_perfs,
        model_train_kwargs,
        test_session_info=test_session["session_info"],
    )

    # Poisson scoring of the trained embedding on held-out spikes (headline)
    if verbose:
        print("     testing performance on held-out session (Poisson) ...")
    test_perf = model.score(
        x=test_session["X"],
        y=test_session["spikes"],
        trials=test_session["trial_ids"],
        **score_kwargs,
    )
    cv_score_df = _get_cluster_cross_val_df(
        test_perf,
        test_session["session_info"],
        test_session["cluster_unique_IDs"],
    )

    # Ridge baseline + permutation null (only useful as a pair, gated together)
    ridge_cv_score_df = None
    perm_cv_score_df = None
    if n_permutations > 0:
        if verbose:
            print("     testing performance on held-out session (Ridge baseline) ...")
        ridge_score_kwargs = {**score_kwargs, "loss": "gaussian"}
        ridge_test_perf = model.score(
            x=test_session["X"],
            y=test_session["spikes"],
            trials=test_session["trial_ids"],
            **ridge_score_kwargs,
        )
        ridge_cv_score_df = _get_cluster_cross_val_df(
            ridge_test_perf,
            test_session["session_info"],
            test_session["cluster_unique_IDs"],
        )

        if verbose:
            print("     testing performance on rotationally permuted held-out session ...")
        perm_cv_score_df = _score_permutations(
            model, test_session, n_permutations, score_kwargs, verbose=verbose, base_seed=fold_idx
        )

    return learning_curve_df, cv_score_df, ridge_cv_score_df, perm_cv_score_df


def _score_permutations(model, test_session, n_permutations, score_kwargs, verbose=False, base_seed=None):
    """Score a trained embedding against `n_permutations` rotationally-permuted draws of
    the held-out spikes; returns a single DataFrame with one row per (neuron, fold,
    permutation). With verbose=True, prints progress every 100 permutations.

    base_seed: when given, the k-th permutation's Q is sampled from
    `np.random.default_rng((base_seed, k))` so the same sequence of K rotations is
    reproducible across calls with the same base_seed (e.g. matched-Q subtraction
    across different model variants on the same test session). When None, falls back
    to numpy's global RNG.

    Uses Ridge (loss="gaussian") since rotated spikes are real-valued — Poisson rejects."""
    ridge_score_kwargs = {**score_kwargs, "loss": "gaussian"}
    test_X = test_session["X"]
    test_spikes = test_session["spikes"]
    test_trial_ids = test_session["trial_ids"]
    perm_dfs = []
    for perm in range(n_permutations):
        rng = np.random.default_rng((base_seed, perm)) if base_seed is not None else None
        rotated_test_spikes = _rotate_spikes(test_spikes, rng=rng)
        test_perf = model.score(
            x=test_X,
            y=rotated_test_spikes,
            trials=test_trial_ids,
            **ridge_score_kwargs,
        )
        perm_df = _get_cluster_cross_val_df(
            test_perf,
            test_session["session_info"],
            test_session["cluster_unique_IDs"],
        )
        perm_df["permutation"] = perm
        perm_dfs.append(perm_df)
        if verbose and (perm + 1) % 100 == 0:
            print(f"          {perm + 1}/{n_permutations} permutations done ...")
    return pd.concat(perm_dfs, axis=0)


def _rotate_spikes(spikes, rng=None):
    """
    Apply a Haar-uniform random rotation across neurons to test whether information
    about encoded variables is factorised across neurons or mixed.

    spikes: array of shape (N_neurons, T_time)
    rng: optional np.random.Generator for deterministic Q. If None, draws from numpy's
         global RNG (legacy behaviour).
    Returns: Q @ spikes, where Q is sampled uniformly from O(N_neurons).
    """
    n_neurons = spikes.shape[0]
    if rng is None:
        Z = np.random.standard_normal((n_neurons, n_neurons))
    else:
        Z = rng.standard_normal((n_neurons, n_neurons))
    Q, R = np.linalg.qr(Z)
    Q = Q * np.sign(np.diag(R))  # sign correction → Haar-uniform on O(n_neurons)
    return Q @ spikes


def _setup_run(fn_name, save_path, overwrite, verbose, seed, input_data_kwargs, params):
    """
    Shared setup for run_*/train_* entry points: validates save_path, returns early if
    outputs already exist, captures model params, sets seeds, and loads input data.

    `params` is the dict captured into model_params.json (typically locals() at the top
    of the caller). Other args are referenced explicitly so static analysis sees them.

    Returns (input_data, model_params, save_path), or None if the run should be skipped.
    """
    save_path = Path(save_path) if save_path is not None else None
    if _outputs_exist(save_path, overwrite, verbose):
        return None

    model_params = copy.deepcopy(params)
    model_params["fn"] = fn_name

    np.random.seed(seed)
    torch.manual_seed(seed)

    if verbose:
        print("Loading input data ...")
    input_data = get_input_data(**input_data_kwargs)

    return input_data, model_params, save_path


def _outputs_exist(save_path, overwrite, verbose):
    """True iff save_path/DONE.txt exists and overwrite is False."""
    if save_path is None or overwrite:
        return False
    if (save_path / "DONE.txt").exists():
        if verbose:
            print("model output already populated, set overwrite=True to overwrite existing results")
        return True
    return False


def _save_outputs(
    save_path,
    model_params=None,
    training_df=None,
    cv_scores_df=None,
    ridge_cv_scores_df=None,
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
    # save model params (don't mutate the caller's dict)
    if model_params is not None:
        params_to_save = {**model_params, "save_path": str(save_path)}  # make .json serializable
        with open(save_path / "model_params.json", "w") as f:
            json.dump(params_to_save, f, indent=4)
    # save model training data
    if training_df is not None:
        training_df.to_csv(save_path / "training.csv", index=False)
    # save cv neuron scores (Poisson D²)
    if cv_scores_df is not None:
        cv_scores_df.to_csv(save_path / "cv_scores.csv", index=False)
    # save Ridge cv neuron scores (R²) — matched baseline for the rotation null
    if ridge_cv_scores_df is not None:
        ridge_cv_scores_df.to_csv(save_path / "ridge_cv_scores.csv", index=False)
    # save permutation cv neuron scores (Ridge R² on rotated spikes)
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
    """One row per recorded training epoch with train/test embedding perf and train loss."""
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
    """Long-form per-(neuron, fold) cv_score table tagged with the test session's metadata."""
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
