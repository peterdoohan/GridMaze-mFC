"""
Library for running exps that train various instantiations of the embedding model with different input data
@ Kris & Peter (mainly Kris)
"""

# %% Imports
import pickle
import torch
import json
import copy
import numpy as np
import sys
import pandas as pd
from pathlib import Path

from .get_input_data import get_input_data
from .embedding_utils import Encoder, train_model

# %% Global Variables

from ...paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

with open(Path(EXPERIMENT_INFO_PATH) / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

EMBEDDING_MODEL_RESULTS = RESULTS_PATH / "embedding_model" / "exps"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # run on GPU if possible

DEFAULT_INPUT_KWARGS = {
    "subject_IDs": ["m2"],
    "maze_name": "maze_1",
    "days_on_maze": "late",
    "input_features": ["distance", "place_direction"],
    "distance_metrics": ("distance_to_goal", "geodesic"),
    "include_multi_unit": False,
    "navigation_only": True,
    "moving_only": True,
    "resolution": 0.1,
    "max_distance": None,
    "max_steps_to_goal": 30,
    "distance_bin_method": "uniform",
    "n_distance_bins": 20,
    "min_spike_count": 300,
}

DEFAULT_INIT_KWARGS = {
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

DEFAULT_TRAIN_KWARGS = {
    "lr": 5e-4,
    "nepochs": 101,
    "test_freq": 100,
}

DEFAULT_EVAL_KWARGS = {
    "crossval_folds": 5,
    "crossval_alpha": 1e-3,
    "crossval_train_sessions": False,
}

# %% Functions


def run_embedding_model_experiment(
    exp_name,
    exp_set=None,
    with_embedding=True,
    run_crossvalidation=True,
    train_full_model=True,
    input_kwargs=DEFAULT_INPUT_KWARGS,
    model_init_kwargs=DEFAULT_INIT_KWARGS,
    model_train_kwargs=DEFAULT_TRAIN_KWARGS,
    model_eval_kwargs=DEFAULT_EVAL_KWARGS,
    overwrite=False,
    notes=None,
    seed=0,
):
    """ """
    # set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # set up results dir
    if exp_set is not None:
        save_dir = EMBEDDING_MODEL_RESULTS / exp_set / exp_name
    else:
        save_dir = EMBEDDING_MODEL_RESULTS / exp_name
    if (save_dir / "DONE.txt").exists() and not overwrite:
        print(f"Exp with name {exp_name} already completed: {save_dir}. To overwrite, set overwrite=True")
        return
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"running {exp_name} and saving results to {save_dir}")

    # organise model inputs
    print(f"\n{input_kwargs}\n\n{model_init_kwargs}\n\n{model_train_kwargs}\n\n{model_eval_kwargs}\n")
    sessions_input_data = get_input_data(**input_kwargs)
    # update input kwargs
    input_kwargs["exp_name"] = exp_name
    input_kwargs["with_embedding"] = with_embedding
    input_kwargs["run_crossvalidation"] = run_crossvalidation
    input_kwargs["train_full_model"] = train_full_model
    # update init kwargs
    model_init_kwargs["input_streams"] = sessions_input_data[0]["X_type_inds"]
    model_init_kwargs["input_stream_names"] = sessions_input_data[0]["input_feature_names"]
    model_init_kwargs["Nout"] = sum(
        [sesh["spikes"].shape[0] for sesh in sessions_input_data]
    )  # total number of neurons

    # save kwargs
    exp_kwargs = {
        "input": input_kwargs,
        "model_init": model_init_kwargs,
        "model_train": model_train_kwargs,
        "model_eval": model_eval_kwargs,
    }
    with open(save_dir / "experiment_kwargs.json", "w") as f:
        json.dump(exp_kwargs, f, indent=4)

    # run quantitative evaluation
    if run_crossvalidation:
        training_crossval_perfs = []
        cluster_crossval_perfs = []
        # train model and test on held out session
        for i in range(len(sessions_input_data)):
            print(f"\nrunning crossvalidation for session {i+1} of {len(sessions_input_data)}")
            encoder = Encoder(**model_init_kwargs)
            train_sessions = sessions_input_data[:i] + sessions_input_data[i + 1 :]
            test_session = sessions_input_data[i]
            # train & test
            if with_embedding:
                encoder, train_losses, test_perfs, train_perfs = train_model(
                    encoder,
                    train_sessions,
                    test_session,
                    device=DEVICE,
                    eval_alpha=model_eval_kwargs["crossval_alpha"],
                    **model_train_kwargs,
                )
                sys.stdout.flush()
                # save out learning curves
                training_crossval_perfs.append(
                    _get_learning_curve_df(test_session, train_losses, test_perfs, train_perfs, exp_kwargs)
                )
            else:
                print("running without embedding")
            # run crossvalidation on a held-out session, and optionally the training sessions as well
            sessions_to_test = (
                ([test_session] + train_sessions) if model_eval_kwargs["crossval_train_sessions"] else [test_session]
            )
            for session in sessions_to_test:
                test_perf, valid_cluster_mask = encoder.eval_representation(
                    session["X"].to(DEVICE),
                    session["spikes"].to(DEVICE),
                    cv=model_eval_kwargs["crossval_folds"],
                    alpha=model_eval_kwargs["crossval_alpha"],
                    embed=with_embedding,
                    return_keep=True,
                    trials=session["trial_ids"],
                )
                # print([vals.mean() for vals in test_perfs])
                sys.stdout.flush()
                # save out cluster corss validated perf score, with and without embedding
                cluster_crossval_perfs.append(
                    _get_cluster_cross_val_df(test_perf, test_session, session, exp_kwargs, valid_cluster_mask)
                )
        # save to disk
        cluster_cross_val_performance_df = pd.concat(cluster_crossval_perfs, axis=0).reset_index(drop=True)
        cluster_cross_val_performance_df.to_csv(save_dir / "cluster_cross_val_performance.htsv", sep="\t", index=False)
        if with_embedding:
            training_performances_df = pd.concat(training_crossval_perfs, axis=0).reset_index(drop=True)
            training_performances_df.to_csv(save_dir / "training_performances.htsv", sep="\t", index=False)

    # train a model on all data and save
    if train_full_model:
        encoder = Encoder(**model_init_kwargs)
        encoder, train_losses, test_perfs, train_perfs = train_model(
            encoder,
            sessions_input_data,
            device=DEVICE,
            **model_train_kwargs,
        )
        model_training_result = {
            "train_losses": train_losses,
            "train_train_embedding_perfs": train_perfs,
            "model": encoder,
        }
        # save out model
        pickle.dump(model_training_result, open(save_dir / "model_training_result.p", "wb"))

    # save notes
    if notes is not None:
        with open(save_dir / "notes.txt", "w") as f:
            for note in notes:
                f.write(note + "\n")
    # if completed, write an empty DONE.txt file
    with open((save_dir / "DONE.txt"), "w") as file:
        pass
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
