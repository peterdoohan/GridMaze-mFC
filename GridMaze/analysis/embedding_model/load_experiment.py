"""
Library for loading embedding model experiments from results
"""

# %% Imports
import json
import pickle
import numpy as np
import pandas as pd


# %% Global Variables
from ...paths import RESULTS_PATH

EMBEDDING_MODEL_EXPS_DIR = RESULTS_PATH / "embedding_model" / "exps"

# %% Functions


def load_encoder(exp_name, exp_set=None):
    """
    Loads Encoder object from disk, specified by Encoder() class in ./embedding_untils.py
    """
    exp_dir = EMBEDDING_MODEL_EXPS_DIR / exp_name if exp_set is None else EMBEDDING_MODEL_EXPS_DIR / exp_set / exp_name
    model_training_result = pickle.load(open(exp_dir / "model_training_result.p", "rb"))
    return model_training_result["model"]


def load_kwargs(exp_name, exp_set=None):
    """ """
    exp_dir = EMBEDDING_MODEL_EXPS_DIR / exp_name if exp_set is None else EMBEDDING_MODEL_EXPS_DIR / exp_set / exp_name
    with open(exp_dir / "experiment_kwargs.json", "r") as f:
        exp_kwargs = json.load(f)
    return exp_kwargs


def load_cluster_crossval_perf(exp_name, exp_set=None, average_over_folds=True, abbrev=None):
    """ """
    exp_dir = EMBEDDING_MODEL_EXPS_DIR / exp_name if exp_set is None else EMBEDDING_MODEL_EXPS_DIR / exp_set / exp_name
    df = pd.read_csv(exp_dir / "cluster_cross_val_performance.htsv", sep="\t")
    if average_over_folds:
        df = (
            df.groupby([c for c in df.columns if c not in ["fold", "cv_performance"]])
            .cv_performance.mean()
            .reset_index()
        )
    if abbrev is not None:
        df["abbrev"] = abbrev
    return df


# %% Old fn
def load_exp_results(
    exp_name,
    exp_set_dir=None,
    data_structure="cluster_crossval_perf",
    average_over_folds=True,
):
    """ """
    exp_dir = _get_exp_dir(exp_name, exp_set_dir)
    all_data_structures = [
        "exp_kwargs",
        "full_model",
        "cluster_crossval_perf",
        "training_crossval_perf",
        "full_model_training_perf",
    ]
    if data_structure not in all_data_structures:
        raise ValueError(f"data_structure must be {all_data_structures}, not {data_structure}")
    # load htsv outputs as pandas dataframes
    if data_structure == "cluster_crossval_perf":
        df = pd.read_csv(exp_dir / "cluster_cross_val_performance.htsv", sep="\t")
        if average_over_folds:
            df = (
                df.groupby([c for c in df.columns if c not in ["fold", "cv_performance"]])
                .cv_performance.mean()
                .reset_index()
            )
        return df

    elif data_structure == "training_crossval_perf":
        return pd.read_csv(exp_dir / "training_performances.htsv", sep="\t")

    elif data_structure == "exp_kwargs":
        with open(exp_dir / "experiment_kwargs.json", "r") as f:
            return json.load(f)
    else:
        # load full model results
        with open(exp_dir / "experiment_kwargs.json", "r") as f:
            exp_kwargs = json.load(f)
        model_training_result = pickle.load(open(exp_dir / "model_training_result.p", "rb"))

        if data_structure == "full_model_training_perf":
            nepochs = exp_kwargs["model_train"]["nepochs"]
            test_freq = exp_kwargs["model_train"]["test_freq"]
            test_epochs = np.arange(0, nepochs, test_freq)
            return {
                "epoch": test_epochs,
                "train_loss": model_training_result["train_losses"],
                "train_perf": model_training_result["train_train_embedding_perfs"],
            }
        elif data_structure == "full_model":
            return model_training_result["model"]  # custom encoder model obj


def _get_exp_dir(exp_name, exp_set_dir):
    """
    Checks if exp_name exists in EMBEDDING_MODEL_EXPS_DIR and returns the path if it does,
    else returns None with print warning
    """
    if exp_set_dir is not None:
        all_exps_dir = EMBEDDING_MODEL_EXPS_DIR / exp_set_dir
    else:
        all_exps_dir = EMBEDDING_MODEL_EXPS_DIR
    all_exp_names = [exp.name for exp in all_exps_dir.iterdir() if exp.is_dir()]
    if not exp_name in all_exp_names:
        raise FileNotFoundError(f"exp_name {exp_name} not found in {all_exps_dir}")
    else:
        return all_exps_dir / exp_name
