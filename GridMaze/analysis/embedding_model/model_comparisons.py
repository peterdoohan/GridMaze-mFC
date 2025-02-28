"""
Refactor: input_feature_comparisons.py
"""

# %% Imports
import json
import numpy as np
import pandas as pd

from GridMaze.analysis.embedding_model import plot_latents as pl
from GridMaze.analysis.embedding_model import load_experiment as le
from matplotlib import pyplot as plt
import seaborn as sns

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

EMBEDDING_MODEL_RESULTS = RESULTS_PATH / "embedding_model" / "exps"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]


# %% Loading Functions


def load_exp_set(exp_set, include_subjects=None, include_mazes=None):
    """
    Results saving convention: an exp_set is a series of embedding_model cross validation
    (see embedding_model/run_experiment.py) with different input features / model parameters
    that we want to compare.

    Args:
    """
    exp_set_dir = EMBEDDING_MODEL_RESULTS / exp_set
    exp_names = [d.name for d in exp_set_dir.iterdir() if d.is_dir()]
    if include_subjects is not None:
        exp_names = [exp for exp in exp_names if exp.split(".")[0] in include_subjects]
    if include_mazes is not None:
        exp_names = [exp for exp in exp_names if exp.split(".")[1] in include_mazes]
    results_dfs = []
    for exp_name in exp_names:
        subject, maze_name, abbrev = exp_name.split(".")  # naming convention: {subject}.{maze_n}.{abbrev}
        try:
            df = le.load_cluster_crossval_perf(exp_name, exp_set, average_over_folds=True, abbrev=abbrev)
            results_dfs.append(df)
        except:
            print(f"{exp_name} not found, probably still running. Returning None.")
            continue
    return pd.concat(results_dfs).reset_index(drop=True)


# %% Plotting Functions


def plot_model_comparison(results_df, normalise_to_best_model=False, ax=None, save_path=None):
    """ """
    if normalise_to_best_model:
        results_pivot = results_df.pivot(
            index=["subject_ID", "cluster_unique_ID"], columns="abbrev", values="cv_performance"
        )
        # get the best model for each subject
        subject2best_model = (
            results_df.groupby(["subject_ID", "abbrev"], observed=True)
            .cv_performance.mean()
            .unstack()
            .idxmax(axis=1)
            .to_dict()
        )
        # normalise performance to best model for each subject
        subject_norm_dfs = []
        for subject in results_df.subject_ID.unique():
            best_model = subject2best_model[subject]
            # results_pivot.loc[subject, :] = results_pivot.loc[subject,:].sub(results_pivot.loc[subject, best_model], axis=0)
            subject_norm = results_pivot.loc[subject, :].sub(results_pivot.loc[subject, best_model], axis=0)
            subject_norm.index = pd.MultiIndex.from_product(
                [[subject], subject_norm.index], names=["subject_ID", "cluster_unique_ID"]
            )
            subject_norm_dfs.append(subject_norm)
        results_df = pd.concat(subject_norm_dfs, axis=0).stack().reset_index(name="cv_performance")
    # order by mean performance
    model_ordering = results_df.groupby("abbrev", observed=False).cv_performance.mean().abs().sort_values().index
    # get subject mean performance
    subject_cond_mean_perf = (
        results_df.groupby(["subject_ID", "abbrev"], observed=True).cv_performance.mean().reset_index()
    )
    # plotting
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(len(model_ordering), 5), clear=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="black", lw=1, ls="--")
    if len(results_df.subject_ID.unique()) > 1:
        hue = "subject_ID"
        dodge = 0.4
    else:
        hue = None
        dodge = 0
    sns.pointplot(
        results_df,
        x="abbrev",
        y="cv_performance",
        order=model_ordering,
        hue=hue,
        dodge=dodge,
        linestyle="none",
        alpha=0.3,
        markeredgewidth=0,
        markersize=5,
        err_kws={"linewidth": 2},
        ax=ax,
    )
    if len(results_df.subject_ID.unique()) > 1:
        sns.pointplot(
            subject_cond_mean_perf,
            x="abbrev",
            y="cv_performance",
            order=model_ordering,
            marker="_",
            markersize=20,
            markeredgewidth=3,
            color="black",
            linestyle="none",
            errorbar=None,
            ax=ax,
        )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("")
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")


# %% Hack
def resave_results():
    """Load and resave subject level results into all subject dataframes and resave for convience."""
    exp_set = "validate_embedding_approach"
    for maze_name in MAZE_NAMES:
        for exp in ["productspace_input_no_embedding", "onehot_inputs_no_embedding"]:
            # load subject level data combine
            subject_dfs = []
            for subject in SUBJECT_IDS:
                exp_name = f"{subject}.{maze_name}.{exp}"
                try:
                    df = le.load_cluster_crossval_perf(exp_name, exp_set=exp_set, average_over_folds=False, abbrev=None)
                    subject_dfs.append(df)
                except:
                    print(f"{exp_name} not found")
                    continue
            if len(subject_dfs) == 0:
                print(f"Missing all subjects data from {maze_name}.{exp}")
                continue
            all_subject_df = pd.concat(subject_dfs)
            all_subject_df.reset_index(drop=True, inplace=True)
            # save combined data to appropraite location
            save_path = (
                EMBEDDING_MODEL_RESULTS
                / exp_set
                / f"all_subjects.{maze_name}.{exp}"
                / "cluster_cross_val_performance.htsv"
            )
            # check if file existis
            if save_path.exists():
                exisiting_df = pd.read_csv(save_path, sep="\t")
                exisiting_subjects = exisiting_df.subject_ID.unique()
                new_subjects = all_subject_df.subject_ID.unique()
                if set(exisiting_subjects) == set(new_subjects):
                    print(f"Skipping {maze_name}.{exp} as already saved.")
                    continue
                else:
                    print(f"Overwriting {maze_name}.{exp} as new subjects found.")
            all_subject_df.to_csv(save_path, sep="\t", index=False)
