"""
Lib for generating permuted place-direction heatmaps for control analyses
"""

# %% Imports
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import permute
from GridMaze.analysis.place_direction import dimensionality_reduction as pdr

# %% Global Variables
from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    subject_IDs = json.load(f)

RESULTS_DIR = RESULTS_PATH / "place_direction" / "permuted_heatmaps"

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]
# %% control analysis for low dimensional structure in place-direction tuning


def plot_permuted_place_direction_pca_auc(true_value, permuted_values, ax=None):
    """ """
    if ax is None:
        f, ax = plt.subplots(figsize=(3, 1.5))
    ax.spines[["top", "right"]].set_visible(False)
    sns.histplot(permuted_values, ax=ax, color="gray", element="step", label="permuted")
    ax.axvline(true_value, color="red", label="true")
    ax.set_xlabel("AUC")
    ax.set_ylabel("Count")


def get_permuted_place_direction_pca_auc(maze_name="maze_1", subject_IDs="all"):
    """ """
    population_tuning_df = pdr.get_population_place_direction_tuning(
        subject_IDs=subject_IDs, maze_name=maze_name, late_sessions=True
    )
    permuted_heatmaps_df = load_permuted_place_direction_heatmaps(maze_name, subject_IDs)
    n_permutations = permuted_heatmaps_df.index.get_level_values(2).max()
    auc_values = []
    for i in range(n_permutations + 1):
        permuted_df = permuted_heatmaps_df.xs(i, level="permutation", drop_level=False)
        auc = pca_auc(permuted_df.values)
        auc_values.append(auc)
    permuted_auc_values = np.array(auc_values)
    true_auc_value = pca_auc(population_tuning_df.values)
    return true_auc_value, permuted_auc_values


def pca_auc(X):
    """
    returns the AUC of the PCA explained variance curve for a given matrix,
    not cross validated
    """
    pca = PCA(random_state=0)
    pca.fit(X)
    explained_variance = pca.explained_variance_ratio_
    auc = np.trapz(np.cumsum(explained_variance), dx=1 / len(explained_variance))
    return auc


# %% load data from disk


def load_permuted_place_direction_heatmaps(maze_name, subject_IDs="all"):
    """ """
    save_path = RESULTS_DIR / f"{maze_name}.parquet"
    if not save_path.exists():
        raise FileNotFoundError(f"Permuted place direction heatmaps for {maze_name} not found at {save_path}.")
    df = pd.read_parquet(save_path)
    if not subject_IDs == "all":
        df = df.loc[:, subject_IDs, :]
    return df


# %% Generate permuted heatmap functions


def populate_permuted_place_direction_heatmaps(n_permutation=5_000, max_jopbs=15, verbose=True, overwrite=False):
    """
    Note update to save as parquet TODO
    """

    for maze in MAZE_NAMES:
        save_path = RESULTS_DIR / f"{maze}.csv"
        if save_path.exists() and not overwrite:
            if verbose:
                print(f"File {save_path} already exists. Skipping.")
            continue
        else:
            if verbose:
                print(f"Generating permuted place direction heatmaps for {maze} ...")
        permuted_heatmaps_df = get_permuted_population_place_direction_tuning(
            maze, n_permutation, verbose=verbose, max_jobs=max_jopbs
        )
        permuted_heatmaps_df.to_csv(save_path)
    return print(f"Permuted place direction heatmaps saved to {RESULTS_DIR}.")


def get_permuted_population_place_direction_tuning(
    maze_name="maze_1",
    n_permutations=15,
    late_sessions=True,
    verbose=True,
    max_jobs=15,
):
    """ """
    # if session objects are not input, generate them from input filters
    days_on_maze = "late" if late_sessions else "all"
    if verbose:
        print("Loading sessions ...")
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names=[maze_name],
        days_on_maze=days_on_maze,
        with_data=[
            "navigation_df",
            "navigation_spike_rates_df",
            "cluster_metrics",
            "cluster_place_direction_tuning_metrics",
        ],
        must_have_data=True,
    )

    permuted_dfs = Parallel(n_jobs=max_jobs)(
        delayed(_process_permutation)(sessions, i, verbose) for i in range(n_permutations)
    )
    output_df = pd.concat(permuted_dfs, axis=0)
    return output_df


def _process_permutation(sessions, i, verbose):
    """ """
    if verbose:
        print(f"permutation {i}")
    dfs = []
    for session in sessions:
        if verbose:
            print(session.name)
        df = get_session_permuted_place_direction_tuning(session)
        if df is None:
            continue  # not pd tuned clusters
        df.index.name = "cluster_unique_ID"
        df[("subject_ID", "")] = session.subject_ID
        df[("permutation", "")] = i
        df.set_index(["subject_ID", "permutation"], append=True, inplace=True)
        dfs.append(df)
    pop_pd_tuning_df = pd.concat(dfs, axis=0)
    return pop_pd_tuning_df


def get_session_permuted_place_direction_tuning(session):
    # load data
    navigation_df = session.navigation_df
    rates_df = session.navigation_spike_rates_df
    # circularly permute spikes relative to behaviour
    _rates_df = permute.random_circular_shift(rates_df)
    # get place_direction heatmaps from permuted data
    navigation_rates_df = pd.concat([navigation_df, _rates_df.reset_index(drop=True)], axis=1)
    place_direction_tuning = pdr.get_session_place_direction_tuning(
        session, navigation_rates_df, fill_nans="mean", normalisation="length"
    )
    return place_direction_tuning
