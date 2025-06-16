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
from GridMaze.analysis.behaviour import dimensionality_reduction as bdr
from GridMaze.analysis.place_direction import efficient_coding as ec

# %% Global Variables
from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

RESULTS_DIR = RESULTS_PATH / "place_direction" / "permuted_heatmaps"

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]

# %% control analysis of permuted place-direction heatmaps expalining the low d strucutre of behaviour


def test(maze_name="maze_1", subject_IDs="all"):
    """
    not cross validataed
    interesting result permuted heatmaps to a better job of explaing variance in
    the behaviour that the real data...
    """
    # load real data
    population_tuning_df = pdr.get_population_place_direction_tuning(
        subject_IDs=subject_IDs,
        maze_name=maze_name,
        late_sessions=True,
        fill_nans="mean",
        normalisation="length",
    )
    behavioural_sequences_df = bdr.get_maze_behavioural_sequences_df(
        subject_IDs, maze_name, late_sessions=True, normalisation="length"
    )
    N = population_tuning_df.values
    B = behavioural_sequences_df.values
    # load permuted data
    permuted_heatmaps_df = load_permuted_place_direction_heatmaps(maze_name, subject_IDs)
    n_permutations = permuted_heatmaps_df.index.get_level_values(2).max()

    # get auc of neurons explain behaviour for every set of permuted heatmaps
    def get_auc(N, B):
        cumsum = ec.get_pca_variance_explained(N, B)
        auc = np.trapz(cumsum, dx=1 / len(cumsum))
        return auc

    permuted_aucs = []
    for i in range(100):
        print(i)
        permuted_df = permuted_heatmaps_df.xs(i, level="permutation", drop_level=False)
        N_perm = permuted_df.values
        auc = get_auc(N_perm, B)
        permuted_aucs.append(auc)
    permuted_aucs = np.array(permuted_aucs)
    # get true value
    true_auc = get_auc(N, B)
    return true_auc, permuted_aucs


# %% control analysis for low dimensional structure in place-direction tuning


def plot_permutation_test_results(true_value, permuted_values, xlabel="AUC", ax=None):
    """ """
    if ax is None:
        f, ax = plt.subplots(figsize=(3, 1.5))
    ax.spines[["top", "right"]].set_visible(False)
    sns.histplot(permuted_values, ax=ax, color="gray", element="step", label="permuted")
    ax.axvline(true_value, color="red", label="true")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")


def get_permuted_place_direction_PC95_comparison(verbose=False):
    """
    how many PCs are needed to explain 95% of the variance in the population place-direction tuning heatmaps
    in true data and in permuted data? Done separately for each subject and maze.

    returns a dict with the following structure:
    {"maze_name": {"subject_ID": {"true": true_value, "permuted": permuted_values}}}
    """
    results = {}
    for maze_name in MAZE_NAMES:
        if verbose:
            print(f"Processing {maze_name} ...")
        subject_results = {}
        for subject in SUBJECT_IDS:
            if verbose:
                print(f"Processing subject {subject} ...")
            population_tuning_df = pdr.get_population_place_direction_tuning(
                subject_IDs=[subject],
                maze_name=maze_name,
                late_sessions=True,
                fill_nans="mean",
                normalisation="length",
            )
            permuted_heatmaps_df = load_permuted_place_direction_heatmaps(maze_name, [subject])
            n_permutations = permuted_heatmaps_df.index.get_level_values(2).max()
            perm_PC95_values = []
            for i in range(n_permutations + 1):
                permuted_df = permuted_heatmaps_df.xs(i, level="permutation", drop_level=False)
                perm_PC95_values.append(pca_n_components(permuted_df.values, target_variance=0.95))
            true_value = pca_n_components(population_tuning_df.values)
            subject_results[subject] = {
                "true": true_value,
                "permuted": perm_PC95_values,
            }
        results[maze_name] = subject_results
    return results


def pca_n_components(X, target_variance=0.95):
    """ """
    pca = PCA(random_state=0)
    pca.fit(X)
    explained_variance = pca.explained_variance_ratio_
    cumsum = np.cumsum(explained_variance)
    n_components = np.searchsorted(cumsum, target_variance) + 1
    return n_components


def pca_auc(X):
    """
    returns the AUC of the PCA explained variance curve for a given matrix,
    not cross validated
    """
    pca = PCA(random_state=0)
    pca.fit(X)
    explained_variance = pca.explained_variance_ratio_
    cumsum = np.cumsum(explained_variance) / np.sum(explained_variance)
    auc = np.trapz(cumsum, dx=1 / len(explained_variance))
    return auc


# %% load data from disk


def load_permuted_place_direction_heatmaps(
    maze_name,
    subject_IDs="all",
    normalisation="length",
):
    """
    later add options to fill nans and normalise at this level
    """
    save_path = RESULTS_DIR / f"{maze_name}.parquet"
    if not save_path.exists():
        raise FileNotFoundError(f"Permuted place direction heatmaps for {maze_name} not found at {save_path}.")
    df = pd.read_parquet(save_path)
    if not subject_IDs == "all":
        df = df.loc[:, subject_IDs, :]
    if normalisation:
        if normalisation == "mean":
            df = df.div(df.mean(axis=1), axis=0)
        elif normalisation == "length":
            df = df.div(df.pow(2).sum(axis=1).pow(0.5), axis=0)
        elif normalisation == "max":
            df = df.div(df.max(axis=1), axis=0)
        else:
            raise ValueError(f"Unknown normalisation method: {normalisation}")
    return df


# %% Generate permuted heatmap functions


def populate_permuted_place_direction_heatmaps(n_permutation=5_000, max_jopbs=15, verbose=True, overwrite=False):
    """
    Note that place_direction heatmaps are saved out normalised to length one, should
    remove this a repopulation TODO
    """

    for maze in MAZE_NAMES:
        save_path = RESULTS_DIR / f"{maze}.parquet"
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
        permuted_heatmaps_df.to_parquet(save_path)
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
        session,
        navigation_rates_df,
        fill_nans="mean",
        normalisation=False,
    )
    return place_direction_tuning
