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
    SUBJECT_IDS = json.load(f)

RESULTS_DIR = RESULTS_PATH / "place_direction" / "permuted_heatmaps"

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]


# %% control analysis for low dimensional structure in place-direction tuning


# def plot_permutation_test_results(true_value, permuted_values, xlabel="AUC", ax=None):
#     """ """
#     if ax is None:
#         f, ax = plt.subplots(figsize=(3, 1.5))
#     ax.spines[["top", "right"]].set_visible(False)
#     sns.histplot(permuted_values, ax=ax, color="gray", element="step", label="permuted")
#     ax.axvline(true_value, color="red", label="true")
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel("Count")


def get_true_vs_permuted_PC95(n_resamples=500, max_jobs=10, verbose=True):
    """
    Compare the number of principal components explaining 95% variance (and AUC of explained variance curve)
    in the true place-direction heatmaps to that of permuted heatmaps.
    subjects are bootstrap resampled for heatmaps going into true and permuted conditions. Need to combine heatmaps across subjects
    due to high number of place-direction pairs on each maze.
    """
    save_path = RESULTS_DIR / "true_vs_permuted_PC95.parquet"
    if save_path.exists():
        if verbose:
            print(f"Loading {save_path} from disk")
        results_df = pd.read_parquet(save_path)
        return results_df
    results = []
    for maze in MAZE_NAMES[:1]:
        print(f"Loading data for {maze} ...")
        population_tuning_df = pdr.get_population_place_direction_tuning(
            subject_IDs="all",
            maze_name="maze_1",
            late_sessions=True,
            fill_nans="mean",
            normalisation="length",
            min_split_corr=0.5,
        )
        permuted_heatmaps_df = load_permuted_place_direction_heatmaps(
            maze,
            subject_IDs="all",
            normalisation="length",
        )
        n_features = population_tuning_df.shape[1]
        n_permutations = permuted_heatmaps_df.index.get_level_values(2).max()
        # prestratify dfs by subject
        subject_data = {}
        for subject in SUBJECT_IDS:
            subject_data[subject] = {
                "true": population_tuning_df.xs(subject, level="subject_ID").reset_index(drop=True),
                "permuted": permuted_heatmaps_df.xs(subject, level="subject_ID").reset_index(level=0, drop=True),
            }
        # bootstrapped resample over subjects
        maze_results = Parallel(n_jobs=max_jobs)(
            delayed(_process_resample)(subject_data, n_features, n_permutations, maze, i, verbose)
            for i in range(n_resamples)
        )
        maze_results = [r for r in results if r is not None]  # filter out None results
        results.extend(maze_results)
    results_df = pd.DataFrame(results)
    # save results
    results_df.to_parquet(save_path)
    if verbose:
        print(f"Results saved to {save_path}")
    return results_df


def _process_resample(subject_data, n_features, n_permutations, maze, i, verbose):
    if verbose:
        print(i)
    sampled_subjects = np.random.choice(SUBJECT_IDS, size=len(SUBJECT_IDS), replace=True)
    # combine neurons across resampled subjects and calculate PC95
    true_df = pd.concat([subject_data[subject]["true"] for subject in sampled_subjects], axis=0)
    if true_df.shape[0] < n_features:
        print(f"Skipping {maze} for {sampled_subjects} as not enough features ({true_df.shape[0]} < {n_features})")
        return None
    true_PC95 = pca_n_components(true_df.values, target_variance=0.95)
    true_auc = pca_auc(true_df.values)
    # for each set of permuted heatmaps, combine neurons across resampled subjects and calculate PC95
    perms_df = pd.concat([subject_data[subject]["permuted"] for subject in sampled_subjects], axis=0)
    perm_PC95s, perm_aucs = [], []
    for j in range(n_permutations):
        _perm_df = perms_df.loc[j]
        permuted_PC95 = pca_n_components(_perm_df.values, target_variance=0.95)
        perm_PC95s.append(permuted_PC95)
        permuted_auc = pca_auc(_perm_df.values)
        perm_aucs.append(permuted_auc)
    mean_perm_PC95 = np.mean(perm_PC95s)
    mean_perm_auc = np.mean(perm_aucs)
    return {
        "maze": maze,
        "n": i,
        "true_PC95": true_PC95,
        "permuted_PC95": mean_perm_PC95,
        "true_auc": true_auc,
        "permuted_auc": mean_perm_auc,
    }


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
        place_direction_tuned=False,  # do all cluster filtering after the fact
        min_split_corr=None,  # better to match units to filtered real data at time of analysis
    )
    return place_direction_tuning
