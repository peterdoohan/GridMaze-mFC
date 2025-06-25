"""
Library for anlysing place-direction single cell tuning patterns across mazes.
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import h5py

from GridMaze.maze import representations as mr
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.place_direction import dimensionality_reduction as pdr
from GridMaze.analysis.unit_match import get_across_maze_matches as mm

# %% Global Variables
from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

from GridMaze.analysis.unit_match.get_across_maze_matches import MAZE_PAIR2VALID_DAYS

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

RESULTS_DIR = RESULTS_PATH / "unit_match" / "place_direction"


# %% true vs permuted place-direction tuning correlation across mazes


def get_cross_maze_corr_summary(
    maze_pair=("maze_1", "maze_2"),
    single_units=True,
    min_split_half_corr=None,
    n_permutations=1_000,
    save=True,
    verbose=True,
):
    """ """
    _maze_pair = f"{maze_pair[0]}.{maze_pair[1]}"
    save_path = RESULTS_DIR / f"{_maze_pair}.corr_summary.h5"
    if not save and save_path.exists():
        if verbose:
            print(f"Loading cross-maze correlation summary from {save_path}")
        out = {}
        with h5py.File(save_path, "r") as f:
            for subj_id, grp in f.items():
                out[subj_id] = {"true_corrs": grp["true_corrs"][()], "permuted_corrs": grp["permuted_corrs"][()]}
        return out
    results = {}
    for subject_ID in SUBJECT_IDS:
        if verbose:
            print(f"processing subject: {subject_ID}")
        true_corrs, permuted_corrs = get_cross_maze_place_direction_corrs(
            subject_ID=subject_ID,
            maze_pair=maze_pair,
            single_units=single_units,
            min_split_half_corr=min_split_half_corr,
            n_permutations=n_permutations,
            verbose=verbose,
        )
        results[subject_ID] = {
            "true_corrs": true_corrs,
            "permuted_corrs": permuted_corrs,
        }
    if save:
        if verbose:
            print(f"Saving cross-maze correlation summary to {save_path}")
        with h5py.File(save_path, "w") as f:
            for subject_ID, subject_results in results.items():
                grp = f.create_group(subject_ID)
                grp.create_dataset("true_corrs", data=subject_results["true_corrs"])
                grp.create_dataset("permuted_corrs", data=subject_results["permuted_corrs"])
    return results


def get_cross_maze_place_direction_corrs(
    subject_ID="m2",
    maze_pair=("maze_1", "maze_2"),
    single_units=True,
    min_split_half_corr=None,
    n_permutations=1_000,
    verbose=True,
):
    """ """
    # get matches units
    if verbose:
        print("Loading true and permuted cross-maze cluster matches ...")
    tuning_metric = None if min_split_half_corr is None else "place_direction"
    true_matches = mm.get_cross_maze_matches(
        subject_ID,
        maze_pair,
        single_units,
        tuning_metric,
        min_split_half_corr,
        return_as="cluster_unique_ID",
        verbose=verbose,
    )
    permuted_matches = mm.get_permuted_cross_maze_matches(
        subject_ID, maze_pair, n_permutations, single_units, tuning_metric, min_split_half_corr
    )
    # get all heatmaps
    if verbose:
        print("Loading place-direction heatmaps ...")
    all_heatmaps = get_heatmaps(subject_ID, maze_pair)
    # get true cross-maze correlation
    if verbose:
        print("Calculating true cross-maze correlations ...")
    true_matches = np.array(true_matches)
    hm_A = all_heatmaps.loc[true_matches[:, 0]]
    hm_B = all_heatmaps.loc[true_matches[:, 1]]
    true_corrs = get_heatmap_corrs(hm_A, hm_B, method="spearman")
    # get permuted cross-maze correlations
    if verbose:
        print("Calculating permuted cross-maze correlations ...")
    permuted_corrs = np.zeros((n_permutations, len(true_corrs)))
    for i, perm_matches in enumerate(permuted_matches):
        if verbose:
            print(i)
        perm_matches = np.array(permuted_matches[i])
        hm_A = all_heatmaps.loc[perm_matches[:, 0]]
        hm_B = all_heatmaps.loc[perm_matches[:, 1]]
        permuted_corrs[i] = get_heatmap_corrs(hm_A, hm_B, method="spearman")
    return true_corrs, permuted_corrs


# %% get heatmaps


def get_heatmap_corrs(hm_A, hm_B, method="spearman"):
    """
    correlates each row of hm_A with the corresponding row of hm_B
    row = heatmap (raw firing rates with NaNs in unvisited place-directions)
    """
    # not the most efficient way but can easily handle nans
    n_matches = hm_A.shape[0]
    corrs = np.zeros(n_matches)
    for i in range(n_matches):
        corrs[i] = hm_A.iloc[i].corr(
            hm_B.iloc[i], method=method
        )  # pandas corr deals with nans (unvitised pds in either maze)
    return corrs


def get_heatmaps(subject_ID="m2", maze_pair=("maze_1", "maze_2"), verbose=False):
    """
    returns all possible relevant place-direction heatmaps for a subject and pair of mazes
    as a df, with filtered place-direction paris common to both mazes
    """
    # get place directions that are common to both mazes
    simple_maze_A, simple_maze_B = [mr.get_simple_maze(m) for m in maze_pair]
    common_place_directions = get_common_maze_place_directions(simple_maze_A, simple_maze_B)
    # for each maze in the pair, generate the place-direction heatamps for comparison
    _maze_pair = f"{maze_pair[0]}.{maze_pair[1]}"
    dfs = []
    for maze in maze_pair:
        if verbose:
            print(f"Loading sessions for {subject_ID} on {maze} maze")
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names=[maze],
            days_on_maze=MAZE_PAIR2VALID_DAYS[_maze_pair][maze],
            with_data=[
                "navigation_df",
                "navigation_spike_rates_df",
                "cluster_metrics",
                "cluster_place_direction_tuning_metrics",
            ],
            must_have_data=True,
        )
        if verbose:
            print(f"generating place-direction heatmaps")
        maze_heatmaps = pdr.get_population_place_direction_tuning(
            sessions=sessions,
            include_multi_unit=True,  # sometimes units are single unit in one recording and multi-unit in another
            fill_nans=False,
            normalisation=False,
            min_split_corr=None,
            max_steps_to_goal=30,
            place_direction_tuned=False,
            verbose=verbose,
        )
        # filter for the common place-directions
        maze_heatmaps = maze_heatmaps[common_place_directions]
        dfs.append(maze_heatmaps)
    # combine heatmaps from both mazes and return
    all_heatmaps = pd.concat(dfs, axis=0)
    all_heatmaps = all_heatmaps.droplevel(1, axis=0).sort_index(axis=1)
    return all_heatmaps


def get_common_maze_place_directions(maze_A, maze_B):
    """
    input twos simple maze nx objects
    returns place-direction pairs that are common to both mazes
    """
    A_pd = mr.get_maze_place_direction_pairs(maze_A)
    B_pd = mr.get_maze_place_direction_pairs(maze_B)
    common_pds = set(A_pd).intersection(set(B_pd))
    return list(common_pds)
