"""
Library for anlysing egocentric-action single cell tuning patterns across mazes.
"""

# %% Imports
import h5py
import json
import numpy as np
import pandas as pd


from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.unit_match import get_across_maze_matches as mm
from GridMaze.analysis.egocentric_action import population_tuning as ept
from GridMaze.analysis.unit_match import place_direction as um_pd

# %% Global Variables

from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

from GridMaze.analysis.unit_match.get_across_maze_matches import MAZE_PAIR2VALID_DAYS

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

RESULTS_DIR = RESULTS_PATH / "unit_match" / "egocentric_action"

MAZE_PAIRS = [("maze_1", "maze_2"), ("maze_2", "rooms_maze")]


# %% Summary function


def get_cross_maze_corr_summary(
    maze_pair=("maze_1", "maze_2"),
    min_split_half_corr=0.3,
    n_permutations=1_000,
    save=False,
    verbose=False,
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
        true_corrs, permuted_corrs = get_cross_maze_egocentric_action_tuning_corrs(
            subject_ID=subject_ID,
            maze_pair=maze_pair,
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


# %% main corr functions


def get_cross_maze_egocentric_action_tuning_corrs(
    subject_ID="m2", maze_pair=("maze_1", "maze_2"), min_split_half_corr=0.3, n_permutations=1_000, verbose=True
):
    """ """
    if verbose:
        print("Loading true and permuted cross-maze cluster matches ...")
    true_matches = mm.get_cross_maze_matches(
        subject_ID,
        maze_pair,
        single_units=True,
        tuning_metric="egocentric_action",
        min_split_half_corr=min_split_half_corr,
        return_as="cluster_unique_ID",
        verbose=verbose,
    )
    true_matches = np.array(true_matches)
    # get permuted matches
    permuted_matches = mm.get_permuted_cross_maze_matches(
        subject_ID,
        maze_pair,
        n_permutations,
        single_units=True,
        tuning_metric="egocentric_action",
        min_split_half_corr=min_split_half_corr,
    )
    # get tuning curves
    if verbose:
        print("Loading ego-action tuning curves ...")
    tuning_curves, metrics_df = get_tuning_curves(subject_ID=subject_ID, maze_pair=maze_pair, verbose=verbose)
    # get true match corrs
    if verbose:
        print("Calculating true cross-maze correlations ...")
    tc_A = tuning_curves.loc[true_matches[:, 0]]
    tc_B = tuning_curves.loc[true_matches[:, 1]]
    true_corrs = tuning_curve_corrs(tc_A, tc_B, method="spearman")
    # get permuted match corrs
    if verbose:
        print("Calculating permuted cross-maze correlations ...")
        permuted_corrs = np.zeros((n_permutations, len(true_corrs)))
    for i, perm_matches in enumerate(permuted_matches):
        if verbose:
            print(i)
        perm_matches = np.array(permuted_matches[i])
        tc_A = tuning_curves.loc[perm_matches[:, 0]]
        tc_B = tuning_curves.loc[perm_matches[:, 1]]
        permuted_corrs[i] = tuning_curve_corrs(tc_A, tc_B, method="spearman")
    return true_corrs, permuted_corrs


def tuning_curve_corrs(tc_A, tc_B, method="spearman"):
    """ """
    n_matches = tc_A.shape[0]
    corrs = np.zeros(n_matches)
    for i in range(n_matches):
        corrs[i] = tc_A.iloc[i].corr(
            tc_B.iloc[i], method=method
        )  # pandas corr deals with nans (unvitised pds in either maze)
    return corrs


def get_tuning_curves(
    subject_ID="m2",
    maze_pair=("maze_1", "maze_2"),
    actions=["turn_left", "turn_right", "go_forward"],
    window=(-2, 2),
    min_split_half_corr=0.3,
    verbose=True,
):
    """
    note no smoothing
    """
    subject_ID = [subject_ID] if subject_ID != "all" else "all"
    _maze_pair = f"{maze_pair[0]}.{maze_pair[1]}"
    with_data = [
        "navigation_df",
        "navigation_spike_rates_df",
        "cluster_metrics",
        "cluster_egocentric_action_tuning_metrics",
    ]
    tuning_dfs, metric_dfs = [], []
    for maze in maze_pair:
        if verbose:
            print(f"Loading sessions for {subject_ID} on {maze} maze")
        sessions = gs.get_maze_sessions(
            subject_IDs=subject_ID,
            maze_names=[maze],
            days_on_maze=MAZE_PAIR2VALID_DAYS[_maze_pair][maze],
            with_data=with_data,
            must_have_data=True,
        )
        tuning_df, metrics_df = ept.get_population_egocentric_action_tuning(
            sessions=sessions,
            actions=actions,
            window=window,
            min_split_half_corr=min_split_half_corr,
            max_jobs=10,
            with_metrics=True,
            verbose=verbose,
        )
        wide_df = tuning_df.unstack().swaplevel(1, 2, axis=1).sort_index(axis=1)
        tuning_dfs.append(wide_df)
        metric_dfs.append(metrics_df)
    return pd.concat(tuning_dfs, axis=0), pd.concat(metric_dfs, axis=0)
