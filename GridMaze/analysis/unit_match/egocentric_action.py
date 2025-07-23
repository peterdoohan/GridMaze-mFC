"""
Library for anlysing egocentric-action single cell tuning patterns across mazes.
"""

# %% Imports
import h5py
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore


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


# %% Matched cell tuning heatmaps


def plot_matched_egocentric_action_tuning_heatmap(
    A,
    B,
    smooth_SD=14,
    normalise="zscore",
    order_by="CV_pref_max",
    crop_window=False,
    actions=["turn_left", "turn_right", "go_forward"],
    fig=None,
    axes=None,
):
    """ """
    if axes is None or fig is None:
        f, axes = plt.subplots(3, 6, figsize=(10, 5))
    # unroll input
    (tc_A, metrics_A), (tc_B, metrics_B) = A, B
    # smooth
    if smooth_SD:
        tc_A = pd.DataFrame(gaussian_filter1d(tc_A.values, smooth_SD, axis=1), index=tc_A.index, columns=tc_A.columns)
        tc_B = pd.DataFrame(gaussian_filter1d(tc_B.values, smooth_SD, axis=1), index=tc_B.index, columns=tc_B.columns)
    # reshape to wide format (Fix below, dup entries means unstack won't work need to use .xs and concat)
    wide_A, wide_B = [
        tuning.unstack(level=1).swaplevel(1, 2, axis=1).sort_index(axis=1).action_aligned_rates
        for tuning in (tc_A, tc_B)
    ]
    # normalise
    if normalise == "zscore":
        wide_A = pd.DataFrame(zscore(wide_A, axis=1), index=wide_A.index, columns=wide_A.columns)
        wide_B = pd.DataFrame(zscore(wide_B, axis=1), index=wide_B.index, columns=wide_B.columns)
    else:
        raise NotImplementedError
    # order and plot each action separately
    for action in actions:
        action_A

    return


def get_matched_egocentric_action_tuning_df(
    min_split_half_corr=0.3,
    min_pref_action_factor=2,
    min_pref_action_frac=0.5,
    shuffle_matched_pairs=True,
    window=(-2, 2),
    actions=["turn_left", "turn_right", "go_forward"],
    verbose=True,
):
    """ """
    # get all matched clusters that past input criteria for acion tuning
    all_matches = []
    for maze_pair in MAZE_PAIRS:
        for subject_ID in SUBJECT_IDS:
            matches = mm.get_cross_maze_matches(
                subject_ID,
                maze_pair,
                single_units=True,
                tuning_metric="egocentric_action",
                tuning_metric_kwargs={
                    "pref_action_factor": min_pref_action_factor,
                    "pref_action_frac": min_pref_action_frac,
                },
                min_split_half_corr=min_split_half_corr,
                return_as="cluster_unique_ID",
                verbose=verbose,
            )
            if matches is None:
                continue
            else:
                all_matches.extend(matches)
    all_matches = np.array(all_matches)

    if shuffle_matched_pairs:
        shuffled = all_matches.copy()
        for i in range(shuffled.shape[0]):
            np.random.shuffle(shuffled[i])
        all_matches = shuffled

    # get all distance tuning curves
    all_tuning_curves, all_metrics_dfs = [], []
    for maze_pair in MAZE_PAIRS:
        tuning_df, metrics_df = get_tuning_curves(
            subject_ID="all",
            maze_pair=maze_pair,
            actions=actions,
            window=window,
            min_split_half_corr=min_split_half_corr,
            wide_format=False,
            verbose=verbose,
        )
        all_tuning_curves.append(tuning_df)
        all_metrics_dfs.append(metrics_df)
    all_tuning_curves = pd.concat(all_tuning_curves, axis=0)
    all_metrics_dfs = pd.concat(all_metrics_dfs, axis=0)
    # index tuning curves
    tc_A = all_tuning_curves.loc[all_matches[:, 0]]
    tc_B = all_tuning_curves.loc[all_matches[:, 1]]
    metrics_A = all_metrics_dfs.loc[all_matches[:, 0]]
    metrics_B = all_metrics_dfs.loc[all_matches[:, 1]]
    return (tc_A, metrics_A), (tc_B, metrics_B)


# %% Summary function


def plot_cross_maze_corrs_summary(results, print_stats=True, min_matches=10, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2, 3))
    um_pd.plot_cross_maze_corrs_summary(results, print_stats, min_matches, ax)
    ax.set_ylabel("egocentric-action \n tuning corr.")


def get_cross_maze_corr_summary(
    maze_pair=("maze_1", "maze_2"),
    min_split_half_corr=0.3,
    n_permutations=1_0,
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
        if true_corrs is None:
            continue  # no matches found for this subject
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
    if true_matches is None:
        if verbose:
            print(f"No matches found for {subject_ID} on {maze_pair[0]} and {maze_pair[1]}")
        return None, None
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
    tuning_curves, _ = get_tuning_curves(subject_ID=subject_ID, maze_pair=maze_pair, wide_format=True, verbose=verbose)
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
    wide_format=True,
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
        if wide_format:
            wide_df = tuning_df.unstack().swaplevel(1, 2, axis=1).sort_index(axis=1)
            tuning_dfs.append(wide_df)
        else:
            tuning_dfs.append(tuning_df)
        metric_dfs.append(metrics_df)
    return pd.concat(tuning_dfs, axis=0), pd.concat(metric_dfs, axis=0)


# %% Misc


def plot_all_subject_matched_clusters(subject="m2", maze_pair=("maze_1", "maze_2")):
    """
    Search through matches to find some nice pairs!
    """
    colors = ["royalblue", "darkviolet"]
    matched_clusters = mm.get_cross_maze_matches(
        subject,
        maze_pair,
        single_units=True,
        tuning_metric="distance_to_goal",
        min_split_half_corr=0.3,
        return_as="cluster_objects",
        verbose=True,
    )
    for i, pair in enumerate(matched_clusters):
        f, axes = plt.subplots(1, 2, figsize=(6, 3))
        for Clust, ax in zip(pair, axes):
            Clust.plot_tuning(
                feature="actions",
                feature_kwargs={
                    "concise": True,
                    "action_type": "all",
                    "smooth_SD": 14,
                    "colors": ["darkviolet", "royalblue", "grey"],
                },
                ax=ax,
            )
            ax.set_title(Clust.cluster_unique_ID)
        plt.show()
