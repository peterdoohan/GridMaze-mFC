"""
Library for anlysing distance-to-goal single cell tuning patterns across mazes.
"""

# %% Imports
import json
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore

from GridMaze.analysis.core import get_sessions as gs

from GridMaze.analysis.unit_match import get_across_maze_matches as mm
from GridMaze.analysis.distance_to_goal import population_tuning as dpt
from GridMaze.analysis.unit_match import place_direction as um_pd

# %% Global Variables

from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

from GridMaze.analysis.unit_match.get_across_maze_matches import MAZE_PAIR2VALID_DAYS

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

RESULTS_DIR = RESULTS_PATH / "unit_match" / "distance_to_goal"

MAZE_PAIRS = [("maze_1", "maze_2"), ("maze_2", "rooms_maze")]


# %% Corss maze heatmaps!


def plot_matched_distance_tuning_heatmaps(
    tc_A, tc_B, smooth_SD=2, normalisation_method="zscore", cmap="coolwarm", v_range=(-2, 2), axes=None
):
    """ """
    pos_mask = tc_A["gamma_4p"]["size"].ge(0).values
    neg_mask = tc_A["gamma_4p"]["size"].lt(0).values
    A_pos_df, B_pos_df = tc_A[pos_mask].copy(), tc_B[pos_mask].copy()
    A_neg_df, B_neg_df = tc_A[neg_mask].copy(), tc_B[neg_mask].copy()
    x = A_pos_df.distance_to_goal.columns.values.astype(float)
    # order pos
    pos_idx_max = A_pos_df.apply(lambda row: dpt.get_idx_order(row, x, fit="gamma_4p", op="max"), axis=1).values
    A_pos_df.loc[:, ("idx_max", "")] = pos_idx_max
    B_pos_df.loc[:, ("idx_max", "")] = pos_idx_max
    A_pos_df = A_pos_df.sort_values(by=[("idx_max", "")], ascending=True)
    B_pos_df = B_pos_df.sort_values(by=[("idx_max", "")], ascending=True)
    # order neg
    neg_idx_min = A_neg_df.apply(lambda row: dpt.get_idx_order(row, x, fit="gamma_4p", op="min"), axis=1).values
    A_neg_df.loc[:, ("idx_min", "")] = neg_idx_min
    B_neg_df.loc[:, ("idx_min", "")] = neg_idx_min
    A_neg_df = A_neg_df.sort_values(by=[("idx_min", "")], ascending=True)
    B_neg_df = B_neg_df.sort_values(by=[("idx_min", "")], ascending=True)
    A_pos, B_pos = A_pos_df.distance_to_goal.values, B_pos_df.distance_to_goal.values
    A_neg, B_neg = A_neg_df.distance_to_goal.values, B_neg_df.distance_to_goal.values
    # smooth and normalise
    if smooth_SD:
        A_pos, B_pos, A_neg, B_neg = [gaussian_filter1d(x, smooth_SD, axis=1) for x in [A_pos, B_pos, A_neg, B_neg]]
    if normalisation_method == "max":
        A_pos, B_pos = [x / np.max(x, axis=1)[:, None] for x in [A_pos, B_pos]]
        A_neg, B_neg = [x / np.max(x, axis=1)[:, None] for x in [A_neg, B_neg]]
    elif normalisation_method == "zscore":
        A_pos, B_pos = [zscore(x, axis=1) for x in [A_pos, B_pos]]
        A_neg, B_neg = [zscore(x, axis=1) for x in [A_neg, B_neg]]
    if axes is None:
        f, axes = plt.subplots(
            2,
            2,
            figsize=(6, 6),
            height_ratios=[1.5, 1],
            sharex=True,
        )
    for ax, data in zip(axes.flatten(), [A_pos, B_pos, A_neg, B_neg]):
        sns.heatmap(
            data,
            cmap=cmap,
            vmax=v_range[1],
            vmin=v_range[0],
            ax=ax,
            cbar_kws={"label": "Firing Rate (z-score)", "shrink": 0.5},
        )
        y_tick = len(data) // 100 * 100
        ax.set_yticks([y_tick])
        ax.set_yticklabels([f"{y_tick}"], rotation=90)
        ax.set_ylabel("Neurons", labelpad=-10)
        ax.set_xlabel("Shortest-path \n Distance to Goal (m)")
        ax.set_xticks(np.arange(0, len(x), 5))
        ax.set_xticklabels(np.arange(0, max(x), 0.25), rotation=0)


def get_matched_distance_tuning_dfs(
    min_split_half_corr=0.3,
    shuffle_matched_pairs=True,
    verbose=False,
):
    """ """
    # get all matched clusters
    all_matches = []
    for maze_pair in MAZE_PAIRS:
        for subject_ID in SUBJECT_IDS:
            matches = mm.get_cross_maze_matches(
                subject_ID,
                maze_pair,
                single_units=True,
                tuning_metric="distance_to_goal",
                min_split_half_corr=min_split_half_corr,
                return_as="cluster_unique_ID",
                verbose=verbose,
            )
            if len(matches) == 0:
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
    all_tuning_curves = []
    for maze_pair in MAZE_PAIRS:
        all_tuning_curves.append(
            get_tuning_curves(subject_ID="all", maze_pair=maze_pair, include_metrics=True, verbose=verbose)
        )
    all_tuning_curves = pd.concat(all_tuning_curves, axis=0)
    # index tuning curves
    tc_A = all_tuning_curves.loc[all_matches[:, 0]]
    tc_B = all_tuning_curves.loc[all_matches[:, 1]]
    return tc_A, tc_B


# %% Cross Maze correlations!


def plot_cross_maze_corrs_summary(results, print_stats=True, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2, 3))
    um_pd.plot_cross_maze_corrs_summary(results, print_stats, ax)
    ax.set_ylabel("distance-to-goal \n tuning corr.")


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
        true_corrs, permuted_corrs = get_cross_maze_distance_to_goal_corrs(
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


def get_cross_maze_distance_to_goal_corrs(
    subject_ID="m2", maze_pair=("maze_1", "maze_2"), min_split_half_corr=0.3, n_permutations=1_000, verbose=True
):
    # get true matched units
    if verbose:
        print("Loading true and permuted cross-maze cluster matches ...")
    true_matches = mm.get_cross_maze_matches(
        subject_ID,
        maze_pair,
        single_units=True,
        tuning_metric="distance_to_goal",
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
        tuning_metric="distance_to_goal",
        min_split_half_corr=min_split_half_corr,
    )
    # get tuning curves
    if verbose:
        print("Loading distance tuning curves ...")
    all_tuning_curves = get_tuning_curves(
        subject_ID=subject_ID,
        maze_pair=maze_pair,
        verbose=verbose,
    )
    # get true match corrs
    if verbose:
        print("Calculating true cross-maze correlations ...")
    tc_A = all_tuning_curves.loc[true_matches[:, 0]]
    tc_B = all_tuning_curves.loc[true_matches[:, 1]]
    true_corrs = distance_tuning_corrs(tc_A, tc_B, method="spearman")
    # get permuted cross-maze correlations
    if verbose:
        print("Calculating permuted cross-maze correlations ...")
    permuted_corrs = np.zeros((n_permutations, len(true_corrs)))
    for i, perm_matches in enumerate(permuted_matches):
        if verbose:
            print(i)
        perm_matches = np.array(permuted_matches[i])
        hm_A = all_tuning_curves.loc[perm_matches[:, 0]]
        hm_B = all_tuning_curves.loc[perm_matches[:, 1]]
        permuted_corrs[i] = distance_tuning_corrs(hm_A, hm_B, method="spearman")
    return true_corrs, permuted_corrs


def distance_tuning_corrs(tc_A, tc_B, method="spearman"):
    """ """
    n_matches = tc_A.shape[0]
    corrs = np.zeros(n_matches)
    for i in range(n_matches):
        corrs[i] = tc_A.iloc[i].corr(
            tc_B.iloc[i], method=method
        )  # pandas corr deals with nans (unvitised pds in either maze)
    return corrs


def get_tuning_curves(subject_ID="m2", maze_pair=("maze_1", "maze_2"), include_metrics=False, verbose=True):
    """ """
    subject_ID = [subject_ID] if subject_ID != "all" else "all"
    _maze_pair = f"{maze_pair[0]}.{maze_pair[1]}"
    with_data = [
        "navigation_df",
        "navigation_spike_rates_df",
        "cluster_metrics",
    ]
    if include_metrics:
        with_data.extend(["cluster_distance_tuning_metrics"])
    dfs = []
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
        if verbose:
            print(f"Generating distance-to-goal tuning curves...")
        for session in sessions:
            if verbose:
                print(session.name)
            df = dpt._get_session_distance_tuning(
                session,
                metrics=("distance_to_goal", "geodesic"),
                bin_spacing=0.05,
                max_steps_to_goal=30,
                moving_only=False,
            )
            if include_metrics:
                metrics_df = session.cluster_distance_tuning_metrics
                if metrics_df.index.name == "cluster_unique_ID":
                    metrics_df = metrics_df.reset_index()
                metrics_df = metrics_df[metrics_df.single_unit]
                metrics_df.set_index("cluster_unique_ID", inplace=True)
                df = pd.concat([df, metrics_df], axis=1)
            dfs.append(df)

    all_tuning_curves = pd.concat(dfs, axis=0)

    return all_tuning_curves


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
        f, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.spines[["top", "right"]].set_visible(False)
        for Clust, color in zip(pair, colors):
            Clust.plot_tuning(
                feature="distance_to_goal",
                ax=ax,
                feature_kwargs={"color": color},
            )
        ax.set_title(f"Match {i}")
        plt.show()
