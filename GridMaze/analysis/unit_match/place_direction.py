"""
Library for anlysing place-direction single cell tuning patterns across mazes.
"""

# %% Imports
import json
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel

from GridMaze.maze import representations as mr
from GridMaze.maze import plotting as mp
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.unit_match import get_across_maze_matches as mm
from GridMaze.analysis.place_direction import dimensionality_reduction as pdr

# %% Global Variables
from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

from GridMaze.analysis.unit_match.get_across_maze_matches import MAZE_PAIR2VALID_DAYS

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

RESULTS_DIR = RESULTS_PATH / "unit_match" / "place_direction"

MAZE_PAIRS = [("maze_1", "maze_2"), ("maze_2", "rooms_maze")]

# %% cross-maze NMF/PCA components


def plot_cross_maze_pca_components(mhm_df, maze_pair, n_components=10, cmap="coolwarm", axes=None):
    """ """
    # setup fig
    if axes is None:
        f, axes = plt.subplots(2, n_components, figsize=(6 * n_components, 12))
    # process data
    nmf_df = pdr.get_pca_df(mhm_df, n_components=n_components)
    simple_maze_A, simple_maze_B = [mr.get_simple_maze(m) for m in maze_pair]
    for i in range(n_components):
        c = nmf_df[i]
        for j, simple_maze in enumerate([simple_maze_A, simple_maze_B]):
            _c = c.loc[maze_pair[j]]
            mp.plot_directed_heatmap(simple_maze, _c, axes[j, i], colormap=cmap, allow_negative=True)
    axes[0, 0].set_title(f"{maze_pair[0]} maze")
    axes[1, 0].set_title(f"{maze_pair[1]} maze")
    return


def plot_cross_maze_nmf_components(mhm_df, maze_pair, n_components=10, cmap="Reds", axes=None):
    """ """
    # setup fig
    if axes is None:
        f, axes = plt.subplots(2, n_components, figsize=(6 * n_components, 12))
    # process data
    nmf_df = pdr.get_nmf_df(mhm_df, n_components=n_components)
    simple_maze_A, simple_maze_B = [mr.get_simple_maze(m) for m in maze_pair]
    for i in range(n_components):
        c = nmf_df[i]
        for j, simple_maze in enumerate([simple_maze_A, simple_maze_B]):
            _c = c.loc[maze_pair[j]]
            mp.plot_directed_heatmap(simple_maze, _c, axes[j, i], colormap=cmap)
    axes[0, 0].set_title(f"{maze_pair[0]}")
    axes[1, 0].set_title(f"{maze_pair[1]}")


def get_matched_heatmaps_df(
    maze_pair=("maze_1", "maze_2"), min_split_half_corr=0.3, fill_nans="mean", normalisation="length", verbose=False
):
    """ """
    # load heatmaps
    heatmaps = []
    _maze_pair = f"{maze_pair[0]}.{maze_pair[1]}"
    for maze in maze_pair:
        if verbose:
            print(f"Loading sessions for {maze} maze")
        sessions = gs.get_maze_sessions(
            subject_IDs="all",
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
        heatmaps.append(
            pdr.get_population_place_direction_tuning(
                sessions=sessions,
                include_multi_unit=False,
                fill_nans=fill_nans,  # for later DR
                normalisation=normalisation,  # for later DR
                min_split_corr=min_split_half_corr,
                place_direction_tuned=False,
                max_steps_to_goal=30,
                verbose=False,
            )
        )
    heatmaps_A, heatmaps_B = heatmaps
    # load all clusters matched across maze pair
    all_matches = []
    for subject_ID in SUBJECT_IDS:
        matches = mm.get_cross_maze_matches(
            subject_ID,
            maze_pair,
            single_units=True,
            tuning_metric="place_direction",
            min_split_half_corr=min_split_half_corr,
            return_as="cluster_unique_ID",
            verbose=verbose,
        )
        all_matches.extend(matches)
    all_matches = np.array(all_matches)
    # index matched clusters in maze_A, maze_B heatmaps
    mhm_A = heatmaps_A.loc[all_matches[:, 0]].droplevel(1, axis=0).reset_index(drop=True)
    mhm_A.columns = pd.MultiIndex.from_tuples([(maze_pair[0], *c) for c in mhm_A.columns])
    mhm_B = heatmaps_B.loc[all_matches[:, 1]].droplevel(1, axis=0).reset_index(drop=True)
    mhm_B.columns = pd.MultiIndex.from_tuples([(maze_pair[1], *c) for c in mhm_B.columns])
    # combine
    mhm_df = pd.concat([mhm_A, mhm_B], axis=1)
    return mhm_df


# %% true vs permuted place-direction tuning correlation across mazes


def plot_cross_maze_corrs_summary2(results, print_stats=True, min_matches=10, ax=None):
    """ """
    # setup fig
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(1, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_ylabel("match correlation \n place-direction tuning")
    ax.set_xlabel("match type")

    # compute subject-level means
    rows = []
    for subject_ID, data in results.items():
        true = data["true_corrs"]
        if len(true) < min_matches:
            continue
        true_mean = np.mean(true)
        permuted_mean = np.nanmean(data["permuted_corrs"])  # mean over all pairs and permutations
        rows.append({"subject": subject_ID, "condition": "true", "corr": true_mean})
        rows.append({"subject": subject_ID, "condition": "permuted", "corr": permuted_mean})
    df = pd.DataFrame(rows)
    # plot individual subject values behind
    sns.stripplot(
        data=df,
        x="condition",
        y="corr",
        color="grey",
        alpha=0.5,
        size=5,
        ax=ax,
        zorder=1,
    )
    # plot mean +/- SEM across subjects
    sns.pointplot(
        data=df,
        x="condition",
        y="corr",
        color="k",
        errorbar="se",
        capsize=0,
        linestyle="none",
        ax=ax,
        zorder=2,
    )
    if print_stats:
        _get_stats(results)
    return


def plot_cross_maze_corrs_summary(results, print_stats=True, min_matches=10, ax=None):
    """ """
    # setup fig
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    colors = sns.color_palette("hls", len(SUBJECT_IDS))
    offset = 0.05
    for i, (subject_ID, data) in enumerate(results.items()):
        true = data["true_corrs"]  # 1
        if len(true) < min_matches:
            continue
        true_mean = np.mean(true)
        true_sem = np.std(true) / np.sqrt(len(true))
        permuted_means = np.nanmean(data["permuted_corrs"], axis=1)  # n_permutations
        permuted_grand_mean = np.nanmean(permuted_means)
        permuted_lower_CI = np.percentile(permuted_means, 2.5)
        permuted_upper_CI = np.percentile(permuted_means, 97.5)
        ax.errorbar(
            0 + i * offset,
            true_mean,
            yerr=true_sem,
            fmt="o",
            color=colors[i],
            label=subject_ID,
            capsize=0,
            elinewidth=2,
        )
        ax.errorbar(
            1 + i * offset,
            permuted_grand_mean,
            yerr=[[permuted_grand_mean - permuted_lower_CI], [permuted_upper_CI - permuted_grand_mean]],
            fmt="o",
            color=colors[i],
            capsize=0,
            elinewidth=2,
        )
    x_mid = len(SUBJECT_IDS) * offset / 2
    ax.legend()
    ax.set_xlim(-1 * x_mid, 1 + 3 * x_mid)
    ax.set_xticks([0 + x_mid, 1 + x_mid])
    ax.set_xticklabels(["True", "Permuted"])
    ax.set_ylabel("place-direction tuning corr.")
    ax.set_xlabel("cross-maze\nmatched neurons")
    if print_stats:
        _get_stats(results)


def _get_stats(results):
    """ """
    # get subject p-values
    df = pd.DataFrame(index=SUBJECT_IDS, columns=["p_value"])
    for subject_ID, data in results.items():
        true = np.mean(data["true_corrs"])
        permuted_means = np.nanmean(data["permuted_corrs"], axis=1)
        df.loc[subject_ID, "p_value"] = (true < permuted_means).mean()
    print("Subject p-values:")
    print(df)
    # get random effects p-value
    true = [np.mean(data["true_corrs"]) for data in results.values()]
    permuted = [np.mean(np.nanmean(data["permuted_corrs"], axis=1)) for data in results.values()]
    t_stat, p_value = ttest_rel(true, permuted)
    print(f"Random effects t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")


def get_all_cross_maze_corrs():
    results = []
    for maze_pair in MAZE_PAIRS:
        results.append(get_cross_maze_corr_summary(maze_pair))
    # combine results into single dict
    all_results = {}
    for subject_ID in SUBJECT_IDS:
        true = []
        permuted = []
        for r in results:
            if subject_ID not in r.keys():
                continue
            t_array = r[subject_ID]["true_corrs"]
            p_array = r[subject_ID]["permuted_corrs"]
            true.append(t_array)
            permuted.append(p_array)
        all_results[subject_ID] = {
            "true_corrs": np.hstack(true),  # n_matches
            "permuted_corrs": np.hstack(permuted),  # n_maches by n_permutations
        }
    return all_results


def get_cross_maze_corr_summary(
    maze_pair=("maze_1", "maze_2"),
    single_units=True,
    min_split_half_corr=None,
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


# %% Misc
def plot_all_subject_matched_clusters(subject="m3", maze_pair=("maze_1", "maze_2")):
    """ """
    matched_clusters = mm.get_cross_maze_matches(
        subject,
        maze_pair,
        single_units=True,
        return_as="cluster_objects",
    )
    for i, pair in enumerate(matched_clusters):
        f, axes = plt.subplots(1, 2, figsize=(12, 6))
        for Clust, ax in zip(pair, axes):
            Clust.plot_tuning(feature="place_direction", ax=ax)
            ax.set_title(f"{Clust.cluster_unique_ID}")
        f.suptitle(f"Match {i}")
        plt.show()
