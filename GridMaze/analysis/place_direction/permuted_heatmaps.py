"""
Lib for generating permuted place-direction heatmaps for control analyses
"""

# %% Imports
import json
from cv2 import line, mean
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.model_selection import ShuffleSplit
from sympy import comp

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import permute
from GridMaze.analysis.place_direction import dimensionality_reduction as pdr

from GridMaze.analysis.place_direction import efficient_coding as ec

# %% Global Variables
from GridMaze.paths import RESULTS2_PATH, EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

RESULTS_DIR = RESULTS2_PATH / "place_direction" / "permuted_heatmaps"

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]


# %% control analysis for low dimensional structure in place-direction tuning


def _stats(auc_df):
    """ """
    auc = auc_df.groupby("resample").mean().drop(columns=["split"])
    p_val = (auc["true_auc"] < (auc["permuted_auc"])).mean()
    print(f"true vs permuted: \n     p_value: {p_val:.3f}")


def plot_permuted_vs_true_summary(auc_df, ve_df, print_stats=True, axes=None):
    # set up figure
    if axes is None:
        f = plt.figure(figsize=(2, 2))
        gs = f.add_gridspec(2, 2, width_ratios=(1, 0.2), height_ratios=(1, 1), hspace=0.3)
        axes = [f.add_subplot(gs[:, 0]), f.add_subplot(gs[1, 1])]
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].plot([0, ve_df.component.max()], [0, 1], color="k", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Prop. \n variance explained")
    axes[0].set_xlabel("n components")
    axes[1].set_ylabel("AUC")

    # process cum ve curve data (top axis)
    conditions = ["true", "permuted"]
    colors = ["red", "gray"]
    components = ve_df.component.unique()
    ve = ve_df.groupby(["resample", "component"]).mean().drop(columns=["split"])
    for cond, color in zip(conditions, colors):
        cond_ve = ve[cond].unstack()
        mean = cond_ve.mean()
        std = cond_ve.std()
        axes[0].plot(components, mean, color=color, linewidth=1, label=cond)
        axes[0].fill_between(
            components,
            mean - std,
            mean + std,
            color=color,
            alpha=0.2,
        )
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)

    # process AUC data (bottom axis)
    auc = auc_df.groupby("resample").mean().drop(columns=["split"])
    mean_auc = auc.mean()
    auc_lower = auc.quantile(0.025)
    auc_upper = auc.quantile(0.975)
    err_lower = mean_auc - auc_lower
    err_upper = auc_upper - mean_auc
    for i, (cond, color) in enumerate(zip(conditions, colors)):
        cond = cond + "_auc"
        axes[1].errorbar(
            x=i,
            y=mean_auc[cond],
            yerr=[[err_lower[cond]], [err_upper[cond]]],
            fmt="o",
            color=color,
            label=cond,
            capsize=0,
            elinewidth=1.5,
            markersize=4,
        )
    axes[1].set_xticks(range(len(conditions)), conditions, rotation=45, ha="right")
    axes[1].set_ylim(0.5, 0.85)
    axes[1].set_xlim(-0.5, len(conditions) - 0.5)

    if print_stats:
        _stats(auc_df)
    return


def get_true_vs_permuted_neural_variance_explained(
    maze_name,
    n_splits=5,
    test_size=0.1,
    n_resamples=500,
    n_permutations=100,
    min_split_corr=0.3,
    late_sessions=False,
    demean=False,
    norm_length=True,
    max_jobs=5,
    verbose=False,
    save=False,
):
    """ """
    auc_save_path = RESULTS_DIR / "variance_explained" / f"{maze_name}_auc_results.parquet"
    ve_save_path = RESULTS_DIR / "variance_explained" / f"{maze_name}_ve_results.parquet"
    if not save and auc_save_path.exists() and ve_save_path.exists():
        if verbose:
            print("loading results from disk...")
        auc_df = pd.read_parquet(auc_save_path)
        ve_df = pd.read_parquet(ve_save_path)
        return auc_df, ve_df
    if verbose:
        print("Loading input data ...")
    input_data = get_input_data(
        maze_name=maze_name,
        n_splits=n_splits,  # no need for splits here
        test_size=test_size,
        n_permutations=n_permutations,
        late=late_sessions,
        min_split_corr=min_split_corr,
        verbose=verbose,
    )
    result_dfs = Parallel(n_jobs=max_jobs)(
        delayed(_process_reamples)(input_data, n_splits, n_permutations, demean, norm_length, i, verbose)
        for i in range(n_resamples)
    )
    auc_results_df = pd.concat([df[0] for df in result_dfs], axis=0)
    ve_results_df = pd.concat([df[1] for df in result_dfs], axis=0)
    if save:
        if verbose:
            print("saving results to disk...")
        auc_save_path.parent.mkdir(parents=True, exist_ok=True)
        ve_save_path.parent.mkdir(parents=True, exist_ok=True)
        auc_results_df.to_parquet(auc_save_path)
        ve_results_df.to_parquet(ve_save_path)
    return auc_results_df, ve_results_df


def _process_reamples(input_data, n_splits, n_permutations, demean, norm_length, n, verbose):
    """ """
    if verbose:
        print(f"Resample {n} ...")
    split_auc_results = []
    split_ve_results = []
    sampled_subjects = np.random.choice(SUBJECT_IDS, size=len(SUBJECT_IDS), replace=True)
    for i in range(n_splits):
        true_train = pd.concat([input_data[subject][i]["true"]["train"] for subject in sampled_subjects], axis=0).values
        true_test = pd.concat([input_data[subject][i]["true"]["test"] for subject in sampled_subjects], axis=0).values
        perm_train_df = pd.concat(
            [input_data[subject][i]["permuted"]["train"] for subject in sampled_subjects], axis=0
        ).droplevel(1)
        perm_test_df = pd.concat(
            [input_data[subject][i]["permuted"]["test"] for subject in sampled_subjects], axis=0
        ).droplevel(1)
        if demean:
            true_train, true_test = ec._demean(true_train), ec._demean(true_test)
        if norm_length:
            true_train, true_test = ec._norm_length(true_train), ec._norm_length(true_test)
        # get variance explained for true and permuted data
        true_cum_ve = ec.get_pca_variance_explained(true_train, true_test)
        if len(true_cum_ve) < true_train.shape[1] + 1:
            true_cum_ve = np.concatenate((true_cum_ve, np.ones(true_train.shape[1] + 1 - len(true_cum_ve))))
        true_auc = np.trapz(true_cum_ve, dx=1 / len(true_cum_ve))
        perm_train_df, perm_test_df = perm_train_df.swaplevel(), perm_test_df.swaplevel()
        perm_aucs, perm_cum_ves = [], []
        for j in range(n_permutations):
            _perm_train = perm_train_df.loc[j].values
            _perm_test = perm_test_df.loc[j].values
            if demean:
                _perm_train, _perm_test = ec._demean(_perm_train), ec._demean(_perm_test)
            if norm_length:
                _perm_train, _perm_test = ec._norm_length(_perm_train), ec._norm_length(_perm_test)
            permuted_cum_ve = ec.get_pca_variance_explained(_perm_train, _perm_test)
            if len(permuted_cum_ve) < _perm_train.shape[1] + 1:
                permuted_cum_ve = np.concatenate(
                    (permuted_cum_ve, np.ones(_perm_train.shape[1] + 1 - len(permuted_cum_ve)))
                )
            perm_cum_ves.append(permuted_cum_ve)
            perm_auc = np.trapz(permuted_cum_ve, dx=1 / len(permuted_cum_ve))
            perm_aucs.append(perm_auc)
        # store ve curve for true and permuted data
        mean_perm_cum_ve = np.mean(perm_cum_ves, axis=0)
        ve_df = pd.DataFrame(index=range(true_train.shape[1] + 1))
        ve_df["split"] = i
        ve_df["resample"] = n
        ve_df["component"] = np.arange(true_train.shape[1] + 1)
        ve_df["true"] = true_cum_ve
        ve_df["permuted"] = mean_perm_cum_ve
        split_ve_results.append(ve_df)
        # store auc summary
        mean_perm_auc = np.mean(perm_aucs)
        split_auc_results.append(
            {
                "split": i,
                "resample": n,
                "true_auc": true_auc,
                "permuted_auc": mean_perm_auc,
            }
        )
    return [pd.DataFrame(split_auc_results), pd.concat(split_ve_results, axis=0)]


def get_input_data(
    maze_name,
    n_splits=5,
    test_size=0.1,
    n_permutations=100,
    late=False,
    max_steps_to_goal=30,
    min_split_corr=0.3,
    verbose=False,
):
    """ """
    permuted_heatmaps = load_permuted_place_direction_heatmaps(
        maze_name, normalisation=False, n_permutations=n_permutations
    )
    days_on_maze = "late" if late == True else "all"
    all_data = {}
    subject2session_names = {}
    for subject in SUBJECT_IDS:
        if verbose:
            print(subject)
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject],
            maze_names=[maze_name],
            days_on_maze=days_on_maze,
            with_data=[
                "navigation_df",
                "navigation_spike_rates_df",
                "trajectory_decisions_df",
                "cluster_metrics",
                "cluster_place_direction_tuning_metrics",
            ],
        )
        session_names = []
        subject_data = {}
        for session in sessions:
            session_name = session.name
            session_names.append(session_name)
            subject_data[session_name] = pdr.get_session_place_direction_tuning(
                session,
                fill_nans="mean",
                normalisation=False,
                min_split_corr=min_split_corr,
                max_steps_from_goal=max_steps_to_goal,
            )
        subject2session_names[subject] = session_names
        all_data[subject] = subject_data
    # remove session_names for the pool that have no data
    for subject in SUBJECT_IDS:
        session_names = subject2session_names[subject]
        session_names = [sn for sn in session_names if all_data[subject][sn] is not None]
        subject2session_names[subject] = session_names
    # organise data into CV splits
    subject2split_data = {}
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
    for subject in SUBJECT_IDS:
        _session_names = np.array(subject2session_names[subject])
        # Generate the splits (session names)
        split2data = {}
        for i, (train_index, test_index) in enumerate(ss.split(_session_names)):
            train, test = _session_names[train_index], _session_names[test_index]
            split_data = {}
            train_data = [all_data[subject][session] for session in train]
            train_df = pd.concat([df for df in train_data if df is not None], axis=0)
            test_data = [all_data[subject][session] for session in test]
            test_df = pd.concat([df for df in test_data if df is not None], axis=0)
            split_data["true"] = {
                "train": train_df,
                "test": test_df,
            }
            split_data["permuted"] = {
                "train": permuted_heatmaps.loc[train_df.index, :, :],
                "test": permuted_heatmaps.loc[test_df.index, :, :],
            }
            split2data[i] = split_data
        subject2split_data[subject] = split2data
    return subject2split_data


# %% load data from disk


def load_permuted_place_direction_heatmaps(
    maze_name,
    subject_IDs="all",
    normalisation="length",
    n_permutations="all",
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
    if n_permutations != "all":
        assert isinstance(n_permutations, int)
        df = df[df.index.get_level_values(2) < n_permutations]

    return df


# %% Generate permuted heatmap functions


def populate_permuted_place_direction_heatmaps(n_permutation=100, max_jopbs=5, verbose=True, overwrite=False):
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
    late_sessions=False,
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
