"""
Library for decoding subject location (place) as a function of distance to goal of event
aligned time. Uses util functions in ./decoding_utils.py
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

from GridMaze.analysis.core import get_sessions as gs
from . import decoding_utils as du

# %% Global Variables
from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

RESULTS_DIR = RESULTS_PATH / "place_decoding"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "maze_day2date.json", "r") as input_file:
    MAZE_DAY2DATE = json.load(input_file)

# %% Plot summary figures


def plot_place_decoding(results_df, distance_metric="geodesic", cue_window=(-8, 8), reward_window=(-8, 8), axes=None):
    """ """
    # set up fig
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(4, 2), sharey=True)
    for ax in axes:
        ax.axvline(0, color="k", ls="--", alpha=0.5)
        ax.axhline(0, color="k", ls="--", alpha=0.5)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_ylabel("Expected distance error")
    # plot cross subject expected distance error
    df = results_df[results_df.distance_metric == distance_metric]
    subject_means = df.groupby(["event", "timepoint", "permuted", "subject_ID"]).ede.mean()
    subject_grouped = subject_means.groupby(["event", "timepoint", "permuted"])
    mean = subject_grouped.mean().unstack()
    sem = subject_grouped.sem().unstack()
    timepoints = mean.index.get_level_values(1).unique().values
    for event, window, ax in zip(["cue", "reward"], [cue_window, reward_window], axes):
        for permuted, color in zip([True, False], ["k", "r"]):
            # plot mean and sem
            _mean = mean.loc[event, :][permuted]
            _sem = sem.loc[event, :][permuted]
            ax.plot(timepoints, _mean, color=color, lw=2)
            ax.fill_between(
                timepoints,
                _mean - _sem,
                _mean + _sem,
                color=color,
                alpha=0.2,
            )
        ax.set_xlim(window)
        ax.set_xlabel(f"{event} (s)")


def plot_session_place_decoding(
    results_df, simple_maze, dist_type="geodesic", axes=None, cue_window=(-4, 10), reward_window=(-10, 4), ymax=15
):
    """ """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(4, 2), sharey=True)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.axvline(0, color="k", ls="--", alpha=0.5)
    axes[0].set_ylabel("Expected distance error")
    # expected distance error (ede)
    true_results = results_df[results_df.permutation.isna()]
    permuted_results = results_df[~results_df.permutation.isna()]
    # plot
    for ax, event, window in zip(
        axes,
        ["cue", "reward"],
        [cue_window, reward_window],
    ):
        # plot true
        true_ede = du.get_expected_distance_error_df(
            true_results.copy(),
            simple_maze,
            decoding_type="place",
            alignment=f"{event}_aligned_time",
            permuted=False,
            return_total_av=True,
        )[dist_type]
        ax.plot(
            true_ede.index.values,
            true_ede.values,
            color="k",
            lw=2,
        )
        # plot chance
        perm_ede = du.get_expected_distance_error_df(
            permuted_results.copy(),
            simple_maze,
            decoding_type="place",
            alignment=f"{event}_aligned_time",
            permuted=True,
            return_total_av=False,
        )[dist_type]
        perm_av_ede = perm_ede.groupby("permutation").mean()
        p_mean = perm_av_ede.mean().values
        p_sem = perm_av_ede.sem().values

        ax.fill_between(
            true_ede.index.values,
            p_mean - p_sem,
            p_mean + p_sem,
            color="k",
            alpha=0.2,
        )
        ax.set_xlim(window)
        ax.set_xlabel(f"{event} (s)")
        ax.set_ylim(0, ymax)


# %% Decoding


def run_session_place_decoding(
    session,
    output_type,
    n_true=10,
    n_permuted=10,
    training_trial_phases="navigation",
    verbose=True,
):
    """ """
    # check if session is a MazeSession object
    if not isinstance(session, gs.MazeSession):
        if verbose:
            print(f"Getting session object for {session}")
        subject_ID, maze_name, day_on_maze = session
        session = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names=[maze_name],
            days_on_maze=[day_on_maze],
            with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "trials_df"],
            must_have_data=True,
        )
    # update inputs
    if training_trial_phases == "navigation":
        _training_trial_phases = ["navigation"]
    elif training_trial_phases == "all":
        _training_trial_phases = ["navigation", "reward_consumption", "ITI"]
    else:
        raise ValueError("training_trial_phases must be 'navigation' or 'all'")
    # check if save path exists
    save_path = RESULTS_DIR / output_type / training_trial_phases / f"{session.name}.parquet"
    if save_path.exists():
        results_df = pd.read_parquet(save_path, engine="pyarrow", use_threads=True)
        return results_df
    # get expected distance error (EDE) for true and permuted data
    true_metrics_df = get_place_decoding(
        session,
        output_type=output_type,
        n_repeats=n_true,
        training_trial_phases=_training_trial_phases,
        permuted=False,
        verbose=verbose,
    )
    permuted_metrics_df = get_place_decoding(
        session,
        output_type=output_type,
        n_repeats=n_permuted,
        training_trial_phases=_training_trial_phases,
        permuted=True,
        verbose=verbose,
    )
    # combine into one df
    results_df = pd.concat([true_metrics_df, permuted_metrics_df], axis=0)
    results_df.reset_index(drop=True, inplace=True)
    # save results
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(save_path, index=False)
    return results_df


def get_place_decoding(
    session,
    output_type="place_direction",
    resolution=0.5,
    include_multi_units=True,
    window=(-10, 10),
    goal_stratified_validation=True,
    n_test_trials=None,
    training_trial_phases=["navigation"],
    inv_alpha="auto",
    permuted=False,
    n_repeats=10,
    verbose=True,
):
    """ """
    # load input data
    simple_maze = session.simple_maze()
    results_dfs = []
    for n in range(n_repeats):
        input_data = du.get_place_decoding_input_data(
            session, resolution, include_multi_units, window, permuted=permuted
        )
        if output_type == "place_direction":
            input_data[("place_direction", "")] = input_data.apply(
                lambda x: f"{x[("maze_position", "simple")]}_{x[("cardinal_movement_direction", "")]}", axis=1
            )
        folds_df = du.get_folds_df(
            session, goal_stratified_validation, return_unique_IDs=True, n_test_trials=n_test_trials
        )
        # find optimal regularisation
        if inv_alpha == "auto":
            _inv_alpha = du.get_opt_reg(
                input_data,
                folds_df["fold_0"],
                simple_maze,
                input_type="spikes",
                output_type=output_type,
                training_trial_phases=training_trial_phases,
                eval_metric="expected_distance_error",
            )

        results_dfs = []
        # get cross validated decoding across folds
        folds = folds_df.columns.levels[0].unique()
        results_dfs = Parallel(n_jobs=len(folds), verbose=False)(
            delayed(_decode_place_fold)(
                n,
                fold,
                input_data,
                folds_df,
                output_type,
                _inv_alpha,
                training_trial_phases,
                verbose,
            )
            for fold in folds
        )
        df = pl.concat(results_dfs, how="vertical")
        results_dfs.append(df)
    decoding_df = pl.concat(results_dfs, how="vertical")
    metrics_df = du.get_decoding_metrics_df(decoding_df, simple_maze, output_type=output_type)
    metrics_df["permuted"] = permuted
    return metrics_df


# %%
def _decode_place_fold(
    repeat,
    fold,
    input_data,
    folds_df,
    output_type,
    inv_alpha,
    training_trial_phases,
    verbose,
):
    """
    Decode one fold of place (or place_direction) for get_place_decoding.
    Returns a single Polars DataFrame.
    """
    if verbose:
        print(f"Decoding {fold}")
    fold_df = folds_df[fold]
    train_df, test_df = du._get_test_train_dfs(input_data, fold_df, training_trial_phases)
    X_train, X_test, y_train, y_test = du._get_test_train_arrays(
        train_df,
        test_df,
        input_type="spikes",
        output_type=output_type,
        whiten_features=True,
    )
    if inv_alpha is None:
        decoder = LogisticRegression(
            penalty=None, max_iter=10_000, random_state=0, class_weight="balanced", verbose=False
        )
    else:
        decoder = LogisticRegression(
            penalty="l2", C=inv_alpha, max_iter=10_000, random_state=0, class_weight="balanced", verbose=False
        )
    decoder.fit(X_train, y_train)
    Yprobs = decoder.predict_proba(X_test)
    features = list(decoder.classes_)
    df = du.get_decoding_results_df(test_df, y_test, Yprobs, features, output_type, engine="polars")
    df = df.with_columns([pl.lit(fold).alias("fold"), pl.lit(repeat).alias("repeat")])
    return df
