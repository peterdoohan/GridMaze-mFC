"""
New Lib for combining decoding analyses to see if goal decoding at cue improve when using decoders that know
about distance to goal while controlling for place coding in the neuronal population.
@peterdoohan
"""

# %% Imports

import json
import numpy as np
import pandas as pd
import polars as pl
import networkx as nx
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

from GridMaze.maze import representations as mr
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import folds
from GridMaze.analysis.goal_coding import decoding_utils as du
from GridMaze.analysis.distance_to_goal import bases as db


# %% Global Variables


from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

RESULTS_DIR = RESULTS_PATH / "distance_to_goal" / "goal_decoding_comparisons"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)


# %%
def plot_main_residual(res_df, cue_window=(-5, 10), reward_window=(-10, 5), axes=None):
    """ """
    # set up figure
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    for ax in axes:
        ax.axvline(0, color="k", linestyle="--", alpha=0.5)
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_ylabel("delta test acc")
    for event, ax, window in zip(["cue", "reward"], axes, [cue_window, reward_window]):
        df = res_df[res_df.event == event]
        df = df.set_index(["subject_ID", "aligned_time"]).residuals
        subject_grouped_df = df.groupby("aligned_time")
        mean = subject_grouped_df.mean()
        sem = subject_grouped_df.sem()
        ax.plot(mean.index, mean.values)
        ax.fill_between(mean.index, mean - sem, mean + sem, alpha=0.2)
        ax.set_xlabel(f"{event} (s)")
        ax.set_xlim(window)
    return


def plot_decoding_comparisons(
    summary_df, metric="test_acc", chance=1 / 12, cue_window=(-5, 10), reward_window=(-10, 5), axes=None
):
    """ """
    # set up figure

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    for ax in axes:
        ax.axvline(0, color="k", linestyle="--", alpha=0.5)
        if metric == "test_acc":
            ax.axhline(chance, color="k", linestyle="--", alpha=0.5)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_ylabel(metric)

    # plot conditions
    conditions = ["spikes", "spikes_by_distance", "place_direction_prob", "place_direction_prob_by_distance"]
    for event, ax, window in zip(["cue", "reward"], axes, [cue_window, reward_window]):
        df = summary_df[summary_df.event == event]
        df = df.set_index(["subject_ID", "aligned_time"])[conditions]
        subject_grouped_df = df.groupby("aligned_time")
        mean_df = subject_grouped_df.mean()
        sem_df = subject_grouped_df.sem()
        for i, condition in enumerate(conditions):
            mean = mean_df[condition]
            sem = sem_df[condition]
            ax.plot(mean.index, mean.values, label=condition)
            ax.fill_between(mean.index, mean - sem, mean + sem, alpha=0.2)
        ax.set_xlabel(f"{event} (s)")
        ax.set_xlim(window)
    axes[0].legend(fontsize=8)


def get_decoding_summary_df(metric="test_acc", permuted=False, min_single_unit=0, as_residuals=False):
    """ """
    all_dfs = []
    for subject_ID in SUBJECT_IDS:
        print(subject_ID)
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names="all",
            days_on_maze="late",
            goal_subsets=["subset_1", "subset_2"],
            with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "trials_df"],
            must_have_data=True,
        )
        # load from disk
        decoding_dfs = []
        for s in sessions:
            # check single unit count to include
            if s.cluster_metrics.single_unit.sum() < min_single_unit:
                print(f"Skipping {s.name} due to low single unit count")
                continue
            try:
                decoding_df = run_goal_decoding_comparison(s, permuted=permuted, verbose=False, load_only=True)
                decoding_dfs.append(decoding_df)
            except FileNotFoundError as e:
                print(e)
                continue
        for event in ["cue", "reward"]:
            event_perf = []
            for decoding_df in decoding_dfs:
                _time = f"{event}_aligned_time"
                if event == "cue":
                    df = decoding_df[~decoding_df[_time].isna()]
                    df = df[  # only include ITI before cue and navigation time after cue
                        ((df[_time].le(0)) & (df.trial_phase == "ITI"))
                        | (df[_time].gt(0)) & (df.trial_phase == "navigation")
                    ]
                else:  # reward
                    df = decoding_df[~decoding_df[_time].isna()]
                    df = df[
                        ((df[_time].gt(0)) & (df.trial_phase == "reward_consumption"))
                        | (df[_time].le(0)) & (df.trial_phase == "navigation")
                    ]
                session_perf = (
                    df.groupby(["input_type", "trial_unique_ID", f"{event}_aligned_time"])[metric].mean().unstack()
                )
                event_perf.append(session_perf)
            subject_df = pd.concat(event_perf, axis=0)
            if as_residuals:
                residuals = subject_df.loc["spikes_by_distance"] - subject_df.loc["place_direction_prob_by_distance"]
                subject_results = residuals.mean().reset_index()
                subject_results.columns = ["aligned_time", "residuals"]
            else:
                subject_results = subject_df.groupby("input_type").mean().T.reset_index()
                subject_results = subject_results.rename(columns={f"{event}_aligned_time": "aligned_time"})
            subject_results["event"] = event
            subject_results["subject_ID"] = subject_ID
            all_dfs.append(subject_results)
    summary_df = pd.concat(all_dfs, axis=0).reset_index(drop=True)
    return summary_df


# %% Functions


def test_plots(subject_ID):
    sessions = gs.get_maze_sessions(
        subject_IDs=[subject_ID],
        maze_names="all",
        days_on_maze="all",
        goal_subsets=["subset_1", "subset_2"],
        with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "trials_df"],
        must_have_data=True,
    )
    for session in sessions:
        decoding_df = run_goal_decoding_comparison(session, permuted=False, verbose=False, load_only=True)
        fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
        quick_plot(decoding_df, axes=axes, metric="test_acc", chance=1 / 12)
        axes[0].set_title(f"{session.maze_name}-{session.day_on_maze}-{session.subject_ID}")

    return


def populate_goal_decoding_comparisons(subject_ID, permuted, n_repeats):
    """ """
    sessions = gs.get_maze_sessions(
        subject_IDs=[subject_ID],
        maze_names="all",
        days_on_maze="all",
        with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "trials_df"],
        must_have_data=True,
    )
    for s in sessions:
        print(s)
        try:
            run_goal_decoding_comparison(s, permuted=permuted, n_repeats=n_repeats)
        except FileNotFoundError as e:
            print(e)
            continue
    return


# %%
def run_goal_decoding_comparison(
    session,
    resolution=0.5,
    input_types=[
        "spikes",
        "spikes_by_distance",
        "place_direction_prob",
        "place_direction_prob_by_distance",
    ],
    include_multi_units=True,
    window=(-10, 10),
    goal_stratified_validation=True,
    n_bases=8,
    basis_type="gamma",
    training_trial_phases=["navigation"],
    max_steps_to_goal=30,
    permuted=False,
    n_repeats=1,
    verbose=True,
    load_only=False,
):
    """ """
    # get session object if strings input (when running jobs on HPC)
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
    # check if results already exist
    session_name = session.name
    _is_permuted = "permuted" if permuted else "true"
    save_path = RESULTS_DIR / _is_permuted / f"{session_name}.parquet"
    if save_path.exists():
        if verbose:
            print(f"Loading results for {session_name} from disk")
        return pd.read_parquet(save_path)
    else:
        if load_only:
            raise FileNotFoundError(f"Results for {session_name} not found on disk")

    # get downsampled input data containing behavioural info and spike data
    all_results = []
    for r in range(n_repeats):
        if verbose:
            print(f"Repeat: {r}")
        input_data = du.get_place_decoding_input_data(session, resolution, include_multi_units, window, permuted=False)
        if permuted:
            if verbose:
                print("Shuffling goals on each trial")
            trial_goal_df = input_data[[("trial", ""), ("goal", "")]].drop_duplicates()
            shuffled_trial2goal = pd.Series(
                index=trial_goal_df.trial, data=trial_goal_df.goal.sample(frac=1).values
            ).to_dict()
            input_data[("goal", "")] = input_data[("trial", "")].map(shuffled_trial2goal)
        # get distance to goal basis functions (for spikes_by_distance condition)
        basis_fn = db.distance_basis_generator(
            n_bases=n_bases, basis=basis_type, max_steps=max_steps_to_goal, plot=False
        )
        simple_maze = session.simple_maze()
        # organise trials into test-train folds
        folds_df = folds.get_folds_df(session, goal_stratified_validation, return_unique_IDs=True)
        # predict plce/place_direction probabilities from spike counts (for control conditions)
        if verbose:
            print(f"Predicting place_direction probabilities from spike counts")
        spatial_probs_df = get_predicted_spatial(
            input_data,
            folds_df,
            simple_maze,
            basis_fn=basis_fn,
            input_type="spikes_by_distance",
            output_type="place_direction",
            training_trial_phases=training_trial_phases,
            verbose=verbose,
        )
        input_data = pd.concat([input_data, spatial_probs_df], axis=1)
        # run xvaled decoding for each condition aross folds
        _folds = folds_df.columns.levels[0].unique()
        if verbose:
            print("Running condition decodings paralleised across folds")
        fold_results = Parallel(n_jobs=len(_folds), verbose=False)(
            delayed(_process_fold)(
                r,
                fold,
                input_data,
                folds_df,
                input_types,
                basis_fn,
                training_trial_phases,
                verbose,
            )
            for fold in _folds
        )
        all_results.extend(fold_results)
    decoding_results_df = pl.concat(all_results, how="vertical")
    decoding_metrics_df = du.get_decoding_metrics_df(
        decoding_results_df, simple_maze, output_type="goal", groupby=["sample_index", "repeat", "input_type"]
    )
    # save results to disk
    save_path.parent.mkdir(parents=True, exist_ok=True)
    decoding_metrics_df.to_parquet(save_path, index=False)
    if verbose:
        print(f"Saved results to {save_path}")
    return decoding_metrics_df


def _process_fold(repeat, fold, input_data, folds_df, input_types, basis_fn, training_trial_phases, verbose):
    """ """
    fold_results = []
    for itype in input_types:
        df = du.get_xvaled_decoding_df(
            input_data,
            folds_df,
            fold,
            training_trial_phases,
            itype,
            output_type="goal",
            basis_fn=basis_fn,
            df_engine="polars",
            verbose=verbose,
        )
        df = df.with_columns(
            [pl.lit(repeat).alias("repeat"), pl.lit(fold).alias("fold"), pl.lit(itype).alias("input_type")]
        )
        fold_results.append(df)
    fold_results = pl.concat(fold_results, how="vertical")
    return fold_results


def quick_plot(df, axes=None, metric="test_acc", chance=1 / 12, cue_window=(-5, 10), reward_window=(-10, 5)):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    for ax in axes:
        ax.axhline(chance, color="k", linestyle="--", alpha=0.5)
        ax.axvline(0, color="k", linestyle="--", alpha=0.5)

    input_types = df.input_type.unique()
    for itype in input_types:
        itype_df = df[df.input_type == itype]
        for event, ax, window in zip(["cue", "reward"], axes, [cue_window, reward_window]):
            _time = f"{event}_aligned_time"
            if event == "cue":
                _df = itype_df[~itype_df[_time].isna()]
                _df = _df[
                    ((_df[_time].between(window[0], 0)) & (_df.trial_phase == "ITI"))
                    | (_df[_time].between(0, window[1])) & (_df.trial_phase == "navigation")
                ]
            else:  # reward
                _df = itype_df[~itype_df[_time].isna()]
                _df = _df[
                    ((_df[_time].between(0, window[1])) & (_df.trial_phase == "reward_consumption"))
                    | (_df[_time].between(window[0], 0)) & (_df.trial_phase == "navigation")
                ]
            trial_df = _df.groupby(["trial_unique_ID", _time])[metric].mean().unstack()
            mean = trial_df.mean()
            ax.plot(mean.index, mean.values, label=itype)
    axes[0].legend(fontsize=8)


# %%
def get_predicted_spatial(
    input_data,
    folds_df,
    simple_maze,
    basis_fn=None,
    input_type="spikes",
    output_type="place_direction",
    training_trial_phases=["navigation"],
    verbose=True,
):
    """
    From some input_data, and folds_df dataframes, preform cross-validated prediction
    of place_direction from spike counts (w/ Logisitic Rergression classifier).
    W/o stratification by distance to goal.

    Outputs the neural representation of place direction in the data as
    a probability distribution over the place directions.

    W/ automatic regularisation optimisation
    """
    if output_type == "place_direction":
        # precompute all place_directions ("A1_N")
        all_features = mr.get_maze_place_direction_pairs(simple_maze)
        all_features = ["_".join(x) for x in all_features]

        # add place_direction column to input_data
        input_data[("place_direction", "")] = input_data.apply(
            lambda x: f"{x[("maze_position", "simple")]}_{x[("cardinal_movement_direction", "")]}", axis=1
        )
    elif output_type == "place":
        all_features = mr.get_maze_locations(simple_maze)
    else:
        raise ValueError(f"Unknown output type {output_type!r}")
    # get x-valed place-direction prob from spikes on each input_data sample
    _folds = folds_df.columns.levels[0].unique()
    dfs = Parallel(n_jobs=len(_folds), verbose=False)(
        delayed(_process_predict_spatial_fold)(
            fold,
            input_data,
            folds_df,
            basis_fn,
            input_type,
            output_type,
            training_trial_phases,
            all_features,
            verbose,
        )
        for fold in _folds
    )
    # combine folds and ensure index lines up with input_data
    probs_df = pd.concat(dfs, axis=0)
    probs_df.sort_index(axis=0, inplace=True)
    assert probs_df.index.equals(input_data.index)
    return probs_df


def _process_predict_spatial_fold(
    fold,
    input_data,
    folds_df,
    basis_fn,
    input_type,
    output_type,
    training_trial_phases,
    all_features,
    verbose,
):
    """ """
    if verbose:
        print(fold)
    probs_df = du.get_xvaled_decoding_df(
        input_data,
        folds_df,
        fold,
        training_trial_phases,
        input_type,
        output_type=output_type,
        basis_fn=basis_fn,
        df_engine="polars",
        verbose=verbose,
        return_as="probs_df",
    )
    features = probs_df.columns.levels[1].unique()
    # check for missing place_directions and add columns with value 0
    missing_features = set(all_features) - set(features)
    if len(missing_features) > 0:
        for missing_direction in missing_features:
            probs_df[(f"{output_type}_prob", missing_direction)] = 0
    return probs_df.sort_index(axis=1)


# %%
