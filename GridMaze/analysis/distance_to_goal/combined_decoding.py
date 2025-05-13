"""
New Lib for combining decoding analyses to see if goal decoding at cue improve when using decoders that know
about distance to goal while controlling for place coding in the neuronal population.
@peterdoohan
"""

# %% Imports

import json
from tkinter import font
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
from GridMaze.analysis.distance_to_goal import decoding_utils as du
from GridMaze.analysis.distance_to_goal import goal_decoding as gd
from GridMaze.analysis.distance_to_goal import bases as db


# %% Global Variables

from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

RESULTS_DIR = RESULTS_PATH / "distance_to_goal" / "goal_decoding_comparisons"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)


# %%


def plot_decoding_comparisons(summary_df, metric="test_acc", chance=1 / 12, cmap="Set1", axes=None):
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
    conditions = [
        "spikes_by_distance",
        "place_direction_prob_by_distance",
    ]  # ["spikes", "spikes_by_distance", "place_direction_prob", "place_direction_prob_by_distance"]
    cmap = plt.get_cmap(cmap, len(conditions))
    for event, ax in zip(["cue", "reward"], axes):
        df = summary_df[summary_df.event == event]
        df = df.set_index(["subject_ID", "aligned_time"])[conditions]
        subject_grouped_df = df.groupby("aligned_time")
        mean_df = subject_grouped_df.mean()
        sem_df = subject_grouped_df.sem()
        for i, condition in enumerate(conditions):
            color = cmap(i)
            mean = mean_df[condition]
            sem = sem_df[condition]
            ax.plot(mean.index, mean.values, label=condition, color=color)
            ax.fill_between(mean.index, mean - sem, mean + sem, alpha=0.2, color=color)
        ax.set_xlabel(f"{event} (s)")
    axes[0].legend(fontsize=8, loc="center left")

    # run stats
    for ax, event in zip(axes, ["cue", "reward"]):
        spikes_by_distance_df = pd.DataFrame()
        residuals_df = pd.DataFrame()
        for subject in SUBJECT_IDS:
            df = summary_df[(summary_df.event == event) & (summary_df.subject_ID == subject)]
            df = df.set_index(["aligned_time"])
            spikes_by_distance_df[subject] = df["spikes_by_distance"]
            residuals_df[subject] = df["spikes_by_distance"] - df["place_direction_prob_by_distance"]
        _plot_p_values(ax, spikes_by_distance_df.T, height=0.46, color="k", chance=chance)
        _plot_p_values(ax, residuals_df.T, height=0.47, color="red", chance=0)


def _plot_p_values(ax, df, height, color, chance=0):
    """"""
    p_values = []
    x = df.columns
    for i in x:
        t_stat, p_val = ttest_1samp(df[i], popmean=chance, alternative="greater")
        p_values.append(p_val)
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
    # indicate significant timepoints with line
    sig_idx = np.where(reject)[0]
    runs = np.split(sig_idx, np.where(np.diff(sig_idx) != 1)[0] + 1)
    for run in runs:
        if run.size > 0:
            x_run = x[run]
            y_run = np.full_like(x_run, height - 0.04, dtype=float)
            ax.plot(x_run, y_run, color=color, linewidth=2)


def get_decoding_comparisons_summary_df(metric="test_acc", cue_window=(-5, 10), reward_window=(-10, 5)):
    """ """
    event2valid_trial_phases = {
        "cue": ["ITI", "navigation"],
        "reward": ["navigation", "reward_consumption"],
    }
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
        cue_aligned_perf, reward_aligned_perf = [], []
        for s in sessions:
            try:
                decoding_df = run_goal_decoding_comparison(s, verbose=False, load_only=True)
            except FileNotFoundError as e:
                print(e)
                continue
            for event, window, perf_df in zip(
                ["cue", "reward"], [cue_window, reward_window], [cue_aligned_perf, reward_aligned_perf]
            ):
                _df = decoding_df[
                    (decoding_df[f"{event}_aligned_time"].between(*window))
                    & (decoding_df.trial_phase.isin(event2valid_trial_phases[event]))
                ]
                perf_df.append(
                    _df.groupby(["condition", "trial_unique_ID", f"{event}_aligned_time"])[metric].mean().unstack()
                )  # conditions_by_trials x timepoints (average over repeats)
        for _df, event in zip([cue_aligned_perf, reward_aligned_perf], ["cue", "reward"]):
            df = pd.concat(_df, axis=0)  # next average trials over conditions
            df = df.groupby("condition").mean().T.reset_index()
            df = df.rename(columns={f"{event}_aligned_time": "aligned_time"})
            df["event"] = event
            df["subject_ID"] = subject_ID
            all_dfs.append(df)
    summary_df = pd.concat(all_dfs, axis=0).reset_index(drop=True)
    return summary_df


# %% Functions


def run_goal_decoding_comparison(
    session,
    resolution=0.5,
    include_multi_units=True,
    window=(-10, 10),
    goal_stratified_validation=True,
    n_test_trials=None,
    n_bases=8,
    basis_type="gamma",
    training_trial_phases=["navigation"],
    max_steps_to_goal=30,
    inv_alpha="auto",
    n_repeats=10,
    verbose=True,
    load_only=False,
):
    """
    CONDITION 1: spikes --(predict)--> goal
    CONDITION 2: spikes_by_distance --(predict)--> goal
    CONDITION 3: spikes_by_distance --(predict)--> place_direction --(predict)--> goal (control)
    CONDITION 4: spikes_by_distance --(predict)--> place_direction_by_distance --(predict)--> goal (control)


    Note deocders are trained on all data defined in training_trial_phases
    not separate decoders for each timepoint aligned to trial events
    """
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
    # define conditions
    conditions = input_types = [
        "spikes",
        "spikes_by_distance",
        "place_direction_prob",
        "place_direction_prob_by_distance",
    ]
    # check if results already exist
    session_name = session.name
    save_path = RESULTS_DIR / f"{session_name}.parquet"
    if save_path.exists():
        if verbose:
            print(f"Loading results for {session_name} from disk")
        return pd.read_parquet(save_path)
    else:
        if load_only:
            raise FileNotFoundError(f"Results for {session_name} not found on disk")
    # else run analysis

    # get distance to goal basis functions (for spikes_by_distance condition)
    basis_fn = db.distance_basis_generator(n_bases=n_bases, basis=basis_type, max_steps=max_steps_to_goal, plot=False)
    simple_maze = session.simple_maze()
    C_dfs = [[] for _ in conditions]  # store condition results here
    for n in range(n_repeats):
        # get downsampled input data containing behavioural info and spike data
        input_data = du.get_place_decoding_input_data(session, resolution, include_multi_units, window, permuted=False)
        # organise trials into test-train folds
        folds_df = folds.get_folds_df(
            session, goal_stratified_validation, return_unique_IDs=True, n_test_trials=n_test_trials
        )
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
            inv_alpha=inv_alpha,
            training_trial_phases=training_trial_phases,
            verbose=verbose,
        )
        input_data = pd.concat([input_data, spatial_probs_df], axis=1)
        if inv_alpha == "auto":
            # get optimal regularisation for each condition
            inv_alphas = []
            for input_type in input_types:
                if verbose:
                    print(input_type)
                inv_alphas.append(
                    du.get_opt_reg(
                        input_data,
                        folds_df["fold_0"],
                        simple_maze,
                        basis_fn,
                        input_type=input_type,
                        output_type="goal",
                        training_trial_phases=training_trial_phases,
                        eval_metric="expected_distance_error",
                    )
                )
        else:
            inv_alphas = [inv_alpha] * len(input_types)
        # run xvaled decoding for each condition aross folds
        folds = folds_df.columns.levels[0].unique()
        if verbose:
            print("Running condition decodings paralleised across folds")
        parallel_outputs = Parallel(n_jobs=len(folds), verbose=False)(
            delayed(_decode_fold_repeat)(
                fold,
                n,
                input_data,
                folds_df,
                input_types,
                basis_fn,
                inv_alphas,
                training_trial_phases,
                verbose,
            )
            for fold in folds
        )
        # parallel_outputs is a list of lists (fold × conditions), assign to C_dfs
        for cond_idx in range(len(input_types)):
            for fold_output in parallel_outputs:
                C_dfs[cond_idx].extend(fold_output[cond_idx])
        del parallel_outputs  # save memory
    # combine folds and repeats
    results_dfs = [pl.concat(_dfs, how="vertical") for _dfs in C_dfs]
    # calculate test_acc and expecte distance error over every xvaled sample-repeat
    decoding_metric_dfs = []
    for df, condition in zip(results_dfs, conditions):
        metrics_df = du.get_decoding_metrics_df(df, simple_maze, output_type="goal")  # output pandas
        metrics_df["condition"] = condition
        decoding_metric_dfs.append(metrics_df)
    # combine decoding metrics across conditions
    output_df = pd.concat(decoding_metric_dfs, axis=0)
    # save results to disk
    save_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(save_path, index=False)
    if verbose:
        print(f"Saved results to {save_path}")
    return output_df


def _decode_fold_repeat(
    fold, repeat, input_data, folds_df, input_types, basis_fn, inv_alphas, training_trial_phases, verbose
):
    """
    Run decoding for a single fold & repeat. Returns a list of pl.DataFrames,
    one per condition.
    """
    fold_df = folds_df[fold]
    C_dfs = [[] for _ in input_types]
    for cond_idx, (input_type, inv_alpha) in enumerate(zip(input_types, inv_alphas)):
        if verbose:
            print(f"{fold}-{repeat}:{input_type}")
        train_df, test_df = folds._get_test_train_dfs(input_data, fold_df, training_trial_phases=training_trial_phases)
        X_train, X_test, y_train, y_test = du._get_test_train_arrays(
            train_df, test_df, input_type=input_type, output_type="goal", whiten_features=True, basis_fn=basis_fn
        )
        if inv_alpha is None:
            decoder = LogisticRegression(penalty=None, max_iter=10_000, random_state=0)
        else:
            decoder = LogisticRegression(penalty="l2", C=inv_alpha, max_iter=10_000, random_state=0)
        decoder.fit(X_train, y_train)
        Yprobs = decoder.predict_proba(X_test)
        C_df = du.get_decoding_results_df(
            test_df, y_test, Yprobs, list(decoder.classes_), output_type="goal", engine="polars"
        )
        assert isinstance(C_df, pl.DataFrame)
        C_df = C_df.with_columns([pl.lit(repeat).alias("repeat"), pl.lit(fold).alias("fold")])
        C_dfs[cond_idx].append(C_df)
    return C_dfs


def quick_plot(df, axes=None, metric="test_acc", cue_window=(-5, 10), reward_window=(-10, 5)):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    conditions = df.condition.unique()
    for condition in conditions:
        _df = df[df.condition == condition]
        for event, ax, window in zip(["cue", "reward"], axes, [cue_window, reward_window]):
            _df = _df[_df[f"{event}_aligned_time"].between(*window)]
            trial_df = _df.groupby(["trial_unique_ID", f"{event}_aligned_time"])[metric].mean().unstack()
            mean = trial_df.mean()
            ax.plot(mean.index, mean.values, label=condition)
    axes[0].legend()


# %%
def get_predicted_spatial(
    input_data,
    folds_df,
    simple_maze,
    basis_fn=None,
    input_type="spikes",
    output_type="place_direction",
    inv_alpha="auto",
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

    # get x-val optimal regularisation
    if inv_alpha == "auto":
        if verbose:
            print("Auto-optimising regularisation")
        inv_alpha = du.get_opt_reg(
            input_data,
            folds_df["fold_0"],
            simple_maze,
            basis_fn=basis_fn,
            input_type=input_type,
            output_type=output_type,
            training_trial_phases=training_trial_phases,
            eval_metric="expected_distance_error",
        )
    # get x-valed place-direction prob from spikes on each input_data sample
    dfs = []
    for fold in folds_df.columns.levels[0].unique():
        if verbose:
            print(fold)
        train_df, test_df = folds._get_test_train_dfs(input_data, folds_df[fold], training_trial_phases)
        X_train, X_test, y_train, y_test = du._get_test_train_arrays(
            train_df, test_df, input_type=input_type, output_type=output_type, whiten_features=True, basis_fn=basis_fn
        )
        if inv_alpha is None:
            decoder = LogisticRegression(penalty=None, max_iter=10_000, random_state=0, class_weight="balanced")
        else:
            decoder = LogisticRegression(
                penalty="l2", C=inv_alpha, max_iter=10_000, random_state=0, class_weight="balanced"
            )
        decoder.fit(X_train, y_train)
        Yprobs = decoder.predict_proba(X_test)
        features = list(decoder.classes_)
        probs_df = pd.DataFrame(
            index=test_df.index,
            columns=pd.MultiIndex.from_product([[f"{output_type}_prob"], features]),
            data=Yprobs,
        )
        # check for missing place_directions and add columns with value 0
        missing_features = set(all_features) - set(features)
        if len(missing_features) > 0:
            for missing_direction in missing_features:
                probs_df[(f"{output_type}_prob", missing_direction)] = 0
        dfs.append(probs_df.sort_index(axis=1))
    # combine folds and ensure index lines up with input_data
    probs_df = pd.concat(dfs, axis=0)
    probs_df.sort_index(axis=0, inplace=True)
    assert probs_df.index.equals(input_data.index)
    return probs_df


# %%
