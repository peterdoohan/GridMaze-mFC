""" """

# %% Imports

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from GridMaze.analysis.core import permute

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.goal_coding import decoding_utils as du

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "simple_decoding"

TRIAL_ALIGNED_RATES_FS = 0.04

# %% pop level functions


def get_event_aligned_decoding_summary(
    resolution=0.5,
    window=(-10, 10),
    verbose=True,
    save=False,
):
    """ """
    # load cached results if already processed
    save_path = RESULTS_DIR / "event_aligned_decoding_summary.parquet"
    if save_path.exists() and not save:
        if verbose:
            print(f"Loading existing results from {save_path}")
        return pd.read_parquet(save_path)
    if verbose:
        print("Loading sessions ...")
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names="all",
        days_on_maze="late",
        with_data=[
            "navigation_df",
            "navigation_spike_counts_df",
            "cluster_metrics",
            "trials_df",
        ],
        must_have_data=True,
    )
    dfs = []
    for session in sessions:
        if verbose:
            print(session.name)
        for event in ["cue", "reward"]:
            results_df = get_event_aligned_goal_decoding(
                session,
                event=event,
                resolution=resolution,
                window=window,
            )
            dfs.append(results_df)
    summary_df = pd.concat(dfs, axis=0)
    summary_df.reset_index(drop=True, inplace=True)
    if save:
        if verbose:
            print(f"Saving results to {save_path}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_parquet(save_path)
    return summary_df


def get_trial_aligned_decoding_summary(
    resolution=0.25,
    verbose=True,
    save=False,
):
    # load cached results if already processed
    save_path = RESULTS_DIR / "trial_aligned_decoding_summary.parquet"
    if save_path.exists() and not save:
        if verbose:
            print(f"Loading existing results from {save_path}")
        return pd.read_parquet(save_path)
    if verbose:
        print("Loading sessions ...")
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names="all",
        days_on_maze="late",
        with_data=["trial_aligned_rates_df", "cluster_metrics", "trials_df"],
        must_have_data=True,
    )
    dfs = []
    for session in sessions:
        if verbose:
            print(session.name)
        results_df = get_trial_aligned_goal_decoding(session, resolution=resolution)
        dfs.append(results_df)
    summary_df = pd.concat(dfs, axis=0)
    summary_df.reset_index(drop=True, inplace=True)
    if save:
        if verbose:
            print(f"Saving results to {save_path}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_parquet(save_path)
    return summary_df


# %% session level functions


def get_event_aligned_goal_decoding(
    session,
    event="cue",
    resolution=0.5,
    window=(-10, 10),
    goal_stratified_validation=True,
    n_test_trials=None,
    include_multi_units=True,
    zscore_spikes=True,
):
    """ """
    input_data = du.get_event_aligned_input_data(session, event, resolution, window, include_multi_units)
    timepoints = sorted(input_data.event_aligned_time[event].unique())
    folds_df = du.get_folds_df(session, goal_stratified_validation, return_unique_IDs=True, n_test_trials=n_test_trials)
    results_dfs = []
    for fold in folds_df.columns.levels[0].unique():
        fold_df = folds_df[fold]
        test_trials = [t for t in fold_df.test.values.flatten() if isinstance(t, str)]
        train_trials = [t for t in fold_df.train.values.flatten() if isinstance(t, str)]
        train_df = input_data[input_data.trial_unique_ID.isin(train_trials)]
        test_df = input_data[input_data.trial_unique_ID.isin(test_trials)]
        decoder = LogisticRegression(penalty=None, max_iter=10000, random_state=0, class_weight="balanced")
        for t in timepoints:
            _train_df = train_df[train_df.event_aligned_time[event] == t]
            _test_df = test_df[test_df.event_aligned_time[event] == t]
            if _train_df.empty or _test_df.empty:
                continue  # rare cases when no trials for that timepoint (eg, end of session trial)
            X_train, y_train = _train_df.spike_count.values, _train_df.goal.values
            X_test, y_test = _test_df.spike_count.values, _test_df.goal.values
            if zscore_spikes:  # zscore features
                scaler = StandardScaler()  # mean=0, std=1 per column
                scaler.fit(X_train)  # learn stats on train
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
            # fit model
            decoder.fit(X_train, y_train)
            # out_df
            y_pred = decoder.predict(X_test)
            correct = (y_pred == y_test).astype(int)
            n_test_samp = len(y_test)
            df = pd.DataFrame(
                {
                    "timepoint": np.repeat(t, n_test_samp),
                    "trial_unique_ID": _test_df.trial_unique_ID.values,
                    "steps_to_goal": _test_df.steps_to_goal.bin_mid,
                    "trial_phase": _test_df.trial_phase.values,
                    "true_goal": y_test,
                    "predicted_goal": y_pred,
                    "accuracy": correct,
                }
            )
            df["fold"] = fold
            results_dfs.append(df)
    results_df = pd.concat(results_dfs, axis=0)
    results_df.reset_index(drop=True, inplace=True)
    # add session info
    results_df["subject_ID"] = session.subject_ID
    results_df["maze_name"] = session.maze_name
    results_df["day_on_maze"] = session.day_on_maze
    results_df["goal_subset"] = session.goal_subset
    return results_df


def get_trial_aligned_goal_decoding(
    session,
    resolution=0.25,
    goal_stratified_validation=True,
    n_test_trials=None,
    include_multi_units=True,
    zscore_spikes=True,
):
    """ """
    # load data
    trial_aligned_rates = session.trial_aligned_rates_df
    cluster_metrics = session.cluster_metrics
    session_info = session.session_info
    # filter clusters
    keep_clusters = gc.filter_clusters(
        cluster_metrics,
        session_info,
        return_unique_IDs=False,
        single_units=True,
        multi_units=include_multi_units,
    )
    trial_aligned_rates = trial_aligned_rates[trial_aligned_rates.cluster_ID.isin(keep_clusters)]
    # downsample resolution
    rates = trial_aligned_rates.firing_rate
    times = rates.columns.astype(float).values
    rates_T = rates.T
    rates_T.index = rates_T.index.astype(float)
    ds_rates = rates_T.groupby(rates_T.index // resolution).mean().T  # average over samples to desired resolution
    new_times = np.linspace(times[0], times[-1], ds_rates.shape[1])
    ds_rates.columns = pd.MultiIndex.from_product([["firing_rate"], new_times])
    input_data = pd.concat(
        [trial_aligned_rates.drop("firing_rate", level=0, axis=1), ds_rates],
        axis=1,
    )  # recombine
    timepoints = new_times
    folds_df = du.get_folds_df(
        session, goal_stratified_validation, return_unique_IDs=False, n_test_trials=n_test_trials
    )
    folds = np.unique(folds_df.columns.get_level_values(0))
    results_dfs = []
    for fold in folds:
        fold_df = folds_df[fold]
        test_trials = fold_df.test.dropna().values
        train_trials = fold_df.train.stack().dropna().values
        train_df = input_data[input_data.trial.isin(train_trials)]
        test_df = input_data[input_data.trial.isin(test_trials)]
        train_df.set_index(["cluster_ID", "trial"], inplace=True)
        test_df.set_index(["cluster_ID", "trial"], inplace=True)
        decoder = LogisticRegression(penalty=None, max_iter=10000, random_state=0, class_weight="balanced")
        y_train = train_df.goal.unstack(level=0)[0].values  # n train trials
        y_test = test_df.goal.unstack(level=0)[0].values  # n test trials
        for timepoint in timepoints:
            X_train = train_df.firing_rate[timepoint].unstack(level=0).values  # n_trials x n_clusters
            X_test = test_df.firing_rate[timepoint].unstack(level=0).values
            if zscore_spikes:  # zscore features
                scaler = StandardScaler()  # mean=0, std=1 per column
                scaler.fit(X_train)  # learn stats on train
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                # fit model
            decoder.fit(X_train, y_train)
            # out_df
            y_pred = decoder.predict(X_test)
            correct = (y_pred == y_test).astype(int)
            n_test_samp = len(y_test)
            df = pd.DataFrame(
                {
                    "timepoint": np.repeat(timepoint, n_test_samp),
                    "trial": test_df.firing_rate[new_times[0]].unstack(level=0).index.values,
                    "true_goal": y_test,
                    "predicted_goal": y_pred,
                    "accuracy": correct,
                }
            )
            df["fold"] = fold
            results_dfs.append(df)
    results_df = pd.concat(results_dfs, axis=0)
    results_df.reset_index(drop=True, inplace=True)
    # add session info
    results_df["subject_ID"] = session.subject_ID
    results_df["maze_name"] = session.maze_name
    results_df["day_on_maze"] = session.day_on_maze
    results_df["goal_subset"] = session.goal_subset
    return results_df


def get_place_control_aligned_goal_decoding():
    # use place or place-direction as X to predict goal as control

    return
