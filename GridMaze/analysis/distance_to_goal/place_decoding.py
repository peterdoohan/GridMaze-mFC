"""
Library for decoding subject location (place) as a function of distance to goal of event
aligned time. Uses util functions in ./decoding_utils.py
@peterdoohan
"""

# %% Imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler


from . import decoding_utils as du
from . import bases as db

# %% Global Variables

# %% Functions


def test(
    session,
    resolution=0.5,
    include_multi_units=True,
    window=(-10, 10),
    goal_stratified_validation=True,
    n_test_trials=None,
    training_trial_phases=["navigation", "reward_consumption", "ITI"],
):
    """ """
    input_data = du.get_place_decoding_input_data(session, resolution, include_multi_units, window)
    folds_df = du.get_folds_df(session, goal_stratified_validation, return_unique_IDs=True, n_test_trials=n_test_trials)
    results_dfs = []
    for fold in folds_df.columns.levels[0].unique():
        print(fold)
        fold_df = folds_df[fold]
        test_trials = [t for t in fold_df.test.values.flatten() if isinstance(t, str)]
        train_trials = [t for t in fold_df.train.values.flatten() if isinstance(t, str)]
        train_df = input_data[input_data.trial_unique_ID.isin(train_trials)]
        train_df = train_df[train_df.trial_phase.isin(training_trial_phases)]
        test_df = input_data[input_data.trial_unique_ID.isin(test_trials)]
        X_train, y_train = train_df.spike_count.values, train_df.maze_position.simple.values
        X_test, y_test = test_df.spike_count.values, test_df.maze_position.simple.values
        decoder = LogisticRegression(
            penalty=None, max_iter=10000, random_state=0, class_weight="balanced", verbose=True
        )
        decoder.fit(X_train, y_train)
        Pprobs = decoder.predict_proba(X_test)
        n_samples, n_places = Pprobs.shape
        places = list(decoder.classes_)
        df = pd.DataFrame(
            {
                "cue_aligned_time": np.repeat(test_df.event_aligned_bin["cue"].values, n_places),
                "reward_aligned_time": np.repeat(test_df.event_aligned_bin["reward"].values, n_places),
                "true_place": np.repeat(y_test, n_places),
                "trial_unique_ID": np.repeat(test_df.trial_unique_ID.values, n_places),
                "predicted_place": np.tile(places, n_samples),
                "predicted_place_prob": Pprobs.ravel(),
            }
        )
        df["fold"] = fold
        results_dfs.append(df)
    results_df = pd.concat(results_dfs, axis=0)
    results_df.reset_index(drop=True, inplace=True)
    return results_df


def get_event_aligned_place_deocoding(
    session,
    event="cue",
    resolution=0.5,
    window=(-10, 10),
    goal_stratified_validation=False,
    n_test_trials=6,
    include_multi_units=True,
    whiten_features=True,
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
            X_train, y_train = _train_df.spike_count.values, _train_df.maze_position.simple.values
            X_test, y_test = _test_df.spike_count.values, _test_df.maze_position.simple.values
            if whiten_features:  # zscore features
                scaler = StandardScaler()  # mean=0, std=1 per column
                scaler.fit(X_train)  # learn stats on train
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
            decoder.fit(X_train, y_train)
            # out_df
            Gprobs = decoder.predict_proba(X_test)
            n_samples, n_places = Gprobs.shape
            places = list(decoder.classes_)
            df = pd.DataFrame(
                {
                    "timepoint": np.repeat(t, n_samples * n_places),
                    "true_place": np.repeat(y_test, n_places),
                    "trial_unique_ID": np.repeat(_test_df.trial_unique_ID.values, n_places),
                    "predicted_place": np.tile(places, n_samples),
                    "predicted_place_prob": Gprobs.ravel(),
                }
            )
            df["fold"] = fold
            results_dfs.append(df)
    results_df = pd.concat(results_dfs, axis=0)
    results_df.reset_index(drop=True, inplace=True)
    return results_df
