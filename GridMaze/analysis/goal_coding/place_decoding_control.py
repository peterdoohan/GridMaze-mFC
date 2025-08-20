"""
Is goal decoding just place decoding?
Control analysis: neurons -> decoded dist over places --> decode goal (all cv)
@peterdoohan
"""

# %% Imports
import pandas as pd
import polars as pl
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


from GridMaze.maze import representations as mr
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import folds
from GridMaze.analysis.goal_coding import decoding_utils as du


# %% Global Variables

from GridMaze.paths import RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "goal_decoding" / "place_decoding_control"


# %% Functions


def run_goal_decoding_comparison(
    session,
    resolution=0.5,
    include_multi_units=True,
    window=(-10, 10),
    goal_stratified_validation=True,
    zscore_features=True,
    n_jobs=-1,
    verbose=True,
):
    """ """
    # get downsampled input data containing behavioural info and spike data
    input_data = du.get_place_decoding_input_data(session, resolution, include_multi_units, window, permuted=False)
    simple_maze = session.simple_maze()
    cue_timepoints = np.sort(input_data.event_aligned_bin.cue.dropna().unique())
    reward_timepoints = np.sort(input_data.event_aligned_bin.reward.dropna().unique())
    # organise trials into test-train folds
    folds_df = folds.get_folds_df(session, goal_stratified_validation, return_unique_IDs=True)
    # predict plce/place_direction probabilities from spike counts (for control conditions)
    spatial_probs_dfs = []
    for spatial_output in ["place", "place_direction"]:
        if verbose:
            print(f"Predicting {spatial_output} probabilities from spikes")
        spatial_probs_df = get_predicted_spatial(
            input_data,
            folds_df,
            simple_maze,
            input_type="spikes",
            output_type=spatial_output,
            training_trial_phases=["navigation"],
            n_jobs=n_jobs,
            verbose=False,
        )
        spatial_probs_dfs.append(spatial_probs_df)
    input_data = pd.concat([input_data] + spatial_probs_dfs, axis=1)
    # run xvaled decoding for each condition aross folds
    _folds = folds_df.columns.levels[0].unique()
    results_dfs = []
    for fold in _folds:
        if verbose:
            print(fold)
        # decode goal from spikes, place_probs and place_direction_probs
        fold_df = folds_df[fold]
        test_trials = [t for t in fold_df.test.values.flatten() if isinstance(t, str)]
        train_trials = [t for t in fold_df.train.values.flatten() if isinstance(t, str)]
        train_df = input_data[input_data.trial_unique_ID.isin(train_trials)]
        test_df = input_data[input_data.trial_unique_ID.isin(test_trials)]
        # NOTE: not finding opt reg with nested CV, features between conditions similar in number (no reg)
        # could adapt later
        decoder = LogisticRegression(penalty=None, max_iter=10000, random_state=0, class_weight="balanced")
        for event, timepoints in zip(["cue", "reward"], [cue_timepoints, reward_timepoints]):
            for t in timepoints:
                _train_df = train_df[train_df.event_aligned_bin[event] == t]
                _test_df = test_df[test_df.event_aligned_bin[event] == t]
                if _train_df.empty or _test_df.empty:
                    continue
                y_train, y_test = _train_df.goal.values, _test_df.goal.values
                n_test_samp = _test_df.shape[0]
                res = pd.DataFrame(
                    {
                        ("timepoint", ""): np.repeat(t, n_test_samp),
                        ("trial_unique_ID", ""): _test_df.trial_unique_ID.values,
                        ("steps_to_goal", ""): _test_df.steps_to_goal.future.values,
                        ("trial_phase", ""): _test_df.trial_phase.values,
                        ("true_goal", ""): y_test,
                    }
                )
                # predict goal from spikes or spatial probs
                for feature_set in ["spike_count", "place_prob", "place_direction_prob"]:
                    X_train, X_test = _train_df[feature_set].values, _test_df[feature_set].values
                    if zscore_features:  # zscore features
                        scaler = StandardScaler()  # mean=0, std=1 per column
                        scaler.fit(X_train)  # learn stats on train
                        X_train = scaler.transform(X_train)
                        X_test = scaler.transform(X_test)
                    decoder.fit(X_train, y_train)
                    y_pred = decoder.predict(X_test)
                    res[("predicted_goal", feature_set)] = y_pred
                    res[("accuracy", feature_set)] = (y_pred == y_test).astype(int)  # eval
                results_dfs.append(res)
    results_df = pd.concat(results_dfs, axis=0)
    results_df.reset_index(drop=True, inplace=True)
    # add session info
    results_df["subject_ID"] = session.subject_ID
    results_df["maze_name"] = session.maze_name
    results_df["day_on_maze"] = session.day_on_maze
    results_df["goal_subset"] = session.goal_subset
    return results_df


# %%
def get_predicted_spatial(
    input_data,
    folds_df,
    simple_maze,
    input_type="spikes",
    output_type="place",
    training_trial_phases=["navigation"],
    n_jobs=False,
    verbose=True,
):
    """
    From some input_data, and folds_df dataframes, preform cross-validated prediction
    of place_direction from spike counts (w/ Logisitic Rergression classifier).

    Outputs the neural representation of place direction in the data as
    a probability distribution over the place directions or just place.

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
    if n_jobs:
        dfs = Parallel(n_jobs=n_jobs, verbose=False)(
            delayed(_process_predict_spatial_fold)(
                fold,
                input_data,
                folds_df,
                input_type,
                output_type,
                training_trial_phases,
                all_features,
                verbose,
            )
            for fold in _folds
        )
    else:
        dfs = [
            _process_predict_spatial_fold(
                fold, input_data, folds_df, input_type, output_type, training_trial_phases, all_features, verbose
            )
            for fold in _folds
        ]
    # combine folds and ensure index lines up with input_data
    probs_df = pd.concat(dfs, axis=0)
    probs_df.sort_index(axis=0, inplace=True)
    assert probs_df.index.equals(input_data.index)
    return probs_df


def _process_predict_spatial_fold(
    fold,
    input_data,
    folds_df,
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
