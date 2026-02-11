"""
Decode both place-direction and distance to goal (over all spikes) and see if errors are dynamically correlated
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from GridMaze.analysis.core import folds
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.theta_mod import double_decoding as tdd

# %% global variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "theta_mod" / "double_decoding_simple"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)


# %% Functions


def get_double_decoding_df(verbose=True, n_jobs=-1, save=False):
    """ """
    save_path = RESULTS_DIR / "double_decoding_simple_df.parquet"
    if save_path.exists() and not save:
        if verbose:
            print(f"Loading existing double results from {save_path} ...")
        return pd.read_parquet(save_path)
    results_dfs = []
    for subject_ID in SUBJECT_IDS:
        for maze_name in ["maze_1", "maze_2", "rooms_maze"]:
            if verbose:
                print(f"Loading sessions for {subject_ID} - {maze_name} ...")
            sessions = gs.get_maze_sessions(
                subject_IDs=[subject_ID],
                maze_names=[maze_name],
                days_on_maze="late",
                with_data=[
                    "navigation_df",
                    "cluster_metrics",
                    "trials_df",
                    "navigation_theta_spike_counts_df",
                ],
                must_have_data=True,
            )
            if n_jobs:
                dfs = Parallel(n_jobs=n_jobs)(delayed(get_session_double_decoding_df)(session) for session in sessions)
            else:
                dfs = [get_session_double_decoding_df(session) for session in sessions]
            results_dfs.extend(dfs)
    results_df = pd.concat(results_dfs, axis=0, ignore_index=True)
    if save:
        results_df.to_parquet(save_path)
        if verbose:
            print(f"Saved double decoding results to {save_path}.")
    return results_df


def get_session_double_decoding_df(
    session,
    resolution=0.1,
    sum_spike_window=0.4,
    moving_only=True,
    bin_spacing=0.08,
    max_distance=None,
    max_steps_from_goal=30,
    n_folds=8,
    sqrt_spikes=True,
    normalise_X=True,
    alpha="opt",
    output="weighted",
    verbose=True,
):
    """ """
    # load data
    if verbose:
        print(f"{session.name}: loading input data...")
    input_data = tdd.get_input_data(
        session,
        theta_split=False,
        resolution=resolution,
        sum_spike_window=sum_spike_window,
        moving_only=moving_only,
        bin_spacing=bin_spacing,
        max_distance=max_distance,
        max_steps_to_goal=max_steps_from_goal,
    )
    input_data = input_data.droplevel(2, axis=1)
    # generate variables to be used across folds, reg validation etc.
    distances = np.sort(input_data.distance_bin_mid.unique())  # in order corresponding to bin_id [0, 1, ...]
    distance_bin_ids = np.sort(input_data.distance_bin_id.unique())
    all_pairs_path_length = tdd._get_all_pairs_path_length(session)
    folds_df = folds.get_folds_df(
        session,
        goal_stratified=False,
        n_folds=n_folds,
        return_unique_IDs=False,
    )

    # init results df
    results_df = pd.concat(
        [
            input_data.drop(["spike_count", "past", "future"], axis=1, level=0).copy(),
            pd.DataFrame(
                index=input_data.index,
                columns=pd.MultiIndex.from_product((["decoded_distance"], ["from_distance", "from_place"])),
            ),
        ],
        axis=1,
    )
    results_df[("place_decoding_info", "traj_defined")] = False
    results_df[("place_decoding_info", "in_train")] = False

    _folds = folds_df.columns.get_level_values(0).unique()
    for fold in _folds:
        if verbose:
            print(fold)
        fold_df = folds_df[fold]
        train_trials, test_trials = [fold_df[t].unstack().dropna().values for t in ["train", "test"]]
        train_df, test_df = [input_data[input_data.trial.isin(trials)] for trials in [train_trials, test_trials]]
        # train decoder on mean spikes across theta phases
        X_train, X_test = [df.spike_count.values for df in [train_df, test_df]]
        if sqrt_spikes:
            X_train, X_test = np.sqrt(X_train), np.sqrt(X_test)
        if normalise_X:
            scaler = StandardScaler().fit(X_train)
            X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
        # decoder either distace-to-goal or place (we will set up different decoders for each)
        Yd_train, Yd_test = [df.distance_bin_id.values for df in [train_df, test_df]]
        Yp_train, Yp_test = [df.maze_position.simple.values for df in [train_df, test_df]]
        # optionaly find optimal xval regularisation
        if alpha == "opt":
            if verbose:
                print("    Finding optimal alpha for distance decoder...")
            d_alpha = tdd.get_opt_alpha(
                fold_df,
                train_df,
                var="distance_to_goal",
                normalise_X=normalise_X,
                sqrt_spikes=sqrt_spikes,
                distances=distances,
                distance_bin_ids=distance_bin_ids,
                output=output,
                verbose=verbose,
            )
            if verbose:
                print("    Finding optimal alpha for place decoder...")
                p_alpha = tdd.get_opt_alpha(
                    fold_df,
                    train_df,
                    var="place",
                    normalise_X=normalise_X,
                    sqrt_spikes=sqrt_spikes,
                    output=output,
                    distance_ref="goal",
                    restrict_to_traj=False,
                    all_pairs_path_length=all_pairs_path_length,
                    verbose=verbose,
                )
        else:
            d_alpha, p_alpha = alpha, alpha
        # train decoders
        d_decoder = LogisticRegression(C=d_alpha, random_state=0, max_iter=10_000, class_weight="balanced")
        d_decoder.fit(X_train, Yd_train)
        train_distances_bin_ids = d_decoder.classes_
        p_decoder = LogisticRegression(C=p_alpha, random_state=0, max_iter=10_000, class_weight="balanced")
        p_decoder.fit(X_train, Yp_train)
        train_locations = p_decoder.classes_
        # test decoders
        Yd_prob = d_decoder.predict_proba(X_test)
        d_pred = tdd._get_distance_pred_distance(
            Yd_prob,
            distances=distances,
            distance_bin_ids=distance_bin_ids,
            decoder_classes=train_distances_bin_ids,
            output=output,
        )
        results_df.loc[test_df.index, ("decoded_distance", "from_distance")] = d_pred
        Yp_prob = p_decoder.predict_proba(X_test)
        p_pred, _, p_in_train = tdd._get_place_pred_distance(
            Yp_prob,
            Yp_test,
            test_df,
            decoder_classes=train_locations,
            all_pairs_path_length=all_pairs_path_length,
            restrict_to_traj=False,
            output=output,
            distance_ref="goal",
            return_as="all",
        )
        results_df.loc[test_df.index, ("decoded_distance", "from_place")] = p_pred
        results_df.loc[test_df.index, ("place_decoding_info", "in_train")] = p_in_train

    return results_df.reset_index(drop=True)
