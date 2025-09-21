"""
is there theta modulation of the place-direction representation?
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

from GridMaze.analysis.place_direction import future_decoding as fd
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import folds
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import convert

from GridMaze.maze import representations as mr
from GridMaze.maze import plotting as mp


# %% Global Variables
from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]

# %% Functions


def filter_decision_points(summary_df, decision_points="future"):
    dfs = []
    for maze_name in ["maze_1", "maze_2"]:
        maze_df = summary_df[summary_df.maze_name == maze_name]
        simple_maze = mr.get_simple_maze(maze_name)
        if decision_points == "future":
            _decision_points = fd.get_decision_points(
                simple_maze, mode="future", edges_only=True, node_only=False, return_as="strings", plot=False
            )
        elif decision_points == "past":
            _decision_points = fd.get_decision_points(
                simple_maze, mode="past", edges_only=False, node_only=True, return_as="strings", plot=False
            )
        dfs.append(maze_df[maze_df.place_direction.isin(_decision_points)])
    df = pd.concat(dfs, axis=0)
    return df


def get_summary_df(verbose=True):
    dfs, failed_sessions = [], []
    for subject in SUBJECT_IDS:
        for maze_name in MAZE_NAMES:
            sessions = gs.get_maze_sessions(
                subject_IDs=[subject],
                maze_names=[maze_name],
                days_on_maze="late",
                with_data=["navigation_df", "navigation_theta_spike_counts_df", "cluster_metrics", "trials_df"],
                must_have_data=True,
            )
            for session in sessions:
                if verbose:
                    print(session.name)
                try:
                    results_df = get_session_theta_mod_trajectory_error(session)  # defualt settings
                    results_df["subject_ID"] = session.subject_ID
                    results_df["maze_name"] = session.maze_name
                    results_df["day_on_maze"] = session.day_on_maze
                    dfs.append(results_df)
                except Exception as e:
                    print(f"Failed: {session.name}")
                    print(e)
                    failed_sessions.append(session.name)
    summary_df = pd.concat(dfs).reset_index(drop=True)
    return summary_df, failed_sessions


def get_session_theta_mod_trajectory_error(
    session,
    include_multi_units=True,
    max_steps_to_goal=30,
    resolution=0.2,
    envelope=10,
    sqrt_spikes=False,
    alpha=5,
    normalise_X=True,
    n_folds=5,
    verbose=True,
):
    """ """
    # input data
    if verbose:
        print("Loading input data...")
    input_data = get_input_data(
        session,
        include_multi_units=include_multi_units,
        max_steps_to_goal=max_steps_to_goal,
        resolution=resolution,
        modes=["past", "future"],
        offset=envelope,
        state_type="place",
    )
    # include only samples where full future + past envelope is defined to avoid bias
    input_data = input_data[input_data[["past", "future"]].notnull().all(axis=1)]
    valid_trials = input_data.trial.unique()
    theta_phases = input_data.spike_count.columns.get_level_values(1).unique().astype(float)
    simple_maze = session.simple_maze()
    all_pairs_path_length = _get_all_pairs_path_length(simple_maze)
    folds_df = folds.get_folds_df(
        session,
        goal_stratified=False,
        valid_trials=valid_trials,
        n_folds=n_folds,
        return_unique_IDs=False,
    )
    _folds = folds_df.columns.get_level_values(0).unique()
    results = []
    for fold in _folds:
        if verbose:
            print(fold)
        fold_df = folds_df[fold]
        train_trials, test_trials = [fold_df[t].unstack().dropna().values for t in ["train", "test"]]
        train_df, test_df = [input_data[input_data.trial.isin(trials)] for trials in [train_trials, test_trials]]
        # train decoder on mean spikes across theta phases
        X_train_mean, X_test_mean = [df.spike_count.T.groupby(level=0).mean().T.values for df in [train_df, test_df]]
        if sqrt_spikes:
            X_train_mean, X_test_mean = np.sqrt(X_train_mean), np.sqrt(X_test_mean)
        if normalise_X:
            scaler = StandardScaler().fit(X_train_mean)
            X_train_mean, X_test_mean = scaler.transform(X_train_mean), scaler.transform(X_test_mean)
        Y_train, Y_test = [df.maze_position.simple.values for df in [train_df, test_df]]
        decoder = LogisticRegression(C=alpha, random_state=0, max_iter=10_000, class_weight="balanced")
        decoder.fit(X_train_mean, Y_train)
        decoder_classes = decoder.classes_
        # test on spikes at each theta phase
        res = test_df.drop(columns=["spike_count", "past", "future"], level=0).droplevel(axis=1, level=[1, 2])
        for phase in theta_phases:
            X_theta_test = test_df.spike_count.xs(phase, level=1, axis=1).values
            if sqrt_spikes:
                X_theta_test = np.sqrt(X_theta_test)
            if normalise_X:
                X_theta_test = scaler.transform(X_theta_test)
            Yprob = decoder.predict_proba(X_theta_test)  # decoding prob across all maze locs
            weighted_errors, traj_prob_mass = get_weighted_trajectory_position_error(
                Yprob, Y_test, test_df, decoder_classes, all_pairs_path_length
            )
            _res = res.copy()
            _res["theta_phase"] = phase
            _res["fold"] = fold
            _res["weighted_trajectory_error"] = weighted_errors
            _res["trajectory_prob_mass"] = traj_prob_mass
            results.append(_res)
    results_df = pd.concat(results).reset_index(drop=True)
    return results_df


def get_weighted_trajectory_position_error(Yprob, Y_test, test_df, decoder_classes, all_pairs_path_length):
    """ """
    samples = Yprob.shape[0]
    traj_envelope = test_df[["past", "future"]]  # past & future parts of trajectory
    errors = np.zeros(samples)
    traj_prob_mass = np.zeros(samples)
    for i in range(samples):
        y = Y_test[i]
        probs = Yprob[i]
        traj = traj_envelope.iloc[i]
        past_locs = traj.loc["past"].iloc[1:].dropna().unique()
        future_locs = traj.loc["future"].iloc[1:].dropna().unique()
        traj_locs = traj.dropna().unique()
        traj_loc_mask = np.isin(decoder_classes, traj_locs)
        traj_pmass = np.sum(probs[traj_loc_mask])
        # filter probs to only include those on the trajectory
        past_loc_mask = np.isin(decoder_classes, past_locs)
        past_probs = probs[past_loc_mask]
        past_dists = np.array([all_pairs_path_length[y][loc] for loc in decoder_classes[past_loc_mask]])
        future_loc_mask = np.isin(decoder_classes, future_locs)
        future_probs = probs[future_loc_mask]
        future_dists = np.array([all_pairs_path_length[y][loc] for loc in decoder_classes[future_loc_mask]])
        # calculate decoding error distance over the trajectory (+ve = more future, -ve = more past)
        traj_prob_sum = traj_pmass + 1e-9  # avoid div by 0
        weighted_error = (np.sum(future_probs * future_dists) - np.sum(past_probs * past_dists)) / traj_prob_sum
        errors[i] = weighted_error
        traj_prob_mass[i] = traj_pmass
    return errors, traj_prob_mass


def _get_all_pairs_path_length(simple_maze):
    extended_maze = mr.get_extended_simple_maze(simple_maze)
    dists = dict(nx.all_pairs_dijkstra_path_length(extended_maze, weight="weight"))
    coord2label = mr.get_maze_coord2label(simple_maze)
    _dists = {}
    for src in dists.keys():
        src_dists = dists[src]
        __dists = {}
        for targ in src_dists.keys():
            __dists[coord2label[targ]] = src_dists[targ]
        _dists[coord2label[src]] = __dists
    return _dists


def get_input_data(
    session,
    include_multi_units=True,
    max_steps_to_goal=30,
    resolution=0.2,
    modes=["future", "past"],
    offset=12,
    state_type="place",
):
    """ """
    # load data
    navigation_df = session.navigation_df
    spike_counts_df = session.navigation_theta_spike_counts_df.reset_index(drop=True)

    # filter clusters
    keep_clusters = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=include_multi_units,
    )
    spike_counts_df = spike_counts_df[
        spike_counts_df.columns[spike_counts_df.columns.get_level_values(1).isin(keep_clusters)]
    ]

    # downsample data
    ds_nav_df, ds_spikes_df = ds.downsample_nav_spikes_data(
        navigation_df, spike_counts_df, resolution=resolution, distance_metrics=[("steps_to_goal", "future")]
    )
    ds_nav_df.columns = pd.MultiIndex.from_tuples([(*col, "") for col in ds_nav_df.columns])
    navigation_spikes_df = pd.concat([ds_nav_df, ds_spikes_df], axis=1)

    # add place_direction column
    navigation_spikes_df[("place_direction", "", "")] = (
        navigation_spikes_df.maze_position.simple + "_" + navigation_spikes_df.cardinal_movement_direction
    )

    # add future, past state information
    future_past_df = fd.get_past_and_future_states(
        navigation_spikes_df, state_type=state_type, past_offset=offset, future_offset=offset
    )
    future_past_df = future_past_df[modes]
    future_past_df.columns = pd.MultiIndex.from_tuples([(*col, "") for col in future_past_df.columns])
    navigation_spikes_df = pd.concat([navigation_spikes_df, future_past_df], axis=1)

    # filter data
    navigation_spikes_df = filt.filter_navigation_rates_df(
        navigation_spikes_df,
        navigation_only=True,
        moving_only=True,
        exclude_time_at_goal=True,
        max_steps_to_goal=max_steps_to_goal,
    )
    return navigation_spikes_df
