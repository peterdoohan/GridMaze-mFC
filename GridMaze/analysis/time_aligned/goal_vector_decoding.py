"""
Script for decoding egocentric goal vector for neural activity
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import euclidean

from GridMaze.maze import representations as mr
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.processing.align_activity import align_signals

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, ANALYSIS_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(ANALYSIS_INFO_PATH / "intra_trial_interval_times.json", "r") as input_file:
    INTRA_TRIAL_INTERVAL_TIMES = json.load(input_file)

SIGNAL_SAMPLE_RATE = 25  # Hz
FRAME_RATE = 60  # Hz
PRE_WINDOW = 5  # s

# %% Functions


"""
TODO:
- swap to event aligned rates
- make goal in pre-cue goal same as post-cue goal DONE
- make euclidean distance the performance error DONE
- normalise this error by the varaince in egocentric goal vector measures at each timepoint DONE
    (distance from the mean, to be the same as error metric)
- add smoothing DONE
"""


def test_decoding(session, plot=True, smooth_SD=4):
    """ """
    # get goal vecotor data
    (
        GVs,  # [n_timepoints, 2(x,y), n_trials]
        valid_trials,
        times1,
        min_max_stretch,
    ) = get_session_goal_vectors(session, smooth_SD=smooth_SD).values()
    # get neural data
    (rates, times2) = get_neural_data(session, smooth_SD=smooth_SD)  # rates [n_trials, n_clusters, n_timepoints]
    # check time alignment
    assert (times1 == times2).all(), "Timepoints are not aligned"
    # partition trials into training and testing sets
    validation_folds_df = filt.get_goal_stratified_validation_folds_df(
        session, test_trials_per_goal=1, only_include_trials=valid_trials
    )
    folds = validation_folds_df.columns.get_level_values(0).unique()
    results = np.zeros((len(folds), len(times1)))
    for f, fold in enumerate(folds):
        fold_df = validation_folds_df[fold]
        train_trials = fold_df.train.values.flatten()
        test_trials = fold_df.test.values.flatten()
        train_trials_mask = np.isin(valid_trials, train_trials)
        test_trials_mask = np.isin(valid_trials, test_trials)
        GV_train = GVs[:, :, train_trials_mask]  # [n_timepoints, 2(x,y), n_train_trials]
        GV_test = GVs[:, :, test_trials_mask]  # [n_timepoints, 2(x,y), n_test_trials]
        rates_train = rates[train_trials_mask, :, :]  # [n_train_trials, n_clusters, n_timepoints]
        rates_test = rates[test_trials_mask, :, :]  # [n_test_trials, n_clusters, n_timepoints]
        # fit model
        model = LinearRegression()
        for i in range(len(times1)):
            X_train = rates_train[:, :, i]  # [n_train_trials, n_clusters]
            X_test = rates_test[:, :, i]  # [n_test_trials, n_clusters]
            y_train = GV_train[i, :, :].T  # [n_train_trials, 2(x,y)]
            y_test = GV_test[i, :, :].T  # [n_test_trials, 2(x,y)]
            # fit model
            model.fit(X_train, y_train)
            # predict
            y_pred = model.predict(X_test)
            # evaluate
            error = np.linalg.norm(y_pred - y_test, axis=1)
            # error = mean_squared_error(y_test, y_pred, multioutput="raw_values").sum()
            results[f, i] = error.mean()
    if plot:
        # normalise results
        d = get_goal_vector_variance(GVs)
        norm_err = results / d
        f, ax = plt.subplots()
        ax.plot(times1, norm_err.mean(axis=0))
        # ax.fill_between(
        #     times1,
        #     norm_err.mean(axis=0) - norm_err.std(axis=0),
        #     norm_err.mean(axis=0) + norm_err.std(axis=0),
        #     alpha=0.5,
        # )
        ax.set_ylabel("Norm Decoding Error")
        for i in INTRA_TRIAL_INTERVAL_TIMES.values():
            ax.axvline(i, color="red", linestyle="--")
        ax.set_ylim(0, 10)

        return norm_err


def get_goal_vector_variance(goal_vectors):
    """
    Input: goal_vectors [n_timepoints, 2(x,y), n_trials]
    Returns: Variance of euclidean distances from mean goal vector at each timepoint
    """
    mean_vectors = goal_vectors.mean(axis=2)
    distances = np.zeros((goal_vectors.shape[0], goal_vectors.shape[2]))  # [n_timepoints, n_trials]
    for i in range(goal_vectors.shape[2]):
        distances[:, i] = np.linalg.norm(mean_vectors - goal_vectors[:, :, i], axis=1)
    return distances.std(axis=1)  # [n_timepoints]


def get_session_goal_vectors(session, smooth_SD=4):
    """
    returns a df with goal vectors [x,y] for each timepoints (warped to trial events same as trial_aligned_rates_df
    analysis data structure) over trials.
    """
    navigation_df = session.navigation_df.copy()
    # goal in ITI currently defined as previous goal, change to def to upcoming goal
    navigation_df.loc[(navigation_df.trial_phase == "ITI"), ("goal", "")] = np.nan
    navigation_df.loc[:, ("goal", "")] = navigation_df.goal.bfill().ffill()
    signal_times = navigation_df.time.values * 1000  # expected in ms for alignment
    ego_goal_vector = get_egocentric_goal_vector(navigation_df, session.simple_maze())
    trials_df = session.trials_df
    trial_times = trials_df.time.drop("ITI_start", axis=1)
    valid_trials_mask = trial_times.diff(axis=1).trial_end.gt(0)
    valid_trials = trials_df[valid_trials_mask].trial.values
    event_times = (
        trial_times[valid_trials_mask].reset_index(drop=True).to_numpy() * 1000
    )  # expected in ms for alignment
    target_times = np.multiply(list(INTRA_TRIAL_INTERVAL_TIMES.values()), 1000)  # expected in ms for alignment
    # return ego_goal_vector, signal_times, trial_times, target_times
    aligned = align_signals(
        ego_goal_vector.T,  # [2, n_frames]
        signal_times,
        event_times,
        target_times,
        pre_win=(PRE_WINDOW * 1000),
        fs_out=SIGNAL_SAMPLE_RATE,
        plot_warp=False,
    )
    if smooth_SD:
        aligned["aligned_signals"] = gaussian_filter1d(aligned["aligned_signals"], smooth_SD, axis=0)
    return {
        "aligned_goal_vectors": aligned["aligned_signals"].T,  # [n_timepoints, 2(x,y), n_trials]
        "trials": valid_trials,
        "times": aligned["t_out"] / 1000,
        "min_max_stretch": aligned["min_max_stretch"],
    }


def get_egocentric_goal_vector(navigation_df, simple_maze):
    """
    Get egocentric goal vector over all trial times. Note cannot use
    precalcualte egocentric angle to goal & distances in navigation_df
    bc these is only defined over the navigation period of each trial.
    """
    label2pos = mr.get_maze_label2position(simple_maze)

    def _map_goal2pos(g):
        if not isinstance(g, str):
            return (np.nan, np.nan)
        else:
            return label2pos[g]

    goal_pos = navigation_df.goal.map(_map_goal2pos)  # pd.Series of tuples
    goal_pos = np.array(goal_pos.tolist())  # [n_frames, 2 (x,y)]
    current_pos = navigation_df.centroid_position.values  # [n_frames, 2 (x,y)]
    # get egocentric angle to goal
    dy = goal_pos[:, 1] - current_pos[:, 1]
    dx = goal_pos[:, 0] - current_pos[:, 0]
    allocentric_angles = np.degrees(np.arctan2(dx, dy)) % 360
    head_directions = navigation_df.head_direction.value.values
    egocentric_angles = (allocentric_angles - head_directions) % 360
    # get euclidean distance to goal
    distance_to_goal = np.linalg.norm(goal_pos - current_pos, axis=1)
    # get goal vector
    ego_angles_rad = np.deg2rad(egocentric_angles)
    V_x = distance_to_goal * np.cos(ego_angles_rad)
    V_y = distance_to_goal * np.sin(ego_angles_rad)
    goal_vector = np.column_stack((V_x, V_y))
    return goal_vector


# %%


def get_neural_data(session, trials="all", include_multi_units=False, return_timepoints=True, smooth_SD=4):
    """ """
    trial_aligned_rates_df = session.trial_aligned_rates_df
    # filter clusters
    keep_clusters = gc.filter_clusters(
        session.cluster_metrics, session.session_info, return_unique_IDs=True, multi_units=include_multi_units
    )
    trial_aligned_rates_df = trial_aligned_rates_df[trial_aligned_rates_df.cluster_unique_ID.isin(keep_clusters)]
    # filter trials
    if trials != "all":
        trial_aligned_rates_df = trial_aligned_rates_df[trial_aligned_rates_df.trial.isin(trials)]
    n_trials = trial_aligned_rates_df.trial.unique().shape[0]
    n_clusters = trial_aligned_rates_df.cluster_unique_ID.unique().shape[0]
    n_timepoints = trial_aligned_rates_df.firing_rate.shape[1]
    _rates = trial_aligned_rates_df.set_index(
        [("trial", ""), ("cluster_unique_ID", "")]
    ).firing_rate.values  # [n_trials * n_clusters, n_timepoints]
    rates = _rates.reshape(n_trials, n_clusters, n_timepoints)  # [n_trials, n_clusters, n_timepoints]
    if smooth_SD:
        rates = gaussian_filter1d(rates, smooth_SD, axis=2)
    if return_timepoints:
        timepoints = trial_aligned_rates_df.firing_rate.columns.values.astype(float)
        return rates, timepoints
    else:
        return rates


# %%


def get_event_aligned_neural_data(session, include_multi_units=False, return_timepoints=True):
    """ """
    event_aligned_rates_df = session.event_aligned_rates_df
    # filter clusters
    keep_clusters = gc.filter_clusters(
        session.cluster_metrics, session.session_info, return_unique_IDs=True, multi_units=include_multi_units
    )
    event_aligned_rates_df = event_aligned_rates_df[event_aligned_rates_df.cluster_unique_ID.isin(keep_clusters)]
    n_trials = event_aligned_rates_df.trial.unique().shape[0]
    n_clusters = event_aligned_rates_df.cluster_unique_ID.unique().shape[0]
    n_timepoints = event_aligned_rates_df.firing_rate.shape[1]
    _rates = event_aligned_rates_df.set_index(
        [("trial", "", ""), ("cluster_unique_ID", "", "")]
    ).firing_rate.values  # [n_trials * n_clusters, n_timepoints]
    rates = _rates.reshape(n_trials, n_clusters, n_timepoints)  # [n_trials, n_clusters, n_timepoints]
    if return_timepoints:
        timepoints = event_aligned_rates_df.firing_rate.columns.values
        return rates, timepoints
    else:
        return rates


def get_event_aligned_egocentric_goal_vector_and_rates(session, event, window=(-10, 10), smooth_SD=False):
    """ """
    # convert window to frames
    window_frames = np.array(window) * FRAME_RATE
    navigation_activity_df = session.get_navigation_activity_df(type="rates")  # only includes single units
    # goal in ITI currently defined as previous goa, change to next goal
    navigation_activity_df.loc[(navigation_activity_df.trial_phase == "ITI"), ("goal", "")] = np.nan
    navigation_activity_df.loc[:, ("goal", "")] = navigation_activity_df.goal.bfill().ffill()
    # get event aligned goal vector & rates
    times = navigation_activity_df.time.values  # [n_timepoints]
    goal_vector = get_egocentric_goal_vector(navigation_activity_df, session.simple_maze())  # [n_timepoints, 2]
    rates = navigation_activity_df.firing_rate.values  # [n_timepoints, n_clusters]
    event_times = session.trials_df.time[event].values  # [n_trials]
    Vs, Rs, trials = [], [], []
    for i, t in enumerate(event_times):
        idx = np.argmin(np.abs(times - t))
        idx_range = np.arange(idx + window_frames[0], idx + window_frames[1])
        if idx_range[0] < 0 or idx_range[-1] > len(times):
            continue
        trials.append(i + 1)
        Vs.append(goal_vector[idx_range])
        Rs.append(rates[idx_range])
    aligned_goal_vectors = np.array(Vs)  # [n_trials, n_timepoints, 2]
    aligned_rates = np.array(Rs)  # [n_trials, n_timepoints, n_clusters]
    if smooth_SD:
        aligned_goal_vectors = gaussian_filter1d(aligned_goal_vectors, smooth_SD, axis=1)
        aligned_rates = gaussian_filter1d(aligned_rates, smooth_SD, axis=1)
    return aligned_goal_vectors, aligned_rates, np.array(trials)


def test_decoding2(session, event="cue", plot=True, smooth_SD=False):
    """"""
    V, R, trials = get_event_aligned_egocentric_goal_vector_and_rates(session, event, smooth_SD=smooth_SD)
    validation_folds_df = filt.get_goal_stratified_validation_folds_df(
        session, test_trials_per_goal=1, only_include_trials=trials
    )
    folds = validation_folds_df.columns.get_level_values(0).unique()
    results = np.zeros((len(folds), V.shape[1]))
    for f, fold in enumerate(folds):
        fold_df = validation_folds_df[fold]
        train_trials = fold_df.train.values.flatten()
        test_trials = fold_df.test.values.flatten()
        train_trials_mask = np.isin(trials, train_trials)
        test_trials_mask = np.isin(trials, test_trials)
        GV_train = V[train_trials_mask, :, :]  # [n_train_trials, n_timepoints, 2(x,y)]
        GV_test = V[test_trials_mask, :, :]  # [ n_test_trials, n_timepoints, 2(x,y)]
        rates_train = R[train_trials_mask, :, :]  # [n_train_trials, n_timepoints, n_clusters]
        rates_test = R[test_trials_mask, :, :]  # [n_test_trials, n_timepoints,  n_clusters]
        # fit model
        model = Ridge(alpha=1e6)
        for i in range(V.shape[1]):
            X_train = rates_train[:, i, :]  # [n_train_trials, n_clusters]
            X_test = rates_test[:, i, :]  # [n_test_trials, n_clusters]
            y_train = GV_train[:, i, :]  # [n_train_trials, 2(x,y)]
            y_test = GV_test[:, i, :]  # [n_test_trials, 2(x,y)]
            # fit model
            model.fit(X_train, y_train)
            # predict
            y_pred = model.predict(X_test)
            # evaluate
            error = np.linalg.norm(y_pred - y_test, axis=1)
            results[f, i] = error.mean()

    # get normalised decoding error
    r = results.mean(axis=0)  # average over folds
    d = get_goal_vector_variance2(V)  # variance in goal vector signal at each timepoint
    # r = r / d  # normalised performance error
    # plot results
    if plot:
        f, ax = plt.subplots()
        ax.plot(r)
        ax.set_ylabel("Norm Decoding Error")
        ax.axvline(len(r) // 2, color="k", linestyle="--", alpha=0.5)
        ax.set_xlabel(event)
    return r


def _run_regression(V, R, trials, validation_folds_df, alpha):
    """ """
    folds = validation_folds_df.columns.get_level_values(0).unique()
    test_perf = np.zeros((len(folds), V.shape[1]))
    train_perf = np.zeros((len(folds), V.shape[1]))
    for f, fold in enumerate(folds):
        fold_df = validation_folds_df[fold]
        train_trials = fold_df.train.values.flatten()
        test_trials = fold_df.test.values.flatten()
        train_trials_mask = np.isin(trials, train_trials)
        test_trials_mask = np.isin(trials, test_trials)
        GV_train = V[train_trials_mask, :, :]  # [n_train_trials, n_timepoints, 2(x,y)]
        GV_test = V[test_trials_mask, :, :]  # [ n_test_trials, n_timepoints, 2(x,y)]
        rates_train = R[train_trials_mask, :, :]  # [n_train_trials, n_timepoints, n_clusters]
        rates_test = R[test_trials_mask, :, :]  # [n_test_trials, n_timepoints,  n_clusters]
        # fit model
        model = Ridge(alpha=alpha)
        for i in range(V.shape[1]):
            X_train = rates_train[:, i, :]  # [n_train_trials, n_clusters]
            X_test = rates_test[:, i, :]  # [n_test_trials, n_clusters]
            y_train = GV_train[:, i, :]  # [n_train_trials, 2(x,y)]
            y_test = GV_test[:, i, :]  # [n_test_trials, 2(x,y)]
            # fit model
            model.fit(X_train, y_train)
            # predict
            y_pred_test = model.predict(X_test)
            # evaluate
            test_error = np.linalg.norm(y_pred_test - y_test, axis=1)
            test_perf[f, i] = test_error.mean()
            # training perf
            y_pred_train = model.predict(X_train)
            train_error = np.linalg.norm(y_pred_train - y_train, axis=1)
            train_perf[f, i] = train_error.mean()
    return test_perf, train_perf


def get_goal_vector_variance2(V):
    """ """
    n_trials, n_timepoints, _ = V.shape
    mean = V.mean(axis=0)
    distances = np.zeros((n_timepoints, n_trials))  # [n_timepoints, n_trials]
    for i in range(n_trials):
        distances[:, i] = np.linalg.norm(mean - V[i, :, :], axis=1)
    return distances.std(axis=1)  # [n_timepoints]


# %% test regularisation


def test_regularisation(session, event="cue", smooth_SD=False):
    """ """
    alphas = np.logspace(8, 16, 10)
    test_perfs, train_perfs = [], []
    V, R, trials = get_event_aligned_egocentric_goal_vector_and_rates(session, event, smooth_SD=smooth_SD)
    validation_folds_df = filt.get_goal_stratified_validation_folds_df(
        session, test_trials_per_goal=1, only_include_trials=trials
    )
    for alpha in alphas:
        test_perf, train_perf = _run_regression(V, R, trials, validation_folds_df, alpha)
        test_perfs.append(test_perf.mean())
        train_perfs.append(train_perf.mean())
    return test_perfs, train_perfs
