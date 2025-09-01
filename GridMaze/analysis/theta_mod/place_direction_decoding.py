"""
Can we test if place-direction decoding is modulated by theta-phase?
@peterdoohan
"""

# %% Imports
import numpy as np
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
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


# %% Functions


def quick_plot(results_df):
    spatial_df = results_df[results_df.input_type == "spatial"]
    spatial_error = spatial_df.groupby(["mode", "offset"]).error.mean()
    theta_df = results_df[(results_df.input_type == "spatial_spikes") & (results_df.theta_phase != "mean")]
    theta_error = theta_df.groupby(["mode", "offset", "theta_phase"]).error.mean().unstack(level=[0, 1])
    # diff from spatial error
    theta_error_res = theta_error.sub(spatial_error, axis=1)
    # demean
    norm_theta_error_df = theta_error_res.sub(theta_error_res.mean(axis=0), axis=1)
    # plot
    return norm_theta_error_df.plot()


def test(
    session,
    include_multi_units=True,
    max_steps_to_goal=30,
    resolution=0.2,
    modes=["future"],
    offset=10,
    sqrt_spikes=False,
    alpha=1,
    normalise_X=True,
    spikes_reg_weight=0.1,
    n_folds=5,
    verbose=True,
):
    # input data
    if verbose:
        print("Loading input data...")
    input_df = get_input_df(
        session,
        include_multi_units,
        max_steps_to_goal,
        resolution,
        modes,
        offset,
        state_type="place",
    )
    theta_phases = input_df.spike_count.columns.get_level_values(1).unique().astype(float)
    simple_maze = session.simple_maze()
    all_pairs_path_length = _get_all_pairs_path_length(simple_maze)
    results = []
    for mode in modes:
        for off in range(0, offset):
            _input_df = input_df[~input_df[mode][off].isnull()]
            # mean spikes over theta phases (for training)
            S_mean = _input_df.spike_count.T.groupby(level=0).mean().T.values
            S_thetas = [_input_df.spike_count.xs(p, level=1, axis=1).values for p in theta_phases]
            if sqrt_spikes:
                S_mean = np.sqrt(S_mean)
                S_thetas = [np.sqrt(S) for S in S_thetas]
            if spikes_reg_weight is not None:
                S_mean = S_mean * spikes_reg_weight
                S_thetas = [S * spikes_reg_weight for S in S_thetas]
            N = convert.place_direction2onehot(
                _input_df.place_direction.values, simple_maze=simple_maze
            )  # nusance regressors for current pd
            Y = _input_df[mode][off].values  # future/past place/place-direction (what we are predicting from spikes)
            # combine spike arrays with current state onehots
            X_mean = np.concat([N, S_mean], axis=1)
            X_thetas = [np.concat([N, x], axis=1) for x in S_thetas]
            # define cv folds based on trials available in this subsampling of the data
            folds_df = folds.get_folds_df(
                session,
                goal_stratified=False,
                valid_trials=_input_df.trial.unique(),
                n_folds=n_folds,
                return_unique_IDs=True,
            )
            _folds = folds_df.columns.get_level_values(0).unique()
            for fold in _folds:
                if verbose:
                    print(f"  fold {fold}")
                fold_df = folds_df[fold]
                train_trials, test_trials = [fold_df[t].unstack().dropna().values for t in ["train", "test"]]
                train_mask, test_mask = [
                    _input_df.trial_unique_ID.isin(trials).values for trials in [train_trials, test_trials]
                ]
                Y_train, Y_test = Y[train_mask], Y[test_mask]
                # init results df (contains info rel to sample predictions eg, trial, moving etc.)
                res = (
                    _input_df[test_mask].drop(columns=["spike_count"] + modes, level=0).droplevel(axis=1, level=[1, 2])
                )
                # first test on baseline spatial input
                _res = res.copy()
                X_train, X_test = N[train_mask, :], N[test_mask, :]
                if normalise_X:
                    scaler = StandardScaler()
                    scaler.fit(X_train)
                    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
                # fit model
                model = LogisticRegression(C=alpha, random_state=0, max_iter=10_000, class_weight="balanced")
                model.fit(X_train, Y_train)
                features = list(model.classes_)
                # get weighted dist error as output
                Yprobs = model.predict_proba(X_test)
                dist_mat = _get_dist_mat(
                    Y_test, Yprobs, features, all_pairs_path_length
                )  # dist between features and true pos
                weighted_dist_err = np.sum(Yprobs * dist_mat, axis=1)
                Y_hat = model.predict(X_test)
                acc = (Y_test == Y_hat).astype(int)
                _res[("error")] = weighted_dist_err
                _res[("input_type")] = "spatial"
                _res[("theta_phase")] = None
                _res[("fold")] = fold
                _res[("mode")] = mode
                _res[("offset")] = off
                results.append(_res)
                # next test spatial input + spikes form different theta phases
                # train model on average theta
                X_train = X_mean[train_mask, :]
                if normalise_X:
                    scaler = StandardScaler()
                    scaler.fit(X_train)
                    X_train = scaler.transform(X_train)
                # fit model
                model = LogisticRegression(C=alpha, random_state=0, max_iter=10_000, class_weight="balanced")
                model.fit(X_train, Y_train)
                # test performance on spikes form different theta phases
                for label, Xtheta in zip(["mean"] + theta_phases.to_list(), [X_mean] + X_thetas):
                    _res = res.copy()
                    X_test = Xtheta[test_mask, :]
                    if normalise_X:
                        X_test = scaler.transform(X_test)
                    # fit model
                    Yprobs = model.predict_proba(X_test)
                    weighted_dist_err = np.sum(Yprobs * dist_mat, axis=1)
                    _res[("error")] = weighted_dist_err
                    _res[("input_type")] = "spatial_spikes"
                    _res[("theta_phase")] = label
                    _res[("fold")] = fold
                    _res[("mode")] = mode
                    _res[("offset")] = off
                    results.append(_res)
    return pd.concat(results, axis=0)


def _get_dist_mat(Y_test, Yprobs, features, all_pairs_path_length):
    dist_mat = np.zeros_like(Yprobs)
    for i, src in enumerate(Y_test):
        for j, targ in enumerate(features):
            dist_mat[i, j] = all_pairs_path_length[src][targ]
    return dist_mat


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


# %%
def get_input_df(
    session,
    include_multi_units=True,
    max_steps_to_goal=30,
    resolution=0.2,
    modes=["future", "past"],
    offset=12,
    state_type="place",
):
    """
    Note slightly hacky way of grabing the future/past states but don't want to change Kris' original code
    """
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
