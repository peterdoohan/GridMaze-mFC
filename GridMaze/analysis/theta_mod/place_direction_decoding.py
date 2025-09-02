"""
Can we test if place-direction decoding is modulated by theta-phase?
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

RESULTS_DIR = RESULTS_PATH / "theta_mod" / "future_place_direction"


# %% plotting


def plot_theta_mod_decoding_summary(
    summary_df,
    maze_names=["maze_1", "maze_2", "rooms_maze"],
    mode="future",
    decision_points="future",
    offset_range=(1, 6),
    demean=True,
    ax=None,
):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xlabel("theta phase")
    ax.set_ylabel("decoding bias (steps)")
    # filter data
    df = summary_df[summary_df["mode"] == mode]
    df = df[df.maze_name.isin(maze_names)]
    if decision_points:
        df = fd._filter_for_decision_points(df, decision_points=decision_points)
    if offset_range:
        df = df[df.offset.between(*offset_range)]
    # decoding from spatial info alone
    spatial_df = df[df.input_type == "spatial"]
    spatial_error = spatial_df.groupby(["subject_ID", "offset"]).error.mean()
    # decoding from spatial + spikes at different theta phases
    theta_df = df[df.input_type == "spatial_spikes_theta"]
    theta_error = theta_df.groupby(["subject_ID", "offset", "theta_phase"]).error.mean().unstack(level=2)
    # diff from spatial error
    theta_error_res = theta_error.sub(spatial_error, axis=0)
    # demean
    if demean:
        theta_error_res = theta_error_res.sub(theta_error_res.mean(axis=1), axis=0)
    subject_grouped = theta_error_res.groupby(level=1)
    grand_mean = subject_grouped.mean()
    grand_sem = subject_grouped.sem()
    # plot
    theta_phases = grand_mean.columns.values
    offsets = grand_mean.index.values
    colors = sns.color_palette("viridis", n_colors=len(offsets))
    for off, color in zip(offsets, colors):
        ax.errorbar(
            theta_phases,
            grand_mean.loc[off],
            yerr=grand_sem.loc[off],
            fmt="o-",
            markersize=6,
            linewidth=2,
            capsize=None,
            elinewidth=2,
            label=off,
            color=color,
        )
    ax.legend(fontsize=8)


# %% summary level decoding


def get_theta_mod_future_decoding_summary(save=False, verbose=True):
    """ """
    save_path = RESULTS_DIR / f"theta_mod_decoding_summary.parquet"
    if not save and save_path.exists():
        if verbose:
            print(f"Loading existing results from {save_path}")
        return pd.read_parquet(save_path)
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
                    results_df = get_session_theta_mod_future_decoding(session)  # defualt settings
                    results_df["subject_ID"] = session.subject_ID
                    results_df["maze_name"] = session.maze_name
                    results_df["day_on_maze"] = session.day_on_maze
                    dfs.append(results_df)
                except Exception as e:
                    print(f"Error processing session {session.name}: {e}")
                    failed_sessions.append(session.name)
    summary_df = pd.concat(dfs, axis=0)
    if save:
        summary_df.to_parquet(save_path)
        if verbose:
            print(f"Saving results to {save_path}")
    return summary_df, failed_sessions


# %% session lvl processing


def get_session_theta_mod_future_decoding(
    session,
    include_multi_units=True,
    max_steps_to_goal=30,
    resolution=0.2,
    modes=["past", "future"],
    offset=12,
    sqrt_spikes=False,
    alpha=1,
    normalise_X=True,
    spikes_reg_weight=0.1,
    n_folds=5,
    n_jobs=-1,
    verbose=False,
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
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_offset)(
            mode,
            off,
            input_df,
            theta_phases,
            sqrt_spikes,
            spikes_reg_weight,
            simple_maze,
            session,
            n_folds,
            normalise_X,
            modes,
            alpha,
            all_pairs_path_length,
            verbose,
        )
        for off in range(1, offset + 1)
        for mode in modes
    )
    _results = []
    for res in results:
        _results.extend(res)
    results_df = pd.concat(_results, axis=0)

    return results_df


def _process_offset(
    mode,
    off,
    input_df,
    theta_phases,
    sqrt_spikes,
    spikes_reg_weight,
    simple_maze,
    session,
    n_folds,
    normalise_X,
    modes,
    alpha,
    all_pairs_path_length,
    verbose,
):
    offset_results = []
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
        res = _input_df[test_mask].drop(columns=["spike_count"] + modes, level=0).droplevel(axis=1, level=[1, 2])
        res["fold"] = fold
        res["mode"] = mode
        res["offset"] = off
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
        dist_mat = _get_dist_mat(Y_test, Yprobs, features, all_pairs_path_length)  # dist between features and true pos
        weighted_dist_err = np.sum(Yprobs * dist_mat, axis=1)
        _res["error"] = weighted_dist_err
        _res["input_type"] = "spatial"
        offset_results.append(_res)
        # next test spatial input + spikes form different theta phases
        # train model on average theta
        X_train, X_test = X_mean[train_mask, :], X_mean[test_mask, :]
        if normalise_X:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
        # fit model
        model = LogisticRegression(C=alpha, random_state=0, max_iter=10_000, class_weight="balanced")
        model.fit(X_train, Y_train)
        _res = res.copy()
        Yprobs = model.predict_proba(X_test)
        weighted_dist_err = np.sum(Yprobs * dist_mat, axis=1)
        _res["error"] = weighted_dist_err
        _res["input_type"] = "spatial_spikes_mean"
        offset_results.append(_res)
        # test performance on spikes form different theta phases
        for label, Xtheta in zip(theta_phases.to_list(), X_thetas):
            _res = res.copy()
            X_test = Xtheta[test_mask, :]
            if normalise_X:
                X_test = scaler.transform(X_test)
            # fit model
            Yprobs = model.predict_proba(X_test)
            weighted_dist_err = np.sum(Yprobs * dist_mat, axis=1)
            _res["error"] = weighted_dist_err
            _res["input_type"] = "spatial_spikes_theta"
            _res["theta_phase"] = label
            offset_results.append(_res)
    return offset_results


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
