"""
is there theta modulation of the place-direction representation?
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from scipy.stats import ttest_1samp
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.multitest import multipletests

from GridMaze.analysis.place_direction import future_decoding as fd
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import folds
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import convert
from GridMaze.analysis.theta_mod import theta_utils as tmu

from GridMaze.maze import representations as mr
from GridMaze.maze import plotting as mp


# %% Global Variables
from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

RESULTS_DIR = RESULTS_PATH / "theta_mod" / "trajectory_decoding"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]

FRAME_RATE = 60

# %% Vis data


def plot_theta_mod_trajectory_error(
    summary_df,
    error="signed",
    normalise=True,
    all_traj_defined=True,
    steps_to_goal=None,
    decision_points=False,
    color="grey",
    print_stats=True,
    ax=None,
):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_ylabel("decoding bias (steps)")
    ax.set_xlabel("theta phase (rad)")
    ax.set_xticks(np.arange(-np.pi, np.pi + 0.1, np.pi / 2))
    ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])
    # filter data
    df = summary_df.copy()
    if all_traj_defined:
        df = df[df.all_traj_defined]
    if steps_to_goal is not None:
        df = df[df.steps_to_goal.between(*steps_to_goal)]
    if decision_points:
        df = filter_decision_points(df, decision_points="future")
    # average data for each subject
    subject_means = df.groupby(["subject_ID", "theta_phase"])[f"{error}_error"].mean().unstack(0)
    if normalise:
        subject_means = subject_means.sub(subject_means.mean(), axis=1)
    grand_mean = subject_means.mean(1)
    grand_sem = subject_means.sem(1)
    # plot
    ax.errorbar(
        grand_mean.index.values,
        grand_mean.values,
        yerr=grand_sem.values,
        fmt="o-",
        color=color,
        markersize=6,
        linewidth=2,
        capsize=None,
        elinewidth=2,
    )
    # stats
    if print_stats:
        tmu.test_theta_modulation(subject_means.T)


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


# %%


def get_summary_df(verbose=True, save=False, control=False):

    def _process_session(session):
        if verbose:
            print(session.name)
        results_df = get_session_theta_mod_trajectory_error(
            session,
            verbose=False,
            use_control_theta_spike_counts=control,
        )
        results_df["subject_ID"] = session.subject_ID
        results_df["maze_name"] = session.maze_name
        results_df["day_on_maze"] = session.day_on_maze
        return results_df

    if control:
        save_path = RESULTS_DIR / "decoding_summary_control_df.parquet"
    else:
        save_path = RESULTS_DIR / "decoding_summary_df2.parquet"
    if save_path.exists() and not save:
        summary_df = pd.read_parquet(save_path)
        return summary_df
    dfs = []
    for subject in SUBJECT_IDS:
        for maze_name in MAZE_NAMES:
            sessions = gs.get_maze_sessions(
                subject_IDs=[subject],
                maze_names=[maze_name],
                days_on_maze="late",
                with_data=["navigation_df", "navigation_theta_spike_counts_df", "cluster_metrics", "trials_df"],
                must_have_data=True,
            )
            _dfs = Parallel(n_jobs=-1)(delayed(_process_session)(session) for session in sessions)
            for df in _dfs:
                dfs.append(df)
    summary_df = pd.concat(dfs).reset_index(drop=True)
    # check for duplicate columns
    summary_df = summary_df.loc[:, ~summary_df.columns.duplicated()].copy()
    if save:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_parquet(save_path)
    return summary_df


def get_session_theta_mod_trajectory_error(
    session,
    include_multi_units=True,
    max_steps_to_goal=30,
    sum_spike_window=0.4,
    resolution=0.1,
    envelope=2,
    sqrt_spikes=True,
    alpha="opt",
    normalise_X=True,
    n_folds=8,
    verbose=False,
    use_control_theta_spike_counts=False,
):
    """ """
    # input data
    if verbose:
        print("Loading input data...")
    input_data = get_input_data(
        session,
        include_multi_units=include_multi_units,
        max_steps_to_goal=max_steps_to_goal,
        sum_spike_window=sum_spike_window,
        resolution=resolution,
        modes=["past", "future"],
        offset=envelope,
        state_type="place",
    )
    if use_control_theta_spike_counts:
        input_data = get_theta_spike_control_input_data(input_data)
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
        if alpha == "opt":
            _alpha = _get_opt_alpha(fold_df, train_df, normalise_X=normalise_X, sqrt_spikes=sqrt_spikes)
        else:
            _alpha = alpha
        decoder = LogisticRegression(
            C=_alpha, random_state=0, max_iter=10_000, class_weight="balanced", verbose=verbose
        )
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
            signed_errors, abs_errors, true_prob_mass, all_traj_defined = get_weighted_trajectory_position_error(
                Yprob, Y_test, test_df, decoder_classes, all_pairs_path_length
            )
            _res = res.copy()
            _res["theta_phase"] = phase
            _res["fold"] = fold
            _res["signed_error"] = signed_errors
            _res["abs_error"] = abs_errors
            _res["true_prob_mass"] = true_prob_mass
            _res["all_traj_defined"] = all_traj_defined
            results.append(_res)
    results_df = pd.concat(results).reset_index(drop=True)
    return results_df


def get_weighted_trajectory_position_error(
    Yprob,
    Y_test,
    test_df,
    decoder_classes,
    all_pairs_path_length,
    verbose=False,
):
    """ """
    samples = Yprob.shape[0]
    traj_envelope = test_df[["past", "future"]]  # past & future parts of trajectory
    signed_errors = np.zeros(samples)
    abs_errors = np.zeros(samples)
    true_prob_mass = np.zeros(samples)
    all_traj_defined = np.ones(samples, dtype=bool)
    for i in range(samples):
        y = Y_test[i]
        probs = Yprob[i]
        loc2prob = dict(zip(decoder_classes, probs))
        traj = traj_envelope.iloc[i]
        # check trajectory in decoder classes
        not_in_train = np.setdiff1d(traj.unique(), decoder_classes)
        if len(not_in_train) > 0:
            if verbose:
                print(not_in_train)
            all_traj_defined[i] = False
        past_locs = traj.loc["past"].iloc[1:].dropna().unique()
        future_locs = traj.loc["future"].iloc[1:].dropna().unique()
        if y not in decoder_classes:
            true_pmass = np.nan
        else:
            true_pmass = loc2prob[y]
        true_prob_mass[i] = true_pmass
        # filter probs to only include those on the trajectory
        past_probs = [loc2prob[loc] if loc in loc2prob.keys() else np.nan for loc in past_locs]
        past_dists = np.array([all_pairs_path_length[y][loc] for loc in past_locs])
        future_probs = [loc2prob[loc] if loc in loc2prob.keys() else np.nan for loc in future_locs]
        future_dists = np.array([all_pairs_path_length[y][loc] for loc in future_locs])
        # calculate decoding error distance over the trajectory (+ve = more future, -ve = more past)
        weighted_past, weighted_future = np.nansum(future_probs * future_dists), np.nansum(past_probs * past_dists)
        signed_errors[i] = weighted_future - weighted_past
        abs_errors[i] = weighted_future + weighted_past
    return (
        signed_errors,
        abs_errors,
        true_prob_mass,
        all_traj_defined,
    )


def _get_opt_alpha(
    fold_df, train_df, normalise_X=True, sqrt_spikes=True, reg_range=np.logspace(-4, 4, 10), verbose=False
):
    vfolds_df = fold_df.train
    vfolds = vfolds_df.columns
    results = np.zeros((len(vfolds), len(reg_range)))
    for i, vfold in enumerate(vfolds):
        if verbose:
            print(f"vfold: {i}")
        val_trials = vfolds_df[vfold].dropna().values
        train_trials = vfolds_df[[t for t in vfolds if t != vfold]].unstack().dropna().values
        _train_df = train_df[train_df.trial.isin(train_trials)]
        _val_df = train_df[train_df.trial.isin(val_trials)]
        X_train, X_val = [df.spike_count.T.groupby(level=0).mean().T.values for df in [_train_df, _val_df]]
        if X_train.shape[0] == 0 or X_val.shape[0] == 0:
            continue
        Y_train, Y_val = [df.maze_position.simple.values for df in [_train_df, _val_df]]
        # standardise
        if sqrt_spikes:
            X_train, X_val = np.sqrt(X_train), np.sqrt(X_val)
        if normalise_X:
            scaler = StandardScaler().fit(X_train)
            X_train, X_val = scaler.transform(X_train), scaler.transform(X_val)
        # fit model
        for j, alpha in enumerate(reg_range):
            decoder = LogisticRegression(C=alpha, random_state=0, max_iter=10_000, class_weight="balanced")
            decoder.fit(X_train, Y_train)
            score = decoder.score(X_val, Y_val)
            results[i, j] = score
    opt_alpha = reg_range[np.nanmean(results, axis=0).argmax()]
    return opt_alpha


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
    resolution=None,
    sum_spike_window=0.4,
    modes=["future", "past"],
    offset=12,
    state_type="place",
):
    """ """
    # load data
    navigation_df = session.navigation_df.copy()
    spike_counts_df = session.navigation_theta_spike_counts_df.reset_index(drop=True).copy()

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

    # sum spikes over spike_window
    sum_frames = int(sum_spike_window * FRAME_RATE)
    spike_counts_df = spike_counts_df.rolling(window=sum_frames, center=True).sum().fillna(0).astype(int)

    navigation_df.columns = pd.MultiIndex.from_tuples([(*col, "") for col in navigation_df.columns])
    navigation_spikes_df = pd.concat([navigation_df, spike_counts_df], axis=1)

    # downsample data
    if resolution is not None:
        every_n_frames = int(resolution * FRAME_RATE)
        navigation_spikes_df = navigation_spikes_df.iloc[::every_n_frames].reset_index(drop=True)

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


# %%


def get_theta_spike_control_input_data(input_data):
    """ """
    # split df back into navigation and spike counts
    navigation_df = input_data.drop(columns="spike_count", level=0, axis=1).copy()
    theta_spike_counts = input_data.spike_count.copy()
    phases = theta_spike_counts.columns.get_level_values(1).unique()
    # get each clusters theta modulation profile (norm.)
    theta_mod = theta_spike_counts.mean().unstack()
    norm_theta_mod = theta_mod.sub(theta_mod.mean(axis=1), axis=0) + 1
    # get mean spike counts across theta phases
    theta_avg = theta_spike_counts.T.groupby(level=0).mean().T
    # check cluster unique ids
    assert all(norm_theta_mod.index == theta_avg.columns), "cluster_ID mismatch"
    # build control "spike_counts" that just reflect average theta modualtion
    dfs = []
    for phase in phases:
        control_phase_counts = theta_avg.mul(norm_theta_mod[phase], axis=1)
        control_phase_counts.columns = pd.MultiIndex.from_product(
            [["spike_count"], [phase], control_phase_counts.columns]
        )
        dfs.append(control_phase_counts)
    control_theta_spike_counts = pd.concat(dfs, axis=1)
    control_theta_spike_counts = control_theta_spike_counts.swaplevel(1, 2, axis=1).sort_index(axis=1)
    # combine with navigation df and return
    return pd.concat([navigation_df, control_theta_spike_counts], axis=1)
