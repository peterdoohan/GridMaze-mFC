"""
Test distance to goal decoding aligned to event times (eg, cue.)
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns

from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.maze import representations as mr
from GridMaze.analysis.core import get_clusters as gc

from sklearn.neural_network import MLPRegressor
from scipy.stats import linregress

# %% Globs
from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]

with open(EXPERIMENT_INFO_PATH / "maze_day2date.json", "r") as input_file:
    MAZE_DAY2DATE = json.load(input_file)

RESULTS_PATH = RESULTS_PATH / "distance_decoding"

FRAME_RATE = 60

# %% plot short distance excluded trials comparisons


def test(subject_IDs=["m2"], maze_names=["maze_1"], last_n_days_on_maze=5, window=(-1, 2), verbose=False):
    """ """
    # load data
    all_trials_results, reduced_trials_results = [], []
    for subject in subject_IDs:
        for maze in maze_names:
            days = [int(d) for d in MAZE_DAY2DATE[maze].keys()]
            if last_n_days_on_maze:
                days = days[-last_n_days_on_maze:]
            for day in days:
                all_trials_dir = RESULTS_PATH / "decoding_error" / f"{subject}.{maze}.{day}.cue"
                reduced_trials_dir = RESULTS_PATH / "decoding_error2" / f"{subject}.{maze}.{day}.cue"
                try:
                    all_trials_df = pd.read_csv(all_trials_dir / "decoding_results.csv", index_col=0)
                    reduced_trials_df = pd.read_csv(reduced_trials_dir / "decoding_results.csv", index_col=0)
                except FileNotFoundError:
                    if verbose:
                        print(f"No results for {all_trials_dir.name}")
                    continue
                # filter timepoints
                all_trials_df = all_trials_df[all_trials_df.timepoint.between(window[0], window[1])]
                reduced_trials_df = reduced_trials_df[reduced_trials_df.timepoint.between(window[0], window[1])]
                # get error
                all_trials_test = all_trials_df.groupby(["timepoint", "shuffle"]).test.mean().unstack()
                reduced_trials_test = reduced_trials_df.groupby(["timepoint", "shuffle"]).test.mean().unstack()
                all_trials_error = all_trials_test[False] - all_trials_test[True]
                reduced_trials_error = reduced_trials_test[False] - reduced_trials_test[True]
                all_trials_results.append(all_trials_error.values)
                reduced_trials_results.append(reduced_trials_error.values)
    # plot
    all_trials_results = np.array(all_trials_results)
    reduced_trials_results = np.array(reduced_trials_results)
    f, ax = plt.subplots(1, 1, figsize=(5, 3), clear=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel(f"cue-aligned time (s)")
    ax.set_ylabel("norm error")
    ax.set_title("Geodesic Distance-to-goal Decoding")
    timepoints = all_trials_error.index.values
    all_trials_mean = all_trials_results.mean(axis=0)
    reduced_trials_mean = reduced_trials_results.mean(axis=0)
    all_trials_sem = all_trials_results.std(axis=0) / np.sqrt(all_trials_results.shape[0])
    reduced_trials_sem = reduced_trials_results.std(axis=0) / np.sqrt(reduced_trials_results.shape[0])
    ax.plot(timepoints, all_trials_mean, label="all trials", color="green")
    ax.fill_between(
        timepoints, all_trials_mean - all_trials_sem, all_trials_mean + all_trials_sem, alpha=0.1, color="green"
    )
    ax.plot(timepoints, reduced_trials_mean, label="excl. short dist. trials", color="grey")
    ax.fill_between(
        timepoints,
        reduced_trials_mean - reduced_trials_sem,
        reduced_trials_mean + reduced_trials_sem,
        alpha=0.1,
        color="grey",
    )
    ax.legend(fontsize="small")
    return all_trials_results, reduced_trials_results


# %% Run different version of the analysis to see decoded vs real distance around cue time


def plot_real_vs_decoded_distance(verbose=False, window=False):
    """ """
    # load all data
    results_dfs = []
    for subject in SUBJECT_IDS:
        for maze in MAZE_NAMES:
            for day in MAZE_DAY2DATE[maze].keys():
                results_dir = RESULTS_PATH / "real_vs_decoded" / f"{subject}.{maze}.{day}.cue"
                try:
                    results_df = pd.read_csv(results_dir / "decoded_vs_real_distance.csv", index_col=0)
                    results_dfs.append(results_df)
                except FileNotFoundError:
                    if verbose:
                        print(f"No results for {results_dir}")
                    continue
    results_df = pd.concat(results_dfs, axis=0).reset_index(drop=True)
    if window:
        results_df = results_df[results_df.timepoints.between(window[0], window[1])]
    # plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3), clear=True)
    sns.histplot(
        results_df, x="real_distance", y="decoded_distance", bins=50, cbar=True, cbar_kws=dict(shrink=0.75), ax=ax
    )
    sns.regplot(results_df, x="real_distance", y="decoded_distance", ax=ax, color="red", scatter=False)
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 2.5)
    ax.plot([0, 2.5], [0, 2.5], color="k", linestyle="--", alpha=0.5)
    # print some stats
    slope, intercept, r_value, p_value, std_err = linregress(
        results_df.real_distance.values, results_df.decoded_distance.values
    )
    print("Slope:", slope)
    print("P-value for slope:", p_value)
    # return results_df


def decoded_vs_real_distance(subject_ID, maze_name, day_on_maze, event, window, resolution, alpha):
    # load session
    try:
        session = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names=[maze_name],
            days_on_maze=[int(day_on_maze)],
            with_data=["navigation_df", "cluster_metrics", "navigation_spike_counts_df", "trials_df"],
            must_have_data=True,
        )
    except FileExistsError:
        print(f"Session {subject_ID}.{maze_name}.{day_on_maze} does not have requisit data")
        return
    # run_analysis
    save_dir = RESULTS_PATH / "real_vs_decoded" / f"{subject_ID}.{maze_name}.{day_on_maze}.{event}"
    save_dir.mkdir(exist_ok=True, parents=True)
    results_df = _decoded_vs_real_distance(session, event, window, resolution, alpha, plot=False, save_dir=save_dir)
    return results_df


def _decoded_vs_real_distance(session, event="cue", window=(0, 1), resolution=0.2, alpha=10, plot=True, save_dir=False):
    """ """
    (
        decoding_distances,
        spike_counts,
        timepoints,
    ) = get_input_data(session, event, window, resolution=resolution)
    # run MLP distance decoder
    results_df = run_mlp_distance_decoder_ALT(decoding_distances, spike_counts, timepoints, alpha)
    # plot
    if plot:
        f, ax = plt.subplots(1, 1, figsize=(3, 3), clear=True)
        sns.histplot(
            results_df, x="real_distance", y="decoded_distance", bins=50, cbar=True, cbar_kws=dict(shrink=0.75), ax=ax
        )
        ax.set_xlim(0, 2.5)
        ax.set_ylim(0, 2.5)
        ax.plot([0, 2.5], [0, 2.5], color="k", linestyle="--")
    if save_dir:
        results_df.to_csv(save_dir / "decoded_vs_real_distance.csv")
    return results_df


def run_mlp_distance_decoder_ALT(decoding_distances, spike_counts, timepoints, alpha=10):
    """ """
    n_clusters = spike_counts.shape[-1]
    trial_ind = np.arange(decoding_distances.shape[0])
    results = []
    for trial in trial_ind:
        test_trial = [trial]
        train_trials = np.setdiff1d(trial_ind, test_trial)
        decoded_distances = np.zeros(len(timepoints))
        real_distances = np.zeros(len(timepoints))
        for t in range(len(timepoints)):
            X_train = spike_counts[train_trials, t, :]  # [n_train_trials, n_clusters]
            X_test = spike_counts[test_trial, t, :]
            y_train = decoding_distances[train_trials, t]
            y_test = decoding_distances[test_trial, t]
            mlp = MLPRegressor(
                hidden_layer_sizes=(n_clusters, n_clusters),  # 2 hidden layers with n_clusters units
                max_iter=10_000,  # brief testing suggests this is ok with enough reg ^^
                alpha=alpha,
                verbose=False,
            )
            mlp.fit(X_train, y_train)
            # predict
            y_pred = mlp.predict(X_test)
            decoded_distances[t] = y_pred
            real_distances[t] = y_test
        results.append(
            pd.DataFrame(
                {
                    "trial": trial + 1,
                    "timepoints": timepoints,
                    "real_distance": real_distances,
                    "decoded_distance": decoded_distances,
                }
            )
        )
    return pd.concat(results, axis=0)


# %% Load results


def plot_random_effects_result(event="cue", maze_names=["maze_1", "maze_2"], last_n_days_on_maze=5, verbose=False):
    """ """
    # load data
    subject_decoding_errors = []
    for subject in SUBJECT_IDS:
        decoding_errors = []  # mean norm decoding error per session
        for maze in maze_names:
            days = [int(d) for d in MAZE_DAY2DATE[maze].keys()][-last_n_days_on_maze:]
            for day in days:
                results_dir = RESULTS_PATH / "decoding_error" / f"{subject}.{maze}.{day}.{event}"
                try:
                    results_df = pd.read_csv(results_dir / "decoding_results.csv", index_col=0)
                except FileNotFoundError:
                    if verbose:
                        print(f"No results for {results_dir}")
                    continue
                # process results to get normalised decoding error (on held_out test data) per session
                test_df = results_df.groupby(["timepoint", "shuffle"]).test.mean().unstack()
                norm_error = test_df[False] - test_df[True]  # chance normalised error
                decoding_errors.append(norm_error.values)
        subject_decoding_errors.append(np.array(decoding_errors))
    # plot
    f, ax = plt.subplots(1, 1, figsize=(5, 3), clear=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel(f"{event} aligned time (s)")
    ax.set_ylabel("norm error")
    ax.set_title("Geodesic Distance-to-goal Decoding")
    timepoints = norm_error.index.values
    subject_means = np.array([x.mean(axis=0) for x in subject_decoding_errors])  # n_subjects, n_timepoints
    mean = subject_means.mean(axis=0)
    sem = subject_means.std(axis=0) / np.sqrt(subject_means.shape[0])
    ax.plot(timepoints, mean, color="green", lw=2)
    ax.fill_between(timepoints, mean - sem, mean + sem, alpha=0.1, color="green")
    return subject_decoding_errors


# %%
def run_analysis():
    return print("See jobs/distance_decoding/submit.py")


# %%
def decoding_distance_to_goal(
    subject_ID,
    maze_name,
    day_on_maze,
    event="cue",
    window=(-1, 1),
    resolution=0.1,
    n_folds=8,
    n_perm=1,
    alpha=10,
    normalise_spikes=False,
    exclude_short_distance_trials=False,
):
    # load session
    try:
        session = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names=[maze_name],
            days_on_maze=[int(day_on_maze)],
            with_data=["navigation_df", "cluster_metrics", "navigation_spike_counts_df", "trials_df"],
            must_have_data=True,
        )
    except FileExistsError:
        print(f"Session {subject_ID}.{maze_name}.{day_on_maze} does not have requisit data")
        return
        # set save dir with standardised naming
    save_dir = RESULTS_PATH / "decoding_error2" f"{subject_ID}.{maze_name}.{day_on_maze}.{event}"
    save_dir.mkdir(exist_ok=True, parents=True)
    # run analysis
    _decode_session_distance_to_goal(
        session,
        event,
        window,
        resolution,
        n_folds,
        n_perm,
        alpha,
        save_dir,
        normalise_spikes,
        exclude_short_distance_trials,
    )
    return


def _decode_session_distance_to_goal(
    session,
    event="cue",
    window=(-2, 2),
    resolution=0.1,
    n_folds=8,
    n_perm=1,
    alpha=10,
    save_dir=False,
    normalise_spikes=False,
    exclude_short_distance_trials=False,
):
    """ """
    trials = session.trials_df.trial.unique()
    # split trials into folds for xval
    validation_folds_df = filt.get_trial_validation_folds_df(trials, splits=n_folds)
    (
        decoding_distances,
        spike_counts,
        timepoints,
    ) = get_input_data(session, event, window, resolution=resolution)

    if exclude_short_distance_trials:
        # exclude trials where distance to goal is less than 0.5
        short_distance_mask = decoding_distances.min(axis=1) < 0.5
        remaining_trials = np.argwhere(~short_distance_mask).flatten() + 1
        decoding_distances = decoding_distances[~short_distance_mask]
        spike_counts = spike_counts[~short_distance_mask]
        # overwrite validation folds
        validation_folds_df = filt.get_trial_validation_folds_df(remaining_trials, splits=n_folds)
        # map trials to new indices
        validation_folds_df = validation_folds_df.replace({t: i + 1 for i, t in enumerate(remaining_trials)})
        trials = remaining_trials

    if normalise_spikes:
        # sqrt transform to make data more normal
        spike_counts = np.sqrt(spike_counts)
        # # normalise to apply regularisation equally across neurons
        spike_counts = (spike_counts - np.mean(spike_counts, axis=(0, 1), keepdims=True)) / np.std(
            spike_counts, axis=(0, 1), keepdims=True
        )
    # run MLP distance decoder
    print("running MLP distance decoder on real data ...")
    results_df = run_mlp_distance_decoder(
        decoding_distances, spike_counts, timepoints, validation_folds_df, alpha=alpha
    )
    results_df["shuffle"] = False
    results_df["shuffle_perm"] = np.nan
    shuffled_results = []
    # find null distribution of decoded errors for n_perm * n_folds itterations
    # enough to estimate mean and variance of null distribution
    print("running MLP distance decoder on shuffled data ...")
    for i in range(n_perm):
        np.random.shuffle(decoding_distances)  # shuffle across trials
        print(f"shuffle {i}")
        shuffle_df = run_mlp_distance_decoder(
            decoding_distances, spike_counts, timepoints, validation_folds_df, alpha=alpha
        )
        shuffle_df["shuffle"] = True
        shuffle_df["shuffle_perm"] = i
        shuffled_results.append(shuffle_df)
    shuffled_results = pd.concat(shuffled_results, axis=0)
    combined_df = pd.concat([results_df, shuffled_results], axis=0)
    # plot
    plot_results(combined_df, event, save_dir=save_dir)
    plot_test_train(combined_df, event, save_dir=save_dir)
    if save_dir:
        combined_df["subject"] = session.subject_ID
        combined_df["maze"] = session.maze_name
        combined_df["day_on_maze"] = session.day_on_maze
        combined_df.to_csv(save_dir / "decoding_results.csv")
    return combined_df


def plot_results(results_df, event, save_dir=False):
    """ """
    results = []
    for fold in results_df.fold.unique():
        fold_results = results_df[results_df.fold == fold]
        true_vs_shuffle = fold_results.groupby(["timepoint", "shuffle"]).test.mean().unstack()
        chance_norm_error = true_vs_shuffle[False] - true_vs_shuffle[True]
        results.append(chance_norm_error.values)
    timepoints = true_vs_shuffle.index.values
    results = np.array(results)
    mean = results.mean(axis=0)
    sem = results.std(axis=0) / np.sqrt(results.shape[0])
    f, ax = plt.subplots()
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot(timepoints, mean, label="mean")
    ax.fill_between(timepoints, mean - sem, mean + sem, alpha=0.1)
    ax.set_xlabel(f"{event} aligned time (s)")
    ax.set_ylabel("norm distance error")
    ax.axhline(0, color="k", linestyle="--")
    ax.axvline(0, color="k", linestyle="--")
    if save_dir:
        plt.savefig(save_dir / f"norm_decoding_error.pdf")
    return


def plot_test_train(results_df, event, save_dir=False):
    """ """
    shuffle_grouped = results_df[results_df.shuffle].groupby("timepoint")
    shuffle_test = shuffle_grouped.test.mean()
    shuffle_train = shuffle_grouped.train.mean()
    true_grouped = results_df[~results_df.shuffle].groupby("timepoint")
    true_test = true_grouped.test.mean()
    true_train = true_grouped.train.mean()
    timepoints = true_test.index
    f, ax = plt.subplots()
    ax.plot(timepoints, shuffle_test, label="shuffle_test", ls="--", color="blue")
    ax.plot(timepoints, shuffle_train, label="shuffle_train", ls="--", color="grey")
    ax.plot(timepoints, true_test, label="true_test", color="blue")
    ax.plot(timepoints, true_train, label="true_train", color="grey")
    ax.set_xlabel(f"{event} aligned time (s)")
    ax.set_ylabel("distance error")
    ax.axvline(0, color="k", linestyle="--")
    ax.legend(fontsize="small")
    if save_dir:
        plt.savefig(save_dir / f"test-train.pdf")
    return


# %%
def run_mlp_distance_decoder(
    decoding_distances,
    spike_counts,
    timepoints,
    validation_folds_df,
    alpha=1,
):
    """"""
    results = []
    n_clusters = spike_counts.shape[-1]
    for fold in validation_folds_df.columns.get_level_values(0).unique():
        fold_df = validation_folds_df[fold]
        test_trials = fold_df.test.dropna().values.astype(int)
        train_trials = fold_df.train.dropna().values.astype(int)
        # fit MLP model
        test_err, train_err = np.zeros(len(timepoints)), np.zeros(len(timepoints))
        for t in range(len(timepoints)):
            # split data
            X_train = spike_counts[train_trials - 1, t, :]  # [n_train_trials, n_clusters]
            X_test = spike_counts[test_trials - 1, t, :]
            y_train = decoding_distances[train_trials - 1, t]  # [n_train_trials]
            y_test = decoding_distances[test_trials - 1, t]
            mlp = MLPRegressor(
                hidden_layer_sizes=(n_clusters, n_clusters),  # 2 hidden layers with n_clusters units
                max_iter=10_000,  # brief testing suggests this is ok with enough reg ^^
                alpha=alpha,
                verbose=False,
            )
            mlp.fit(X_train, y_train)
            # predict
            y_pred = mlp.predict(X_test)
            dist_error = np.abs(y_pred - y_test)
            test_err[t] = dist_error.mean()
            # get training error
            y_train_pred = mlp.predict(X_train)
            train_dist_error = np.abs(y_train_pred - y_train)
            train_err[t] = train_dist_error.mean()
        results.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "test": test_err,
                    "train": train_err,
                    "timepoint": timepoints,
                }
            )
        )
    return pd.concat(results, axis=0)


# %%
#
def get_optimal_alpha(
    session, event="reward", window=(-4, 0), alphas=[1e-1, 1, 5, 1e1, 5e1, 1e2], plot=True, save_dir=False
):
    """ """
    trials = session.trials_df.trial.unique()
    validation_folds_df = filt.get_trial_validation_folds_df(trials, splits=4)
    (
        decoding_distances,
        spike_counts,
        timepoints,
    ) = get_input_data(session, event, window, resolution=0.2)
    # sqrt transform to make data more normal
    spike_counts = np.sqrt(spike_counts)
    # normalise
    spike_counts = (spike_counts - np.mean(spike_counts, axis=(0, 1), keepdims=True)) / np.std(
        spike_counts, axis=(0, 1), keepdims=True
    )
    # run MLP distance decoder
    test_perf, train_perf = [], []
    for alpha in alphas:
        results_df = run_mlp_distance_decoder(
            decoding_distances, spike_counts, timepoints, validation_folds_df, alpha=alpha
        )
        test_perf.append(results_df.test.mean())  # mean across timepoints
        train_perf.append(results_df.train.mean())
    opt_alpha = alphas[np.argmin(test_perf)]
    # plot results
    if plot:
        plt.plot(alphas, test_perf, label="test")
        plt.plot(alphas, train_perf, label="train")
        plt.xlabel("alpha")
        plt.xscale("log")
        plt.ylabel(f"mean distance error ({event})")
        plt.legend()
        plt.scatter(opt_alpha, test_perf[np.argmin(test_perf)], color="r")
    if plot and save_dir:
        plt.savefig(save_dir / "optimal_alpha.pdf")
    return opt_alpha


# %%


def get_input_data(session, event, window=(-10, 10), resolution=0.1):
    """ """
    # load data
    navigation_df = get_updated_navigation_df(
        session, log_distance=False
    )  # [frames, navigation varaibales incl. decoding distance]
    spike_counts_df = session.navigation_spike_counts_df.reset_index(drop=True).droplevel(
        0, axis=1
    )  # [frames, clusters]
    # filter for sinlge units
    keep_clusters = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=False,
    )
    spike_counts_df = spike_counts_df[keep_clusters]
    # get distances to goal and spike counts aligned to event times
    times = navigation_df.time
    trial_df = session.trials_df
    event_times = trial_df.time[event].values
    decoding_distances = []
    spike_counts = []
    for t in event_times:
        event_idx = times.sub(t).abs().argmin()
        select_frames = np.arange(event_idx + window[0] * FRAME_RATE, event_idx + window[1] * FRAME_RATE)
        nav_df = navigation_df.loc[select_frames]
        decoding_distances.append(nav_df[("distance_to_goal", "for_decoding")].values)
        spikes_df = spike_counts_df.loc[select_frames]
        spike_counts.append(spikes_df.values)  # [n_frames, n_clusters]
    decoding_distances = np.vstack(decoding_distances)  # [n_trials, window_size]
    spike_counts = np.stack(spike_counts)  # [n_trials, window_size, n_clusters]
    if resolution is not None:
        res_frames = int(resolution * FRAME_RATE)
        # average distance values over frames to get to specified resolution
        n_trials, n_timepoints = decoding_distances.shape
        decoding_distances = decoding_distances.reshape(n_trials, int(n_timepoints / res_frames), res_frames).mean(
            axis=2
        )
        # sum spike counts over frames to get to specified resolition
        n_clusters = spike_counts.shape[-1]
        spike_counts = spike_counts.reshape(n_trials, int(n_timepoints / res_frames), res_frames, n_clusters).sum(
            axis=2
        )
    # timepoints array
    timepoints = np.arange(window[0], window[1], resolution)
    return decoding_distances, spike_counts, timepoints


# %% Update distance to goal definition


def get_updated_navigation_df(session, log_distance=False):
    """
    Distance to goal not defined in reward_consumption and ITI periods. Update so that distance
    in these periods are defined for the next trials's goal.
    """
    skeleton_maze = session.skeleton_maze()
    coord2label = mr.get_skeleton_maze_node_labels_dict()
    label2coord = {v: k for k, v in coord2label.items()}
    all_sk_path_distances = dict(nx.all_pairs_dijkstra_path_length(skeleton_maze))
    navigation_df = session.navigation_df
    trial2next_goal = get_trial2next_goal(session)
    navigation_df[("next_goal", "")] = navigation_df.trial.map(trial2next_goal).bfill().ffill()
    navigation_df
    distances = navigation_df.apply(
        _update_distance, axis=1, all_sk_path_distances=all_sk_path_distances, label2coord=label2coord
    )
    if log_distance:
        distances = np.log(distances + 1e-3)
    navigation_df[("distance_to_goal", "for_decoding")] = distances
    return navigation_df


def _update_distance(row, all_sk_path_distances, label2coord):
    """ """
    current_coord = label2coord[row[("maze_position", "skeleton")]]
    trial_phase = row[("trial_phase", "")]
    if not isinstance(trial_phase, str):  # start or end of session
        next_goal = row[("next_goal", "")]
        next_goal_coord = label2coord[next_goal + "_C"]
        return all_sk_path_distances[current_coord][next_goal_coord]
    if trial_phase == "navigation":
        current_goal = row[("goal", "")]
        current_goal_coord = label2coord[current_goal + "_C"]
        return all_sk_path_distances[current_coord][current_goal_coord]
    else:
        next_goal = row[("next_goal", "")]
        next_goal_coord = label2coord[next_goal + "_C"]
        return all_sk_path_distances[current_coord][next_goal_coord]


def get_trial2next_goal(session):
    """
    returns a dictionary mapping trial number to the goal on the next trial
    """
    trials_df = session.trials_df
    next_goal = trials_df.goal.shift(-1)
    # replace last None value with random goal
    next_goal.iloc[-1] = np.random.choice(session.goals)
    trials_df[("next_goal", "")] = next_goal
    return trials_df.set_index("trial").next_goal.to_dict()
