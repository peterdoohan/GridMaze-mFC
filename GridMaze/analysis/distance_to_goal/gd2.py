"""
Library for distance-to-goal alaigned goal decoding.
Eg, build separate decoders for neural activity 1, step from goal, 2 steps from goal, etc.
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
import test


from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import convert
from GridMaze.maze import representations as mr

from . import bases as db
from . import decoding_utils as dutils


# %% Global Variables

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "distance_to_goal"


with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

FRAME_RATE = 60

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]
GOAL_SETS = ["subset_1", "subset_2", "all"]


# %% dev


# %% results plotting functions


def plot_distance_aligned_results(results_df, ax=None, color="rosybrown", sig_color="slategrey", ymax=0.45):
    """ """
    # set up plot
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(4, 3), clear=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Steps to goal")
    ax.set_ylabel("Decoding Acc. \n (chance subtracted)")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax.set_ylim(-0.02, ymax)
    # average chance subtracted decoding acc over steps_to_goal across subjects
    df = results_df.groupby(["steps_to_goal", "subject_ID"]).norm_acc.mean().unstack().T
    steps = df.columns.values
    mean = df.mean(axis=0)
    sem = df.sem(axis=0)
    # plot
    ax.plot(steps, mean, color=color, lw=2)
    ax.fill_between(steps, mean - sem, mean + sem, color=color, alpha=0.2)
    ax.set_xlim(0, steps.max())
    # run stats
    _plot_p_values(ax, df, ymax, sig_color)


def plot_event_aligned_results(
    results_df, event, ax=None, chance=1 / 12, color="darkorange", sig_color="sandybrown", ymax=0.55
):
    """ """
    # set up plot
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(4, 3), clear=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(f"{event} (s)")
    ax.set_ylabel("Decoding Acc. \n (chance subtracted)")
    ax.axhline(y=chance, color="k", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="k", linestyle="--", alpha=0.5)
    ax.set_ylim(0, ymax)
    # average chance subtracted decoding acc over steps_to_goal across subjects
    df = results_df.groupby(["timepoint", "subject_ID"]).test_acc.mean().unstack().T
    timepoints = df.columns.values
    mean = df.mean(axis=0)
    sem = df.sem(axis=0)
    # plot
    ax.plot(timepoints, mean, color=color, lw=2)
    ax.fill_between(timepoints, mean - sem, mean + sem, color=color, alpha=0.2)
    ax.set_xlim(timepoints.min(), timepoints.max())
    ax.set_xticks([-5, 0, 5])
    _plot_p_values(ax, df, ymax, sig_color, chance=chance)
    return


def _plot_p_values(ax, df, height, color, chance=0):
    """"""
    p_values = []
    x = df.columns
    for i in x:
        t_stat, p_val = ttest_1samp(df[i], popmean=chance, alternative="greater")
        p_values.append(p_val)
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
    # indicate significant timepoints with line
    sig_idx = np.where(reject)[0]
    runs = np.split(sig_idx, np.where(np.diff(sig_idx) != 1)[0] + 1)
    for run in runs:
        if run.size > 0:
            x_run = x[run]
            y_run = np.full_like(x_run, height - 0.04, dtype=float)
            ax.plot(x_run, y_run, color=color, linewidth=2)


def plot_event_aligned_decoding_heatmap_summary(cue_results_df, reward_results_df, axes=None, cmap="Oranges", vmax=0.6):
    """
    Split decoding reusults by maze and goal subset to plot decoding acc summary of conditions
    in a heatmap.
    """
    if axes is None:
        f, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    (
        cue_grouped_df,
        reward_grouped_df,
    ) = [
        df.groupby(["goal_subset", "maze_name", "subject_ID", "timepoint"]).test_acc.mean().unstack()
        for df in [cue_results_df, reward_results_df]
    ]
    cue_mean_df, reward_mean_df = [
        df.groupby(["goal_subset", "maze_name"]).mean() for df in [cue_grouped_df, reward_grouped_df]
    ]
    # get complementary dfs that are True when value is sig above chance
    cue_sig_df = pd.DataFrame(index=cue_mean_df.index, columns=cue_mean_df.columns)
    reward_sig_df = pd.DataFrame(index=reward_mean_df.index, columns=reward_mean_df.columns)

    for df, sig_df in zip([cue_grouped_df, reward_grouped_df], [cue_sig_df, reward_sig_df]):
        times = df.columns
        for maze in cue_grouped_df.index.get_level_values(1).unique():
            for goal_subset in cue_grouped_df.index.get_level_values(0).unique():
                chance = (1 / 24) if goal_subset == "all" else (1 / 12)
                _df = df.loc[(goal_subset, maze)]
                # get p-values for these trials
                p_values = []
                for t in times:
                    t_stat, p_val = ttest_1samp(_df[t], popmean=chance, alternative="greater")
                    p_values.append(p_val)
                reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
                sig_df.loc[(goal_subset, maze), times] = reject

    # reorder goalset index
    for df in [cue_mean_df, reward_mean_df, cue_sig_df, reward_sig_df]:
        df.index = pd.MultiIndex.from_product(
            [["subset_1", "subset_2", "all"], df.index.levels[1]], names=["goal_subset", "maze_name"]
        )
    for ax, mean_df, sig_df, event in zip(
        axes, [cue_mean_df, reward_mean_df], [cue_sig_df, reward_sig_df], ["cue", "reward"]
    ):
        cbar = True if event == "reward" else False
        sns.heatmap(
            mean_df[sig_df],
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            ax=ax,
            rasterized=True,
            cbar=cbar,
            cbar_kws={"label": "Decoding Acc."},
        )
        times = mean_df.columns.values.astype(float)
        zero_point = np.argmin(np.abs(times))
        ax.axvline(zero_point, color="k", ls="--", alpha=0.5)
        tick_labels = [-5, 0, 5]
        tick_positions = [np.argmin(np.abs(times - tick)) for tick in tick_labels]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0)
        ax.set_xlabel(f"{event} time (s)")
        if event == "reward":
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
            spine.set_linewidth(0.5)

    return


def plot_distance_aligned_decoding_heatmap_summary(results_df, ax=None, cmap="Reds", vmax=0.5):
    """ """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(4, 4), clear=True)
    # average decoding acc for each subject then average over subjects
    subject_mean_df = (
        results_df.groupby(["goal_subset", "maze_name", "subject_ID", "steps_to_goal"]).norm_acc.mean().unstack()
    )
    mean_df = subject_mean_df.groupby(["goal_subset", "maze_name"]).mean(0)
    # calc sig from cross subject variance
    sig_df = pd.DataFrame(index=mean_df.index, columns=mean_df.columns)
    steps = mean_df.columns
    for maze in subject_mean_df.index.get_level_values(1).unique():
        for goal_subset in subject_mean_df.index.get_level_values(0).unique():
            _df = subject_mean_df.loc[(goal_subset, maze)]
            # get p-values for these trials
            p_values = []
            for t in steps:
                t_stat, p_val = ttest_1samp(_df[t], popmean=0, alternative="greater", nan_policy="omit")
                p_values.append(p_val)
            reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
            sig_df.loc[(goal_subset, maze), steps] = reject
    for df in [mean_df, sig_df]:
        df.index = pd.MultiIndex.from_product(
            [["subset_1", "subset_2", "all"], df.index.levels[1]], names=["goal_subset", "maze_name"]
        )
    # plot
    sns.heatmap(
        mean_df[sig_df],
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        ax=ax,
        rasterized=True,
        cbar=True,
        cbar_kws={"label": "Decoding Acc."},
    )
    tick_labels = steps[::4]
    tick_positions = [np.argmin(np.abs(steps - tick)) for tick in tick_labels]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_xlabel(f"Steps to goal")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(0.5)
    ax.set_ylabel("")


# %% Single reference frame exp average decoding


def get_aligned_decoding(reference, maze_names="all", goal_sets="all", verbose=True):
    """ """
    maze_names = MAZE_NAMES if maze_names == "all" else maze_names
    goal_sets = GOAL_SETS if goal_sets == "all" else goal_sets
    # run separately for all sessions for each subject
    results_dfs = []
    for subject_ID in SUBJECT_IDS:
        if verbose:
            print(f"Loading {subject_ID} data...")
        sessions = dutils.get_sessions_for_analysis([subject_ID], maze_names, goal_sets)
        for session in sessions:
            if verbose:
                print(f"Decoding: {session.name}")
            if reference == "distance":
                results_df = get_session_distance_aligned_decoding(session)
            elif reference in ["cue", "reward"]:
                results_df = get_session_event_aligned_decoding(session, event=reference)
            else:
                NotImplementedError
            results_df["subject_ID"] = subject_ID
            results_df["maze_name"] = session.maze_name
            results_df["goal_subset"] = session.goal_subset
            results_df["days_on_maze"] = session.day_on_maze
            results_dfs.append(results_df)
    return pd.concat(results_dfs, axis=0)


# %% Cross reference frame decoding


# %% distance aligned analyses


def get_sessions_distance_basis_decoding(
    session,
    resolution=0.5,
    n_bins=20,
    binning_method="uniform",
    max_steps_from_goal=20,
    n_bases=4,
    basis_type="gamma",
    goal_stratified_validation=True,
    n_test_trials=None,
    whiten_features=True,
):
    """ """
    # get input data
    input_data = dutils.get_distance_aligned_input_data(
        session,
        resolution,
        include_multi_units=False,
        include_trial_phases=["navigation"],
        max_steps_to_goal=max_steps_from_goal,
        n_bins=n_bins,
        binning_method=binning_method,
    )
    # load basis functions
    gamma_basis_shape_params = db.get_gamma_basis_shape_params(n_bases, btype="steps", max_steps=max_steps_from_goal)
    basis_fn = db.distance_basis_generator(
        gamma_basis_shape_params,
        basis=basis_type,
        btype="steps",
        normalise=True,
        max_steps=max_steps_from_goal,
        plot=True,
    )

    folds_df = dutils.get_folds_df(session, goal_stratified_validation, n_test_trials=n_test_trials)
    results = []
    for fold in folds_df.columns.levels[0].unique():
        fold_df = folds_df[fold]
        test_trials = [t for t in fold_df.test.values.flatten() if isinstance(t, str)]
        train_trials = [t for t in fold_df.train.values.flatten() if isinstance(t, str)]
        train_df = input_data[input_data.trial_unique_ID.isin(train_trials)]
        test_df = input_data[input_data.trial_unique_ID.isin(test_trials)]
        # get input as neurons x distance basis
        Xs = []
        for _df in [train_df, test_df]:
            basis_activations = basis_fn(_df.steps_to_goal.future)
            spikes = _df.spike_count.values
            A = spikes[:, :, None] * basis_activations[:, None, :]  # [n_timepoints, n_neurons, n_bases]
            Xs.append(A.reshape(A.shape[0], -1))  # [n_timepoints, n_neurons * n_bases]
        train_X, test_X = Xs
        if whiten_features:  # zscore features
            scaler = StandardScaler()  # mean=0, std=1 per column
            scaler.fit(train_X)  # learn stats on train
            train_X = scaler.transform(train_X)
            test_X = scaler.transform(test_X)
        train_y, test_y = train_df.goal.values, test_df.goal.values
        # fit single model
        decoder = LogisticRegression(max_iter=10000, penalty="l2", C=1, random_state=0, class_weight="balanced")
        decoder.fit(train_X, train_y)
        chance = 1 / len(decoder.classes_)
        # test decoder
        test_pred = decoder.predict(test_X)
        return test_y, test_X, decoder
        for y, yhat, trial, steps in zip(
            test_y, test_pred, test_df.trial_unique_ID.values, test_df.steps_to_goal.future.values
        ):
            results.append(
                {
                    "fold": fold,
                    "steps_to_goal": steps,
                    "trial": trial,
                    "goal": y,
                    "predicted_goal": yhat,
                    "test_acc": int(y == yhat),
                    "chance": chance,
                }
            )
    results_df = pd.DataFrame(results)
    results_df["norm_acc"] = results_df.test_acc - results_df.chance
    return results_df


def get_session_distance_aligned_decoding(
    session,
    inputs=["spikes"],
    trial_phases=["navigation"],
    resolution=0.5,
    binning_method="uniform",
    max_steps_from_goal=20,
    n_bins=20,
    goal_stratified_validation=True,
    n_test_trials=None,
    include_multi_units=False,
    whiten_features=True,
):
    """ """
    input_data = dutils.get_distance_aligned_input_data(
        session,
        resolution,
        include_multi_units,
        trial_phases,
        max_steps_to_goal=max_steps_from_goal,
        n_bins=n_bins,
        binning_method=binning_method,
    )
    bin_mids = sorted(input_data.steps_to_goal.bin_mid.dropna().unique())
    results_df = []
    for steps in bin_mids:
        steps_df = input_data[input_data.steps_to_goal.bin_mid == steps]
        valid_trials = steps_df.trial.unique()
        folds_df = dutils.get_folds_df(
            session, goal_stratified_validation, valid_trials, return_unique_IDs=True, n_test_trials=n_test_trials
        )
        if folds_df.shape[0] < 2:
            continue  # only one valid goal, cannot run classifer
        folds = folds_df.columns.levels[0].unique()
        for fold in folds:
            # get test and train data
            fold_df = folds_df[fold]
            test_trials = [t for t in fold_df.test.values.flatten() if isinstance(t, str)]
            train_trials = [t for t in fold_df.train.values.flatten() if isinstance(t, str)]
            test_df = steps_df[steps_df.trial_unique_ID.isin(test_trials)]
            train_df = steps_df[steps_df.trial_unique_ID.isin(train_trials)]
            train_y, test_y = train_df.goal.values, test_df.goal.values
            train_X, test_X = [], []
            if "spikes" in inputs:
                train_X.append(train_df.spike_count.values)
                test_X.append(test_df.spike_count.values)
            if "place" in inputs:
                train_X.append(train_df.place_onehot.values)
                test_X.append(test_df.place_onehot.values)
            train_X, test_X = np.concatenate(train_X, axis=1), np.concatenate(test_X, axis=1)
            if whiten_features:  # zscore features
                scaler = StandardScaler()  # mean=0, std=1 per column
                scaler.fit(train_X)  # learn stats on train
                train_X = scaler.transform(train_X)
                test_X = scaler.transform(test_X)
            # fit model
            decoder = LogisticRegression(max_iter=10000, penalty=None, random_state=0, class_weight="balanced")
            decoder.fit(train_X, train_y)
            goals = list(decoder.classes_)
            chance = 1 / len(goals)
            # test decoder
            test_pred = decoder.predict(test_X)
            for (
                y,
                yhat,
                trial,
            ) in zip(test_y, test_pred, test_trials):
                results_df.append(
                    {
                        "steps_to_goal": steps,
                        "fold": fold,
                        "trial": trial,
                        "goal": y,
                        "predicted_goal": yhat,
                        "test_acc": int(y == yhat),
                        "chance": chance,
                    }
                )
    results_df = pd.DataFrame(results_df)
    results_df["norm_acc"] = results_df.test_acc - results_df.chance
    return results_df


# %% event aligned analyses


def get_session_event_aligned_goal_decoding(
    session,
    event="cue",
    resolution=0.5,
    window=(-10, 10),
    goal_stratified_validation=True,
    n_test_trials=None,
    include_multi_units=True,
    whiten_features=True,
):
    """
    Chance is always 1 / n_goals.
    """
    input_data = dutils.get_event_aligned_input_data(session, event, resolution, window, include_multi_units)
    timepoints = sorted(input_data.event_aligned_time[event].unique())
    folds_df = dutils.get_folds_df(
        session, goal_stratified_validation, return_unique_IDs=True, n_test_trials=n_test_trials
    )
    results_dfs = []
    for fold in folds_df.columns.levels[0].unique():
        fold_df = folds_df[fold]
        test_trials = [t for t in fold_df.test.values.flatten() if isinstance(t, str)]
        train_trials = [t for t in fold_df.train.values.flatten() if isinstance(t, str)]
        train_df = input_data[input_data.trial_unique_ID.isin(train_trials)]
        test_df = input_data[input_data.trial_unique_ID.isin(test_trials)]
        decoder = LogisticRegression(penalty=None, max_iter=10000, random_state=0)
        for t in timepoints:
            _train_df = train_df[train_df.event_aligned_time[event] == t]
            _test_df = test_df[test_df.event_aligned_time[event] == t]
            if _train_df.empty or _test_df.empty:
                continue  # rare cases when no trials for that timepoint (eg, end of session trial)
            X_train, y_train = _train_df.spike_count.values, _train_df.goal.values
            X_test, y_test = _test_df.spike_count.values, _test_df.goal.values
            if whiten_features:  # zscore features
                scaler = StandardScaler()  # mean=0, std=1 per column
                scaler.fit(train_X)  # learn stats on train
                train_X = scaler.transform(train_X)
                test_X = scaler.transform(test_X)
            # fit model
            decoder.fit(X_train, y_train)
            # out_df
            Gprobs = decoder.predict_proba(X_test)
            n_samples, n_goals = Gprobs.shape
            goals = list(decoder.classes_)
            df = pd.DataFrame(
                {
                    "timepoint": np.repeat(t, n_samples * n_goals),
                    "true_goal": np.repeat(y_test, n_goals),
                    "trial_unique_ID": np.repeat(_test_df.trial_unique_ID.values, n_goals),
                    "predicted_goal": np.tile(goals, n_samples),
                    "predicted_goal_prob": Gprobs.ravel(),
                }
            )
            df["fold"] = fold
            results_dfs.append(df)
    results_df = pd.concat(results_dfs, axis=0)
    results_df.reset_index(drop=True, inplace=True)
    return results_df


# %% dev
