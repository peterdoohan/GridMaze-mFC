"""
Library for distance-to-goal alaigned goal decoding.
Eg, build separate decoders for neural activity 1, step from goal, 2 steps from goal, etc.
@peterdoohan
"""

# %% Imports
import json
from math import dist
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import multipletests


from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import convert


# %% Global Variables

from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

FRAME_RATE = 60
# %% Functions


def plot_distance_aligned_decoding(
    results_df, ax=None, color="rosybrown", n_bootstraps=1000, ymax=0.45, plot_sig=False
):
    """
    Plot chance substracted accuracy
    """
    # set up plot
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(4, 3), clear=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Steps to goal")
    ax.set_ylabel("Decoding Acc. \n (chance subtracted)")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax.set_ylim(-0.02, ymax)
    # normalise decoding acc to chance level on each session-step-fold
    results_df["norm_acc"] = results_df.test_acc - results_df.chance
    # get average decoding acc over steps_to_goal across subjects
    if n_bootstraps:
        rng = np.random.default_rng()
        subject_perms = rng.choice(
            SUBJECT_IDS,
            size=(n_bootstraps, len(SUBJECT_IDS)),
            replace=True,
        )
        perm_accs = []
        for i, p in enumerate(subject_perms):
            df = pd.concat([results_df[results_df.subject_ID == s] for s in p], axis=0).reset_index(drop=True)
            perm_accs.append(df.groupby("steps_to_goal").norm_acc.mean())
        perm_df = pd.concat(perm_accs, axis=1).T.reset_index(drop=True)
        lower_bound = perm_df.quantile(0.025, axis=0)
        upper_bound = perm_df.quantile(0.975, axis=0)
        mean = perm_df.mean(axis=0)
        steps = perm_df.columns.values
        # plot decoding acc with var
        ax.plot(steps, mean, color=color, lw=2)
        ax.fill_between(steps, lower_bound, upper_bound, color=color, alpha=0.2)
        # calc above chance timepoints
        if plot_sig:
            step_pvalues = 1 - perm_df.gt(0).mean(0)
            reject, pvals_corrected, _, _ = multipletests(step_pvalues, alpha=0.05, method="hs", maxiter=1)
            sig_steps = steps[reject]
            if len(sig_steps) > 1:
                ax.scatter(sig_steps, np.ones(len(sig_steps)) * ymax, marker="s", color=color, s=5)

    else:  # plot mean and sem across subjects
        df = results_df.groupby(["steps_to_goal", "subject_ID"]).norm_acc.mean().unstack().T
        steps = df.columns.values
        mean = df.mean(axis=0)
        sem = df.sem(axis=0)
        # plot
        ax.plot(steps, mean, color=color, lw=2)
        ax.fill_between(steps, mean - sem, mean + sem, color=color, alpha=0.2)
    ax.set_xlim(0, steps.max())


def distance_aligned_goal_decoding(maze_names=["maze_1", "maze_2"], goal_sets=["subset_1", "subset_2"], verbose=True):
    """"""
    results_dfs = []
    for subject_ID in SUBJECT_IDS:
        if verbose:
            print(f"Loading {subject_ID} data...")
        sessions = get_sessions_for_analysis([subject_ID], maze_names, goal_sets)
        for session in sessions:
            if verbose:
                print(f"Decoding: {session.name}")
            results_df = get_session_distance_aligned_goal_decoding(session)
            results_df["subject_ID"] = subject_ID
            results_df["maze_name"] = session.maze_name
            results_df["goal_subset"] = session.goal_subset
            results_df["days_on_maze"] = session.day_on_maze
            results_dfs.append(results_df)
    return pd.concat(results_dfs, axis=0)


def get_sessions_for_analysis(subject_IDs, maze_names, goal_subsets):
    """ """
    days_on_maze = "late" if "all" in goal_subsets else "all"
    sessions = gs.get_maze_sessions(
        subject_IDs=subject_IDs,
        maze_names=maze_names,
        days_on_maze=days_on_maze,
        goal_subsets=goal_subsets,
        with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "trials_df"],
        must_have_data=False,
    )
    sessions = [sessions] if isinstance(sessions, gs.MazeSession) else sessions
    # check sessions have at least 2 trials per goal
    keep_sessions = []
    for session in sessions:
        trials_df = session.trials_df
        if trials_df.groupby("goal").trial.count().ge(2).all():
            keep_sessions.append(session)
    if len(keep_sessions) == 0:
        raise FileNotFoundError(f"No valid session for: {subject_IDs}, {maze_names}, {goal_subsets}")
    return keep_sessions


def get_session_distance_aligned_goal_decoding(session, max_steps_from_goal=30):
    """ """
    input_df = get_session_input_data(
        session, resolution=0.2, include_multi_units=True, max_steps_to_goal=max_steps_from_goal
    )
    results_df = []
    for steps in range(max_steps_from_goal + 1):
        steps_df = input_df[input_df.steps_to_goal.future == steps]
        valid_trials = steps_df.trial.unique()
        folds_df, chance = get_folds_df(session, valid_trials, return_chance=True)
        if folds_df.shape[0] < 2:
            # only one valid goal, cannot run classifer
            continue
        folds = folds_df.columns.levels[0].unique()
        for fold in folds:
            # get test and train data
            fold_df = folds_df[fold]
            test_trials = fold_df.test.unstack().dropna().values
            train_trials = fold_df.train.unstack().dropna().values
            test_df = steps_df[steps_df.trial_unique_ID.isin(test_trials)]
            test_X, test_y = test_df.spike_count.values, test_df.goal.values
            train_df = steps_df[steps_df.trial_unique_ID.isin(train_trials)]
            train_X, train_y = train_df.spike_count.values, train_df.goal.values
            # fit model
            decoder = LogisticRegression(penalty=None, max_iter=10000)
            decoder.fit(train_X, train_y)
            # test decoder
            test_pred = decoder.predict(test_X)
            test_acc = np.mean(test_pred == test_y)
            train_pred = decoder.predict(train_X)
            train_acc = np.mean(train_pred == train_y)
            results_df.append(
                {
                    "steps_to_goal": steps,
                    "fold": fold,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "chance": chance,
                }
            )
    return pd.DataFrame(results_df)


def get_session_input_data(session, resolution=0.2, include_multi_units=True, max_steps_to_goal=20):
    """ """
    navigation_df = session.navigation_df
    spike_counts_df = session.navigation_spike_counts_df.reset_index(drop=True)
    # filter clusters
    keep_clusters = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=include_multi_units,
    )
    spike_counts_df = spike_counts_df[spike_counts_df.columns[spike_counts_df.spike_count.columns.isin(keep_clusters)]]
    # downsample spike counts by suming spikes within resolution window
    ds_frames = int(FRAME_RATE * resolution)
    ds_spike_counts_df = spike_counts_df.groupby(spike_counts_df.index // ds_frames).sum().reset_index(drop=True)
    # keep only relevant navigation info
    nav_info = navigation_df[
        [
            ("time", ""),
            ("trial_unique_ID", ""),
            ("trial", ""),
            ("goal", ""),
            ("trial_phase", ""),
            ("steps_to_goal", "future"),
        ]
    ]
    # downsample navigation info by taking values in mid window
    mid_window_inds = (spike_counts_df.index // ds_frames).unique() * ds_frames + (ds_frames // 2)
    mid_window_inds = mid_window_inds[mid_window_inds < len(nav_info)]
    nav_info = nav_info.iloc[mid_window_inds]
    nav_info.reset_index(drop=True, inplace=True)
    if nav_info.shape[1] < ds_spike_counts_df.shape[1]:
        ds_spike_counts_df = ds_spike_counts_df.iloc[:-1]

    ds_nav_rates_df = pd.concat([nav_info, ds_spike_counts_df], axis=1)
    # filter for navigation only and to max_steps_to_goal
    ds_nav_rates_df = ds_nav_rates_df[ds_nav_rates_df.steps_to_goal.future.le(max_steps_to_goal)]
    ds_nav_rates_df[("trial", "")] = ds_nav_rates_df[("trial", "")].astype(int)
    return ds_nav_rates_df


def get_goals_df(session, valid_trials=None, return_unique_IDs=True):
    """
    returns df with goals in index and corresponding session trials in columns
    """
    trials_df = session.trials_df
    if valid_trials is not None:
        trials_df = trials_df[trials_df.trial.isin(valid_trials)].reset_index(drop=True)
    goal2trials = {}
    for goal in session.goals:
        goal2trials[goal] = trials_df[trials_df.goal == goal].trial.to_list()
    goals_df = pd.DataFrame.from_dict(goal2trials, orient="index")
    if return_unique_IDs:
        session_info = session.session_info
        goals_df = goals_df.apply(lambda x: convert.trial2trial_unique_ID(session_info, x))
    return goals_df


def get_folds_df(session, valid_trials=None, return_chance=True):
    """ """
    goals_df = get_goals_df(session, valid_trials, return_unique_IDs=True)
    # check there are at least 2 trials per goal (needed for test/train split)
    valid_goals_df = goals_df[goals_df.count(axis=1).ge(2)]
    chance = 1 / valid_goals_df.shape[0]
    # shuffle
    valid_goals_df = valid_goals_df.apply(lambda x: np.random.choice(x, size=len(x), replace=False), axis=1).apply(
        pd.Series
    )
    # split into test and train folds
    cols = valid_goals_df.columns
    fold_dfs = []
    for i in cols:
        fold = f"fold_{i}"
        test_df = pd.DataFrame(valid_goals_df[cols[i]])
        test_df.columns = pd.MultiIndex.from_product([[fold], ["test"], test_df.columns])
        train_df = valid_goals_df.drop(columns=cols[i])
        train_df.columns = pd.MultiIndex.from_product([[fold], ["train"], train_df.columns])
        fold_dfs.append(pd.concat([test_df, train_df], axis=1))
    # return as df
    folds_df = pd.concat(fold_dfs, axis=1)
    if not return_chance:
        return folds_df
    else:
        return folds_df, chance
