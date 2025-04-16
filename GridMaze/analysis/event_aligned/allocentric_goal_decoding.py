"""This module contains funtions to decoding navigational goals from time-aligned population activity"""

# %% Imports
import json
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed
from sklearn.neural_network import MLPClassifier

from ..core import get_sessions as gs
from ..core import get_clusters as gc

# %% Global variables

RATES_SAMPLE_RATE = 0.04

from ...paths import EXPERIMENT_INFO_PATH, ANALYSIS_INFO_PATH, RESULTS_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(ANALYSIS_INFO_PATH / "intra_trial_interval_times.json", "r") as f:
    INTRA_TRIAL_INTERVAL_TIMES = json.load(f)

RESULTS_DIR = RESULTS_PATH / "goal_coding"

# %% Plotting results


def plot_trial_aligned_decoding_results(results_df, ax=None, sem=True, chance=1 / 12):
    """ """
    df = results_df.xs("test", level=2, axis=0)
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(5, 3), clear=True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Acc.")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-5, INTRA_TRIAL_INTERVAL_TIMES["ITI_end"] + 0.5)
    ax.set_xticks(list(INTRA_TRIAL_INTERVAL_TIMES.values()))
    ax.set_xticklabels(["cue", "reward", "erc", "end"])
    for time in INTRA_TRIAL_INTERVAL_TIMES.values():
        ax.axvline(time, color="black", linestyle="--", alpha=0.5)
    ax.axhline(chance, color="black", linestyle="--", alpha=0.5)
    # plot results
    mean_acc = df.mean(axis=1)  # across folds
    time = mean_acc.index.values.astype(float)
    av = mean_acc.values.astype(float)
    ax.plot(time, av, color="deepskyblue")
    if sem:
        sem = df.sem(axis=1).values.astype(float)
        ax.fill_between(time, av - sem, av + sem, color="deepskyblue", alpha=0.3)


def plot_event_aligned_decoding_results(results_df, axes=None, chance=1 / 12):
    """ """
    df = results_df.xs("test", level=2, axis=0)
    if axes is None:
        f, axes = plt.subplots(1, 2, figsize=(6, 3), clear=True, sharey=True)
    for ax, label in zip(axes, ["Cue", "Reward"]):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("Acc.")
        ax.set_xlabel(label)
        ax.set_ylim(0, 1)
        ax.set_xlim(-10, 10)
        ax.axhline(chance, color="black", linestyle="--", alpha=0.5)
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    # plot results
    mean_acc = df.mean(axis=1)  # across folds
    for ax, ind in zip(axes, ["cue_aligned", "reward_aligned"]):
        time = mean_acc[ind].index.values.astype(float)
        av = mean_acc[ind].values.astype(float)
        ax.plot(time, av, color="deepskyblue")


# %% run bootstrapped permutation tests


def run_bootstrapped_allocentric_goal_deocding(maze_name, goal_subset, alignment="trial", n_permutations=8, n_jobs=4):
    """ """
    # set up somewhere to save results bc. this is going to take a while
    save_dir = RESULTS_DIR / "permutation_results" / alignment / maze_name / goal_subset
    save_dir.mkdir(parents=True, exist_ok=True)
    # load sessions once
    subject2sessions = {}
    for subject in SUBJECT_IDS:
        subject2sessions[subject] = get_sessions_for_analysis([subject], maze_name, goal_subset, alignment)
    # generate session list permutations bootstrapping over subjects
    rng = np.random.default_rng()
    subject_perms = rng.choice(
        SUBJECT_IDS,
        size=(n_permutations, len(SUBJECT_IDS)),
        replace=True,
    )
    session_perms = [
        [session for subject in subjects for session in subject2sessions[subject]] for subjects in subject_perms
    ]
    # get save paths for each permutation
    save_paths = [save_dir / f"perm_{i}.csv" for i in range(n_permutations)]
    Parallel(n_jobs=n_jobs)(
        delayed(_run_allocentric_goal_decoding)(
            session_perms[i],
            alignment=alignment,
            include_multi_units=True,
            window_size=0.2,
            smooth_SD=4,
            save_path=save_paths[i],
        )
        for i in range(n_permutations)
    )


def _run_allocentric_goal_decoding(
    sessions,
    alignment="trial",
    include_multi_units=True,
    window_size=0.2,
    smooth_SD=4,
    save_path=None,
):
    activity_df = get_activity_df(sessions, alignment, include_multi_units, window_size, smooth_SD)
    validation_fold_df = get_validation_folds_df(sessions, n_training_trial_sets=15, alignment=alignment)
    results_df = _get_decoding_accurary(activity_df, validation_fold_df)
    if save_path is not None:
        results_df.to_csv(save_path)
    else:
        return results_df


# %% allocentric goal decoding


def run_allocentric_goal_decoding(
    subject_IDs="all",
    maze_name="maze_1",
    goal_subset="subset_1",
    alignment="trial",
    include_multi_units=True,
    window_size=0.5,
    smooth_SD=4,
    plot=True,
):
    """ """
    sessions = get_sessions_for_analysis(subject_IDs, maze_name, goal_subset, alignment)
    activity_df = get_activity_df(sessions, alignment, include_multi_units, window_size, smooth_SD)
    validation_fold_df = get_validation_folds_df(sessions, n_training_trial_sets=100, alignment=alignment)
    results_df = _get_decoding_accurary(activity_df, validation_fold_df)
    if plot:
        if alignment == "event":
            plot_event_aligned_decoding_results(results_df)
        elif alignment == "trial":
            plot_trial_aligned_decoding_results(results_df)
    return results_df


def get_sessions_for_analysis(subject_IDs, maze_name, goal_subset, alignment):
    """ """
    subject_IDs = SUBJECT_IDS if subject_IDs == "all" else subject_IDs
    data = ["event_aligned_rates_df", "trial_aligned_rates_df", "cluster_metrics"]
    sessions = []  # loop over subjects so we can add the same session twice when running bootstrapped permutation tests
    for subject in subject_IDs:
        s = gs.get_maze_sessions(
            subject_IDs=[subject],
            maze_names=[maze_name],
            days_on_maze="all",
            goal_subsets=[goal_subset],
            with_data=data,
        )
        s = [s] if isinstance(s, gs.MazeSession) == 1 else s
        sessions.extend(s)
    # check sessions have enough trials (>=2) per goal for test-train split (cannot combine data across sessions)
    for session in sessions:
        trials_per_goal = get_trials_per_goal(session, alignment)
        if trials_per_goal.apply(len).lt(2).any():
            print(f"{session} has some goals have less than 2 valid trials")
            sessions.remove(session)
    return sessions


def get_activity_df(sessions, alignment="event", include_multi_units=True, window_size=0.2, smooth_SD=4):
    """"""
    # combine data across sessions w/ select units
    if alignment == "event":
        data_struc = "event_aligned_rates_df"
    elif alignment == "trial":
        data_struc = "trial_aligned_rates_df"
    else:
        raise ValueError("alignment must be 'event' or 'trial'")

    aligned_rates_dfs = []
    for session in sessions:
        event_aligned_rates_df = getattr(session, data_struc)
        keep_clusters = gc.filter_clusters(
            session.cluster_metrics,
            session.session_info,
            return_unique_IDs=True,
            single_units=True,
            multi_units=include_multi_units,
        )
        aligned_rates_dfs.append(event_aligned_rates_df[event_aligned_rates_df.cluster_unique_ID.isin(keep_clusters)])
    aligned_rates_df = pd.concat(aligned_rates_dfs, axis=0)
    # and add trial_unique_IDs
    if alignment == "event":
        col = ("trial_unique_ID", "", "")
    elif alignment == "trial":
        col = ("trial_unique_ID", "")
    aligned_rates_df.loc[:, col] = (
        aligned_rates_df[["subject_ID", "maze_name", "day_on_maze", "trial"]].astype(str).agg("_".join, axis=1)
    )
    # reduce resolution to specified window length
    window_length = int(window_size / RATES_SAMPLE_RATE)
    if alignment == "event":
        ds_rates_dfs = []
        for event in ["cue_aligned", "reward_aligned"]:
            rates_df = aligned_rates_df.firing_rate[event]
            ds_rates_df = rates_df.T.rolling(window=window_length, center=True).mean()[::window_length].dropna().T
            ds_rates_df.columns = pd.MultiIndex.from_tuples([("firing_rate", event, c) for c in ds_rates_df.columns])
            if smooth_SD:
                ds_rates = ds_rates_df.values
                smooted_rates = gaussian_filter1d(ds_rates, smooth_SD, axis=1)
                ds_rates_df = pd.DataFrame(smooted_rates, index=ds_rates_df.index, columns=ds_rates_df.columns)
            ds_rates_dfs.append(ds_rates_df)
        ds_rates_df = pd.concat(ds_rates_dfs, axis=1)
    elif alignment == "trial":
        rates_df = aligned_rates_df.firing_rate
        ds_rates_df = rates_df.T.rolling(window=window_length, center=True).mean()[::window_length].dropna().T
        ds_rates_df.columns = pd.MultiIndex.from_tuples([("firing_rate", c) for c in ds_rates_df.columns])
        if smooth_SD:
            ds_rates = ds_rates_df.values
            smooted_rates = gaussian_filter1d(ds_rates, smooth_SD, axis=1)
            ds_rates_df = pd.DataFrame(smooted_rates, index=ds_rates_df.index, columns=ds_rates_df.columns)
    # replace firing_rates columns with downsampled version
    aligned_rates_df = aligned_rates_df.drop("firing_rate", axis=1, level=0)
    activity_df = pd.concat([aligned_rates_df, ds_rates_df], axis=1)
    return activity_df


def _get_decoding_accurary(
    activity_df, validation_folds_df, classifier="logreg", decoder_kwargs={"inv_alpha": None}, verbose=False
):
    """Returns decoding accuracy for each timepoint and fold"""
    timepoints = activity_df.firing_rate.columns
    folds = validation_folds_df.columns.get_level_values(0).unique()
    results_df = pd.DataFrame(
        index=pd.MultiIndex.from_tuples([(*col, new_level) for col in timepoints for new_level in ["test", "train"]]),
        columns=folds,
    )
    for fold in folds:
        if verbose:
            print(fold)
        # get test, train data for logistic regression
        fold_df = validation_folds_df[fold]
        if len(fold_df.test.columns[0][0]) > 1:  # remove empty index when combining data across sessions
            test_df = fold_df.test.droplevel(0, axis=1)
        else:
            test_df = fold_df.test
        test_X, test_y = get_synthetic_activity_matrix(
            activity_df, test_df
        )  # [n_goals, n_session_clusters, n_timepoints], [n_goals]
        training_df = fold_df.training
        training_Xs, training_ys = [], []
        for training_set in training_df.columns.get_level_values(0).unique():
            training_set_df = training_df[training_set]
            X, y = get_synthetic_activity_matrix(activity_df, training_set_df)
            training_Xs.append(X)
            training_ys.append(y)
        training_X = np.vstack(training_Xs)  # [n_training_sets x n_goals, n_session_clusters, n_timepoints]
        training_y = np.hstack(training_ys)  # [n_training_sets x n_goals]

        # set up decoder based on speified inputs
        if classifier == "logreg":
            if decoder_kwargs["inv_alpha"] is None:
                decoder = LogisticRegression(penalty=None, max_iter=10000)
            else:
                decoder = LogisticRegression(
                    penalty="l2",
                    solver="lbfgs’",
                    C=decoder_kwargs["inv_alpha"],
                    max_iter=10000,
                )
        elif classifier == "mlp":
            decoder = MLPClassifier(
                hidden_layer_sizes=decoder_kwargs["Nhid"],
                alpha=decoder_kwargs["alpha"],
                max_iter=10000,
            )
        else:
            raise NotImplementedError

        # run decoding at each timepoint
        for i in range(len(timepoints)):
            test_activity = test_X[:, :, i]  # [n_goals, n_session_clusters]
            training_activity = training_X[:, :, i]  # [n_training_sets x n_goals, n_session_clusters]
            decoder.fit(training_activity, training_y)
            test_predictions = decoder.predict(test_activity)
            test_accuracy = (test_predictions == test_y).mean()
            train_predictions = decoder.predict(training_activity)
            train_accuracy = (train_predictions == training_y).mean()
            results_df.loc[(*timepoints[i], "test"), fold] = test_accuracy
            results_df.loc[(*timepoints[i], "train"), fold] = train_accuracy
    return results_df


def get_synthetic_activity_matrix(activity_df, test_df):
    """ """
    activity_df = activity_df.set_index("trial_unique_ID")
    n_session_clusters = len(activity_df.cluster_unique_ID.unique())
    n_goals = test_df.shape[0]
    n_timepoints = activity_df.firing_rate.shape[1]
    synthetic_activity_matrix = np.full((n_goals, n_session_clusters, n_timepoints), np.nan)
    goal_vector = []
    for i, goal in enumerate(test_df.index):
        test_trials = test_df.loc[goal]
        test_trials_activity_df = activity_df.loc[test_trials]
        test_trials_activity_df = test_trials_activity_df[
            [c for c in test_trials_activity_df.columns if c[0] in ["cluster_unique_ID", "firing_rate"]]
        ].reset_index()  # [n_test_trials x n_neurons, n_timepoints + 2]
        synthetic_trial_activity_df = test_trials_activity_df.pivot(
            index="trial_unique_ID", columns="cluster_unique_ID"
        )  # [n_test_trials, n_timespoints x n_session-clusters], only has values for clusters recorded on trials from the same session
        synthetic_trial_activity = synthetic_trial_activity_df.sum(
            axis=0
        )  # collapse to [n_session-clusters, n_timespoints, 1] (pd.Series)
        # assign synthetic trial activity to synthetic activity matrix
        synthetic_activity_matrix[i, :, :] = (
            synthetic_trial_activity.unstack().values.T
        )  # [n_session-clusters, n_timespoints (cue and reward aligned)]
        goal_vector.append(goal)
    return synthetic_activity_matrix, np.array(goal_vector)


# %% Functions building validation_folds_df


def get_validation_folds_df(sessions, n_training_trial_sets, alignment="event"):
    """ """
    exp_max_trials_per_goal = get_max_trials_per_goal(sessions, alignment)
    session_fold_dfs = []
    for session in sessions:
        session_fold_df = get_session_validation_folds_df(
            session, exp_max_trials_per_goal, n_training_trial_sets, alignment
        )
        # turn from trial float value to trial_unique_ID string
        session_fold_df = session_fold_df.map(
            lambda x: f"{session.subject_ID}_{session.maze_name}_{session.day_on_maze}_{int(x)}"
        )
        session_fold_dfs.append(session_fold_df)
    validation_folds_df = pd.concat(session_fold_dfs, axis=1).sort_index(axis=1)
    return validation_folds_df


def get_max_trials_per_goal(sessions, alignment):
    """returns the maximum trials compelted for any one goal in a session or list of sessions"""
    sessions = [sessions] if isinstance(sessions, gs.MazeSession) else sessions
    session_maxes = []
    for session in sessions:
        trials_per_goal = get_trials_per_goal(session, alignment)
        max_trials_per_goal = np.max([len(t) for t in trials_per_goal])
        session_maxes.append(max_trials_per_goal)
    return np.max(session_maxes)


def get_trials_per_goal(session, alignment):
    if alignment == "event":  # has all trials even if very short etc.
        df = session.event_aligned_rates_df
    elif alignment == "trial":  # note some trials are missing in trial_aligned_rates_df bc/ erc after ITI
        df = session.trial_aligned_rates_df
    trials2goal_df = df.loc[:, (df.columns.get_level_values(0).isin(["trial", "goal"]))]
    trials_per_goal = trials2goal_df.dropna().groupby("goal")["trial"].apply(lambda x: np.unique(x))
    return trials_per_goal


def get_session_validation_folds_df(session, exp_max_trials_per_goal, n_training_trial_sets, alignment):
    """"""
    session_name = session.name
    if alignment == "event":  # has all trials even if very short etc.
        df = session.event_aligned_rates_df
    elif alignment == "trial":  # note some trials are missing in trial_aligned_rates_df bc/ erc after ITI
        df = session.trial_aligned_rates_df
    trials2goal_df = df.loc[:, (df.columns.get_level_values(0).isin(["trial", "goal"]))]
    trials_per_goal = trials2goal_df.dropna().groupby("goal")["trial"].apply(lambda x: np.unique(x))
    max_trials_per_goal = np.max([len(t) for t in trials_per_goal])
    # goals must have 2 or more valid trials (enough for test_train split)
    if trials_per_goal.apply(len).le(2).any():
        raise ValueError(f"{session} has some goals have less than 2 valid trials")
    # test-train splits
    fold_dfs = []
    f = 0
    trials_per_goal = trials_per_goal.apply(lambda x: np.random.choice(x, size=len(x), replace=False))
    trials_per_goal_df = trials_per_goal.apply(pd.Series)
    for fold in range(exp_max_trials_per_goal):
        if f > max_trials_per_goal - 1:  # shuffle and reset fold counter
            trials_per_goal = trials_per_goal.apply(lambda x: np.random.choice(x, size=len(x), replace=False))
            trials_per_goal_df = trials_per_goal.apply(pd.Series)
            f = 0
        tpg_df = trials_per_goal_df.copy()
        test_trial_set = tpg_df[f]
        training_trial_sets = tpg_df.drop(f, axis=1)
        while test_trial_set.isna().any():  # choose another fold for test if includes NaNs
            replacement_fold = np.random.choice(training_trial_sets.columns)
            replacement_set = tpg_df[replacement_fold]
            if replacement_set.isna().any():
                continue
            test_trial_set = test_trial_set.fillna(replacement_set)
            training_trial_sets = training_trial_sets.drop(replacement_fold, axis=1)
        # extend training trial sets to desired length
        n_synthetic_trial_sets = n_training_trial_sets - training_trial_sets.shape[-1]
        extra_training_trial_sets = get_synthetic_trial_sets(
            training_trial_sets, n_synthetic_trial_sets, max_trials_per_goal
        )
        training_trial_sets = fill_training_trial_sets_df(training_trial_sets)
        training_trial_sets = pd.concat([training_trial_sets, extra_training_trial_sets], axis=1)
        training_trial_sets.columns = pd.MultiIndex.from_tuples(
            [(f"fold_{fold}", "training", c, session_name) for c in range(n_training_trial_sets)]
        )
        # add training sets to fold
        fold_df = training_trial_sets
        fold_df[(f"fold_{fold}", "test", "", session_name)] = test_trial_set
        fold_dfs.append(fold_df)
        f += 1
    session_fold_df = pd.concat(fold_dfs, axis=1)
    return session_fold_df


def fill_training_trial_sets_df(training_trial_sets):
    df = training_trial_sets.copy()
    for index, row in df.iterrows():
        nan_cols = row.index[row.isnull()]
        valid_cols = row.index[row.notnull()]
        if len(nan_cols) > len(valid_cols):
            sample = row.dropna().sample(n=len(nan_cols), replace=True)
        else:  # sample wo/ replacement possible
            sample = row.dropna().sample(n=len(nan_cols), replace=False)
        df.loc[index, nan_cols] = sample.values
    return df


def get_synthetic_trial_sets(training_trial_sets, n_extra_sets, max_trials_per_goal):
    synth_trials = []
    for index, row in training_trial_sets.iterrows():
        trial_generator = sample_without_replacement(list(row.dropna()))
        sampled_trials = []
        for i in range(n_extra_sets):
            sampled_trials.append(next(trial_generator))
        synth_trials.append(sampled_trials)
    return pd.DataFrame(
        np.array(synth_trials),
        columns=[f"s{max_trials_per_goal+i}" for i in range(n_extra_sets)],
        index=training_trial_sets.index,
    )


def sample_without_replacement(lst):
    assert len(lst) > 0, "list must have at least one item"
    working_copy = lst.copy()  # Create a copy of the list to work with
    while True:
        if not working_copy:  # If the working copy is empty, refresh it
            working_copy = lst.copy()
        choice = random.choice(working_copy)  # Randomly select an item
        working_copy.remove(choice)  # Remove the selected item to prevent future selection
        yield choice


# %% Validation folds quality control functions


def validation_fold_qc(validation_folds_df):
    """Checks test-train contamination, goal_stratification and similarity of training set folds"""
    if not _check_no_test_train_contamination(validation_folds_df):
        raise ValueError("test trials found in training sets")
    else:
        print("no test trials found in training sets")
    goal_stratification = _check_goal_stratification(validation_folds_df)
    print(f"goal stratification: {goal_stratification}")
    test_set_similarity = _check_similarity_of_training_sets(validation_folds_df)
    print(f"average similarity of training sets: {test_set_similarity}")
    return


def _check_goal_stratification(validation_folds_df):
    goals, counts = np.unique(validation_folds_df.index.to_numpy(), return_counts=True)
    return dict(zip(goals, counts))


def _check_similarity_of_training_sets(validation_folds_df):
    """Returns average similairy of training sets across folds. value between (0,1)"""
    similarities = []
    for fold in validation_folds_df.columns.get_level_values(0).unique():
        fold_df = validation_folds_df[fold]
        n_sets = fold_df.training.shape[-1]
        for i in range(n_sets):
            for j in range(i + 1, n_sets):
                set_i = fold_df.training[i]
                set_j = fold_df.training[j]
                similarity = (set_i == set_j).mean()
                similarities.append(similarity)
    return np.mean(similarities)


def _check_no_test_train_contamination(validation_folds_df):
    """Checks that there is no test trials in eany of the training data for each fold"""
    for fold in validation_folds_df.columns.get_level_values(0).unique():
        fold_df = validation_folds_df[fold]
        test_trials = fold_df.test.to_numpy().flatten()
        training_trials = fold_df.training.to_numpy().flatten()
        if np.in1d(test_trials, training_trials).any():
            return False
    return True


# %% Test regularisation param


def run_hyperparameter_search(maze_name="maze_1", goal_subset="subset_1"):
    sessions = get_sessions_for_analysis(
        subject_IDs="all",
        maze_name=maze_name,
        goal_subset=goal_subset,
        alignment="event",
    )
    validation_folds_df = get_validation_folds_df(sessions, n_training_trial_sets=100, alignment="event")
    for window_size in [0.1, 0.2, 0.5]:
        for smooth_SD in [False, 2, 4, 6]:
            activity_df = get_activity_df(
                sessions, alignment="event", include_multi_units=True, window_size=window_size, smooth_SD=smooth_SD
            )
            # filter activity df to only include the window at reward where decoding is best
            _df = activity_df.firing_rate.reward_aligned
            reward_time = _df.columns[np.abs(_df.columns).argmin()]
            activity_df = activity_df[
                [
                    c
                    for c in activity_df.columns
                    if c[0] != "firing_rate" or (c[1] == "reward_aligned" and c[2] == reward_time)
                ]
            ]
            # linear decoder hyperparameter search
            hp_search_results = []
            for alpha in [None, 1e-6, 1e-4, 1e-2, 1, 1e2]:
                results_df = _get_decoding_accurary(
                    activity_df,
                    validation_folds_df,
                    classifier="logreg",
                    decoder_kwargs={"inv_alpha": alpha},
                )
                mean_acc = results_df.droplevel([0, 1]).mean(axis=1)
                hp_result = {
                    "alpha": alpha,
                    "smooth_SD": smooth_SD,
                    "window_size": window_size,
                    "test_acc": mean_acc.test,
                    "train_acc": mean_acc.train,
                }
                print(hp_result)
                hp_search_results.append(hp_result)
    linear_hp_search_results = pd.DataFrame(hp_search_results)
    save_path = RESULTS_DIR / "hyperparameter_search" / maze_name / goal_subset / "logreg.csv"
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
    linear_hp_search_results.to_csv(save_path, index=False)
    return hp_search_results
