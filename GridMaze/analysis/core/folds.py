"""
Library for generating folds for cross-validation in various analyses.
"""

# %% Imports
import numpy as np
import pandas as pd
from math import ceil


from GridMaze.analysis.core import convert

# %% Global Vairiables


# %% test train splits functions


def _get_test_train_dfs(input_data, fold_df, training_trial_phases=["navigation"]):
    """ """
    test_trials = [t for t in fold_df.test.values.flatten() if isinstance(t, str)]
    train_trials = [t for t in fold_df.train.values.flatten() if isinstance(t, str)]
    train_df = input_data[input_data.trial_unique_ID.isin(train_trials)]
    # include only specified trial phases in training data
    if training_trial_phases:
        train_df = train_df[train_df.trial_phase.isin(training_trial_phases)]
    test_df = input_data[input_data.trial_unique_ID.isin(test_trials)]
    return train_df, test_df


# %% session level validation folds


def get_folds_df(
    session, goal_stratified=True, valid_trials=None, return_unique_IDs=True, n_folds=5, min_goal_repeats=3
):
    """
    if goal_stratified is True, then hold one goal out for test per fold
    else split trials randomly into n_folds with 1 test fold and n-1 training folds
    """
    n_trials = session.trials_df.trial.max()
    if goal_stratified:
        # check there are are enogh trials to stratify by goals if not split trials randomly
        # only applies to early sessions
        if n_trials < len(session.goals) * min_goal_repeats:
            folds_df = _get_folds_non_stratified(session, valid_trials, n_folds, return_unique_IDs)
        else:
            folds_df = _get_folds_goal_stratified(session, valid_trials, return_unique_IDs)
    else:
        folds_df = _get_folds_non_stratified(session, valid_trials, n_folds, return_unique_IDs)
    return folds_df


def _get_folds_goal_stratified(session, valid_trials=None, return_unique_IDs=True):
    """ """
    goals_df = get_goals_df(session, valid_trials, return_unique_IDs)
    # check there are at least 2 trials per goal (needed for test/train split)
    valid_goals_df = goals_df[goals_df.count(axis=1).ge(2)]
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
    return folds_df


def get_goals_df(session, valid_trials=None, return_unique_IDs=True):
    """
    returns df with goals in index and corresponding session trials in columns
    """
    trials_df = session.trials_df
    assert trials_df.trial.max() > len(session.goals), "Session does not have enough trials to stratify by goals"
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


def _get_folds_non_stratified(
    session,
    valid_trials=None,
    n_folds=5,
    return_unique_IDs=True,
):
    """
    No goal stratified validation folds. Instead specify how many trials per fold.
    Can be useful for all goals sessions where you want more training data per fold.
    """
    assert n_folds is not None, "n_test_trials must be specified for non goal-stratified validation folds"
    session_info = session.session_info
    trials_df = session.trials_df
    trials = trials_df.trial.values if valid_trials is None else valid_trials
    if len(trials) < n_folds:
        print(f"not enough trials to stratify by {n_folds}, using {len(trials)} folds")
        n_folds = len(trials)
    # shuffle trials
    trials = np.random.choice(trials, size=len(trials), replace=False)
    trials_per_fold = ceil(len(trials) / n_folds)
    trials_nan_pad = list(trials) + [np.nan] * (trials_per_fold * n_folds - len(trials))
    df = pd.DataFrame(
        data=np.array(trials_nan_pad).reshape(trials_per_fold, n_folds),
    )
    fold_dfs = []
    for i in range(n_folds):
        fold = f"fold_{i}"
        test_df = pd.DataFrame(df[i])
        test_df.columns = pd.MultiIndex.from_product([[fold], ["test"], test_df.columns])
        train_df = df.drop(columns=i)
        train_df.columns = pd.MultiIndex.from_product([[fold], ["train"], train_df.columns])
        fold_dfs.append(pd.concat([test_df, train_df], axis=1))
    folds_df = pd.concat(fold_dfs, axis=1)
    if return_unique_IDs:
        folds_df = folds_df.apply(lambda x: convert.trial2trial_unique_ID(session_info, x))
    return folds_df
