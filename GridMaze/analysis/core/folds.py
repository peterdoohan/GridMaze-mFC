"""
Library for generating folds for cross-validation in various analyses.
"""

# %% Imports
import numpy as np
import pandas as pd

from GridMaze.analysis.core import convert

# %% Global Vairiables


# %% session level validation folds


def get_folds_df(session, goal_stratified=True, valid_trials=None, return_unique_IDs=True, n_test_trials=None):
    """ """
    n_trials = session.trials_df.trial.max()
    if goal_stratified:
        # check there are are enogh trials to stratify by goals if not split trials randomly
        # only applies to early sessions
        if n_trials < len(session.goals) * 2:
            folds_df = _get_folds_non_stratified(session, valid_trials, n_test_trials=(n_trials // 5))
        else:
            folds_df = _get_folds_goal_stratified(session, valid_trials, return_unique_IDs)
    else:
        folds_df = _get_folds_non_stratified(session, valid_trials, n_test_trials, return_unique_IDs)
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
    n_test_trials=5,
    return_unique_IDs=True,
):
    """
    No goal stratified validation folds. Instead specify how many trials per fold.
    Can be useful for all goals sessions where you want more training data per fold.
    """
    assert n_test_trials is not None, "n_test_trials must be specified for non goal-stratified validation folds"
    session_info = session.session_info
    trials_df = session.trials_df
    trials = trials_df.trial.values if valid_trials is None else valid_trials
    # shuffle trials
    trials = np.random.choice(trials, size=len(trials), replace=False)
    fold_dfs = []
    for fold, i in enumerate(range(0, len(trials), n_test_trials)):
        test_trials = trials[i : i + n_test_trials]
        train_trials = np.concatenate([trials[:i], trials[i + n_test_trials :]])
        fold_df = pd.DataFrame(
            {
                "test": pd.Series(test_trials),
                "train": pd.Series(train_trials),
            }
        )
        fold_df.columns = pd.MultiIndex.from_product([[f"fold_{fold}"], fold_df.columns])
        fold_dfs.append(fold_df)
    folds_df = pd.concat(fold_dfs, axis=1)
    if return_unique_IDs:
        folds_df = folds_df.apply(lambda x: convert.trial2trial_unique_ID(session_info, x))
    return folds_df
