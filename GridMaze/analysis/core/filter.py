"""
Library for filtering GridMaze data objects, based on certain criteria for analysis.
Eg, filtering navigation df to only include times when moving OR select only kilosort "good" clusters.
"""

# %% Imports
import numpy as np
import pandas as pd

# %% Global Variables

FRAME_RATE = 60

# %% Functions


def filter_navigation_rates_df(
    navigation_rates_df,
    navigation_only=True,
    moving_only=True,
    exclude_time_at_goal=True,
    max_steps_to_goal=None,
):
    """
    Filter navigation rates dataframe based on specified criteria.
    Args:
        navigation_rates_df (pd.DataFrame): Dataframe of navigation rates (either frames.navigation.parquet or conctenated navigation rates)
        navigation_only (bool): Only include navigation trials
        moving_only (bool): Only include times when the animal is moving
        exclude_time_at_goal (bool): Exclude times when the animal is at the goal
    """
    # filter columns (frames)
    conditions = []
    if navigation_only:
        conditions.append(navigation_rates_df.trial_phase == "navigation")
    else:
        # ignore out of session times
        conditions.append(navigation_rates_df.trial_phase.isin(["navigation", "ITI", "reward_consumption"]))
    if moving_only:
        conditions.append(navigation_rates_df.moving)
    if exclude_time_at_goal:
        conditions.append(navigation_rates_df.goal != navigation_rates_df.maze_position.simple)
    if max_steps_to_goal is not None:
        conditions.append(navigation_rates_df.steps_to_goal.future.lt(max_steps_to_goal))
    combined_condition = np.logical_and.reduce(conditions)
    if len(conditions) > 0:
        navigation_rates_df = navigation_rates_df[combined_condition]
        return navigation_rates_df[:-1]  # exclude last row incase action is not defined
    else:
        return navigation_rates_df


def split_trials(navigation_rates_df):
    """splits trials from a navigation_df-containing dataframe into two halves, stratified by goal if possible."""
    if len(pd.Series(navigation_rates_df.goal.unique()).dropna()) == navigation_rates_df.trial.max():
        # too few trial to stratify split by goal -> split randomly instead
        trials = navigation_rates_df.trial.unique()
        trials = trials[~np.isnan(trials)]
        np.random.shuffle(trials)
        split_size = len(trials) // 2
        return trials[:split_size], trials[split_size:]
    else:
        trials2goal_df = navigation_rates_df.loc[
            :, (navigation_rates_df.columns.get_level_values(0).isin(["trial", "goal"]))
        ]
        trials_per_goal = trials2goal_df.dropna().groupby("goal")["trial"].apply(lambda x: np.unique(x))
        trials_per_goal = trials_per_goal.apply(
            lambda x: np.random.choice(x, size=len(x), replace=False)
        )  # shuffle trials
        goal_split_trials_df = trials_per_goal.apply(pd.Series)
        middle_index = len(goal_split_trials_df.columns) // 2
        split_1 = goal_split_trials_df.iloc[:, :middle_index].to_numpy().flatten()
        split_1 = split_1[~np.isnan(split_1)]
        split_2 = goal_split_trials_df.iloc[:, middle_index:].to_numpy().flatten()
        split_2 = split_2[~np.isnan(split_2)]
        return split_1, split_2


# %% Function for generating KFolds splits


def get_trial_splits(session, goal_stratified=True, split=0.5, n_splits=10):
    """"""
    trials_df = session.trials_df
    goal2trials = trials_df.groupby(("goal", "")).trial.apply(lambda x: np.unique(x))
    goal2trials_df = goal2trials.apply(pd.Series)
    # shuffle
    goal2trials_df = goal2trials_df.apply(_shuffle, axis=1)
    max_trials_per_goal = goal2trials_df.count(axis=1).max()
    if goal_stratified:
        n_split = int(max_trials_per_goal * split)
        splits = []
        for i in range(n_splits):
            goal2trials_df = goal2trials_df.apply(_shuffle, axis=1)
            split_1 = goal2trials_df.iloc[:, :n_split].unstack().to_numpy()
            split_2 = goal2trials_df.iloc[:, n_split:].unstack().to_numpy()
            splits.append((split_1, split_2))
        return splits


def _shuffle(x):
    return pd.Series(np.random.choice(x, size=len(x), replace=False))


def get_goal_stratified_validation_folds_df(session, test_trials_per_goal=1, only_include_trials=None):
    """
    Generates a DataFrame containing stratified validation folds based on goals.
    This function takes a session object and generates a DataFrame where trials are stratified by goals.
    The trials are shuffled and divided into training and testing sets for cross-validation purposes.
    Parameters:
    -----------
    session : object
        An object containing session data, which includes a DataFrame `trials_df` with trial information.
    test_trials_per_goal : int, optional
        The number of test trials per goal for each fold. Default is 1.
    Returns:
    --------
    pd.DataFrame
        A DataFrame with multi-index columns representing the folds. Each fold contains 'test' and 'train' sets
        for each goal. The columns are structured as follows:
        - Level 0: Fold identifier (e.g., 'fold_0', 'fold_1', ...)
        - Level 1: Set type ('test' or 'train')
        - Level 2: Original trial indices
    """
    trials_df = session.trials_df
    if only_include_trials is not None:
        trials_df = trials_df[trials_df.trial.isin(only_include_trials)]
    goal2trials = trials_df.groupby(("goal", "")).trial.apply(lambda x: np.unique(x))
    goal2trials_df = goal2trials.apply(pd.Series)
    goal2trials_df = goal2trials_df.apply(_shuffle, axis=1)  # shuffle
    n_folds = goal2trials_df.shape[1] // test_trials_per_goal
    fold_dfs = []
    for i in range(n_folds):
        test_cols = goal2trials_df.columns[i * test_trials_per_goal : (i + 1) * test_trials_per_goal]
        train_cols = goal2trials_df.columns[~goal2trials_df.columns.isin(test_cols)]
        test_df = goal2trials_df.loc[:, test_cols]
        train_df = goal2trials_df.loc[:, train_cols]
        # construct fold df
        test_df.columns = pd.MultiIndex.from_product([[f"fold_{i}"], ["test"], test_df.columns])
        train_df.columns = pd.MultiIndex.from_product([[f"fold_{i}"], ["train"], train_df.columns])
        fold_dfs.append(pd.concat([test_df, train_df], axis=1))
    return pd.concat(fold_dfs, axis=1)


def get_trial_validation_folds_df(trials, splits=10):
    """
    Generates a DataFrame containing train and test trial indices for cross-validation.
    This function splits the trials from a given session into a specified number of folds for cross-validation.
    Each fold contains a set of test trials and the remaining trials are used for training. The resulting DataFrame
    has a MultiIndex with fold numbers and 'train'/'test' labels.
    Parameters:
    session (object): An object containing trial data, expected to have a 'trials_df' attribute which is a DataFrame.
    splits (int): The number of folds to split the trials into. Default is 10.
    trials (array-like, optional): A list or array of trial indices to be used. If None, all trials from the session
                                   will be used. Default is None.
    Returns:
    pd.DataFrame: A DataFrame with MultiIndex columns where each column represents a fold and whether the trials
                  are for training or testing. The DataFrame contains trial indices for each fold.
    """
    np.random.shuffle(trials)
    n_trials = len(trials)
    split_size = n_trials // splits
    fold_dfs = []
    for i in range(splits):
        test_trials = trials[i * split_size : (i + 1) * split_size]
        train_trials = trials[~np.isin(trials, test_trials)]
        test_df = pd.DataFrame(test_trials, columns=pd.MultiIndex.from_tuples([(f"fold_{i}", "test")]))
        train_df = pd.DataFrame(train_trials, columns=pd.MultiIndex.from_tuples([(f"fold_{i}", "train")]))
        fold_dfs.append(pd.concat([test_df, train_df], axis=1))
    return pd.concat(fold_dfs, axis=1)


# %% Downsampling functions


def downsample_navigation_activity_df(navigation_activity_df, window_length=0.1):
    """
    Reduces resolution of navigation_spike_counts_df, that natively expresses spike counts per frame, to spike counts
    per window (defined in seconds). Note the resultant dataframe counts spike in non-overlapping blocks and for convience
    takes the place, direction, distance_to_goal (and other variables) for the mid window index of the original data structure.
    This could introduce errors where eg, positions change within a window (don't worry about this for now).
    """
    combine_n_frames = int(FRAME_RATE * window_length)
    cols = navigation_activity_df.columns.get_level_values(0).unique()
    if "spike_count" in cols:
        activity_type = "spike_count"
    elif "firing_rate" in cols:
        activity_type = "firing_rate"
    else:
        raise ValueError("No activity type found in columns")
    # plit navigation cols from activity cols
    nav_df = navigation_activity_df.drop(columns=activity_type, level=0, axis=0)
    activity_df = navigation_activity_df.xs(activity_type, level=0, axis=1, drop_level=False)
    index = activity_df.index
    # downsample (sum spikes or average rates over frames)
    if activity_type == "firing_rate":
        ds_activity_df = activity_df.groupby(index // combine_n_frames).mean()
    elif activity_type == "spike_count":
        ds_activity_df = activity_df.groupby(index // combine_n_frames).sum()
    ds_activity_df.reset_index(drop=True, inplace=True)
    # take the corresponding navigation info for each downsampled window as the mid window index of the original data
    mid_window_indicies = (index // combine_n_frames).unique() * combine_n_frames + (combine_n_frames // 2)
    mid_window_indicies = mid_window_indicies[mid_window_indicies < len(nav_df)]
    ds_nav_info_df = nav_df.iloc[mid_window_indicies]
    ds_nav_info_df.reset_index(drop=True, inplace=True)
    # check if the activity df has one less row than the navigation df and remove the last row if so
    if ds_nav_info_df.shape[1] < ds_activity_df.shape[1]:
        ds_activity_df = ds_activity_df.iloc[:-1]
    # combine navigation info and activity info together again and return
    return pd.concat([ds_nav_info_df, ds_activity_df], axis=1)
