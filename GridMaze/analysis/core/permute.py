"""
Core lib for permuting data for shuffle controls
"""

# %% Imports

import numpy as np
import pandas as pd

# %% Global variables


# %% Functions


def random_circular_shift(df):
    n = len(df)
    k = np.random.randint(0, n)
    # np.roll on the integer positions, then .iloc to reorder
    new_order = np.roll(np.arange(n), k)
    df_shifted = df.iloc[new_order]
    df_shifted.index = df.index
    return df_shifted


def shuffle_trials(df, trial_unique_ID=False):
    """ """
    _t = "trial_unique_ID" if trial_unique_ID else "trial"
    if isinstance(df, pd.DataFrame):
        trials = df[_t].dropna().unique()
        shuffled_trials = np.random.permutation(trials)
        df[_t] = df[_t].replace(trials, shuffled_trials)
        return df
    elif isinstance(df, list):
        trial_sets = [_df[_t].dropna().unique() for _df in df]
        t0 = np.unique(trial_sets[0])
        assert all(np.array_equal(t0, np.unique(a)) for a in trial_sets[1:]), "Trials are different across input dfs"
        shuffled_trials = np.random.permutation(t0)
        new_dfs = [i.copy() for i in df]
        for _df, new_df in zip(df, new_dfs):
            new_df[_t] = _df[_t].replace(t0, shuffled_trials)
        return new_dfs
    else:
        raise ValueError(f"Unknown type {type(df)}")
