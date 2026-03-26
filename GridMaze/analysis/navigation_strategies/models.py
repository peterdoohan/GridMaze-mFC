"""
This moduel models choices during maze navigation as a function of different strategies
"""

# %% Imports
import sys
import numpy as np
from scipy.optimize import minimize
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.navigation_strategies import get_input_data as gid

# %% Global variables

INVALID_TRANSITION = -100
LOG_MAX_FLOAT = np.log(sys.float_info.max / 2.1)

NSEW = ["N", "S", "E", "W"]
# %% Modelling functions


def get_navigation_strategy_weights(
    navigation_strategies_df,
    strategies=["vector", "structure", "backtracking_penalty"],
    zscore=False,
):
    """
    Calculates the weight of each input strategy for explain subject's navigational
    decisions using maximum likelihood estimation.

    df should be generated from GridMaze.analysis.strategies.get_input_data.get_navigation_strategies_df
    using the same strategies as provided here.

    If zscore=True, each strategy's action values are z-scored (mean and std computed
    over available actions only) before fitting, making fitted weights directly comparable
    across strategies and conditions.
    """
    df = navigation_strategies_df.copy()
    if zscore:
        available = df["available"][NSEW].to_numpy(dtype=bool)
        for s in strategies:
            vals = df[s][NSEW].to_numpy(dtype=float)
            available_vals = vals[available]
            std = available_vals.std()
            if std > 0:
                df[s] = (df[s] - available_vals.mean()) / std
    # fit weights to data
    initial_weights = np.zeros(len(strategies))
    result = minimize(
        get_neg_loglikelihood,
        initial_weights,
        args=(strategies, df),
        method="BFGS",
    )
    return {s: w for s, w in zip(strategies, result.x)}


def get_neg_loglikelihood(weights, strategies, df):
    """
    Minimal generalisation: weights is an iterable of scalars, strategies is an iterable
    of matching strategy name strings. Assumes `INVALID_TRANSITION` and `softmax`
    are defined in the same scope as in your original code.
    """
    if len(weights) != len(strategies):
        raise ValueError("weights and strategies must have same length")
    # start with zeros and accumulate weighted strategy columns
    V = np.zeros((len(df), 4), dtype=float)
    for w, s in zip(weights, strategies):
        if s not in df.columns:
            raise KeyError(f"strategy '{s}' not found in input df")
        V += w * df[s][NSEW].to_numpy(dtype=float)

    # action availability handling (imposed by maze struct.)
    A_bool = df.available.to_numpy()
    A = np.where(A_bool, 0, INVALID_TRANSITION)
    V = V + A

    # subject choice mask
    choice_mask = df.subject_choice.to_numpy().astype(bool)
    P = softmax(V, choice_mask)

    loglikelihood = np.log(P)
    if np.any(np.isnan(loglikelihood)):
        raise ValueError("Log likelihood contains NaN(s).")
    return -np.sum(loglikelihood)


def softmax(V, choice_mask):
    """Calculates softmax probabilities for choices in a given state."""
    V[V > LOG_MAX_FLOAT] = LOG_MAX_FLOAT  # Protection against overflow in exponential.
    expV = np.exp(V)
    return expV[choice_mask] / np.sum(expV, axis=1)
