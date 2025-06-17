"""
Library for visualising population tuning aligned to egocentric actions
"""

# %% Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d

from GridMaze.analysis.cluster_tuning import actions as act
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc

# %% Global Variables

# %% Functions


def test(session):
    navigation_rates_df = session.get_navigation_activity_df(type="rates", cluster_kwargs={"single_units": True})
    action_aligned_rates_df = act._get_basic_action_tuning(navigation_rates_df)

    return


def get_session_egocentric_action_tuning(
    session,
    actions=["turn_left", "turn_right", "go_forward"],
    action_type="all",
    min_split_half_corr=0.3,
    window=(-3, 3),
    normalise="zscore",
    smooth_SD=10,
):
    """ """
    # load data
    navigation_rates_df = session.get_navigation_activity_df(type="rates", cluster_kwargs={"single_units": True})
    egocentric_metrics_df = session.cluster_egocentric_action_tuning_metrics
    # keep only clusters with some egocentric aciton tuning
    keep_clusters = egocentric_metrics_df[
        egocentric_metrics_df.split_half_corr[f"{action_type}_action"].value.gt(min_split_half_corr)
    ].cluster_unique_ID.values
    all_clusters = navigation_rates_df.firing_rate.columns.to_numpy()
    reject_clusters = np.setdiff1d(all_clusters, keep_clusters)
    navigation_rates_df = navigation_rates_df.drop(columns=reject_clusters, level=1)
    # get action aligned rates
    action_aligned_rates = act._get_basic_action_tuning(navigation_rates_df, window=window)
    # filter for specified actions
    action_aligned_rates = action_aligned_rates[action_aligned_rates.basic_action.isin(actions)]
    # filter for specified action type
    if action_type == "all":
        pass
    elif action_type == "free":
        action_aligned_rates = action_aligned_rates[action_aligned_rates.choice_degree.gt(2)]
    elif action_type == "forced":
        action_aligned_rates = action_aligned_rates[action_aligned_rates.choice_degree.le(2)]
    else:
        raise ValueError(f"Action type {action_type} not recognised")
    # get tuning curves for each action
    tuning_curves = action_aligned_rates.groupby(
        ["cluster_unique_ID", "basic_action"]
    ).action_aligned_rates.mean()  # [clusters x actions, timepoints]
    # normalise tuning curves for each cluster
    if normalise == "zscore":
        # concat tuning curves [actions, clusters x timepoints]
        long_tuning_curves = tuning_curves.action_aligned_rates.unstack().swaplevel(0, 1, axis=1).sort_index(axis=1)
        # zscore each cluster
        long_tuning_curves = long_tuning_curves.apply(zscore, axis=1)
        # restack
        tuning_curves = long_tuning_curves.stack(level=[0], future_stack=True)
    else:
        raise ValueError(f"Normalisation {normalise} not recognised")
    # smooth tuning curves if specified
    if smooth_SD:
        rates = tuning_curves.values
        smoothed_rates = gaussian_filter1d(rates, smooth_SD, axis=1)
        tuning_curves = pd.DataFrame(
            smoothed_rates,
            index=tuning_curves.index,
            columns=tuning_curves.columns,
        )
    return tuning_curves
