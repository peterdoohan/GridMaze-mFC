"""
Come up with some way of visualising velocity tuning across the population
@ peterdoohan
"""

# %% Imports
from ast import Not
import pandas as pd
import numpy as np

from GridMaze.analysis.cluster_tuning import movement as mv
from GridMaze.analysis.core import get_sessions as gs

# %% Global Variables

FRAME_RATE = 60

# %% Functions


def get_population_velocity_tuning(late_session=False, verbose=True, sessions=None):
    """ """
    if sessions is None:
        if verbose:
            print("Loading sessions...")
        days_on_maze = "late" if late_session else "all"
        sessions = gs.get_maze_sessions(
            subject_IDs="all",
            days_on_maze=days_on_maze,
            maze_names="all",
            with_data=["cluster_movement_metrics", "navigation_df", "navigation_spike_rates_df"],
        )
    tuning_dfs, metric_dfs = [], []
    for session in sessions:
        if verbose:
            print(session.name)
        tuning_df, tuning_metrics = get_session_velocity_tuning(session, return_with_metrics=True)
        if tuning_df is np.nan:
            continue
        tuning_dfs.append(tuning_df)
        metric_dfs.append(tuning_metrics)

    return tuning_dfs, metric_dfs


def get_session_velocity_tuning(
    session,
    min_corr=0.6,
    x_range=(-0.3, 0.3),
    y_range=(-0.3, 0.3),
    bin_size=0.025,
    smooth_SD=False,
    min_occ=0.5,
    return_with_metrics=False,
):
    # load data
    navigation_df = session.navigation_df
    navigation_rates_df = session.navigation_spike_rates_df.reset_index(drop=True)
    movement_metrics = session.cluster_movement_metrics

    # filter clusters
    if min_corr is not None:
        movement_metrics = movement_metrics[movement_metrics.velocity.mean_corr.gt(min_corr)]
    keep_clusters = movement_metrics.cluster_unique_ID.values

    # if no velocity tuned cluster from session return nan
    if len(keep_clusters) == 0:
        if return_with_metrics:
            return np.nan, np.nan
        else:
            return np.nan

    navigation_rates_df = navigation_rates_df.firing_rate[keep_clusters]
    navigation_rates_df.columns = pd.MultiIndex.from_product([["firing_rate"], keep_clusters])
    navigation_rates_df = pd.concat([navigation_df, navigation_rates_df], axis=1)

    # get smoothed velocity data and update navigation df
    speeds, velocities, trang_acc = mv.get_movement_tuning_data(navigation_df)
    navigation_rates_df[("velocity", "x")] = velocities[:, 0]
    navigation_rates_df[("velocity", "y")] = velocities[:, 1]

    # bin velocity data
    x_bin_edges = np.arange(x_range[0], x_range[1] + bin_size, bin_size)
    y_bin_edges = np.arange(y_range[0], y_range[1] + bin_size, bin_size)
    navigation_rates_df[("velocity", "x_bin")] = pd.cut(
        navigation_rates_df[("velocity", "x")], bins=x_bin_edges, labels=(x_bin_edges[:-1] + bin_size / 2)
    )
    navigation_rates_df[("velocity", "y_bin")] = pd.cut(
        navigation_rates_df[("velocity", "y")], bins=y_bin_edges, labels=(y_bin_edges[:-1] + bin_size / 2)
    )

    # get tuning curves
    grouped_df = navigation_rates_df.groupby([("velocity", "x_bin"), ("velocity", "y_bin")], observed=True)
    tuning_df = grouped_df.firing_rate.mean().firing_rate  # (n_x bins x n_y_bins), n_clusters

    # optionally smooth tuning curves in 2D
    if smooth_SD:
        raise NotImplementedError()

    # remove low occ bins
    sub_min_occ = grouped_df.count().time.lt(min_occ * FRAME_RATE)
    tuning_df = tuning_df.mask(sub_min_occ)

    if return_with_metrics:
        return tuning_df, movement_metrics
    else:
        return tuning_df
