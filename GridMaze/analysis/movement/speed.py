"""
Basic characterisation of speed tuning across the mFC population
"""

# %% Imports
import numpy as np
import pandas as pd

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.cluster_tuning import movement as mv

# %% Globs


# %% functions


def get_population_speed_tuning(late_session=False, verbose=True):
    # load sessions
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
            tuning_df, tuning_metrics = get_session_speed_tuning(session, return_with_metrics=True)
            tuning_dfs.append(tuning_df)
            metric_dfs.append(tuning_metrics)
    pop_tuning_df = pd.concat(tuning_dfs, axis=0)
    pop_metrics_df = pd.concat(metric_dfs, axis=0)
    return pop_tuning_df, pop_metrics_df


def get_session_speed_tuning(
    session,
    single_units=True,
    min_corr=0.8,
    speed_range=(0, 0.3),
    speed_bin_size=0.01,
    return_with_metrics=False,
):
    """ """
    # load data
    navigation_df = session.navigation_df
    navigation_rates_df = session.navigation_spike_rates_df.reset_index(drop=True)
    movement_metrics = session.cluster_movement_metrics
    # filter clusters
    if single_units:
        movement_metrics = movement_metrics[movement_metrics.single_unit]
    movement_metrics = movement_metrics[movement_metrics.speed.mean_corr.gt(min_corr)]
    keep_clusters = movement_metrics.cluster_unique_ID.values
    navigation_rates_df = navigation_rates_df.firing_rate[keep_clusters]
    navigation_rates_df.columns = pd.MultiIndex.from_product([["firing_rate"], keep_clusters])
    navigation_rates_df = pd.concat([navigation_df, navigation_rates_df], axis=1)
    # get smooted movement data and update navigation df
    speeds, _ = mv.get_movement_tuning_data(navigation_df)
    navigation_rates_df[("speed", "")] = speeds
    # get tuning (average fr per speed bin)
    stat_bin_edges = np.arange(speed_range[0], speed_range[1] + speed_bin_size, speed_bin_size)
    navigation_rates_df[("speed_binned", "")] = pd.cut(
        navigation_rates_df[("speed", "")], bins=stat_bin_edges, labels=(stat_bin_edges[:-1] + speed_bin_size / 2)
    )
    tuning_df = navigation_rates_df.groupby("speed_binned", observed=True).firing_rate.mean().firing_rate.T
    if return_with_metrics:
        return tuning_df, movement_metrics
    else:
        return tuning_df
