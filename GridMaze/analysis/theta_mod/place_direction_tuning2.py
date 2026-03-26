"""
Another idea for measuring theta mod place-direction tuning
@peterdoohan
"""

# %% Imports
import pandas as pd
import numpy as np

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import filter as filt

# %% Global variables

# %% Functions


def plot_theta_and_direction_stratified_heatmaps(session):
    """ """
    simple_maze = session.simple_maze()
    # get navigation spikes (theta strat.) df filtered for navigation moving etc.
    nav_spikes_df = _get_theta_nav_spikes_df(session, split_half_corr_thres, verbose=False)
    nav_spikes_df = nav_spikes_df.sort_index(axis=1)

    return


def get_direction_stratified_heatmaps():

    return


def get_theta_stratified_nav_spikes_df(
    session,
    split_half_corr_thres,
    moving_thres=0.05,
    verbose=True,
):
    # load data
    place_dir_metrics = session.cluster_place_direction_tuning_metrics.copy()
    navigation_df = session.navigation_df.copy()
    spikes_df = session.navigation_theta_spike_counts_df.reset_index(drop=True)
    # filter for clusters with strong pd tuning
    consider_clusters = place_dir_metrics[
        place_dir_metrics.split_half_corr.value.gt(split_half_corr_thres)
    ].index.values
    if len(consider_clusters) == 0:
        if verbose:
            print(f"No place-dir. tuned cluster for session: {session.name}")
        return None
    spikes_df = spikes_df[spikes_df.columns[spikes_df.columns.get_level_values(1).isin(consider_clusters)]]
    # combine spikes and nav data
    navigation_df.columns = pd.MultiIndex.from_tuples([(*col, "") for col in navigation_df.columns])
    nav_spikes_df = pd.concat([navigation_df, spikes_df], axis=1).copy()
    # filter data same as normal place-direction heatmaps
    nav_spikes_df = filt.filter_navigation_rates_df(
        nav_spikes_df, navigation_only=True, moving_only=False, exclude_time_at_goal=True, max_steps_to_goal=30
    )
    # apply custom movement threshold to keep as much data as possbible
    nav_spikes_df = nav_spikes_df[nav_spikes_df.speed.gt(moving_thres)]
    nav_spikes_df = nav_spikes_df.reset_index(drop=True).sort_index(axis=0)
    return nav_spikes_df
