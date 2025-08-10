"""
Can we decode the goal between if we known about theta phase. Either train and test on different theta-phases
OR do we need to know all theta phases to decode the goal?
@peterdoohan
"""

# %% Imports
import pandas as pd
import numpy as np
import networkx as nx

from GridMaze.maze import representations as mr
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import convert


# %% Globs

FRAME_RATE = 60

# %%


def get_session_theta_phase_goal_decoding():
    return


def get_session_theta_mod_goal_decoding():
    """
    Within session compare goal decoding from feature = n_neurons OR
    features = n_neurons x n_theta_phases. Need xval per fold opt regularisation
    bc of different number of features
    """
    return


# %% Functions


def get_input_data(
    session,
    event="cue",
    resolution=0.5,
    window=(-10, 10),
    include_multi_units=True,
):
    """
    Returns a dataframe with spike counts aligned to event (cue & reward) times.
    """
    # load data
    session_info = session.session_info
    trials_df = session.trials_df
    navigation_df = session.navigation_df
    theta_spike_counts_df = session.navigation_theta_spike_counts_df.reset_index(drop=True)
    # filter clusters
    keep_clusters = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=include_multi_units,
    )
    theta_spike_counts_df = theta_spike_counts_df[
        theta_spike_counts_df.columns[theta_spike_counts_df.columns.get_level_values(1).isin(keep_clusters)]
    ]
    # get rates aligned to event
    frames_before, frames_after = int(window[0] * FRAME_RATE), int(window[1] * FRAME_RATE)
    event_times = trials_df.set_index("trial").time[event]
    trial2goal = trials_df.set_index("trial").goal
    nav_info_dfs, spike_count_dfs = [], []
    for trial, event_time in event_times.items():
        event_frame = (navigation_df.time - event_time).abs().argmin()
        nav_aligned_df = navigation_df.iloc[event_frame + frames_before : event_frame + frames_after].reset_index(
            drop=True
        )
        spikes_aligned_df = theta_spike_counts_df.iloc[
            event_frame + frames_before : event_frame + frames_after
        ].reset_index(drop=True)
        # downsample to speficied resolution
        ds_nav_aligned_df, ds_spikes_aligned_df = ds.downsample_nav_spikes_data(
            nav_aligned_df, spikes_aligned_df, resolution
        )
        # add event aligned time info
        timepoints = np.arange(window[0], window[1], resolution)
        if len(timepoints) > ds_nav_aligned_df.shape[0]:
            # can happen for last trial in session (no more frames)
            timepoints = timepoints[: ds_nav_aligned_df.shape[0]]
        ds_nav_aligned_df[("event_aligned_time", event)] = timepoints
        # update distnace outside navigation where they are not defined (use shortest path
        # upcoming goal (event=cue) or shortest path to just visted goal (event=reward))
        ds_nav_aligned_df[("goal", "")] = trial2goal[trial]
        # update trial info so it is consistent across all aligned times
        ds_nav_aligned_df[("trial", "")] = trial
        ds_nav_aligned_df[("trial_unique_ID", "")] = convert.trial2trial_unique_ID(session_info, trial)
        nav_info_dfs.append(ds_nav_aligned_df)
        spike_count_dfs.append(ds_spikes_aligned_df)
    # combine over trials
    nav_info_df = pd.concat(nav_info_dfs, axis=0).reset_index(drop=True)
    spike_count_df = pd.concat(spike_count_dfs, axis=0).reset_index(drop=True)
    # combine nav_info and spike counts
    nav_info_df.columns = pd.MultiIndex.from_tuples([(*c, "") for c in nav_info_df.columns])
    event_aligned_nav_rates_df = pd.concat([nav_info_df, spike_count_df], axis=1)
    return event_aligned_nav_rates_df
