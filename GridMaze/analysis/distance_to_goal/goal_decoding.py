"""
Library for distance-to-goal alaigned goal decoding.
Eg, build separate decoders for neural activity 1, step from goal, 2 steps from goal, etc.
@peterdoohan
"""

# %% Imports
import numpy as np
import pandas as pd

from ..core import get_sessions as gs

# %% Global Variables

FRAME_RATE = 60
# %% Functions


def _downsample_place_shifted_rates(place_shifted_rates_df, window_length):
    """"""
    combine_n_frames = int(FRAME_RATE * window_length)
    nav_info_df = place_shifted_rates_df.drop(columns="firing_rate", level=0, axis=0)
    firing_rates_df = place_shifted_rates_df.xs("firing_rate", level=0, axis=1, drop_level=False)
    ds_firing_rates_df = firing_rates_df.groupby(firing_rates_df.index // combine_n_frames).mean()
    ds_firing_rates_df.reset_index(drop=True, inplace=True)
    mid_window_indicies = (firing_rates_df.index // combine_n_frames).unique() * combine_n_frames + (
        combine_n_frames // 2
    )
    mid_window_indicies = mid_window_indicies[mid_window_indicies < len(nav_info_df)]
    ds_nav_info_df = nav_info_df.iloc[mid_window_indicies]
    ds_nav_info_df.reset_index(drop=True, inplace=True)
    if ds_nav_info_df.shape[1] < ds_firing_rates_df.shape[1]:
        ds_firing_rates_df = ds_firing_rates_df.iloc[:-1]

    return pd.concat([ds_nav_info_df, ds_firing_rates_df], axis=1)


def _downsample_navigation_spike_counts(navigation_spike_counts_df, window_length):
    """
    Reduces resolution of navigation_spike_counts_df, that natively expresses spike counts per frame, to spike counts
    per window (defined in seconds). Note the resultant dataframe counts spike in non-overlapping blocks and for convience
    takes the place, direction, distance_to_goal (and other variables) for the mid window index of the original data structure.
    This could introduce errors where eg, positions change within a window (don't worry about this for now).
    """
    combine_n_frames = int(FRAME_RATE * window_length)
    nav_info_df = navigation_spike_counts_df.drop(columns="spike_count", level=0, axis=0)
    spike_counts_df = navigation_spike_counts_df.xs("spike_count", level=0, axis=1, drop_level=False)
    ds_spike_counts_df = spike_counts_df.groupby(spike_counts_df.index // combine_n_frames).sum()
    ds_spike_counts_df.reset_index(drop=True, inplace=True)
    mid_window_indicies = (spike_counts_df.index // combine_n_frames).unique() * combine_n_frames + (
        combine_n_frames // 2
    )
    mid_window_indicies = mid_window_indicies[mid_window_indicies < len(nav_info_df)]
    ds_nav_info_df = nav_info_df.iloc[mid_window_indicies]
    ds_nav_info_df.reset_index(drop=True, inplace=True)
    if ds_nav_info_df.shape[1] < ds_spike_counts_df.shape[1]:
        ds_spike_counts_df = ds_spike_counts_df.iloc[:-1]
    return pd.concat([ds_nav_info_df, ds_spike_counts_df], axis=1)
