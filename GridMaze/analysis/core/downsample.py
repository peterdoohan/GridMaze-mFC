"""
Common set of functions for downsampling data for analysis.
Eg, you want to downsample navigating input data from FRAME RATe (native) to 0.5 window resolution.
"""

# %% Imports
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# %% Global Variables
FRAME_RATE = 60  # Hz

# %% Functions


def downsample_nav_spikes_data(
    navigation_df, spike_counts_df, resolution=0.2, distance_metrics=[("steps_to_goal", "future")]
):
    """ """
    # downsample spike counts by suming spikes within resolution window
    ds_frames = int(FRAME_RATE * resolution)
    if ds_frames < 1:
        raise ValueError("resolution must be >= 1/FRAME_RATE seconds")
    ds_spike_counts_df = spike_counts_df.groupby(spike_counts_df.index // ds_frames).sum().reset_index(drop=True)
    # keep only relevant navigation info
    nav_info = navigation_df[
        [
            ("time", ""),
            ("trial_unique_ID", ""),
            ("trial", ""),
            ("goal", ""),
            ("trial_phase", ""),
            ("moving", ""),
            ("maze_position", "simple"),
            ("cardinal_movement_direction", ""),
        ]
    ]
    # downsample navigation info by taking values in mid window
    mid_window_inds = (spike_counts_df.index // ds_frames).unique() * ds_frames + (ds_frames // 2)
    mid_window_inds = mid_window_inds[mid_window_inds < len(nav_info)]
    nav_info = nav_info.iloc[mid_window_inds]
    nav_info.reset_index(drop=True, inplace=True)
    # downsample continous nav variables by taking mean of values in window
    nav_cont = navigation_df[[*distance_metrics, ("speed", "")]]
    ds_nav_cont = nav_cont.groupby(nav_cont.index // ds_frames).mean()
    ds_nav_cont.reset_index(drop=True, inplace=True)
    # add to nav_info
    nav_info = pd.concat([nav_info, ds_nav_cont], axis=1)
    # account for differences in ds methods
    if nav_info.shape[0] < ds_spike_counts_df.shape[0]:
        ds_spike_counts_df = ds_spike_counts_df.iloc[:-1]
    return nav_info, ds_spike_counts_df
