"""
Common set of functions for downsampling data for analysis.
Eg, you want to downsample navigating input data from FRAME RATe (native) to 0.5 window resolution.
"""

# %% Imports
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import circmean

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


def downsample_navigation_activity_df(df, resolution):
    """
    similar to downsample_nav_spikes_data but carefully downsamples all navigation variables
    """
    # split into navigation and spike dfs
    act_type = "spike_count" if "spike_count" in df.columns.get_level_values(0) else "firing_rate"
    spikes_df = df.xs(act_type, level=0, axis=1, drop_level=False)
    navigation_df = df.drop(columns=act_type, level=0, axis=1)

    # split navigation data into, continous, polar, and categorical
    polar_cols = [("head_direction", "value"), ("angle_to_goal", "allocentric"), ("angle_to_goal", "egocentric")]
    cont_cols = [
        c
        for i in [
            "time",
            "centroid_position",
            "velocity",
            "speed",
            "distance_to_goal",
            "progress_to_goal",
            "steps_to_goal",
        ]
        for c in navigation_df.xs(i, level=0, axis=1, drop_level=False).columns.to_list()
    ]
    cat_cols = [c for c in navigation_df.columns if c not in polar_cols and c not in cont_cols]

    # define downsample window
    combine_n_frames = int(FRAME_RATE * resolution)
    ds_frames = int(FRAME_RATE * resolution)
    if ds_frames < 1:
        raise ValueError("resolution must be >= 1/FRAME_RATE seconds")
    window_groups = navigation_df.index // ds_frames

    # downsample neurons
    if act_type == "spike_count":
        # sum spikes within res window
        ds_spikes_df = spikes_df.groupby(window_groups).sum().reset_index(drop=True)
    else:  # act_type == "firing_rate"
        # average firing rates within res window
        ds_spikes_df = spikes_df.groupby(window_groups).mean().reset_index(drop=True)

    # downsample categorical data by taking values in mid window
    mid_window_inds = window_groups.unique() * ds_frames + (ds_frames // 2)
    mid_window_inds = mid_window_inds[mid_window_inds < len(navigation_df)]
    ds_cat_df = navigation_df.iloc[mid_window_inds].reset_index(drop=True)

    # downsample continous data by taking mean of values in window
    ds_cont_df = navigation_df[cont_cols].groupby(window_groups).mean().reset_index(drop=True)

    # downsample polar variables by taking the circular mean
    ds_polar_df = navigation_df[polar_cols].groupby(window_groups).agg(lambda x: circmean(x, low=0, high=360))

    # 2) pull out the polar data & convert to radians
    rad = np.deg2rad(navigation_df[polar_cols].values)

    # 3) vectorized sin/cos
    sin_vals = np.sin(rad)
    cos_vals = np.cos(rad)

    # 4) group & mean (fast!)
    _polar_cols = pd.MultiIndex.from_tuples(polar_cols)
    sin_mean = pd.DataFrame(sin_vals, columns=_polar_cols).groupby(window_groups).mean()
    cos_mean = pd.DataFrame(cos_vals, columns=_polar_cols).groupby(window_groups).mean()

    # 5) back to angles in [0,360)
    ds_polar_df2 = np.rad2deg(np.arctan2(sin_mean, cos_mean)).mod(360)  # gives [-180,180]  # wrap to [0,360)

    return


def _polar_downsample(navigation_df, polar_cols, window_groups):
    rad = np.deg2rad(navigation_df[polar_cols].values)
    sin_vals = np.sin(rad)
    cos_vals = np.cos(rad)
    _polar_cols = pd.MultiIndex.from_tuples(polar_cols)
    sin_mean = pd.DataFrame(sin_vals, columns=_polar_cols).groupby(window_groups).mean()
    cos_mean = pd.DataFrame(cos_vals, columns=_polar_cols).groupby(window_groups).mean()
    return np.rad2deg(np.arctan2(sin_mean, cos_mean)).mod(360)  # gives [-180,180]  # wrap to [0,360)
