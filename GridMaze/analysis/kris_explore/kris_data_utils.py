import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from ..core import get_sessions as gs
from ..core import filter as filt
from ...maze import representations as mr

def get_model_input_data(
    session,
    distance_metric="geodesic",
    include_multi_unit=False,
    navigation_only=True,
    moving_only=True,
    resolution=0.5,  # s
    max_distance=1.8,
    n_distance_bins=20,
):
    # get data
    navigation_spikes_df = session.get_navigation_activity_df(
        type="spikes", cluster_kwargs={"single_units": True, "multi_units": include_multi_unit}
    )
    if resolution:
        navigation_spikes_df = _downsample_navigation_spike_counts(navigation_spikes_df, resolution)
    # filter data
    navigation_spikes_df = filt.filter_navigation_rates_df(
        navigation_spikes_df, navigation_only=navigation_only, moving_only=moving_only, exclude_time_at_goal=False
    )
    navigation_spikes_df = _remove_invalid_frames(navigation_spikes_df, distance_metric)
    navigation_spikes_df = navigation_spikes_df[
        navigation_spikes_df[("distance_to_goal", distance_metric)].le(max_distance)
    ]
    distance_bins_col = ("distance_to_goal", distance_metric + "_binned")
    bins = pd.interval_range(start=0, end=max_distance, freq=max_distance / n_distance_bins, closed="left")
    navigation_spikes_df[distance_bins_col] = pd.cut(
        navigation_spikes_df[("distance_to_goal", distance_metric)],
        bins=bins,
    )
    dist_onehot = _dist_bin2onehot(navigation_spikes_df[distance_bins_col].values, bins)
    pd_by_frame = (
        navigation_spikes_df[("maze_position", "simple")]
        + "_"
        + navigation_spikes_df[("cardinal_movement_direction", "")]
    )  # convert to unique string eg, "A1_N", expected by onehot encoder
    place_direction_onehot = _place_direction2onehot(pd_by_frame.values, session.simple_maze())
    spike_counts = navigation_spikes_df.spike_count.values
    # organise output into Kris' expected dict
    return {
        "X": torch.from_numpy(np.concatenate([dist_onehot, place_direction_onehot], axis=1).T).to(torch.float32),
        "spikes": torch.from_numpy(spike_counts.T).to(torch.float32),
    }




def get_subject_input_data(subject, maze_name):
    sessions = gs.get_maze_sessions(
        subject_IDs=[subject],
        maze_names=[maze_name],
        days_on_maze="late",
        with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics"],
        must_have_data=True,
    )
    input_data = [get_model_input_data(s) for s in sessions]
    # add neuron inds
    ind = 0
    for data in input_data:
        n_clusters = data["spikes"].shape[0]
        data["inds"] = torch.from_numpy(np.arange(ind, ind + n_clusters)).to(torch.int32)
        ind += n_clusters
    return input_data



def _downsample_navigation_spike_counts(navigation_spike_counts_df, window_length, FRAME_RATE = 60):
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


def _remove_invalid_frames(navigation_spikes_df, distance_metric):
    """Where there are tracking errors, cardinal_movement_direction not defined (nan), remove
    these frames"""
    nan_mask = []
    for cols in [
        ("maze_position", "simple"),
        ("cardinal_movement_direction", ""),
        ("distance_to_goal", distance_metric),
    ]:
        nan_mask.append(navigation_spikes_df[cols].isna())
    return navigation_spikes_df[~np.logical_or.reduce(nan_mask)]


def _dist_bin2onehot(bins_by_frame, bins):
    enc = OneHotEncoder(categories=[bins], sparse_output=False)
    onehot = enc.fit_transform(bins_by_frame.reshape(-1, 1))
    return onehot



def _place_direction2onehot(pd_by_frame, simple_maze):
    all_place_direction_pairs = mr.get_maze_place_direction_pairs(simple_maze)
    all_place_direction_pairs = np.array(
        [x[0] + "_" + x[1] for x in all_place_direction_pairs], dtype=object
    )  # transform to unique string from tuples
    enc = OneHotEncoder(categories=[all_place_direction_pairs], sparse_output=False)
    onehot = enc.fit_transform(pd_by_frame.reshape(-1, 1))
    return onehot



