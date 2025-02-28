"""
Library for generating navigation_spike_counts and navigation_rates analysis data structures that are paired with navigation_df
(see get_navigation_df.py)
"""

# %% Imports

import pandas as pd
import numpy as np
from scipy.stats import norm
from ..core import get_sessions as gs
from ..core import load_data
from ..core import convert

# %% Global Variables


# %% Functions


def get_navigation_spike_rates_df(processed_data_path, analysis_data_path):
    """
    Returns a dataframe with the firing rate of each neuron at each frame of video during a session.
    """
    try:
        session_info = load_data.load(processed_data_path / "session_info.json")
        navigation_aligned_activity = get_navigation_spike_rates(processed_data_path)
    except FileNotFoundError:
        print(f"Missing prerequist processed data to generate navigation_spike_rates_df for {processed_data_path}")
        return None
    cluster_rates = navigation_aligned_activity["navigation_spike_rates"].T
    cluster_IDs = navigation_aligned_activity["cluster_IDs"]
    unique_cluster_IDs = convert.cluster_IDs2scluster_unique_IDs(session_info, cluster_IDs)
    navigation_aligned_rates_df = pd.DataFrame(
        data=cluster_rates,
        index=navigation_aligned_activity["t_out"],
        columns=pd.MultiIndex.from_product([["firing_rate"], unique_cluster_IDs]),
    )
    navigation_aligned_rates_df.index.name = "time"
    return navigation_aligned_rates_df


def get_navigation_spike_counts_df(processed_data_path, analysis_data_path):
    """
    Returns a dataframe with the number of spikes that each neuron produces during a frame of video during a session.
    """
    try:
        session_info = load_data.load(processed_data_path / "session_info.json")
        navigation_aligned_activity = get_navigation_spike_counts(processed_data_path)
    except FileNotFoundError:
        print(f"Missing prerequist processed data to generate navigation_spike_counts_df for {processed_data_path}")
        return None
    cluster_spike_counts = navigation_aligned_activity["navigation_spike_counts"].T
    cluster_IDs = navigation_aligned_activity["cluster_IDs"]
    unique_cluster_IDs = convert.cluster_IDs2scluster_unique_IDs(session_info, cluster_IDs)
    navigation_aligned_spike_counts_df = pd.DataFrame(
        data=cluster_spike_counts,
        index=navigation_aligned_activity["t_out"],
        columns=pd.MultiIndex.from_product([["spike_count"], unique_cluster_IDs]),
    )
    navigation_aligned_spike_counts_df.index.name = "time"
    return navigation_aligned_spike_counts_df


def get_navigation_spike_rates(processed_data_path, smooth_SD="default", window_size=1000):
    """ """
    # Load processed data
    trajectories_df = load_data.load(processed_data_path / "frames.trajectories.htsv")
    spike_times = load_data.load(processed_data_path / "spikes.times.npy").reshape(-1)
    spike_clusters = load_data.load(processed_data_path / "spikes.clusters.npy").reshape(-1)
    frame_times = trajectories_df.time.to_numpy()
    frame_rate = round(len(frame_times) / (frame_times[-1] - frame_times[0]))
    if smooth_SD == "default":
        smooth_SD = 1 / frame_rate
    cluster_IDs = np.sort(np.unique(spike_clusters)).astype(np.float64)
    navigation_aligned_rates = np.ones((len(cluster_IDs), len(frame_times)))
    for i, cluster_ID in enumerate(cluster_IDs):  # loop over clusters
        cluster_spike_times = spike_times[spike_clusters == cluster_ID]
        for start_ind in range(0, len(frame_times), window_size):  # use sliding windows to handle memory usage
            end_ind = min(start_ind + window_size, len(frame_times))
            t_window = frame_times[start_ind:end_ind]
            spike_times_window = cluster_spike_times[
                (cluster_spike_times >= frame_times[start_ind]) & (cluster_spike_times <= frame_times[end_ind - 1])
            ]
            kernel_matrix = norm.pdf(t_window[:, None] - spike_times_window, scale=smooth_SD)
            navigation_aligned_rates[i, start_ind:end_ind] = np.sum(kernel_matrix, axis=1)
    return {
        "navigation_spike_rates": navigation_aligned_rates,
        "t_out": frame_times,
        "cluster_IDs": cluster_IDs,
    }


def get_navigation_spike_counts(processed_data_path):
    """ """
    # Load processed data
    trajectories_df = load_data.load(processed_data_path / "frames.trajectories.htsv")
    spike_clusters = load_data.load(processed_data_path / "spikes.clusters.npy").reshape(-1)
    spike_times = load_data.load(processed_data_path / "spikes.times.npy").reshape(-1)
    frame_times = trajectories_df.time.to_numpy()
    frame_rate = round(len(frame_times) / (frame_times[-1] - frame_times[0]))
    cluster_IDs = np.sort(np.unique(spike_clusters)).astype(np.float64)
    navigation_aligned_counts = np.ones((len(cluster_IDs), len(frame_times)), dtype=np.int32)
    for i, cluster_ID in enumerate(cluster_IDs):  # Loop over clusters
        cluster_spike_times = spike_times[spike_clusters == cluster_ID]
        start_frame_times = frame_times - (
            1 / (2 * frame_rate)
        )  # offset by half a frame to get spike counts within each frame
        cumsum_spike_counts = np.searchsorted(cluster_spike_times, start_frame_times)
        navigation_aligned_counts[i, :] = np.diff(
            cumsum_spike_counts, prepend=cumsum_spike_counts[0]
        )  # Count spikes within each frame
    return {
        "navigation_spike_counts": navigation_aligned_counts,
        "t_out": frame_times,
        "cluster_IDs": cluster_IDs,
    }
