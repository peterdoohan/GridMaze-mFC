"""
Create analysis data structure with neuron spikes counted in per theta phase bin
"""

# %% Imports
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert

from GridMaze.analysis.core import load_data
from GridMaze.analysis.core import convert
from GridMaze.analysis.event_aligned import lfp_utils as lu

# %% Global Variables
FS_LFP = 1500

THETA_RANGE = (7, 11)

_4HZ_RANGE = (2, 5)  # 4Hz oscillation range

FRAME_RATE = 60  # Hz
# %% Functions


def get_navigation_theta_spike_counts_df(processed_data_path, n_bins=12):
    """
    see _get_navigation_lfp_phase_binned_spike_counts_df
    """
    return _get_navigation_lfp_phase_binned_spike_counts_df(processed_data_path, lfp_phase="theta", n_bins=n_bins)


def get_navigation_4Hz_spike_counts_df(processed_data_path, n_bins=12):
    """
    see _get_navigation_lfp_phase_binned_spike_counts_df
    """
    return _get_navigation_lfp_phase_binned_spike_counts_df(processed_data_path, lfp_phase="4Hz", n_bins=n_bins)


def _get_navigation_lfp_phase_binned_spike_counts_df(processed_data_path, lfp_phase="theta", n_bins=12):
    """ """
    try:
        session_info = load_data.load(processed_data_path / "session_info.json")
        lfp_phase_binned_spike_counts = get_lfp_phase_binned_spike_counts(
            processed_data_path, lfp_phase=lfp_phase, n_bins=n_bins
        )
    except FileNotFoundError:
        print(
            f"Missing prerequist processed data to generate navigation_lfp_phase_binned_spike_counts_df for {processed_data_path}"
        )
        return None
    # convert to DataFrame
    cluster_phase_spike_counts = lfp_phase_binned_spike_counts[
        "aligned_spike_counts"
    ].T  # [n_frames, n_bins, n_clusters]
    cluster_phase_spike_counts = cluster_phase_spike_counts.reshape(
        cluster_phase_spike_counts.shape[0], -1
    )  # [n_frames, n_bins * n_clusters]
    cluster_IDs = lfp_phase_binned_spike_counts["cluster_IDs"]
    cluster_unique_IDs = convert.cluster_IDs2scluster_unique_IDs(session_info, cluster_IDs)
    phase_bin_edges = lfp_phase_binned_spike_counts["bin_edges"]
    phase_bin_centers = (phase_bin_edges[:-1] + phase_bin_edges[1:]) / 2
    time = lfp_phase_binned_spike_counts["t_out"]
    df = pd.DataFrame(
        data=cluster_phase_spike_counts,
        index=time,
        columns=pd.MultiIndex.from_product(
            [["spike_count"], phase_bin_centers, cluster_unique_IDs],
            names=["", f"{lfp_phase}_phase_bin", "cluster_unique_ID"],
        ),
    )
    df = df.swaplevel(2, 1, axis=1).sort_index(axis=1)
    return df


# %% combine spikes and lfp phase aligned to video frames


def get_lfp_phase_binned_spike_counts(processed_data_path, lfp_phase="theta", n_bins=12):
    """ """
    # load spike data
    spike_clusters = load_data.load(processed_data_path / "spikes.clusters.npy").reshape(-1)
    spike_times = load_data.load(processed_data_path / "spikes.times.npy").reshape(-1)
    cluster_IDs = np.sort(np.unique(spike_clusters)).astype(np.float64)

    # get lfp phase data
    lfp_signal = get_LFP(processed_data_path)
    lfp_times = load_data.load(processed_data_path / "lfp.times.npy")
    if lfp_phase == "theta":
        lfp_phase = get_lfp_phase(lfp_signal, freq_range=THETA_RANGE, N=4)
    elif lfp_phase == "4Hz":
        lfp_phase = get_lfp_phase(lfp_signal, freq_range=_4HZ_RANGE, N=3)
    else:
        raise ValueError("lfp_phase must be 'theta' or '4Hz'")
    bin_edges, lfp_phase_bins = bin_lfp_phase(lfp_phase, n_bins=n_bins)

    # load frame times data
    trajectories_df = load_data.load(processed_data_path / "frames.trajectories.htsv")
    frame_times = trajectories_df.time.to_numpy()
    start_frame_times = frame_times - (  # offset by half a frame to get spike counts within each frame
        1 / (2 * FRAME_RATE)
    )
    # get spike counter per cluster x lfp_phase in each video frame
    aligned_spike_counts = np.zeros((len(cluster_IDs), n_bins, len(frame_times)), dtype=np.int32)
    for i, cluster_id in enumerate(cluster_IDs):
        cluster_spike_times = spike_times[spike_clusters == cluster_id]
        # get the lfp phase bin for each spike
        spike_lfp_idx = np.searchsorted(lfp_times, cluster_spike_times, side="left")
        spike_lfp_idx = np.clip(spike_lfp_idx - 1, 0, len(lfp_times) - 1)  # step back to the last sample ≤ spike time
        spike_phase_bins = lfp_phase_bins[spike_lfp_idx]
        # count spikes in each bin for each frame
        for bin_idx in range(n_bins):
            spike_times_in_bin = cluster_spike_times[spike_phase_bins == bin_idx]
            # count spikes in each frame
            cumsum_spike_counts = np.searchsorted(spike_times_in_bin, start_frame_times)
            aligned_spike_counts[i, bin_idx, :] = np.diff(cumsum_spike_counts, prepend=cumsum_spike_counts[0])
    return {
        "aligned_spike_counts": aligned_spike_counts,  # shape (n_clusters, n_bins, n_frames)
        "t_out": frame_times,  # time of each frame
        "bin_edges": bin_edges,  # edges of the phase bins
        "cluster_IDs": cluster_IDs,  # unique cluster IDs
    }


# %% LFP wrangling functions


def bin_lfp_phase(lfp_phase, n_bins=12):
    """bin osc phases from filtered lfp into n bins"""
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_indices = np.digitize(lfp_phase, bin_edges[1:-1])
    return bin_edges, bin_indices


def get_lfp_phase(
    lfp_signal,
    freq_range=(7, 11),
    N=4,
):
    """
    only for LFP (currently no CSD)
    """
    # filter for input frequency range
    nyq = FS_LFP / 2
    b, a = butter(N, [(freq_range[0] / nyq), (freq_range[1] / nyq)], btype="bandpass")
    filt_osc = filtfilt(b, a, lfp_signal)
    analytic = hilbert(filt_osc)
    phase_hilbert = np.angle(analytic)
    return phase_hilbert


def get_LFP(processed_data, shank=3, single_channel=False, remove_artifacts=True):
    """
    copy from event_aligned/lfp_units.py
    to work straight from processed data
    """
    # load data
    lfp_metrics = load_data.load(processed_data / "lfp.metrics.htsv")
    cluster_metrics = load_data.load(processed_data / "clusters.metrics.htsv")
    lfp_signal = load_data.load(processed_data / "lfp.signal.npy")
    if single_channel:  # converting from channel_id to channel_index in lfp_signal to get the signal
        channel_id = lu._get_single_channel_for_LFP(lfp_metrics, cluster_metrics, shank)
        channel_index = lfp_metrics[lfp_metrics.contact.id == channel_id].index[0]
        lfp = lfp_signal[:, channel_index]
    else:  # average lfp over multiple channels
        channel_ids = lu._get_shank_channels_for_LFP(lfp_metrics, cluster_metrics, shank)
        channel_indices = lfp_metrics.contact.id.isin(channel_ids).index.values
        lfp = lfp_signal[:, channel_indices].mean(axis=1)
    if remove_artifacts:
        lfp = lu._remove_artifacts(lfp, thres=500)
    return lfp
