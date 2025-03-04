"""
Preprocessing library for extracting and processing LFP data. Converts raw ephys data from raw_data/ephys to 3 GridMaze processed
data structures: 1) lfp.signal.npy (float16, uV), 2) lfp.time (float64, s from start of session), 3) lfp.metrics.htsv (tabular).

@peterdoohan
"""

# %% Imports
import numpy as np
import pandas as pd
from spikeinterface import extractors as se
from spikeinterface import preprocessing as sp
from scipy.interpolate import interp1d
from . import pycontrol_data_import as di
from .rsync import Rsync_aligner
import probeinterface as pi

# %% Global variables

# %% Functions


def get_LFP_signal(
    session_dir,
    bandpass_max=450,
    downsample_frequency=1500,
):
    """
    Function to extract and process LFP data from raw open-ephys data.
    Steps: 1) Load raw LFP data with spikeinterface
           2) Subselect channels from neuropixel probe. Defualt = 8 which is every 80um vertically
           3) Bandpass filter LFP data (also de-means data)
           4) Downsample LFP data
           5) Convert LFP data to float16, units = uV
    Args:
        ephys_path (Path): Path to raw ephys data folder
        LFP_stream_name (str): Name of LFP stream in open-ephys data
        channel_downsample_factor (int): Factor to downsample channels by
        bandpass_max (int): Maximum frequency for bandpass filter
        downsample_frequency (int): Frequency to downsample LFP data to
    """
    # load data and configre probe with spike interface
    raw_rec = _load_recording(session_dir)
    bp_recording_LFP = sp.bandpass_filter(recording=raw_rec, freq_min=0.1, freq_max=bandpass_max)
    downsampled_LFP = sp.resample(recording=bp_recording_LFP, resample_rate=downsample_frequency)
    lfp_np32 = downsampled_LFP.get_traces(return_scaled=True)  # units = uV
    lfp_np16 = lfp_np32.astype(np.float16)  # minimal loss of precision while decreasing file size
    return lfp_np16


def get_LFP_times(session_dir, downsample_frequency):
    """
    Returns LFP times (associated with LFP signals) in seconds from the start of the pycontrol session.
    Args:
        ephys_path (Path): Path to raw ephys data folder
        LFP_stream_name (str): Name of LFP stream in open-ephys data
        LFP_timestamps_path (Path): Path to LFP timestamps file
        pycontrol_path (Path): Path to pycontrol data file
        downsample_frequency (int): Frequency to downsample LFP data to
    Returns:
        lfp_times (np.array): Array of LFP times in seconds from start of session
    """
    raw_rec = _load_recording(session_dir)
    original_sample_rate = int(raw_rec.get_sampling_frequency())  # Hz
    ephys_sync_pulse_times = np.load(session_dir.ephys_sync_path)[::2]  # timestamps in 2_500 Hz sample ids
    pycontrol_sync_pulse_times = di.Session(session_dir.pycontrol_path).times["rsync"]  # seconds
    rsync_aligner = Rsync_aligner(
        ephys_sync_pulse_times, pycontrol_sync_pulse_times, units_A=1 / original_sample_rate, units_B=1
    )
    lfp_times = rsync_aligner.A_to_B(np.arange(raw_rec.get_num_frames()), extrapolate=True)  # sample rate still 2500 Hz
    n_downsampled = int(len(lfp_times) * downsample_frequency / original_sample_rate)
    # downsample with linear interpolation
    interp_func = interp1d(np.arange(len(lfp_times)), lfp_times, kind="linear")
    lfp_times = interp_func(np.linspace(0, len(lfp_times) - 1, n_downsampled))
    return lfp_times


def get_LFP_metrics(session_dir, downsample_frequency):
    """
    Generates a dataframe of LFP metrics.
    """
    raw_rec = _load_recording(session_dir)
    probe = pi.get_probe(manufacturer="cambridgeneurotech", probe_name="ASSY-156-F")
    probe.wiring_to_device("cambridgeneurotech_mini-amp-64")
    probe_df = probe.to_dataframe()
    # select channels to keep (one from every column on each shank, two rows per shank)
    channels_to_keep = get_lfp_channels_to_keep(probe_df)
    probe_df = probe_df[probe_df.contact_ids.isin(channels_to_keep)]
    lfp_metrics_df = pd.DataFrame()
    lfp_metrics_df["channel_ind"] = probe_df["contact_ids"]
    lfp_metrics_df["channel_id"] = raw_rec.get_channel_ids()
    lfp_metrics_df["x_pos"] = probe_df["x"]
    lfp_metrics_df["y_pos"] = probe_df["y"]
    lfp_metrics_df["shank_id"] = probe_df["shank_ids"]
    lfp_metrics_df["sampling_rate"] = downsample_frequency
    return lfp_metrics_df


def _load_recording(session_dir):
    probe = pi.get_probe(manufacturer="cambridgeneurotech", probe_name="ASSY-236-F")
    probe.wiring_to_device("cambridgeneurotech_mini-amp-64")
    raw_rec = se.read_openephys(session_dir.ephys_data_path)
    raw_rec = raw_rec.set_probe(probe)
    return raw_rec


# %%


def get_lfp_channels_to_keep(probe_df):
    """
    Takes one column of contacts for each shank of the cambridge neurotech probes, returning the channel IDs to keep
    under this criteria
    """
    keep_channel_ids = []
    for shank in range(6):
        shank_df = probe_df[probe_df["shank_ids"] == str(shank)]
        x_pos_to_keep = shank_df.x.min()
        keep_channel_ids.extend(shank_df[shank_df.x == x_pos_to_keep].contact_ids)
    return keep_channel_ids
