"""
Preprocessing library for extracting and processing LFP data. Converts raw ephys data from raw_data/ephys to 3 GridMaze processed
data structures: 1) lfp.signal.npy (float16, uV), 2) lfp.time (float64, s from start of session), 3) lfp.metrics.htsv (tabular).

@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from pathlib import Path
from spikeinterface import extractors as se
from spikeinterface import preprocessing as sp
from scipy.interpolate import interp1d
from . import pycontrol_data_import as di
from .rsync import Rsync_aligner
import probeinterface as pi
from . import get_ephys_data as ed

# %% Global variables
from spikeinterface import core as si

si.set_global_job_kwargs(n_jobs=80, chunk_duration="1s", progress_bar=True)

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
    probe = raw_rec.get_probe()
    probe_df = probe.to_dataframe()
    contact_id2channel_label = (  # contact id in probe_df mapped to channel labels in raw_rec (non-fucking trivial)
        pd.DataFrame(raw_rec.get_property("contact_vector"))
        .set_index("contact_ids")
        .device_channel_indices.apply(lambda x: f"CH{x+1}")
    ).to_dict()
    bp_recording_LFP = sp.bandpass_filter(recording=raw_rec, freq_min=0.1, freq_max=bandpass_max)
    downsampled_LFP = sp.resample(recording=bp_recording_LFP, resample_rate=downsample_frequency)
    channels_to_keep = get_lfp_channels_to_keep(probe_df)  # "contact ids" from probe_df, not channel labels in raw_rec
    downchanneled_LFP = downsampled_LFP.channel_slice(
        [contact_id2channel_label[str(c)] for c in channels_to_keep]
    )  # convert contact ids to channel labels for slicing
    lfp_np32 = downchanneled_LFP.get_traces(return_scaled=True)  # units = uV
    lfp_np16 = lfp_np32.astype(np.float16)  # minimal loss of precision while decreasing file size
    return lfp_np16


def get_LFP_times(session_dir, downsample_frequency=1500):
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
    if session_dir.session_type == "rest":
        lfp_times = raw_rec.get_times()  # seconds
    else:  # "maze"
        ephys_sync_pulse_times = np.load(session_dir.ephys_sync_path)[::2]  # timestamps in 2_500 Hz sample ids
        pycontrol_sync_pulse_times = di.Session(session_dir.pycontrol_path).times["rsync"] / 1000  # seconds
        rsync_aligner = Rsync_aligner(
            ephys_sync_pulse_times,
            pycontrol_sync_pulse_times,
            units_A=(1 / original_sample_rate),
            units_B=1,
        )
        lfp_times = rsync_aligner.A_to_B(
            np.arange(raw_rec.get_num_frames()), extrapolate=True
        )  # sample rate still 2500 Hz
    n_downsampled = int(len(lfp_times) * downsample_frequency / original_sample_rate)
    # downsample with linear interpolation
    interp_func = interp1d(np.arange(len(lfp_times)), lfp_times, kind="linear")
    lfp_times = interp_func(np.linspace(0, len(lfp_times) - 1, n_downsampled))
    return lfp_times


def get_LFP_metrics(session_dir, downsample_frequency=1500):
    """
    Generates a dataframe of LFP metrics.
    """
    raw_rec = _load_recording(session_dir)
    probe = raw_rec.get_probe()
    probe_df = probe.to_dataframe()
    contact_id2channel_label = (  # contact id in probe_df mapped to channel labels in raw_rec (non-fucking trivial)
        pd.DataFrame(raw_rec.get_property("contact_vector"))
        .set_index("contact_ids")
        .device_channel_indices.apply(lambda x: f"CH{x+1}")
    ).to_dict()
    channel_label2contact_id = {v: k for k, v in contact_id2channel_label.items()}
    channel_assignments = _load_channel_assignments(session_dir)
    # map from channel labels to contact ids
    channel_assignments = {channel_label2contact_id[k]: v for k, v in channel_assignments.items()}
    probe_anatomy_df = ed.load_subject_probe(session_dir.subject_ID)
    tissue_sample = ed._get_tissue_sample(session_dir.subject_ID, session_dir.date)
    probe_anatomy_df = probe_anatomy_df[probe_anatomy_df.tissue_sample == tissue_sample]
    channels_to_keep = get_lfp_channels_to_keep(probe_df)  # "contact ids" so can map directly to contact id in probe_df
    lfp_metrics_df = probe_anatomy_df[probe_anatomy_df.contact.id.isin(channels_to_keep)].copy()
    lfp_metrics_df.loc[:, ("contact", "qc")] = lfp_metrics_df.contact.id.map(channel_assignments)
    lfp_metrics_df.loc[:, ("sampling_rate", "")] = downsample_frequency
    return lfp_metrics_df


# %% supporting functions


def _load_channel_assignments(session_dir):
    """
    Correct instances where channel assignments start from 65 instead of 1.
    Consistant with _load_recording function
    """
    # load channel assignmnets from spikesorting (see code/SpikeSorting/spikesort_sessions.py)
    with open(Path(session_dir.spikesorting_path) / "channel_assignments.json", "r") as file:
        channel_assigments = json.load(file)
    # check channel IDs start from 1 (sometimes start from 65, fault in raw recording)
    # correct consistent with _load_recordings function below.
    raw_rec = se.read_openephys(session_dir.ephys_data_path, block_index=0)
    channels = [c for c in raw_rec.get_channel_ids() if "CH" in c]  # exclude accelerometer channels (AUX1, etc.)
    if "CH1" not in channels:
        n_channels = len(channels)  # should be 64
        channel_name_corrections = {f"CH{i}": f"CH{i-n_channels}" for i in [int(c.split("CH")[1]) for c in channels]}
        channel_assigments = {channel_name_corrections[k]: v for k, v in channel_assigments.items()}
    return channel_assigments


def _load_recording(session_dir):
    """
    Note checks for channel IDs starting from 1, if not corrects to start from 1.
    (overwrites spikeinterface probe and recording object)
    """
    probe = pi.get_probe(manufacturer="cambridgeneurotech", probe_name="ASSY-236-F")
    probe.wiring_to_device("cambridgeneurotech_mini-amp-64")
    raw_rec = se.read_openephys(session_dir.ephys_data_path, block_index=0)
    channel_IDs = raw_rec.channel_ids
    if "CH1" not in channel_IDs:  # sometimes channel IDs start from 65? fix this to set probe correclty
        new_channel_IDs = [f"CH{i}" for i in np.arange(1, raw_rec.get_num_channels() + 1)]
        raw_rec = raw_rec.channel_slice(channel_IDs, new_channel_IDs)
    raw_rec = raw_rec.set_probe(probe)
    return raw_rec


def get_lfp_channels_to_keep(probe_df):
    """
    Takes one column of contacts for each shank of the cambridge neurotech probes, returning the channel IDs to keep
    under this criteria
    output is actually probe_df "contact_ids"
    """
    keep_channel_ids = []
    for shank in range(6):
        shank_df = probe_df[probe_df["shank_ids"] == str(shank)]
        x_pos_to_keep = shank_df.x.min()
        keep_channel_ids.extend(shank_df[shank_df.x == x_pos_to_keep].contact_ids)
    return [int(c) for c in keep_channel_ids]
