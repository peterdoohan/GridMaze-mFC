"""Script for converting ephys/Kilosort spike times from ephys refernece to pycontorl reference"""

# %% Imports
import numpy as np
import pandas as pd
from .rsync import Rsync_aligner
from . import pycontrol_data_import as di

from spikeinterface import extractors as se

import probeinterface as pi
#%% Global variables

EPHYS_SAMPLING_RATE = 30_000 # Hz



# %%
def get_spike_pycontrol_times(session_dir):
    """Converts spike times from ephys reference to pycontrol reference"""
    spike_times = np.load(session_dir.kilosort_path /"spike_times.npy")  # in 30_000 Hz samples
    ephys_sync_pulse_times = np.load(session_dir.ephys_sync_path)[::2]  # timestamps in 30_000 Hz sample id
    pycontrol_sync_pulse_times = di.Session(session_dir.pycontrol_path).times["rsync"]/1000  # seconds
    spike_pycontrol_times = Rsync_aligner(
        ephys_sync_pulse_times, pycontrol_sync_pulse_times, units_A=1 / 30000, units_B=1,
    ).A_to_B(spike_times, extrapolate=True)
    return spike_pycontrol_times.reshape(-1)


def get_spike_clusters(session_dir):
    """Returns the spike cliusters corresponding to each spike times for a given session"""
    spike_clusters = np.load(session_dir.kilosort_path /  "spike_clusters.npy")
    return spike_clusters.reshape(-1)


def get_cluster_metrics(session_dir):
    """
    Returns a DataFrame with cluster metrics, including cluster ID, KSLabel, primary channel index and ID
    Add anatomical information ot this + more detailed quality metrics at a later date.
    """
    try:
        ks_metrics = pd.read_csv(session_dir.kilosort_path / "cluster_KSLabel.tsv", sep="\t")
        cluster_metrics = pd.DataFrame()
        cluster_IDs = ks_metrics.cluster_id
        cluster_metrics["cluster_ID"] = cluster_IDs
        cluster_metrics["KSLabel"] = ks_metrics.KSLabel
        # get cluster locations
        primary_channel_ind, primary_channel_ids = _get_primary_channel(session_dir)
        cluster_metrics["primary_channel_ind"] = primary_channel_ind
        cluster_metrics["primary_channel_ID"] = primary_channel_ids
        cluster_metrics["average_firing_rate"] = _get_av_firing_rates(cluster_IDs, session_dir)
    except ValueError:
        print(f"Could not calculate cluster metrics for {session_dir}")
        cluster_metrics = None
    return cluster_metrics

def _get_av_firing_rates(cluster_IDs, session_dir):
    """Returns the average firing rate for each cluster in a session"""
    spike_times = get_spike_pycontrol_times(session_dir)
    session_length = spike_times[-1] # seconds
    spike_clusters = get_spike_clusters(session_dir)
    cluster_av_rates = []
    for cluster in cluster_IDs:
        cluster_spikes = spike_times[spike_clusters == cluster]
        cluster_av_rates.append(len(cluster_spikes) / session_length)
    return cluster_av_rates


def _get_primary_channel(session_dir):
    """
    Returns the primary channel (channel where the waveform aplitude is maximum)
    index and ID for each cluster
    """
    templates = np.load(session_dir.kilosort_path / "templates.npy")
    primary_channel_ind = (templates**2).sum(axis=1).argmax(axis=-1)
    # get channel IDs using spike interface
    probe = pi.get_probe(manufacturer='cambridgeneurotech', probe_name="ASSY-156-F")
    probe.wiring_to_device('cambridgeneurotech_mini-amp-64')
    raw_rec = se.read_openephys(session_dir.ephys_data_path, block_index=0) #some sessions are multiblock? 
    raw_rec = raw_rec.set_probe(probe)
    primary_channel_ids = raw_rec.channel_ids[primary_channel_ind]
    return primary_channel_ind, primary_channel_ids

#Add anatomical information to the data once processed