"""
Process preprocessed ephys data with spikeinterface into easy to work with GridMaze processed data :)
"""

# %% Imports
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date
from GridMaze.preprocessing.rsync import Rsync_aligner
from GridMaze.preprocessing import pycontrol_data_import as di

# %% Global Variables
AP_SAMPLING_RATE = 30_000

from GridMaze.paths import PROCESSED_DATA_PATH, EXPERIMENT_INFO_PATH

PROBE_DEPTHS_DF = pd.read_csv(EXPERIMENT_INFO_PATH / "probe_depths.htsv", sep="\t")


# %% Main Functions
def get_spike_times(session_dir):
    """Converts spike times from ephys reference to pycontrol reference"""
    internal_ks_path = Path(session_dir.spikesorting_path) / "kilosort4/sorter_output"
    spike_times = np.load(internal_ks_path / "spike_times.npy")  # in 30_000 Hz samples
    if session_dir.session_type == "rest":
        # for rest sessions we do not need to align time relative to pycontrol
        # just convert from units = min
        spike_times = spike_times / (AP_SAMPLING_RATE * 60)
    else:
        ephys_sync_pulse_times = np.load(session_dir.ephys_sync_path)[::2]  # timestamps in 30_000 Hz sample id
        pycontrol_sync_pulse_times = di.Session(session_dir.pycontrol_path).times["rsync"] / 1000  # seconds
        spike_pycontrol_times = Rsync_aligner(
            ephys_sync_pulse_times,
            pycontrol_sync_pulse_times,
            units_A=1 / AP_SAMPLING_RATE,
            units_B=1,
        ).A_to_B(spike_times, extrapolate=True)
        spike_times = spike_pycontrol_times
    return spike_times


def get_spike_clusters(session_dir):
    """Retrieves spike_clusters output from kilosort"""
    kilosort_path = Path(session_dir.spikesorting_path) / "kilosort4/sorter_output"
    spike_clusters = np.load(kilosort_path / "spike_clusters.npy")
    return spike_clusters


def get_cluster_metrics(
    session_dir,
    keep_metrics=[
        "unit_id",
        "presence_ratio",
        "firing_rate",
        "isi_violations_ratio",
        "amplitude_cutoff",
        "amplitude_median",
        "sd_ratio",
    ],
):
    """ """
    quality_metrics_path = Path(session_dir.spikesorting_path) / "quality_metrics.htsv"
    quality_metrics_df = pd.read_csv(quality_metrics_path, sep="\t")
    # get single_unit, multi_unit and noise_unit labels
    single_units = _is_single_unit(quality_metrics_df)
    noise_units = _is_noise_unit(quality_metrics_df)
    multi_units = np.logical_and(  # define mutli unit activity as sorted clusters that are not noise or single units
        ~single_units, ~noise_units
    )
    # process quality metrics
    quality_metrics_df.rename(columns={"unit_id": "cluster_ID"}, inplace=True)
    quality_metrics_df.set_index("cluster_ID", inplace=True)
    quality_metrics_df.drop(columns=np.setdiff1d(quality_metrics_df.columns, keep_metrics), inplace=True)
    quality_metrics_df.columns = pd.MultiIndex.from_product([["quality_metrics"], quality_metrics_df.columns])
    quality_metrics_df[("single_unit", "")] = single_units
    quality_metrics_df[("multi_unit", "")] = multi_units
    quality_metrics_df[("noise_unit", "")] = noise_units
    # process anatomy
    probe_df = load_subject_probe(session_dir.subject_ID)
    tissue_sample = _get_tissue_sample(session_dir.subject_ID, session_dir.date)
    session_probe_df = probe_df[probe_df.tissue_sample == tissue_sample]
    primary_channels = _get_primary_channel(session_dir)
    cluster_anatomy_df = session_probe_df.set_index(("contact", "id")).loc[primary_channels].reset_index()
    # combine
    cluster_metrics_df = pd.concat([quality_metrics_df.reset_index(), cluster_anatomy_df], axis=1)
    return cluster_metrics_df


# %% supporitng cluster metrics
def load_subject_probe(subject_ID):
    """
    Loads subject probe information containing anatomical information about each contact
    at different timepoints in the experiment
    """
    # load from subject folder
    probe_df = pd.read_csv(PROCESSED_DATA_PATH / subject_ID / "probe.htsv", sep="\t")
    # add multindex
    probe_df.columns = pd.MultiIndex.from_tuples(
        [tuple(c.split(".")) if "." in c else (c, "") for c in probe_df.columns]
    )
    return probe_df


def _get_tissue_sample(subject_ID, _date):
    df = PROBE_DEPTHS_DF.copy()
    df["date"] = df.date.apply(date.fromisoformat)
    subject_df = df[(df["subject"] == subject_ID) & (df["date"] <= _date)]
    # Get the row with the latest date (i.e. the most recent measurement)
    latest_row = subject_df.sort_values("date", ascending=False).iloc[0]
    return latest_row.tissue_sample


# %% supporting functions


def _is_single_unit(
    quality_metric_df,
    isi_violations_ratio_thres=0.1,
    amplitude_cutoff_thres=0.1,
    firing_rate_thres=0.05,
    presence_ratio_thres=0.8,
    amplitude_median_thres=30,
):
    """
    Returns boolian list of whether cluster passes single unit QC
    """
    query = f"amplitude_cutoff < {amplitude_cutoff_thres} and firing_rate > {firing_rate_thres} and presence_ratio > {presence_ratio_thres} and amplitude_median > {amplitude_median_thres} and isi_violations_ratio < {isi_violations_ratio_thres}"
    qc_pass_df = quality_metric_df.query(query)
    return quality_metric_df.unit_id.isin(qc_pass_df.unit_id.values).to_numpy()


def _is_noise_unit(quality_metric_df, amplitude_median_thres=20, firing_rate_thres=0.049):
    """
    Defines kilosort clusters as noise if they have low waveform template amplitude
    and average firing rate
    """
    noise_clusters_df = quality_metric_df.query(
        f"amplitude_median < {amplitude_median_thres} or firing_rate < {firing_rate_thres}"
    )
    return quality_metric_df.unit_id.isin(noise_clusters_df.unit_id.values).to_numpy()


def _get_primary_channel(session_dir):
    """
    Returns the primary channel (channel where the waveform aplitude is maximum)
    index and ID for each cluster
    """
    kilosort_path = Path(session_dir.spikesorting_path) / "kilosort4/sorter_output"
    templates = np.load(kilosort_path / "templates.npy")
    channel_map = np.load(kilosort_path / "channel_map.npy")
    channel_map += 1  # index from 1 to be consistant with probe_anatomy_df
    primary_channel = (templates**2).sum(axis=1).argmax(axis=-1)
    primary_channel = channel_map[primary_channel]
    return primary_channel
