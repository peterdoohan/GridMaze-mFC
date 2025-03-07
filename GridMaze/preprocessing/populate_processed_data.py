"""This module populates processes pycontrol, video and raw ephys data into processed data in the processed data folder"""

# %% Imports
import numpy as np
import json
from pathlib import Path
from .get_data_directory import get_sessions_data_directory
from .get_session_info import get_session_info
from .get_pycontrol_dfs import get_events_df, get_trials_df
from . import get_frames_dfs as fd
from . import get_ephys_data as ed
from . import get_lfp_data as ld
from . import get_UnitMatch_data as gud

from concurrent.futures import ProcessPoolExecutor

# %% Global Variables
from ..paths import PROCESSED_DATA_PATH

if not PROCESSED_DATA_PATH.exists():
    PROCESSED_DATA_PATH.mkdir()


# %% Updated functions


def populate_subject_probes(overwrite=False):
    """
    Populate probe.htsv files in subject processed data folders.
    Note this needs to be run in an environemtn with allensdk installed,
    separate to the main environment defined in requirements.txt
    """
    from GridMaze.preprocessing import probe_fit as pf

    pf.save_subject_probe_dfs(overwrite)
    return


def populate_session_processed_data():
    """ """

    return


def populate_processed_data(data_streams=["session_info", "pycontrol", "video", "spikes", "lfp"], overwrite=False):
    """
    Top level function for populating processed data for entire experiment.
    Args:
        data_streams (list): List of data streams to populate. Options are ["session_info", "pycontrol", "video", "spikes", "lfp"]
        overwrite (bool): If True, will overwrite existing processed data files
    """
    data_stream2fn = {
        "session_info": _populate_session_info,
        "pycontrol": _populate_pycontrol_data,
        "video": _populate_video_data,
        "spikes": _populate_spike_data,
        "lfp": _populate_lfp_data,
        "unit_match": _populate_unit_match_data,
    }
    sessions_data_directory = get_sessions_data_directory()
    for data_stream in data_streams:
        print(f"Processing {data_stream} data")
        preprocessing_fn = data_stream2fn[data_stream]
        for session_dir in sessions_data_directory.itertuples():
            processed_data_folder = (
                PROCESSED_DATA_PATH
                / session_dir.subject_ID
                / (session_dir.date.isoformat() + "." + session_dir.session_type)
            )
            if not processed_data_folder.exists():
                processed_data_folder.mkdir(parents=True)
            preprocessing_fn(session_dir, processed_data_folder, overwrite)
    return


def populate_processed_data_multiprocessed(
    data_streams=["session_info", "pycontrol", "video", "spikes", "lfp"], n_processes=6, overwrite=False
):
    """
    Top level function for populating processed data for entire experiment using multiple processes.
    Should be faster than populate_processed_data"""
    data_stream2fn = {
        "session_info": _populate_session_info,
        "pycontrol": _populate_pycontrol_data,
        "video": _populate_video_data,
        "spikes": _populate_spike_data,
        "lfp": _populate_lfp_data,
    }
    session_data_directory = get_sessions_data_directory()
    n_sessions = len(session_data_directory)
    session_directories = list(session_data_directory.itertuples())
    processed_data_folders = [
        PROCESSED_DATA_PATH / s.subject_ID / (s.date.isoformat() + "." + s.session_type) for s in session_directories
    ]
    save_functions = [data_stream2fn[data_stream] for data_stream in data_streams]
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        executor.map(
            _save_processed_data,
            [save_functions] * n_sessions,
            session_directories,
            processed_data_folders,
            [overwrite] * n_sessions,
        )
    return


def _save_processed_data(processing_functions, session_dir, processed_data_folder, overwrite):
    """Saves all processed data for a given session (or data specified by input processing functions)"""
    print(f"Processing {session_dir.subject_ID} {session_dir.date.isoformat()} {session_dir.session_type}")
    if not processed_data_folder.exists():
        processed_data_folder.mkdir(parents=True)
    for processing_fn in processing_functions:
        processing_fn(session_dir, processed_data_folder, overwrite)
    return


# %% Data stream functions


def _populate_session_info(data_directory, processed_data_folder, overwrite):
    """
    Saves session_info.json for a given session to the corresponding processed_data_folder.
    If overwirte=True, the function will overwrite an existing session_info.json file.
    """
    if not overwrite and (processed_data_folder / "session_info.json").exists():
        return
    else:
        session_info = get_session_info(data_directory)
        # save
        with open((processed_data_folder / "session_info.json"), "w") as outfile:
            outfile.write(json.dumps(session_info, indent=4))
        return


def _populate_pycontrol_data(data_directory, processed_data_folder, overwrite):
    """
    Saves trials.htsv and events.htsv for a given session to the corresponding processed_data_folder.
    If overwirte=True, the function will overwrite an existing trials.htsv and events.htsv files.
    """
    if data_directory.session_type == "rest":
        # no pycontrol data for rest/sleep sessions
        return
    if not overwrite and (processed_data_folder / "trials.htsv").exists():
        pass
    else:
        trials_df = get_trials_df(data_directory)
        trials_df.columns = _flatten_multiindex_columns(trials_df)
        trials_df.to_csv(processed_data_folder / "trials.htsv", index=False, sep="\t")
    if not overwrite and (processed_data_folder / "events.htsv").exists():
        return
    else:
        events_df = get_events_df(data_directory)
        events_df.to_csv(processed_data_folder / "events.htsv", index=False, sep="\t")
        return


def _populate_video_data(data_directory, processed_data_folder, overwrite):
    """
    Saves tracking.htsv, trajectories.htsv and trialInfo.htsv for a given session to the corresponding processed_data_folder.
    If overwirte=True, the function will overwrite an existing files.
    """
    if data_directory.session_type == "rest":
        # no video data for rest/sleep sessions
        return
    # save tracking data
    if not overwrite and (processed_data_folder / "frames.tracking.htsv").exists():
        pass
    else:
        tracking_df = fd.get_tracking_df(data_directory)
        tracking_df.columns = _flatten_multiindex_columns(tracking_df)
        tracking_df.to_csv(processed_data_folder / "frames.tracking.htsv", index=False, sep="\t")
    # save trajectories data
    if not overwrite and (processed_data_folder / "frames.trajectories.htsv").exists():
        pass
    else:
        trajectories_df = fd.get_trajectories_df(data_directory)
        trajectories_df.columns = _flatten_multiindex_columns(trajectories_df)
        trajectories_df.to_csv(processed_data_folder / "frames.trajectories.htsv", index=False, sep="\t")
    # save trial info data
    if not overwrite and (processed_data_folder / "frames.trialInfo.htsv").exists():
        return
    else:
        trial_info_df = fd.get_trial_info_df(data_directory)
        trial_info_df.to_csv(processed_data_folder / "frames.trialInfo.htsv", index=False, sep="\t")
    return


def _populate_spike_data(data_directory, processed_data_folder, overwrite):
    """
    Saves spikes.times.npy, spikes.clusters.npy and cluster.metrics.htsv for a given session to the corresponding processed_data_folder.
    If overwrite is True, the function will overwrite existing files.

    Note rest sessions are not processed here because they have not yet been preprocessed with Kilosort (as of 2024-09-24)
    """
    if data_directory.session_type == "rest":
        # spikes data for rest/sleep has not been processed yet
        return
    if not isinstance(data_directory.kilosort_path, Path):  # if not path (eg, np.nan)
        print(f"No spike data for {data_directory.subject_ID} {data_directory.date.isoformat()}, missing kilosort data")
        # no spike data for this session
        return
    if not overwrite and (processed_data_folder / "spikes.times.npy").exists():
        pass
    else:
        spike_pytimes = ed.get_spike_pycontrol_times(data_directory)
        np.save(processed_data_folder / "spikes.times.npy", spike_pytimes)
    if not overwrite and (processed_data_folder / "spikes.clusters.npy").exists():
        pass
    else:
        spike_clusters = ed.get_spike_clusters(data_directory)
        np.save(processed_data_folder / "spikes.clusters.npy", spike_clusters)
    if not overwrite and (processed_data_folder / "clusters.metrics.htsv").exists():
        pass
    else:
        cluster_metrics = ed.get_cluster_metrics(data_directory)
        if cluster_metrics is None:
            return
        else:
            cluster_metrics.to_csv(processed_data_folder / "clusters.metrics.htsv", sep="\t", index=False)
    return


def _populate_lfp_data(data_directory, processed_data_folder, overwrite):
    """
    Saves lfp.signal.npy, lfp.time.npy and lfp.metrics.htsv for a given session to the corresponding processed_data_folder.
    If overwrite is True, the function will overwrite existing files.

    Note rest sessions are not processed here because they have not yet been preprocessed with Kilosort (as of 2024-09-24)
    """
    if not overwrite and (processed_data_folder / "lfp.signal.npy").exists():
        pass
    else:
        lfp_signal = ld.get_LFP_signal(data_directory)
        np.save(processed_data_folder / "lfp.signal.npy", lfp_signal)
    if not overwrite and (processed_data_folder / "lfp.time.npy").exists():
        pass
    else:
        lfp_times = ld.get_LFP_times(data_directory)
        np.save(processed_data_folder / "lfp.time.npy", lfp_times)
    if not overwrite and (processed_data_folder / "lfp.metrics.htsv").exists():
        lfp_metrics = ld.get_LFP_metrics(data_directory)
        lfp_metrics.to_csv(processed_data_folder / "lfp.metrics.htsv", sep="\t", index=False)
    return


def _populate_unit_match_data(session_dir, processed_data_path, overwrite, max_duration_delta=5):
    """ """
    session_ID = f"{session_dir.subject_ID}-{session_dir.date}-{session_dir.session_type}"
    if (
        not isinstance(session_dir.ephys_path, str)  # missing ephys completely
        or session_dir.ephys_corrupt  # something wrong with ephys data - could not be preprocesed
        or not session_dir.spikesorting_completed  # spikesorting failed (also usually something wrong with ephys)
    ):
        print(f"Missing ephys data for {session_ID} cannot populate spike data")
        return
    if session_dir.session_type != "rest":
        # if ephys stop before end of session don't process spike data
        if session_dir.ephys_duration - session_dir.pycontrol_duration < -max_duration_delta:
            print(f"Ephys reocrding incomplete for {session_ID} cannot populate spike data")
            return
    if not overwrite and (processed_data_path / "UnitMatch").exists():
        pass
    else:
        preprocessed_UM_path = gud.get_unit_match_folder(session_dir)
        gud.copy_unit_match_folder(preprocessed_UM_path, processed_data_path)
    return


# %% Misc


def _flatten_multiindex_columns(df):
    """Returns a list of flat column names (str) where columns that were previously multiindex become level0_name.level1_name
    and single index columns stay level0_name"""
    return [f"{x[0]}.{x[1]}" if x[1] != "" else x[0] for x in df.columns.to_flat_index()]


# %%


def rename_processed_data(original_name, new_name):
    """This function looks through all processed_data folder and replaces a given filename with a new specified name"""
    for subject_folder in PROCESSED_DATA_PATH.iterdir():
        for session_folder in subject_folder.iterdir():
            if (session_folder / original_name).exists():
                (session_folder / original_name).rename(session_folder / new_name)
    return
