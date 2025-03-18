"""This module populates processes pycontrol, video and raw ephys data into processed data in the processed data folder"""

# %% Imports
import numpy as np
import json
from pathlib import Path
from datetime import date
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


def populate_processed_data(
    data_streams=["session_info", "pycontrol", "video", "spikes", "lfp", "unit_match"],
    overwrite=False,
    verbose=False,
):
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
            if verbose:
                print(f"Processing {session_dir.subject_ID} {session_dir.date.isoformat()} {session_dir.session_type}")
            processed_data_folder = (
                PROCESSED_DATA_PATH
                / session_dir.subject_ID
                / (session_dir.date.isoformat() + "." + session_dir.session_type)
            )
            if not processed_data_folder.exists():
                processed_data_folder.mkdir(parents=True)
            preprocessing_fn(session_dir, processed_data_folder, overwrite)
    return


# %% HPC parallel processing
def populate_session_processed_data(
    subject_ID,
    _date,
    session_type,
    data_streams=["session_info", "pycontrol", "video", "spikes", "lfp"],
    overwrite=False,
):
    """
    Populate processed data for a single session.
    Slighly hacky but designed to be called by a slurm script and run in parellel on an HPC.
    """
    data_stream2fn = {
        "session_info": _populate_session_info,
        "pycontrol": _populate_pycontrol_data,
        "video": _populate_video_data,
        "spikes": _populate_spike_data,
        "lfp": _populate_lfp_data,
        "unit_match": _populate_unit_match_data,
    }
    # load all data directories
    sessions_data_directory = get_sessions_data_directory()
    # isolate session from data directory
    filtered_session = sessions_data_directory[
        (sessions_data_directory.subject_ID == subject_ID)
        & (sessions_data_directory.date == date.fromisoformat(_date))
        & (sessions_data_directory.session_type == session_type)
    ].iloc[0]
    # define processed data folder (standardised naming)
    processed_data_folder = PROCESSED_DATA_PATH / subject_ID / (_date + "." + session_type)
    if not processed_data_folder.exists():
        processed_data_folder.mkdir(parents=True)
    # run processing for each data stream
    for data_stream in data_streams:
        print(f"Populating {data_stream} data")
        fn = data_stream2fn[data_stream]
        fn(filtered_session, processed_data_folder, overwrite)
    return print(f"Finished processing {subject_ID} {_date} {session_type}")


# %% Local parallel processing


def populate_processed_data_multiprocessed(
    data_streams=["session_info", "pycontrol", "video", "spikes", "lfp", "unit_match"], n_processes=6, overwrite=False
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
        "unit_match": _populate_unit_match_data,
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


def _populate_session_info(session_dir, processed_data_folder, overwrite):
    """
    Saves session_info.json for a given session to the corresponding processed_data_folder.
    If overwirte=True, the function will overwrite an existing session_info.json file.
    """
    if not overwrite and (processed_data_folder / "session_info.json").exists():
        return
    else:
        session_info = get_session_info(session_dir)
        # save
        with open((processed_data_folder / "session_info.json"), "w") as outfile:
            outfile.write(json.dumps(session_info, indent=4))
        return


def _populate_pycontrol_data(session_dir, processed_data_folder, overwrite):
    """
    Saves trials.htsv and events.htsv for a given session to the corresponding processed_data_folder.
    If overwirte=True, the function will overwrite an existing trials.htsv and events.htsv files.
    """
    if session_dir.session_type == "rest":  # no pycontrol data for rest/sleep sessions
        return
    if not pass_data_QC(session_dir, "pycontrol"):  # issues with raw data
        return
    if not overwrite and (processed_data_folder / "trials.htsv").exists():
        pass
    else:
        trials_df = get_trials_df(session_dir)
        trials_df.columns = _flatten_multiindex_columns(trials_df)
        trials_df.to_csv(processed_data_folder / "trials.htsv", index=False, sep="\t")
    if not overwrite and (processed_data_folder / "events.htsv").exists():
        return
    else:
        events_df = get_events_df(session_dir)
        events_df.to_csv(processed_data_folder / "events.htsv", index=False, sep="\t")
        return


def _populate_video_data(session_dir, processed_data_folder, overwrite):
    """
    Saves tracking.htsv, trajectories.htsv and trialInfo.htsv for a given session to the corresponding processed_data_folder.
    If overwirte=True, the function will overwrite an existing files.
    """
    if session_dir.session_type == "rest":  # no video data for rest/sleep sessions
        return
    if not pass_data_QC(session_dir, "video"):  # issues with raw data
        return
    # save tracking data
    if not overwrite and (processed_data_folder / "frames.tracking.htsv").exists():
        pass
    else:
        tracking_df = fd.get_tracking_df(session_dir)
        tracking_df.columns = _flatten_multiindex_columns(tracking_df)
        tracking_df.to_csv(processed_data_folder / "frames.tracking.htsv", index=False, sep="\t")
    # save trajectories data
    if not overwrite and (processed_data_folder / "frames.trajectories.htsv").exists():
        pass
    else:
        trajectories_df = fd.get_trajectories_df(session_dir)
        trajectories_df.columns = _flatten_multiindex_columns(trajectories_df)
        trajectories_df.to_csv(processed_data_folder / "frames.trajectories.htsv", index=False, sep="\t")
    # save trial info data
    if not overwrite and (processed_data_folder / "frames.trialInfo.htsv").exists():
        return
    else:
        trial_info_df = fd.get_trial_info_df(session_dir)
        trial_info_df.to_csv(processed_data_folder / "frames.trialInfo.htsv", index=False, sep="\t")
    return


def _populate_spike_data(session_dir, processed_data_folder, overwrite):
    """
    Saves spikes.times.npy, spikes.clusters.npy and cluster.metrics.htsv for a given session to the corresponding processed_data_folder.
    If overwrite is True, the function will overwrite existing files.

    Note rest sessions are not processed here because they have not yet been preprocessed with Kilosort (as of 2024-09-24)
    """
    if not pass_data_QC(session_dir, "spikes"):  # issues with raw data
        return
    # save spike times
    if not overwrite and (processed_data_folder / "spikes.times.npy").exists():
        pass
    else:
        spike_pytimes = ed.get_spike_times(session_dir)
        np.save(processed_data_folder / "spikes.times.npy", spike_pytimes)
    # save spike clusters
    if not overwrite and (processed_data_folder / "spikes.clusters.npy").exists():
        pass
    else:
        spike_clusters = ed.get_spike_clusters(session_dir)
        np.save(processed_data_folder / "spikes.clusters.npy", spike_clusters)
    # save cluster metrics
    if not overwrite and (processed_data_folder / "clusters.metrics.htsv").exists():
        pass
    else:
        cluster_metrics = ed.get_cluster_metrics(session_dir)
        cluster_metrics.columns = _flatten_multiindex_columns(cluster_metrics)
        cluster_metrics.to_csv(processed_data_folder / "clusters.metrics.htsv", sep="\t", index=False)
    return


def _populate_lfp_data(session_dir, processed_data_folder, overwrite):
    """
    Saves lfp.signal.npy, lfp.time.npy and lfp.metrics.htsv for a given session to the corresponding processed_data_folder.
    If overwrite is True, the function will overwrite existing files.

    Note rest sessions are not processed here because they have not yet been preprocessed with Kilosort (as of 2024-09-24)
    """
    if not pass_data_QC(session_dir, "lfp"):  # issues with raw data
        return
    # save lfp signal
    if not overwrite and (processed_data_folder / "lfp.signal.npy").exists():
        pass
    else:
        lfp_signal = ld.get_LFP_signal(session_dir)
        np.save(processed_data_folder / "lfp.signal.npy", lfp_signal)
    # save lfp times
    if not overwrite and (processed_data_folder / "lfp.times.npy").exists():
        pass
    else:
        lfp_times = ld.get_LFP_times(session_dir)
        np.save(processed_data_folder / "lfp.times.npy", lfp_times)
    # save lfp metrics
    if not overwrite and (processed_data_folder / "lfp.metrics.htsv").exists():
        pass
    else:
        lfp_metrics = ld.get_LFP_metrics(session_dir)
        lfp_metrics.columns = _flatten_multiindex_columns(lfp_metrics)
        lfp_metrics.to_csv(processed_data_folder / "lfp.metrics.htsv", sep="\t", index=False)
    return


def _populate_unit_match_data(session_dir, processed_data_path, overwrite):
    """ """
    if not pass_data_QC(session_dir, "unit_match"):  # issues with raw data
        return
    # save unit match data
    if not overwrite and (processed_data_path / "UnitMatch").exists():
        pass
    else:
        preprocessed_UM_path = gud.get_unit_match_folder(session_dir)
        gud.copy_unit_match_folder(preprocessed_UM_path, processed_data_path)
    return


# %% Misc


def pass_data_QC(session_dir, data_stream, duration_thres=5):
    """
    Checks if the there are any issues with the raw data that prevent processing.
    Eg, missing ephys, missing video, incomplete total session etc.
    Data could be salvaged from all the sessions excluded through this method, but
    easier to avoid for now.
    """
    session_ID = f"{session_dir.subject_ID}-{session_dir.date.isoformat()}-{session_dir.session_type}"
    # if session is short (incomplete) skip processing for everything except session_info
    if data_stream != "session_info":
        if session_dir.short_session:
            print(f"Sesssion: {session_ID}: Incomplete session, skip processing")
            return False
    # check ephys and video if durations line up with pycontrol (indiciative of missing data)
    if session_dir.session_type == "maze":
        if data_stream in ["spikes", "lfp", "unit_match"]:
            ephys_diff = session_dir.ephys_duration - session_dir.pycontrol_duration
            if ephys_diff > duration_thres:
                print(f"Session: {session_ID}: Ephys duration too short, missing data, skipping processing")
                return False
        elif data_stream == "video":
            video_diff = session_dir.video_duration - session_dir.pycontrol_duration
            if video_diff > duration_thres:
                print(f"Session: {session_ID}: Video duration too short, missing data, skipping processing")
                return False
    # check if spikesorting failed
    if not session_dir.spikesorting_complete and data_stream in ["spikes", "lfp", "unit_match"]:
        print(f"Session: {session_ID}: Spikesorting not completed, skipping processing")
        return False
    return True


def _flatten_multiindex_columns(df):
    """Returns a list of flat column names (str) where columns that were previously multiindex become level0_name.level1_name
    and single index columns stay level0_name"""
    return [f"{x[0]}.{x[1]}" if x[1] != "" else x[0] for x in df.columns.to_flat_index()]


# %% Fixes
def rename_processed_data(original_name, new_name):
    """This function looks through all processed_data folder and replaces a given filename with a new specified name"""
    for subject_folder in [d for d in PROCESSED_DATA_PATH.iterdir() if d.is_dir()]:
        for session_folder in [d for d in subject_folder.iterdir() if d.is_dir()]:
            if (session_folder / original_name).exists():
                (session_folder / original_name).rename(session_folder / new_name)
    return
