"""
Library to organise experiment datafiles. Generates a session data directory that points to all the raw
and preprocessed data files for a given session. Rows of this data frame are passed to functions in the
GridMaze preprocessing module to generate the standardised processed data file types.
"""
#%% Imports
import json
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path 
from GridMaze.preprocessing.pycontrol_data_import import session_dataframe
from GridMaze.preprocessing.get_frames_dfs import open_dlc_output_as_df
#%% Global Variables

from GridMaze.paths import PYCONTROL_PATH, EPHYS_PATH, VIDEO_PATH, DLC_PATH, SPIKESORTING_PATH, EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "maze_configs.json", "r") as infile:
    MAZE_CONFIGS = json.load(infile)

# define experiment start date to ignore ephys recordings before day 1 on big maze
EXPERIMENT_START_DATE = date.fromisoformat(MAZE_CONFIGS['maze_1']['start'])

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as infile:
    SUBJECT_IDS = json.load(infile)

SESSION_TYPES = ["maze", "rest"] # recorded in that order

IGNORE_SESSIONS = pd.read_csv(EXPERIMENT_INFO_PATH / "ignore_sessions.htsv", sep="\t")

FRAME_RATE = 60
#%% 
def get_sessions_data_directory(overwrite=False):
    """
    Returns a dataframe with rows corresponding to each session recorded in the experiment,
    each row includes the subject, session date, session type (used to distinguish multiple sessions
    run on a given day for a given subject), along with paths to the raw and preprocessed data files
    (e.g. pycontrol, video, ephys, dlc, kilosort).

    Rows of this dataframe are passed to functions in the GridMaze preprocessing module to generate the
    standardised processed data file types.
    """
    # initialise data directory from experiment info (see global variables)
    data_directory = init_data_directory()
    # add pycontrol data
    pycontrol_paths, pycontrol_datetimes = add_pycontrol_paths(data_directory)
    data_directory["pycontrol_path"] = pycontrol_paths
    data_directory["pycontrol_datetime"] = pycontrol_datetimes
    data_directory["pycontrol_duration"] = data_directory.pycontrol_path.apply(get_pycontrol_duration)
    # add video data
    video_paths, sync_paths = add_video_paths(data_directory)
    data_directory["video_path"] = video_paths
    data_directory["video_sync_path"] = sync_paths
    # add dlc data
    data_directory["dlc_path"] = add_dlc_paths(data_directory)
    data_directory["video_duration"] = data_directory.dlc_path.apply(get_video_duration)
    # add ephys data
    ephys_data_paths, ephys_sync_paths, ephys_datetimes = add_ephys_paths(data_directory)
    data_directory["ephys_data_path"] = ephys_data_paths
    data_directory["ephys_sync_path"] = ephys_sync_paths
    data_directory["ephys_datetime"] = ephys_datetimes
    # add spikesorting data
    spikesorting_paths, spikesorting_completed = add_spikesorting_paths(data_directory)
    data_directory["spikesorting_path"] = spikesorting_paths
    data_directory["spikesorting_complete"] = spikesorting_completed
    return data_directory

def init_data_directory():
    """
    Returns a pandas DataFrame with rows corresponding to each session recorded in the experiment. Later
    columns will be added to this DataFrame to point to the raw and preprocessed data files for each session.
    """
    session_info = []
    for maze_name in MAZE_CONFIGS.keys():
        start_date = date.fromisoformat(MAZE_CONFIGS[maze_name]['start'])
        end_date = date.fromisoformat(MAZE_CONFIGS[maze_name]['end'])
        d = start_date
        while d <= end_date:
            for subject_ID in SUBJECT_IDS:
                for session_type in SESSION_TYPES:
                    session_info.append({
                        "maze_name": maze_name,
                        "subject_ID": subject_ID,
                        "date": d,
                        "session_type": session_type
                    })
            d += timedelta(days=1)   
    return pd.DataFrame(session_info)   

#%% Pycontrol

def get_pycontrol_duration(pycontrol_path):
    """Returns duration of pycontrol session in mins"""
    if pycontrol_path is np.nan:
        return np.nan
    else:
        return session_dataframe(pycontrol_path).time.iloc[-1]/(60*1000)

def add_pycontrol_paths(init_data_directory):
    """
    Matches pycontrol files to each session (row) of the init_data_directory.
    Returns a list of paths ordered by the rows of the init_data_directory. Note 'rest' 
    sessions will not have pycontrol files, values will be np.nan.

    Note newer version of pycontrol output files in a .tsv format whereas older versions
    like the data from this exp, outputted a .txt file.
    """
    pycontrol_paths_df = _get_pycontrol_paths_df()
    sorted_pycontrol_paths = []
    sorted_pycontrol_datetimes = []
    for row in init_data_directory.itertuples():
        if row.session_type == "rest":
            sorted_pycontrol_paths.append(np.nan)
            sorted_pycontrol_datetimes.append(np.nan)
        else:
            # find corresponding session in pycontrol_paths_df
            subject_mask = pycontrol_paths_df.subject_ID == row.subject_ID
            date_mask = pycontrol_paths_df.datetime.apply(lambda x: x.date()) == row.date
            filted_pycontrol_path = pycontrol_paths_df[subject_mask & date_mask]
            if not len(filted_pycontrol_path) == 1:
                raise FileNotFoundError(f"No unique pycontrol file found for {row}")
            sorted_pycontrol_paths.append(filted_pycontrol_path.pycontrol_filepath.values[0])
            sorted_pycontrol_datetimes.append(filted_pycontrol_path.datetime)
    return sorted_pycontrol_paths, sorted_pycontrol_datetimes

def _get_pycontrol_paths_df():
    """Extracts subject ID and datetime from pycontrol file names and returns a DataFrame"""
    all_pycontrol_paths = [p for p in PYCONTROL_PATH.glob("*.txt") if not p.name.startswith(".")]
    pycontrol_info = []
    for filepath in all_pycontrol_paths:
        filename = filepath.name
        subject_ID = filename.split("-")[0]
        dt = datetime.strptime(filename.split(".")[0].split("-",1)[1], '%Y-%m-%d-%H%M%S')
        if _ignore_session(subject_ID, dt, reason="any"):
            continue
        else:
            pycontrol_info.append({
                "subject_ID": subject_ID,
                "datetime": dt,
                "pycontrol_filepath": filepath
            })
    return pd.DataFrame(pycontrol_info)

#%% Video

def add_video_paths(init_data_directory):    
    """
    Matches video & vidoe sync files to each session (row) of the init_data_directory.
    """
    video_paths_df = _get_video_paths_df()
    sorted_video_paths = []
    sorted_sync_paths = []
    sorted_datetimes = []
    for row in init_data_directory.itertuples():
        # rest sessions do not have associated video files
        if row.session_type == "rest":
            sorted_video_paths.append(np.nan)
            sorted_sync_paths.append(np.nan)
            sorted_datetimes.append(np.nan)
            continue
        # find corresponding session in video_paths_df
        subject_mask = video_paths_df.subject_ID == row.subject_ID
        date_mask = video_paths_df.datetime.apply(lambda x: x.date()) == row.date
        filtered_video_path = video_paths_df[subject_mask & date_mask]
        if not len(filtered_video_path) == 1:
            raise FileNotFoundError(f"No unique video file found for {row}")
        sorted_video_paths.append(filtered_video_path.video_filepath.values[0])
        sorted_sync_paths.append(filtered_video_path.video_sync_filepath.values[0])
        sorted_datetimes.append(filtered_video_path.datetime.values[0])
    return sorted_video_paths, sorted_sync_paths

def _get_video_paths_df():
    """
    Extracts subject ID and datetime from video file names (.mp4) and video sync file names (.csv)
    and returns a DataFrame the video file paths and video sync file paths associated with each 
    session (one subject, one datetime).
    Note session in ignore session that were rerun in the afternoon will have multiple video files and must
    be ignored.
    """
    # process video files
    all_video_paths = list(VIDEO_PATH.glob("*.mp4"))
    video_info = []
    for filepath in all_video_paths:
        filename = filepath.name
        subject_ID = filename.split("_")[0]
        dt = datetime.strptime(filename.split(".")[0].split("_")[1], '%Y-%m-%d-%H%M%S')
        if _ignore_session(subject_ID, dt, reason="reran session in afternoon", window=timedelta(seconds=20)):
            continue
        else:
            video_info.append({
                "subject_ID": subject_ID,
                "datetime": dt,
                "video_filepath": filepath
            })
    video_df = pd.DataFrame(video_info)

    # process sync files
    all_sync_paths = list(VIDEO_PATH.glob("*.csv"))
    sync_info = []
    for filepath in all_sync_paths:
        filename = filepath.name
        subject_ID = filename.split("_")[0]
        dt =datetime.strptime(filename.split(".")[0].split("_",-1)[-1], '%Y-%m-%d-%H%M%S')
        if _ignore_session(subject_ID, dt, reason="reran session in afternoon", window=timedelta(seconds=20)):
            continue
        else:
            sync_info.append({
                "subject_ID": subject_ID,
                "datetime": dt,
                "video_sync_filepath": filepath
            })
    sync_df = pd.DataFrame(sync_info)

    # merge video and sync dataframes
    return video_df.merge(sync_df, on=["subject_ID", "datetime"], how="outer")

#%% Ephys

def add_ephys_paths(init_data_directory):
    """
    Adds ephys data folder and ephys sync file path to each corresponding session in the init_data_directory.
    See _get_ephys_paths_df function for more details about how the ephys data folder and ephys sync file path
    are extracted.
    """
    ephys_paths_df = _get_ephys_paths_df()
    sorted_ephys_data_paths = []
    sorted_ephys_sync_paths = []
    sorted_ephys_datetimes = []
    for row in init_data_directory.itertuples():
        subject_mask = ephys_paths_df.subject_ID == row.subject_ID
        date_mask = ephys_paths_df.datetime.apply(lambda x: x.date()) == row.date
        session_type_mask = ephys_paths_df.session_type == row.session_type
        filtered_ephys_path = ephys_paths_df[subject_mask & date_mask & session_type_mask]
        if not len(filtered_ephys_path) == 1:
            raise FileNotFoundError(f"No unique ephys file found for {row}")
        sorted_ephys_data_paths.append(filtered_ephys_path.ephys_data_folder.values[0])
        sorted_ephys_sync_paths.append(filtered_ephys_path.ephys_sync_filepath.values[0])
        sorted_ephys_datetimes.append(filtered_ephys_path.datetime.values[0])
    return sorted_ephys_data_paths, sorted_ephys_sync_paths, sorted_ephys_datetimes

def _get_ephys_paths_df():
    """
    Extracts the subject_ID and datetime from ephys information, with the ephys data folder and ephys sync file path.
    Returns a DataFrame. Note internal_sync_filepath which is hardcoded in the function below represents the interal filepath
    to the timestamps.npy file in the ephys data folder that holds the sync pulses for aligning data. Note the function also
    includes some logic to interate through each base ephys folder to find the correct Record node subfolder, either:
    "Record Node 121" or "Record Node 122" (a silly feature of how Open Ephys recorded the data from rest or maze).
    Function also assigns session type to each session (maze or rest) based on the order they were conducted. See 
    _get_session_type_order function for more details about execptions are handeled.
    """
    internal_sync_filepath="experiment1/recording1/events/Rhythm_FPGA-109.0/TTL_1/timestamps.npy"
    all_ephys_paths = [x for p in EPHYS_PATH.iterdir() if p.is_dir() for x in p.iterdir() if x.is_dir() and not x.name == "ignore"]
    ephys_info = []
    for filepath in all_ephys_paths:
        subject_ID = filepath.parts[-2]
        dt = datetime.strptime(filepath.name, '%Y-%m-%d_%H-%M-%S')
        if dt.date() < EXPERIMENT_START_DATE:
            # some ephys recordings were made before the start of the experiment
            # might become useful for control analyses but not processed here
            continue
        if _ignore_session(subject_ID, dt, reason="reran session in afternoon", window=timedelta(minutes=1)):
            # sessions re run in afternoon will have two associated ephys files
            # sessions where pycontrol was restarted will have just one assoicated ephys file
            continue
        else:
            ephys_info.append({
                "subject_ID": subject_ID,
                "datetime": dt,
                "ephys_data_folder": filepath,
                "ephys_sync_filepath": list(filepath.iterdir())[0] / internal_sync_filepath #first interate through to the folder rocord node then add internal filepath
            })
    ephys_paths_df = pd.DataFrame(ephys_info)
    # assign sessions as maze or rest, knowing they were conducted in that order
    ephys_paths_df["session_type"] = None
    all_subject_IDs = ephys_paths_df.subject_ID.unique()
    all_dates = ephys_paths_df.datetime.apply(lambda x: x.date()).unique()
    for subject_ID in all_subject_IDs:
        for d in all_dates:
            filtered_paths_df = ephys_paths_df[(ephys_paths_df.subject_ID == subject_ID) & (ephys_paths_df.datetime.apply(lambda x: x.date()) == d)]
            filtered_paths_df = filtered_paths_df.sort_values("datetime")
            if not len(filtered_paths_df) == 2:
                raise FileNotFoundError(f"Expected 2 ephys files for {subject_ID} {d}, got {len(filtered_paths_df)}")
            session_type_order = _get_session_type_order(filtered_paths_df)
            for row, session_type in zip(filtered_paths_df.itertuples(), session_type_order):
                subject_ID = row.subject_ID
                d = row.datetime.date()
                ephys_paths_df.loc[row.Index, "session_type"] = session_type
    return ephys_paths_df


#%% DeepLabCut
    
def get_video_duration(dlc_path):
    """
    Returns video duration in mins by reading DLC file
    (quicker than reading length of video fiel)
    """
    if dlc_path is np.nan:
        return np.nan
    else:
        with open(dlc_path, "rb") as f:
            rows = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1024 * 1024), b""))
        return (rows-1)/FRAME_RATE/60 # mins


def add_dlc_paths(init_data_directory):
    """
    Adds DeepLabCut file paths to each session in the init_data_directory
    """
    dlc_paths_df = _get_dlc_paths_df()
    sorted_dlc_paths = []
    for row in init_data_directory.itertuples():
        if row.session_type == "rest":
            sorted_dlc_paths.append(np.nan)
            continue
        subject_mask = dlc_paths_df.subject_ID == row.subject_ID
        date_mask = dlc_paths_df.datetime.apply(lambda x: x.date()) == row.date
        filtered_dlc_path = dlc_paths_df[subject_mask & date_mask]
        if not len(filtered_dlc_path) == 1:
            print(filtered_dlc_path)
            raise FileNotFoundError(f"No unique dlc file found for {row}")
        sorted_dlc_paths.append(filtered_dlc_path.dlc_filepath.values[0])
    return sorted_dlc_paths

def _get_dlc_paths_df():
    """
    Extracts subject ID and datetime from DeepLabCut file names and returns a DataFrame
    """
    all_dlc_paths = list(DLC_PATH.glob("*.csv"))
    dlc_info = []
    for filepath in all_dlc_paths:
        filename = filepath.name
        subject_ID = filename.split("_")[0]
        dt = datetime.strptime(filename.split('_')[1].split('DLC')[0], '%Y-%m-%d-%H%M%S')
        if _ignore_session(subject_ID, dt, reason="reran session in afternoon"):
            # sessions where pycontrol was restarted video as not, therefore no need to ignore
            continue
        else:
            dlc_info.append({
                "subject_ID": subject_ID,
                "datetime": dt,
                "dlc_filepath": filepath
            })
    return pd.DataFrame(dlc_info)


#%% updates spikesorting paths

def add_spikesorting_paths(data_directory):
    """
    Add spikesorting paths and if spikesorting has run to completion
    by directly matching to ephys datetimes and subject_IDs
    """
    spikesorting_paths_df = _get_spikesorting_paths_df()
    matched_spikesorting_paths = []
    spikesorting_completed = []
    for _, session in data_directory.iterrows():
        session_ID = f"{session.subject_ID}-{session.date}-{session.session_type}"
        if not isinstance(session.ephys_data_path, Path):  # no valid ephys recording for session
            matched_spikesorting_paths.append(np.nan)
            spikesorting_completed.append(np.nan)
        else:
            matched_ss = spikesorting_paths_df[
                (spikesorting_paths_df.datetime == session.ephys_datetime)
                & (spikesorting_paths_df.subject == session.subject_ID)
            ]
            if matched_ss.empty:
                print(f"Missing spikesorting data for {session_ID}")
            elif len(matched_ss) > 1:
                print(f"Multiple spikesorting folders fround for {session_ID}")
            else:
                matched_spikesorting_paths.append(matched_ss.spikesorting_path.values[0])
                spikesorting_completed.append(matched_ss.spikesorting_completed.values[0])
    return matched_spikesorting_paths, spikesorting_completed


def _get_spikesorting_paths_df():
    """
    Returns paths to spike sorted ephys data in an orghanised DataFrame
    """
    all_spikesorting_paths = [
        f
        for d in SPIKESORTING_PATH.iterdir()
        if d.name not in ["kilosort_optim", "probe_params"] and d.is_dir()
        for f in d.iterdir()
        if f.is_dir()
    ]
    paths_info = []
    for ss_path in all_spikesorting_paths:
        subject, dt_string = ss_path.parts[-2:]
        dt = datetime.fromisoformat(dt_string)
        paths_info.append(
            {
                "subject": subject,
                "datetime": dt,
                "spikesorting_path": ss_path,
                "spikesorting_completed": (ss_path / "DONE.txt").exists(),
            }
        )
    return pd.DataFrame(paths_info)



#%% Handling ignored sessions & processing exceptions

def _ignore_session(subject_ID, date_time, reason="any", window=timedelta(minutes=1)):
    """
    Checks if a session defined by a subject_ID and a datetime should be ignored. returns a boolean (True if session is to be ignored)
    This is done by checking if the session is in IGNORE_SESSIONS (defined in get_experiment_info.py),
    Note that datetimes are checked to be within a window (set by window variable, seconds) of the datetime in IGNORE_SESSIONS, in the case
    of ephys and video sessions that are started slighly before pycontrol. Sessions ignored for a particular reason:
    eg, "restarted pycontrol", "reran session in afternoon" can be checked specifically as well by setting the reason variable.

    """
    subject_mask = IGNORE_SESSIONS.subject == subject_ID
    if not any(subject_mask):
        return False
    datetime_mask = IGNORE_SESSIONS.datetime.apply(lambda x: (abs(datetime.fromisoformat(x) - date_time)) < window)
    if not any(datetime_mask):
        return False
    reason_mask = IGNORE_SESSIONS.reason == reason if reason in IGNORE_SESSIONS.reason.values else np.ones(len(IGNORE_SESSIONS), dtype=bool)
    return any(subject_mask & datetime_mask & reason_mask)

def _get_session_type_order(filtered_paths_df):
    """
    Returns the order session types were conducted (maze --> rest OR rest --> maze).
    All sessions were run maze --> rest, unless the session was rerun in the afternoon, the order
    will be rest --> maze.

    Needed to hard code instance for session m7 2022-07-11 where session was rerun immediately with sleep/rest recroded after.
    """
    # check if all sessions are for the same subject and date
    subject_ID = np.unique(filtered_paths_df.subject_ID)
    if not len(subject_ID) == 1:
        raise ValueError(f"Expected one subject ID, got {subject_ID}")
    subject_ID = subject_ID[0]
    date = np.unique(filtered_paths_df.datetime.apply(lambda x: x.date()))
    if not len(date) == 1:
        raise ValueError(f"Expected one date, got {date}")
    else:
        date = date[0]
    # hard code exception for session m7 2022-07-11
    if subject_ID == "m7" and date == date.fromisoformat("2022-07-11"):
        return ["maze", "rest"]
    # reference ignore sessions list
    subject_mask = IGNORE_SESSIONS.subject == subject_ID
    date_mask = IGNORE_SESSIONS.datetime.apply(lambda x: datetime.fromisoformat(x).date()) == date
    reason_mask = IGNORE_SESSIONS.reason == "reran session in afternoon"
    if any(subject_mask & date_mask & reason_mask):
        return ["rest", "maze"]
    else:
        return ["maze", "rest"]

