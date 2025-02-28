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
from ..paths import DATA_PATH

#%% Global Variables

# set paths to raw and preprocessed data
PYCONTROL_PATH = DATA_PATH / "raw_data" / "pycontrol"
EPHYS_PATH = DATA_PATH / "raw_data" / "ephys"
VIDEO_PATH = DATA_PATH / "raw_data" / "video"
DLC_PATH = DATA_PATH / "preprocessed_data" / "DeepLabCut"
KILOSORT_PATH = DATA_PATH / "preprocessed_data" / "kilosort"

# load experiment info
EXPERIMENT_INFO_PATH = DATA_PATH / "experiment_info"

with open(EXPERIMENT_INFO_PATH / "maze_configs.json", "r") as infile:
    MAZE_CONFIGS = json.load(infile)

# define experiment start date to ignore ephys recordings before day 1 on big maze
EXPERIMENT_START_DATE = date.fromisoformat(MAZE_CONFIGS['maze_1']['start'])

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as infile:
    SUBJECT_IDS = json.load(infile)

SESSION_TYPES = ["maze", "rest"] # recorded in that order

IGNORE_SESSIONS = pd.read_csv(EXPERIMENT_INFO_PATH / "ignore_sessions.htsv", sep="\t")
#%% 
def get_sessions_data_directory():
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
    data_directory["pycontrol_path"] = add_pycontrol_paths(data_directory)
    # add video data
    video_paths, sync_paths = add_video_paths(data_directory)
    data_directory["video_path"] = video_paths
    data_directory["video_sync_path"] = sync_paths
    # add ephys data
    ephys_data_paths, ephys_sync_paths = add_ephys_paths(data_directory)
    data_directory["ephys_data_path"] = ephys_data_paths
    data_directory["ephys_sync_path"] = ephys_sync_paths
    # add dlc data
    data_directory["dlc_path"] = add_dlc_paths(data_directory)
    # add kilosort data
    data_directory["kilosort_path"] = add_kilosort_paths(data_directory)
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
    for row in init_data_directory.itertuples():
        if row.session_type == "rest":
            sorted_pycontrol_paths.append(np.nan)
        else:
            # find corresponding session in pycontrol_paths_df
            subject_mask = pycontrol_paths_df.subject_ID == row.subject_ID
            date_mask = pycontrol_paths_df.datetime.apply(lambda x: x.date()) == row.date
            filted_pycontrol_path = pycontrol_paths_df[subject_mask & date_mask]
            if not len(filted_pycontrol_path) == 1:
                raise FileNotFoundError(f"No unique pycontrol file found for {row}")
            sorted_pycontrol_paths.append(filted_pycontrol_path.pycontrol_filepath.values[0])
    return sorted_pycontrol_paths

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
    for row in init_data_directory.itertuples():
        # rest sessions do not have associated video files
        if row.session_type == "rest":
            sorted_video_paths.append(np.nan)
            sorted_sync_paths.append(np.nan)
            continue
        # find corresponding session in video_paths_df
        subject_mask = video_paths_df.subject_ID == row.subject_ID
        date_mask = video_paths_df.datetime.apply(lambda x: x.date()) == row.date
        filtered_video_path = video_paths_df[subject_mask & date_mask]
        if not len(filtered_video_path) == 1:
            raise FileNotFoundError(f"No unique video file found for {row}")
        sorted_video_paths.append(filtered_video_path.video_filepath.values[0])
        sorted_sync_paths.append(filtered_video_path.video_sync_filepath.values[0])
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
    for row in init_data_directory.itertuples():
        subject_mask = ephys_paths_df.subject_ID == row.subject_ID
        date_mask = ephys_paths_df.datetime.apply(lambda x: x.date()) == row.date
        session_type_mask = ephys_paths_df.session_type == row.session_type
        filtered_ephys_path = ephys_paths_df[subject_mask & date_mask & session_type_mask]
        if not len(filtered_ephys_path) == 1:
            raise FileNotFoundError(f"No unique ephys file found for {row}")
        sorted_ephys_data_paths.append(filtered_ephys_path.ephys_data_folder.values[0])
        sorted_ephys_sync_paths.append(filtered_ephys_path.ephys_sync_filepath.values[0])
    return sorted_ephys_data_paths, sorted_ephys_sync_paths

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


#%% Kilosort
def add_kilosort_paths(data_directory):
    """
    Adds Kilosort file paths to each session in the init_data_directory.
    Note data_directory must have ephys paths already added (output from add_ephys_paths function).
    """
    kilosort_paths_df = _get_kilosort_paths_df()
    sorted_kilosort_paths = []
    for row in data_directory.itertuples():
        # easiest to match ks files to the ephys files they were processed from (no need to match session types)
        subject_mask = kilosort_paths_df.subject_ID == row.subject_ID
        date_mask = kilosort_paths_df.datetime == datetime.strptime(row.ephys_data_path.name, '%Y-%m-%d_%H-%M-%S')
        filtered_kilosort_path = kilosort_paths_df[subject_mask & date_mask]
        if not len(filtered_kilosort_path) == 1:
            if row.session_type == "rest":
                # rest sessions currently not spikesorted
                sorted_kilosort_paths.append(np.nan)
                continue
            else:
                print(f"No unique kilosort file found for {row}")
                sorted_kilosort_paths.append(np.nan)
                continue
        sorted_kilosort_paths.append(filtered_kilosort_path.kilosort_folder.values[0])
    return sorted_kilosort_paths

def _get_kilosort_paths_df():
    """
    Extracts subject ID and datetime from kilosor sorter output file paths for all spikesorted sessions, returned as a DataFrame
    """
    all_kilosort_paths = [x for p in KILOSORT_PATH.iterdir() if p.is_dir() for x in p.iterdir() if x.is_dir() and not x.name =="ignore"]
    kilosort_info = []
    for filepath in all_kilosort_paths:
        subject_ID = filepath.parts[-2]
        dt = datetime.strptime('_'.join(filepath.name.split('_')[:2]), '%Y-%m-%d_%H-%M-%S')
        if dt.date() < EXPERIMENT_START_DATE:
            # some ephys recordings were made before the start of the experiment
            # might become useful for control analyses but not processed here
            continue
        if _ignore_session(subject_ID, dt, reason="reran session in afternoon"):
            # only sessions rerun in the afternoon will have multiple kilosort folders
            continue
        else:
            kilosort_info.append({
                "subject_ID": subject_ID,
                "datetime": dt,
                "kilosort_folder": filepath / "Phy"
            })
    return pd.DataFrame(kilosort_info)

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

