# %% Imports
import csv
import json
import numpy as np
import pandas as pd
from . import rsync
from . import pycontrol_data_import as di
from . import get_head_direction as hd
from . import maze_registration as maze_reg
from . import pixels_to_position as pix2pos
from . import get_centroid_position as cpos
from . import get_maze_trajectories as mtraj
from ..maze import representations as maze_reps
from .get_pycontrol_dfs import get_trials_df

from ..paths import EXPERIMENT_INFO_PATH

# %% Pathnames & global variables
# SESSION_DATA_DIRECTORY_DF = get_data_directory_df()
IMAGE_SIZE = maze_reg.get_image_size_from_video()
DLC_LIKELIHOOD_THRESHOLD = 0.99

with open(EXPERIMENT_INFO_PATH / "bodypart_outlier_thresholds.json") as f:
    BODYPART_OUTLIER_THRESHOLDS = json.load(f)

PART_NAME_CHANGES = {
    "nose": "head_front",
    "head1": "head_mid",
    "head2": "head_back",
    "lear": "ear_L",
    "rear": "ear_R",
    "mid1": "body_front",
    "mid2": "body_mid",
    "tail": "body_back",
}


# %% Main functions
def get_tracking_df(session_directory):
    return get_cleaned_dlc_df(session_directory.dlc_path)


def get_trajectories_df(session_directory):
    """
    Returns a Pandas DataFrame containing frame-by-frame information from a session directory.

    Parameters
    ----------
    session_directory : single row for the SESSION_DATA_DIRECTROY_DF DataFrame
        Series containing raw data filepaths and metadata relevant for the session.

    Returns
    -------
    trajectories_df : Pandas DataFrame
        A DataFrame containing frame-by-frame information, including:
        - Time: the time of each frame (in seconds) referenced to pyControl.
        - Head direction: the head direction (in degrees) of the animal at each frame.
        - Head direction interpolated: a boolean value indicating whether the head direction value was interpolated.
        - Centroid position x/y: the x/y-coordinates (physical-space) of the animal's centroid  at each frame.
            Centroid defined as back of the head
        - Centroid position interpolated: a boolean value indicating whether the centroid position value was interpolated.
        - Maze position simple: the position of the animal in the simple maze (node or edge label) at each frame.
        - Maze position skeleton: the position of the animal in the skeleton maze (node label only) at each frame.
        - DLC features: the output of DeepLabCut analysis of the video frames.

    """
    dlc_df = get_cleaned_dlc_df(session_directory.dlc_path)
    bodypart_positions = get_dlc_positions_array(dlc_df)
    head_direction = hd.get_head_direction(dlc_df)
    centroid_positions = cpos.get_centroid_positions(bodypart_positions)
    x_pos = centroid_positions[0][:, 0]
    y_pos = centroid_positions[0][:, 1]
    pycontrol_times = get_frame_pytimes(session_directory.video_sync_path, session_directory.pycontrol_path)
    if len(pycontrol_times) - len(dlc_df) == 1:  # sometimes DLC output is missing the first/last frame
        pycontrol_times = pycontrol_times[:-1]
    # add maze trajectories (simple and skeleton)
    simple_maze_trajectory = mtraj.get_valid_simple_maze_trajectory(
        centroid_positions[0], maze_reps.get_simple_maze(session_directory.maze_name)
    )
    skeleton_maze_trajectory = mtraj.get_skeleton_maze_trajectory(
        centroid_positions[0], maze_reps.get_skeleton_maze(session_directory.maze_name)
    )
    # create features dataframe
    trajectories_df = pd.DataFrame(
        {
            ("time"): pycontrol_times,
            ("head_direction", "value"): head_direction[0],
            ("head_direction", "interpolated"): head_direction[1],
            ("centroid_position", "x"): x_pos,
            ("centroid_position", "y"): y_pos,
            ("centroid_position", "interpolated"): centroid_positions[1],
            ("maze_position", "simple"): simple_maze_trajectory,
            ("maze_position", "skeleton"): skeleton_maze_trajectory,
        }
    )
    # create a MultiIndex for the column names
    columns = pd.MultiIndex.from_tuples(
        [(col, "") if isinstance(col, str) else col for col in trajectories_df.columns],
        names=["feature", "measurement"],
    )
    trajectories_df.columns = columns
    return trajectories_df


def get_trial_info_df(session_directory):
    trials_df = get_trials_df(session_directory)
    frame_times = get_frame_pytimes(session_directory.video_sync_path, session_directory.pycontrol_path)
    dlc_df = get_cleaned_dlc_df(session_directory.dlc_path)
    if len(frame_times) - len(dlc_df) == 1:  # sometimes DLC output is missing the first/last frame
        frame_times = frame_times[:-1]
    session_trials = convert_times2trials(trials_df, frame_times)
    trial2goal = {k: v for k, v in zip(trials_df.trial.to_numpy(), trials_df.goal.to_numpy())}
    session_goals = pd.Series(session_trials).map(trial2goal).to_numpy()
    trial_phase = convert_times2trial_phases(trials_df, frame_times)
    trial_info_df = pd.DataFrame({"trial": session_trials, "trial_phase": trial_phase, "goal": session_goals})
    return trial_info_df


# %% Trial info df subfunctions
def convert_times2trials(trials_df, session_times):
    """Converts session times to trial numbers active at that timepoint"""
    trial_array = np.full(len(session_times), np.nan, dtype=float)
    trial_starts = trials_df["time"]["cue"].values
    trial_ends = trials_df["time"]["trial_end"].values
    trial_numbers = trials_df["trial"].values
    trial_indices = np.searchsorted(trial_starts, session_times, side="right") - 1
    valid_indices = np.logical_and(trial_indices >= 0, session_times <= trial_ends[trial_indices])
    trial_array[valid_indices] = trial_numbers[trial_indices[valid_indices]]
    return trial_array


def convert_times2trial_phases(trials_df, session_times):
    cue_times = trials_df.time["cue"].values
    reward_times = trials_df.time["reward"].values
    end_reward_consumption_times = trials_df.time["end_reward_consumption"].values
    trial_end_times = trials_df.time["trial_end"].values
    navigation_mask = np.logical_and(cue_times <= session_times[:, None], session_times[:, None] < reward_times)
    reward_consumption_mask = np.logical_and(
        reward_times <= session_times[:, None], session_times[:, None] < end_reward_consumption_times
    )
    ITI_mask = np.logical_and(
        end_reward_consumption_times <= session_times[:, None], session_times[:, None] <= trial_end_times
    )
    phases = np.full(len(session_times), np.nan, dtype="object")
    phases[navigation_mask.any(axis=1)] = "navigation"
    phases[ITI_mask.any(axis=1)] = "ITI"
    phases[reward_consumption_mask.any(axis=1)] = "reward_consumption"
    return phases


# %% velocity speed functions
def get_velocities(x, y, times):
    """Calculates velocity from x,y coordinates and times"""
    dt = np.diff(times)
    dx = np.diff(x)
    dy = np.diff(y)
    vx = dx / dt
    vy = dy / dt
    velocity = np.column_stack((vx, vy))
    # Linear extrapolation for the last velocity value
    last_vx = vx[-1] + (vx[-1] - vx[-2])
    last_vy = vy[-1] + (vy[-1] - vy[-2])
    last_velocity = np.array([last_vx, last_vy]).reshape(1, 2)
    velocity = np.append(velocity, last_velocity, axis=0)
    return velocity


def get_speeds(velocities):
    """Calculates speed from velocities"""
    return np.linalg.norm(velocities, axis=1)


# %% open dlc output files
def open_dlc_output_as_df(filepath, change_headers=False):
    """Read in DLC data from a csv file, and return a pandas dataframe with the data.
    Issue: #dlc output is shifted to make aspect ratio square? Hackfix by subtracting difference in true x,y pixels from the image
    """
    headers = []
    with open(filepath) as read_file:
        for row in csv.reader(read_file):
            if row[0] == "scorer":
                pass
            elif row[0] == "bodyparts":
                bodyparts = row
            else:
                if row[0] == "coords":
                    for index, value in enumerate(row):
                        headers.append(bodyparts[index] + "_" + value)
                break
        headers[0] = "frame"
        df = pd.read_csv(filepath, skiprows=3, names=headers).set_index("frame")
        part = df.columns.str.split("_").str[0]
        tracking = df.columns.str.split("_").str[1]
        if change_headers:
            part = part.map(PART_NAME_CHANGES)
        df.columns = pd.MultiIndex.from_tuples([(i, j) for i, j in zip(part, tracking)])
    for part in df.columns.get_level_values(0).unique():
        df[[(part, "y")]] = IMAGE_SIZE[1] - df[[(part, "y")]]  # change pixel coord origin to be bottom left
        df[[(part, "y")]] = df[[(part, "y")]] - np.diff(
            IMAGE_SIZE
        )  # HACK_FIX: dlc output is shifted to make aspect ratio square?
    return df


def translate_to_physical_coords(dlc_df):
    """Translate pixel coordinate x,y pixel positions in dlc_df to physical coordinates (meters)."""
    body_parts = dlc_df.columns.get_level_values(0).unique()
    for part in body_parts:
        pix_coords = dlc_df[[(part, "x"), (part, "y")]].to_numpy()
        physical_coords = pix2pos.translate_pixel2physical_coords(pix_coords)
        dlc_df[[(part, "x"), (part, "y")]] = physical_coords
    return dlc_df


def get_dlc_positions_array(dlc_df):
    """Returns a numpy array of shape (no_frames, no_bodyparts, 2) containing the x and y coordinates of each body part on each frame."""
    no_bodyparts = len(dlc_df.columns.get_level_values(0).unique())
    no_frames = len(dlc_df)
    return dlc_df.to_numpy().reshape((no_frames, no_bodyparts, 2))


# %% Choose outlier thresholds
def save_bodypart_outlier_thresholds():
    bodypart_outlier_thresholds = get_bodypart_outlier_thresholds()
    with open(EXPERIMENT_INFO_PATH / "bodypart_outlier_thresholds.json", "w") as fp:
        json.dump(bodypart_outlier_thresholds, fp, indent=4)


def get_bodypart_outlier_thresholds():
    """Defines the outlier threshold for each body part as 5x the estimated standard deviation (sd) of the distance between body parts and body center.
    sd is estimated as the interquartile range of the distribution of distances between each body part and the body center (median position of all body parts),
    over all frames from a representative set of videos."""
    filepaths = get_representative_dlc_filepaths()
    distances_dfs = pd.DataFrame()
    for file in filepaths:
        dlc_df = open_dlc_output_as_df(file, change_headers=PART_NAME_CHANGES)
        dlc_df = translate_to_physical_coords(dlc_df).drop("likelihood", axis=1, level=1)
        median_x = dlc_df.xs("x", axis=1, level=1).median(axis=1)
        median_y = dlc_df.xs("y", axis=1, level=1).median(axis=1)
        centroids = np.stack((median_x, median_y), axis=1)
        part2distances = {}
        for part in dlc_df.columns.get_level_values(0).unique():
            part_coords = dlc_df[[(part, "x"), (part, "y")]].to_numpy()
            distances = np.linalg.norm(part_coords - centroids, axis=1)
            part2distances[part] = distances
        distances_df = pd.DataFrame(part2distances)
        distances_dfs = pd.concat((distances_dfs, distances_df), axis=0, ignore_index=True)
    sd = (distances_df.quantile(0.886) - distances_df.quantile(0.114)) / 2
    threshold_multiplier = 5
    threshold_distances = distances_df.median() + sd * threshold_multiplier
    return dict(threshold_distances)


def get_representative_dlc_filepaths():
    """Returns a list of filepaths to DLC output files from the first and last video of each subject."""
    from .get_data_directory import get_sessions_data_directory

    sessions_data_directory = get_sessions_data_directory()
    dlc_paths = []
    for subject in sessions_data_directory.subject_ID.unique():
        subject_mask = sessions_data_directory["subject_ID"] == subject
        session_type_mask = sessions_data_directory["session_type"] == "maze"
        subject_filepath_df = sessions_data_directory[subject_mask & session_type_mask]
        dlc_paths.append(subject_filepath_df.iloc[np.argmin(subject_filepath_df.date)]["dlc_path"])
        dlc_paths.append(subject_filepath_df.iloc[np.argmax(subject_filepath_df.date)]["dlc_path"])
    return dlc_paths


# %% dlc_df cleaning functions
def remove_far_away_points(dlc_df):
    """Remove outliers in the input DataFrame by setting points farther than a threshold from the centroid to NaN."""
    median_x = dlc_df.xs("x", axis=1, level=1).median(axis=1)
    median_y = dlc_df.xs("y", axis=1, level=1).median(axis=1)
    centroids = np.stack((median_x, median_y), axis=1)
    for part in dlc_df.columns.get_level_values(0).unique():
        part_coords = dlc_df[[(part, "x"), (part, "y")]].to_numpy()
        distances = np.linalg.norm(part_coords - centroids, axis=1)
        qc_threshold = BODYPART_OUTLIER_THRESHOLDS[part]
        qc_mask = distances > qc_threshold
        part_coords[qc_mask] = np.nan
        dlc_df[[(part, "x"), (part, "y")]] = part_coords
    return dlc_df


def remove_low_likelihood_points(dlc_df):
    """Remove points with low likelihood (estimated by dlc) from the input DataFrame by setting them to NaN."""
    for part in dlc_df.columns.get_level_values(0).unique():
        part_coords = dlc_df[[(part, "x"), (part, "y")]].to_numpy()
        qc_mask = dlc_df[(part, "likelihood")] < DLC_LIKELIHOOD_THRESHOLD
        part_coords[qc_mask] = np.nan
        dlc_df[[(part, "x"), (part, "y")]] = part_coords
    return dlc_df


def get_cleaned_dlc_df(filepath):
    """Returns a cleaned version of the dlc_df from the input filepath."""
    dlc_df = open_dlc_output_as_df(filepath, change_headers=PART_NAME_CHANGES)
    dlc_df = translate_to_physical_coords(dlc_df)
    dlc_df = remove_low_likelihood_points(dlc_df)
    dlc_df = remove_far_away_points(dlc_df)
    return dlc_df.drop("likelihood", axis=1, level=1)


# %% convert video frames to pyControl seconds
def get_frame_pytimes(video_pinstate_filepath, pycontrol_filepath):
    """Get the pycontrol times corresponding to each frame in the video file."""
    video_sync_pulse_times = pd.read_csv(video_pinstate_filepath, header=None, names=["pinstate"])  # 60fps
    video_sync_pulse_frames = get_sync_pulse_frames(video_sync_pulse_times)
    pycontrol_sync_pulse_times = di.Session(pycontrol_filepath).times["rsync"] / 1000  # seconds
    pytime_videotime_aligner = rsync.Rsync_aligner(
        video_sync_pulse_frames, pycontrol_sync_pulse_times, units_A=1 / 60, units_B=1
    )
    frame_times = pytime_videotime_aligner.A_to_B(video_sync_pulse_times.index.to_numpy(), extrapolate=True)
    if any(np.isnan(frame_times)):
        frame_times = interpolate_missing_times(frame_times)
    return frame_times


def get_sync_pulse_frames(video_sync_pulse_times):
    """Get the frame numbers of the sync pulses in the video file.
    Each sync pulse corresponds to three frames. This function returns
    only the first from each tripplet. Ouput units are frames, recorded
    at 60frames/s."""
    sync_pulse_id = video_sync_pulse_times["pinstate"].value_counts().idxmin()
    return np.where(video_sync_pulse_times["pinstate"] == sync_pulse_id)[0][::3]


def interpolate_missing_times(frame_times):
    missing_inds = np.isnan(frame_times)
    frame_times[missing_inds] = np.interp(
        np.flatnonzero(missing_inds), np.flatnonzero(~missing_inds), frame_times[~missing_inds]
    )
    return frame_times


# %%
# if __name__ == '__main__':
#     DLC_PATH = '../data/raw_data/DeepLabCut'
#     save_bodypart_outlier_thresholds(DLC_PATH)
