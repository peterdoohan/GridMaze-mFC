"""This file registers video pixel coordinates of select towers on a maze for later use to correct for fish-eye distortion. A quality control step is included
to ensure that the camera does not move over the recording period."""

# %% imports
import os
import ast
import json
import cv2
import numpy as np
import pandas as pd
from datetime import datetime as dt
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
from ..maze.representations import _get_node_positions_dict, get_simple_nodes_dict

from ..paths import DATA_PATH
# %% Global variables
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
ALIGNMENT_POINTS = ["A1", "A4", "A7", "C3", "C5", "D1", "D4", "D7", "E3", "E5", "G1", "G4", "G7"]

RAW_VIDEO_PATH = DATA_PATH / "raw_data" / "video"

with open("../data/experiment_info/maze_configs.json") as input_file:
    MAZE_CONFIGS = json.load(input_file)

EXPERIMENT_INFO_PATH = Path("../data/experiment_info")

MAZE_REGISTRATION_FILE = EXPERIMENT_INFO_PATH / "maze_registration.tsv"


# %% Main Function
def get_maze_registration_df(maze_registration_path=MAZE_REGISTRATION_FILE):
    """Load maze registration file if it exists, otherwise run registration. Returns dataframe with the pixel and physical coordinates
    of each maze tower used for later alignmnet steps."""
    try:
        maze_registration_df = pd.read_csv(str(maze_registration_path), sep="\t")
        # convert saved strings to floats
        maze_registration_df["pixel_coords"] = maze_registration_df["pixel_coords"].apply(ast.literal_eval)
        maze_registration_df["physical_coords"] = maze_registration_df["physical_coords"].apply(ast.literal_eval)
    except FileNotFoundError:
        print(f"No maze registration file found in {EXPERIMENT_INFO_PATH}. Running registration now.")
        maze_registration_df = run_maze_registration()
    return maze_registration_df


def run_maze_registration(n_click_replicates=3, plot=True, force_save=False):
    """
    Performs maze registration process with user input.
    Steps:
        1. Get sample videos to compare camera alignment across sessions, see get_sample_maze_videos(). Note this function
        can be adapted to select videos arbitrarily.
        2. For each video, get pixel coordinates of select towers specified in ALIGNMENT_NODES (repeated 3 times per video
        to get an estimate of click variance).
        3. Calculate the mean and standard deviation of the pixel coordinates for each tower across replicates.
        4. Plot the pixel coordinates of each tower for each video to visually check for consistency.
        5. Run a quality control check to ensure that the camera alignment is consistent across registration sessions. This
        is done by comparing the standard deviation of the distance between the pixel coordinates of the same tower in different
        registration sessions to the mean standard deviation of the pixel coordinates of the same tower across registration sessions.
        6. If the camera alignment is consistent, save the average pixel coordinates of each tower to a .tsv file in the raw video folder
        as maze_registration.tsv. If the camera alignment is not consistent, multiple registrations might be needed (not implemented here.)

    Args:
    - n_click_replicates: int, default=3, number of times to repeat the click registration process for each video.
    - plot: bool, default=True, whether to plot the pixel coordinates of each tower for each video.
    - force_save: bool, default=False, whether to save the average pixel coordinates of each tower to a .tsv file in the raw video folder

    Notes:
    - If running this function in a jupyter notebook, use the magic command %matplotlib qt to enable interactive plotting.
    """
    sample_video_paths = get_sample_videos()
    registration_click_dfs = []
    for video_path in sample_video_paths:
        tower2pixel_coord_replicates = []
        for _ in range(n_click_replicates):
            tower2pixel_coord_replicates.append(get_alignment_point_pixel_coords(video_path, ALIGNMENT_POINTS))
        registration_click_df = pd.DataFrame(tower2pixel_coord_replicates).apply(calculate_replicate_mean_and_std)
        registration_click_dfs.append(registration_click_df)
    if plot:
        plot_click_registration(registration_click_dfs)
    # quality control
    session_variance = get_cross_session_variance(registration_click_dfs)
    click_variance = pd.concat(registration_click_dfs, axis=1).mean(axis=1)
    click_variance = np.mean([click_variance["std_x"], click_variance["std_y"]])
    if session_variance < 10 * click_variance or force_save:
        print("camera alignment is consitent across registration sessions, saving out results.")
        # convert to usefull dataframe
        alignment_point2pixel_coords = get_average_alignment_coords(registration_click_dfs)
        maze_registration_df = _get_maze_registration_df(alignment_point2pixel_coords)
        # save
        maze_registration_df.to_csv(EXPERIMENT_INFO_PATH / "maze_registration.htsv", sep="\t", index=False)
    else:
        print(
            "camera alignment is not consistent across registration sessions. Multiple registrations might be needed."
        )


def get_sample_videos():
    """
    Returns a list of video filepaths from the last day of each maze config in the experiment.
    Later use these sample videos to check if camera alignment is conistent over the experiment.
    This function can be changed to return videos arbitrarily.
    """
    all_video_paths = [
        p
        for p in RAW_VIDEO_PATH.iterdir()
        if p.suffix == ".mp4"
    ]
    video_path_datetimes = [
        dt.strptime(p.name.split("_")[-1].split(".")[0], "%Y-%m-%d-%H%M%S") for p in all_video_paths
    ]
    sample_video_paths = []
    for maze_name in MAZE_CONFIGS.keys():
        end_date = dt.fromisoformat(MAZE_CONFIGS[maze_name]["end"])
        difference_from_end_date = [abs(i - end_date) for i in video_path_datetimes]
        sample_video_paths.append(
            all_video_paths[difference_from_end_date.index(min(difference_from_end_date))]
        )
    return sample_video_paths


def _get_maze_registration_df(alignment_point2pixel_coords):
    """Returns a dataframe with pixel and physical coordinates for each tower, relies on
    having run maze registration and having the tower pixel coordinates saved under
    MAZE_REGISTRATION_FILE path variable."""
    alignment_point2simple_node = get_simple_nodes_dict()
    simple_node2physical_coords = _get_node_positions_dict()
    alignment_point2info = [
        {
            "alignment_point": p,
            "pixel_coords": alignment_point2pixel_coords[p],
            "physical_coords": simple_node2physical_coords[alignment_point2simple_node[p]],
        }
        for p in ALIGNMENT_POINTS
    ]
    maze_registration_df = pd.DataFrame(alignment_point2info)
    return maze_registration_df


# %% Supporting Functions


def get_alignment_point_pixel_coords(video_path, alignment_points):
    video = cv2.VideoCapture(str(video_path))
    temp_image_path = RAW_VIDEO_PATH / "temp_image.png"
    ret, first_frame = video.read()
    cv2.imwrite(str(temp_image_path), first_frame)
    image = cv2.imread(str(temp_image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    alignment_point2pixel_coord = {}
    for point in alignment_points:
        print(f"Click on {point}")
        pixel_coord = get_pixel_coords_from_image(image, label=point)[0]
        alignment_point2pixel_coord[point] = pixel_coord
    # remove temp image
    temp_image_path.unlink()
    return alignment_point2pixel_coord


def calculate_replicate_mean_and_std(column):
    x_values = [item[0] for item in column]
    y_values = [item[1] for item in column]
    return pd.Series(
        {"mean_x": np.mean(x_values), "mean_y": np.mean(y_values), "std_x": np.std(x_values), "std_y": np.std(y_values)}
    )


def plot_click_registration(registration_click_dfs, scaling_factor=5):
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    pallet = sns.color_palette("tab10", n_colors=len(registration_click_dfs))
    for reg_df, color in zip(registration_click_dfs, pallet):
        for tower in reg_df.columns:
            mean_x = reg_df.loc["mean_x", tower]
            mean_y = reg_df.loc["mean_y", tower]
            std_x = reg_df.loc["std_x", tower]
            std_y = reg_df.loc["std_y", tower]
            radius = max(std_x, std_y) * scaling_factor
            radius = 1 if radius == 0 else radius
            circle = plt.Circle((mean_x, mean_y), radius, color=color, alpha=0.5)
            ax.add_patch(circle)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_xlim(0, 1200)  # Set appropriate limits for x and y axis based on your data
    ax.set_ylim(0, 1000)
    return


def get_cross_session_variance(registration_click_dfs):
    """Returns the standard deviation of the distance between the pixel coordinates of the same tower in a list of tower_coord_dicts"""
    n = len(registration_click_dfs)
    towers = registration_click_dfs[0].columns
    permutations = [(i, j) for i in range(n) for j in range(n) if i < j]
    offset_distances = []
    for a, b in permutations:
        for t in towers:
            pos_a = (registration_click_dfs[a][t]["mean_x"], registration_click_dfs[a][t]["mean_y"])
            pos_b = (registration_click_dfs[b][t]["mean_x"], registration_click_dfs[b][t]["mean_y"])
            offset_distance = euclidean(pos_a, pos_b)
            offset_distances.append(offset_distance)
    return np.std(offset_distances)


def get_average_alignment_coords(registration_click_dfs):
    combined_click_dfs = pd.concat(registration_click_dfs)
    tower_x_means = combined_click_dfs.loc["mean_x"].mean()
    tower_y_means = combined_click_dfs.loc["mean_y"].mean()
    average_tower_coords = {tower: (tower_x_means[tower], tower_y_means[tower]) for tower in tower_x_means.keys()}
    return average_tower_coords


def get_pixel_coords_from_image(image, label):
    """Returns pixel coordinates of a click on an image"""
    plt.figure(figsize=(15, 10))
    orientation = [0, image.shape[1], 0, image.shape[0]]
    plt.imshow(image, extent=orientation)
    plt.title(f"click on the center of {label}")
    plt.tight_layout()
    plt.axis("off")
    plt.show()
    pixel_coords = plt.ginput(n=1, timeout=0, show_clicks=True)
    plt.close()
    return pixel_coords


# %% Other


def get_image_size_from_video():
    """Returns the image size of the videos in the raw_data directory, using 1st frame from the 1st video as an example image
    Output is (height, width)"""
    example_video_path = str(get_sample_videos()[0])
    temp_image_path = os.path.join(RAW_VIDEO_PATH, "temp_image.png")
    video = cv2.VideoCapture(example_video_path)
    temp_image_path = os.path.join(RAW_VIDEO_PATH, "temp_image.png")
    ret, first_frame = video.read()
    if ret:
        cv2.imwrite(temp_image_path, first_frame)
    else:
        print(f"Could not read first frame from {example_video_path}")
    image = cv2.imread(temp_image_path)
    image_size = image.shape[:2]
    os.remove(temp_image_path)
    return image_size