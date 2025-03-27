"""
Library for calculating rate of change of distance to goal.
Used to inform decoding analyses & maybe in the future isolate the moment after cue where,
animals lock onto the goal
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import networkx as nx
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt


from GridMaze.analysis.core import get_sessions as gs

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with open(EXPERIMENT_INFO_PATH / "maze_configs.json", "r") as input_file:
    MAZE_CONFIGS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

FRAME_RATE = 60

# %% Functions


def get_session_delta_dtg(session, cue_window=3):  # dtg = distance to goal
    """"""
    navigation_df = session.navigation_df
    skeleton_maze = session.skeleton_maze()
    window_frames = cue_window * FRAME_RATE
    skeleton_label2skeleton_coord = {v: k for k, v in nx.get_node_attributes(skeleton_maze, "label").items()}
    shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(skeleton_maze, weight="weight"))
    trial_dD_dts = []
    trials = navigation_df.dropna().trial.unique()
    for trial in trials:
        trial_df = navigation_df[navigation_df.trial == trial]
        goal = trial_df.goal.unique()[0]
        goal_coord = skeleton_label2skeleton_coord[goal + "_C"]
        cue_indx = trial_df.index[0]
        sk_locations = navigation_df.iloc[
            cue_indx - window_frames : cue_indx + window_frames + 1
        ].maze_position.skeleton.to_numpy()
        if len(sk_locations) == 0:
            continue  # window out of session bounds
        sk_coords = [skeleton_label2skeleton_coord[loc] for loc in sk_locations]
        geo_distance_to_goal = np.array([shortest_path_lengths[c][goal_coord] for c in sk_coords])
        dD_dt = np.diff(geo_distance_to_goal) * FRAME_RATE  # m/s
        trial_dD_dts.append(dD_dt)
    return np.vstack(trial_dD_dts)  # [trials, timepoints]


def get_subject_detla_dtg(subject, window_length):
    sessions = gs.get_maze_sessions(
        subject_IDs=[subject],
        maze_names="all",
        days_on_maze="late",
        with_data=["navigation_df"],
    )
    subject_dD_dts = []
    for session in sessions:
        session_dD_dts = get_session_delta_dtg(session, cue_window=window_length)
        subject_dD_dts.append(session_dD_dts)
    return np.vstack(subject_dD_dts)  # [sessions * trials, timepoints]


def get_delta_dtg_df(window_length=5, smooth_SD=5):
    """ """
    save_path = RESULTS_PATH / "behaviour" / "rate_of_change_of_distance_to_goal_df.htsv"
    if save_path.exists():
        ddtg_df = pd.read_csv(save_path, sep="\t", index_col=0)
    else:
        dD_dts = []
        for subject in SUBJECT_IDS:
            subject_dD_dts = get_subject_detla_dtg(subject, window_length=window_length)
            av_subject_dD_dt = np.mean(gaussian_filter1d(subject_dD_dts, sigma=smooth_SD), axis=0)
            dD_dts.append(np.array(av_subject_dD_dt))
        dD_dt = np.array(dD_dts)
        time = np.linspace(-window_length, window_length, dD_dt.shape[1])
        ddtg_df = pd.DataFrame(data=dD_dt, index=SUBJECT_IDS, columns=time)
        # save
        ddtg_df.to_csv(save_path, sep="\t", index=True)
    return ddtg_df


def plot_cross_subject_rate_of_change_of_distance_to_goal(
    ax=None, plot_single_subjects=False, window_length=5, smooth_SD=5
):
    ddtg_df = get_delta_dtg_df(window_length=window_length, smooth_SD=smooth_SD)
    av_dD_dt = ddtg_df.mean(axis=0)
    sem_dD_dt = ddtg_df.sem(axis=0)
    time = ddtg_df.columns.values.astype(float)
    # plot figure
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(5, 5), clear=True)
    if plot_single_subjects:
        for s in SUBJECT_IDS:
            d = ddtg_df.loc[s].values
            ax.plot(time, d, color="yellowgreen", lw=0.5, alpha=0.5)
    ax.plot(time, av_dD_dt, color="darkolivegreen", lw=2)
    ax.fill_between(
        time,
        av_dD_dt - sem_dD_dt,
        av_dD_dt + sem_dD_dt,
        color="darkolivegreen",
        alpha=0.3,
    )
    ax.axvline(0, color="silver", linestyle="--")
    ax.set_xlabel("Cue-aligned Time (s)")
    ax.set_ylabel("Rate of Change of Geodesic Distance to Goal (m/s)")
    f.tight_layout()
