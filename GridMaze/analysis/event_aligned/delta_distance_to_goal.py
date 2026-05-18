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
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS2_PATH

with open(EXPERIMENT_INFO_PATH / "maze_configs.json", "r") as input_file:
    MAZE_CONFIGS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "maze_day2date.json", "r") as input_file:
    MAZE_DAY2DATE = json.load(input_file)

FRAME_RATE = 60

# %% dev


def get_all_sessions_ddtg_at_cue(window=(1, 3), overwrite=False, multi_index=False):
    """ """
    save_path = RESULTS2_PATH / "event_aligned" / "delta_distance_to_goal" / f"cue_aligned_ddtg.htsv"
    if not overwrite and save_path.exists():
        cue_ddtg_df = pd.read_csv(save_path, sep="\t")
    else:
        ddtgs = []
        sessions = gs.get_maze_sessions(
            subject_IDs="all",
            maze_names="all",
            days_on_maze="all",
            with_data=["navigation_df"],
        )
        for session in sessions:
            df = get_session_delta_dtg(session, window=window)
            ddtg_df = df.loc[:, [("subject_ID", ""), ("maze_name", ""), ("day_on_maze", ""), ("trial", "")]]
            ddtg_df["ddtg"] = df.cue_aligned_time.mean(1).values  # mean ddtg in first s seconds of each trial
            ddtgs.append(ddtg_df)
        cue_ddtg_df = pd.concat(ddtgs, axis=0).reset_index(drop=True)
        # save
        cue_ddtg_df.to_csv(save_path, sep="\t", index=False)
    if multi_index:
        cue_ddtg_df.columns = pd.MultiIndex.from_product([cue_ddtg_df.columns, [""]])
    return cue_ddtg_df.dropna().reset_index(drop=True)


# %% Functions


def get_session_delta_dtg(session, window=(-3, 3)):  # dtg = distance to goal
    """"""
    navigation_df = session.navigation_df
    skeleton_maze = session.skeleton_maze()
    window_frames = (window[0] * FRAME_RATE, window[1] * FRAME_RATE)
    skeleton_label2skeleton_coord = {v: k for k, v in nx.get_node_attributes(skeleton_maze, "label").items()}
    shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(skeleton_maze, weight="weight"))
    trial_dD_dts = []
    trials = navigation_df.dropna().trial.unique()

    keep_trials = []
    for trial in trials:
        trial_df = navigation_df[navigation_df.trial == trial]
        goal = trial_df.goal.unique()[0]
        goal_coord = skeleton_label2skeleton_coord[goal + "_C"]
        cue_indx = trial_df.index[0]
        sk_locations = navigation_df.iloc[
            cue_indx + window_frames[0] : cue_indx + window_frames[1] + 1
        ].maze_position.skeleton.to_numpy()
        if len(sk_locations) == 0:
            continue  # window out of session bounds
        sk_coords = [skeleton_label2skeleton_coord[loc] for loc in sk_locations]
        geo_distance_to_goal = np.array([shortest_path_lengths[c][goal_coord] for c in sk_coords])
        dD_dt = np.diff(geo_distance_to_goal) * FRAME_RATE  # m/s
        trial_dD_dts.append(dD_dt)
        keep_trials.append(trial)
    # return as df
    info_df = pd.DataFrame(index=range(len(keep_trials)))
    info_df[("subject_ID", "")] = session.subject_ID
    info_df[("maze_name", "")] = session.maze_name
    info_df[("day_on_maze", "")] = session.day_on_maze
    info_df[("trial", "")] = keep_trials
    info_df.columns = pd.MultiIndex.from_tuples(info_df.columns)
    times = np.arange(window[0], window[1], 1 / FRAME_RATE)
    ddtg_df = pd.DataFrame(
        data=np.vstack(trial_dD_dts), columns=pd.MultiIndex.from_product([["cue_aligned_time"], times])
    )
    return pd.concat([info_df, ddtg_df], axis=1)


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
    save_path = RESULTS2_PATH / "event_aligned" / "delta_distance_to_goal" / "rate_of_change_of_distance_to_goal_df.htsv"
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
    ax=None,
    plot_single_subjects=False,
    window_length=5,
    smooth_SD=5,
    color="rosybrown",
):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(5, 5), clear=True)
    ax.spines[["top", "right"]].set_visible(False)

    ddtg_df = get_delta_dtg_df(window_length=window_length, smooth_SD=smooth_SD)
    av_dD_dt = ddtg_df.mean(axis=0)
    sem_dD_dt = ddtg_df.sem(axis=0)
    time = ddtg_df.columns.values.astype(float)
    # plot figure
    if plot_single_subjects:
        for s in SUBJECT_IDS:
            d = ddtg_df.loc[s].values
            ax.plot(time, d, color=color, lw=0.5, alpha=0.5)
    ax.plot(time, av_dD_dt, color=color, lw=2)
    ax.fill_between(
        time,
        av_dD_dt - sem_dD_dt,
        av_dD_dt + sem_dD_dt,
        color=color,
        alpha=0.3,
    )
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Cue (s)")
    ax.set_ylabel("Δ Dist. to goal \n (m/s)")
