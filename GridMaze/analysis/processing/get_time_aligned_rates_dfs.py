"""This file contains functions that build dataframes which organise processed data
   by subject, maze, session, trial, goal and neuron """

# %% Imports
import json
import numpy as np
import pandas as pd

from scipy.stats import norm
from .align_activity import align_spikes
from ..core import get_sessions as gs
from ..core import load_data

# %% Global variables
from ...paths import EXPERIMENT_INFO_PATH, ANALYSIS_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

try:
    with open(ANALYSIS_INFO_PATH / "intra_trial_interval_times.json", "r") as input_file:
        INTRA_TRIAL_INTERVAL_TIMES = json.load(input_file)
except FileNotFoundError:
    print(
        "No intra trial interval times found in analysis_info folder.\n"
        "Please run get_analysis_info.py to save out necessary data for this script.\n"
        "Then run save_analysis_info to save out the intra_trial_interval_times."
    )
    pass


# %% Main functions
def get_trial_aligned_rates_df(processed_data_path, analysis_data_path):
    """
    Generates a dataframe with trial-warped firing rates (raw or filtered)
    single unit firing rates for all trials in a list of sessions.
        -INPUT: session_path from preprocessing/analysis_data folder: eg, 'm8/2022-07-05-135156'
        -OUTPUT: pandas dataframe with event aligned unit rates organised by
                 subject, maze, session, trial, goal and neuron and trial-aligned
                 rates (Hz)

    """
    try:
        session_info = load_data.load(processed_data_path / "session_info.json")
        trials_df = load_data.load(processed_data_path / "trials.htsv")
    except FileNotFoundError:
        print(
            f" required processed data not available for {processed_data_path}, cannot generate trial_aligned_rates_df"
        )
        return None
    valid_trials_df = get_valid_trials_df(trials_df)
    if len(valid_trials_df) == 0:  # if no valid sessions can't make dataframe
        return None
    valid_trial2goal = dict(valid_trials_df[["trial", "goal"]].to_numpy())
    session_ID = gs.get_session_name(session_info)
    (
        trial_aligned_rates,
        trial_aligned_times,  # see get_trial_aligned_acitvity fn for how this is done
        min_max_stretch,
        trial_aligned_clusters,
    ) = get_trial_aligned_activity(processed_data_path).values()
    # loop over trials
    trial_rates_dfs = []
    for t, trial_no in enumerate(valid_trial2goal.keys()):
        trial_info = pd.DataFrame(
            {
                "subject_ID": session_info["subject_ID"],
                "maze_name": session_info["maze_name"],
                "day_on_maze": session_info["day_on_maze"],
                "trial": trial_no,
                "goal": valid_trial2goal[trial_no],
                "cluster_unique_ID": [f"{session_ID}_cluster{int(cID)}" for cID in trial_aligned_clusters],
                "cluster_ID": trial_aligned_clusters.astype(int),
            }
        )
        # add cluster type info for units that spike in trial
        trial_info.columns = pd.MultiIndex.from_product(
            [
                [
                    "subject_ID",
                    "maze_name",
                    "day_on_maze",
                    "trial",
                    "goal",
                    "cluster_unique_ID",
                    "cluster_ID",
                ],
                [""],
            ],
        )
        trial_activity = trial_aligned_rates[t, :, :]
        trial_rates = pd.DataFrame(
            data=trial_activity,
            columns=pd.MultiIndex.from_product([["firing_rate"], trial_aligned_times]),
        )
        trial_rates_dfs.append(pd.concat([trial_info, trial_rates], axis=1))
    session_rates_df = pd.concat(trial_rates_dfs, axis=0).reset_index(drop=True)
    # add alignmnet stretch info
    min_max_stretch_column = np.repeat(min_max_stretch, len(trial_aligned_clusters), axis=0)
    session_rates_df[("stretch", "min")] = min_max_stretch_column[:, 0]
    session_rates_df[("stretch", "max")] = min_max_stretch_column[:, 1]
    return session_rates_df


def get_event_aligned_rates_df(processed_data_path, analysis_data_path, window_size=15, sampling_rate=25):
    """
    Generates a dataframe with cue_aligned and reward_aligned single unit
    firing rates (raw or filtered) for all trials in a list of sessions.
        -INPUT: session_path from preprocessing/analysis_data folder: eg, 'm8/2022-07-05-135156'
        -OUTPUT: pandas dataframe with event aligned unit rates with units
                 organised subject, maze, session, trial, goal and neuron and
                 rates organised by cue_aligned, reward_aligned and raw
                 (firing_rate) and filtered (filtered_rate)
    """
    try:
        session_info = load_data.load(processed_data_path / "session_info.json")
        trials_df = load_data.load(processed_data_path / "trials.htsv")
    except FileNotFoundError:
        print(
            f" required processed data not available for {processed_data_path}, cannot generate event_aligned_rates_df"
        )
        return None
    if len(trials_df) == 0:  # if no valid sessions can't make dataframe
        return None
    valid_trial2goal = dict(trials_df[["trial", "goal"]].to_numpy())
    session_ID = gs.get_session_name(session_info)
    (
        cue_aligned_rates,
        cue_aligned_times,
        cue_aligned_clusters,
    ) = get_event_aligned_activity(
        processed_data_path, event="cue", window_size=window_size, fs_out=sampling_rate
    ).values()
    (
        reward_aligned_rates,
        reward_aligned_times,
        _,
    ) = get_event_aligned_activity(
        processed_data_path,
        event="reward",
        window_size=window_size,
        fs_out=sampling_rate,
    ).values()
    # loop over trials
    trial_dfs = []
    for t, trial_no in enumerate(valid_trial2goal.keys()):
        trial_info = pd.DataFrame(
            {
                "subject_ID": session_info["subject_ID"],
                "maze_name": session_info["maze_name"],
                "day_on_maze": session_info["day_on_maze"],
                "trial": trial_no,
                "goal": valid_trial2goal[trial_no],
                "cluster_unique_ID": [
                    f"{session_ID}_cluster{int(cID)}" for cID in cue_aligned_clusters
                ],  # cue and reward aligned clusters are the same
                "cluster_ID": cue_aligned_clusters.astype(int),
            }
        )
        trial_info.columns = pd.MultiIndex.from_product(
            [
                [
                    "subject_ID",
                    "maze_name",
                    "day_on_maze",
                    "trial",
                    "goal",
                    "cluster_unique_ID",
                    "cluster_id",
                ],
                [""],
                [""],
            ],
            names=["datatype", "aligned_event", "timepoint"],
        )
        cue_aligned_activity = cue_aligned_rates[t, :, :]
        cue_aligned_df = pd.DataFrame(
            data=cue_aligned_activity,
            columns=pd.MultiIndex.from_product([["firing_rate"], ["cue_aligned"], cue_aligned_times.tolist()]),
        )
        reward_aligned_activity = reward_aligned_rates[t, :, :]
        reward_aligned_df = pd.DataFrame(
            data=reward_aligned_activity,
            columns=pd.MultiIndex.from_product([["firing_rate"], ["reward_aligned"], reward_aligned_times.tolist()]),
        )
        trial_dfs.append(pd.concat([trial_info, cue_aligned_df, reward_aligned_df], axis=1))
    event_aligned_rates_df = pd.concat(trial_dfs, axis=0).reset_index(drop=True)
    return event_aligned_rates_df


# %% Sub-functions


def get_valid_trials_df(trials_df):
    valid_trials_df = trials_df.time.drop("ITI_start", axis=1)
    valid_trials_df = trials_df[valid_trials_df.diff(axis=1)["trial_end"] > 0].reset_index(drop=True)
    return valid_trials_df


# %% Aligned Rates Functions


def get_trial_aligned_activity(processed_data_path):
    """warps firing rates around trial events such that the warped timecourse of each trial is the same across all anaimals, session
    and trials
    - see aligned_activity.py for further details"""
    trials_df = load_data.load(processed_data_path / "trials.htsv") * 1000  # align spikes expects times in ms
    trial_times = get_valid_trial_times(trials_df)
    spike_clusters = load_data.load(processed_data_path / "spikes.clusters.npy").reshape(-1)
    spike_times = load_data.load(processed_data_path / "spikes.times.npy").reshape(-1) * 1000  # spike times in ms
    spikes = np.vstack((spike_clusters, spike_times))
    itt = np.multiply(list(INTRA_TRIAL_INTERVAL_TIMES.values()), 1000)  # intratrial interval times in ms
    aligned_activity = align_spikes(trial_times, itt, spikes, pre_win=5000, plot=False)  # require time units  in ms
    trial_aligned_rates = aligned_activity["aligned_rates"]
    t_out = aligned_activity["t_out"] / 1000  # convert to seconds
    min_max_stretch = aligned_activity["min_max_stretch"]
    cluster_IDs = aligned_activity["cluster_IDs"]
    return {
        "trial_aligned_rates": trial_aligned_rates,
        "t_out": t_out,  # seconds
        "min_max_stretch": min_max_stretch,
        "cluster_IDs": cluster_IDs,
    }


def get_event_aligned_activity(processed_data_path, event, window_size=15, fs_out=25, smooth_SD="default"):
    """aligns firing rates on each trial to a particular event (eg, cue or reward).
    INPUTS:
        - processed_data_path: processed data folder for the session containing: trials.tsv, spike_clusters.npy, spike_times.npy
        - event: trial event you want to align to, either 'cue','reward', 'end_reward_consumption', 'ITI_start' or 'trial_end'
        - window_size: how many seconds pre and post event you want to capture in the alignment
        - fs_out: ampling rate of output firing rate (Hz)
        - smooth_SD: Standard deviation of gaussian smoothing applied to ouput rate.
                     If set to default, smooth_SD is set to the inter sample interval.
    OUTPUTS:
        - event_aligned_rates: array of size [n_trials, n_neurons, n_timepoints], containing event aligned firing rates
                                each neuron on each trial
        - t_out: times of each output firing rate time point (s), where t=0=event
        - cluster_IDs: neuron_ID output by kilosort
    """
    if smooth_SD == "default":
        smooth_SD = 1 / fs_out
    trials_df = load_data.load(processed_data_path / "trials.htsv")
    trial_times = trials_df.time.to_numpy()  # no need to filter out trials with erc after next cue
    event2event_ind = {event: i for i, event in enumerate(trials_df.time.columns)}
    event_ind = event2event_ind[event]
    spike_clusters = load_data.load(processed_data_path / "spikes.clusters.npy").reshape(-1)
    spike_times = load_data.load(processed_data_path / "spikes.times.npy").reshape(-1)
    spikes = np.vstack(
        (
            spike_clusters,
            spike_times,
        )
    )
    n_trials = trial_times.shape[0]
    cluster_IDs = np.sort(np.unique(spike_clusters)).astype(np.float64)
    n_clusters = len(cluster_IDs)
    t_out = np.arange(-window_size, window_size, 1 / fs_out)
    pad_len = smooth_SD * 4
    event_aligned_rates = np.zeros([n_trials, n_clusters, len(t_out)])
    for tr in range(n_trials):  # Loop over trials.
        event_aligned_spikes = spikes[
            :,
            (trial_times[tr, event_ind] - window_size - pad_len < spikes[1, :])
            & (spikes[1, :] < trial_times[tr, event_ind] + window_size + pad_len),
        ]
        event_aligned_spike_IDs = event_aligned_spikes[0, :]
        event_aligned_spike_times = event_aligned_spikes[1, :] - trial_times[tr, event_ind]
        for j, n in enumerate(cluster_IDs):  # Loop over clusters.
            if n in event_aligned_spike_IDs:
                neuron_mask = event_aligned_spike_IDs == n
                n_spike_times = event_aligned_spike_times[neuron_mask]
                event_aligned_rates[tr, j, :] = np.sum(
                    norm.pdf(n_spike_times[None, :] - t_out[:, None], scale=smooth_SD),
                    axis=1,
                )
    return {
        "event_aligned_rates": event_aligned_rates,
        "t_out": t_out,
        "cluster_IDs": cluster_IDs,
    }


def get_valid_trial_times(trials_df):
    """Returns a numpy array ([trials, 4] times of cue, reward, end of reward consumption (erc) and
    end of trial for trials in which erc occurs before the next cue."""
    trial_times = trials_df.time.drop("ITI_start", axis=1)
    trial_times = trial_times[trial_times.diff(axis=1)["trial_end"] > 0].reset_index(drop=True).to_numpy()
    return trial_times


# %%
def get_av_intra_trial_times():
    """finds the average intra-trial interval times (ITT, seconds) across the experient:
    cue_on->reward, reward->end_reward_consumption, erc->end_trial
    - first finds median interval times for each subject on each session
    - then finds median interval times for each subject across sessions
    - finaly takes average interval times across subjects = global ITT av
    """
    subject_ITTs = []
    for subject in SUBJECT_IDS:
        sessions = gs.get_maze_sessions(subject_IDs=[subject], with_data=["trials_df"], must_have_data=True)
        session_ITTs = []
        for session in sessions:
            trials_df = session.trials_df
            trial_times = get_valid_trial_times(trials_df)
            trial_ITT = np.diff(trial_times)
            median_ITT = np.nanmedian(trial_ITT, axis=0)
            session_ITTs.append(np.concatenate([[0], np.cumsum(median_ITT)]))
        subject_ITTs.append(np.nanmedian(np.stack(session_ITTs), axis=0))
    av_ITTs = np.stack(subject_ITTs).mean(axis=0)
    return {x: time for x, time in zip(["cue", "reward", "end_reward_consumption", "ITI_end"], av_ITTs)}
