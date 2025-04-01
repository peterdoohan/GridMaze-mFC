"""
Library for event aligned LFP analysis.
@peterdoohan
"""

# %% Imports
import pandas as pd
import numpy as np
import json
import mne
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import load_data
from scipy.stats import zscore
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from scipy.signal import welch
from mne.time_frequency import psd_array_welch
from mne.time_frequency import psd_array_multitaper

from GridMaze.analysis.event_aligned import delta_distance_to_goal as ddtg

from GridMaze.analysis.event_aligned import lfp_utils as lu

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

LFP_RESULTS = RESULTS_PATH / "event_aligned" / "lfp"
if not LFP_RESULTS.exists():
    LFP_RESULTS.mkdir(parents=True)

with open(EXPERIMENT_INFO_PATH / "maze_configs.json", "r") as input_file:
    MAZE_CONFIGS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "maze_day2date.json", "r") as input_file:
    MAZE_DAY2DATE = json.load(input_file)

FS = 1500  # lfp sampling frequency

# %% Main Functions


def plot_exp_average_spectrograms(signal_type, axes=None, smooth_SD=False):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), width_ratios=[0.9, 1])
    if signal_type == "CSD":
        save_path = LFP_RESULTS / "aligned_spectrograms_CSD.parquet"
    elif signal_type == "LFP":
        save_path = LFP_RESULTS / "aligned_spectrograms_LFP.parquet"
    else:
        NotImplementedError

    df = load_data._load_multiindex_parquet(save_path)
    # issue with freqs having diffent precisions HACK fix
    df[("frequencies", "")] = df.frequencies.apply(lambda f: round(f, 5))
    df = df[df.late_session]
    cue_df = df.groupby("frequencies").cue_aligned_time.mean()
    reward_df = df.groupby("frequencies").reward_aligned_time.mean()
    # get common max, min values
    _max = max(cue_df.max().max(), reward_df.max().max())
    _min = min(cue_df.min().min(), reward_df.min().min())
    freqs = cue_df.index.values
    times = df.cue_aligned_time.columns.astype("float")
    events = ["cue", "reward"]
    for i, _df in enumerate([cue_df, reward_df]):
        spec = _df.values
        if smooth_SD:
            spec = gaussian_filter(spec, sigma=smooth_SD)
        cbar = True if i == 1 else False
        _plot_spectrogram(spec, times, freqs, ax=axes[i], _min=_min, _max=_max, colorbar=cbar)
        axes[i].set_title("")
        axes[i].set_xlabel(f"{events[i]}-aligned time (s)")
        if i == 1:
            axes[1].set_ylabel("")
    fig.tight_layout()


def load_spectrogram_dfs_from_disk(signal_type="CSD", subject_IDs="all"):
    save_paths = list((LFP_RESULTS / "aligned_spectrograms" / signal_type).iterdir())
    if subject_IDs != "all":
        save_paths = [p for p in save_paths if p.name.split(".")[0] in subject_IDs]
    dfs = []
    for p in save_paths:
        print(f"loading {p.name}")
        dfs.append(load_data._load_multiindex_parquet(p))
    return dfs


# %% Plotting functions (TODO: refac)


def _plot_spectrogram_from_df(df, axes=None, window=False):
    """ """
    _df = df.copy()
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), width_ratios=[0.9, 1])
    if window:
        new_columns = []
        for col in _df.columns:
            if col[0] in ["cue_aligned_time", "reward_aligned_time"]:
                time_val = float(col[1])
                if window[0] <= time_val <= window[1]:
                    new_columns.append(col)
            else:
                new_columns.append(col)  # Keep metadata columns
        _df = _df[new_columns]
    # get common max, min values
    _max = _df[["cue_aligned_time", "reward_aligned_time"]].max().max()
    _min = _df[["cue_aligned_time", "reward_aligned_time"]].min().min()
    freqs = _df.frequencies.values
    for i, event in enumerate(["cue", "reward"]):
        times = _df[f"{event}_aligned_time"].columns.astype("float")
        spec = _df[f"{event}_aligned_time"].values
        cbar = True if i == 1 else False
        _plot_spectrogram(spec, times, freqs, ax=axes[i], _min=_min, _max=_max, colorbar=cbar)
        axes[i].set_title("")
        axes[i].set_xlabel(f"{event}-aligned time (s)")
        if i == 1:
            axes[1].set_ylabel("")
    fig.tight_layout()


def _plot_spectrogram(x, times, freqs, ax=None, _min=None, _max=None, colorbar=True):
    """ """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    _min = x.min() if _min is None else _min
    _max = x.max() if _max is None else _max
    pcmesh = ax.pcolormesh(times, freqs, x, shading="auto", cmap="coolwarm", vmin=_min, vmax=_max)
    ax.axvline(0, color="white", linestyle="--")
    ax.grid(False)
    ax.set_yscale("log")
    ax.set_title(f"Wavelet Decomposition")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    if colorbar:
        cbar = plt.colorbar(pcmesh, ax=ax, orientation="vertical")
        cbar.set_label("Power (z-scored)")
        for spine in cbar.ax.spines.values():
            spine.set_visible(False)


# %% Load functions

# %% Save functions


def save_all_spectrogram_dfs(overwrite=False, signal_type="CSD", verbose=False, save_session_dfs=True):
    save_path = LFP_RESULTS / f"aligned_spectrograms_{signal_type}.parquet"
    if save_path.exists() and not overwrite:
        print(f"event aligned spectrogram dfs already populated: {save_path.name}")
        return
    spectrogram_dfs = []
    for subject in SUBJECT_IDS:
        for maze in MAZE_CONFIGS.keys():
            for day in [int(d) for d in MAZE_DAY2DATE[maze].keys()]:
                try:
                    session = gs.get_maze_sessions(
                        subject_IDs=[subject],
                        maze_names=[maze],
                        days_on_maze=[day],
                        with_data=[
                            "trials_df",
                            "lfp_times",
                            "lfp_signal",
                            "lfp_metrics",
                            "cluster_metrics",
                            "navigation_df",
                        ],
                        must_have_data=True,
                    )
                except FileNotFoundError as e:
                    if verbose:
                        print(f"skipping session: {subject} {maze} {day} - MISSING DATA")
                        continue
                try:
                    spectrogram_df = lu._get_spectrogram_df(
                        session,
                        events=[
                            "cue",
                            "reward",
                            "end_reward_consumption",
                            "cue_goal_directed",
                            "cue_non_goal_directed",
                            "cue_not_moving",
                        ],
                        signal_type=signal_type,
                        single_channel=False,
                        window=(-3, 3),
                    )
                    if save_session_dfs:
                        save_path = LFP_RESULTS / "aligned_spectrograms" / signal_type / f"{session.name}.parquet"
                        if not overwrite and save_path.exists():
                            continue
                        else:
                            print(f"Saving {save_path.name}")
                            spectrogram_df.columns = spectrogram_df.columns.map(lambda x: str(x))
                            spectrogram_df.to_parquet(save_path, compression="gzip")
                    spectrogram_dfs.append(spectrogram_df)
                except ValueError as e:
                    if verbose:
                        print(e)
                        print(f"Failed to generate spectrogram for session: {session.name}")
    combined_dfs = pd.concat(spectrogram_dfs, axis=0).reset_index(drop=True)
    print(f"Saving {save_path.name}")
    combined_dfs.columns = combined_dfs.columns.map(lambda x: str(x))
    combined_dfs.to_parquet(save_path, compression="gzip")
    return combined_dfs


def save_all_signal_dfs(overwrite=False, signal_type="CSD", verbose=False):
    """Generates a dataframe of event-aligned LFP/CSD data for all sessions"""
    save_path = LFP_RESULTS / f"aligned_{signal_type}.parquet"
    if save_path.exists() and not overwrite:
        print(f"event aligned signal dfs already populated: {save_path.name}")
        return
    signal_dfs = []
    for subject in SUBJECT_IDS:
        for maze in MAZE_CONFIGS.keys():
            for day in [int(d) for d in MAZE_DAY2DATE[maze].keys()]:
                try:
                    session = gs.get_maze_sessions(
                        subject_IDs=[subject],
                        maze_names=[maze],
                        days_on_maze=[day],
                        with_data=["trials_df", "lfp_times", "lfp_signal", "lfp_metrics", "cluster_metrics"],
                        must_have_data=True,
                    )
                except FileNotFoundError as e:
                    if verbose:
                        print(f"skipping session: {subject} {maze} {day} - MISSING DATA")
                        continue
                if verbose:
                    print(session.name)
                try:
                    signal_dfs.append(
                        lu._get_signal_df(
                            session,
                            events=["cue", "reward", "end_reward_consumption"],
                            signal_type=signal_type,
                            single_channel=False,
                            window=(-3, 3),
                        )
                    )
                except ValueError as e:
                    if verbose:
                        print(e)
                        print(f"Failed generate event aligned signal for session: {session.name}")
    df = pd.concat(signal_dfs, axis=0)
    # save
    df.columns = df.columns.map(lambda x: str(x))
    df.to_parquet(save_path, compression="gzip")
    return df


def save_all_PSD_dfs(signal_type="CSD", overwrite=False, verbose=False):
    """ """
    save_path = LFP_RESULTS / f"trial_PSD_{signal_type}.parquet"
    if save_path.exists() and not overwrite:
        print(f"event aligned signal dfs already populated: {save_path.name}")
        return
    PSD_dfs = []
    for subject in SUBJECT_IDS:
        for maze in MAZE_CONFIGS.keys():
            for day in [int(d) for d in MAZE_DAY2DATE[maze].keys()]:
                try:
                    session = gs.get_maze_sessions(
                        subject_IDs=[subject],
                        maze_names=[maze],
                        days_on_maze=[day],
                        with_data=["trials_df", "lfp_times", "lfp_signal", "lfp_metrics", "cluster_metrics"],
                        must_have_data=True,
                    )
                except FileNotFoundError as e:
                    if verbose:
                        print(f"skipping session: {subject} {maze} {day} - MISSING DATA")
                        continue
                if verbose:
                    print(session.name)
                try:
                    PSD_dfs.append(lu._get_trial_phase_PSD_df(session, signal_type=signal_type))
                except ValueError as e:
                    if verbose:
                        print(e)
                        print(f"Failed generate event aligned signal for session: {session.name}")
    df = pd.concat(PSD_dfs, axis=0)
    # save
    df.columns = df.columns.map(lambda x: str(x))
    df.to_parquet(save_path, compression="gzip")
    return df
