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

# %% Spectrogram plots


def plot_cue_aligned_spectrogram_residuals(
    spectrogram_df,
    events=["cue_goal_directed", "cue_non_goal_directed"],
    axes=None,
    window=(-1, 1),
    bands=[(4, 5), (7, 9), (9, 12)],  # Hz
):
    """ """
    # prepare axes
    if axes is None:
        f, axes = plt.subplots(1, 2, figsize=(6, 3), width_ratios=[1, 0.75])
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("Time (s)")

    # process data for spectrogram plot
    df = spectrogram_df[spectrogram_df.late_session]
    event_dfs = [df[df.event == e] for e in events]
    av_spec_dfs = [_df.groupby("frequency").time.mean().time for _df in event_dfs]
    residual_df = av_spec_dfs[0] - av_spec_dfs[1]
    t = residual_df.columns.values.astype(np.float64)
    freqs = residual_df.index.values
    spec = residual_df.values
    t_mask = (t >= window[0]) & (t <= window[1])
    _min = spec[:, t_mask].min()
    _max = spec[:, t_mask].max()
    # plot
    _plot_spectrogram(spec, t, freqs, ax=axes[0], _min=_min, _max=_max)
    axes[0].set_xlim(*window)

    # process data for band power plot
    band_results = np.zeros((len(SUBJECT_IDS), len(bands), df.time.shape[1]))
    for i, subject in enumerate(SUBJECT_IDS):
        subject_df = df[df.subject_ID == subject]
        event_dfs = [subject_df[subject_df.event == e] for e in events]
        av_spec_dfs = [_df.groupby("frequency").time.mean().time for _df in event_dfs]
        residual_df = av_spec_dfs[0] - av_spec_dfs[1]
        for j, band in enumerate(bands):
            band_df = residual_df.loc[residual_df.reset_index().frequency.between(*band).values]
            band_mean = band_df.mean(axis=0)
            band_results[i, j, :] = band_mean
    band_results_mean = band_results.mean(axis=0)
    band_results_sem = band_results.std(axis=0) / np.sqrt(len(SUBJECT_IDS))
    # plot
    for i, band in enumerate(bands):
        band_mean = band_results_mean[i, :]
        band_sem = band_results_sem[i, :]
        axes[1].plot(t, band_mean, label=f"{band[0]}-{band[1]} Hz")
        axes[1].fill_between(t, band_mean - band_sem, band_mean + band_sem, alpha=0.2)
    axes[1].legend()
    axes[1].set_xlim(*window)
    axes[1].axhline(0, color="black", linestyle="--")
    f.tight_layout()


def plot_average_spectrogram(
    spectrogram_df, axes=None, windows={"cue": (-1, 1), "reward": (-3, 3), "end_reward_consumption": (-1, 1)}
):
    """ """
    # prepare axes
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(6, 3), width_ratios=[0.4, 1, 0.5], sharey=True)
    # process data for plot
    events = ["cue", "reward", "end_reward_consumption"]
    df = spectrogram_df[spectrogram_df.late_session]
    event_dfs = [df[df.event == e] for e in events]
    av_spec_dfs = [_df.groupby("frequency").time.mean().time for _df in event_dfs]
    _max = max([av_spec_df.max().max() for av_spec_df in av_spec_dfs])
    _min = min([av_spec_df.min().min() for av_spec_df in av_spec_dfs])
    for i, (ax, av_spec_df) in enumerate(zip(axes, av_spec_dfs)):
        t = av_spec_df.columns.values.astype(np.float64)
        spec = av_spec_df.values
        freqs = av_spec_df.index.values
        cbar = True if i == 2 else False
        y_label = True if i == 0 else False
        _plot_spectrogram(
            spec,
            t,
            freqs,
            ax=ax,
            _min=_min,
            _max=_max,
            colorbar=cbar,
            y_label=y_label,
        )
        event = events[i]
        ax.set_title(event)
        ax.set_xlim(*windows[event])
    fig.tight_layout()
    return


def _plot_spectrogram(x, times, freqs, ax=None, _min=None, _max=None, colorbar=True, y_label=True):
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
    if y_label:
        ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    if colorbar:
        cbar = plt.colorbar(pcmesh, ax=ax, orientation="vertical")
        cbar.set_label("Power (z-scored)")
        for spine in cbar.ax.spines.values():
            spine.set_visible(False)


# %% PSD plots


def plot_PSD(
    PSD_df,
    axes=None,
    normalise=False,
):
    """
    Looks much better with LFP, maybe too low s/n in CSD
    """
    # prepare axes
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
        ax.set_xscale("log")
    # process data for plot
    df = PSD_df[PSD_df.late_session]
    trial_events = ["cue", "reward", "ERC"]
    trial_phases = ["navigation", "RC", "ITI"]
    av_subject_psd = df.groupby(["subject_ID", "frequency"]).power.mean()
    if normalise:
        subject_max_power = av_subject_psd.groupby(level="subject_ID")["power"].max().max(1)
        av_subject_psd = av_subject_psd.div(
            av_subject_psd.index.get_level_values("subject_ID").map(subject_max_power), axis=0
        )
    subject_grouped_psd = av_subject_psd.reset_index("subject_ID").groupby("frequency").power
    psd_mean = subject_grouped_psd.mean().power
    psd_sem = subject_grouped_psd.sem().power
    freqs = psd_mean.index.values
    for ax, trial_sections in zip(axes, [trial_events, trial_phases]):
        for s in trial_sections:
            mean = psd_mean[s].values
            sem = psd_sem[s].values
            ax.plot(freqs, mean, label=s)
            ax.fill_between(freqs, mean - sem, mean + sem, alpha=0.2)
        ax.legend()


# %% Event aligned signal plots


def plot_av_event_aligned_signal(
    signal_df, axes=None, windows={"cue": (-0.5, 0.5), "reward": (-1.5, 1.5), "end_reward_consumption": (-0.5, 0.5)}
):
    """mean, sem across subjects plotted"""
    # prepare axes
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(6, 3), width_ratios=[1, 3, 1], sharey=True)
    for ax in axes.flatten():
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("uV")
        ax.axhline(0, color="black", linestyle="--", alpha=0.2)
        ax.axvline(0, color="black", linestyle="--", alpha=0.2)
    # process data for plot
    df = signal_df[signal_df.late_session]
    event_dfs = [df[df.event == e] for e in ["cue", "reward", "end_reward_consumption"]]
    av_subject_dfs = [df.groupby("subject_ID").time.mean().time for df in event_dfs]
    t = av_subject_dfs[0].columns.values.astype(np.float64)
    for event, ax, av_subject_df in zip(["cue", "reward", "end_reward_consumption"], axes, av_subject_dfs):
        mean = av_subject_df.mean(axis=0).values
        sem = av_subject_df.sem(axis=0).values
        ax.plot(t, mean, label=event, color="k")
        ax.fill_between(t, mean - sem, mean + sem, alpha=0.3, color="k")
        ax.set_title(event)
        ax.set_xlim(*windows[event])


def plot_av_subject_signal(
    signal_df,
    event="cue",
    window=(-1, 1),
    axes=None,
):
    """subjects plotted in separate pannels"""
    # prepare axes
    if axes is None:
        fig, axes = plt.subplots(2, 3, figsize=(6, 4), sharey=True)
    for ax in axes.flatten():
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.axhline(0, color="black", linestyle="--", alpha=0.2)
        ax.axvline(0, color="black", linestyle="--", alpha=0.2)
    # process data for plot
    df = signal_df[signal_df.late_session]
    df = df[df.event == event]
    times = df.time.columns.values.astype(np.float64)
    for subject, ax in zip(SUBJECT_IDS, axes.flatten()):
        subject_df = df[df.subject_ID == subject]
        mean = subject_df.time.mean(0).values
        ax.plot(times, mean, c="k")
        ax.set_title(subject)
    for ax in axes.flatten():
        ax.set_xlim(*window)
    fig.tight_layout()


# %% Load functions


def load_spectrogram_df(signal_type):
    """ """
    save_path = LFP_RESULTS / f"aligned_spectrograms_{signal_type}.parquet"
    if not save_path.exists():
        print(f"event aligned spectrogram dfs not found, save with fn: save_all_spectrogram_dfs")
    df = load_data._load_multiindex_parquet(save_path)
    return df


def load_PSD_df(signal_type):
    """ """
    save_path = LFP_RESULTS / f"trial_PSD_{signal_type}.parquet"
    if not save_path.exists():
        print(f"event aligned PSD dfs not found, save with fn: save_all_PSD_dfs")
    df = load_data._load_multiindex_parquet(save_path)
    return df


def load_signal_df(signal_type):
    """ """
    save_path = LFP_RESULTS / f"aligned_{signal_type}.parquet"
    if not save_path.exists():
        print(f"event aligned signal dfs not found, save with fn: save_all_signal_dfs")
    df = load_data._load_multiindex_parquet(save_path)
    return df


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
    save_path = LFP_RESULTS / f"aligned_spectrograms_{signal_type}.parquet"
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
