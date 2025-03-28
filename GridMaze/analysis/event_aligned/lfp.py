"""
library for plotting lfp aligned to trial events
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

from GridMaze.analysis.event_aligned import delta_distance_to_goal as ddtg


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

# %% high level analysis


def test(ddtg_thres=-0.1):
    """
    Split trials into goal-directed and non-goal directed at cue from ddtg and
    plot average CSD aligned to cue to see if there is a difference
    """
    ddtg_df = ddtg.get_all_sessions_ddtg_at_cue(s=2, multi_index=True)
    csd_df = get_aligned_signal_df(signal_type="LFP")

    def _get_trial_unique_ID(row):
        return f"{row.subject_ID.values[0]}_{row.maze_name.values[0]}_{row.day_on_maze.values[0]}_{int(row.trial.values[0])}"

    ddtg_df["trial_unique_ID"] = ddtg_df.apply(_get_trial_unique_ID, axis=1)
    csd_df["trial_unique_ID"] = csd_df.apply(_get_trial_unique_ID, axis=1)
    goal_directed_trials = ddtg_df[ddtg_df.ddtg != 0].trial_unique_ID
    non_goal_directed_trials = ddtg_df[ddtg_df.ddtg == 0].trial_unique_ID
    goal_directed_csd = csd_df[csd_df.trial_unique_ID.isin(goal_directed_trials)]
    non_goal_directed_csd = csd_df[csd_df.trial_unique_ID.isin(non_goal_directed_trials)]
    for subject in SUBJECT_IDS:
        f, ax = plt.subplots()
        goal_directed_csd[goal_directed_csd.subject_ID == subject].cue_aligned_time.mean(0).plot(ax=ax)
        non_goal_directed_csd[non_goal_directed_csd.subject_ID == subject].cue_aligned_time.mean(0).plot(ax=ax)
    f, ax = plt.subplots()
    goal_directed_csd.cue_aligned_time.mean(0).plot(ax=ax)
    non_goal_directed_csd.cue_aligned_time.mean(0).plot(ax=ax)


def plot_exp_average_spectrograms(axes=None, smooth_SD=False):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), width_ratios=[0.9, 1])
    save_path = LFP_RESULTS / "aligned_spectrograms_CSD.parquet"
    df = load_data._load_multiindex_parquet(save_path)
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


def plot_bands(axes=None, low_band=(1, 3), theta_band=(8, 12), high_band=(150, 250)):
    """ """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    save_path = LFP_RESULTS / "aligned_spectrograms_CSD.parquet"
    df = load_data._load_multiindex_parquet(save_path)
    df = df[df.late_session]
    for band, label, color in zip(
        [low_band, theta_band, high_band], ["low", "theta", "high"], ["blue", "green", "red"]
    ):
        band_df = df.loc[df.frequencies.between(*band).values]
        cue_responses, reward_responses = [], []
        for subject in SUBJECT_IDS:
            subject_df = band_df[band_df.subject_ID == subject].groupby("frequencies")
            cue_responses.append(subject_df.cue_aligned_time.mean().mean().values)
            reward_responses.append(subject_df.reward_aligned_time.mean().mean().values)
        cue_mean = np.array(cue_responses).mean(axis=0)
        cue_sem = np.array(cue_responses).std(axis=0) / np.sqrt(len(SUBJECT_IDS))
        reward_mean = np.array(reward_responses).mean(axis=0)
        reward_sem = np.array(reward_responses).std(axis=0) / np.sqrt(len(SUBJECT_IDS))
        times = band_df.cue_aligned_time.columns.astype("float")
        for i, (mean, sem) in enumerate(zip([cue_mean, reward_mean], [cue_sem, reward_sem])):
            axes[i].plot(times, mean, label=f"{label}: {band[0]}-{band[1]} Hz", color=color)
            axes[i].fill_between(times, mean - sem, mean + sem, alpha=0.3, color=color)
    axes[1].legend()
    axes[0].set_title("cue aligned time (s)")
    axes[1].set_title("reward aligned time (s)")
    axes[0].set_ylabel("Power (z-scored)")
    return


# %% top level save/load data functions
# def load_spectrogram_dfs_from_disk(signal_type="CSD", subject_IDs="all"):
#     save_paths = list((LFP_RESULTS / "aligned_spectrograms" / signal_type).iterdir())
#     if subject_IDs != "all":
#         save_paths = [p for p in save_paths if p.name.split(".")[0] in subject_IDs]
#     dfs = []
#     for p in save_paths:
#         print(f"loading {p.name}")
#         dfs.append(load_data._load_multiindex_parquet(p))
#     return dfs


def save_all_spectrogram_dfs(overwrite=False, signal_type="CSD", verbose=False):
    spectrogram_dfs = []
    skipped_sessions = []
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
                        print(e)
                        continue
                try:
                    spectrogram_dfs.append(
                        get_spectrogram_df(
                            session,
                            signal_type=signal_type,
                            overwrite=overwrite,
                        )
                    )
                except ValueError as e:
                    if verbose:
                        print(e)
                        print(f"skipping session: {session.name}")
                    skipped_sessions.append(session.name)
    if verbose:
        print(f"skipped sessions: {skipped_sessions}")
    return spectrogram_dfs


def get_aligned_signal_df(overwrite=False, signal_type="CSD", single_channel=False, window=(-2, 2), verbose=False):
    """Generates a dataframe of event-aligned CSD data for all sessions"""
    save_path = LFP_RESULTS / f"aligned_{signal_type}.parquet"
    if not overwrite and save_path.exists():
        df = load_data._load_multiindex_parquet(save_path)
    else:
        print(f"Computing aligned {signal_type} signal df")
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
                            print(e)
                            continue
                    if verbose:
                        print(session)
                    try:
                        signal_dfs.append(_get_signal_df(session, signal_type, single_channel, window))
                    except ValueError as e:
                        if verbose:
                            print(e)
                            print(f"skipping session: {session.name}")
        df = pd.concat(signal_dfs, axis=0)
        # save
        df.columns = df.columns.map(lambda x: str(x))
        df.to_parquet(save_path, compression="gzip")
    return df


# %%


def get_spectrogram_df(
    session,
    signal_type="CSD",
    single_channel=False,
    window=(-2, 2),
    freqs=np.geomspace(1, 250, 100),
    overwrite=False,
):
    """
    Session level function.
    Calculates the average spectogram pwoer of LFP/CSD signal aliged to cue and reward (store in combined
    dataframe for convience).

    Note: Function tries to load data from disk if available, otherwise computes and saves to disk.
    Set overwrite to True to force recomputation and
    """
    # try to load from disk
    save_path = LFP_RESULTS / "aligned_spectrograms" / signal_type / f"{session.name}.parquet"
    if not overwrite and save_path.exists():
        df = load_data._load_multiindex_parquet(save_path)
    else:
        print(f"Computing spectrogram for {session.name}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dfs = []
        for event in ["cue", "reward"]:
            av_spec = _get_session_event_aligned_spectrogram(
                session, event, signal_type, single_channel, window, freqs, plot=False
            )
            times = np.linspace(*window, av_spec.shape[1])
            spec_df = pd.DataFrame(av_spec, columns=times)
            spec_df.columns = pd.MultiIndex.from_product([[f"{event}_aligned_time"], spec_df.columns])
            dfs.append(spec_df)
        # add info columns useful when combining multiple sessions
        info_df = _get_info_df(session, spec_df.index, signal_type, single_channel)
        info_df[("frequencies", "")] = freqs
        # combine dataframes
        df = pd.concat([info_df] + dfs, axis=1)
        # save to disk
        if not save_path.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        df.columns = df.columns.map(lambda x: str(x))
        df.to_parquet(save_path, compression="gzip")
    return df


def _get_signal_df(
    session,
    signal_type="CSD",
    single_channel=False,
    window=(-2, 2),
    zscore_signal=True,
):
    """ """
    # load data
    trials_df = session.trials_df
    times = session.lfp_times
    if signal_type == "LFP":
        signal = get_LFP(session, shank=3, single_channel=single_channel)
    elif signal_type == "CSD":
        signal = get_CSD(session, orientation="horizontal", single_channel=single_channel)
    else:
        raise NotImplementedError
    if zscore_signal:
        signal = zscore(signal, axis=0)
    # average normalised signal around event times
    sig_dfs = []
    samples_before, samples_after = int(window[0] * FS), int(window[1] * FS)
    for event in ["cue", "reward"]:
        event_times = trials_df.time[event].values
        trials = trials_df.trial
        nearest_event_samples = np.array([np.argmin(np.abs(times - t)) for t in event_times])
        signal_windows = [signal[s + samples_before : s + samples_after] for s in nearest_event_samples]
        expected_samples = np.abs(samples_before) + np.abs(samples_after)
        keep_sig, keep_trials = [], []
        for i, sig in enumerate(signal_windows):
            if sig.shape[0] != expected_samples:
                print(f"trial: {trials[i]} window out of bounds")
            else:
                keep_sig.append(sig)
                keep_trials.append(trials[i])
        t = np.linspace(*window, expected_samples)
        sig_dfs.append(
            pd.DataFrame(data=np.array(keep_sig), columns=pd.MultiIndex.from_product([[f"{event}_aligned_time"], t]))
        )
    # return as dataframe with some session info
    info_df = _get_info_df(session, sig_dfs[0].index, signal_type, single_channel)
    info_df[("trial", "")] = keep_trials
    df = pd.concat([info_df] + sig_dfs, axis=1)
    return df


def _get_info_df(session, index, signal_type, single_channel):
    late_session = (
        True if session.day_on_maze in [int(d) for d in MAZE_DAY2DATE[session.maze_name].keys()][-7:] else False
    )
    info_df = pd.DataFrame(index=index)
    info_df[("subject_ID", "")] = session.subject_ID
    info_df[("maze_name", "")] = session.maze_name
    info_df[("day_on_maze", "")] = session.day_on_maze
    info_df[("late_session", "")] = late_session
    info_df[("signal_type", "")] = signal_type
    info_df[("single_channel", "")] = single_channel
    info_df[("signal_type", "")] = signal_type
    info_df[("probe_depth", "")] = session.probe_depth
    info_df[("tissue_sample", "")] = session.tissue_sample
    info_df.columns = pd.MultiIndex.from_tuples(info_df.columns)
    return info_df


def _get_session_event_aligned_spectrogram(
    session,
    event="cue",
    signal_type="CSD",
    single_channel=False,
    window=(-2, 2),
    freqs=np.geomspace(1, 250, 100),
    zscore_freqs=True,
    plot=False,
):
    """ """
    # load data
    trials_df = session.trials_df
    times = session.lfp_times
    if signal_type == "LFP":
        signal = get_LFP(session, shank=3, single_channel=single_channel)
    elif signal_type == "CSD":
        signal = get_CSD(session, orientation="horizontal", single_channel=single_channel)
    else:
        raise NotImplementedError
    # wavelet transform entire session
    cwt = compute_wavelet_transform_fft(signal, freqs, FS)
    spec = np.abs(cwt) ** 2  # freq x time x power (spectrogram)
    # zscore spectrogram (across frequencies)
    if zscore_freqs:
        spec = zscore(spec, axis=1)
    # average normalised spectrogram around event times
    event_times = trials_df.time[event].values
    nearest_event_samples = np.array([np.argmin(np.abs(times - t)) for t in event_times])
    samples_before, samples_after = int(window[0] * FS), int(window[1] * FS)
    spec_windows = [spec[:, s + samples_before : s + samples_after] for s in nearest_event_samples]
    expected_samples = np.abs(samples_before) + np.abs(samples_after)
    spec_windows = [s for s in spec_windows if s.shape[1] == expected_samples]
    av_spec = np.array(spec_windows).mean(axis=0)
    if plot:
        t = np.linspace(*window, av_spec.shape[1])
        _plot_spectrogram(av_spec, t, freqs)
    return av_spec


# %% Plotting functions


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


# %% get CSD/LFP functions


def get_LFP(session, shank=3, single_channel=False, remove_artifacts=True):
    """ """
    # load data
    lfp_metrics = session.lfp_metrics
    cluster_metrics = session.cluster_metrics
    lfp_signal = session.lfp_signal
    if single_channel:  # converting from channel_id to channel_index in lfp_signal to get the signal
        channel_id = _get_single_channel_for_LFP(lfp_metrics, cluster_metrics, shank)
        channel_index = lfp_metrics[lfp_metrics.contact.id == channel_id].index[0]
        lfp = lfp_signal[:, channel_index]
    else:  # average lfp over multiple channels
        channel_ids = _get_shank_channels_for_LFP(lfp_metrics, cluster_metrics, shank)
        channel_indices = lfp_metrics.contact.id.isin(channel_ids).index.values
        lfp = lfp_signal[:, channel_indices].mean(axis=1)
    if remove_artifacts:
        lfp = _remove_artifacts(lfp, thres=500)
    return lfp


def get_CSD(session, orientation="horizontal", single_channel=False, remove_artifacts=True):
    """
    CSD = c2 - ((c1 + c3) / 2)
    for c1, c2, c3 colinear contacts in the same region
    """
    # load data
    lfp_metrics = session.lfp_metrics
    cluster_metrics = session.cluster_metrics
    lfp_signal = session.lfp_signal
    if single_channel:  # converting from channel_id to channel_index in lfp_signal to get the signal
        c1_id, c2_id, c3_id = _get_single_channels_for_CSD(lfp_metrics, cluster_metrics, orientation)
        c1_ind, c2_ind, c3_ind = [lfp_metrics[lfp_metrics.contact.id == c].index[0] for c in [c1_id, c2_id, c3_id]]
        c1, c2, c3 = [lfp_signal[:, c] for c in [c1_ind, c2_ind, c3_ind]]
    else:  # average lfp over multiple channels before calculating CSD
        c1_ids, c2_ids, c3_ids = _get_shank_channels_for_CSD(lfp_metrics, cluster_metrics, orientation)
        c1_inds, c2_inds, c3_inds = [
            lfp_metrics[lfp_metrics.contact.id.isin(cs)].index.values for cs in [c1_ids, c2_ids, c3_ids]
        ]
        c1, c2, c3 = [lfp_signal[:, cs].mean(axis=1) for cs in [c1_inds, c2_inds, c3_inds]]
    # calculate CSD (2nd spatial derivate of LFP)
    CSD = c2 - (c1 + c3) / 2
    if remove_artifacts:
        CSD = _remove_artifacts(CSD, thres=100)
    return CSD


def _remove_artifacts(signal, thres=500, window=(-100, 100)):
    """ """
    sig_med = np.median(signal)
    sig_mean_dev = np.abs(signal - sig_med)
    artifact_mask = sig_mean_dev > thres
    # Create a mask for indices to remove.
    removal_mask = np.zeros(signal.shape, dtype=bool)
    artifact_indices = np.where(artifact_mask)[0]
    for idx in artifact_indices:
        start = max(0, idx + window[0])
        end = min(len(signal), idx + window[1] + 1)
        removal_mask[start:end] = True
    good_indices = np.where(~removal_mask)[0]
    # Check that we have at least two good points for interpolation.
    if len(good_indices) < 2:
        raise ValueError("Not enough good points for interpolation.")
    new_signal = signal.copy()
    x = np.arange(len(signal))
    # Interpolate over the removed indices.
    new_signal[removal_mask] = np.interp(x[removal_mask], x[good_indices], signal[good_indices])
    return new_signal


# %% Select channels for LFP / CSD analysis


def _get_shank_channels_for_LFP(lfp_metrics, cluster_metrics, shank=3, verbose=False, min_good=2):
    """ """
    cluster_metrics = cluster_metrics[cluster_metrics.single_unit]
    lfp_shank = lfp_metrics[(lfp_metrics.contact.shank == shank) & (lfp_metrics.contact.qc == "good")]
    if len(lfp_shank) < min_good:
        raise ValueError(f"Shank {shank} has less than {min_good} good contacts")
    contact_set = lfp_shank.contact.id.values
    if verbose:
        print(f"Shank {shank} contacts: {contact_set}")
    return contact_set


def _get_single_channel_for_LFP(lfp_metrics, cluster_metrics, shank=3, verbose=False):
    """
    Get a single LFP channel from the middle of the probe that has passed QC and is associated with single unit(s).
    """
    cluster_metrics = cluster_metrics[cluster_metrics.single_unit]
    single_unit_contact_ids = set(cluster_metrics.contact.id.unique())
    lfp_shank = lfp_metrics[
        (lfp_metrics.contact.shank == shank)
        & (lfp_metrics.contact.qc == "good")
        & (lfp_metrics.contact.id.isin(single_unit_contact_ids))
    ].sort_values(("contact", "y"))
    if lfp_shank.empty:
        raise ValueError(f"No suitable channels found for shank: {shank}")
    mid_index = len(lfp_shank) // 2
    selected_contact = lfp_shank.iloc[mid_index].contact.id
    if verbose:
        print(f"Selected channel: {selected_contact}")
    return selected_contact


def _get_shank_channels_for_CSD(lfp_metrics, cluster_metrics, orientation="horizontal", verbose=False, min_good=2):
    """ """
    cluster_metrics = cluster_metrics[cluster_metrics.single_unit]
    if orientation == "horizontal":
        # choose every other shank, check which option has best signal (proxy single unit counts)
        set_1 = [0, 2, 4]
        set_2 = [1, 3, 5]
        set_1_total_units = len(cluster_metrics[cluster_metrics.contact.shank.isin(set_1)])
        set_2_total_units = len(cluster_metrics[cluster_metrics.contact.shank.isin(set_2)])
        if set_1_total_units > set_2_total_units:
            horizontal_shanks = set_1
        else:
            horizontal_shanks = set_2
        # get contacts
        contact_sets = []
        for shank in horizontal_shanks:
            shank_contacts = lfp_metrics[(lfp_metrics.contact.shank == shank) & (lfp_metrics.contact.qc == "good")]
            if len(shank_contacts) < min_good:
                raise ValueError(f"Shank {shank} has less than {min_good} good contacts")
            contact_set = shank_contacts.contact.id.values
            if verbose:
                print(f"Shank {shank} contacts: {contact_set}")
            contact_sets.append(contact_set)
    else:  # havn't implemented vertical version
        raise NotImplementedError
    return contact_sets


def _get_single_channels_for_CSD(
    lfp_metrics,
    cluster_metrics,
    orientation="horizontal",
    verbose=False,
):
    """
    NOTE only works for 6 shank cambridge neurotech probes
    Get channel ids for computing the current source density (CSD)
    Inputs:
        lfp_metrics - DataFrame containing LFP channel information (MultiIndex, e.g., lfp_metrics.contact.id)
        cluster_metrics - DataFrame containing cluster information (MultiIndex)
        orientation - Orientation of the CSD ("horizontal" or "vertical")
    Note:
        - For horizontal CSD, channels are taken across shanks 1, 3, and 5 at the same depth.
        - For vertical CSD, channels are taken from a single shank (top, middle, bottom).
        - The function searches through available options in both orientations, giving
          preference to channels with nearby single units and that passed QC ("good")
          during spikesorting.
    Returns:
        channels - List of channel ids for computing the CSD
    """
    # Restrict to contacts with single units (marker of good quality)
    cluster_metrics = cluster_metrics[cluster_metrics.single_unit]
    single_unit_contact_ids = set(cluster_metrics.contact.id.unique())

    if orientation == "horizontal":
        horizontal_shanks = [1, 3, 5]
        lfp_horiz = lfp_metrics[lfp_metrics.contact.shank.isin(horizontal_shanks)]
        # Get all unique depths and sort from bottom to top (assuming higher y is deeper)
        depths = sorted(lfp_horiz.contact.y.unique(), reverse=True)

        valid_candidates = []
        for depth in depths:
            subset = lfp_horiz[lfp_horiz.contact.y == depth]
            # Ensure channels from all three required shanks exist
            if set(subset.contact.shank) != set(horizontal_shanks):
                continue
            # Ensure exactly three channels (one per shank)
            if len(subset) != 3:
                continue
            # All channels must pass QC
            if not (subset.contact.qc == "good").all():
                continue
            # All channels must be associated with a single unit
            if not set(subset.contact.id).issubset(single_unit_contact_ids):
                continue
            valid_candidates.append((depth, subset))
        if not valid_candidates:
            raise ValueError("No suitable channels found for horizontal CSD computation")

        if verbose:
            print(f"Found {len(valid_candidates)} valid channel sets for horizontal CSD")
        # Choose the candidate from the bottom-most valid depth (first in our sorted order)
        chosen_depth, candidate_subset = valid_candidates[0]
        if verbose:
            print(f"Choosing channels at depth {chosen_depth}")
        # Order channels by the required shank order (1, 3, 5)
        candidate_subset = candidate_subset.set_index(("contact", "shank")).loc[horizontal_shanks]
        contacts = tuple(candidate_subset.contact.id)
        if verbose:
            print(f"Channels chosen: {contacts}")
        return contacts

    elif orientation == "vertical":
        valid_candidates = []
        # Iterate over available shanks
        for shank in sorted(lfp_metrics.contact.shank.unique()):
            # consider every other channel (roughly top, middle, bottom)
            subset = lfp_metrics[lfp_metrics.contact.shank == shank].sort_values(("contact", "y")).iloc[::2]
            # Ensure exactly three channels (one per shank)
            if len(subset) != 3:
                continue
            # All channels must pass QC
            if not (subset.contact.qc == "good").all():
                continue
            # All channels must be associated with a single unit
            if not set(subset.contact.id).issubset(single_unit_contact_ids):
                continue
            valid_candidates.append((shank, subset))
        if not valid_candidates:
            raise ValueError("No suitable channels found for vertical CSD computation")

        if verbose:
            print(f"Found {len(valid_candidates)} valid channel sets for vertical CSD")
        # Choose the candidate with the earliest shank (lowest shank number)
        valid_candidates.sort(key=lambda x: x[0])
        chosen_shank, candidate_subset = valid_candidates[0]
        if verbose:
            print(f"Choosing channels from shank {chosen_shank}")
        # Order channels top-to-bottom (ascending y)
        candidate_subset = candidate_subset.sort_values(("contact", "y"))
        contacts = tuple(candidate_subset.contact.id)
        if verbose:
            print(f"Channels chosen: {contacts}")
        return contacts


# %% Custom Wavelet functions


def _morlet(M, gaussian_width=1.5, window_length=1.0, precision=8):
    """
    Generate a complex Morlet wavelet kernel.
    """
    x = np.linspace(-precision, precision, M)
    return (
        ((np.pi * gaussian_width) ** (-0.25))
        * np.exp(-(x**2) / gaussian_width)
        * np.exp(1j * 2 * np.pi * window_length * x)
    )


def generate_morlet_filterbank(freqs, fs, gaussian_width=1.5, window_length=1.0, precision=16):
    """
    Generate a bank of Morlet wavelet filters for a set of frequencies.

    Parameters:
      freqs         - 1D numpy array of positive frequency values
      fs            - Sampling rate (Hz)
      gaussian_width- Width of the Gaussian envelope
      window_length - Base frequency of the mother wavelet
      precision     - Controls the number of points in the wavelet (2**precision)

    Returns:
      filter_bank   - Array of shape (n_freqs, max_len) containing padded filters
      time          - Time vector corresponding to the wavelets
    """
    filter_bank = []
    cutoff = 8  # Determines the time support of the wavelet
    M = 2**precision
    # FIX: Remove precision from the call to _morlet so that it uses its default precision value (8)
    morlet_wavelet = np.conj(_morlet(M, gaussian_width, window_length))
    x = np.linspace(-cutoff, cutoff, M)
    max_len = 0
    time = None

    for freq in freqs:
        scale = window_length / (freq / fs)
        # Determine the subsampling indices to get the desired frequency scaling.
        j = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * (x[1] - x[0]))
        j = np.ceil(j).astype(int)
        j = j[j < morlet_wavelet.size]
        # Reverse the scaled wavelet
        scaled_wavelet = morlet_wavelet[j][::-1]
        if len(scaled_wavelet) > max_len:
            max_len = len(scaled_wavelet)
            time = np.linspace(-cutoff * window_length / freq, cutoff * window_length / freq, max_len)
        filter_bank.append(scaled_wavelet)

    # Pad all wavelet filters to have the same length.
    padded_filters = []
    for filt in filter_bank:
        pad_width = max_len - len(filt)
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded_filters.append(np.pad(filt, (pad_left, pad_right), mode="constant"))

    return np.array(padded_filters), time


def compute_wavelet_transform_fft(sig, freqs, fs, gaussian_width=1.5, window_length=1.0, precision=16, norm=False):
    filter_bank, _ = generate_morlet_filterbank(freqs, fs, gaussian_width, window_length, precision)
    n_freqs, _ = filter_bank.shape
    n_time = len(sig)
    cwt = np.zeros((n_freqs, n_time), dtype=complex)

    for i in range(n_freqs):
        conv_real = fftconvolve(sig, np.real(filter_bank[i]), mode="same")
        conv_imag = fftconvolve(sig, np.imag(filter_bank[i]), mode="same")
        cwt[i] = conv_real + 1j * conv_imag

    # Normalize the coefficients if desired.
    if norm == "l1":
        cwt = cwt / (fs / freqs[:, np.newaxis])
    elif norm == "l2":
        cwt = cwt / (fs / np.sqrt(freqs)[:, np.newaxis])

    return cwt


def plot_filterbank(filter_bank, time, freqs):
    fig, ax = plt.subplots(1, constrained_layout=True, figsize=(5, 15))
    ax.spines[["top", "right", "left"]].set_visible(False)
    for f_i in range(filter_bank.shape[0]):
        ax.plot(time, np.real(filter_bank[f_i, :]) + f_i * 1.5)
        ax.text(-1.5, 1.5 * f_i, f"{np.round(freqs[f_i], 2)}Hz", va="center", ha="left")

    ax.set_yticks([])
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Time (s)")
    ax.set_title("Wavlet Filterbank")


# %% QC functions


def plot_CSD_QC(session, times=(5, 7), filter_CSD=True):
    lfp_signal = session.lfp_signal
    lfp_metrics = session.lfp_metrics
    lfp_times = session.lfp_times
    mask = (lfp_times > times[0]) & (lfp_times < times[1])
    for ort in ["horizontal", "vertical"]:
        c1, c2, c3 = _get_single_channels_for_CSD(
            lfp_metrics,
            session.cluster_metrics,
            ort,
            verbose=True,
        )
        # check contact info
        contact_infos = []
        for c in [c1, c2, c3]:
            contact_info = lfp_metrics[lfp_metrics.contact.id == c]
            contact_infos.append(
                {
                    "id": contact_info.contact.id.values[0],
                    "indx": contact_info.index.values[0],
                    "x": contact_info.contact.x.values[0],
                    "y": contact_info.contact.y.values[0],
                }
            )
        c1_info, c2_info, c3_info = contact_infos
        # isolate lfp from contacts of interest
        contact_indices = [lfp_metrics[lfp_metrics.contact.id == c].index[0] for c in [c1, c2, c3]]
        c1, c2, c3 = [lfp_signal[:, c] for c in contact_indices]
        # get CSD
        CSD = c2 - (c1 + c3) / 2
        # plot (just specficied times)
        times, c1, c2, c3, CSD = [x[mask] for x in [lfp_times, c1, c2, c3, CSD]]
        if filter_CSD:
            CSD = mne.filter.filter_data(
                CSD[:, np.newaxis].T.astype(np.float64),
                FS,
                l_freq=None,
                h_freq=300,
                method="fir",
                fir_design="firwin",
                verbose=False,
            ).T
        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, sharey=True)
        for ax in axes:
            ax.spines[["top", "right", "bottom"]].set_visible(False)
            ax.spines["left"].set_linewidth(0.5)
        for ax in axes[:-1]:
            ax.set_xticks([])
            ax.set_ylabel("Voltage (uV)")
        axes[0].plot(times, c1, color="black")
        axes[0].set_title(f"Contact {c1_info['id']} at ({c1_info['x']}, {c1_info['y']})")
        axes[1].plot(times, c2, color="black")
        axes[1].set_title(f"Contact {c2_info['id']} at ({c2_info['x']}, {c2_info['y']})")
        axes[2].plot(times, c3, color="black")
        axes[2].set_title(f"Contact {c3_info['id']} at ({c3_info['x']}, {c3_info['y']})")
        axes[3].plot(times, CSD, color="blue")
        axes[3].set_title("CSD")
        axes[3].set_xlabel("Time (s)")
    return
