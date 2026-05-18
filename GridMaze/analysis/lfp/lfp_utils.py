"""
Ultility functions to support main lfp.py library
"""

# %% Imports
import pandas as pd
import numpy as np
import json
import mne
from scipy.stats import zscore
import matplotlib.pyplot as plt
from scipy.signal import butter, fftconvolve, filtfilt, hilbert, welch
from mne.time_frequency import psd_array_multitaper

from GridMaze.analysis.event_aligned import delta_distance_to_goal as ddtg

# %% Global Variables

from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "maze_day2date.json", "r") as input_file:
    MAZE_DAY2DATE = json.load(input_file)

FS = 1500  # lfp sampling frequency

THETA_RANGE = (7, 10)

DDTG_LOWER_THRES = -0.1  # rate of change of distance to goal thres for goal-directed behaviour at cue

DDTG_UPPER_THRES = 0.015  #  ... for non-goal-directed at cue

DDTG_WINDOW = (1, 3)  # seconds post cue to calculate ddtg


# %%
def _get_trial_phase_PSD_df(
    session,
    signal_type="LFP",
    single_channel=False,
    event_windows={"cue": (0, 0.5), "reward": (-0.5, 0.5), "ERC": (-0.5, 0.5)},
    trial_phase_windows={"navigation": (0.5, -0.5), "RC": (0.5, -0.5), "ITI": (0.5, -0.5)},
    min_window=0.5,
):
    """
    Compute the average power spectral density (PSD) of LFP/CSD signals
    in different trial phases: navigation, reward consumption (RC), ITI,
    and different trial events: cue, reward, end reward consumption (ERC),
    using the Welsh method.

    Note:
    - event_windows: specifies the window (s) around event to calculate PSD
    - trial_phase_windows: specifies the times pre-post surrounding events to calc PSD:
        - "navigation": cue+t[0], reward-t[1]
        - "RC": reward+t[0], ERC-t[1]
        - "ITI": ERC+t[0], cue-t[1]
    """
    # load data
    trials_df = session.trials_df
    times = session.lfp_times
    if signal_type == "LFP":
        signal = get_LFP(session, shank=3, single_channel=single_channel)
    elif signal_type == "CSD":
        signal = get_CSD(session, orientation="horizontal", single_channel=single_channel)

    # get trial_phase/event windows
    cue_times = trials_df.time.cue.values
    reward_times = trials_df.time.reward.values
    ERC_times = trials_df.time.end_reward_consumption.values
    trial_end_times = trials_df.time.trial_end.values
    label2windows = {
        "cue": list(zip(cue_times + event_windows["cue"][0], cue_times + event_windows["cue"][1])),
        "reward": list(zip(reward_times + event_windows["reward"][0], reward_times + event_windows["reward"][1])),
        "ERC": list(zip(ERC_times + event_windows["ERC"][0], ERC_times + event_windows["ERC"][1])),
    }
    nav_windows = list(
        zip(cue_times + trial_phase_windows["navigation"][0], reward_times + trial_phase_windows["navigation"][1])
    )
    RC_windows = list(zip(reward_times + trial_phase_windows["RC"][0], ERC_times + trial_phase_windows["RC"][1]))
    ITI_windows = list(zip(ERC_times + trial_phase_windows["ITI"][0], trial_end_times + trial_phase_windows["ITI"][1]))
    # ensure windows are all > min_window (if not remove that trial's window)
    label2windows["navigation"] = [x for x in nav_windows if np.diff(x) - min_window > 0]
    label2windows["RC"] = [x for x in RC_windows if np.diff(x) - min_window > 0]
    label2windows["ITI"] = [x for x in ITI_windows if np.diff(x) - min_window > 0]
    # get segments for each event/trial_phase
    segments = {}
    for label, windows in label2windows.items():
        segments[label] = []
        for window in windows:
            start_sample = np.argmin(np.abs(times - window[0]))
            end_sample = np.argmin(np.abs(times - window[1]))
            segments[label].append(signal[start_sample:end_sample])
    # calculate average PSD for each segment and return in df
    label2psd = []
    for label, segs in segments.items():
        f, psd = _av_psd_multitaper(segs)
        label2psd.append((label, psd))
    results_df = pd.DataFrame()
    for label, psd in label2psd:
        results_df[label] = psd
    results_df.columns = pd.MultiIndex.from_product([["power"], results_df.columns])
    # info df
    info_df = _get_info_df(session, results_df.index, signal_type, single_channel)
    info_df[("frequency", "")] = f
    return pd.concat([info_df, results_df], axis=1)


def _av_psd_multitaper(
    segments,
    fs=FS,
    fmin=1,
    fmax=250,
    n_freqs=100,
    normalization="full",
):
    """
    Compute the average PSD over many segments using MNE's multitaper method.
    Note: Because the segments have variable lengths (and thus may yield different frequency bins),
    each segment's PSD is interpolated onto a common frequency grid before averaging.
    Returns:
    - common_freqs: 1D numpy array with the common frequency grid.
    - avg_psd: 1D numpy array with the averaged PSD across segments.
    """
    common_freqs = np.geomspace(fmin, fmax, n_freqs)
    psd_list = []
    for seg in segments:
        psd, freqs = psd_array_multitaper(
            seg, sfreq=fs, fmin=fmin, fmax=fmax, normalization=normalization, verbose=False
        )
        # Interpolate this segment's PSD onto the common frequency grid.
        interp_psd = np.interp(common_freqs, freqs, psd)
        psd_list.append(interp_psd)
    psd_array = np.array(psd_list)
    avg_psd = np.mean(psd_array, axis=0)
    return common_freqs, avg_psd


def _av_psd_welch(segments):
    """
    Compute the average PSD over many segments using Welch's method.
    Returns:
    - f: Frequency bins.
    - avg_psd: Averaged power spectral density across segments.
    """
    psds = []
    for seg in segments:
        f, psd = welch(seg, fs=FS, nperseg=FS // 2, noverlap=None, window="hann", scaling="spectrum")
        psds.append(psd)
    psds = np.array(psds)
    avg_psd = np.mean(psds, axis=0)
    return f, avg_psd


# %% Session level spectrogram and signals


def _get_signal_df(
    session,
    events=["cue", "reward", "end_reward_consumption"],
    signal_type="CSD",
    single_channel=False,
    window=(-3, 3),
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
    for event in events:
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
        df = pd.DataFrame(data=np.array(keep_sig), columns=pd.MultiIndex.from_product([["time"], t]))
        info_df = _get_info_df(session, df.index, signal_type, single_channel)
        info_df[("event", "")] = event
        info_df[("trial", "")] = keep_trials
        sig_dfs.append(pd.concat([info_df, df], axis=1))
    return pd.concat(sig_dfs, axis=0).reset_index(drop=True)


def _get_spectrogram_df(
    session,
    events=["cue", "reward", "end_reward_consumption", "cue_goal_directed", "cue_non_goal_directed", "cue_not_moving"],
    signal_type="CSD",
    single_channel=False,
    window=(-3, 3),
    freqs=np.geomspace(1, 250, 100),
    zscore_freqs=True,
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
    if any(["goal_directed" or "not_moving" in e for e in events]):
        ddtg_df = ddtg.get_session_delta_dtg(session, window=DDTG_WINDOW)
        mean_ddtg = ddtg_df.cue_aligned_time.mean(1)
    # wavelet transform entire session
    cwt = compute_wavelet_transform_fft(signal, freqs, FS)
    spec = np.abs(cwt) ** 2  # freq x time x power (spectrogram)
    # zscore spectrogram (across frequencies)
    if zscore_freqs:
        spec = zscore(spec, axis=1)
    # average normalised spectrogram around event times
    av_specs = []
    for event in events:
        # special instance split trials based on goal-directed criteria
        if event == "cue_goal_directed":
            goal_directed_trials = ddtg_df[mean_ddtg.le(DDTG_LOWER_THRES)].trial.values
            event_times = trials_df[trials_df.trial.isin(goal_directed_trials)].time.cue.values
        elif event == "cue_non_goal_directed":
            non_goal_directed_trials = ddtg_df[mean_ddtg.ge(DDTG_UPPER_THRES)].trial.values
            event_times = trials_df[trials_df.trial.isin(non_goal_directed_trials)].time.cue.values
        elif event == "cue_not_moving":  # control: stationary trials
            stationary_trials = ddtg_df[mean_ddtg == 0].trial.values
            event_times = trials_df[trials_df.trial.isin(stationary_trials)].time.cue.values
        else:  # normal instance: compute over all trials
            event_times = trials_df.time[event].values
        if event_times.size == 0:
            print(f"no trials for event: {event}")
            continue
        nearest_event_samples = np.array([np.argmin(np.abs(times - t)) for t in event_times])
        samples_before, samples_after = int(window[0] * FS), int(window[1] * FS)
        spec_windows = [spec[:, s + samples_before : s + samples_after] for s in nearest_event_samples]
        av_specs.append(np.array(spec_windows).mean(axis=0))
    times = np.linspace(*window, av_specs[0].shape[1])
    # combine into dataframe
    dfs = []
    for event, av_spec in zip(events, av_specs):
        av_spec = pd.DataFrame(av_spec, columns=pd.MultiIndex.from_product([["time"], times]))
        info_df = _get_info_df(session, av_spec.index, signal_type, single_channel)
        info_df[("event", "")] = event
        info_df[("frequency", "")] = freqs
        dfs.append(pd.concat([info_df, av_spec], axis=1))
    return pd.concat(dfs, axis=0).reset_index(drop=True)


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


# %% Oscillation phase extraction


def get_nearest_theta_phase(
    session,
    times,
    signal_type="LFP",
    return_binned=False,
    n_bins=8,
):
    """
    get the oscillatory phase at the given times from the session's lfp/csd data.

    Note returns as a pd.Series so its easily encorporated into a DataFrame.
    """
    osc_phase = get_osc_phase(
        session,
        signal_type,
        freq_range=THETA_RANGE,
        N=4,
    )
    lfp_times = session.lfp_times
    idx = np.searchsorted(lfp_times, times, side="left")  # find insertion points
    idx[idx == len(lfp_times)] = len(lfp_times) - 1  # clip any out-of-bounds to the valid range
    idx_lo = np.maximum(idx - 1, 0)  # also consider the point just before each insertion point
    idx_hi = idx
    dist_lo = np.abs(
        times - lfp_times[idx_lo]
    )  # pick whichever of times[idx_lo] or times[idx_hi] is closer to each new_times
    dist_hi = np.abs(lfp_times[idx_hi] - times)
    take_hi = dist_hi < dist_lo  # where hi is closer than lo, take hi, else lo
    nearest_idx = np.where(take_hi, idx_hi, idx_lo)

    nearest_phase = pd.Series(osc_phase[nearest_idx])
    if not return_binned:
        return nearest_phase
    else:
        breaks = np.arange(-np.pi, np.pi + (np.pi / n_bins), (2 * np.pi / n_bins))
        bins = pd.IntervalIndex.from_breaks(breaks, closed="left")
        nearest_phase_binned = pd.cut(nearest_phase, bins=bins, include_lowest=True)
        return nearest_phase_binned


def get_osc_phase(
    session,
    signal_type="LFP",
    freq_range=(7, 10),
    N=4,
):
    """ """
    # get preprocessed signal
    if signal_type == "LFP":
        signal = get_LFP(session)
    elif signal_type == "CSD":
        signal = get_CSD(session)
    else:
        raise ValueError("signal_type must be 'LFP' or 'CSD'")

    # filter for input frequency range
    nyq = FS / 2
    b, a = butter(N, [(freq_range[0] / nyq), (freq_range[1] / nyq)], btype="bandpass")
    filt_osc = filtfilt(b, a, signal)
    analytic = hilbert(filt_osc)
    phase_hilbert = np.angle(analytic)
    return phase_hilbert


def quick_plot(session, freq_range=(6, 10), N=4, time_range=(95, 100)):
    # set up plot
    f, axes = plt.subplots(3, 1, figsize=(15, 5), sharex=True)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    lfp = get_LFP(session)
    nyq = FS / 2
    b, a = butter(N, [freq_range[0] / nyq, freq_range[1] / nyq], btype="bandpass")
    filt_osc = filtfilt(b, a, lfp)
    analytic = hilbert(filt_osc)
    phase_hilbert = np.angle(analytic)
    t = session.lfp_times
    t_mask = (t >= time_range[0]) & (t <= time_range[1])
    lfp = lfp[t_mask]
    filt_osc = filt_osc[t_mask]
    phase_hilbert = phase_hilbert[t_mask]
    t = t[t_mask]
    for ax, y, label in zip(axes.flatten(), [lfp, filt_osc, phase_hilbert], (["lfp", "filt", "phase"])):
        ax.plot(t, y)
        ax.set_ylabel(label)


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
