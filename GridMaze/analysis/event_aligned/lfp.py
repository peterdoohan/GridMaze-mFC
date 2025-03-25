"""
library for plotting lfp aligned to trial events
"""

# %% Imports
import pandas as pd
import numpy as np
import json
import mne
from GridMaze.analysis.core import get_sessions as gs
from scipy.stats import zscore
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


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

# %% average over all session


def test2(event="cue", window=(-2, 2)):
    """
    Quick plot average CSD for a session
    """
    av_csds = []
    subject = "m6"
    for maze in MAZE_CONFIGS.keys():
        all_days = [int(d) for d in MAZE_DAY2DATE[maze].keys()]
        days = all_days[-7:]  # last 7 days (late sessions)
        for day in days:
            session = gs.get_maze_sessions(
                subject_IDs=[subject],
                maze_names=[maze],
                days_on_maze=[day],
                with_data=["trials_df", "lfp_times", "lfp_signal", "lfp_metrics", "cluster_metrics"],
                must_have_data=True,
            )
            # get csd
            csd = get_CSD(session, single_channel=False)
            # zscore
            csd = zscore(csd, axis=0)
            # get signal around event
            trials_df = session.trials_df
            times = session.lfp_times
            event_times = trials_df.time[event].values
            nearest_event_samples = np.array([np.argmin(np.abs(times - t)) for t in event_times])
            samples_before, samples_after = int(window[0] * FS), int(window[1] * FS)
            csd_windows = [csd[s + samples_before : s + samples_after] for s in nearest_event_samples]
            av_csd = np.array(csd_windows).mean(axis=0)
            av_csds.append(av_csd)
    x = np.array([i for i in av_csds if i.shape[0] != 0]).mean(axis=0)  # hack
    times = np.linspace(*window, x.shape[0])
    plt.plot(times, x)
    plt.axvline(0, color="k", ls="--")
    return x


def test():
    """
    Note cannot load all session objects at once, will overload memory
    """
    event = "cue"
    signal_type = "CSD"
    single_channel = False
    window = (-2, 2)
    freqs = np.geomspace(3, 250, 100)
    session_specs = []
    skipped_session = []
    for subject in SUBJECT_IDS:
        for maze in MAZE_CONFIGS.keys():
            all_days = [int(d) for d in MAZE_DAY2DATE[maze].keys()]
            days = all_days[-7:]  # last 7 days (late sessions)
            for day in days:
                session = gs.get_maze_sessions(
                    subject_IDs=[subject],
                    maze_names=[maze],
                    days_on_maze=[day],
                    with_data=["trials_df", "lfp_times", "lfp_signal", "lfp_metrics", "cluster_metrics"],
                    must_have_data=True,
                )
                print(session)
                try:
                    session_specs.append(
                        _get_session_event_aligned_spectrogram(
                            session, event, signal_type, single_channel, window, freqs, plot=False
                        )
                    )
                except ValueError as e:
                    print(e)
                    print(f"skipping session: {session.name}")
                    skipped_session.append(session.name)
    # should save this out in a nice way where it can be loaded and used for different analyses
    # eg, random effect, fixed effects etc.

    return session_specs


def _get_spectrogram_df(
    session,
    signal_type="CSD",
    single_channel=False,
    window=(-2, 2),
    freqs=np.geomspace(3, 250, 100),
    overwrite=False,
):
    """
    Calculates the average spectogram pwoer of LFP/CSD signal aliged to cue and reward (store in combined
    dataframe for convience).

    Note: Function tries to load data from disk if available, otherwise computes and saves to disk.
    Set overwrite to True to force recomputation and
    """
    # process av specfram for each event
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
    info_df = pd.DataFrame(index=spec_df.index)
    info_df[("subject_ID", "")] = session.subject_ID
    info_df[("maze_name", "")] = session.maze_name
    info_df[("day_on_maze", "")] = session.day_on_maze
    info_df[("signal_type", "")] = signal_type
    info_df[("single_channel", "")] = single_channel
    info_df[("signal_type", "")] = signal_type
    info_df.columns = pd.MultiIndex.from_tuples(info_df.columns)
    # combine dataframes
    df = pd.concat([info_df] + dfs, axis=1)
    if save:
        save_path = LFP_RESULTS / f"{session.name}.parquet"
        df.to_parquet(save_path, compression="gzip")
    return df


# %% Event (Cue or Reward) aligned spectrograms (from wavelet decomposition)


def _get_session_event_aligned_spectrogram(
    session,
    event="cue",
    signal_type="LFP",
    single_channel=False,
    window=(-2, 2),
    freqs=np.geomspace(3, 250, 100),
    zscore_freqs=True,
    plot=True,
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
    av_spec = np.array(spec_windows).mean(axis=0)
    if plot:
        times = np.linspace(*window, av_spec.shape[1])
        _plot_spectrogram(av_spec, times, freqs, event, f"{signal_type} single channel: {single_channel}")
    return av_spec


def _plot_spectrogram(x, times, freqs, event, signal_type, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1, clear=True, figsize=(10, 3))
    im = ax.imshow(
        x,
        aspect="auto",
        extent=[times[0], times[-1], freqs[-1], freqs[0]],
        cmap="coolwarm",
    )
    ax.set_xlabel(f"{event} Aligned Time (s)")
    ax.axvline(0, color="white", linestyle="--")
    ax.invert_yaxis()
    # ax.set_yscale("log")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{signal_type}")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Power (z-scored)")


# %% CSD functions


def get_LFP(session, shank=3, single_channel=False):
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
    return lfp


def get_CSD(session, orientation="horizontal", single_channel=False):
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
    return CSD


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
        set_1 = [1, 3, 5]
        set_2 = [2, 4, 6]
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


# %% Wavelet functions


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
    # Create a finely-sampled, conjugated Morlet wavelet as a template.
    morlet_wavelet = np.conj(_morlet(M, gaussian_width, window_length, precision))
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


def compute_wavelet_transform_fft(sig, freqs, fs, gaussian_width=1.5, window_length=1.0, precision=16, norm="l1"):
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
