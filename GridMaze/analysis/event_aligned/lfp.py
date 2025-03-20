"""
library for plotting lfp aligned to trial events
based on code from Pynapple: https://github.com/pynapple-org/pynapple/blob/main/pynapple/process/wavelets.py!
"""

# %% Imports
import numpy as np
import mne
from GridMaze.analysis.core import get_sessions as gs
import matplotlib.pyplot as plt


# %% Global Variables

FS = 1_500  # lfp sampling frequency

# %% Dev Functions


def test(session):
    fs = FS  # Sampling frequency in Hz
    sig, t = get_test_csd(session)
    freqs = np.geomspace(3, 250, 100)
    cwt_result = compute_wavelet_transform(sig, freqs, fs)

    # For visualization (optional)
    plt.imshow(np.abs(cwt_result), aspect="auto", extent=[t[0], t[-1], freqs[-1], freqs[0]])
    plt.xlabel("Time (s)")
    plt.gca().invert_yaxis()
    plt.yscale("log")
    plt.ylabel("Frequency (Hz)")
    plt.title("Wavelet Transform Power")
    plt.colorbar(label="Power")
    plt.show()

    plt.plot(t, sig)
    plt.show()


def get_test_lfp():
    session = gs.get_maze_sessions(
        subject_IDs=["m2"],
        maze_names=["maze_2"],
        days_on_maze=[11],
        with_data=[
            "trials_df",
            "lfp_signal",
            "lfp_times",
            "lfp_metrics",
            "cluster_metrics",
        ],
    )
    # load_data
    lfp_metrics = session.lfp_metrics
    lfp_signal = session.lfp_signal
    lfp_times = session.lfp_times
    # change dtype
    lfp_signal = lfp_signal.astype(np.float64)
    # choose times
    lfp_signal = lfp_signal[250_000:255_000]
    lfp_times = lfp_times[250_000:255_000]
    # low pass
    lfp_signal = mne.filter.filter_data(
        lfp_signal.T, FS, l_freq=None, h_freq=300, method="fir", fir_design="firwin", verbose=False
    ).T
    # remove bad channels
    good_channel_mask = lfp_metrics.contact.qc == "good"
    lfp_metrics = lfp_metrics[good_channel_mask].reset_index(drop=True)
    lfp_signal = lfp_signal[:, good_channel_mask]
    # common average reference
    # lfp_signal = lfp_signal - lfp_signal.mean(axis=1)[:, None]
    # choose a channel with lots of single units as an example channel
    cluster_metrics = session.cluster_metrics
    best_channels = cluster_metrics[cluster_metrics.single_unit].contact.id.mode().values
    for c in best_channels:
        contact_info = lfp_metrics[lfp_metrics.contact.id == c]
        if not contact_info.empty:
            break
    channel_ind = contact_info.index[0]
    lfp_signal = lfp_signal[:, channel_ind]
    return lfp_signal, lfp_times


def get_test_csd(session):
    csd_signal = get_CSD(session, "horizontal")
    csd_times = session.lfp_times
    # choose times
    csd_signal = csd_signal[10_000:80_000]
    csd_times = csd_times[10_000:80_000]
    return csd_signal, csd_times


# %% CSD functions


def plot_CSD_QC(session, times=(5, 7), filter_CSD=True):
    lfp_signal = session.lfp_signal
    lfp_metrics = session.lfp_metrics
    lfp_times = session.lfp_times
    mask = (lfp_times > times[0]) & (lfp_times < times[1])
    for ort in ["horizontal", "vertical"]:
        c1, c2, c3 = _get_channels_for_CSD(
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


def get_CSD(session, orientation="horizontal"):
    """
    CSD = c2 - ((c1 + c3) / 2)
    for c1, c2, c3 colinear contacts in the same region
    """
    # load data
    lfp_metrics = session.lfp_metrics
    cluster_metrics = session.cluster_metrics
    lfp_signal = session.lfp_signal
    contact_ids = _get_channels_for_CSD(lfp_metrics, cluster_metrics, orientation)
    # get corresponding indcies for the contact ids
    contact_indices = [lfp_metrics[lfp_metrics.contact.id == c].index[0] for c in contact_ids]
    c1, c2, c3 = [lfp_signal[:, c] for c in contact_indices]
    CSD = c2 - (c1 + c3) / 2
    return CSD


def _get_channels_for_CSD(lfp_metrics, cluster_metrics, orientation="horizontal", verbose=False):
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


def compute_wavelet_transform(sig, freqs, fs, gaussian_width=1.5, window_length=1.0, precision=16, norm="l1"):
    filter_bank, _ = generate_morlet_filterbank(freqs, fs, gaussian_width, window_length, precision)
    n_freqs, filter_len = filter_bank.shape
    n_time = len(sig)
    cwt = np.zeros((n_freqs, n_time), dtype=complex)

    for i in range(n_freqs):
        # Compute full convolution
        full_conv_real = np.convolve(sig, np.real(filter_bank[i]), mode="full")
        full_conv_imag = np.convolve(sig, np.imag(filter_bank[i]), mode="full")

        # Calculate the starting index for extracting the central portion
        start_idx = (len(full_conv_real) - n_time) // 2
        conv_real = full_conv_real[start_idx : start_idx + n_time]
        conv_imag = full_conv_imag[start_idx : start_idx + n_time]

        cwt[i] = conv_real + 1j * conv_imag

    # Normalize the coefficients if desired.
    if norm == "l1":
        cwt = cwt / (fs / freqs[:, np.newaxis])
    elif norm == "l2":
        cwt = cwt / (fs / np.sqrt(freqs)[:, np.newaxis])

    return cwt
