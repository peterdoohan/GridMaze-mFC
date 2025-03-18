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

# %% Functions


def test():
    fs = FS  # Sampling frequency in Hz
    sig, t = get_test_lfp()
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


# %%


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
