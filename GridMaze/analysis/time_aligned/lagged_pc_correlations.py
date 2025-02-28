"""This srcipt is for looking at the lagged correlations of neural PCs during navigation"""
# %% Imports
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from ..processing.filter_navigation_rates_df import _filter_navigation_df

# %% Global variables
FRAME_RATE = 60


# %% Functions


def get_navigation_pc_df(session, navigation_only=True, exclude_time_at_goal=True, standardise=True, n_components=None):
    """Note changing cluster_type to all will include mua clusters in the reward consumption period, leading to supurious theta patterns that
    are most likely licking artifacts."""
    navigation_rates_df = session.get_navigation_activity_df(activity_type="firing_rate", cluster_type="all")
    # filter down to what we want to calculate PCs over
    navigation_rates_df = _filter_navigation_df(
        navigation_rates_df,
        minimum_firing_rate=0.25,
        navigation_only=navigation_only,
        moving_only=False,
        exclude_time_at_goal=exclude_time_at_goal,
    )
    rates_array = navigation_rates_df.firing_rate.to_numpy()
    if standardise:
        scaler = StandardScaler()
        rates_array = scaler.fit_transform(rates_array)
    model = PCA(n_components=n_components, random_state=0)
    principal_components = model.fit_transform(rates_array)
    # replace firing rates with PCs
    navigation_rates_df = navigation_rates_df.drop(columns=navigation_rates_df.filter(like="firing_rate", axis=1))
    pc_df = pd.DataFrame(
        data=principal_components,
        columns=pd.MultiIndex.from_product([["principle_component"], range(len(principal_components[1]))]),
        index=navigation_rates_df.index,
    )
    navigation_pc_df = navigation_rates_df.join(pc_df)
    return navigation_pc_df


def plot_session_lagged_pc_correlations_summary(session, n_components=10, max_lag=2):
    navigation_pc_df = get_navigation_pc_df(
        session, navigation_only=True, exclude_time_at_goal=True, n_components=n_components
    )
    # compute pc pair corr separately for each trial to avoid disontinuities
    trial_fds = []
    trial_lagged_correlations = []
    for trial in navigation_pc_df.trial.unique():
        trial_df = navigation_pc_df[navigation_pc_df.trial == trial]
        pc_timeseries = trial_df.principle_component.to_numpy().T
        if pc_timeseries.shape[1] < max_lag * FRAME_RATE + 1:
            continue
        lagged_correlations_array = get_lagged_correlations_array(pc_timeseries, max_lag, n_components)
        trial_lagged_correlations.append(lagged_correlations_array)
        freq, av_frequency_decomposition = pc_pair_average_frequency_decomposition(lagged_correlations_array)
        trial_fds.append(av_frequency_decomposition)
    pc_pair_lagged_corrs = np.nanmean(np.stack(trial_lagged_correlations, axis=0), axis=0)
    trial_av_fd = np.nanmean(np.stack(trial_fds, axis=0), axis=0)
    # plotting
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), clear=True)
    _plot_pc_pair_lagged_corr_heatmap(ax1, pc_pair_lagged_corrs, n_components, max_lag)
    ax2.plot(freq, trial_av_fd, color="k")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Amplitude")
    ax2.set_ylim(0, 0.01)
    return trial_fds


def get_lagged_correlations_array(pc_timeseries, max_lag, n_components):
    max_lag_frames = int(max_lag * FRAME_RATE)
    n_pairs = n_components * (n_components - 1) // 2
    lags = np.arange(-max_lag_frames, max_lag_frames + 1)  # at frame rate resolution
    autocorr_results = np.empty((n_pairs, len(lags)))
    pair_index = 0
    for i in range(n_components):
        for j in range(i + 1, n_components):
            corrs = []
            for lag in lags:
                if lag < 0:
                    corrs.append(pearsonr(pc_timeseries[i, :lag], pc_timeseries[j, -lag:])[0])
                elif lag > 0:
                    corrs.append(pearsonr(pc_timeseries[i, lag:], pc_timeseries[j, :-lag])[0])
                else:
                    corrs.append(pearsonr(pc_timeseries[i, :], pc_timeseries[j, :])[0])
            autocorr_results[pair_index, :] = corrs
            pair_index += 1
    return autocorr_results


def _plot_pc_pair_lagged_corr_heatmap(ax, lagged_corr_array, n_features, max_lag):
    sns.heatmap(ax=ax, data=lagged_corr_array, cmap="coolwarm", center=0)
    ax.set_xlabel("Time Lag (seconds)")
    ax.set_ylabel("Feature Pair")
    pc_labels = []
    row = 0
    for i in range(n_features):
        for j in range(i + 1, n_features):
            pc_labels.append(f"PC{i} - PC{j}")
            row += 1
    yticks = np.arange(len(pc_labels)) + 0.5
    ax.set_yticks(yticks)
    ax.set_yticklabels(pc_labels, rotation=0, ha="right", size="small")
    # Set xticks and xticklabels
    total_ticks = 11  # 5 on either side of 0, plus the 0
    tick_locations = np.linspace(0, lagged_corr_array.shape[1] - 1, total_ticks) + 0.5
    tick_labels = [str(round(val, 2)) for val in np.linspace(-max_lag, max_lag, total_ticks)]
    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    ax.axvline(x=lagged_corr_array.shape[1] // 2 + 0.5, color="black", linewidth=1, ls="--", alpha=0.5)
    return


# %% FTequency decomposition functions


def plot_frequency_spectrum(signal, FRAME_RATE):
    """
    Plots the frequency spectrum of a signal.

    Parameters:
    - signal: A numpy array with the signal values.
    - FRAME_RATE: The frame rate at which the signal was sampled.

    """

    # Calculate the FFT and the frequency bins
    fft_result = np.fft.fft(signal)
    n = len(signal)
    frequency = np.fft.fftfreq(n, d=1 / FRAME_RATE)

    # Only plot the first half of the frequencies (due to symmetry for real-valued signals)
    n_half = n // 2
    fft_result_half = fft_result[:n_half]
    frequency_half = frequency[:n_half]

    # Compute the magnitude of the FFT and normalize
    magnitude = np.abs(fft_result_half) / n_half

    # Plotting the spectrum
    plt.figure(figsize=(12, 6))
    plt.plot(frequency_half, magnitude)
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
    return frequency_half, magnitude


def get_frequency_decomposition(signal):
    fft_result = np.fft.fft(signal)
    n = len(signal)
    frequency = np.fft.fftfreq(n, d=1 / FRAME_RATE)
    n_half = n // 2
    fft_result_half = fft_result[:n_half]
    frequency_half = frequency[:n_half]
    magnitude = np.abs(fft_result_half) / n_half
    return frequency_half, magnitude


def pc_pair_average_frequency_decomposition(lagged_correlations_array):
    pc_pair_fds = []
    for a in lagged_correlations_array:
        freq, magnitude = get_frequency_decomposition(a)
        pc_pair_fds.append(magnitude)
    return freq, np.mean(np.stack(pc_pair_fds, axis=0), axis=0)
