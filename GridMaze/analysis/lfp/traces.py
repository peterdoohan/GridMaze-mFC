"""
Script for visualising example LFP traces
@peterdoohan
"""

# %% Imports
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from GridMaze.analysis.event_aligned import lfp_utils as lu

# %% Global Variables

FS = 1500  # lfp sampling rate

# %% Functions


def plot_lfp_with_osc(session, window=(10, 10.5), freq_range=(7, 11), N=4, ax=None):
    # get lfp
    lfp_signal = lu.get_LFP(session)
    lfp_time = session.lfp_times
    # extract osc
    nyq = FS / 2
    b, a = butter(N, [(freq_range[0] / nyq), (freq_range[1] / nyq)], btype="bandpass")
    filt_osc = filtfilt(b, a, lfp_signal)
    # plotting
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(5, 1.5))
    ax.spines[["top", "right"]].set_visible(False)
    window_mask = (lfp_time >= window[0]) & (lfp_time <= window[1])
    ax.plot(lfp_time[window_mask], lfp_signal[window_mask], color="k", alpha=1, label="LFP", lw=0.5)
    ax.plot(
        lfp_time[window_mask],
        filt_osc[window_mask],
        color="crimson",
        label=f"{freq_range[0]}-{freq_range[1]} Hz",
        lw=1.5,
        alpha=0.8,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("LFP (uV)")
    ax.legend()
