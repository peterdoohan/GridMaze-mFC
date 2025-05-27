"""
supporting functions to get oscillatory phase data from lfp/cpd
"""

# %% Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, sosfiltfilt
from GridMaze.analysis.event_aligned import lfp_utils as lu

# %% Global Variables
FS = 1500  # lfp sampling frequency

THETA_RANGE = (7, 11)

_4HZ_RANGE = (2, 5)  # 4Hz oscillation range
# %% Convience Function


def get_nearest_osc_phase(
    session,
    new_times,
    signal_type="LFP",
    band="theta",
    return_binned=True,
    n_bins=8,
):
    """
    get the oscillatory phase at the given times from the session's lfp/csd data.

    Note returns as a pd.Series so its easily encorporated into a DataFrame.
    """
    if band == "theta":
        osc_phase = get_osc_phase(
            session,
            signal_type,
            freq_range=THETA_RANGE,
            N=4,
        )
    elif band == "4Hz":
        osc_phase = get_osc_phase(
            session,
            signal_type,
            freq_range=_4HZ_RANGE,
            N=3,
        )  # 4th order butterworth filter returns NaNs?
    else:
        raise ValueError(f"band must be 'theta' or '4Hz', not {band}")
    times = session.lfp_times
    idx = np.searchsorted(times, new_times, side="left")  # find insertion points
    idx[idx == len(times)] = len(times) - 1  # clip any out-of-bounds to the valid range
    idx_lo = np.maximum(idx - 1, 0)  # also consider the point just before each insertion point
    idx_hi = idx
    dist_lo = np.abs(
        new_times - times[idx_lo]
    )  # pick whichever of times[idx_lo] or times[idx_hi] is closer to each new_times
    dist_hi = np.abs(times[idx_hi] - new_times)
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


# %% Core function


def get_osc_phase(
    session,
    signal_type="LFP",
    freq_range=(7, 11),
    N=4,
):
    """ """
    # get preprocessed signal
    if signal_type == "LFP":
        signal = lu.get_LFP(session)
    elif signal_type == "CSD":
        signal = lu.get_CSD(session)
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
    lfp = lu.get_LFP(session)
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
    axes[-1].set_xlabel("time (s)")
