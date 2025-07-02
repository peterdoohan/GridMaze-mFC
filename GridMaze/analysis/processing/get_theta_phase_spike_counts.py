"""
Create analysis data structure with neuron spikes counted in per theta phase bin
"""

# %% Imports
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from GridMaze.analysis.event_aligned import lfp_utils as lu

# %% Global Variables
FS_LFP = 1500

# %% Functions


def test():
    """ """
    return


# %% LFP wrangling functions


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
    nyq = FS_LFP / 2
    b, a = butter(N, [(freq_range[0] / nyq), (freq_range[1] / nyq)], btype="bandpass")
    filt_osc = filtfilt(b, a, signal)
    analytic = hilbert(filt_osc)
    phase_hilbert = np.angle(analytic)
    return phase_hilbert


def get_LFP(session, shank=3, single_channel=False, remove_artifacts=True):
    """
    copy from event_aligned/lfp_units.py
    to work straight from processed data
    """
    # load data
    lfp_metrics = session.lfp_metrics
    cluster_metrics = session.cluster_metrics
    lfp_signal = session.lfp_signal
    if single_channel:  # converting from channel_id to channel_index in lfp_signal to get the signal
        channel_id = lu._get_single_channel_for_LFP(lfp_metrics, cluster_metrics, shank)
        channel_index = lfp_metrics[lfp_metrics.contact.id == channel_id].index[0]
        lfp = lfp_signal[:, channel_index]
    else:  # average lfp over multiple channels
        channel_ids = lu._get_shank_channels_for_LFP(lfp_metrics, cluster_metrics, shank)
        channel_indices = lfp_metrics.contact.id.isin(channel_ids).index.values
        lfp = lfp_signal[:, channel_indices].mean(axis=1)
    if remove_artifacts:
        lfp = lu._remove_artifacts(lfp, thres=500)
    return lfp
