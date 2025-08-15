"""
Test theta amplitude at different depths and shanks (A->P) across subjects
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from scipy.signal import butter, filtfilt, hilbert

from GridMaze.analysis.core import get_sessions as gs

# %% Globs

FS = 1500
FRAME_RATE = 60
THETA_RANGE = (7, 11)

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

RESULTS_DIR = RESULTS_PATH / "lfp"

# %% Functions


def get_all_channel_av_theta_powers(verbose=False, save=False, n_jobs=-1):
    """ """
    save_path = RESULTS_DIR / "channel_avg_theta_powers.csv"
    if save_path.exists() and not save:
        if verbose:
            print(f"Loading population theta modulation from {save_path}")
        df = pd.read_csv(save_path, index_col=0, header=[0, 1])
        df.columns = pd.MultiIndex.from_tuples([c if not "Unnamed" in c[1] else (c[0], "") for c in df.columns])
        return df

    dfs = []
    for subject_ID in SUBJECT_IDS:
        if verbose:
            for maze in ["maze_1", "maze_2", "rooms_maze"]:
                print(f"Loading sessions for {subject_ID} {maze}")
                sessions = gs.get_maze_sessions(
                    subject_IDs=[subject_ID],
                    maze_names=[maze],
                    days_on_maze="all",
                    with_data=[
                        "navigation_df",
                        "lfp_signal",
                        "lfp_times",
                        "lfp_metrics",
                    ],
                    must_have_data=True,
                )
                if n_jobs is not None:
                    subject_dfs = Parallel(n_jobs=n_jobs, verbose=verbose)(
                        delayed(get_session_channel_theta_powers)(session, verbose=verbose) for session in sessions
                    )
                else:
                    subject_dfs = []
                    for session in sessions:
                        if verbose:
                            print(session.name)
                        subject_dfs.append(get_session_channel_theta_powers(session, verbose=verbose))
                dfs.extend(subject_dfs)

    theta_powers_df = pd.concat(dfs, axis=0)
    if save:
        if verbose:
            print(f"Saving channel average theta powers to {save_path}")
        theta_powers_df.to_csv(save_path)
    return theta_powers_df


def get_session_channel_theta_powers(
    session,
    freq_range=THETA_RANGE,
    N=4,
    verbose=True,
):
    """ """
    if verbose:
        print(session.name)
    # load data
    lfp_signal = session.lfp_signal
    lfp_metrics = session.lfp_metrics
    lfp_times = session.lfp_times
    # catch rare inst where times is one sample too long
    if lfp_signal.shape[0] != lfp_times.shape[0]:
        lfp_times = lfp_times[:-1]
    navigation_df = session.navigation_df
    # get mask for moving and navigation times
    nav_moving_mask = _mask_lfp(lfp_times, navigation_df)
    n_channels = lfp_signal.shape[1]
    av_theta_powers = np.zeros(n_channels)
    for i in range(n_channels):
        if not lfp_metrics.iloc[i].contact.qc == "good":
            av_theta_powers[i] = np.nan
        else:
            _lfp_signal = lfp_signal[:, i]
            # filter for input frequency range
            nyq = FS / 2
            b, a = butter(N, [(freq_range[0] / nyq), (freq_range[1] / nyq)], btype="bandpass")
            filt_osc = filtfilt(b, a, _lfp_signal)
            analytic_signal = hilbert(filt_osc)
            amplitude_envelope = np.abs(analytic_signal)
            av_theta_powers[i] = np.mean(amplitude_envelope[nav_moving_mask])
    lfp_metrics[("av_theta_power", "")] = av_theta_powers
    lfp_metrics[("subject_ID", "")] = session.subject_ID
    lfp_metrics[("maze_name", "")] = session.maze_name
    lfp_metrics[("day_on_maze", "")] = session.day_on_maze
    return lfp_metrics


def _mask_lfp(lfp_times, navigation_df):
    """ """
    df = navigation_df.copy()
    tfames = df.time.to_numpy()
    maskframes = ((df.trial_phase == "navigation") & df.moving).to_numpy().astype(np.int8)

    # edge finding on boolian series
    d = np.diff(np.r_[0, maskframes, 0])
    starts = np.flatnonzero(d == 1)  # indices where True runs start
    stops = np.flatnonzero(d == -1)

    mids = (tfames[:-1] + tfames[1:]) / 2
    left_edges = np.r_[tfames[0] - 0.5 * (tfames[1] - tfames[0]), mids]
    right_edges = np.r_[mids, tfames[-1] + 0.5 * (tfames[-1] - tfames[-2])]

    interval_starts = left_edges[starts]
    interval_ends = right_edges[stops - 1]

    # Map intervals to indices on the high-rate time axis
    i0 = np.searchsorted(lfp_times, interval_starts, side="left")
    i1 = np.searchsorted(lfp_times, interval_ends, side="left")

    lfp_mask = np.zeros(lfp_times.shape[0], dtype=bool)
    for a, b in zip(i0, i1):
        if a < b:  # guard against empty intervals
            lfp_mask[a:b] = True
    return lfp_mask
