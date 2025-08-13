"""
meausre displacement in time across high-d theta trajectories
"""

# %% Imports
import json
import numpy as np
import pandas as pd

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.theta_mod import utils as tmu

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "theta_mod" / "trajectory_displacement"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "maze_day2date.json", "r") as input_file:
    MAZE_DAY2DATE = json.load(input_file)
MAZE2LATE_DAYS = {maze: [int(d) for d in list(MAZE_DAY2DATE[maze].keys())[-7:]] for maze in MAZE_DAY2DATE.keys()}


FRAME_RATE = 60
# %% aggregate data over sessions function


def get_theta_displacement_summary_df(
    smooth_SD=2.5,
    time_shift_range=(-1, 1),
    min_comparison_time=2,
    n_pcs=5,
    pcs_from="all_spikes",
    verbose=True,
    save=False,
):
    """ """
    save_path = RESULTS_DIR / f"theta_displacement_summary_{pcs_from}.parquet"
    if not save and save_path.exists():
        if verbose:
            print(f"Loading existing results from {save_path}")
        return pd.read_parquet(save_path)
    results = []
    failed_sessions = []
    for subject_ID in SUBJECT_IDS:
        for maze in MAZE_DAY2DATE.keys():
            # loop over subjects and mazes to not load too much data at once
            if verbose:
                print(f"Loading data: {subject_ID}, {maze}")
            sessions = gs.get_maze_sessions(
                subject_IDs=[subject_ID],
                maze_names=[maze],
                days_on_maze="late",
                with_data=[
                    "navigation_df",
                    "navigation_spike_counts_df",
                    "navigation_theta_spike_counts_df",
                    "cluster_metrics",
                    "navigation_spike_rates_df",  # need these for tuning curves when getting PCs
                    "cluster_place_direction_tuning_metrics",
                    "cluster_egocentric_action_tuning_metrics",
                    "cluster_distance_tuning_metrics",
                ],
                must_have_data=True,
            )
            for session in sessions:
                if verbose:
                    print(session.name)
                try:
                    alignment_df = get_session_theta_time_displacement(
                        session,
                        smooth_SD=smooth_SD,
                        time_shift_range=time_shift_range,
                        min_comparison_time=min_comparison_time,
                        n_pcs=n_pcs,
                    )
                    results.append(alignment_df)
                except Exception as e:
                    if verbose:
                        print(f"Failed to process session {session.name}: {e}")
                    failed_sessions.append(session.name)
    results_df = pd.concat(results, axis=0).reset_index(drop=True)
    if save:
        if verbose:
            print(f"Saving results to {save_path}")
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(save_path)
    if verbose:
        print(f"Failed sessions: {failed_sessions}")
    return results_df


# %% main fn


def get_session_theta_time_displacement(
    session,
    smooth_SD=2.5,  # s
    include_multi_unit=True,
    sqrt_spikes=False,
    zscore_spikes=False,
    n_pcs=5,
    pcs_from="all_spikes",
    time_shift_range=(-1, 1),
    min_comparison_time=2,  # s
):
    _kwargs = {
        "include_multi_unit": include_multi_unit,
        "sqrt_spikes": sqrt_spikes,
        "zscore_spikes": zscore_spikes,
        "smooth_SD": smooth_SD,
    }
    if pcs_from == "all_spikes":
        # # run PCA on on-task, navigation time data
        pca = tmu.get_pcs(session, n_pcs=n_pcs, **_kwargs)
    elif pcs_from == "distance_to_goal_tuning":
        # try PCs generated from distance tuning curves
        pca = tmu.get_distance_to_goal_pcs(session, n_pcs=n_pcs, include_multi_unit=include_multi_unit)
    elif pcs_from == "place_direction_tuning":
        # try on PCs generate from place_direction tuning curves
        pca = tmu.get_place_direction_pcs(session, n_pcs=n_pcs, include_multi_unit=include_multi_unit)
    elif pcs_from == "egocentric_action_tuning":
        # try on PCs generated from egocentric_action tuning curves
        pca = tmu.get_egocentric_action_pcs(session, include_multi_unit=include_multi_unit, n_pcs=n_pcs)
    else:
        raise ValueError("Invalid pcs_from input")

    # project spikes split by theta phase onto the same PC basis
    theta_pc_df = tmu.get_theta_pc_df(session, pca=pca, **_kwargs)
    phases = np.array(sorted([c for c in theta_pc_df.pc.columns.get_level_values(0).unique() if c != "theta_mean"]))
    trials = theta_pc_df.trial.dropna().unique()

    # measure displacement (mse) from mean theta traj at diff theta phases across diff timeshifts
    timeshift_frames = (time_shift_range[0] * FRAME_RATE, time_shift_range[1] * FRAME_RATE)
    timeshifts = np.arange(*timeshift_frames)
    min_comparison_frames = min_comparison_time * FRAME_RATE
    mses = np.full((len(trials), len(phases), len(timeshifts)), np.nan, dtype=float)
    for i, trial in enumerate(trials):
        _mask = (theta_pc_df.trial == trial) & (theta_pc_df.trial_phase == "navigation")
        _theta_df = theta_pc_df[_mask]
        if _theta_df.empty:
            continue

        mean_traj = _theta_df.pc.theta_mean.to_numpy(copy=False)
        if mean_traj.ndim == 1:
            mean_traj = mean_traj[:, None]

        for j, phase in enumerate(phases):
            phase_traj = _theta_df.pc[phase].to_numpy(copy=False)
            # compute all shifts' MSEs in one shot
            mses[i, j, :] = mse_over_shifts(phase_traj, mean_traj, timeshifts, min_comparison_frames)

    # return as df cols = x_shift, rows = phase x trial
    mses_rs = mses.reshape(-1, mses.shape[-1])
    time_shifts_seconds = timeshifts / FRAME_RATE
    df = pd.DataFrame(
        index=pd.MultiIndex.from_product([trials, phases], names=["trial", "theta_phase"]),
        columns=pd.MultiIndex.from_product([["mse"], time_shifts_seconds]),
        data=mses_rs,
    )
    # add other info
    df["subject_ID"] = session.subject_ID
    df["maze_name"] = session.maze_name
    df["day_on_maze"] = session.day_on_maze
    df.reset_index(inplace=True)
    return df


def mse_over_shifts(X, M, timeshifts, min_frames):
    """
    X: (n, d) phase trajectory (float, can contain NaN)
    M: (n, d) mean trajectory aligned to X (same shape)
    timeshifts: 1D iterable of ints (positive = shift down)
    returns: 1D array of length len(timeshifts) with MSEs (NaN if < min_frames)
    """
    X = np.asarray(X, dtype=float)
    M = np.asarray(M, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if M.ndim == 1:
        M = M[:, None]
    n, d = X.shape
    S = len(timeshifts)

    # allocate all shifted versions at once: (S, n, d)
    shifted = np.full((S, n, d), np.nan, dtype=float)

    for k, s in enumerate(timeshifts):
        if s >= 0:
            if s < n:
                shifted[k, s:, :] = X[: n - s, :]
            # else: whole plane stays NaN
        else:
            s_abs = -s
            if s_abs < n:
                shifted[k, : n - s_abs, :] = X[s_abs:, :]

    # rows valid for a given shift: no NaN in any channel after shifting
    valid = ~np.isnan(shifted).any(axis=2)  # (S, n)

    # squared diffs, masked where rows invalid for that shift
    diffs2 = (shifted - M[None, :, :]) ** 2  # (S, n, d)
    diffs2[~valid, :] = np.nan

    # require at least min_frames valid rows before averaging
    counts = valid.sum(axis=1)  # (S,)
    mse = np.nanmean(diffs2, axis=(1, 2))  # (S,)
    mse[counts <= min_frames] = np.nan
    return mse
