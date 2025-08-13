"""
meausre displacement in time across high-d theta trajectories
"""

# %% Imports
import numpy as np

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.theta_mod import utils
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter1d

# %% Global Variables
FRAME_RATE = 60
# %% Functions


def get_session_theta_time_displacement(
    session,
    smooth_SD=2,  # s
    include_multi_unit=True,
    sqrt_spikes=False,
    zscore_spikes=False,
    n_pcs=5,
    frac_var_exp=None,
    time_shift_range=(-1, 1),
    min_comparison_time=2,  # s
):
    timeshift_frames = (time_shift_range[0] * FRAME_RATE, time_shift_range[1] * FRAME_RATE)
    timeshifts = np.arange(*timeshift_frames)
    min_comparison_frames = min_comparison_time * FRAME_RATE
    _kwargs = {
        "include_multi_unit": include_multi_unit,
        "sqrt_spikes": sqrt_spikes,
        "zscore_spikes": zscore_spikes,
        "smooth_SD": smooth_SD,
    }
    # run PCA on on-task, navigation time data
    pca, n_pcs = utils.get_pcs(session, n_pcs=n_pcs, frac_var_exp=frac_var_exp, **_kwargs)
    # project spikes split by theta phase onto the same PC basis
    theta_pc_df = utils.get_theta_pc_df(session, pca=pca, n_pcs=n_pcs, **_kwargs)
    phases = np.array(sorted([c for c in theta_pc_df.pc.columns.get_level_values(0).unique() if c != "theta_mean"]))
    trials = theta_pc_df.trial.dropna().unique()
    # loop over trials, optimal displacement
    mses = np.full((len(trials), len(phases), len(timeshifts)), np.nan)
    for i, trial in enumerate(trials):
        print(trial)
        _mask = (theta_pc_df.trial == trial) & (theta_pc_df.trial_phase == "navigation")
        _theta_df = theta_pc_df[_mask]
        if _theta_df.empty:
            continue
        # get average trajectory
        mean_traj = _theta_df.pc.theta_mean
        for j, phase in enumerate(phases):
            phase_traj = _theta_df.pc[phase]
            # calculate displacement under differnt time shifts
            for k, time_shift in enumerate(timeshifts):
                shift_phase_traj = phase_traj.shift(time_shift)
                nan_mask = shift_phase_traj.isna().any(axis=1)
                _mean_traj = mean_traj[~nan_mask].values
                _shift_traj = shift_phase_traj[~nan_mask].values
                if not _mean_traj.shape[0] > min_comparison_frames:
                    continue
                mses[i, j, k] = mean_squared_error(_mean_traj, _shift_traj)
    return mses


def get_mse(trials, timeshifts, phases, theta_pc_df, min_comparison_frames):
    mses = np.full((len(trials), len(phases), len(timeshifts)), np.nan)
    for i, trial in enumerate(trials):
        print(trial)
        _mask = (theta_pc_df.trial == trial) & (theta_pc_df.trial_phase == "navigation")
        _theta_df = theta_pc_df[_mask]
        if _theta_df.empty:
            continue
        # get average trajectory
        mean_traj = _theta_df.pc.theta_mean
        for j, phase in enumerate(phases):
            phase_traj = _theta_df.pc[phase]
            # calculate displacement under differnt time shifts
            for k, time_shift in enumerate(timeshifts):
                shift_phase_traj = phase_traj.shift(time_shift)
                nan_mask = shift_phase_traj.isna().any(axis=1)
                _mean_traj = mean_traj[~nan_mask].values
                _shift_traj = shift_phase_traj[~nan_mask].values
                if not _mean_traj.shape[0] > min_comparison_frames:
                    continue
                mses[i, j, k] = mean_squared_error(_mean_traj, _shift_traj)
    return


def get_mse2(trials, timeshifts, phases, theta_pc_df, min_comparison_frames):
    # Pre‐allocate output:
    n_trials = len(trials)
    n_phases = len(phases)
    n_shifts = len(timeshifts)
    mses = np.full((n_trials, n_phases, n_shifts), np.nan)

    for i, trial in enumerate(trials):
        # extract once per trial
        df_trial = theta_pc_df[theta_pc_df.trial == trial]
        if df_trial.empty:
            continue

        # pre‐compute mean trajectory (T, D)
        mean_traj = df_trial.pc.theta_mean.values
        mask_mean = np.isnan(mean_traj).any(axis=1)

        for j, phase in enumerate(phases):
            X = df_trial.pc[phase].values  # shape (T, D)
            mask_X = np.isnan(X).any(axis=1)

            for k, shift in enumerate(timeshifts):
                if shift >= 0:
                    A = mean_traj[: -shift or None]
                    B = X[shift:]
                    mask = mask_mean[: -shift or None] | mask_X[shift:]
                else:
                    A = mean_traj[-shift:]
                    B = X[: shift or None]
                    mask = mask_mean[-shift:] | mask_X[:-shift]

                # only keep rows where neither is masked
                valid = ~mask
                if valid.sum() <= min_comparison_frames:
                    continue

                diff = (A[valid] - B[valid]) ** 2
                mses[i, j, k] = diff.mean()
    return mses


def get_mse3(trials, timeshifts, phases, theta_pc_df, min_comparison_frames):
    # Pre‐allocate output:
    n_trials = len(trials)
    n_phases = len(phases)
    n_shifts = len(timeshifts)
    mses = np.full((n_trials, n_phases, n_shifts), np.nan)

    for i, trial in enumerate(trials):
        # extract once per trial
        df_trial = theta_pc_df[theta_pc_df.trial == trial]
        if df_trial.empty:
            continue

        # pre‐compute mean trajectory (T, D)
        mean_traj = df_trial.pc.theta_mean.values
        mask_mean = np.isnan(mean_traj).any(axis=1)

        for j, phase in enumerate(phases):
            X = df_trial.pc[phase].values  # shape (T, D)
            shifted = []
            for shift in timeshifts:
                if shift >= 0:
                    pad = np.full((shift, X.shape[1]), np.nan)
                    shifted.append(np.vstack([pad, X[:-shift]]))
                else:
                    pad = np.full((-shift, X.shape[1]), np.nan)
                    shifted.append(np.vstack([X[-shift:], pad]))
            shifted = np.stack(shifted)  # (K, T, D)

            # Now compute squared errors broadcasted:
            #   mean_traj      -> shape (1, T, D)
            #   shifted        -> shape (K, T, D)
            err2 = (mean_traj[None] - shifted) ** 2  # (K, T, D)

            # mask out any row with nan in either
            mask_mean = np.isnan(mean_traj).any(axis=1)  # (T,)
            mask_shift = np.isnan(shifted).any(axis=2)  # (K, T)
            valid = ~(mask_mean[None] | mask_shift)  # (K, T)

    # flatten time+dim:
    err2[~valid[..., None]] = np.nan
    mse_all = err2.reshape(len(timeshifts), -1).mean(axis=1)
    return
