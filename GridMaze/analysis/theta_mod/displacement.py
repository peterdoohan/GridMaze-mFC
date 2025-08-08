"""
meausre displacement in time across high-d theta trajectories
"""

# %% Imports
import numpy as np

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.theta_mod import utils
from sklearn.metrics import mean_squared_error

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
    time_shift_range=(-5, 5),
    min_comparison_time=2,  # s
):
    timeshift_frames = (time_shift_range[0] * FRAME_RATE, time_shift_range[1] * FRAME_RATE)
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
    output_dfs = []
    # loop over trials, optimal displacement
    for trial in trials:
        _mask = (theta_pc_df.trial == trial) & (theta_pc_df.trial_phase == "navigation")
        _theta_df = theta_pc_df[_mask]
        if _theta_df.empty:
            continue
        # get average trajectory
        mean_traj = _theta_df.pc.theta_mean
        timeshifts = np.arange(*timeshift_frames)
        phase_shift_mse = np.array((len(phases), len(timeshifts)))
        for i, phase in enumerate(phases):
            phase_traj = _theta_df.pc[phase]
            # calculate displacement under differnt time shifts
            for j, time_shift in enumerate(timeshifts):
                shift_phase_traj = phase_traj.shift(time_shift)
                nan_mask = shift_phase_traj.isna()
                _mean_traj = mean_traj[~nan_mask].values
                _shift_traj = shift_phase_traj[~nan_mask].values
                if not _mean_traj.shape[0] > min_comparison_frames:
                    continue
                phase_shift_mse[i, j] = mean_squared_error(_mean_traj, _shift_traj)
