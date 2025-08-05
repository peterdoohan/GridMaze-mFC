"""
Neural activity unfolds over a trial as a trajectory though a high-dimensional space (with some smoothing).
Neighboring timepoints on this trajectory define the current direction of movement (vector) through this trajectory.
Points between different phases of a theta oscillation can also define a vector in this high-d space. If representations
move from the present to the future (including up to a goal a long way-away) over theta cycles, vectors between theta phases and
vectors between neural timepoints should be aligned (non-orthogoal as predicted by chance).
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.theta_mod import utils as tmu

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

FRAME_RATE = 60  # Hz

# %% Run on all sessions


def get_theta_alignment_summary_df(smooth_SD=2, vector_window=2, verbose=True):
    """ """
    for subject_ID in SUBJECT_IDS:
        if verbose:
            print(f"Loading data: {subject_ID}")
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names="all",
            days_on_maze="all",
            with_data=[
                "navigation_df",
                "navigation_spike_counts_df",
                "navigation_theta_spike_counts_df",
                "cluster_metrics",
            ],
            must_have_data=True,
        )
        results = []
        for session in sessions:
            if verbose:
                print(session.name)
            alignment_df = get_session_alignment_angles(session, smooth_SD=smooth_SD, vector_window=vector_window)
            results.append(alignment_df)
    return pd.concat(results, axis=0).reset_index(drop=True)


# %% Functions


def get_session_alignment_angles(
    session,
    smooth_SD=2,  # s
    include_multi_unit=True,
    sqrt_spikes=False,
    zscore_spikes=False,
    vector_window=2,  # s
):
    _kwargs = {
        "include_multi_unit": include_multi_unit,
        "sqrt_spikes": sqrt_spikes,
        "zscore_spikes": zscore_spikes,
        "smooth_SD": smooth_SD,
    }
    # run PCA on on-task, navigation time data
    pca, n_pcs = tmu.get_pcs(session, frac_var_exp=0.9, **_kwargs)
    # project all spikes onto the PC basis defined above (organised in df)
    neural_pc_df = tmu.get_neural_pc_df(session, pca=pca, n_pcs=n_pcs, **_kwargs)
    # project spikes split by theta phase onto the same PC basis
    theta_pc_df = tmu.get_theta_pc_df(session, pca=pca, n_pcs=n_pcs, **_kwargs)
    # theta phases not including "theta_mean"
    phases = np.array(sorted([c for c in theta_pc_df.pc.columns.get_level_values(0).unique() if c != "theta_mean"]))
    trials = neural_pc_df.trial.dropna().unique()
    output_dfs = []
    # loop over trials, calculate alignment angles (to goal and current trajectory)
    for trial in trials:
        _mask = (neural_pc_df.trial == trial) & (neural_pc_df.trial_phase == "navigation")
        _neural_df = neural_pc_df[_mask]
        _theta_df = theta_pc_df[_mask]
        if _neural_df.empty or _theta_df.empty:
            continue
        # define vector window within trial
        idx = _neural_df.index
        window_edges = np.arange(idx[0], idx[-1], (vector_window * FRAME_RATE))
        window_mids = window_edges[:-1] + 0.5 * np.diff(window_edges)
        window_mids = window_mids.astype(int)  # convert to int for indexing
        # get theta phase vectors (n_phases, n_samples, n_pcs)
        theta_vectors = _get_theta_phase_vectors(_theta_df.loc[window_mids], phases)
        # get neural vectors (vectors between window edges, n_samples, n_pcs)
        traj_vectors = _get_neural_trajectory_vectors(_neural_df.loc[window_edges])
        # get vector between now and goal (n_samples, n_pcs)
        goal_vectors = _get_goal_vectors(_neural_df.loc[window_mids], at_goal=_neural_df.iloc[-1])
        # construct output df
        trajectory_alignment_df = pd.DataFrame(
            index=window_mids,
            columns=pd.MultiIndex.from_product([["trajectory_angle"], phases]),
            data=get_theta_phase_alignment(traj_vectors, theta_vectors),
        )
        goal_alignment_df = pd.DataFrame(
            index=window_mids,
            columns=pd.MultiIndex.from_product([["goal_angle"], phases]),
            data=get_theta_phase_alignment(goal_vectors, theta_vectors),
        )
        info_df = _neural_df[
            [("subject_ID", ""), ("maze_name", ""), ("day_on_maze", ""), ("trial", ""), ("trial_unique_ID", "")]
        ].loc[window_mids]
        grouped_df = _neural_df.groupby(pd.cut(idx, bins=window_edges, labels=False, include_lowest=True))
        info_df[("moving", "")] = grouped_df.moving.any().values
        info_df[("distance_to_goal", "geodesic")] = grouped_df.distance_to_goal.mean().distance_to_goal.geodesic.values
        info_df[("steps_to_goal", "future")] = grouped_df.steps_to_goal.mean().steps_to_goal.future.values
        assert all((goal_alignment_df.index == info_df.index)), ValueError("index mismatch")
        trial_output_df = pd.concat([info_df, trajectory_alignment_df, goal_alignment_df], axis=1)
        output_dfs.append(trial_output_df)
    return pd.concat(output_dfs, axis=0)


# %% calc angles


def get_theta_phase_alignment(base_vectors, theta_vectors):
    n_phases = theta_vectors.shape[0]
    n_samples = base_vectors.shape[0]
    phase_angles = np.zeros((n_phases, n_samples))
    for i in range(n_phases):
        theta_vector = theta_vectors[i]
        dots = np.einsum("ij,ij->i", base_vectors, theta_vector)  # dot products per timepoint
        n_norm = np.linalg.norm(base_vectors, axis=1)  # norms per timepoint
        t_norm = np.linalg.norm(theta_vector, axis=1)
        den = n_norm * t_norm
        cos_sim = np.clip(dots / np.maximum(den, 1e-12), -1.0, 1.0)
        phase_angles[i] = np.arccos(cos_sim)  # shape (n_timepoints,), radian
    return phase_angles.T  # n_samples, n_phases


# %% calc vectors
def _get_theta_phase_vectors(df, phases):
    """
    return vectors between each theta phase and the theta phase mean
    for each sample (df row): np.array of shape (n_phases, n_samples, n_pcs)
    """
    vectors = np.zeros((len(phases), df.shape[0], len(df.pc.columns.get_level_values(1).unique())))
    theta_mean = df.xs("theta_mean", level=1, axis=1).values
    for i, phase in enumerate(phases):
        vectors[i] = df.xs(phase, level=1, axis=1).values - theta_mean
    return vectors  # n_phases x n_samples x n_pcs


def _get_neural_trajectory_vectors(df):
    pcs = df.pc.values
    # vectors between each timepoint and the next
    vectors = pcs[1:] - pcs[:-1]
    return vectors


def _get_goal_vectors(df, at_goal):
    # vectors between each timepoint and the goal
    pt_goal = at_goal.pc.values.astype(float)
    vectors = pt_goal - df.pc.values
    return vectors
