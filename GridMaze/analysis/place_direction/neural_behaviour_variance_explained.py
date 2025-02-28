"""This analysis file is to test if behaviour and mFC neural activity occupy the same low dimensional subspace"""

# %% Imports
import numpy as np
import pandas as pd
from scipy.linalg import svd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from GridMaze.analysis.core import get_sessions as gs

from GridMaze.analysis.place_direction import get_neural_place_direction_df as npd
from GridMaze.analysis.place_direction import get_behavioural_place_direction_df as bpd

# from .dimensionality_reduction2 import get_analysis_sessions


# %% Global variables


# %% Functions


def plot_neural_behavioural_variance_explained(sessions, ve_method="pca"):
    neural_place_direction_df = npd.get_multisession_neural_place_direction_df(sessions, normalisation_method="length")
    behavioural_place_direction_df = bpd.get_multisession_behavioural_place_direction_df(
        sessions, normalisation_method="length"
    )
    # avoid column scrambling
    column_order = sorted(neural_place_direction_df.columns)
    neural_place_direction_df = neural_place_direction_df.reindex(column_order, axis=1)
    behavioural_place_direction_df = behavioural_place_direction_df.reindex(column_order, axis=1)
    N = neural_place_direction_df.to_numpy()  # [n_neurons, n_features]
    B = behavioural_place_direction_df.to_numpy()  # [n_trials, n_features]
    if ve_method == "pca":
        ve_fn = get_pca_variance_explained
    elif ve_method == "svd":
        ve_fn = get_svd_variance_explained
    else:
        assert ValueError(f"ve_method {ve_method} not recognised")
    N_explains_N = ve_fn(N, N)
    B_explains_N = ve_fn(B, N)
    B_explains_B = ve_fn(B, B)
    N_explains_B = ve_fn(N, B)
    # plotting
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(N_explains_N, label="Neural explains Neural", color="red")
    ax.plot(B_explains_N, label="Behaviour explains Neural", color="red", ls="--")
    ax.plot(B_explains_B, label="Behaviour explains Behavioural", color="blue")
    ax.plot(N_explains_B, label="Neural explains Behavioural", color="blue", ls="--")
    ax.legend()
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative variance explained")
    return


def get_svd_variance_explained(A, B):  # A & B: [n_samples, n_features]
    """Calculates the cumulative variance of matrix B that's explained by the first i orthonormal bases of matrix A using SVD."""
    U, Sigma, Vt = np.linalg.svd(A, full_matrices=False)
    M = B @ Vt.T  # B projected onto the svd bases of A
    pc_exp_var = np.square(M).sum(axis=0)
    cumsum_exp_var = np.cumsum(pc_exp_var) / pc_exp_var.sum()
    return np.concatenate(([0], cumsum_exp_var))


def get_pca_variance_explained(A, B):  # A & B: [n_samples, n_features]
    """Calculates the cumulative variance of matrix B that's explained by the first i components of matrix A."""
    model = PCA(random_state=0)
    model.fit(A)
    M = model.transform(B)  # [n_samples, n_components]
    pc_exp_var = np.square(M).sum(axis=0)
    cumsum_exp_var = np.cumsum(pc_exp_var) / pc_exp_var.sum()
    return cumsum_exp_var


def get_analysis_sessions(maze, subject, late=True):
    subject = [subject] if not subject == "all" else subject
    days_on_maze = "late" if late == True else "all"
    sessions = gs.get_maze_sessions(
        subject_IDs=subject,
        maze_names=[maze],
        days_on_maze=days_on_maze,
        with_data=[
            "navigation_df",
            "navigation_spike_counts_df",
            "navigation_spike_rates_df",
            "cluster_metrics",
            # "cluster_analysis_metrics_df",
            "trajectory_decisions_df",
        ],
    )
    return sessions
