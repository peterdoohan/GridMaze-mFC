"""This scrip contains functions that perform dimensionality reduction (NMF, PCA) on neuron place-direction tuning data"""

# %% Imports
import pandas as pd
from scipy.linalg import svd
from sklearn.decomposition import NMF, PCA

from . import get_behavioural_place_direction_df as bpd
from . import get_neural_place_direction_df as npd
from . import plot_components as pc
from .. import get_sessions as gs


# %% Global variables
NMF_KWARGS = {
    "init": "random",
    "random_state": 0,
    "solver": "mu",
    "beta_loss": "kullback-leibler",
    "max_iter": 1000,
}

# %% Functions


def get_nmf_df(place_direction_df, n_components):
    model = NMF(n_components=n_components, **NMF_KWARGS)
    data_matrix = place_direction_df.to_numpy()
    decomp_components = model.fit(data_matrix).components_
    nmf_df = pd.DataFrame(data=decomp_components.T, index=place_direction_df.columns, columns=range(n_components))
    return nmf_df


def get_pca_df(place_direction_df, n_components=None):
    model = PCA(random_state=0, n_components=n_components)
    data_matrix = place_direction_df.to_numpy()
    decomp_components = model.fit(data_matrix).components_
    pca_df = pd.DataFrame(
        data=decomp_components.T, index=place_direction_df.columns, columns=range(len(decomp_components))
    )
    return pca_df


def get_svd_df(place_direction_df, centered=False, n_components=None):
    data_matrix = place_direction_df.to_numpy()
    if centered:
        data_matrix = data_matrix - data_matrix.mean(axis=0)
    U, s, V = svd(data_matrix, full_matrices=False)
    if n_components is not None:
        V = V[:n_components]
    svd_df = pd.DataFrame(data=V.T, index=place_direction_df.columns, columns=range(len(V)))
    return svd_df


# %% plot nmf components from behaviour and neural data


def plot_nmf_components(data_type="neural", maze_number=1, n_components=8):
    sessions = get_analysis_sessions(maze_number)
    simple_maze = sessions[0].simple_maze()
    if data_type == "neural":
        place_direction_df = npd.get_multisession_neural_place_direction_df(sessions)
        cmap = "Reds"
    elif data_type == "behaviour":
        place_direction_df = bpd.get_multisession_behavioural_place_direction_df(sessions)
        cmap = "Blues"
    nmf_df = get_nmf_df(place_direction_df, n_components)
    pc.plot_nmf_components(nmf_df, simple_maze, title=f"{data_type.capitalize()} NMF Components", colormap=cmap)
    return


# %% Compare neural and behaviour place-direction components


def plot_neural_behaviour_place_direction_components(sessions, method="pca", n_components=8):
    simple_maze = sessions[0].simple_maze()
    neural_place_direction_df = npd.get_multisession_neural_place_direction_df(sessions, normalisation_method="length")
    behavioural_place_direction_df = bpd.get_multisession_behavioural_place_direction_df(
        sessions, normalisation_method="length"
    )
    if method == "pca":
        neural_decomp_df = get_pca_df(neural_place_direction_df, n_components=n_components)
        behavioural_decomp_df = get_pca_df(behavioural_place_direction_df, n_components=n_components)
    elif method == "nmf":
        neural_decomp_df = get_nmf_df(neural_place_direction_df, n_components=n_components)
        behavioural_decomp_df = get_nmf_df(behavioural_place_direction_df, n_components=n_components)
    elif method == "svd":
        neural_decomp_df = get_svd_df(neural_place_direction_df, n_components=n_components)
        behavioural_decomp_df = get_svd_df(behavioural_place_direction_df, n_components=n_components)
    # plotting
    if method == "svd" or method == "pca":
        pc.plot_pca_components(neural_decomp_df, simple_maze, title="Neural", pos_cmap="Reds", neg_cmap="Blues")
        pc.plot_pca_components(
            behavioural_decomp_df, simple_maze, title="Behavioural", pos_cmap="Greens", neg_cmap="Purples"
        )
    elif method == "nmf":
        pc.plot_nmf_components(neural_decomp_df, simple_maze, title="Neural", colormap="Reds")
        pc.plot_nmf_components(behavioural_decomp_df, simple_maze, title="Behavioural", colormap="Blues")
    return


def get_analysis_sessions(maze_number):
    sessions = gs.get_sessions(
        subject_IDs="all",
        maze_number=[maze_number],
        day_on_maze="late",
        with_data=[
            "navigation_df",
            "navigation_spike_counts_df",
            "navigation_spike_rates_df",
            "cluster_metrics",
            "cluster_analysis_metrics_df",
            "trajectory_decisions_df",
        ],
    )
    return sessions
