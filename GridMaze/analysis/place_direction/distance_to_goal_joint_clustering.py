"""This module if for joint clustering place_direction tuning vectors and distance to goal tuning vectors to establish the 
relationship between these two variables"""

# %% Imports
import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import matplotlib.gridspec as gridspec


from . import plot_components as pc
from . import get_neural_place_direction_df as npd
from . import dimensionality_reduction as dr
from .. import get_sessions as gs
from ...maze import plotting as mp
from ...maze import representations as mr

# %% Global varaibles
RESULTS_PATH = "../results/joint_clustering"

# %% Analysis functions


def plot_distance_place_direction_loading_sparceness(sessions, n_components=8):
    """From a list of sessions, compiles a joint distance-to-goal and place-direction tuning vector [n_neurons, n_distance_tuning_feature +
    n_place-direction tuning features], does dimensionaliy reduction using NMF, and plots the sum of the ordered loadings for each component
    """
    distance_place_direction_dfs = [
        get_joint_distance_place_direction_df(
            s,
            normalise_distance_tuning="length",
            normalise_place_direction_tuning="length",
            distance_to_goal_weight=1,
            place_direction_weight=1,
        )
        for s in sessions
    ]
    multisession_distance_place_direction_df = pd.concat(distance_place_direction_dfs, axis=0).to_numpy()
    nmf = NMF(n_components=n_components, **dr.NMF_KWARGS)  # use nmf conditions consistant with other analyses
    component_loadings = nmf.fit_transform(multisession_distance_place_direction_df)
    ordered_loadings = np.sort(component_loadings, axis=1)[
        :, ::-1
    ]  # order componetents for each cell by max loading and sum over neurons
    normalised_loadings = ordered_loadings / ordered_loadings.sum(axis=1).reshape(-1, 1)
    av_loadings = normalised_loadings.mean(axis=0)
    # plotting
    f, ax = plt.subplots(1, 1, figsize=(5, 5), clear=True)
    ax.plot(av_loadings, color="black")
    ax.set_xlabel("Component")
    ax.set_ylabel("Sum Loading")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return


def plot_joint_distance_place_direction_components(
    sessions, n_components=8, distance_smoothing=2, shuffled=False, save=False
):
    """From a list of sessions, compiles a joint distance-to-goal and place-direction tuning vector [n_neurons, n_distance_tuning_feature +
    n_place-direction tuning features], does dimensionaliy reduction using NMF and plots the distance-to-goal tuning and place-direction
    tuning of the resulting components"""
    maze_number = sessions[0].maze_number
    simple_maze = mr.get_simple_maze(maze_number=maze_number)
    maze_joint_vector_dfs = [
        get_joint_distance_place_direction_df(
            s,
            normalise_distance_tuning="length",
            normalise_place_direction_tuning="length",
            distance_to_goal_weight=1,
            place_direction_weight=1,
        )
        for s in sessions
    ]
    if shuffled:
        maze_joint_vector_dfs = [shuffle_joint_distance_place_direction_df(df) for df in maze_joint_vector_dfs]
    joint_vector_dfs = pd.concat(maze_joint_vector_dfs)
    joint_nmf_df = dr.get_nmf_df(joint_vector_dfs, n_components).T
    distance_to_goal_df = joint_nmf_df.distance_to_goal
    distances = distance_to_goal_df.columns.to_numpy(dtype=float)
    place_direction_tuning_df = joint_nmf_df.place_direction
    place_direction_tuning_df.columns = pd.MultiIndex.from_tuples(
        [tuple(c.split("_")) for c in place_direction_tuning_df.columns]
    )
    place_direction_tuning_df = place_direction_tuning_df.T  # now equiv to pca or nmf dfs
    place_direction_tuning_df.index.names = ["maze_position", "direction"]
    for component in range(n_components):
        # distance tunning
        av = distance_to_goal_df.loc[component]
        if distance_smoothing:
            av = gaussian_filter1d(av, sigma=distance_smoothing)
        # place direction
        place_direction_values = place_direction_tuning_df[component]
        print(place_direction_tuning_df)
        # plotting
        f = plot_joint_distance_to_goal_and_place_direction_tuning(
            component, distances, av, None, None, place_direction_values, simple_maze
        )
        if save:
            filename = f"maze{maze_number}_component{component}.pdf"
            f.savefig(os.path.join(RESULTS_PATH, filename))
    return joint_nmf_df


def get_joint_nmf_df(sessions, n_components=8, shuffled=False):
    maze_joint_vector_dfs = [
        get_joint_distance_place_direction_df(
            s,
            normalise_distance_tuning=False,
            normalise_place_direction_tuning=False,
            distance_to_goal_weight=1,
            place_direction_weight=1,
        )
        for s in sessions
    ]
    if shuffled:
        maze_joint_vector_dfs = [shuffle_joint_distance_place_direction_df(df) for df in maze_joint_vector_dfs]
    joint_vector_dfs = pd.concat(maze_joint_vector_dfs)
    joint_nmf_df = dr.get_nmf_df(joint_vector_dfs, n_components).T
    return joint_nmf_df


def get_joint_distance_place_direction_df(
    session,
    normalise_distance_tuning="length",
    normalise_place_direction_tuning="length",
    distance_to_goal_weight=1,
    place_direction_weight=1,
):
    distance_tuning_df = session.distance_to_goal_aligned_rates_df.geodesic_distance_to_goal.average
    place_direction_tuning_df = npd.get_place_direction_df(
        session, normalisation_method=normalise_place_direction_tuning
    )
    if distance_tuning_df is None or place_direction_tuning_df is None:
        return None
    # note place_direction_df will have fewer neurons bc it holds a minimum firing rate constraint
    distance_tuning_df = distance_tuning_df[distance_tuning_df.index.isin(place_direction_tuning_df.index)]
    # reindex each df
    place_direction_tuning_df.columns = pd.Index(
        [("place_direction", c[0] + "_" + c[1]) for c in place_direction_tuning_df.columns]
    )
    distance_tuning_df.columns = pd.Index([("distance_to_goal", c) for c in distance_tuning_df.columns])
    assert (distance_tuning_df.index == place_direction_tuning_df.index).all()
    if normalise_distance_tuning == "length":
        distance_tuning_df = distance_tuning_df.div(distance_tuning_df.sum(axis=1), axis=0)
    elif normalise_distance_tuning == "max":
        distance_tuning_df = distance_tuning_df.div(distance_tuning_df.max(axis=1), axis=0)
    # weight each df
    distance_tuning_df = distance_tuning_df.mul(distance_to_goal_weight)
    place_direction_tuning_df = place_direction_tuning_df.mul(place_direction_weight)
    joint_vectors_df = pd.concat([distance_tuning_df, place_direction_tuning_df], axis=1)
    return joint_vectors_df


def shuffle_joint_distance_place_direction_df(joint_distance_place_direction_df, shuffled_property="distance_to_goal"):
    """This function takes an input df generated by the get_joint_distance_place_direction_df function
    and shuffles either the distance to goal tuning on place direction tuning across neurons.
    """
    index = joint_distance_place_direction_df.index
    other_property = "place_direction" if shuffled_property == "distance_to_goal" else "distance_to_goal"
    shuffled_df = joint_distance_place_direction_df.loc[:, pd.IndexSlice[shuffled_property, :]].sample(frac=1, axis=0)
    shuffled_df.index = index
    other_df = joint_distance_place_direction_df.loc[:, pd.IndexSlice[other_property, :]]
    return pd.concat([shuffled_df, other_df], axis=1)


# %% Supporting functions


def plot_joint_distance_to_goal_and_place_direction_tuning(
    cluster_no,
    distances,
    distance_av,
    distance_lower,
    distance_upper,
    place_direction_values,
    simple_maze,
):
    fgrid = gridspec.GridSpec(2, 3)
    f = plt.figure(figsize=(6, 4))
    f.tight_layout()
    ax1 = plt.subplot(fgrid[0, 0])
    pos1 = ax1.get_position()
    new_pos1 = [pos1.x0, pos1.y0 - 0.2, pos1.width, pos1.height]
    ax1.set_position(new_pos1)
    ax2 = plt.subplot(fgrid[0:2, 1:3])
    ax1.plot(distances, distance_av, color="k", lw=3)
    if not distance_lower is None and not distance_upper is None:
        ax1.fill_between(
            distances,
            distance_lower,
            distance_upper,
            color="silver",
            alpha=0.5,
        )
    ax1.set_xlabel("Distance-to-goal (m)")
    ax1.set_ylabel("Component Loading")
    ax1.set_title(f"Component {cluster_no}")
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    # place-direction tuning
    mp.plot_directed_heatmap(
        simple_maze,
        place_direction_values,
        ax=ax2,
        colormap="Reds",
        title="",
        value_label="Component Loading",
        silhouette_color="silver",
        silhouette_node_size=250,
        silhouette_edge_size=8,
    )
    return f


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
            "distance_to_goal_aligned_rates_df",
        ],
    )
    return sessions


# %%
