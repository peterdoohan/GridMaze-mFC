"""
Library for generating bases that efficiently code for place-direction, used for encoding analyses.
Eg, for given session, use all other late sessions from that subjects to generate place-direction bases and then
use those bases to efficently ecnode place-direction for the current session.
@peterdoohan
"""

# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, PCA

from GridMaze.maze import plotting as mp
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.cluster_tuning.spatial import _get_place_direction_df

# %% Global Variables


# %% Functions


def get_test_sessions(subject_ID="m3", maze_name="maze_1"):
    """ """
    return gs.get_maze_sessions(
        subject_IDs=[subject_ID],
        maze_names=[maze_name],
        days_on_maze="late",
        with_data=["navigation_df", "cluster_metrics", "navigation_spike_rates_df"],
        must_have_data=False,
    )


def get_place_direction_bases(
    sessions,
    n_bases=8,
    dim_red="nmf",
    navigation_only=True,
    moving_only=True,
    exclude_time_at_goal=True,
    minimum_occupancy=0.5,
    max_steps_from_goal=30,
    plot=False,
):
    """
    input sessions must be from the same maze
    """
    # combine place-direction heatmaps across sessions
    simple_maze = sessions[0].simple_maze()
    dfs = []
    for session in sessions:
        navigation_rates_df = session.get_navigation_activity_df(type="rates", cluster_kwargs={"single_units": True})
        place_direction_tuning_df = _get_place_direction_df(
            simple_maze,
            navigation_rates_df,
            navigation_only,
            moving_only,
            exclude_time_at_goal,
            minimum_occupancy,
            max_steps_from_goal,
        )
        dfs.append(place_direction_tuning_df)
    place_direction_df = pd.concat(dfs, axis=0)  # n_neurons x n_place_directions
    # replace NaNs (with cluser mean) and normalise heatmaps (sum to 1)
    place_direction_df = place_direction_df.apply(lambda row: row.fillna(row.mean()), axis=1)
    place_direction_df = place_direction_df.apply(lambda row: row / row.sum(), axis=1)

    # set up model for dimensionality reduction
    if dim_red == "pca":
        model = PCA(n_components=n_bases, random_state=0)
    elif dim_red == "nmf":
        model = NMF(
            n_components=n_bases,
            init="random",
            random_state=0,
            solver="mu",
            beta_loss="kullback-leibler",
            max_iter=10_000,
        )
    else:
        raise ValueError("dim_red must be either 'pca' or 'nmf'")

    # fit model to data
    data_matrix = place_direction_df.to_numpy()
    decomp_components = model.fit(data_matrix).components_
    # create dataframe of components
    bases_df = pd.DataFrame(data=decomp_components, index=range(n_bases), columns=place_direction_df.columns).T
    if plot:
        plot_bases(bases_df, simple_maze, dim_red=dim_red)
    return bases_df


def plot_bases(bases_df, simple_maze, dim_red="pca", axes=None):
    """ """
    if dim_red == "pca":
        cmap = "coolwarm"
        neg = True
    elif dim_red == "nmf":
        cmap = "silver2red"
        neg = False
    else:
        raise ValueError("dim_red must be either 'pca' or 'nmf'")
    n_bases = bases_df.shape[1]
    if axes is None:
        fig, axes = plt.subplots(2, n_bases // 2, figsize=(20, 10), sharex=True)
        axes = axes.flatten()
    for i in range(n_bases):
        ax = axes[i]
        basis = bases_df[i]
        mp.plot_directed_heatmap(
            simple_maze,
            basis,
            ax=ax,
            colormap=cmap,
            allow_negative=neg,
            silhouette_node_size=400,
            silhouette_edge_size=8,
        )
