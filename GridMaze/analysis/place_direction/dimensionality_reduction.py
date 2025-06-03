""" """

# %% Imports
import pandas as pd
import numpy as np

from sklearn.decomposition import NMF, PCA


from GridMaze.analysis.cluster_tuning import spatial
from GridMaze.analysis.core import get_sessions as gs

# %% Global Variables


# %% Functions


def get_nmf_df(
    place_direction_df,
    n_components=8,
    kwargs={
        "init": "random",
        "random_state": 0,
        "solver": "mu",
        "beta_loss": "kullback-leibler",
        "max_iter": 1000,
    },
):
    model = NMF(n_components=n_components, **kwargs)
    data_matrix = place_direction_df.to_numpy()
    decomp_components = model.fit(data_matrix).components_
    nmf_df = pd.DataFrame(data=decomp_components.T, index=place_direction_df.columns, columns=range(n_components))
    return nmf_df


def get_pca_df(place_direction_df, n_components=8):
    model = PCA(random_state=0, n_components=n_components)
    data_matrix = place_direction_df.to_numpy()
    decomp_components = model.fit(data_matrix).components_
    pca_df = pd.DataFrame(
        data=decomp_components.T, index=place_direction_df.columns, columns=range(len(decomp_components))
    )
    return pca_df


def get_population_place_direction_tuning(subject_IDs="all", maze_name="maze_1", late_sessions=True):
    """ """
    days_on_maze = "late" if late_sessions else "all"
    sessions = gs.get_maze_sessions(
        subject_IDs=subject_IDs,
        maze_names=[maze_name],
        days_on_maze=days_on_maze,
        with_data=[
            "navigation_df",
            "navigation_spike_rates_df",
            "cluster_metrics",
            "cluster_place_direction_tuning_metrics",
        ],
        must_have_data=True,
    )
    dfs = []
    for session in sessions:
        print(session.name)
        df = _get_session_place_direction_tuning(session)
        if df is None:
            continue  # not pd tuned clusters
        dfs.append(df)
    pop_pd_tuning_df = pd.concat(dfs, axis=0, ignore_index=True)
    return pop_pd_tuning_df


def _get_session_place_direction_tuning(
    session,
    fill_nans="mean",
    normalisation="length",
    place_direction_tuned=True,
    min_split_corr=0.4,
    navigation_only=True,
    moving_only=True,
    exclude_time_at_goal=True,
    minimum_occupancy=0.5,
    max_steps_from_goal=30,
    verbose=True,
):
    """
    Returns place-direction tuning for all place-direction tuned clusters in a session.
    w/ options for filtering clusters going in, data going into heatmap calculation, then
    further value filling and normalisation of the heatmaps
    """
    # load data
    simple_maze = session.simple_maze()
    navigation_rates_df = session.get_navigation_activity_df(type="rates", cluster_kwargs={"single_units": True})
    pd_tuning_metrics = session.cluster_place_direction_tuning_metrics
    # filter for place_direction tuned clusters (roughly partitioned via split correlations)
    # see analysis/processing/get_place_direction_tuning_metrics.py
    cluster_filters = [pd_tuning_metrics.single_unit]
    if place_direction_tuned:
        cluster_filters.append(pd_tuning_metrics.place_direction_tuned)
    if min_split_corr is not None:
        cluster_filters.append(pd_tuning_metrics.split_half_corr.value.ge(min_split_corr))
    keep_clusters = pd_tuning_metrics[np.logical_and.reduce(cluster_filters)].index
    if len(keep_clusters) == 0:
        if verbose:
            print(f"No place-direction cluster found with split_half_corr >= {min_split_corr}")
        return None
    reject_clusters = [c for c in navigation_rates_df.firing_rate.columns.values if c not in keep_clusters]
    navigation_rates_df = navigation_rates_df.drop([("firing_rate", c) for c in reject_clusters], axis=1)
    # get average place direction tuning
    place_direction_df = spatial._get_place_direction_df(
        simple_maze,
        navigation_rates_df,
        navigation_only,
        moving_only,
        exclude_time_at_goal,
        minimum_occupancy,
        max_steps_from_goal,
    )
    # fill nan values of unvisited place-directions
    if fill_nans == "mean":
        place_direction_df.T.fillna(place_direction_df.mean(axis=1), inplace=True)  # replace nans with the mean
    elif fill_nans == "zero":
        place_direction_df.fillna(0, inplace=True)
    else:
        raise ValueError(f"Unknown fill_nans method: {fill_nans}")
    # normalise over clusters
    if normalisation == "mean":
        place_direction_df = place_direction_df.div(place_direction_df.mean(axis=1), axis=0)
    elif normalisation == "length":
        place_direction_df = place_direction_df.div(place_direction_df.pow(2).sum(axis=1).pow(0.5), axis=0)
    elif normalisation == "max":
        place_direction_df = place_direction_df.div(place_direction_df.max(axis=1), axis=0)
    else:
        raise ValueError(f"Unknown normalisation method: {normalisation}")
    # return df
    place_direction_df.columns.names = ["maze_position", "direction"]
    place_direction_df.sort_index(axis=1, inplace=True)
    return place_direction_df
