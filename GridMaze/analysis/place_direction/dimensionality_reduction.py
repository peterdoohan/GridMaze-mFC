""" """

# %% Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.metrics import explained_variance_score

from GridMaze.maze import plotting as mp
from GridMaze.analysis.cluster_tuning import spatial
from GridMaze.analysis.core import get_sessions as gs


# %% Global Variables


# %% Functions


def plot_nmf_components(population_tuning_df, simple_maze, n_components=8, cmap="Reds", axes=None):
    """ """
    nmf_df = get_nmf_df(population_tuning_df, n_components)
    if axes is None:
        f, axes = plt.subplots(1, n_components, figsize=(6 * n_components, 6))
    for i in range(n_components):
        c = nmf_df[i]
        ax = axes[i]
        mp.plot_directed_heatmap(
            simple_maze,
            c,
            ax,
            colormap=cmap,
            silhouette_node_size=500,
            silhouette_edge_size=10,
            star_base_length=0.045,
            max_point_length=0.03,
        )

    return


def get_nmf_df(
    place_direction_df,
    n_components=8,
):
    model = NMF(n_components=n_components, init="random", random_state=0, max_iter=10_000)
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


def get_svd_df(place_direction_df, n_components=8):
    """ """
    model = TruncatedSVD(n_components=n_components, random_state=0)
    data_matrix = place_direction_df.to_numpy()
    decomp_components = model.fit(data_matrix).components_
    svd_df = pd.DataFrame(
        data=decomp_components.T, index=place_direction_df.columns, columns=range(len(decomp_components))
    )
    return svd_df


def get_population_place_direction_tuning(
    subject_IDs="all",
    maze_name="maze_1",
    late_sessions=True,
    sessions=None,
    return_list=False,
    fill_nans="mean",
    normalisation="length",
    verbose=False,
):
    """ """
    # if session objects are not input, generate them from input filters
    if sessions is None:
        days_on_maze = "late" if late_sessions else "all"
        if verbose:
            print("Loading sessions ...")
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
        if verbose:
            print(session.name)
        df = get_session_place_direction_tuning(session, fill_nans, normalisation)
        if df is None:
            continue  # not pd tuned clusters
        dfs.append(df)
    if return_list:
        return dfs, sessions
    else:
        pop_pd_tuning_df = pd.concat(dfs, axis=0, ignore_index=True)
        return pop_pd_tuning_df


def get_session_place_direction_tuning(
    session,
    navigation_rates_df=None,
    fill_nans="mean",
    normalisation="length",
    place_direction_tuned=True,
    min_split_corr=0.5,
    navigation_only=True,
    moving_only=True,
    exclude_time_at_goal=True,
    minimum_occupancy=0.5,
    max_steps_from_goal=30,
    verbose=False,
):
    """
    Returns place-direction tuning for all place-direction tuned clusters in a session.
    w/ options for filtering clusters going in, data going into heatmap calculation, then
    further value filling and normalisation of the heatmaps
    """
    # load data
    simple_maze = session.simple_maze()
    pd_tuning_metrics = session.cluster_place_direction_tuning_metrics
    if navigation_rates_df is None:
        navigation_rates_df = session.get_navigation_activity_df(type="rates", cluster_kwargs={"single_units": True})
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
    if fill_nans:
        if fill_nans == "mean":
            place_direction_df.T.fillna(place_direction_df.mean(axis=1), inplace=True)  # replace nans with the mean
        elif fill_nans == "zero":
            place_direction_df.fillna(0, inplace=True)
        else:
            raise ValueError(f"Unknown fill_nans method: {fill_nans}")
    # normalise over clusters
    if normalisation:
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


# %% get CV var exp to determine best number of components


def test(tuning_dfs, sessions, component_range=(1, 20), max_jobs=20):
    """ """
    results = []
    for i in range(len(sessions)):
        test_session = sessions[i]
        print(test_session.name)
        test_df = tuning_dfs[i]
        train_df = pd.concat(tuning_dfs[:i] + tuning_dfs[i + 1 :], axis=0)
        # do CV NMF for each number of components
        X_train = train_df.values  # [n_neurons, n_place_directions]
        X_test = test_df.values
        fold_results = Parallel(n_jobs=max_jobs)(
            delayed(_process_fold)(X_train, X_test, test_session, n_components)
            for n_components in range(component_range[0], component_range[1] + 1)
        )
        results.extend(fold_results)
    return results


def _process_fold(X_train, X_test, test_session, n_components):
    """ """
    nmf = NMF(n_components=n_components, random_state=0, max_iter=10_000)
    W_train = nmf.fit_transform(X_train)
    H = nmf.components_
    X_train_pred = W_train @ H
    # project test data onto learned components
    W_test = nmf.transform(X_test)
    X_test_pred = W_test @ H
    # calculate variance explained
    return {
        "subject_ID": test_session.subject_ID,
        "maze_name": test_session.maze_name,
        "day_on_maze": test_session.day_on_maze,
        "n_components": n_components,
        "train_var_exp": explained_variance_score(X_train, X_train_pred),
        "test_var_exp": explained_variance_score(X_test, X_test_pred),
    }
