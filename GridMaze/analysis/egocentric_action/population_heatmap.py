"""
Library for visualising population tuning aligned to egocentric actions
"""

# %% Imports
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans


from GridMaze.analysis.cluster_tuning import actions as act
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc

# %% Global Variables

# %% Functions


def plot_KMeans_cluster_centroids(tuning_df, n_clusters=6, min_action_diff=3, axes=None):
    """ """
    # process data
    n_timepoints = tuning_df.shape[1]
    actions = ["turn_left", "go_forward", "turn_right"]
    wide_df = tuning_df.unstack().swaplevel(0, 1, axis=1).sort_index(axis=1)  # neurons, actions x timepoints
    wide_df = wide_df.reindex(columns=actions, level=0)  # top level order: left, forward, right
    # further filter neurons for diff between left and right tuning
    if min_action_diff:
        max_diff = _get_max_action_diff(wide_df)
        wide_df = wide_df[max_diff >= min_action_diff]  # keep neurons with sufficient left-right diff
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    centroids = kmeans.cluster_centers_
    
    return


def plot_egocentric_action_tuning_heatmap(
    tuning_df, cluster_method="KMeans", n_clusters=6, min_action_diff=3, axes=None
):
    # process data
    n_timepoints = tuning_df.shape[1]
    actions = ["turn_left", "go_forward", "turn_right"]
    wide_df = tuning_df.unstack().swaplevel(0, 1, axis=1).sort_index(axis=1)  # neurons, actions x timepoints
    wide_df = wide_df.reindex(columns=actions, level=0)  # top level order: left, forward, right
    # further filter neurons for diff between left and right tuning
    if min_action_diff:
        max_diff = _get_max_action_diff(wide_df)
        wide_df = wide_df[max_diff >= min_action_diff]  # keep neurons with sufficient left-right diff
    # group neurons by KMeans cluster and plot in clustees together
    if cluster_method == "KMeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(wide_df.values)  # fit and predict clusters
        centroids = kmeans.cluster_centers_
    else:
        raise ValueError(f"Clustering method {cluster_method} not recognised")
    wide_df[("KMeans_cluster", "id")] = labels
    # order clusters by their perfered action
    centroid2pref_action = {}
    centroid2argmax = {}
    for i, ct in enumerate(centroids):
        centroid_tuning = ct.reshape(len(actions), n_timepoints)
        pref_action_ind = np.argmax(np.max(centroid_tuning, axis=1))
        pref_action_argmax = np.argmax(centroid_tuning[pref_action_ind])
        centroid2argmax[i] = pref_action_argmax
        centroid2pref_action[i] = pref_action_ind
    wide_df[("KMeans_cluster", "prefered_action")] = wide_df[("KMeans_cluster", "id")].map(centroid2pref_action)
    wide_df[("KMeans_cluster", "argmax")] = wide_df[("KMeans_cluster", "id")].map(centroid2argmax)
    # order cluster by av cluster argmax
    wide_df.sort_values(by=[("KMeans_cluster", "prefered_action"), ("KMeans_cluster", "argmax")], inplace=True)
    # plotting
    if axes is None:
        f, axes = plt.subplots(1, len(actions), figsize=(3 * len(actions), 6), sharey=True)
    for ax, action in zip(axes, actions):
        action_tuning = wide_df[action]
        sns.heatmap(data=action_tuning, ax=ax, cmap="bwr", cbar=False, vmin=-1.5, vmax=3)

    return centroid2pref_action


def _get_max_action_diff(tuning_df):
    left_right_diff = tuning_df.xs("turn_left", axis=0, level=1) - tuning_df.xs("turn_right", axis=0, level=1)
    left_forward_diff = tuning_df.xs("turn_left", axis=0, level=1) - tuning_df.xs("go_forward", axis=0, level=1)
    right_forward_diff = tuning_df.xs("turn_right", axis=0, level=1) - tuning_df.xs("go_forward", axis=0, level=1)
    max_diff = np.max(
        [
            left_right_diff.abs().max(axis=1),
            left_forward_diff.abs().max(axis=1),
            right_forward_diff.abs().max(axis=1),
        ],
        axis=0,
    )
    return max_diff


# %%


def get_population_egocentric_action_tuning(
    subject_IDs="all",
    maze_names="all",
    late_sessions=True,
    actions=["turn_left", "turn_right", "go_forward"],
    action_type="all",
    window=(-3, 3),
    normalisation="zscore",
    smooth_SD=8,
    min_split_half_corr=0.3,
    max_jobs=10,
    verbose=False,
):
    """ """
    days_on_maze = "late" if late_sessions else "all"
    if verbose:
        print("Loading sessions ...")
    sessions = gs.get_maze_sessions(
        subject_IDs=subject_IDs,
        maze_names=maze_names,
        days_on_maze=days_on_maze,
        with_data=[
            "navigation_df",
            "navigation_spike_rates_df",
            "cluster_metrics",
            "cluster_egocentric_action_tuning_metrics",
        ],
        must_have_data=True,
    )
    tc_dfs = Parallel(n_jobs=max_jobs)(
        delayed(get_session_egocentric_action_tuning)(
            session,
            actions=actions,
            action_type=action_type,
            min_split_half_corr=min_split_half_corr,
            window=window,
            normalisation=normalisation,
            smooth_SD=smooth_SD,
            verbose=verbose,
        )
        for session in sessions
    )
    # remove None dfs
    tc_dfs = [df for df in tc_dfs if df is not None]
    pop_ego_action_tuning = pd.concat(tc_dfs, axis=0)
    return pop_ego_action_tuning


def get_session_egocentric_action_tuning(
    session,
    actions=["turn_left", "turn_right", "go_forward"],
    action_type="all",
    min_split_half_corr=0.3,
    window=(-3, 3),
    normalisation="zscore",
    smooth_SD=5,
    verbose=False,
):
    """ """
    if verbose:
        print(session.name)
    # load data
    navigation_rates_df = session.get_navigation_activity_df(type="rates", cluster_kwargs={"single_units": True})
    egocentric_metrics_df = session.cluster_egocentric_action_tuning_metrics
    # keep only clusters with some egocentric aciton tuning
    keep_clusters = egocentric_metrics_df[
        egocentric_metrics_df.split_half_corr[f"{action_type}_action"].value.gt(min_split_half_corr)
    ].cluster_unique_ID.values
    if len(keep_clusters) == 0:
        if verbose:
            print(f"No clusters with egocentric action tuning for {session.name}")
        return None  # no clusters with egocentric action tuning
    all_clusters = navigation_rates_df.firing_rate.columns.to_numpy()
    reject_clusters = np.setdiff1d(all_clusters, keep_clusters)
    navigation_rates_df = navigation_rates_df.drop(columns=reject_clusters, level=1)
    # get action aligned rates
    action_aligned_rates = act._get_basic_action_tuning(navigation_rates_df, window=window)
    # filter for specified actions
    action_aligned_rates = action_aligned_rates[action_aligned_rates.basic_action.isin(actions)]
    # filter for specified action type
    if action_type == "all":
        pass
    elif action_type == "free":
        action_aligned_rates = action_aligned_rates[action_aligned_rates.choice_degree.gt(2)]
    elif action_type == "forced":
        action_aligned_rates = action_aligned_rates[action_aligned_rates.choice_degree.le(2)]
    else:
        raise ValueError(f"Action type {action_type} not recognised")
    # get tuning curves for each action
    tuning_curves = action_aligned_rates.groupby(
        ["cluster_unique_ID", "basic_action"]
    ).action_aligned_rates.mean()  # [clusters x actions, timepoints]
    # normalisation tuning curves for each cluster
    if normalisation == "zscore":
        # concat tuning curves [actions, clusters x timepoints]
        long_tuning_curves = tuning_curves.action_aligned_rates.unstack().swaplevel(0, 1, axis=1).sort_index(axis=1)
        # zscore each cluster
        long_tuning_curves = long_tuning_curves.apply(zscore, axis=1)
        # restack
        tuning_curves = long_tuning_curves.stack(level=[0], future_stack=True)
    else:
        raise ValueError(f"Normalisation {normalisation} not recognised")
    # smooth tuning curves if specified
    if smooth_SD:
        rates = tuning_curves.values
        smoothed_rates = gaussian_filter1d(rates, smooth_SD, axis=1)
        tuning_curves = pd.DataFrame(
            smoothed_rates,
            index=tuning_curves.index,
            columns=tuning_curves.columns,
        )
    return tuning_curves
