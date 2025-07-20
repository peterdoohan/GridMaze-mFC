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
    timepoints = tuning_df.columns.values.astype(float)
    n_timepoints = len(timepoints)
    actions = ["turn_left", "go_forward", "turn_right"]
    wide_df = tuning_df.unstack().swaplevel(0, 1, axis=1).sort_index(axis=1)  # neurons, actions x timepoints
    wide_df = wide_df.reindex(columns=actions, level=0)  # top level order: left, forward, right
    # further filter neurons for diff between left and right tuning
    if min_action_diff:
        max_diff = _get_max_action_diff(tuning_df)
        wide_df = wide_df[max_diff >= min_action_diff]  # keep neurons with sufficient left-right diff
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(wide_df.values)
    centroids = kmeans.cluster_centers_
    # determine preferred action for each centroid for plotting
    centroid2pref_action = {}
    centroid2argmax = {}
    for i, ct in enumerate(centroids):
        centroid_tuning = ct.reshape(len(actions), n_timepoints)
        pref_action_ind = np.argmax(np.max(centroid_tuning, axis=1))
        pref_action_argmax = np.argmax(centroid_tuning[pref_action_ind])
        centroid2argmax[i] = pref_action_argmax
        centroid2pref_action[i] = actions[pref_action_ind]
    # order cluster by their prefered action and their max tuning (time)
    left_prefering = [c for c, a in centroid2pref_action.items() if a == "turn_left"]
    left_order = sorted(left_prefering, key=lambda x: centroid2argmax[x])
    right_prefering = [c for c, a in centroid2pref_action.items() if a == "turn_right"]
    right_order = sorted(right_prefering, key=lambda x: centroid2argmax[x])
    print(left_order, right_order)
    # plotting
    if axes is None:
        f, axes = plt.subplots(n_clusters // 2, 2, figsize=(3, 1.5 * (n_clusters // 2)), sharey=True, sharex=True)
    # plot left tuning clusters
    for i, clust_order in enumerate([left_order, right_order]):
        for j, ax in enumerate(axes[:, i]):
            ax.spines[["top", "right"]].set_visible(False)
            cluster_id = clust_order[j]
            tuning = centroids[cluster_id].reshape(len(actions), n_timepoints)
            ax.plot(timepoints, tuning[0], label="turn_left", color="darkorchid", lw=2)
            ax.plot(timepoints, tuning[1], label="go_forward", color="grey", lw=2)
            ax.plot(timepoints, tuning[2], label="turn_right", color="steelblue", lw=2)
            if i == 0 and j == 2:
                ax.set_ylabel("Activity (z-scored)")
                ax.set_xlabel("Time (s)")
                ax.legend()
            ax.axvline(0, color="k", linestyle="--", alpha=0.5)


def plot_egocentric_action_tuning_heatmap(
    tuning_df, cluster_method="KMeans", n_clusters=6, min_action_diff=3, axes=None
):
    # process data
    timepoints = tuning_df.columns.values.astype(float)
    n_timepoints = len(timepoints)
    actions = ["turn_left", "go_forward", "turn_right"]
    wide_df = tuning_df.unstack().swaplevel(0, 1, axis=1).sort_index(axis=1)  # neurons, actions x timepoints
    wide_df = wide_df.reindex(columns=actions, level=0)  # top level order: left, forward, right
    # further filter neurons for diff between left and right tuning
    if min_action_diff:
        max_diff = _get_max_action_diff(tuning_df)
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
        f, axes = plt.subplots(1, len(actions), figsize=(3 * len(actions), 6), sharey=True, width_ratios=[1, 1, 1.2])
    for ax, action in zip(axes, actions):
        action_tuning = wide_df[action]
        cbar = True if action == "turn_right" else False
        sns.heatmap(data=action_tuning, ax=ax, cmap="bwr", cbar=cbar, vmin=-1.5, vmax=3)
        y_tick = round(len(wide_df), -2)
        ax.set_yticks([y_tick])
        ax.set_yticklabels([f"{y_tick}"], rotation=90)
        ax.set_xlabel("Time (s)")
        ax.set_xticks(np.linspace(0, n_timepoints, 7))
        ax.set_xticklabels(np.arange(min(timepoints), max(timepoints) + 1, 1), rotation=0)
        ax.axvline(n_timepoints // 2, color="k", linestyle="--", alpha=0.5)
        ax.set_title(action)
        if action == "turn_left":
            ax.set_ylabel("Neurons", labelpad=-10)
        else:
            ax.set_ylabel("")


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
    late_sessions=False,
    sessions=None,
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
    if sessions is None:
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
    return tuning_curves
