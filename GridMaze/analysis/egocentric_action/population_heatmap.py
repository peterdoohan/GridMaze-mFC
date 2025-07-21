"""
Library for visualising population tuning aligned to egocentric actions
"""

# %% Imports
from ast import Not
from itertools import groupby
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


def plot_egocentric_action_tuning_heatmap(
    tuning_df,
    metrics_df,
    min_pref_action_factor=2,
    min_pref_action_frac=0.65,
    normalise="zscore",
    smooth_SD=12,
    order_by="pref_max",
    crop_window=False,
    cmap="coolwarm",
    v_range=(-1.5, 2.5),
    axes=None,
):
    """ """
    if axes is None:
        f, axes = plt.subplots(3, 3, figsize=(4, 4), height_ratios=[0.8, 0.3, 1])
    f.subplots_adjust(wspace=0.05, hspace=0.05)
    for ax in axes.flatten():
        ax.axis("off")

    for i, ego_action in enumerate(["turn_left", "go_forward", "turn_right"]):
        heatmap_df = _get_heatmap_df(
            tuning_df,
            metrics_df,
            ego_action,
            min_pref_action_factor,
            min_pref_action_frac,
            normalise,
            smooth_SD,
            order_by,
            crop_window,
        )
        for j, _ego_action in enumerate(["turn_left", "go_forward", "turn_right"]):
            ax = axes[i, j]
            T = heatmap_df[_ego_action].values
            sns.heatmap(
                T,
                ax=ax,
                cmap=cmap,
                cbar=False,
                vmin=v_range[0],
                vmax=v_range[1],
            )


def _get_heatmap_df(
    tuning_df,
    metrics_df,
    pref_action="turn_left",
    min_pref_action_factor=2,
    min_pref_action_frac=0.65,
    normalise="zscore",
    smooth_SD=12,
    order_by="CV_pref_max",
    crop_window=False,
):
    """ """
    # filter clusters based on input metric thresholds
    metrics = metrics_df[metrics_df.pref_action.all_action.name == pref_action]
    if min_pref_action_factor is not None:
        metrics = metrics[metrics.pref_action.all_action.factor.gt(min_pref_action_factor)]
    if min_pref_action_frac is not None:
        metrics = metrics[metrics.pref_action.all_action.frac.gt(min_pref_action_frac)]
    keep_clusters = metrics.index.values
    tuning = tuning_df.iloc[tuning_df.index.get_level_values(0).isin(keep_clusters)]
    # normalise tuning curves
    if normalise == "zscore":
        tcs = zscore(tuning.values, axis=1)
        tuning = pd.DataFrame(tcs, index=tuning.index, columns=tuning.columns)
    else:
        raise NotImplementedError
    # smooth tuning curves
    if smooth_SD:
        tcs = gaussian_filter1d(tuning.values, smooth_SD, axis=1)
        tuning = pd.DataFrame(tcs, index=tuning.index, columns=tuning.columns)
    # reshape to wide format for final heatmap
    wide_df = (
        tuning.unstack(level=1).swaplevel(1, 2, axis=1).sort_index(axis=1).action_aligned_rates
    )  # n_neurons, n_actions x n_timepoints
    wide_df = wide_df[["turn_left", "go_forward", "turn_right"]]  # reorder
    # order clusters by CV t_max (precomputed in metrics_df)
    if order_by == "CV_pref_max":
        wide_df[("t_max", "")] = wide_df.index.map(metrics.pref_action.all_action.t_max.to_dict()).values
    elif order_by == "pref_max":
        wide_df[("t_max", "")] = wide_df[pref_action].idxmax(axis=1).values.astype(float)
    wide_df.sort_values(by=("t_max", ""), inplace=True)
    wide_df.drop(columns=("t_max", ""), inplace=True)
    if crop_window:
        timepoints = wide_df.columns.get_level_values(1).astype(float)
        crop_mask = (timepoints >= crop_window[0]) & (timepoints <= crop_window[1])
        wide_df = wide_df.loc[:, crop_mask]
    return wide_df  # n_neurons, n_actions x n_timepoints


# %%


def get_population_egocentric_action_tuning(
    subject_IDs="all",
    maze_names="all",
    late_sessions=False,
    sessions=None,
    actions=["turn_left", "turn_right", "go_forward"],
    include_action_type=False,
    window=(-3, 3),
    min_split_half_corr=0.4,
    max_jobs=10,
    with_metrics=True,
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

    def _process_session(session, actions, include_action_type, min_split_half_corr, window, verbose):
        # get tuning curves
        tuning_df = get_session_egocentric_action_tuning(
            session, actions, include_action_type, min_split_half_corr, window, verbose
        )
        if tuning_df is None:
            return None, None
        # get associated metrics (action pref, spit half corr, etc.)
        metrics_df = session.cluster_egocentric_action_tuning_metrics
        metrics_df = metrics_df[metrics_df.split_half_corr.all_action.value.gt(min_split_half_corr)]
        metrics_df.set_index("cluster_unique_ID", inplace=True)
        return tuning_df, metrics_df

    dfs = Parallel(n_jobs=max_jobs)(
        delayed(_process_session)(
            session,
            actions,
            include_action_type,
            min_split_half_corr,
            window,
            verbose,
        )
        for session in sessions
    )
    tuning_df = pd.concat([x[0] for x in dfs if x[0] is not None], axis=0)
    metrics_df = pd.concat([x[1] for x in dfs if x[1] is not None], axis=0)
    if with_metrics:
        return tuning_df, metrics_df
    else:
        return tuning_df


def get_session_egocentric_action_tuning(
    session,
    actions=["turn_left", "turn_right", "go_forward"],
    include_action_type=False,
    min_split_half_corr=0.4,
    window=(-3, 3),
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
        egocentric_metrics_df.split_half_corr.all_action.value.gt(min_split_half_corr)
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
    # add free/forced label and add to axis if desired
    if include_action_type:
        action_aligned_rates = action_aligned_rates.assign(
            action_type=action_aligned_rates.choice_degree.gt(2).map({True: "free", False: "forced"})
        )
        groupby_cols = ["cluster_unique_ID", "basic_action", "action_type"]
    else:
        groupby_cols = ["cluster_unique_ID", "basic_action"]
    # get tuning curves for each action
    tuning_curves = action_aligned_rates.groupby(
        groupby_cols
    ).action_aligned_rates.mean()  # [clusters x actions, timepoints]
    return tuning_curves
