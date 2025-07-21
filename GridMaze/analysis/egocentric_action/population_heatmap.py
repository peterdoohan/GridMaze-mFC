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
from torch import normal


from GridMaze.analysis.cluster_tuning import actions as act
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc

# %% Global Variables

# %% Functions


def plot_heatmap_quantiles(
    tuning_df,
    metrics_df,
    pref_action="turn_left",
    min_pref_action_factor=2,
    min_pref_action_frac=0.65,
    normalise="zscore",
    smooth_SD=12,
    order_by="CV_pref_max",
    crop_window=(-1, 1),
    n_quantiles=4,
    axes=None,
):
    """ """
    if axes is None:
        f, axes = plt.subplots(2, 1, figsize=(3, 3), height_ratios=[1, 0.75], sharex=False)
    axes[0].spines[["top", "right", "bottom"]].set_visible(False)
    axes[0].set_xticks([])
    axes[1].spines[["top", "right"]].set_visible(False)
    for ax in axes:
        ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    # get heatmap
    heatmap_df = _get_heatmap_df(
        tuning_df,
        metrics_df,
        pref_action,
        min_pref_action_factor,
        min_pref_action_frac,
        normalise,
        smooth_SD,
        order_by,
        crop_window,
    )
    n_neurons = heatmap_df.shape[0]
    neuron_group_size = n_neurons // n_quantiles
    _n_groups = np.minimum(np.arange(n_neurons) // neuron_group_size, n_quantiles - 1)
    quantile_df = heatmap_df.groupby(_n_groups).mean()
    # plot
    # for action, cmap in zip(["turn_left", "go_forward", "turn_right"], ["Purples", "Greys", "Blues"]):
    if pref_action == "turn_left":
        action_order = ["turn_left", "turn_right"]
        cmaps = ["Purples", "Blues"]
    elif pref_action == "turn_right":
        action_order = ["turn_right", "turn_left"]
        cmaps = ["Blues", "Purples"]
    else:
        raise ValueError("pref_action must be either 'turn_left' or 'turn_right'")

    for action, cmap, ax in zip(action_order, cmaps, axes):
        aq_df = quantile_df[action]
        colors = sns.color_palette(cmap, n_colors=n_quantiles)
        for i in range(n_quantiles):
            q = aq_df.loc[i]
            x = q.index.astype(float).values
            y = q.values
            ax.plot(x, y, label=f"{action}: Q{i}", color=colors[i], lw=2)
    # ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1), fontsize=8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("firing rate (z-scored)")
    return heatmap_df


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
    # if normalise == "zscore":
    #     tcs = zscore(tuning.values, axis=1)
    #     tuning = pd.DataFrame(tcs, index=tuning.index, columns=tuning.columns)
    # else:
    #     raise NotImplementedError
    # smooth tuning curves
    if smooth_SD:
        tcs = gaussian_filter1d(tuning.values, smooth_SD, axis=1)
        tuning = pd.DataFrame(tcs, index=tuning.index, columns=tuning.columns)
    # reshape to wide format for final heatmap
    wide_df = (
        tuning.unstack(level=1).swaplevel(1, 2, axis=1).sort_index(axis=1).action_aligned_rates
    )  # n_neurons, n_actions x n_timepoints
    wide_df = wide_df[["turn_left", "go_forward", "turn_right"]]  # reorder
    if normalise == "zscore":
        tcs = zscore(wide_df.values, axis=1)
        wide_df = pd.DataFrame(tcs, index=wide_df.index, columns=wide_df.columns)
    else:
        raise NotImplementedError
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
