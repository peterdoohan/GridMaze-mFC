"""This model is for visualising and anlysing event/trial aligned activty during sessions"""

# %% Imports
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from sklearn.cluster import KMeans
import seaborn as sns

from .. import get_sessions as gs
from ..distance_to_goal.gpr_decoding import get_goal_stratified_Kfolds_df

# %% Global variables
with open("../data/experiment_info.json") as input_file:
    EXP_INFO = json.load(input_file)
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 12

# %% Functions


def get_session_for_trial_aligned_analysis():
    sessions = gs.get_sessions(
        subject_IDs="all", maze_number="all", day_on_maze="late", with_data=["trial_aligned_rates_df"]
    )
    return sessions


def get_trial_aligned_activity_heatmap_df(
    sessions,
    min_av_rate_threshold=0.25,
    smooth_SD=10,
    normalisation="zscore",
    n_clusters=6,
    argsort=False,
    plot=False,
):
    cluster_aligned_rates_dfs = []
    for session in sessions:
        aligned_rates_df = session.trial_aligned_rates_df
        aligned_rates_df = aligned_rates_df[aligned_rates_df.cluster_type == "good"]
        cluster_aligned_rates_df = (
            aligned_rates_df.set_index("cluster_unique_ID")
            .groupby("cluster_unique_ID")
            .mean()
            .xs("firing_rate", axis=1, level=0, drop_level=False)
        )
        if argsort:
            goal2trials_df = get_goal_stratified_Kfolds_df(aligned_rates_df, extend_trials=False, shuffle_trials=True)
            random_half_trials = (
                goal2trials_df.T[: goal2trials_df.shape[-1] // 2].to_numpy().flatten()
            )  # goal stratified
            ordering_rates_df = aligned_rates_df[aligned_rates_df.trial.isin(random_half_trials)]
            ordering_mean_rates = (
                ordering_rates_df.set_index("cluster_unique_ID").groupby("cluster_unique_ID").mean().firing_rate
            )
            cluster_aligned_rates_df[("arg_max", "")] = ordering_mean_rates.idxmax(axis=1)
            cluster_aligned_rates_df[("arg_median", "")] = (
                ordering_mean_rates.T.cumsum() > ordering_mean_rates.T.sum() / 2
            ).idxmax()
        if min_av_rate_threshold:
            sup_min_rate_mask = cluster_aligned_rates_df.firing_rate.mean(axis=1) > min_av_rate_threshold
            cluster_aligned_rates_df = cluster_aligned_rates_df[sup_min_rate_mask]
        cluster_aligned_rates_dfs.append(cluster_aligned_rates_df)
    exp_aligned_rates_df = pd.concat(cluster_aligned_rates_dfs, axis=0)
    if smooth_SD:
        rates = gaussian_filter1d(exp_aligned_rates_df.firing_rate.to_numpy(), sigma=smooth_SD, axis=1)
        exp_aligned_rates_df.loc[:, ("firing_rate")] = rates
    # normalise rates by maximum firing rate
    if normalisation:
        rates = exp_aligned_rates_df.firing_rate.to_numpy()
        if normalisation == "max":
            normalised_rates = rates / rates.max(axis=1)[:, None]
        elif normalisation == "zscore":
            normalised_rates = zscore(rates, axis=1)
        exp_aligned_rates_df.loc[:, ("firing_rate")] = normalised_rates
    # KMeans cluster the neurons
    if n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(exp_aligned_rates_df.firing_rate.to_numpy())
        exp_aligned_rates_df["Kmeans_cluster"] = kmeans.labels_
        # Order cluster by av cluster argmax
        centroids = kmeans.cluster_centers_
        peak_times = np.argmax(centroids, axis=1)
        sorted_order = peak_times.argsort()
        cluster_mapping = {original: new for new, original in enumerate(sorted_order)}
        exp_aligned_rates_df["Kmeans_cluster"] = exp_aligned_rates_df["Kmeans_cluster"].map(cluster_mapping)
        # order cells by cluster and arg max/median
        if argsort:
            if argsort == "arg_max":
                sort_cols = ["Kmeans_cluster", "arg_max"]
            elif argsort == "arg_median":
                sort_cols = ["Kmeans_cluster", "arg_median"]
        else:
            sort_cols = ["Kmeans_cluster"]
        exp_aligned_rates_df.sort_values(by=sort_cols, inplace=True)
    if argsort:
        exp_aligned_rates_df.drop(columns=["arg_max", "arg_median"], inplace=True)
    if plot:
        f, ax = plt.subplots(1, 1, figsize=(5, 5), clear=True)
        plot_trial_aligned_heatmap(exp_aligned_rates_df.firing_rate, normalisation, ax)
    return exp_aligned_rates_df


def plot_trial_aligned_heatmap(norm_aligned_rates_df, normalisation_method, ax):
    abs_max = 5  # max(norm_aligned_rates_df.abs().max().max(), 0)
    cmap = "viridis" if normalisation_method == "max" else "bwr"
    sns.heatmap(
        norm_aligned_rates_df,
        cmap=cmap,
        vmin=-abs_max,
        vmax=abs_max,
        ax=ax,
        cbar_kws={"shrink": 0.5, "label": "z-scored Firing Rate"},
    )
    event_times = EXP_INFO["intra_trial_interval_times"][:-1]
    timepoints = [float(col) for col in norm_aligned_rates_df.columns]
    event_inds = [np.argmin(np.abs(np.array(timepoints) - time)) for time in event_times]
    for ind in event_inds:
        ax.axvline(ind, color="silver", linewidth=1, alpha=0.5)
    ax.set_xticks(event_inds)
    ax.set_xticklabels(
        [
            f"Cue\n({event_times[0]:.1f})",
            f"Reward\n({event_times[1]:.1f})",
            f"ITI\n({event_times[2]:.1f})",
        ],
        rotation=0,
    )
    y_tick = round(len(norm_aligned_rates_df), -3)
    ax.set_yticks([y_tick])
    ax.set_yticklabels([f"{y_tick}"], rotation=90)
    ax.set_ylabel("Neurons", labelpad=-10)
    ax.set_xlabel("Time (s)")
    return


# %% Event aligned Heatmaps


def get_session_for_event_aligned_analysis():
    sessions = gs.get_sessions(
        subject_IDs="all", maze_number="all", day_on_maze="late", with_data=["event_aligned_rates_df"]
    )
    return sessions


def event_aligned_rates_heatmap_df(
    sessions, min_av_rate_threshold=0.25, smoothed=True, smooth_SD=10, normalised=True, normalisation_method="zscore"
):
    event_aligned_rates_dfs = []
    for session in sessions:
        aligned_rates_df = session.event_aligned_rates_df
        aligned_rates_df = aligned_rates_df[aligned_rates_df.cluster_type == "good"]
        # get cluster argmaxes
        goal2trials_df = get_goal_stratified_Kfolds_df(aligned_rates_df, extend_trials=False, shuffle_trials=True)
        random_half_trials = goal2trials_df.T[: goal2trials_df.shape[-1] // 2].to_numpy().flatten()  # goal stratified
        ordering_rates_df = aligned_rates_df[aligned_rates_df.trial.isin(random_half_trials)]
        event_arg_maxes = []
        event_arg_medians = []
        for event in ["cue_aligned", "reward_aligned"]:
            ordering_mean_rates = (
                ordering_rates_df.set_index("cluster_unique_ID").groupby("cluster_unique_ID").mean().firing_rate[event]
            )
            event_arg_maxes.append(ordering_mean_rates.idxmax(axis=1))
            event_arg_medians.append((ordering_mean_rates.T.cumsum() > ordering_mean_rates.T.sum() / 2).idxmax())
        cue_arg_maxes, reward_arg_maxes = event_arg_maxes
        cue_arg_medians, reward_arg_medians = event_arg_medians
        cluster_aligned_rates_df = (
            aligned_rates_df.set_index("cluster_unique_ID")
            .groupby("cluster_unique_ID")
            .mean()
            .xs("firing_rate", axis=1, level=0, drop_level=False)
        )
        cluster_aligned_rates_df[("arg_max", "cue", "")] = cue_arg_maxes
        cluster_aligned_rates_df[("arg_max", "reward", "")] = reward_arg_maxes
        cluster_aligned_rates_df[("arg_median", "cue", "")] = cue_arg_medians
        cluster_aligned_rates_df[("arg_median", "reward", "")] = reward_arg_medians
        if min_av_rate_threshold is not None:
            sup_min_rate_mask = cluster_aligned_rates_df.firing_rate.mean(axis=1) > min_av_rate_threshold
            cluster_aligned_rates_df = cluster_aligned_rates_df[sup_min_rate_mask]
        event_aligned_rates_dfs.append(cluster_aligned_rates_df)
        exp_aligned_rates_df = pd.concat(event_aligned_rates_dfs, axis=0).reset_index(drop=True)

    if smoothed:
        for event in ["cue_aligned", "reward_aligned"]:
            rates = gaussian_filter1d(exp_aligned_rates_df.firing_rate[event].to_numpy(), sigma=smooth_SD, axis=1)
            exp_aligned_rates_df.loc[:, ("firing_rate", event)] = rates
    if normalised:
        for event in ["cue_aligned", "reward_aligned"]:
            rates = exp_aligned_rates_df.firing_rate[event].to_numpy()
            if normalisation_method == "max":
                normalised_rates = rates / rates.max(axis=1)[:, None]
            elif normalisation_method == "zscore":
                normalised_rates = zscore(rates, axis=1)
            exp_aligned_rates_df.loc[:, ("firing_rate", event)] = normalised_rates
    return exp_aligned_rates_df


def plot_event_aligned_heatmap(event_aligned_rates_df, axes, order_by="arg_max", order_method="separate"):
    abs_max = 5  # max(event_aligned_rates_df.abs().max().max(), 0)
    cmap = "bwr"
    order_method = ["cue", "reward"] if order_method == "separate" else [order_method, order_method]
    for event, order, ax in zip(["cue_aligned", "reward_aligned"], order_method, axes):
        e = event.split("_")[0]
        event_aligned_rates_df.sort_values(by=[(order_by, order, "")], inplace=True)
        aligned_rates_df = event_aligned_rates_df.firing_rate[event]
        sns.heatmap(
            aligned_rates_df,
            cmap=cmap,
            vmin=-abs_max,
            vmax=abs_max,
            ax=ax,
            cbar_kws={"shrink": 0.5, "label": "z-scored Firing Rate"},
        )
        event_times = [-5, 0, 5]
        timepoints = [float(col) for col in aligned_rates_df.columns]
        event_inds = [np.argmin(np.abs(np.array(timepoints) - time)) for time in event_times]
        ax.axvline(event_inds[1], color="silver", linewidth=1, alpha=0.5)
        ax.set_xticks(event_inds)
        ax.set_xticklabels(["-5", f"{e.capitalize()}", "5"], rotation=0)
        ax.set_xlabel("Time (s)")
        ax.set_xlim([np.argmin(np.abs(np.array(timepoints) - time)) for time in [-10, 10]])
        if event == "cue_aligned":
            y_tick = round(len(aligned_rates_df), -3)
            ax.set_yticks([y_tick])
            ax.set_yticklabels([f"{y_tick}"], rotation=90)
            ax.set_ylabel("Neurons", labelpad=-10)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])
    return
