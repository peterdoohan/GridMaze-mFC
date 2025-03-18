"""
Library for plotting heatmaps of cluster trial event tuning
"""

# %%
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from scipy.stats import zscore
from GridMaze.analysis.core import get_sessions as gs

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, ANALYSIS_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(ANALYSIS_INFO_PATH / "intra_trial_interval_times.json", "r") as input_file:
    INTRA_TRIAL_INTERVAL_TIMES = json.load(input_file)

FRAME_RATE = 60


# %%
def get_sessions_for_analysis():
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names="all",
        days_on_maze="late",
        with_data=["trial_aligned_rates_df", "cluster_metrics"],
    )
    return sessions


def get_trial_aligned_activity_heatmap_df(
    sessions,
    smooth_SD=10,
    normalisation="zscore",
    n_clusters=6,
    argsort=False,
    plot=False,
):
    cluster_aligned_rates_dfs = []
    for session in sessions:
        aligned_rates_df = session.trial_aligned_rates_df
        # only include single units
        cluster_metrics = session.cluster_metrics
        single_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID.values
        aligned_rates_df = aligned_rates_df[aligned_rates_df.cluster_ID.isin(single_units)]
        cluster_aligned_rates_df = (
            aligned_rates_df.groupby("cluster_unique_ID")
            .firing_rate.mean()
            .xs("firing_rate", axis=1, level=0, drop_level=False)
        )
        if argsort:
            trials = aligned_rates_df.trial.unique()
            split_half_trails = np.random.choice(trials, len(trials) // 2, replace=False)
            ordering_rates_df = aligned_rates_df[aligned_rates_df.trial.isin(split_half_trails)]
            ordering_mean_rates = ordering_rates_df.groupby("cluster_unique_ID").firing_rate.mean().firing_rate
            cluster_aligned_rates_df[("arg_max", "")] = ordering_mean_rates.idxmax(axis=1)
            cluster_aligned_rates_df[("arg_median", "")] = (
                ordering_mean_rates.T.cumsum() > ordering_mean_rates.T.sum() / 2
            ).idxmax()
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
    event_times = list(INTRA_TRIAL_INTERVAL_TIMES.values())[:-1]
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
