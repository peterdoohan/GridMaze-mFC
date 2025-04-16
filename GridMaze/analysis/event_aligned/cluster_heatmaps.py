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
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

from scipy.stats import zscore
from GridMaze.analysis.core import get_sessions as gs

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, ANALYSIS_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(ANALYSIS_INFO_PATH / "intra_trial_interval_times.json", "r") as input_file:
    INTRA_TRIAL_INTERVAL_TIMES = json.load(input_file)


# %%
def get_trial_aligned_activity_heatmap(
    sessions, smooth_SD=10, normalisation="zscore", cluster_method="KMeans", n_clusters=6, plot=False
):
    """
    New version of below but with cross validation of cluster ordering
    """
    # get cluster tuning to trial events (either from splits halfs or all trials)
    rates_1, rates_2, rates_full = [], [], []
    for session in sessions:
        aligned_rates_df = session.trial_aligned_rates_df
        # only include single units
        cluster_metrics = session.cluster_metrics
        single_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID.values
        aligned_rates_df = aligned_rates_df[aligned_rates_df.cluster_ID.isin(single_units)]
        # split trials randomly to make splits for cross validation
        trials = aligned_rates_df.trial.unique()
        split_half_trails = np.random.choice(trials, len(trials) // 2, replace=False)
        aligned_rates_1 = aligned_rates_df[aligned_rates_df.trial.isin(split_half_trails)]
        aligned_rates_2 = aligned_rates_df[~aligned_rates_df.trial.isin(split_half_trails)]

        def get_tuning(x):  # aveage of trials to get tuning to trial events
            return (
                x.groupby("cluster_unique_ID").firing_rate.mean().xs("firing_rate", axis=1, level=0, drop_level=False)
            )

        rates_1.append(get_tuning(aligned_rates_1))
        rates_2.append(get_tuning(aligned_rates_2))
        rates_full.append(get_tuning(aligned_rates_df))

    rates_1 = pd.concat(rates_1, axis=0)
    rates_2 = pd.concat(rates_2, axis=0)
    rates_full = pd.concat(rates_full, axis=0)

    # smooth rates if requested
    if smooth_SD:

        def _smooth_rates(x):
            y = x.copy()
            rates = gaussian_filter1d(y.firing_rate.to_numpy(), sigma=smooth_SD, axis=1)
            y.loc[:, ("firing_rate")] = rates
            return y

        rates_1 = _smooth_rates(rates_1)
        rates_2 = _smooth_rates(rates_2)
        rates_full = _smooth_rates(rates_full)

    # normalise rates by maximum firing rate
    if normalisation == "zscore":

        def _norm_rates(x):
            y = x.copy()
            rates = zscore(y.firing_rate.to_numpy(), axis=1)
            y.loc[:, ("firing_rate")] = rates
            return y

        rates_1 = _norm_rates(rates_1)
        rates_2 = _norm_rates(rates_2)
        rates_full = _norm_rates(rates_full)
    else:
        raise NotImplementedError

    # group tuning curves into clusters for plotting
    tuning_1, tuning_2 = rates_1.firing_rate.to_numpy(), rates_2.firing_rate.to_numpy()

    if cluster_method == "KMeans":
        # use split halfs to cross validate cluster ordering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(tuning_1)
        labels = kmeans.predict(tuning_2)
        centroids = kmeans.cluster_centers_

    elif cluster_method == "Agglomerative":
        # Cluster tuning_1 using Agglomerative Clustering
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        labels_1 = agg.fit_predict(tuning_1)

        # Compute centroids for each cluster in tuning_1
        centroids = np.array([tuning_1[labels_1 == i].mean(axis=0) for i in range(n_clusters)])

        # For tuning_2, assign each sample to the nearest centroid
        dists = cdist(tuning_2, centroids)
        labels = dists.argmin(axis=1)

    else:
        raise NotImplementedError

    # use xvaled cluster labels to order full tuning curves in heatmap
    rates_full["cluster"] = labels
    # Order cluster by av cluster argmax
    peak_times = np.argmax(centroids, axis=1)
    sorted_order = peak_times.argsort()
    cluster_mapping = {original: new for new, original in enumerate(sorted_order)}
    rates_full["cluster"] = rates_full["cluster"].map(cluster_mapping)
    rates_full.sort_values(by="cluster", inplace=True)
    if plot:
        f, ax = plt.subplots(1, 1, figsize=(5, 5), clear=True)
        plot_trial_aligned_heatmap(rates_full.firing_rate, normalisation, ax)
    return rates_full


# %%
def get_sessions_for_analysis(aligned_to="trial"):
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names="all",
        days_on_maze="late",
        with_data=[f"{aligned_to}_aligned_rates_df", "cluster_metrics"],
    )
    return sessions


def plot_trial_aligned_heatmap(norm_aligned_rates_df, normalisation_method, ax):
    abs_max = 5  # max(norm_aligned_rates_df.abs().max().max(), 0)
    cmap = "viridis" if normalisation_method == "max" else "bwr"
    sns.heatmap(
        norm_aligned_rates_df,
        cmap=cmap,
        vmin=-abs_max,
        vmax=abs_max,
        ax=ax,
        linewidths=0,
        linecolor=None,
        rasterized=True,
        cbar_kws={"shrink": 0.5, "label": "Firing Rate (z-scored)"},
    )
    event_times = list(INTRA_TRIAL_INTERVAL_TIMES.values())[:-1]
    timepoints = [float(col) for col in norm_aligned_rates_df.columns]
    event_inds = [np.argmin(np.abs(np.array(timepoints) - time)) for time in event_times]
    for ind in event_inds:
        ax.axvline(ind, color="white", linewidth=1, ls="--")
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


# %% Event aligned heatmap versions


def get_event_aligned_rates(
    sessions,
    with_rates_for_ordering=True,
    smooth_SD=5,
    normalisation="zscore",
):
    """"""
    rates, rates_for_ordering = [], []
    for session in sessions:
        aligned_rates_df = session.event_aligned_rates_df
        # only include single units
        cluster_metrics = session.cluster_metrics
        single_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID.values
        aligned_rates_df = aligned_rates_df[aligned_rates_df.cluster_ID.isin(single_units)]
        # select random half of trials
        trials = aligned_rates_df.trial.unique()
        random_half_trails = np.random.choice(trials, len(trials) // 2, replace=False)
        _rates_for_ordering = aligned_rates_df[aligned_rates_df.trial.isin(random_half_trails)]
        # average reponses over trials
        rates.append(aligned_rates_df.groupby("cluster_unique_ID").firing_rate.mean().firing_rate)
        rates_for_ordering.append(_rates_for_ordering.groupby("cluster_unique_ID").firing_rate.mean().firing_rate)

    rates_df = pd.concat(rates, axis=0)
    rates_for_ordering_df = pd.concat(rates_for_ordering, axis=0)

    # smooth rates if requested
    if smooth_SD:

        def _smooth_rates(x):
            y = x.copy()
            rates = gaussian_filter1d(y.to_numpy(), sigma=smooth_SD, axis=1)
            y.loc[:, x.columns] = rates
            return y

        rates_df = _smooth_rates(rates_df)
        rates_for_ordering_df = _smooth_rates(rates_for_ordering_df)

    # normalise rates by maximum firing rate
    if normalisation == "zscore":

        def _norm_rates(x):
            y = x.copy()
            rates = zscore(y.to_numpy(), axis=1)
            y.loc[:, x.columns] = rates
            return y

        rates_df = _norm_rates(rates_df)
        rates_for_ordering_df = _norm_rates(rates_for_ordering_df)

    else:
        raise NotImplementedError

    # return results
    if with_rates_for_ordering:
        return rates_df, rates_for_ordering_df
    else:
        return rates_df


def plot_event_aligned_rates(
    aligned_rates_df,
    rates_for_ordering_df,
    xlims={"cue": (-5, 5), "reward": (-5, 5), "end_reward_consumption": (-5, 5)},
    vmin=-2.5,
    vmax=7.5,
    axes=None,
):
    """ """
    if axes is None:
        f, axes = plt.subplots(1, 3, figsize=(6, 5), clear=True, width_ratios=[1, 1, 1.15])
    for event, ax in zip(["cue", "reward", "end_reward_consumption"], axes):
        key = (
            f"{event}_aligned" if event != "end_reward_consumption" else f"{event}"
        )  # HACK need to fix label in analysis data df
        df = aligned_rates_df.xs(key, axis=1, level=0)
        ordering_df = rates_for_ordering_df.xs(key, axis=1, level=0)
        # filter response within window
        df = df.loc[:, (df.columns >= xlims[event][0]) & (df.columns <= xlims[event][1])]
        ordering_df = ordering_df.loc[
            :, (ordering_df.columns >= xlims[event][0]) & (ordering_df.columns <= xlims[event][1])
        ]
        # order by cluster argmax
        order = np.argmax(ordering_df.to_numpy(), axis=1).argsort()
        df = df.iloc[order]
        # plot heatmap
        cbar = True if event == "end_reward_consumption" else False
        sns.heatmap(
            df,
            cmap="coolwarm",
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
            cbar=cbar,
            cbar_kws={"shrink": 0.5, "label": "Firing Rate (z-scored)"},
            zorder=0,
        )
        times = df.columns.values.astype(float)
        zero_point = np.argmin(np.abs(times))
        ax.axvline(zero_point, color="white", linewidth=1, ls="--", zorder=3)
        ax.set_yticks([])
        tick_labels = [-4, -2, 0, 2, 4]
        tick_positions = [np.argmin(np.abs(times - tick)) for tick in tick_labels]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Single Units ({event} aligned ordering)")
