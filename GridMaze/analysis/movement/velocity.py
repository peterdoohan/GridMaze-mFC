"""
Come up with some way of visualising velocity tuning across the population
@ peterdoohan
"""

# %% Imports
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from GridMaze.analysis.cluster_tuning import movement as mv
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc

# %% Global Variables

FRAME_RATE = 60

# %% Functions


def plot_velocity_population_symmetry_heatmap(
    pop_df,
    n_clusters=8,
    min_ordering_power=30,
    ax=None,
    vmax=100,
    cmap="viridis",
):
    """ """
    df = pop_df.copy()
    harmonics = df.columns
    df.columns = pd.MultiIndex.from_product([["harmonic"], harmonics])
    # cluster neurons
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(df.values)
    df[("KMeans_cluster", "")] = kmeans.labels_
    cluster_means = df.groupby("KMeans_cluster").mean().harmonic
    # split into velocity and speed clusters based on min_ordering_power
    cluster_max = cluster_means.max(axis=1)
    speed_clusters = cluster_max[cluster_max.lt(min_ordering_power)].index
    velocity_clusters = cluster_max[cluster_max.ge(min_ordering_power)].index
    velocity_sorted_clusters = cluster_means.loc[velocity_clusters].sort_values(
        by=list(harmonics), ascending=[False, False, False, False]
    )
    cluster_order = list(velocity_sorted_clusters.index) + list(speed_clusters)
    cluster2new_order = {c: i for i, c in enumerate(cluster_order)}
    cluster2max_harm = cluster_means.loc[cluster_order].reset_index(drop=True).idxmax(axis=1).to_dict()
    # rename clusters by ordering
    df[("KMeans_cluster", "")] = df[("KMeans_cluster", "")].map(cluster2new_order)
    # further order neurons within each cluster
    _dfs = []
    for i in range(n_clusters):
        cluster_df = df[df[("KMeans_cluster", "")] == i].harmonic
        if i in velocity_clusters:
            max_harm = cluster2max_harm[i]
            cluster_df = cluster_df.sort_values(by=[max_harm], ascending=[False])
        _dfs.append(cluster_df)
    ordered_df = pd.concat(_dfs, axis=0)
    # plotting
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2, 6))
    sns.heatmap(
        ordered_df.astype(float),
        cmap=cmap,
        cbar_kws={"label": "Power", "shrink": 0.4},
        yticklabels=False,
        ax=ax,
        rasterized=True,
        vmax=vmax,
    )
    y_tick = (df.shape[0] // 100) * 100
    ax.set_yticks([y_tick])
    ax.set_yticklabels([y_tick])
    ax.set_ylabel("neurons (ordered by rotational symmetry)")
    ax.set_xlabel("harmonic")
    ax.set_xticklabels(harmonics + 1)
    return


def get_population_velocity_summary(late_session=False, verbose=True, sessions=None):
    """ """
    if sessions is None:
        if verbose:
            print("Loading sessions...")
        days_on_maze = "late" if late_session else "all"
        sessions = gs.get_maze_sessions(
            subject_IDs="all",
            days_on_maze=days_on_maze,
            maze_names="all",
            with_data=["cluster_movement_metrics", "navigation_df", "navigation_spike_rates_df"],
        )
    dfs = []
    for session in sessions:
        if verbose:
            print(session.name)
        harm_df = get_session_velocity_rotational_harmonics(session)
        if harm_df is np.nan:
            continue
        dfs.append(harm_df)
    return pd.concat(dfs, axis=0)


# %% test


def get_session_velocity_rotational_harmonics(
    session,
    min_corr=0.75,
    x_range=(-0.3, 0.3),
    y_range=(-0.3, 0.3),
    bin_size=0.025,
    smooth_SD=False,
    min_occ=0.5,
    harm_range=(1, 4),
):
    # load data
    navigation_df = session.navigation_df
    navigation_rates_df = session.navigation_spike_rates_df.reset_index(drop=True)
    movement_metrics = session.cluster_movement_metrics

    # filter clusters
    if min_corr is not None:
        movement_metrics = movement_metrics[movement_metrics.velocity.mean_corr.gt(min_corr)]
    keep_clusters = movement_metrics.cluster_unique_ID.values

    # if no velocity tuned cluster from session return nan
    if len(keep_clusters) == 0:
        return np.nan

    navigation_rates_df = navigation_rates_df.firing_rate[keep_clusters]
    navigation_rates_df.columns = pd.MultiIndex.from_product([["firing_rate"], keep_clusters])
    navigation_rates_df = pd.concat([navigation_df, navigation_rates_df], axis=1)

    # get smoothed velocity data and update navigation df
    speeds, velocities, trang_acc = mv.get_movement_tuning_data(navigation_df)
    navigation_rates_df[("velocity", "x")] = velocities[:, 0]
    navigation_rates_df[("velocity", "y")] = velocities[:, 1]

    # bin velocity data
    x_bin_edges = np.arange(x_range[0], x_range[1] + bin_size, bin_size)
    y_bin_edges = np.arange(y_range[0], y_range[1] + bin_size, bin_size)
    navigation_rates_df[("velocity", "x_bin")] = pd.cut(
        navigation_rates_df[("velocity", "x")], bins=x_bin_edges, labels=(x_bin_edges[:-1] + bin_size / 2)
    )
    navigation_rates_df[("velocity", "y_bin")] = pd.cut(
        navigation_rates_df[("velocity", "y")], bins=y_bin_edges, labels=(y_bin_edges[:-1] + bin_size / 2)
    )

    # get tuning curves
    grouped_df = navigation_rates_df.groupby([("velocity", "x_bin"), ("velocity", "y_bin")], observed=True)
    tuning_df = grouped_df.firing_rate.mean().firing_rate  # (n_x bins x n_y_bins), n_clusters
    tuning_occ = grouped_df.time.count().unstack(level=0)
    sub_min_occ = tuning_occ.lt(min_occ * FRAME_RATE)

    # get harmonic for each velocity heatmap
    harm_df = pd.DataFrame(index=tuning_df.columns, columns=np.arange(harm_range[1]))
    for c in tuning_df.columns:
        tuning_heatmap = tuning_df[c].unstack(level=0)
        # low occ aware smoothing
        if smooth_SD:
            # Convert to arrays
            mean_arr = tuning_heatmap.to_numpy(dtype=float)
            occ_arr = tuning_occ.to_numpy(dtype=float)
            # Numerator: sum of rates per bin = mean * occ
            num_arr = np.where(np.isfinite(mean_arr), mean_arr * occ_arr, 0.0)
            # Smooth numerator and occupancy with the same kernel
            num_s = gaussian_filter(num_arr, sigma=smooth_SD, mode="constant", cval=0.0)
            occ_s = gaussian_filter(occ_arr, sigma=smooth_SD, mode="constant", cval=0.0)
            # Safe division; where occ_s ~ 0 keep NaN
            with np.errstate(invalid="ignore", divide="ignore"):
                smoothed_mean = np.where(occ_s > 0, num_s / occ_s, np.nan)
            # Put back into DataFrame with original indexing
            tuning_heatmap = pd.DataFrame(smoothed_mean, index=tuning_heatmap.index, columns=tuning_heatmap.columns)
        # mask low occupancy bins
        tuning_heatmap = tuning_heatmap.mask(sub_min_occ)
        # get rotational symmetry
        rot_corrs, angles = mv.get_rotational_autocorr(tuning_heatmap)
        # fourier decomposition
        power = mv.rotational_spectrum(rot_corrs)
        harm_df.loc[c] = power[1 : harm_range[1] + 1]
    return harm_df


# %%


def test_plots(cluster_unique_IDs, plot_range=(0, 100)):
    if plot_range is None:
        plot_clusters = cluster_unique_IDs
    else:
        plot_clusters = cluster_unique_IDs[plot_range[0] : plot_range[1]]
    for cluster in plot_clusters:
        fig = plt.figure(figsize=(5, 2.5), clear=True)
        gsc = GridSpec(2, 2, figure=fig, width_ratios=[2, 1], wspace=0.5, hspace=0.8)
        ax1 = fig.add_subplot(gsc[0:2, 0])  # v heatmap
        ax2 = fig.add_subplot(gsc[0, 1])  # rot corr
        ax3 = fig.add_subplot(gsc[1, 1])  # harmonics
        Cluster = gc.get_cluster(cluster_unique_ID=cluster)
        Cluster.plot_tuning(feature="velocity", feature_kwargs={"with_symmetry": True}, ax=(ax1, ax2, ax3))
        fig.suptitle(cluster)
