"""
Library to look at low dimensional structure in route aligned rates across the population
"""
#%% Imports
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from ..core import get_clusters as gc
from ..core import get_sessions as gs

#%% Global Variables


#%% Functions 
def get_subject_route_aligned_rates(subject, smooth_SD=2, n_clusters=10, plot=True):
    """ """
    subject = [subject] if not subject == "all" else subject
    sessions = gs.get_maze_sessions(subject_IDs=subject, maze_names="all", days_on_maze="late", with_data=["route_aligned_rates_df", "cluster_metrics"],
                                    must_have_data=True)
    session_aligned_rates = [get_session_route_aligned_rates(session) for session in sessions]
    aligned_rates_df = pd.concat(session_aligned_rates, axis=0)
    if smooth_SD:
        smoothed_rates = gaussian_filter1d(aligned_rates_df.values, sigma=smooth_SD, axis=1)
        aligned_rates_df = pd.DataFrame(smoothed_rates, index=aligned_rates_df.index, columns=aligned_rates_df.columns)
    # normalise data before KMeans
    aligned_rates_df = aligned_rates_df.div(aligned_rates_df.max(axis=1), axis=0)
    # do KMeans clustering on the data
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(aligned_rates_df.values)
    aligned_rates_df[("KMeans_cluster", "")] = kmeans.labels_
    aligned_rates_df = aligned_rates_df.sort_values(("KMeans_cluster", ""))
    if plot:
        # heatmap 
        fig = plt.figure(figsize=(10, 10))
        gsc = GridSpec(n_clusters,2, width_ratios=[1,2], figure=fig)
        h_ax = fig.add_subplot(gsc[:, 0])
        c_axes = [fig.add_subplot(gsc[i, 1]) for i in range(n_clusters)]
        n_timepoints = aligned_rates_df.shape[1]-1
        sns.heatmap(aligned_rates_df.drop(("KMeans_cluster", ""), axis=1).values, ax=h_ax, cmap="Blues", 
                    cbar=False, xticklabels=False, yticklabels=False)
        h_ax.set_ylabel("Units")
        h_ax.set_xlabel("Route Aligned Time")
        h_ax.axvline(n_timepoints//2, color="black", linestyle="--", lw=1, alpha=0.8)
        h_ax.set_xticks([n_timepoints/4, 3*n_timepoints/4])
        h_ax.set_xticklabels(["route_1", "route_0"])
        # cluster means
        cluster_means = aligned_rates_df.groupby(("KMeans_cluster", "")).mean()
        colors = sns.color_palette("bright", n_clusters)
        for i, cluster_mean in cluster_means.iterrows():
            c_axes[i].plot(cluster_mean.values, color=colors[i], lw=2)        
            c_axes[i].set_xticklabels([])
            c_axes[i].set_yticklabels([])
            c_axes[i].axvline(n_timepoints//2, color="black", linestyle="--", lw=1, alpha=0.8)
            c_axes[i].set_xticks([n_timepoints/4, 3*n_timepoints/4])
            c_axes[i].spines["top"].set_visible(False)
            c_axes[i].spines["right"].set_visible(False)
            if i == n_clusters-1:
                c_axes[i].set_xticklabels(["ordered_route_1", "ordered_route_0"])
                c_axes[i].set_xlabel("Route Aligned Time")
                c_axes[i].set_ylabel("Norm Rate")
    return aligned_rates_df



def get_session_route_aligned_rates(session,
                                    remove_cue_events=False,
                                    optimal_only=True,
                                    max_routes=2,
                                    min_routes=2,
                                    stretch_max=5,
                                    stretch_min=0):
    """ """
    # run analysis for only single units
    route_aligned_rates_df = session.route_aligned_rates_df
    keep_clusters = gc.filter_clusters(session.cluster_metrics, 
                                        session.session_info, 
                                        return_unique_IDs=True)
    route_aligned_rates_df = route_aligned_rates_df[route_aligned_rates_df.cluster_unique_ID.isin(keep_clusters)]
    # filter data at trial level
    filter_masks = []
    if max_routes:
        filter_masks.append(route_aligned_rates_df.n_routes.le(max_routes))
    if min_routes:
        filter_masks.append(route_aligned_rates_df.n_routes.ge(min_routes))
    if stretch_max:
        filter_masks.append(route_aligned_rates_df.trial_stretch["max"].lt(stretch_max))
    if stretch_min:
        filter_masks.append(route_aligned_rates_df.trial_stretch["min"].gt(stretch_min))
    route_aligned_rates_df = route_aligned_rates_df[np.logical_and.reduce(filter_masks)]
    route_aligned_rates_df = route_aligned_rates_df.set_index("cluster_unique_ID")
    column_fields = route_aligned_rates_df.columns.get_level_values(0).unique().to_numpy()
    ordered_routes = column_fields[["route_order" in c for c in column_fields]]
    ordered_routes = ordered_routes[-max_routes:]
    ordered_route_rates = []
    for ordered_route in ordered_routes:
        df = route_aligned_rates_df[ordered_route]
        # filter within route data
        masks = []
        if remove_cue_events:
            masks.append(df.latent["l-1"] != "cue")
        if optimal_only:
            masks.append(df.latent.optimal_route == 1)
        df = df[np.logical_and.reduce(masks)]
        # groupby cluster and get mean firing rate
        mean_rates = df.groupby("cluster_unique_ID").firing_rate.mean()
        mean_rates.columns = pd.MultiIndex.from_product([[ordered_route], mean_rates.columns.get_level_values(1)])
        ordered_route_rates.append(mean_rates)
    return pd.concat(ordered_route_rates, axis=1)
        


def plot_route_aligned_tuning(
    route_aligned_rates_df,
    remove_cue_events=False,
    optimal_only=True,
    max_routes=2,
    min_routes=2,
    smooth_SD=1,
    stretch_max=5,
    stretch_min=0,
    ax=None,
):
    """
    Input is df with warped firing rates aligned to routes before reward for a single cluster
    """
    # filter trials based on n_routes
    if max_routes or min_routes:
        route_aligned_rates_df = route_aligned_rates_df[route_aligned_rates_df.n_routes.le(max_routes)]
        route_aligned_rates_df = route_aligned_rates_df[route_aligned_rates_df.n_routes.ge(min_routes)]
    # filter trials based on min max stretch
    valid_stetch_mask = route_aligned_rates_df.trial_stretch["max"].lt(
        stretch_max
    ) & route_aligned_rates_df.trial_stretch["min"].gt(stretch_min)
    route_aligned_rates_df = route_aligned_rates_df[valid_stetch_mask]
    # find detials from route_aligned_rates_df
    column_fields = route_aligned_rates_df.columns.get_level_values(0).unique().to_numpy()
    ordered_routes = column_fields[["route_order" in c for c in column_fields]]
    ordered_routes = ordered_routes[-max_routes:]
    n_ordered_routes = len(ordered_routes)
    # timepoints per route
    trp = len(route_aligned_rates_df.xs("firing_rate", level=1, axis=1).columns.get_level_values(1).unique())
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(7, 3))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Firing Rate (Hz)")
    ax.set_xlabel("Route Aligned Trial Progress")
    ax.set_xlim(0, n_ordered_routes * trp)
    for i in range(n_ordered_routes):
        ax.axvline(i * trp, color="black", linestyle="--", lw=0.5, alpha=0.3)
    ax.set_xticks([i * trp + trp / 2 for i in range(n_ordered_routes)])
    ax.set_xticklabels(ordered_routes)
    # filter df for each ordered route
    combined_rates = []
    combined_sem = []
    for ordered_route in ordered_routes:
        df = route_aligned_rates_df[ordered_route]
        if remove_cue_events:
            df = df[df.latent["l-1"] != "cue"]
        if optimal_only:
            df = df[df.latent.optimal_route == 1]
        rates = df.firing_rate
        combined_rates.append(rates.mean().values)
        combined_sem.append(rates.sem().values)
    combined_rates = np.hstack(combined_rates)
    combined_sem = np.hstack(combined_sem)
    if smooth_SD:
        combined_rates = gaussian_filter1d(combined_rates, sigma=smooth_SD)
        combined_sem = gaussian_filter1d(combined_sem, sigma=smooth_SD)
    ax.plot(combined_rates, color="black")
    ax.fill_between(
        np.arange(len(combined_rates)),
        combined_rates - combined_sem,
        combined_rates + combined_sem,
        color="black",
        alpha=0.3,
    )
    return