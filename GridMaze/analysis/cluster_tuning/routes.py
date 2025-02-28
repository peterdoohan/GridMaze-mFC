"""
Library for plotting cluster tuning to latent routes used for navigation found by Xiao et al.
"""

# %% Imports
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from ..core import get_clusters as gc

# %% Global Variables
FRAME_RATE = 60


# %%
def plot_session_route_sequence_tuning(session):
    """ """
    navigation_df = session.navigation_df
    navigation_routes_df = session.navigation_routes_df
    navigation_rates_df = pd.concat([navigation_df, navigation_routes_df.reset_index(drop=True)], axis=1)
    navigation_spikes_rates_df = session.navigation_spike_rates_df
    keep_clusters = gc.filter_clusters(session.cluster_metrics, session.session_info, return_unique_IDs=True)
    for cluster in keep_clusters[:1]:
        cluster_rates = navigation_spikes_rates_df.xs(cluster, level=1, axis=1).firing_rate.to_numpy()
        nav_rates_df = navigation_rates_df.copy()
        nav_rates_df.loc[:, ("firing_rate", "")] = cluster_rates
        past_tuning_df = get_route_sequence_tuning_df(nav_rates_df, sequence="past")
        future_tuning_df = get_route_sequence_tuning_df(nav_rates_df, sequence="future")
        _get_route_tuning_vmax(future_tuning_df, past_tuning_df)
        plot_routes_tuning(future_tuning_df, title="Future Route Tuning")
        plot_routes_tuning(past_tuning_df, title="Past Route Tuning")
    return


def get_route_sequence_tuning_df(
    navigation_rates_df,
    sequence="future",
    smooth_SD=1,
    route_max=3,
    route_min=2,
    optimal=True,
    moving=True,
    distance_to_goal_decreasing=True,
    min_time_per_estimate=0.5,  # seconds
    min_trials_per_route_shift=2,
):
    """ """
    # smoothin
    if smooth_SD:
        cluster_rates = navigation_rates_df.firing_rate.to_numpy()
        cluster_rates = gaussian_filter1d(cluster_rates, sigma=smooth_SD * FRAME_RATE, axis=0)
        navigation_rates_df.loc[:, ("firing_rate", "")] = cluster_rates
    # filter data
    filter_masks = []
    if route_max:
        filter_masks.append(navigation_rates_df.n_routes.le(route_max).to_numpy())
    if route_min:
        filter_masks.append(navigation_rates_df.n_routes.ge(route_min).to_numpy())
    if optimal:
        filter_masks.append((navigation_rates_df.optimal_route == 1).to_numpy())
    if moving:
        filter_masks.append(navigation_rates_df.moving.to_numpy())
    if distance_to_goal_decreasing:
        filter_masks.append(navigation_rates_df.distance_to_goal.geodesic.diff().le(0).to_numpy())
    navigation_rates_df = navigation_rates_df[np.logical_and.reduce(filter_masks)]
    # group data at different route shifts
    if sequence == "future":
        route_cols = ["r"] + [f"r+{i}" for i in range(1, route_max)]
    elif sequence == "past":
        route_cols = [f"r-{i}" for i in range(route_max - 1, 0, -1)] + ["r"]
    dfs = []
    for future in route_cols:
        grouped_df = navigation_rates_df.groupby([("route", future), ("trial", "")]).firing_rate
        df = grouped_df.mean().reset_index()
        df.columns = ["route", "trial", "firing_rate"]
        df["route_shift"] = future
        df["frame_count"] = grouped_df.count().reset_index(drop=True)
        dfs.append(df)
    results_df = pd.concat(dfs, axis=0)
    if min_time_per_estimate:
        mask = results_df.frame_count.gt(min_time_per_estimate * FRAME_RATE)
        results_df = results_df[mask]
    if min_trials_per_route_shift:
        trial_counts = results_df.groupby(["route", "route_shift"]).trial.count()
        invalid_estimates = trial_counts[~trial_counts.ge(min_trials_per_route_shift)].index
        mask = results_df.set_index(["route", "route_shift"]).index.isin(invalid_estimates)
        results_df = results_df[~mask]
    # remove frame count
    results_df = results_df.drop(columns="frame_count")
    return results_df


def plot_routes_tuning(tuning_df, axes=None, vmax=None, title=None):
    """ """
    # sort data so plotting is consistant 
    tuning_df = tuning_df.sort_values(['route', 'route_shift'])
    if axes is None:
        f, axes = plt.subplots(1, 2, figsize=(10, 4), width_ratios=[1, 3])
    heatmap_df = tuning_df.groupby(["route", "route_shift"]).firing_rate.mean().unstack()
    # plot heatmap
    if vmax is None:
        vmax = heatmap_df.max().max()
    sns.heatmap(heatmap_df, ax=axes[0], cmap="viridis", vmin=0, vmax=vmax)
    axes[0].set_ylabel("")
    if title is not None:
        axes[0].set_title(title)
    # plot bar graph
    n_shifts = len(tuning_df.route_shift.unique())
    sns.stripplot(
        data=tuning_df,
        x="route",
        y="firing_rate",
        hue="route_shift",
        ax=axes[1],
        palette="bright",
        dodge=True,
        size=5,
        alpha=0.1,
        legend=False,
    )
    sns.pointplot(
        data=tuning_df,
        x="route",
        y="firing_rate",
        hue="route_shift",
        linestyle="none",
        dodge=0.45,
        ax=axes[1],
        palette="bright",
        markersize=5,
        markeredgewidth=0,
        err_kws={"linewidth": 2},
        errorbar="se",
        
    )
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=len(labels))
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Firing Rate (Hz)")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].tick_params(axis="x", which="both", rotation=45)  # Ensure tick labels are rotated correctly
    return




def _get_route_tuning_vmax(future_tuning_df, past_tuning_df):
    future = future_tuning_df.groupby(["route", "route_shift"]).firing_rate.mean().max()
    past = past_tuning_df.groupby(["route", "route_shift"]).firing_rate.mean().max()
    vmax = max(future.max(), past.max())
    return vmax



# %% Route Progress Tuning (not as refined as route aligned tuning, does not exclude cue events)


def get_session_route_progress_tuning(session, metric="time", smooth_SD=False, n_bins=15, n_previous_routes=3):
    """ """
    navigation_df = session.navigation_df
    routes_info_df = session.navigation_routes_df.reset_index(drop=True)
    navigation_routes_df = pd.concat([navigation_df, routes_info_df], axis=1)
    navigation_rates_df = session.navigation_spike_rates_df.reset_index(drop=True)
    keep_clusters = gc.filter_clusters(session.cluster_metrics, session.session_info, return_unique_IDs=True)
    # bin route_progress
    binned = pd.cut(navigation_routes_df.progress_to_goal[metric], bins=n_bins, include_lowest=True)
    navigation_routes_df.loc[:, ("route_progress", "binned")] = binned
    for cluster in keep_clusters:
        nav_df = navigation_routes_df.copy()
        cluster_rates = navigation_rates_df.xs(cluster, level=1, axis=1).firing_rate.to_numpy()
        if smooth_SD:
            cluster_rates = gaussian_filter1d(cluster_rates, sigma=smooth_SD * FRAME_RATE, axis=0)
        nav_df.loc[:, ("firing_rate", "")] = cluster_rates
        trial_av_rates = nav_df.groupby(
            [("trial", ""), ("route_progress", "binned"), ("route_order", "from_goal")], observed=True
        ).firing_rate.mean()
        trial_av_grouped_rates = trial_av_rates.reset_index().groupby(
            [("route_progress", "binned"), ("route_order", "from_goal")], observed=True
        )
        tuning_mean = trial_av_grouped_rates.mean().firing_rate.unstack()
        tuning_sem = trial_av_grouped_rates.sem().firing_rate.unstack()
        plot_route_progress_tuning(tuning_mean, tuning_sem, n_previous_routes=n_previous_routes)
    return


def plot_route_progress_tuning(tuning_mean, tuning_sem, n_previous_routes=3, axes=None):
    """ """
    if axes is None:
        f, axes = plt.subplots(n_previous_routes, 1, figsize=(5, 5), sharex=True, sharey=True)
    for i, ax in enumerate(axes):
        ax.set_ylabel(f"{i}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Route Progress (path_length)")
    f.supylabel("Routes from Goal (Hz)", x=-0.05)
    for i in range(n_previous_routes):
        bin_mids = np.array([b.mid for b in tuning_mean.index])
        mean = tuning_mean[i].to_numpy()
        sem = tuning_sem[i].to_numpy()
        axes[i].plot(bin_mids, mean, color="black")
        axes[i].fill_between(bin_mids, mean - sem, mean + sem, color="black", alpha=0.3)
    return


# %% Route change aligned ratrd (Not the best plot)


def plot_session_route_change_aligned_tuning(session):
    """ """
    route_change_aligned_rates_df = session.route_change_aligned_rates_df
    keep_clusters = gc.filter_clusters(session.cluster_metrics, session.session_info, return_unique_IDs=True)
    for cluster in keep_clusters:
        cluster_route_aligned_rates = route_change_aligned_rates_df[
            route_change_aligned_rates_df.cluster_unique_ID == cluster
        ]
        plot_route_change_aligned_tuning(cluster_route_aligned_rates, None, route_stratified=False)
    return


def plot_route_change_aligned_tuning(
    route_aligned_rates, routes_df=None, include_change_from_cue=True, route_stratified=False, smooth_SD=5, axes=None
):
    """ """
    route_changes = np.unique([c for c in route_aligned_rates.columns.get_level_values(0) if "route_change" in c])
    if axes is None:
        f, axes = plt.subplots(1, len(route_changes), figsize=(16, 4), sharex=True, sharey=True)
    for ax in axes:
        ax.axvline(0, color="black", linestyle="--", lw=0.5, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Firing Rate (Hz)")
    axes[3].set_xlabel("Reward")
    axes[2].set_xlabel("-1")
    axes[1].set_xlabel("-2")
    axes[0].set_xlabel("-3")
    f.supxlabel("Route Change Relative to Reward", y=-0.1)
    for ax, route_change in zip(axes, route_changes[::-1]):
        df = route_aligned_rates[route_change]
        if include_change_from_cue:
            df = df[df.latent.pre != "cue"]
        times = df.firing_rate.columns.to_numpy(dtype=float)
        if route_stratified:
            route2color = _get_route2color(routes_df)
            route2color["reward"] = "black"
            routes = df.latent.post.dropna().unique()
            for route in routes:
                color = route2color[route]
                route_rates = df[df.latent.post == route].firing_rate
                mean = route_rates.mean().values
                sem = route_rates.sem().values
                if smooth_SD:
                    mean = gaussian_filter1d(mean, sigma=smooth_SD)
                    sem = gaussian_filter1d(sem, sigma=smooth_SD)
                ax.plot(times, mean, color=color)
                ax.fill_between(times, mean - sem, mean + sem, color=color, alpha=0.3)
        else:
            df_rates = df.firing_rate
            times = df_rates.columns.to_numpy(dtype=float)
            mean = df_rates.mean().values
            sem = df_rates.sem().values
            if smooth_SD:
                mean = gaussian_filter1d(mean, sigma=smooth_SD)
                sem = gaussian_filter1d(sem, sigma=smooth_SD)
            ax.plot(times, mean, color="black")
            ax.fill_between(times, mean - sem, mean + sem, color="black", alpha=0.3)
    return


def _get_route2color(routes_df, colormap="nipy_spectral"):
    """ """
    route_ids = routes_df.index.to_list() + ["non_route"]
    cmap = plt.get_cmap(colormap, len(route_ids))
    return {route_id: cmap(i) for i, route_id in enumerate(route_ids)}


# %% Clean up route change aligned rates plotting


def plot_session_route_aligned_tuning(session):
    """ """
    route_aligned_rates_df = session.route_aligned_rates_df
    keep_clusters = gc.filter_clusters(session.cluster_metrics, session.session_info, return_unique_IDs=True)
    for cluster in keep_clusters:
        cluster_route_aligned_rates = route_aligned_rates_df[route_aligned_rates_df.cluster_unique_ID == cluster]
        plot_route_aligned_tuning(cluster_route_aligned_rates)
    return


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


def plot_route_stratified_aligned_tuning(
    route_aligned_rates_df,
    routes_df,
    remove_cue_events=True,
    max_routes=False,
    min_routes=False,
    smooth_SD=2,
    stretch_max=5,
    stretch_min=0,
    ax=None,
):
    """
    Input is df with warped firing rates aligned to routes before reward for a single cluster
    """
    # get route colors
    route2color = _get_route2color(routes_df)
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
    # filter df for each ordered route and route
    for i, ordered_route in enumerate(ordered_routes):
        x_coords = np.arange(i * trp, i * trp + trp)
        df = route_aligned_rates_df[ordered_route]
        routes = df.latent.l.dropna().unique()
        for route in routes:
            color = route2color[route]
            routes_df = df[df.latent.l == route]
            if remove_cue_events:
                routes_df = routes_df[routes_df.latent["l-1"] != "cue"]
            rates = routes_df.firing_rate
            mean = rates.mean().values
            sem = rates.sem().values
            if smooth_SD:
                rates = gaussian_filter1d(mean, sigma=smooth_SD)
                sem = gaussian_filter1d(sem, sigma=smooth_SD)
            ax.plot(x_coords, mean, color=color)
            ax.fill_between(x_coords, mean - sem, mean + sem, color=color, alpha=0.3)
    return


# %% Quick Population Analysis


def get_population_route_alaigned_rate_of_change(
    session, single_units=True, remove_cue_events=True, smooth_SD=1, stretch_max=5, stretch_min=0, ax=None
):
    """ """
    route_aligned_rates_df = session.route_aligned_rates_df
    keep_clusters = gc.filter_clusters(session.cluster_metrics, session.session_info, return_unique_IDs=True)
    if single_units:
        route_aligned_rates_df = route_aligned_rates_df[route_aligned_rates_df.cluster_unique_ID.isin(keep_clusters)]
    valid_stetch_mask = route_aligned_rates_df.trial_stretch["max"].lt(
        stretch_max
    ) & route_aligned_rates_df.trial_stretch["min"].gt(stretch_min)
    route_aligned_rates_df = route_aligned_rates_df[valid_stetch_mask]
    # find detials from route_aligned_rates_df
    column_fields = route_aligned_rates_df.columns.get_level_values(0).unique().to_numpy()
    ordered_routes = column_fields[["route_order" in c for c in column_fields]]
    n_ordered_routes = len(ordered_routes)
    # timepoints per route
    trp = len(route_aligned_rates_df.xs("firing_rate", level=1, axis=1).columns.get_level_values(1).unique())
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(7, 3))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("Δ Firing Rate")
        ax.set_xlabel("Route Aligned Trial Progress")
        ax.set_xlim(0, n_ordered_routes * trp)
        for i in range(n_ordered_routes):
            ax.axvline(i * trp, color="black", linestyle="--", lw=0.5, alpha=0.3)
    population_rates = []
    for cluster in keep_clusters:
        cluster_rates = route_aligned_rates_df[route_aligned_rates_df.cluster_unique_ID == cluster]
        combined_rates = []
        for ordered_route in ordered_routes:
            df = cluster_rates[ordered_route]
            if remove_cue_events:
                df = df[df.latent["l-1"] != "cue"]
            rates = df.firing_rate
            rates_mean = rates.mean().values
            combined_rates.append(rates_mean)
        population_rates.append(np.hstack(combined_rates))
    # population rate of change
    population_rates = np.vstack(population_rates)
    if smooth_SD:
        population_rates = gaussian_filter1d(population_rates, sigma=smooth_SD, axis=1)
    population_diff = np.diff(population_rates, axis=1)
    mean = np.mean(population_diff, axis=0)
    sem = np.std(population_diff, axis=0) / np.sqrt(len(keep_clusters))
    # plot
    ax.plot(mean, color="black")
    ax.fill_between(np.arange(len(mean)), mean - sem, mean + sem, color="black", alpha=0.3)
