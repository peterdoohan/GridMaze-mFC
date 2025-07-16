"""New library for plotting distance to goal aligned rates"""

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import normal
from GridMaze.analysis.distance_to_goal import distributions as dd
from GridMaze.analysis.core import convert
from GridMaze.maze import plotting as mp
from scipy.ndimage import gaussian_filter1d

# %% Global Variables
FRAME_RATE = 60

# %% Functions


def plot_session_distance_to_goal_tuning(session, metrics=("distance_to_goal", "geodesic"), goal_stratified=False):
    navigation_rates_df = session.get_navigation_activity_df(type="rates", cluster_kwargs={"single_units": True})
    distance_info = navigation_rates_df[
        [("goal", ""), ("trial", ""), ("moving", ""), ("steps_to_goal", "future"), metrics]
    ]
    distance_info = distance_info.droplevel(1, axis=1)
    cluster_unique_IDs = navigation_rates_df.firing_rate.columns.to_numpy()
    for cluster in cluster_unique_IDs:
        cluster_rates = navigation_rates_df.xs(cluster, level=1, axis=1)
        distance_rates_df = pd.concat([distance_info, cluster_rates], axis=1)
        distance_tuning_df = get_distance_to_goal_tuning_df(distance_rates_df, metrics=metrics)
        plot_distance_tuning(distance_tuning_df, metrics, goal_stratified=goal_stratified)


def get_distance_to_goal_tuning_df(
    distance_rates_df,
    metrics=("distance_to_goal", "geodesic"),
    bin_spacing=0.05,
    max_steps_to_goal=30,
    moving_only=False,
):
    """ """
    trial2goal = distance_rates_df.set_index("trial").goal.dropna().to_dict()
    # deal with moving only
    if moving_only:
        distance_rates_df = distance_rates_df[distance_rates_df.moving]
    if max_steps_to_goal is not None:
        distance_rates_df = distance_rates_df[distance_rates_df.steps_to_goal < max_steps_to_goal]
    # remove frames where distance is above max (treat as outliers)
    if metrics[0] == "distance_to_goal":
        max_distance = dd.get_distance_percentile(metrics, 0.85)
        n_bins = int(max_distance / bin_spacing)
        distance_rates_df = distance_rates_df[distance_rates_df[metrics[0]] < max_distance]
        bins = convert._get_distance_bins(
            binning_method="uniform",
            n_distance_bins=n_bins,
            distance_metrics=metrics,
            max_distance=max_distance,
        )
    else:
        NotImplementedError()
    # bin distances
    distance_rates_df.loc[:, "distance_bin"] = pd.cut(
        distance_rates_df[metrics[0]], bins=bins, include_lowest=True
    ).to_numpy()
    # average over frames in each bin over trials
    trial_av_rates = distance_rates_df.groupby(["trial", "distance_bin"], observed=True).firing_rate.mean().unstack()
    # organise tuning df
    tuning_df = pd.DataFrame()
    tuning_df[("trial", "")] = trial_av_rates.index
    tuning_df[("goal", "")] = trial_av_rates.index.map(trial2goal)
    tuning_df.columns = pd.MultiIndex.from_tuples(tuning_df.columns)
    trial_av_rates.columns = pd.MultiIndex.from_tuples([("distance", c.mid) for c in trial_av_rates.columns])
    tuning_df = pd.concat([tuning_df, trial_av_rates.reset_index(drop=True)], axis=1)
    return tuning_df


def plot_distance_tuning(
    distance_tuning_df, metrics, goal_stratified=False, normalisation=None, smooth_SD=1, ax=None, color="darkcyan"
):
    """"""
    # format axis
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(6, 3), clear=True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Firing Rate (Hz)")
    ax.set_xlabel(f"{metrics[0]}: {metrics[1]}")
    if metrics[0] == "distance_to_goal":
        ax.set_xlim(0, dd.get_distance_percentile(metrics, 0.85))
    else:  # progress_to_goal
        ax.set_xlim(0, 1)
    # process data for plotting
    if not goal_stratified:
        mean_rates = distance_tuning_df.distance.mean(axis=0)
        sem_rates = distance_tuning_df.distance.sem(axis=0)
        distances = mean_rates.index.to_numpy().astype(float)
        mean = mean_rates.to_numpy()
        sem = sem_rates.to_numpy()
        if smooth_SD:
            mean = gaussian_filter1d(mean, smooth_SD)
            sem = gaussian_filter1d(sem, smooth_SD)
        _plot_distance_tuning(mean, sem, distances, ax, color, normalisation=normalisation)
    else:
        goal2color = mp.get_goal2standard_color()
        for goal in distance_tuning_df.goal.unique():
            mean_rates = distance_tuning_df[distance_tuning_df.goal == goal].distance.mean(axis=0)
            sem_rates = distance_tuning_df[distance_tuning_df.goal == goal].distance.sem(axis=0)
            distances = mean_rates.index.to_numpy().astype(float)
            mean = mean_rates.to_numpy()
            sem = sem_rates.to_numpy()
            if smooth_SD:
                mean = gaussian_filter1d(mean, smooth_SD)
                sem = gaussian_filter1d(sem, smooth_SD)
            _plot_distance_tuning(mean, sem, distances, ax, goal2color[goal])
    return


def _plot_distance_tuning(mean, sem, distances, ax, color, normalisation=None):
    if normalisation == None:
        ax.plot(distances, mean, color=color)
        ax.fill_between(distances, mean - sem, mean + sem, color=color, alpha=0.2)
    elif normalisation == "max":
        lower = (mean - sem) / mean.max()
        upper = (mean + sem) / mean.max()
        _mean = mean / mean.max()
        ax.plot(distances, _mean, color=color)
        ax.fill_between(distances, lower, upper, color=color, alpha=0.2)
    else:
        raise ValueError(f"Unknown normalisation: {normalisation}")


# %% theta mod distance tuning


def plot_session_theta_mod_distance_to_goal_tuning(
    session,
    metrics=("distance_to_goal", "geodesic"),
    theta_peak_ind=[4, 5, 6, 7],
    theta_trough_ind=[0, 1, 10, 11],
    bin_spacing=0.04,
    max_steps_to_goal=30,
    moving_only=True,
    smooth_SD=2,
):
    """
    Note: theta_peak_ind work with the 12 theta bins used in navigation_theta_spike_counts_df.
    and from GridMaze.analysis.lfp.theta_mod we know the peak of theta is in the middle of the theta cycle,
    and the trough of theta is at the start and end of the theta cycle.

    Would be good to have a more full proof way to define this in the function inputs
    """
    # load data
    navigation_df = session.navigation_df.copy()
    distance_info = navigation_df[[("goal", ""), ("trial", ""), ("moving", ""), ("steps_to_goal", "future"), metrics]]
    theta_spike_counts = session.navigation_theta_spike_counts_df.reset_index(drop=True)
    cluster_metrics = session.cluster_metrics
    session_info = session.session_info

    # filter for single units
    single_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID
    single_units = convert.cluster_IDs2scluster_unique_IDs(session_info, single_units)

    # get theta phases
    phases = theta_spike_counts.columns.get_level_values(2).unique().astype(float)
    theta_peak_cols = phases[theta_peak_ind]
    theta_trough_cols = phases[theta_trough_ind]

    for cluster in single_units:
        theta_spikes = theta_spike_counts.spike_count[cluster]
        theta_peak_spikes = theta_spikes[theta_peak_cols].sum(axis=1)
        theta_trough_spikes = theta_spikes[theta_trough_cols].sum(axis=1)
        distance_spikes_df = distance_info.copy()
        distance_spikes_df.loc[:, ("theta", "peak")] = theta_peak_spikes
        distance_spikes_df.loc[:, ("theta", "trough")] = theta_trough_spikes
        distance_theta_tuning_df = get_theta_distance_to_goal_tuning(
            distance_spikes_df,
            metrics=metrics,
            bin_spacing=bin_spacing,
            max_steps_to_goal=max_steps_to_goal,
            moving_only=moving_only,
        )
        f, ax = plt.subplots(1, 1, figsize=(4, 2), clear=True)
        plot_theta_distance_tuning(
            distance_theta_tuning_df,
            metrics=metrics,
            smooth_SD=smooth_SD,
            ax=ax,
        )
        ax.set_title(cluster)


def get_theta_distance_to_goal_tuning(
    distance_spikes_df,
    metrics=("distance_to_goal", "geodesic"),
    bin_spacing=0.05,
    max_steps_to_goal=30,
    moving_only=False,
):
    """
    Note input is spikes so we need to convert to rates
    """
    # deal with moving only
    if moving_only:
        distance_spikes_df = distance_spikes_df[distance_spikes_df.moving]
    if max_steps_to_goal is not None:
        distance_spikes_df = distance_spikes_df[distance_spikes_df.steps_to_goal.future < max_steps_to_goal]
    # remove frames where distance is above max (treat as outliers)
    if metrics[0] == "distance_to_goal":
        max_distance = dd.get_distance_percentile(metrics, 0.85)
        n_bins = int(max_distance / bin_spacing)
        distance_spikes_df = distance_spikes_df[distance_spikes_df[metrics] < max_distance]
        bins = convert._get_distance_bins(
            binning_method="uniform",
            n_distance_bins=n_bins,
            distance_metrics=metrics,
            max_distance=max_distance,
        )
    else:
        NotImplementedError()
    # bin distances
    distance_spikes_df.loc[:, ("distance_bin", "")] = pd.cut(
        distance_spikes_df[metrics], bins=bins, include_lowest=True
    ).to_numpy()

    # get total spikes in each distance bin and div by occupancy to get rates
    grouped_df = distance_spikes_df.groupby(["trial", "distance_bin"])
    distance_occ = grouped_df.theta.count() * (1 / FRAME_RATE)  # convert to seconds
    distance_spikes = grouped_df.theta.sum()
    distance_theta_rates = distance_spikes / distance_occ
    return distance_theta_rates  # trials x distance bins, 2 (theta peak and trough)


def plot_theta_distance_tuning(distance_theta_rates, metrics, ax=None, colors=("darkcyan", "royalblue"), smooth_SD=1):
    """ """
    # format axis
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(6, 3), clear=True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Firing Rate (Hz)")
    ax.set_xlabel(f"{metrics[0]}: {metrics[1]}")
    if metrics[0] == "distance_to_goal":
        ax.set_xlim(0, dd.get_distance_percentile(metrics, 0.85))
    else:  # progress_to_goal
        ax.set_xlim(0, 1)

    # process
    distance_grouped = distance_theta_rates.groupby("distance_bin")
    mean = distance_grouped.mean()
    sem = distance_grouped.sem()
    bin_mid = [c.mid for c in mean.index]
    mean.index = bin_mid
    sem.index = bin_mid
    # plot
    for i, (theta_phase, color) in enumerate(zip(["peak", "trough"], colors)):
        _mean = mean.theta[theta_phase].values
        _sem = sem.theta[theta_phase].values
        if smooth_SD:
            _mean = gaussian_filter1d(_mean, smooth_SD)
            _sem = gaussian_filter1d(_sem, smooth_SD)
        ax.plot(bin_mid, _mean, color=color, label=theta_phase)
        ax.fill_between(
            bin_mid,
            _mean - _sem,
            _mean + _sem,
            color=color,
            alpha=0.2,
        )
    ax.legend()
