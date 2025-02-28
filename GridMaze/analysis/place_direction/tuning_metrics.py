""" This analysis is for looking at the correlation of place-direction tuning across split halves of data """
# %% Imports
import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import seaborn as sns
from .. import get_sessions as gs
from ..cluster_tuning import place_direction as pdt

# %% Global variables


# %% Functions


def get_cluster_place_direction_tuning_distribution(visualise_r2_range=(0.2, 1)):
    """"""
    sessions = gs.get_sessions(
        subject_IDs="all",
        maze_number="all",
        day_on_maze="late",
        with_data=["navigation_df", "navigation_spike_rates_df", "cluster_metrics"],
    )
    cluster_place_direction_tuning_df = pd.concat([get_place_direction_tuning_metrics_df(s) for s in sessions], axis=0)
    # drop clusters with slope values that are nan or inf
    cluster_place_direction_tuning_df = cluster_place_direction_tuning_df.replace([np.inf, -np.inf], np.nan)
    cluster_place_direction_tuning_df = cluster_place_direction_tuning_df.dropna()
    r2_cut_offs = np.arange(0, 1, 0.02)

    r2_scores = cluster_place_direction_tuning_df.r2_score
    r2_visualisation_mask = np.logical_and(r2_scores > visualise_r2_range[0], r2_scores < visualise_r2_range[1])
    plotting_data = cluster_place_direction_tuning_df[r2_visualisation_mask]
    median_thetas = [plotting_data[plotting_data.r2_score < cut_off].theta.median() for cut_off in r2_cut_offs]
    median_slopes = [plotting_data[plotting_data.r2_score < cut_off].slope.median() for cut_off in r2_cut_offs]
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5), clear=True)
    sns.histplot(data=plotting_data, x="slope", y="r2_score", ax=ax1, cbar=True)
    ax1.axvline(1, color="black", alpha=0.5)
    ax1.plot(median_slopes, r2_cut_offs, color="red")
    ax1.set_xlim(0, 3)
    sns.histplot(data=plotting_data, x="theta", y="r2_score", ax=ax2, cbar=True)
    ax2.axvline(np.pi / 4, color="black", alpha=0.5)
    ax2.plot(median_thetas, r2_cut_offs, color="red")
    ax2.set_xlim(0, np.pi / 2)
    return


def get_place_direction_tuning_metrics_df(
    session, navigation_only=True, moving_only=True, min_occupancy_threshold=1, plot_cluster_fits=False
):
    """X"""
    maze_place_directions = pdt.get_all_location_cdirs(session.simple_maze())
    navigation_rates_df = session.get_navigation_activity_df(activity_type="firing_rate", cluster_type="good")
    cluster_unique_IDs = navigation_rates_df.firing_rate.columns
    if navigation_only:
        navigation_rates_df = navigation_rates_df[navigation_rates_df.trial_phase == "navigation"]
    if moving_only:
        navigation_rates_df = navigation_rates_df[navigation_rates_df.moving]
    place_direction_rates_dfs = []
    trial_splits = split_trials(navigation_rates_df)
    np.random.shuffle(list(trial_splits))
    for trials in trial_splits:
        rates_df = navigation_rates_df[navigation_rates_df.trial.isin(trials)]
        place_direction_columns = [("maze_position", "simple"), ("cardinal_movement_direction", "")]
        place_direction_averaged_rates_df = rates_df.groupby(place_direction_columns).mean().firing_rate
        sub_min_occ_mask = (
            rates_df.groupby(place_direction_columns).count().time < min_occupancy_threshold * FRAME_RATE
        )
        place_direction_averaged_rates_df[sub_min_occ_mask] = np.nan  # low occupancy locations = nan
        unvistied_place_directions = list(set(maze_place_directions) - set(place_direction_averaged_rates_df.index))
        if len(unvistied_place_directions) > 0:
            unvisited_nan_activity = pd.DataFrame(
                data=np.nan,
                index=pd.MultiIndex.from_tuples(unvistied_place_directions),
                columns=place_direction_averaged_rates_df.columns,
            )
            place_direction_averaged_rates_df = pd.concat(
                (place_direction_averaged_rates_df, unvisited_nan_activity), axis=0
            ).sort_index()
        place_direction_rates_dfs.append(place_direction_averaged_rates_df)
    place_direction_rates1, place_direction_rates2 = place_direction_rates_dfs
    cluster2split_halves_correlation_slope = {}
    cluster2split_halves_r2_score = {}
    cluster2split_halves_theta = {}
    for cluster in cluster_unique_IDs:
        rate_vector1 = place_direction_rates1[cluster].to_numpy()
        rate_vector2 = place_direction_rates2[cluster].to_numpy()
        # remove nan values
        nan_mask = np.logical_or(np.isnan(rate_vector1), np.isnan(rate_vector2))  #
        if all(nan_mask):
            continue
        rate_vector1 = rate_vector1[~nan_mask]
        rate_vector2 = rate_vector2[~nan_mask]
        # compute metrics
        lr = LinearRegression()
        lr.fit(rate_vector1.reshape(-1, 1), rate_vector2)
        r2 = lr.score(rate_vector1.reshape(-1, 1), rate_vector2)
        slope, intercept = deming_regression(rate_vector1, rate_vector2, delta=1)
        theta = np.arctan(slope)
        cluster2split_halves_correlation_slope[cluster] = slope
        cluster2split_halves_r2_score[cluster] = r2
        cluster2split_halves_theta[cluster] = theta
        # plotting
        if plot_cluster_fits:
            f, ax = plt.subplots()
            ax.scatter(rate_vector1, rate_vector2)
            ax.plot(rate_vector1, slope * rate_vector1 + intercept, color="red")
            ax.text(0.5, 0.8, f"slope: {slope}\nr2: {r2:.2f}\ntheta: {theta:.2f}", transform=ax.transAxes)
    df = pd.DataFrame(
        data={
            "slope": cluster2split_halves_correlation_slope,
            "r2_score": cluster2split_halves_r2_score,
            "theta": cluster2split_halves_theta,
        }
    )
    return df


def split_trials(navigation_rates_df):
    """splits trials from a navigation_df-containing dataframe into two halves, stratified by goal if possible."""
    if len(pd.Series(navigation_rates_df.goal.unique()).dropna()) == navigation_rates_df.trial.max():
        # too few trial to stratify split by goal -> split randomly instead
        trials = navigation_rates_df.trial.unique()
        trials = trials[~np.isnan(trials)]
        np.random.shuffle(trials)
        split_size = len(trials) // 2
        return trials[:split_size], trials[split_size:]
    else:
        trials2goal_df = navigation_rates_df.loc[
            :, (navigation_rates_df.columns.get_level_values(0).isin(["trial", "goal"]))
        ]
        trials_per_goal = trials2goal_df.dropna().groupby("goal")["trial"].apply(lambda x: np.unique(x))
        trials_per_goal = trials_per_goal.apply(
            lambda x: np.random.choice(x, size=len(x), replace=False)
        )  # shuffle trials
        goal_split_trials_df = trials_per_goal.apply(pd.Series)
        middle_index = len(goal_split_trials_df.columns) // 2
        split_1 = goal_split_trials_df.iloc[:, :middle_index].to_numpy().flatten()
        split_1 = split_1[~np.isnan(split_1)]
        split_2 = goal_split_trials_df.iloc[:, middle_index:].to_numpy().flatten()
        split_2 = split_2[~np.isnan(split_2)]
        return split_1, split_2


def deming_regression(x, y, delta=1):
    """Compute the slope and intercept for a Deming regression of y against x.
    delta is the ratio of the error variances in y over error variances in x.
    If delta is set to one then gives orthogonal regression.
    return slope and intercept for y = slope*x + intercept
    see https://en.wikipedia.org/wiki/Deming_regression"""
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    covmat = np.cov(x, y, bias=True)
    cov_xx = covmat[0, 0]
    cov_yy = covmat[1, 1]
    cov_xy = covmat[0, 1]
    slope = (cov_yy - delta * cov_xx + np.sqrt((cov_yy - delta * cov_xx) ** 2 + 4 * delta * cov_xy**2)) / (
        2 * cov_xy
    )
    intercept = mean_y - slope * mean_x
    return slope, intercept
