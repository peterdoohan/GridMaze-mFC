"""This script is for visualling place-direction heatmaps split by distance to goal (short, medium, long)"""
# %% Imports
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import itertools

from .. import get_sessions as gs
from ...maze import representations as mr
from ...maze import plotting as mp

from ..processing.filter_navigation_rates_df import _filter_navigation_df
from ..distance_to_goal import neural_tensor_decomposition as ntd

from .tuning_metrics import deming_regression
from sklearn.linear_model import LinearRegression


# %% Global variables
ANALYSIS_DATA_PATH = "../data/analysis_data"
with open(os.path.join(ANALYSIS_DATA_PATH, "analysis_info.json"), "r") as infile:
    ANALYSIS_INFO = json.load(infile)

FRAME_RATE = 60

DISTANCE_METRIC = "geodesic"
DISTANCE_CUTOFFS = (
    [0] + [ANALYSIS_INFO["distance_to_goal_quantiles"][DISTANCE_METRIC][str(q)] for q in [0.33, 0.67]] + [np.inf]
)

with open("../data/experiment_info.json") as input_file:
    EXP_INFO = json.load(input_file)


# %%
def plot_add_mull_slope_test():
    """"""
    slopes_df = pd.DataFrame(columns=["subject_ID", "slope"])
    for subject in EXP_INFO["subject_IDs"]:
        split_distance_fits_df = get_subject_split_distance_fits_df(subject, plot=False)
        slope_value = get_subject_slope_value(split_distance_fits_df)
        slopes_df = slopes_df.append({"subject_ID": subject, "slope": slope_value}, ignore_index=True)
    # plotting
    f, ax = plt.subplots(1, 1, figsize=(1, 3), clear=True)
    sns.swarmplot(
        data=slopes_df,
        y="slope",
        color="violet",
        ax=ax,
        size=8,
    )
    ax.set_ylabel("slope")
    ax.axhline(0, ls="--", lw=0.5, color="b")
    ax.axhline(1, ls="--", lw=0.5, color="k")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return slopes_df


def get_subject_split_distance_fits_df(subject_ID, plot=True):
    """ """
    sessions = gs.get_sessions(
        subject_IDs=[subject_ID],
        maze_number="all",
        day_on_maze="late",
        with_data=["navigation_df", "navigation_spike_rates_df", "cluster_metrics"],
    )
    slopes = []
    ffs = []
    for session in sessions:
        cluster_distance_split_dfs = get_distance_split_place_direction_rates(session)
        for df in cluster_distance_split_dfs:
            slope, ff = get_split_distance_slope(df)
            slopes.append(slope)
            ffs.append(ff)
    split_distance_fits_df = pd.DataFrame(columns=["slope", "ff"])
    split_distance_fits_df["slope"] = slopes
    split_distance_fits_df["ff"] = ffs

    if plot:
        _plot_subject_split_distance_fits(split_distance_fits_df)
    return split_distance_fits_df


def get_subject_slope_value(split_distance_fits_df, slope_range=(-50, 50), ff_range=(0, 10)):
    # filter
    outlier_mask = np.logical_and(
        split_distance_fits_df.slope.between(*slope_range), split_distance_fits_df.ff.between(*ff_range)
    )
    fits_df = split_distance_fits_df[outlier_mask]
    x = fits_df.ff.values.reshape(-1, 1)
    y = fits_df.slope.values.reshape(-1, 1)
    lin_reg = LinearRegression(fit_intercept=True)
    lin_reg.fit(x, y)
    lr_slope = lin_reg.coef_[0][0]
    return lr_slope


def _plot_subject_split_distance_fits(split_distance_fits_df, slope_range=(-50, 50), ff_range=(0, 10)):
    # filter
    outlier_mask = np.logical_and(
        split_distance_fits_df.slope.between(*slope_range), split_distance_fits_df.ff.between(*ff_range)
    )
    fits_df = split_distance_fits_df[outlier_mask]
    # fit linear regression
    x = fits_df.ff.values.reshape(-1, 1)
    max_x = np.max(x)
    y = fits_df.slope.values.reshape(-1, 1)
    lin_reg = LinearRegression(fit_intercept=True)
    lin_reg.fit(x, y)
    lr_slope = lin_reg.coef_[0][0]
    lr_int = lin_reg.intercept_[0]
    # plot
    f, ax = plt.subplots(1, 1, figsize=(3, 5))
    sns.histplot(data=fits_df, x="ff", y="slope", bins=200, ax=ax)
    sns.regplot(
        data=fits_df,
        x="ff",
        y="slope",
        robust=True,
        scatter=False,
        color="red",
        line_kws={"lw": 1, "alpha": 0.5},
        ax=ax,
    )
    ax.plot([1, max_x], [1, max_x], lw=0.5, ls="--", color="k")
    ax.plot([1, max_x], [1, 1], lw=0.5, ls="--", color="b")
    ax.set_xlabel("f2/f1")
    ax.set_ylabel("slope")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return


# %% Functions


def get_distance_split_place_direction_rates(session, minimum_occupancy=0.5):
    """ """
    navigation_rates_df = session.get_navigation_activity_df(activity_type="firing_rate", cluster_type="good")
    navigation_rates_df = _filter_navigation_df(
        navigation_rates_df,
        navigation_only=True,
        moving_only=True,
        exclude_time_at_goal=True,
    )
    cluster_unique_IDs = navigation_rates_df.firing_rate.columns.to_numpy()
    place_direction_cols = [("maze_position", "simple"), ("cardinal_movement_direction", "")]
    distance_cutoff_ranges = [(-np.inf, np.inf)] + [
        tuple(DISTANCE_CUTOFFS[i : i + 2]) for i in range(len(DISTANCE_CUTOFFS) - 1)
    ]
    all_place_directions = mr.get_maze_place_direction_pairs(session.simple_maze())
    distance_split_place_direction_dfs = []
    for lower_lim, upper_lim in distance_cutoff_ranges:
        # get place_direction pd.Series for each distance range
        distance_to_goal = navigation_rates_df.distance_to_goal[DISTANCE_METRIC]
        distance_rates_df = navigation_rates_df[
            np.logical_and(distance_to_goal >= lower_lim, distance_to_goal < upper_lim)
        ]
        place_direction_grouped_rates = distance_rates_df.set_index(place_direction_cols).groupby(place_direction_cols)
        place_direction2sub_min_occ = place_direction_grouped_rates.count().time < minimum_occupancy * FRAME_RATE
        place_direction_rates_df = place_direction_grouped_rates.mean().firing_rate
        place_direction_rates_df[place_direction2sub_min_occ] = np.nan
        # add unvisted locations as nan for all clusters
        visited_place_directions = place_direction_rates_df.index.to_numpy()
        unvistied_place_directions = list(set(all_place_directions) - set(visited_place_directions))
        if len(unvistied_place_directions) > 0:
            unvisitied_place_direction_df = pd.DataFrame(
                index=pd.MultiIndex.from_tuples(unvistied_place_directions), columns=cluster_unique_IDs, data=np.nan
            )
            place_direction_rates_df = pd.concat([place_direction_rates_df, unvisitied_place_direction_df], axis=0)
        place_direction_rates_df.sort_index(inplace=True)
        place_direction_rates_df.index.names = ["maze_position", "direction"]
        #
        distance_split_place_direction_dfs.append(place_direction_rates_df)
    # refactor into a df for each cluster instead of a df for each distance split
    cluster_distance_split_dfs = [
        pd.concat([df[c] for df in distance_split_place_direction_dfs], axis=1) for c in cluster_unique_IDs
    ]
    for df in cluster_distance_split_dfs:
        df.columns = ["all", "close", "medium", "far"]
    return cluster_distance_split_dfs


def plot_session_distance_split_heatmaps(
    cluster_distance_split_dfs, simple_maze, labels=["all", "close", "medium", "far"]
):
    for df in cluster_distance_split_dfs:
        ntd.plot_split_place_direction_heatmaps(df, simple_maze, labels=labels)


# %%


def get_split_distance_slope(
    distance_split_place_direction_df,
    choose_splits="max_rates",
    min_rate_threshold=0.0,
    plot_fit=False,
    regression_method="deming",
    valid_points_threshold=40,
):
    if not regression_method in ["deming", "linear"]:
        raise ValueError("regression_methods must be 'deming', 'linear', or both type: list")
    if "all" in distance_split_place_direction_df.columns:
        distance_split_place_direction_df.drop(columns="all", inplace=True)
    chosen_splits = choose_distance_splits(distance_split_place_direction_df, method=choose_splits)
    splits_df = distance_split_place_direction_df[list(chosen_splits)].copy()
    # only consider place-direcitons with > min_rate_threshold in both splits
    valid_splits_df = splits_df[splits_df.ge(min_rate_threshold).all(axis=1)]
    if len(valid_splits_df) < valid_points_threshold:
        return (np.nan, np.nan)  # unreliable slope estimate
    max_rate = valid_splits_df.max().max()
    # order splits to mean(y) > mean(x) consistanlty
    splits = valid_splits_df.to_numpy().T
    split_mean = splits.mean(axis=1)
    bigger_split_mask = split_mean == split_mean.max()
    y = splits[bigger_split_mask][0]
    x = splits[~bigger_split_mask][0]
    # fit regression
    ff = np.mean(y) / np.mean(x)
    if regression_method == "linear" or plot_fit:
        lin_reg = LinearRegression(fit_intercept=True)
        lin_reg.fit(y.reshape(-1, 1), x)
        lr_slope = lin_reg.coef_[0]
        lr_int = lin_reg.intercept_
        if regression_method == "linear":
            slope = lr_slope
    if regression_method == "deming" or plot_fit:
        dr_slope, dr_int = deming_regression(x, y, delta=1)
        if regression_method == "deming":
            slope = dr_slope
    if plot_fit:
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.scatter(x, y, color="darkviolet", alpha=0.5)
        ax.plot(x, lr_slope * x + lr_int, color="red", lw=3, label="linear regression")
        ax.plot(x, dr_slope * x + dr_int, color="orange", lw=3, label="deming regression")
        ax.plot([0, 100], [0, 100], lw=0.5, ls="--", color="black")
        ax.set_xlim(-0.1, max_rate)
        ax.set_ylim(-0.1, max_rate)
        ax.set_title(f"LR slope = {lr_slope:.2f}, DR slope = {dr_slope:.2f}, f2/f1 = {ff:.2f}", size="8")
        ax.set_xlabel("d1")
        ax.set_ylabel("d2")
        ax.legend()
    return slope, ff


def choose_distance_splits(df, method="valid_overlap"):
    """
    only columns to compared should be in df ('all' column should be dropped)
    """
    if not method in ["valid_overlap", "max_rates"]:
        raise ValueError("choose_splits must be 'valid_overlap' or 'max_rates'")
    split_labels = df.columns.to_numpy()
    unique_pairs = list(itertools.combinations(split_labels, 2))
    if method == "valid_overlap":
        # choice_metric = valid overlap
        choice_metric = [df[list(pair)].dropna().all(axis=1).sum() for pair in unique_pairs]
    if method == "max_rates":
        # choice_metric = sum of av firing rates
        choice_metric = [df[list(pair)].mean(axis=0).sum() for pair in unique_pairs]
    return unique_pairs[np.argmax(choice_metric)]


# %%


def plot_distance_split_session_trajectories(session, smooth_SD=5, labels=["all", "close", "medium", "far"]):
    """
    sanity check function that plots the trajectory snippits inside each distance split
    to make sure they are not different in unexpected ways
    """
    simple_maze = session.simple_maze()
    navigation_df = session.navigation_df
    navigation_df = _filter_navigation_df(
        navigation_df,
        minimum_firing_rate=False,
        navigation_only=True,
        moving_only=True,
        exclude_time_at_goal=True,
    )
    distance_cutoff_ranges = [(-np.inf, np.inf)] + [
        tuple(DISTANCE_CUTOFFS[i : i + 2]) for i in range(len(DISTANCE_CUTOFFS) - 1)
    ]
    f, axes = plt.subplots(1, 4, figsize=(16, 4))
    for (lower_lim, upper_lim), ax, label in zip(distance_cutoff_ranges, axes, labels):
        # plot background simple_maze
        ax.set_title(label)
        mp.plot_simple_maze_silhouette(
            simple_maze,
            ax,
            color="silver",
            node_size=250,
            edge_size=6,
        )
        # plot trajectories
        distance_to_goal = navigation_df.distance_to_goal[DISTANCE_METRIC]
        distance_split_navigation_df = navigation_df[
            np.logical_and(distance_to_goal >= lower_lim, distance_to_goal < upper_lim)
        ].copy()
        # add a navigation_segment column that captures each continous seg of behaviour separately
        idx = distance_split_navigation_df.index.to_numpy()
        distance_split_navigation_df.loc[:, ("navigation_segment", "")] = np.cumsum(
            np.diff(idx, prepend=(idx[0] - 1)) > 1
        )
        # plot each behavioural segment w/ optional smoothing
        for segment in distance_split_navigation_df.navigation_segment.unique():
            segment_df = distance_split_navigation_df[distance_split_navigation_df.navigation_segment == segment]
            x_traj = segment_df.centroid_position.x
            y_traj = segment_df.centroid_position.y
            if smooth_SD:
                x_traj = gaussian_filter1d(x_traj, sigma=smooth_SD)
                y_traj = gaussian_filter1d(y_traj, sigma=smooth_SD)
            ax.plot(x_traj, y_traj, lw=4, alpha=0.2, color="darkviolet", zorder=2)
    return
