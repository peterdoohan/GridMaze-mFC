"""
Library that generates the experiment level distributions of different distance to goal metrics,
and saves them to analysis info.
"""

# %% Imports
import json
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from GridMaze.maze import partitions as mt
from GridMaze.analysis.core import get_sessions as gs

from GridMaze.analysis.distance_to_goal import decoding_utils as du

# %% Global Variables
from GridMaze.paths import ANALYSIS_INFO_PATH


# %% Look at distance-to-goal distributions stratified by goal in differnet maze partitions


def plot_test_not_train_AB_splits(session, s=3, max_steps_to_goal=30):
    """ """
    simple_maze = session.simple_maze()
    A_locs, B_locs = mt.get_AB_split(simple_maze, s=s, plot=True)

    input_data = du.get_place_decoding_input_data(session, resolution=0.5)
    input_data = input_data[input_data.steps_to_goal.future.le(max_steps_to_goal)]
    folds_df = du.get_folds_df(session, goal_stratified=True)
    folds = folds_df.columns.get_level_values(0).unique()
    goal_distance_count_dfs = []
    for fold in folds:
        fold_df = folds_df[fold]
        train_df, test_df = du._get_test_train_dfs(input_data, fold_df, training_trial_phases=["navigation"])
        train_A_df, train_B_df = (
            train_df[train_df.maze_position.simple.isin(A_locs)],
            train_df[train_df.maze_position.simple.isin(B_locs)],
        )
        test_A_df, test_B_df = (
            test_df[test_df.maze_position.simple.isin(A_locs)],
            test_df[test_df.maze_position.simple.isin(B_locs)],
        )
        train_dfs = []
        for df in [train_A_df, train_B_df]:
            counts_df = df.groupby([("goal", ""), ("steps_to_goal", "future")]).time.count().unstack()
            counts_df = counts_df.replace({np.nan: 0})
            train_dfs.append(counts_df)
        train_A_counts, train_B_counts = train_dfs

        test_dfs = []
        for df in [test_A_df, test_B_df]:
            counts_df = df.groupby([("goal", ""), ("steps_to_goal", "future")]).time.count().unstack()
            test_dfs.append(counts_df)
        test_A_counts, test_B_counts = test_dfs

        for train_counts, test_counts in zip([train_A_counts, train_B_counts], [test_B_counts, test_A_counts]):
            r = train_counts.lt(1) & test_counts.gt(0)  # instances with no training data where the test data is
            r.infer_objects(copy=True)
            r = r.fillna(False)
            r = r.astype(int)
            goal_distance_count_dfs.append(r)
    df = pd.concat(goal_distance_count_dfs, axis=0)
    df = df.groupby(df.index).mean()
    f, ax = plt.subplots()
    sns.heatmap(df, ax=ax)
    return df


# %% distance metric distributions


def get_distance_percentile(distance_metric, percentile):
    """
    Get the value at a specific percentile from the histogram data.

    Parameters:
        distance_metric (tuple): The names of the metric to load.
        percentile (float): The desired percentile (0.0 to 1.0).

    Returns:
        float: The interpolated value at the specified percentile.
    """
    data = _load_distribution(distance_metric)
    bin_edges = data["bin_edges"]
    counts = data["counts"]
    total_counts = sum(counts)
    percentile_counts = total_counts * percentile

    cumulative_counts = 0
    for i, count in enumerate(counts):
        cumulative_counts += count
        if cumulative_counts >= percentile_counts:
            # Interpolate within the bin
            previous_cumulative = cumulative_counts - count
            fraction_within_bin = ((percentile_counts - previous_cumulative) / count) if count > 0 else 0
            return bin_edges[i] + fraction_within_bin * (bin_edges[i + 1] - bin_edges[i])

    # If we didn't find the percentile, return the maximum value
    return bin_edges[-1]


def bin_distribution_evenly(
    distance_metric,
    n_bins,
    max_distance=None,
    plot=False,
):
    """
    Splits the distribution into n_bins such that each bin has approximately equal total counts.
    The final bin edge will be capped by max_distance.

    Parameters:
        distance_metric (str): The name of the metric or distribution to load.
        n_bins (int): The number of bins to divide the distribution into.
        max_distance (float): The upper limit for the distance values.

    Returns:
        list: A list of bin edges that split the distribution into n_bins with equal counts.
    """
    # Load the distribution data
    data = _load_distribution(distance_metric)
    bin_edges = data["bin_edges"]
    counts = data["counts"]
    if max_distance is None:
        max_distance = max(bin_edges)
    if bin_edges[-1] > max_distance:
        bin_edges = [edge for edge in bin_edges if edge <= max_distance]
        counts = counts[: len(bin_edges) - 1]
    cumulative_counts = [0] + list(np.cumsum(counts))
    total_counts = cumulative_counts[-1]
    counts_per_bin = total_counts / n_bins
    new_bin_edges = [bin_edges[0]]
    current_cumulative_target = counts_per_bin
    for i in range(1, len(bin_edges)):
        while current_cumulative_target <= cumulative_counts[i]:
            fraction_within_bin = (current_cumulative_target - cumulative_counts[i - 1]) / (
                cumulative_counts[i] - cumulative_counts[i - 1]
            )
            new_bin_edge = bin_edges[i - 1] + fraction_within_bin * (bin_edges[i] - bin_edges[i - 1])
            if new_bin_edge > max_distance:
                new_bin_edge = max_distance
            if len(new_bin_edges) == n_bins and new_bin_edge != max_distance:
                new_bin_edge = max_distance

            new_bin_edges.append(new_bin_edge)
            current_cumulative_target += counts_per_bin
        if len(new_bin_edges) == n_bins + 1:
            break
    if len(new_bin_edges) < n_bins + 1:
        if new_bin_edges[-1] < max_distance:
            new_bin_edges.append(max_distance)
    if plot:
        f, ax = plt.subplots()
        _plot_hist(data, ax=ax, n_bins=False)
        ax.vlines(new_bin_edges, 0, max(counts), color="red")
        if max_distance:
            ax.set_xlim(0, max_distance)
    return new_bin_edges


# %% Save load data functions


def _load_distribution(distance_metric, plot=False, plot_n_bins=40):
    """ """
    save_path = ANALYSIS_INFO_PATH / f"{distance_metric[0]}-{distance_metric[1]}_distribution.json"
    with open(save_path, "r") as infile:
        data = json.load(infile)
    if plot:
        _plot_hist(data, n_bins=plot_n_bins, distance_metric=distance_metric)
    return data


def _plot_hist(data, n_bins=40, distance_metric=None, ax=None):
    """ """
    if ax is None:
        f, ax = plt.subplots()
    counts = data["counts"]
    bins = data["bin_edges"]
    if n_bins:
        n = int(len(bins) / n_bins)
        counts = [sum(counts[i : i + n]) for i in range(0, len(counts), n)]
        bins = bins[::n] + [bins[-1]]
    ax.stairs(counts, bins, fill=True)
    if distance_metric is not None:
        max_x = get_distance_percentile(distance_metric, percentile=0.99)
        ax.set_xlim(0, max_x)


def save_distance_distributions(
    distance_metrics=[
        ("distance_to_goal", "geodesic"),
        ("distance_to_goal", "euclidean"),
        ("distance_to_goal", "manhattan"),
        ("distance_to_goal", "future"),
        ("progress_to_goal", "time"),
        ("progress_to_goal", "path_length"),
        ("steps_to_goal", "geodesic"),
        ("steps_to_goal", "future"),
    ],
    subject_IDs="all",
    mazes=["maze_1", "maze_2"],
    days_on_maze="late",
):
    """
    Run this function to populate distance to goal distributions throughout navigation in
    analysis info data, so they can be quickly loaded later, for various analyses where they
    need to be queried. Eg, get 95th percentile of distance to goal for a given metric.
    """
    print("Loading data...")
    maze_sessions = gs.get_maze_sessions(
        subject_IDs=subject_IDs,
        maze_names=mazes,
        days_on_maze=days_on_maze,
        with_data=["navigation_df"],
        must_have_data=True,
    )
    combined_data = pd.concat([s.navigation_df for s in maze_sessions], axis=0)
    print("Saving distance distributions...")
    for distance_metric in distance_metrics:
        print(distance_metric)
        distances = combined_data[distance_metric].dropna().values
        counts, bin_edges = np.histogram(distances, bins=500)
        histogram_data = {  # adjust data types to be jsonable
            "counts": [int(c) for c in counts],
            "bin_edges": [float(b) for b in bin_edges],
        }
        # save to analysis info
        save_path = ANALYSIS_INFO_PATH / f"{distance_metric[0]}-{distance_metric[1]}_distribution.json"
        with open(save_path, "w") as outfile:
            outfile.write(json.dumps(histogram_data, indent=4))
    return
