""" 
Library that generates the experiment level distributions of different distance to goal metrics,
and saves them to analysis info.
"""

# %% Imports
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from GridMaze.analysis.core import get_sessions as gs

# %% Global Variables
from GridMaze.paths import ANALYSIS_INFO_PATH


# %% Utility Functions


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
    max_distance,
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

    # Adjust the distribution to account for the max_distance
    if bin_edges[-1] > max_distance:
        # Remove counts beyond max_distance
        bin_edges = [edge for edge in bin_edges if edge <= max_distance]
        counts = counts[: len(bin_edges) - 1]

    # Compute the cumulative distribution up to max_distance
    cumulative_counts = [0] + list(np.cumsum(counts))
    total_counts = cumulative_counts[-1]

    # Target count per bin
    counts_per_bin = total_counts / n_bins

    # Calculate the bin edges for even splitting
    new_bin_edges = [bin_edges[0]]
    current_cumulative_target = counts_per_bin
    for i in range(1, len(bin_edges)):
        while current_cumulative_target <= cumulative_counts[i]:
            # Interpolate the exact edge for the current cumulative target
            fraction_within_bin = (current_cumulative_target - cumulative_counts[i - 1]) / (
                cumulative_counts[i] - cumulative_counts[i - 1]
            )
            new_bin_edge = bin_edges[i - 1] + fraction_within_bin * (bin_edges[i] - bin_edges[i - 1])

            # Ensure new bin edge does not exceed max_distance
            if new_bin_edge > max_distance:
                new_bin_edge = max_distance

            # If the last bin is being added, make sure to stop at max_distance
            if len(new_bin_edges) == n_bins and new_bin_edge != max_distance:
                new_bin_edge = max_distance

            new_bin_edges.append(new_bin_edge)
            current_cumulative_target += counts_per_bin

        # Stop if enough edges are found
        if len(new_bin_edges) == n_bins + 1:
            break

    # Ensure the last bin edge matches the original upper bound or max_distance
    if len(new_bin_edges) < n_bins + 1:
        # Add max_distance only if the last edge hasn't been set already
        if new_bin_edges[-1] < max_distance:
            new_bin_edges.append(max_distance)

    if plot:
        # Plot distribution with new bin edges marked
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
        # ("distance_to_goal", "geodesic"),
        # ("distance_to_goal", "euclidean"),
        # ("distance_to_goal", "manhattan"),
        # ("distance_to_goal", "future"),
        # ("progress_to_goal", "time"),
        # ("progress_to_goal", "path_length"),
        # ("steps_to_goal", "geodesic"),
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
