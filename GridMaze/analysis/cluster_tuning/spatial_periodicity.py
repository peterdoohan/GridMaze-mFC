"""This file is for visualising the autocorreltation in firing rates between locations on a cells heatmap and the distance between those locations."""
# %% Imports
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from .. import get_sessions as gs
from ..processing import get_cluster_analysis_metrics as cam
from ...maze import representations as mr
from ...maze import plotting as mp


# %% Grobal Variables
FRAME_RATE = 60


# %% Functions


def get_maze_periodicity_df(session, max_distance=20):
    place_averaged_rates_df = get_place_averaged_rates_df(session)
    split_df1, split_df2 = get_place_averaged_rates_df(session, split_halves=True)
    cluster_unique_IDs = place_averaged_rates_df.columns.to_numpy()
    cluster_spatial_correlations = split_df1.corrwith(split_df2)
    simple_maze = session.simple_maze()
    distance_correlations_df = pd.DataFrame(
        columns=pd.MultiIndex.from_product([["distance_correlations"], np.arange(1, max_distance + 1)])
    )
    fit_params_df = pd.DataFrame(
        columns=pd.MultiIndex.from_product(
            [["fit_params"], ["exp_scale", "exp_length", "sin_scale", "freq", "phase", "offest", "r2"]]
        )
    )
    for cluster in cluster_unique_IDs:
        place2rate = place_averaged_rates_df[cluster].to_dict()
        maze_distance_correlations = get_maze_distance_correlations(place2rate, simple_maze, max_distance=max_distance)
        fit, fit_params = fit_oscilating_exponential_decay(
            maze_distance_correlations, return_fit=True, return_params=True
        )
        if not np.all(np.isnan(fit_params)):
            fit_params = np.append(fit_params, r2_score(maze_distance_correlations, fit))
        distance_correlations_df.loc[cluster] = maze_distance_correlations
        fit_params_df.loc[cluster] = fit_params
    maze_spatial_correlation_df = pd.concat([distance_correlations_df, fit_params_df], axis=1)
    maze_spatial_correlation_df[("spatial_correlation", "")] = cluster_spatial_correlations
    return maze_spatial_correlation_df


def get_place_averaged_rates_df(
    session, split_halves=False, navigation_only=True, moving_only=True, exclude_time_at_goal=True, minimum_occupancy=1
):
    if not gs.check_session_has_data(session, ["navigation_df", "navigation_spike_rates_df", "cluster_metrics"]):
        pass
    simple_maze = session.simple_maze()
    navigation_rates_df = session.get_navigation_activity_df(activity_type="firing_rate", cluster_type="good")
    cluster_unique_IDs = navigation_rates_df.firing_rate.columns.to_numpy()
    if navigation_only:
        navigation_rates_df = navigation_rates_df[navigation_rates_df.trial_phase == "navigation"]
    if moving_only:
        navigation_rates_df = navigation_rates_df[navigation_rates_df.moving]
    if exclude_time_at_goal:
        navigation_rates_df = navigation_rates_df[navigation_rates_df.maze_position.simple != navigation_rates_df.goal]
    if split_halves:
        place_averaged_rates_dfs = []
        for trials in cam.split_trials(navigation_rates_df):
            split_navigation_rates_df = navigation_rates_df[navigation_rates_df.trial.isin(trials)]
            place_averaged_rates_df = _get_place_averaged_rates_df(
                split_navigation_rates_df, simple_maze, minimum_occupancy, cluster_unique_IDs
            )
            place_averaged_rates_dfs.append(place_averaged_rates_df)
        return tuple(place_averaged_rates_dfs)

    else:
        place_averaged_rates_df = _get_place_averaged_rates_df(
            navigation_rates_df, simple_maze, minimum_occupancy, cluster_unique_IDs
        )
        return place_averaged_rates_df


def _get_place_averaged_rates_df(navigation_rates_df, simple_maze, minimum_occupancy, cluster_unique_IDs):
    place_direction_grouped_df = navigation_rates_df.set_index([("maze_position", "simple")]).groupby(
        [("maze_position", "simple")]
    )
    place_averaged_rates_df = place_direction_grouped_df.mean().firing_rate
    place_averaged_rates_df[place_direction_grouped_df.count().time < minimum_occupancy * FRAME_RATE] = np.nan
    all_places = mr.get_maze_locations(simple_maze)
    unvisited_places = list(set(all_places) - set(place_averaged_rates_df.index))
    place_averaged_rates_df = pd.concat(
        [place_averaged_rates_df, pd.DataFrame(index=unvisited_places, columns=cluster_unique_IDs, data=np.nan)]
    )
    place_averaged_rates_df = place_averaged_rates_df.reindex(sorted(place_averaged_rates_df.index))
    return place_averaged_rates_df


def plot_session_spatial_periodicity_summary(session):
    simple_maze = session.simple_maze()
    place_averaged_rates_df = get_place_averaged_rates_df(session)
    split_df1, split_df2 = get_place_averaged_rates_df(session, split_halves=True)
    cluster_unique_IDs = place_averaged_rates_df.columns.to_numpy()
    for cluster in cluster_unique_IDs:
        split_halves_corr = split_df1[cluster].corr(split_df2[cluster])
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), clear=True)
        f.subplots_adjust(wspace=0.3)
        place2rate = place_averaged_rates_df[cluster].to_dict()
        mp.plot_simple_heatmap(
            simple_maze, place2rate, ax1, title=cluster, value_label="Firing Rate (Hz)", node_size=400
        )
        plot_maze_spatial_correlation(place2rate, simple_maze, ax2, color="black", lw=2, alpha=1)
        spatial_periodicity = get_maze_distance_correlations(place2rate, simple_maze)
        try:
            fit, params = fit_oscilating_exponential_decay(spatial_periodicity, return_params=True)
            if np.all(np.isnan(fit)):
                continue
            exp_amp, exp_ls, sin_amp, freq, phase, offset = params
            r2 = r2_score(spatial_periodicity, fit)
            ax2.plot(range(1, len(spatial_periodicity) + 1), fit, color="red", lw=2, alpha=1)
            ax2.text(
                0.95,
                1.0,
                f"r2 = {r2:.2f}\n A = {sin_amp:.2f}\n f = {freq:.2f}",
                transform=ax2.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=12,
                color="black",
            )
        except RuntimeError:
            pass
        except ValueError:
            pass
        ax1.text(
            0.05,
            -0.05,
            f"r = {split_halves_corr:.2f}",
            transform=ax1.transAxes,
            verticalalignment="bottom",
            horizontalalignment="left",
            fontsize=12,
            color="black",
        )
    return


# %% Spatial Correlation Functions


def get_maze_distance_correlations(place2rate, simple_maze, max_distance=20):
    extended_simple_maze = mr.get_extended_simple_maze(simple_maze)
    if max_distance is None:
        max_distance = get_max_maze_distance(extended_simple_maze)
    distances = np.arange(1, max_distance + 1)
    correlation_by_distance = []
    for d in distances:
        node_pairs = node_pairs_n_edges_apart(extended_simple_maze, d)
        node_pair_df = pd.DataFrame(node_pairs, columns=["node1", "node2"])
        node_pair_df["rate1"] = node_pair_df.node1.map(place2rate)
        node_pair_df["rate2"] = node_pair_df.node2.map(place2rate)
        correlation = node_pair_df.rate1.corr(node_pair_df.rate2)
        correlation_by_distance.append(correlation)
    return correlation_by_distance


def node_pairs_n_edges_apart(graph, x):
    """
    Returns pairs of nodes in the graph that are x edges apart.
    """
    seen = set()
    result = []
    for node in graph.nodes():
        lengths = nx.single_source_shortest_path_length(graph, node)
        nodes_x_away = [n for n, length in lengths.items() if length == x]
        for n in nodes_x_away:
            if (node, n) not in seen and (n, node) not in seen:
                result.append((node, n))
                seen.add((node, n))
    # convert results for coords to labels
    if len(nx.get_node_attributes(graph, "label")) == 0:
        raise ValueError("Graph does not have node labels")
    coord2label = nx.get_node_attributes(graph, "label")
    l_result = [(coord2label[n1], coord2label[n2]) for n1, n2 in result]
    return l_result


def get_max_maze_distance(graph):
    """
    Returns the maximum distance between any two nodes in the graph.
    """
    all_lengths = nx.all_pairs_shortest_path_length(graph)
    max_len = 0
    for source, lengths in all_lengths:
        max_len = max(max_len, max(lengths.values()))
    return max_len


def plot_maze_spatial_correlation(place2rate, simple_maze, ax, color="black", alpha=1, lw=2):
    correlation_by_distance = get_maze_distance_correlations(place2rate, simple_maze, max_distance=20)
    distances = np.arange(1, len(correlation_by_distance) + 1)
    ax.plot(distances, correlation_by_distance, color=color, lw=lw, alpha=alpha)
    ax.set_xlabel("Maze Distance")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="silver", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return


# %% Curve fitting functions


def fit_oscilating_exponential_decay(
    spatial_periodicity_curve, return_fit=True, return_params=False, max_itter=100, top_fit_tol=1e-4, plot=False
):
    x = np.array(range(1, len(spatial_periodicity_curve) + 1))
    lower_bounds = [0, 0, 0, 0, 0, -1]
    upper_bounds = [2, 5, 1, 4, 2 * np.pi, 1]

    top_fits = [{"fit": None, "params": None, "residuals": float("inf")} for _ in range(3)]

    itter_count = 0

    while itter_count < max_itter:
        # Sample initial values from the bounds
        p0 = [np.random.uniform(low, high) for low, high in zip(lower_bounds, upper_bounds)]

        try:
            params, _ = curve_fit(
                oscilatting_exponential_decay, x, spatial_periodicity_curve, p0=p0, bounds=(lower_bounds, upper_bounds)
            )
            fit = oscilatting_exponential_decay(x, *params)

            # Calculate the sum of square residuals
            residuals = np.sum((spatial_periodicity_curve - fit) ** 2)

            # Check if this fit is better than any of the top 3
            max_residual_in_top = max([f["residuals"] for f in top_fits])
            if residuals < max_residual_in_top:
                worst_top_fit_index = np.argmax([f["residuals"] for f in top_fits])
                top_fits[worst_top_fit_index] = {"fit": fit, "params": params, "residuals": residuals}

            # Check convergence criterion
            top_residuals = [f["residuals"] for f in top_fits]
            if max(top_residuals) - min(top_residuals) < top_fit_tol:
                break

        except Exception as e:
            # Handle any curve_fit exceptions and continue to the next iteration
            pass
        itter_count += 1
    try:
        avg_fit = np.mean([f["fit"] for f in top_fits], axis=0)
        avg_params = np.mean([f["params"] for f in top_fits], axis=0)
    except TypeError:
        avg_fit = np.nan
        avg_params = np.nan
    if plot:
        f, ax = plt.subplots()
        ax.scatter(x, spatial_periodicity_curve, s=20, label="Data", color="blue")
        ax.plot(x, avg_fit, color="red", label="Fit")
        ax.legend()
        ax.set_xlabel("Maze Distance")
        ax.set_ylabel("Correlation")
        ax.set_ylim(-1, 1)
        ax.axhline(0, color="silver", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    if return_fit and return_params:
        return avg_fit, avg_params
    elif return_fit:
        return avg_fit
    elif return_params:
        return avg_params


def oscilatting_exponential_decay(x, A, B, C, D, phi, E):
    return A * np.exp(-B * x) + C * np.sin(D * x + phi) + E


# %% vectorised version of get maze distance correlations
