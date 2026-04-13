"""
Calculate various metrics derived from maze structure
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import networkx as nx
from GridMaze.maze import representations as mr
from scipy.spatial.distance import euclidean

# %% Global Variables


# %% Functions


def get_maze_RDM_df(simple_maze, metric, kwargs={}):
    if metric == "euclidean_distance":
        return _euclidean_distance_RDM(simple_maze, **kwargs)
    elif metric == "geodesic_distance":
        return _geodesic_distance_RDM(simple_maze, **kwargs)
    elif metric == "node_degree":
        return _get_node_degree_RDM(simple_maze, **kwargs)
    elif metric == "betweenness_centrality":
        return _betweenness_centrality_RDM(simple_maze, **kwargs)
    elif metric == "decision_point_distance":
        return _decision_point_distance_RDM(simple_maze, **kwargs)
    elif metric == "decision_point":
        return _decision_point_RDM(simple_maze, **kwargs)
    elif metric == "corner":
        return _corner_RDM(simple_maze)
    elif metric == "boundary_distance":
        return _boundary_distance_RDM(simple_maze, **kwargs)
    elif metric == "node_fitness":
        return _get_node_fitness_RDM(simple_maze, **kwargs)
    else:
        raise NotImplementedError(f"Unknown metric: {metric}")


def _euclidean_distance_RDM(simple_maze, norm="max"):
    """
    without norm, units = m
    """
    extended_maze = mr.get_extended_simple_maze(simple_maze)
    label2pos = mr.get_maze_label2position(extended_maze)
    labels = list(label2pos.keys())
    RDM_df = pd.DataFrame(index=labels, columns=labels)
    for label_1 in labels:
        for label_2 in labels:
            RDM_df.loc[label_1, label_2] = euclidean(label2pos[label_1], label2pos[label_2])
    if norm == "max":
        RDM_df = RDM_df / RDM_df.values.max()
    else:
        raise NotImplementedError()
    # sort index and columns
    RDM_df = RDM_df.sort_index().sort_index(axis=1)
    return RDM_df.astype(float)


def _geodesic_distance_RDM(simple_maze, norm="max"):
    """
    without norm, units = steps
    """
    extended_maze = mr.get_extended_simple_maze(simple_maze)
    coord2label = mr.get_maze_coord2label(simple_maze)
    labels = list(coord2label.values())
    all_pairs_path_dist = dict(nx.all_pairs_dijkstra_path_length(extended_maze))
    RDM_df = pd.DataFrame(index=labels, columns=labels)
    for coord_1, pair_lengths in all_pairs_path_dist.items():
        label_1 = coord2label[coord_1]
        for coord_2, length in pair_lengths.items():
            label_2 = coord2label[coord_2]
            RDM_df.loc[label_1, label_2] = length
    if norm == "max":
        RDM_df = RDM_df / RDM_df.values.max()
    else:
        raise NotImplementedError()
    # sort index and columns
    RDM_df = RDM_df.sort_index().sort_index(axis=1)
    return RDM_df.astype(float)


def _get_node_degree_RDM(simple_maze, norm="max"):
    """ """
    coord2label = mr.get_maze_coord2label(simple_maze)
    extended_maze = mr.get_extended_simple_maze(simple_maze)
    labels = list(coord2label.values())
    RDM_df = pd.DataFrame(index=labels, columns=labels)
    for coord_1 in extended_maze.nodes:
        label_1 = coord2label[coord_1]
        degree_1 = extended_maze.degree[coord_1]
        for coord_2 in extended_maze.nodes:
            label_2 = coord2label[coord_2]
            degree_2 = extended_maze.degree[coord_2]
            RDM_df.loc[label_1, label_2] = abs(degree_1 - degree_2)
    if norm == "max":
        RDM_df = RDM_df / RDM_df.values.max()
    else:
        raise NotImplementedError()
    # sort index and columns
    RDM_df = RDM_df.sort_index().sort_index(axis=1)
    return RDM_df.astype(float)


def _betweenness_centrality_RDM(simple_maze, norm="max"):
    """ """
    extended_maze = mr.get_extended_simple_maze(simple_maze)
    coord2label = mr.get_maze_coord2label(simple_maze)
    labels = list(coord2label.values())
    betweenness_centrality = nx.betweenness_centrality(extended_maze)
    RDM_df = pd.DataFrame(index=labels, columns=labels)
    for coord_1, bc_1 in betweenness_centrality.items():
        label_1 = coord2label[coord_1]
        for coord_2, bc_2 in betweenness_centrality.items():
            label_2 = coord2label[coord_2]
            RDM_df.loc[label_1, label_2] = abs(bc_1 - bc_2)
    # sort index and columns
    RDM_df = RDM_df.sort_index().sort_index(axis=1)
    if norm == "max":
        RDM_df = RDM_df / RDM_df.values.max()
    else:
        raise NotImplementedError()
    RDM_df = RDM_df.sort_index().sort_index(axis=1)
    return RDM_df.astype(float)


def _decision_point_distance_RDM(simple_maze, norm="max", decision_point_degrees=[3, 4]):
    """
    RDM based on geodesic distance to nearest decision point (node with degree in decision_point_degrees).
    """
    coord2label = mr.get_maze_coord2label(simple_maze)
    decision_points = [coord for coord in simple_maze.nodes if simple_maze.degree[coord] in decision_point_degrees]
    extended_maze = mr.get_extended_simple_maze(simple_maze)
    labels = list(coord2label.values())
    all_pairs_path_dist = dict(nx.all_pairs_dijkstra_path_length(extended_maze))
    RDM_df = pd.DataFrame(index=labels, columns=labels)
    for coord_1, pair_lengths in all_pairs_path_dist.items():
        label_1 = coord2label[coord_1]
        decision_point_distance_1 = min([all_pairs_path_dist[coord_1][dp] for dp in decision_points])
        for coord_2, length in pair_lengths.items():
            label_2 = coord2label[coord_2]
            decision_point_distance_2 = min([all_pairs_path_dist[coord_2][dp] for dp in decision_points])
            RDM_df.loc[label_1, label_2] = abs(decision_point_distance_1 - decision_point_distance_2)
    if norm == "max":
        RDM_df = RDM_df / RDM_df.values.max()
    else:
        raise NotImplementedError()
    # sort index and columns
    RDM_df = RDM_df.sort_index().sort_index(axis=1)
    return RDM_df.astype(float)


def _decision_point_RDM(simple_maze, decision_point_degrees=[3, 4]):
    """Binary RDM: 0 if both locations are decision points, 1 otherwise."""
    coord2label = mr.get_maze_coord2label(simple_maze)
    decision_point_labels = [
        coord2label[coord] for coord in simple_maze.nodes if simple_maze.degree[coord] in decision_point_degrees
    ]
    labels = list(coord2label.values())
    RDM_df = pd.DataFrame(index=labels, columns=labels)
    for label_1 in labels:
        for label_2 in labels:
            RDM_df.loc[label_1, label_2] = (
                0 if label_1 in decision_point_labels and label_2 in decision_point_labels else 1
            )
    return RDM_df.astype(float)


def _corner_RDM(simple_maze):
    """ """
    corner_nodes = [
        "A1",
        "A7",
        "G1",
        "G7",
    ]
    coord2label = mr.get_maze_coord2label(simple_maze)
    labels = list(coord2label.values())
    RDM_df = pd.DataFrame(index=labels, columns=labels)
    for label_1 in labels:
        for label_2 in labels:
            RDM_df.loc[label_1, label_2] = 0 if label_1 in corner_nodes and label_2 in corner_nodes else 1
    return RDM_df.astype(float)


def _boundary_distance_RDM(simple_maze, norm="max", maze_width=7):
    """ """
    label2coord = mr.get_maze_label2coord(simple_maze)
    labels = list(label2coord.keys())
    RDM_df = pd.DataFrame(index=labels, columns=labels)
    for label_1, coord_1 in label2coord.items():
        if "-" in label_1:  # edge
            coord_1 = np.array(coord_1).mean(axis=0)
        else:  # node
            coord_1 = np.array(coord_1)
        x_dist_1 = min(coord_1[0], maze_width - coord_1[0])
        y_dist_1 = min(coord_1[1], maze_width - coord_1[1])
        boundary_dist_1 = min(x_dist_1, y_dist_1)
        for label_2, coord_2 in label2coord.items():
            if "-" in label_2:  # edge
                coord_2 = np.array(coord_2).mean(axis=0)
            else:  # node
                coord_2 = np.array(coord_2)
            x_dist_2 = min(coord_2[0], maze_width - coord_2[0])
            y_dist_2 = min(coord_2[1], maze_width - coord_2[1])
            boundary_dist_2 = min(x_dist_2, y_dist_2)
            RDM_df.loc[label_1, label_2] = abs(boundary_dist_1 - boundary_dist_2)
    if norm == "max":
        RDM_df = RDM_df / RDM_df.values.max()
    else:
        raise NotImplementedError()
    # sort index and columns
    RDM_df = RDM_df.sort_index().sort_index(axis=1)
    return RDM_df.astype(float)


def _get_node_fitness_RDM(simple_maze, norm="max"):
    """ """
    coord2label = mr.get_maze_coord2label(simple_maze)
    extended_maze = mr.get_extended_simple_maze(simple_maze)
    labels = list(coord2label.values())
    all_pairs_path_dist = dict(nx.all_pairs_dijkstra_path_length(extended_maze))

    def _get_node_fitness(coord_1, label_1, pair_lengths):
        if "-" in label_1:  # edge
            coord_1 = np.array(coord_1).mean(axis=0)
        else:  # node
            coord_1 = np.array(coord_1)
        path_dist_1 = np.array(list(pair_lengths.values())) / 2
        eucl_dist_1 = []
        for coord_2 in pair_lengths.keys():
            if "-" in coord2label[coord_2]:  # edge
                coord_2 = np.array(coord_2).mean(axis=0)
            else:  # node
                coord_2 = np.array(coord_2)
            eucl_dist_1.append(euclidean(coord_1, coord_2))
        eucl_dist_1 = np.array(eucl_dist_1)
        fitness = 1 - np.corrcoef(path_dist_1, eucl_dist_1)[0, 1]
        return fitness

    RDM_df = pd.DataFrame(index=labels, columns=labels)
    for coord_1, pair_lengths in all_pairs_path_dist.items():
        label_1 = coord2label[coord_1]
        fitness_1 = _get_node_fitness(coord_1, label_1, pair_lengths)
        for coord_2, length in pair_lengths.items():
            label_2 = coord2label[coord_2]
            fitness_2 = _get_node_fitness(coord_2, label_2, all_pairs_path_dist[coord_2])
            RDM_df.loc[label_1, label_2] = abs(fitness_1 - fitness_2)
    if norm == "max":
        RDM_df = RDM_df / RDM_df.values.max()
    else:
        raise NotImplementedError()
    # sort index and columns
    RDM_df = RDM_df.sort_index().sort_index(axis=1)
    return RDM_df.astype(float)
