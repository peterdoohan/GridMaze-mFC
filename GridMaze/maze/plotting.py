"""This module contains functions for visualising maze representations."""

# %% Imports
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap

from . import representations as mr

from ..paths import EXPERIMENT_INFO_PATH

# %% Set gobal variables
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 12

with open(EXPERIMENT_INFO_PATH / "maze_configs.json", "r") as input_file:
    MAZE_CONFIGS = json.load(input_file)


# define custom colormaps
CUSTOM_COLORMAPS = {
    f"silver2{c}": LinearSegmentedColormap.from_list(f"SilverTo{c}", ["silver", c])
    for c in [
        "blue",
        "lime",
        "red",
        "firebrick",
        "darkorange",
        "olivedrab",
        "teal",
        "cyan",
        "fuchsia",
        "darkviolet",
        "yellow",
    ]
}
CUSTOM_COLORMAPS["heat"] = LinearSegmentedColormap.from_list(
    "heat", ["silver", "gold", "darkorange", "red", "firebrick"]
)
# %% Simple Heatmap plotting functions


def plot_simple_heatmap(
    simple_maze,
    place_values,
    ax=None,
    colormap="plasma",
    title="Simple Maze Heatmap",
    value_label="Value Label",
    highlight_nodes=False,
    highlight_color="deepskyblue",
    node_size=450,
    edge_size=10,
    allow_negative=False,
):
    """Plots a heatmap of the simple maze.
    INPUTS:
     - simple_maze: is a networkx object with specified nodes and edgees of each maze (see ./maze_representations.py) for details). Note that it is important
     that this network x object has node attributes 'position' and 'label' and edge attributes 'position' and 'label' for this function to work.
     - simple_label2value: is a dictionary of node and edge labels (same as label attributes in simple maze object) to values, with nans if there is no value associated
     with node/edge (which get plotted in grey)
    """
    # check place_values contains a value for every location:
    unvistied_locations = np.setdiff1d(mr.get_maze_locations(simple_maze), place_values.index)
    if len(unvistied_locations) > 0:
        for ul in unvistied_locations:
            place_values[ul] = np.nan
    label2coord = mr.get_maze_label2coord(simple_maze)
    place_vals = place_values.copy()
    place_vals.index = place_values.index.map(label2coord).to_flat_index()
    vmax = place_values.max()
    vmin = 0 if not allow_negative else place_values.min()
    node2color = {node: value2hex(place_vals[node], vmin, vmax, colormap=colormap) for node in simple_maze.nodes}
    edge2color = {edge: value2hex(place_vals[edge], vmin, vmax, colormap=colormap) for edge in simple_maze.edges}
    if not highlight_nodes:
        node_border_colors = [c for c in node2color.values()]
    else:  # highlighted specified nodes in gold
        node_label2node_coord = {v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()}
        highlight_nodes = [node_label2node_coord[node] for node in highlight_nodes]
        node_border_colors = [
            node2color[node] if node not in highlight_nodes else mcolors.to_hex(highlight_color)
            for node in simple_maze.nodes
        ]
    node_positions = nx.get_node_attributes(simple_maze, "position")
    # plotting
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_title(title, fontdict={"family": "Courier", "size": 10}, loc="left", pad=-5, x=0.05)
    ax.set_facecolor("none")
    ax.set_aspect("equal")
    ax.axis("off")
    nx.draw_networkx(
        simple_maze,
        pos=node_positions,
        ax=ax,
        node_color=node2color.values(),
        edgecolors=node_border_colors,
        linewidths=1.5,
        edge_color=edge2color.values(),
        node_size=node_size,
        node_shape="8",
        width=edge_size,
        with_labels=False,
    )
    if value_label:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.set_ylim(ax.get_ylim())
        cbar = plt.colorbar(get_colorbar(vmin, vmax, colormap), cax=cax)
        cbar.outline.set_visible(False)
        cbar.set_label(value_label, labelpad=10, fontsize=14)


def value2hex(value, vmin, vmax, colormap="cool"):
    if np.isnan(value):
        return "#d3d3d3"  # light gray
    if colormap not in CUSTOM_COLORMAPS.keys():
        cmap = plt.get_cmap(colormap)
    else:
        cmap = CUSTOM_COLORMAPS[colormap]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    color = sm.to_rgba(value)
    hex_color = mcolors.to_hex(color)
    return hex_color


def get_colorbar(vmin, vmax, colormap):
    if colormap not in CUSTOM_COLORMAPS.keys():
        cmap = plt.get_cmap(colormap)
    else:
        cmap = CUSTOM_COLORMAPS[colormap]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    return sm


# %% Directed Heatmap plotting functions


def plot_directed_heatmap(
    simple_maze,
    place_direction_values,
    ax=None,
    fixed_vmax=False,
    fixed_vmin=False,
    allow_negative=False,
    show_unvisitied_place_directions=False,
    colormap="plasma",
    colorbar=True,
    title="",
    value_label="",
    silhouette_color="silver",
    silhouette_node_size=500,
    silhouette_edge_size=10,
    star_base_length=0.045,
    max_point_length=0.03,
    highlight_nodes=False,
    highlight_color="lime",
):
    """Updated function so that NSEW values are normalised within function. will save alot of headache"""
    # check place_direction_values contains a value for every location:
    place_direction_values.index.names = ["maze_position", "direction"]  # ensure names are consistant for indexing
    unvistied_locations = np.setdiff1d(
        mr.get_maze_place_direction_pairs(simple_maze), list(place_direction_values.index)
    )
    if len(unvistied_locations) > 0:
        for ul in unvistied_locations:
            place_direction_values[ul] = np.nan
    # translate average location values to colors for heatmap
    pd_values = place_direction_values.copy()
    label2coord = mr.get_maze_label2coord(simple_maze)
    place_values = pd_values.groupby("maze_position").mean()
    place_df = pd.DataFrame(
        index=place_values.index.map(label2coord).to_flat_index(), columns=["value", "color", "position"]
    )
    place_df["value"] = place_values.values
    if fixed_vmin:
        vmin = fixed_vmin
    elif allow_negative:
        vmin = place_values.min()
    else:
        vmin = 0
    vmax = place_df.value.max() if not fixed_vmax else fixed_vmax
    place_df["color"] = place_df.value.apply(lambda x: value2hex(x, vmin, vmax, colormap=colormap))
    coord2position = {
        **nx.get_node_attributes(simple_maze, "position"),
        **nx.get_edge_attributes(simple_maze, "position"),
    }
    place_df["position"] = place_df.index.map(coord2position).to_list()
    norm_place_direction_values = normalise_place_direction_rates(
        pd_values, label2coord, allowe_negative=allow_negative
    )
    if show_unvisitied_place_directions:
        pd_unvisited = pd_values.isna()
        pd_unvisited.index = norm_place_direction_values.index  # convert to corrds
    # set axis
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # plot base maze silhoutte
    plot_simple_maze_silhouette(
        simple_maze,
        ax,
        color=silhouette_color,
        highlight_nodes=highlight_nodes,
        highlight_color=highlight_color,
        node_size=silhouette_node_size,
        edge_size=silhouette_edge_size,
    )
    # add star markers
    for place, row in place_df.iterrows():
        direction2norm_value = norm_place_direction_values.loc[place].to_dict()
        star_points = draw_directed_marker(
            ax, row.position, star_base_length, max_point_length, row.color, direction2norm_value, zorder=2
        )
        # add dot markers to indicate unvisited or low condifence place directions (if specified)
        if show_unvisitied_place_directions:
            direction2unvisited = pd_unvisited.loc[place].to_dict()
            plot_unvisited_marker(direction2unvisited, row.position, star_points, ax, zorder=3)

    # add colorbar
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(get_colorbar(vmin, vmax, colormap), cax=cax)
        cbar.outline.set_visible(False)
    if value_label and colorbar:
        cbar.set_label(value_label, labelpad=10, fontsize=14)
    # more plotting params
    if title:
        ax.set_title(title, fontdict={"family": "Courier", "size": 10}, loc="left", pad=-5, x=0.02)
    ax.set_facecolor("none")
    ax.set_aspect("equal")
    ax.axis("off")


def draw_directed_marker(ax, pos, side_length, max_point_length, color, NSEW, zorder=0):
    """
    Draw a star point on a given axis.

    Parameters:
    ax (matplotlib.axes.Axes): The axes to draw on.
    pos (tuple): The (x, y) position of the star point.
    side_length (float): The length of each side of the star's square.
    max_point_length (float): The maximum length of the star points.
    color (str): Color of the star.
    NSEW (dict): Dictionary with values for North, South, East, West extensions.
    zorder (int, optional): The z-order for layering of the plot. Default is 0.
    """

    # Calculate half side length and corner coordinates
    half_length = side_length / 2
    x, y = pos
    corners = {
        "NW": [x - half_length, y + half_length],
        "NE": [x + half_length, y + half_length],
        "SE": [x + half_length, y - half_length],
        "SW": [x - half_length, y - half_length],
    }

    # Draw a square if NSEW is not specified
    if NSEW is np.nan:
        square = patches.Rectangle(
            (corners["SW"][0], corners["SW"][1]), side_length, side_length, color=color, zorder=zorder
        )
        ax.add_patch(square)
        return

    # Initialize star points and adjust based on NSEW values
    star_points = {"N": corners["NW"], "S": corners["SE"], "E": corners["NE"], "W": corners["SW"]}

    for direction, value in NSEW.items():
        if np.isnan(value):
            value = 0
        scaled_length = max_point_length * value
        if direction == "N":
            star_points["N"] = [x, y + half_length + scaled_length]
        elif direction == "S":
            star_points["S"] = [x, y - half_length - scaled_length]
        elif direction == "E":
            star_points["E"] = [x + half_length + scaled_length, y]
        elif direction == "W":
            star_points["W"] = [x - half_length - scaled_length, y]

    # Define the star polygon and add it to the axis
    star_polygon = patches.Polygon(
        [
            corners["NW"],
            star_points["N"],
            corners["NE"],
            star_points["E"],
            corners["SE"],
            star_points["S"],
            corners["SW"],
            star_points["W"],
        ],
        facecolor=color,
        zorder=zorder,
    )
    ax.add_patch(star_polygon)
    return star_points


def plot_unvisited_marker(direction2unvisited, pos, star_points, ax, invalid_radius=0.005, zorder=0):
    """Takes a dictionary of directions to unvisited (True/False) and plots a circle marker at the direction's position if unvisited is True"""
    for direction in direction2unvisited:
        if direction2unvisited[direction]:
            marker_pos = [(star_points[direction][0] + pos[0]) / 2, (star_points[direction][1] + pos[1]) / 2]
            ax.add_patch(patches.Circle(marker_pos, invalid_radius, color="silver", zorder=zorder))
    return


def normalise_place_direction_rates(place_direction_values, label2coord, replace_nans=True, allowe_negative=False):
    """Normalises direction values at a location to sum to 1 for start heatmap plotting"""
    # change position labels for position coords
    norm_place_direction_values = place_direction_values.reset_index()
    location_coords = [label2coord[loc] for loc in norm_place_direction_values.maze_position]
    norm_place_direction_values.maze_position = location_coords
    norm_place_direction_values = norm_place_direction_values.set_index(["maze_position", "direction"]).squeeze()
    for loc in location_coords:
        loc_values = norm_place_direction_values.loc[loc]
        if allowe_negative:
            loc_values = loc_values.abs()
        values_sum = loc_values.sum()
        for dir, value in loc_values.items():
            if values_sum == 0 or (replace_nans and np.isnan(value)):
                norm_place_direction_values[(loc, dir)] = 0
            else:
                norm_place_direction_values[(loc, dir)] = value / values_sum
    return norm_place_direction_values.squeeze()  # make pd.Series


def plot_simple_maze_silhouette(
    simple_maze,
    ax,
    color,
    highlight_nodes=False,
    highlight_color="deepskyblue",
    special_location2color=None,
    node_size=500,
    edge_size=10,
):
    """"""
    node_coord2node_label = nx.get_node_attributes(simple_maze, "label")
    node_label2node_coord = {v: k for k, v in node_coord2node_label.items()}
    edge_cord2edge_label = nx.get_edge_attributes(simple_maze, "label")

    # Distinguish between special nodes and edges based on the key format
    special_node2color = (
        {k: v for k, v in special_location2color.items() if "-" not in k} if special_location2color else None
    )
    special_edge2color = (
        {k: v for k, v in special_location2color.items() if "-" in k} if special_location2color else None
    )

    if not special_node2color:
        node_colors = [color] * len(simple_maze.nodes)
        node_border_colors = node_colors
    else:
        node_colors = [
            mcolors.to_hex(special_node2color[node]) if node in special_node2color.keys() else mcolors.to_hex(color)
            for node in [node_coord2node_label[node] for node in simple_maze.nodes]
        ]
        node_border_colors = node_colors

    if highlight_nodes:
        highlight_nodes = [node_label2node_coord[node] for node in highlight_nodes]
        node_border_colors = [
            mcolors.to_hex(highlight_color) if node in highlight_nodes else node_border_colors[i]
            for i, node in enumerate(simple_maze.nodes)
        ]

    if not special_edge2color:
        edge_colors = [color] * len(simple_maze.edges)
    else:
        edge_colors = [
            mcolors.to_hex(special_edge2color[edge]) if edge in special_edge2color.keys() else mcolors.to_hex(color)
            for edge in [edge_cord2edge_label[edge] for edge in simple_maze.edges]
        ]

    node_positions = nx.get_node_attributes(simple_maze, "position")
    nx.draw_networkx(
        simple_maze,
        pos=node_positions,
        ax=ax,
        node_color=node_colors,
        edgecolors=node_border_colors,
        edge_color=edge_colors,
        node_size=node_size,
        node_shape="8",
        linewidths=1.5,
        width=edge_size,
        with_labels=False,
    )
    ax.set_facecolor("none")
    ax.set_aspect("equal")
    ax.axis("off")


# %% Vector heatmap plotting functions (want to update to new pd.Series input format)


def plot_simple_vector_heatmap(
    simple_maze,
    location2value,
    location2vector,
    ax,
    colormap="plasma",
    title="Simple Maze Vector Heatmap",
    value_label="Value Label",
    silhouette_color="silver",
    vector_point_radius=0.03,
    highlight_nodes=False,
    highlight_color="lime",
):
    """Plots a vector heatmap for of a simple maze, with a silhoutte of the maze structure in the background and colored tear-drops
    that indicate a value, vector magnitude and vector direction for each node/edge.
    INPUTS:
        - simple_maze: is a networkx object with specified nodes and edgees of each maze (see ./maze_representations.py) for details). Note that it is important
        that this network x object has node attributes 'position' and 'label' and edge attributes 'position' and 'label' for this function to work.
        - simple_label2value: is a dictionary of node and edge labels (same as label attributes in simple maze object) to values, with nans if there is no value associated
        with node/edge (which get plotted in grey)
        - simple_label2vector: is a dictionary of node and edge labels (same as label attributes in simple maze object) to normalised vectors (magnitude between 0 and 1),
        with nans if there is no vector associated.
    """
    node2value, edge2value = get_coords2value(simple_maze, location2value)
    node2vector, edge2vector = get_coords2value(simple_maze, location2vector)
    vmax = max(location2value.values())
    vmin = 0
    node2color = {node: value2hex(node2value[node], vmin, vmax, colormap=colormap) for node in simple_maze.nodes}
    edge2color = {edge: value2hex(edge2value[edge], vmin, vmax, colormap=colormap) for edge in simple_maze.edges}
    node_positions = nx.get_node_attributes(simple_maze, "position")
    edge_positions = nx.get_edge_attributes(simple_maze, "position")
    # plotting
    ax.set_title(title, fontdict={"family": "Courier", "size": 10}, loc="left", pad=-5, x=0.05)
    ax.set_facecolor("none")
    ax.set_aspect("equal")
    ax.axis("off")

    plot_simple_maze_silhouette(
        simple_maze, ax, color=silhouette_color, highlight_nodes=highlight_nodes, highlight_color=highlight_color
    )
    for node in node2value.keys():
        vector = node2vector[node]
        position = node_positions[node]
        color = node2color[node]
        plot_vector_point(ax, position, vector_point_radius, color, vector, zorder=2)
    for edge in edge2value.keys():
        vector = edge2vector[edge]
        position = edge_positions[edge]
        color = edge2color[edge]
        plot_vector_point(ax, position, vector_point_radius, color, vector, zorder=3)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(get_colorbar(vmin, vmax, colormap), cax=cax)
    cbar.outline.set_visible(False)
    cbar.set_label(value_label, labelpad=10, fontsize=14)


def plot_vector_point(ax, pos, radius, color, vector, zorder):
    color = mcolors.to_rgba(color)
    circle = plt.Circle(pos, radius, color=color, zorder=zorder)
    ax.add_artist(circle)
    if not np.any(np.isnan(vector)):  # don't plot direction if vector is nan
        v_magnitude = np.linalg.norm(vector)
        v_angle = np.arctan2(*vector)
        triangle_verticies = get_triangle_vertices(pos, radius, v_magnitude, v_angle)
        triangle = patches.Polygon(triangle_verticies, closed=True, color=color, zorder=zorder + 1)
        ax.add_patch(triangle)
    return


def get_triangle_vertices(center, radius, magnitude, angle):
    extension = radius * magnitude if magnitude > 0 else 0
    tangent_angle = np.arcsin(radius / (radius + extension))
    tip_x = center[0]
    tip_y = center[1] + radius + extension
    base_left_x = center[0] - radius * np.cos(tangent_angle)
    base_left_y = center[1] + radius * np.sin(tangent_angle)
    base_right_x = center[0] + radius * np.cos(tangent_angle)
    base_right_y = center[1] + radius * np.sin(tangent_angle)
    # Rotate (clockwise) each vertex around the circle center by the given angle
    tip_x, tip_y = rotate_point_around_center(tip_x, tip_y, center[0], center[1], -angle)
    base_left_x, base_left_y = rotate_point_around_center(base_left_x, base_left_y, center[0], center[1], -angle)
    base_right_x, base_right_y = rotate_point_around_center(base_right_x, base_right_y, center[0], center[1], -angle)
    vertices = np.array([[tip_x, tip_y], [base_left_x, base_left_y], [base_right_x, base_right_y]])
    return vertices


def rotate_point_around_center(x, y, x_center, y_center, angle):
    """Rotate a point around a center by a given angle in radians"""
    x_translated = x - x_center  # Translate the point so that the center becomes the origin
    y_translated = y - y_center
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    input_coords = np.array([x_translated, y_translated])
    x_rotated, y_rotated = np.dot(rotation_matrix, input_coords)
    x_rotated += x_center  # Translate the rotated point back to its original position
    y_rotated += y_center
    return x_rotated, y_rotated


def get_coords2value(maze, nodes_and_edges2value):
    node_label2value = {k: v for k, v in nodes_and_edges2value.items() if len(k.split("-")) == 1}
    edge_label2value = {k: v for k, v in nodes_and_edges2value.items() if len(k.split("-")) == 2}
    node_label2coord = {v: k for k, v in nx.get_node_attributes(maze, "label").items()}
    edge_label2coord = {v: k for k, v in nx.get_edge_attributes(maze, "label").items()}
    node_coord2value = {node_label2coord[node]: value for node, value in node_label2value.items()}
    edge_coord2value = {edge_label2coord[edge]: value for edge, value in edge_label2value.items()}
    return node_coord2value, edge_coord2value


# %% Figure legend plotting


def plot_simple_maze_for_figure_legend(simple_maze, ax, goals, colormap, node_size=15, edge_size=1):
    node2position = nx.get_node_attributes(simple_maze, "position")
    node_labels = node_labels = np.array([i for i in nx.get_node_attributes(simple_maze, "label").values()])
    goal2standard_color = get_goal2standard_color(colormap)
    node2color = {
        node_label: goal2standard_color[node_label] if node_label in goals else "lightgrey"
        for node_label in node_labels
    }
    ax.set_facecolor("none")
    ax.set_aspect("equal")
    ax.axis("off")
    nx.draw_networkx(
        simple_maze,
        pos=node2position,
        ax=ax,
        node_color=node2color.values(),
        node_size=node_size,
        edge_color="lightgrey",
        width=edge_size,
        with_labels=False,
    )
    return


def get_goal2standard_color(colormap="gist_rainbow"):
    all_goals = MAZE_CONFIGS["maze_1"]["goal_sets"]["all"]
    cmap = cm.get_cmap(colormap, len(all_goals) + 2)
    color_list = [mcolors.rgb2hex(cmap(i)) for i in range(len(all_goals) + 2)][1:-1]
    return {goal: color for goal, color in zip(all_goals, color_list)}


# %% Other functions (some depreciated)


def get_sim_simple_label2value(simple_maze):
    node_coord2label = nx.get_node_attributes(simple_maze, "label")
    node_label2value = {node_coord2label[node]: np.random.randint(0, 100) for node in simple_maze.nodes}
    edge_coord2label = nx.get_edge_attributes(simple_maze, "label")
    edge_label2value = {edge_coord2label[edge]: np.random.randint(0, 100) for edge in simple_maze.edges}
    return {**node_label2value, **edge_label2value}


def get_simple_label2betweenness_centrality(simple_maze):
    node_coord2betweenness_centrality = nx.betweenness_centrality(simple_maze)
    edge_coord2betweenness_centrality = nx.edge_betweenness_centrality(simple_maze)
    node_coord2label = nx.get_node_attributes(simple_maze, "label")
    edge_coord2label = nx.get_edge_attributes(simple_maze, "label")
    node_label2betweenness_centrality = {
        node_coord2label[node]: value for node, value in node_coord2betweenness_centrality.items()
    }
    edge_label2betweenness_centrality = {
        edge_coord2label[edge]: value for edge, value in edge_coord2betweenness_centrality.items()
    }
    return {**node_label2betweenness_centrality, **edge_label2betweenness_centrality}


def generate_random_vector():
    angle = np.random.uniform(0, 2 * np.pi)  # Random angle between 0 and 2π
    magnitude = np.random.uniform(0, 1)  # Random magnitude between 0 and 1
    x = magnitude * np.cos(angle)
    y = magnitude * np.sin(angle)
    return np.array([x, y])


def get_sim_location2NSEW(simple_maze):
    node_coord2label = nx.get_node_attributes(simple_maze, "label")
    edge_coord2label = nx.get_edge_attributes(simple_maze, "label")
    node2NSEW = {}
    for node in simple_maze.nodes:
        neighbors = list(simple_maze.neighbors(node))
        node_neighbors_NSEW = []
        for neighbor in neighbors:
            if neighbor[0] == node[0] + 1:
                node_neighbors_NSEW.append("E")
            elif neighbor[0] == node[0] - 1:
                node_neighbors_NSEW.append("W")
            elif neighbor[1] == node[1] + 1:
                node_neighbors_NSEW.append("N")
            elif neighbor[1] == node[1] - 1:
                node_neighbors_NSEW.append("S")
        node_NSEW = {
            direction: {"value": np.random.uniform(0, 1), "valid": np.random.choice([True, False])}
            for direction in node_neighbors_NSEW
        }
        node2NSEW[node_coord2label[node]] = node_NSEW
    edge2NSEW = {}
    for edge in simple_maze.edges:
        node1, node2 = edge
        edge_neighbors_NSEW = []
        if node1[0] == (node2[0] + 1) or node1[0] == (node2[0] - 1):
            edge_neighbors_NSEW.append("E")
            edge_neighbors_NSEW.append("W")
        elif node1[1] == (node2[1] + 1) or node1[1] == (node2[1] - 1):
            edge_neighbors_NSEW.append("N")
            edge_neighbors_NSEW.append("S")
        edge_NSEW = {
            direction: {"value": np.random.uniform(0, 1), "valid": np.random.choice([True, False])}
            for direction in edge_neighbors_NSEW
        }
        edge2NSEW[edge_coord2label[edge]] = edge_NSEW
    return {**node2NSEW, **edge2NSEW}


# %%
def plot_maze_series(axes=None):
    if axes is None:
        f, axes = plt.subplots(1, 3, figsize=(12, 4), clear=True)
    for i, maze_name in enumerate(MAZE_CONFIGS.keys()):
        simple_maze = mr.get_simple_maze(maze_name)
        plot_simple_maze_silhouette(
            simple_maze,
            axes[i],
            color="silver",
            edge_size=7.5,
            node_size=300,
        )
        axes[i].set_title(f"{maze_name}")


def plot_goal_subsets(
    all_goals_color="deepskyblue",
    subset_1_color="yellowgreen",
    subset_2_color="plum",
    axes=None,
):
    if axes is None:
        f, axes = plt.subplots(1, 2, figsize=(8, 4), clear=True)
    subset2goals = MAZE_CONFIGS["maze_1"]["goal_sets"]  # same for each maze
    simple_maze = mr.get_simple_maze("maze_1")
    # plot all goals
    plot_simple_maze_silhouette(
        simple_maze,
        axes[0],
        color="silver",
        edge_size=7.5,
        node_size=300,
        special_location2color={goal: all_goals_color for goal in subset2goals["all"]},
    )
    # plot subset 1 & 2
    plot_simple_maze_silhouette(
        simple_maze,
        axes[1],
        color="silver",
        edge_size=7.5,
        node_size=300,
        special_location2color={
            **{goal: subset_1_color for goal in subset2goals["subset_1"]},
            **{goal: subset_2_color for goal in subset2goals["subset_2"]},
        },
    )
    axes[0].set_title("All Goals")
    axes[1].set_title("Goal Subsets 1 & 2")
