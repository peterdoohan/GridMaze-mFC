"""
This module contains classes to build two networkx representations of a maze: simple and skeleton.
"""

# %% Imports
import json
import numpy as np
import networkx as nx

from ..paths import EXPERIMENT_INFO_PATH

# %% Global Variables

with open(EXPERIMENT_INFO_PATH / "maze_configs.json") as input_file:
    MAZE_CONFIGS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "maze_measurements.json") as input_file:
    MAZE_MEASURMENTS = json.load(input_file)

MAZE_NODE_DIMENSIONS = MAZE_MEASURMENTS["maze_node_dimensions"]
LOWER_LEFT = MAZE_MEASURMENTS["lower_left_node_cartesian_center"]  # meters
DISTNACE_BETWEEN_TOWERS = MAZE_MEASURMENTS["distance_between_node_centers"]  # meters
TOWER_WIDTH = MAZE_MEASURMENTS["tower_width"]  # meters

# %% Useful analysis functions


def get_extended_simple_maze(simple_maze):
    """"""
    edge_graph = nx.line_graph(simple_maze)
    extended_simple_maze = nx.Graph()
    extended_node2position = {
        **nx.get_node_attributes(simple_maze, "position"),
        **nx.get_edge_attributes(simple_maze, "position"),
    }
    extended_node2label = {
        **nx.get_node_attributes(simple_maze, "label"),
        **nx.get_edge_attributes(simple_maze, "label"),
    }
    extended_simple_maze.add_nodes_from(simple_maze.nodes)
    extended_simple_maze.add_nodes_from(edge_graph.nodes)
    nx.set_node_attributes(extended_simple_maze, extended_node2position, "position")
    nx.set_node_attributes(extended_simple_maze, extended_node2label, "label")
    for edge_node in edge_graph:
        extended_simple_maze.add_edge(edge_node[0], edge_node)
        extended_simple_maze.add_edge(edge_node, edge_node[1])
    # add edge labels
    extended_edge2label = {edge: tuple([extended_node2label[p] for p in edge]) for edge in extended_simple_maze.edges()}
    nx.set_edge_attributes(extended_simple_maze, extended_edge2label, "label")
    return extended_simple_maze


def get_maze_locations(simple_maze, edges_only=False, nodes_only=False):
    node_coord2label = nx.get_node_attributes(simple_maze, "label")
    edge_coord2label = nx.get_edge_attributes(simple_maze, "label")
    if nodes_only:
        return [node_coord2label[n] for n in simple_maze.nodes]
    if edges_only:
        return [edge_coord2label[e] for e in simple_maze.edges]
    return [node_coord2label[n] for n in simple_maze.nodes] + [edge_coord2label[e] for e in simple_maze.edges]


def get_maze_place_direction_pairs(simple_maze, nodes=True, edges=True):
    node_coord2label = nx.get_node_attributes(simple_maze, "label")
    edge_coord2label = nx.get_edge_attributes(simple_maze, "label")
    location_action_pairs = []
    if nodes:
        for node in simple_maze.nodes:
            neighbors = list(simple_maze.neighbors(node))
            for neighbor in neighbors:
                if neighbor[0] == node[0] + 1:
                    direction = "E"
                elif neighbor[0] == node[0] - 1:
                    direction = "W"
                elif neighbor[1] == node[1] + 1:
                    direction = "N"
                elif neighbor[1] == node[1] - 1:
                    direction = "S"
                location_action_pairs.append((node_coord2label[node], direction))
    if edges:
        for edge in simple_maze.edges:
            node1, node2 = edge
            if node1[0] == (node2[0] + 1) or node1[0] == (node2[0] - 1):
                directions = ["E", "W"]
            elif node1[1] == (node2[1] + 1) or node1[1] == (node2[1] - 1):
                directions = ["N", "S"]
            location_action_pairs.append((edge_coord2label[edge], directions[0]))
            location_action_pairs.append((edge_coord2label[edge], directions[1]))
    return location_action_pairs


def get_maze_location2NSEW(simple_maze):
    node_coord2label = nx.get_node_attributes(simple_maze, "label")
    edge_coord2label = nx.get_edge_attributes(simple_maze, "label")
    location2NSEW = {}
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
        location2NSEW[node_coord2label[node]] = node_neighbors_NSEW
    for edge in simple_maze.edges:
        node1, node2 = edge
        edge_neighbors_NSEW = []
        if node1[0] == (node2[0] + 1) or node1[0] == (node2[0] - 1):
            edge_neighbors_NSEW.append("E")
            edge_neighbors_NSEW.append("W")
        elif node1[1] == (node2[1] + 1) or node1[1] == (node2[1] - 1):
            edge_neighbors_NSEW.append("N")
            edge_neighbors_NSEW.append("S")
        location2NSEW[edge_coord2label[edge]] = edge_neighbors_NSEW
    return location2NSEW


def get_simple_maze(maze_name):
    edges = MAZE_CONFIGS[maze_name]["structure"]
    return simple_maze(edges)


def get_skeleton_maze(maze_name):
    edges = MAZE_CONFIGS[maze_name]["structure"]
    return skeleton_maze(edges)


def get_maze_label2coord(simple_maze):
    """
    Returns a dictionary of node and edge labels (same as label attributes in simple maze object) to coordinates
    (network x node and edge positions).
    """
    node_coord2label = get_maze_coord2label(simple_maze)
    return {v: k for k, v in node_coord2label.items()}


def get_maze_coord2label(simple_maze):
    """
    Returns a dictionary of node and edge coordinates (network x node and edge positions) to
    standard alpha-neumeric labels
    """
    node_coord2label = nx.get_node_attributes(simple_maze, "label")
    edge_coord2label = nx.get_edge_attributes(simple_maze, "label")
    return {**node_coord2label, **edge_coord2label}


def get_maze_label2position(simple_maze):
    label2coord = {v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()}
    coord2pos = nx.get_node_attributes(simple_maze, "position")
    label2pos = {k: coord2pos[v] for k, v in label2coord.items()}
    return label2pos


# %%


def simple_maze(edges):
    """
    Creates a networkx graph representation of a maze, where nodes are towers, and edges are walkways
    Input: edges (list of str): list of edges in the maze, in the format 'A1-A2'
    Output: maze (networkx graph): graph representation of the maze
    """
    maze = nx.Graph()
    maze.add_nodes_from(nx.grid_2d_graph(*MAZE_NODE_DIMENSIONS).nodes())
    edge_coords = get_edge_coords(edges)
    maze.add_weighted_edges_from([i + (DISTNACE_BETWEEN_TOWERS,) for i in edge_coords])
    nx.set_node_attributes(maze, _get_node_positions_dict(), "position")
    nx.set_node_attributes(maze, {v: k for k, v in get_simple_nodes_dict().items()}, "label")
    nx.set_edge_attributes(maze, {key: _get_center_edge_positions_dict()[key] for key in edge_coords}, "position")
    nx.set_edge_attributes(maze, {key: get_edge_coords2label_dict()[key] for key in edge_coords}, "label")
    return maze


def skeleton_maze(edges):
    """
    Creates a networkx graph representation of a maze, where towers are discritsed into 5 nodes, and walkways are
    discrised into 3 nodes
    Input: edges (list of str): list of edges in the maze, in the format 'A1-A2'
    Output: maze (networkx graph): graph representation of the maze
    Notes:
        - Tower nodes are represented by their discrete coordinates on the simple maze with an additional dimension
        representing their place in the tower: eg, (0,0,0) is the center node in the tower at (0,0). Explicitly:
        (.,.,0) = 'center', (.,.,1) = 'top-left', (.,.,2) = 'top-right', (.,.,3) = 'bottom-left',
        (.,.,4) = 'bottom-right'
        - Edge nodes are represented by their discrete edge coordinates on the simple maze with an additional dimension
        representing their place between adjacent nodes, this is separate for horizontal and vertical edges:
        - Horizontal edges: ((.,.),(.,.),0) = 'west', ((.,.),(.,.),1) = 'middle', (.,.),(.,.),2) = 'east'
        - Vertical edges: ((.,.),(.,.),0) = 'south', ((.,.),(.,.),1) = 'middle', (.,.),(.,.),2) = 'north'
        - Positions (x,y: cartesian coordinates) and labels are attributed to each node
    """
    skeleton_maze = nx.Graph()
    edges = get_edge_coords(edges)
    skeleton_maze_nodes = get_skeleton_maze_nodes(edges)
    skeleton_maze.add_nodes_from(skeleton_maze_nodes)
    skeleton_maze_weighted_edges = _get_skeleton_maze_weighted_edges(edges)
    skeleton_maze.add_weighted_edges_from(skeleton_maze_weighted_edges)
    skeleton_node2position = {
        k: v for k, v in get_skeleton_maze_node_positions_dict().items() if k in skeleton_maze_nodes
    }
    skeleton_node2label = {k: v for k, v in get_skeleton_maze_node_labels_dict().items() if k in skeleton_maze_nodes}
    nx.set_node_attributes(skeleton_maze, skeleton_node2position, "position")
    nx.set_node_attributes(skeleton_maze, skeleton_node2label, "label")
    return skeleton_maze


# %% Simple maze functions


def _get_letter_codes(n):
    """Returns a list of n capital letters of the alphabet"""
    return [chr(i) for i in range(65, 65 + n)]


def get_simple_nodes_dict():
    """Returns a dictionary of alphanumeric nodes ('A1') to simple maze node coordinates (0,0)"""
    nodes_letter2coord = {}
    letter_codes = _get_letter_codes(MAZE_NODE_DIMENSIONS[1])
    for i, l in enumerate(letter_codes):
        for j in np.arange(0, MAZE_NODE_DIMENSIONS[0]):
            letter_node = l + str(j + 1)
            coord_node = (i, j)
            nodes_letter2coord[letter_node] = coord_node
    return nodes_letter2coord


def convert_labels_to_coords(nodes):
    """converts a list of alphanumeric nodes ('A1') to simple maze coordinates (0,0)"""
    nodes_dict = get_simple_nodes_dict()
    nodes_coord = []
    for n in nodes:
        nodes_coord.append(nodes_dict[n])
    return nodes_coord


def get_edge_coords(edges):
    """converts a list of alphanumeric edges ('A1-A2') to simple maze coordinates ((0,0),(0,1))"""
    nodes_dict = get_simple_nodes_dict()
    edges_letter_tuple = [edge.split("-") for edge in edges]
    edge_coords = []
    for i, edge in enumerate(edges_letter_tuple):
        edge_coords.append((nodes_dict[edge[0]], nodes_dict[edge[1]]))
    return edge_coords


def get_edge_coords2label_dict():
    """Returns a dictionary of edge coordinates ((0,0),(0,1)) to alphanumeric edge labels ('A1-A2')"""
    all_edges = nx.grid_2d_graph(*MAZE_NODE_DIMENSIONS).edges()
    node_coord2label = {v: k for k, v in get_simple_nodes_dict().items()}
    edge_coords2label = {}
    for edge in all_edges:
        edge_coords2label[edge] = node_coord2label[edge[0]] + "-" + node_coord2label[edge[1]]
    return edge_coords2label


def _get_node_positions_dict():
    """Returns a dictionary of node positions to cartesian coordinates, defined as center of maze tower"""
    nodes_dict = get_simple_nodes_dict()
    discrete_nodes2cartesian_nodes = {}
    node_discrete_coords = nodes_dict.values()
    for discrete_x, discrete_y in node_discrete_coords:
        cartesian_x = discrete_x * DISTNACE_BETWEEN_TOWERS + LOWER_LEFT[0]
        cartesian_y = discrete_y * DISTNACE_BETWEEN_TOWERS + LOWER_LEFT[1]
        discrete_nodes2cartesian_nodes[(discrete_x, discrete_y)] = (cartesian_x, cartesian_y)
    return discrete_nodes2cartesian_nodes


def _split_edges_into_horizontal_vertical(edges):
    """Splits a list of edge coors into horizontal and vertical edges (separate list outputs as tuple)"""
    horizontal_edges = []
    vertical_edges = []
    for edge in edges:
        if edge[0][0] == edge[1][0]:
            vertical_edges.append(edge)
        elif edge[0][1] == edge[1][1]:
            horizontal_edges.append(edge)
    return horizontal_edges, vertical_edges


def _get_center_edge_positions_dict():
    """Returns a dictionary of edge positions to cartesian coordinates"""
    center_edge_positions = {}
    all_edges = nx.grid_2d_graph(*MAZE_NODE_DIMENSIONS).edges()
    node_positions = _get_node_positions_dict()
    # sort edge coords into horizontal and vertical edges
    horizontal_edges, vertical_edges = _split_edges_into_horizontal_vertical(all_edges)
    # get center positions of horizontal edges
    for edge in horizontal_edges:
        node1, node2 = edge
        x1, y1 = node_positions[node1]
        x2, y2 = node_positions[node2]
        x = (x1 + x2) / 2
        y = y1
        center_edge_positions[edge] = (x, y)
    # get center positions of vertical edges
    for edge in vertical_edges:
        node1, node2 = edge
        x1, y1 = node_positions[node1]
        x2, y2 = node_positions[node2]
        x = x1
        y = (y1 + y2) / 2
        center_edge_positions[edge] = (x, y)
    return center_edge_positions


# %% skeleton maze functions


def _get_skeleton_tower_nodes():
    """
    Returns a list of nodes for the skeleton maze that define a maze tower and cover the same space as a single node
    in the simple maze.
        - Coordinates are defined similar to the simple maze, but with an additional z coordinate to represent relative position
          from the center of the tower: (.,.,0) = 'center', (.,.,1) = 'top-left', (.,.,2) = 'top-right', (.,.,3) = 'bottom-left',
          (.,.,4) = 'bottom-right'
    """
    skeleton_tower_nodes = []
    for node in nx.grid_2d_graph(*MAZE_NODE_DIMENSIONS).nodes():
        tower_nodes = []
        for i in range(5):
            tower_nodes.append(node + (i,))
        skeleton_tower_nodes.extend(tower_nodes)
    return skeleton_tower_nodes


def _get_skeleton_bridge_nodes(edges):
    """
    Returns a list of nodes for the skeleton maze that define a maze bridge and cover the same space as a single edge in the
    simple maze. Coordinates are defined similar to the simple maze, but with an additional z coordinate to represent relative
    position from the center of the bridge, separately for horizontal and vertical bridges:
        - Horizontal edges: ((.,.),(.,.),0) = 'west', ((.,.),(.,.),1) = 'middle', (.,.),(.,.),2) = 'east'
        - Vertical edges: ((.,.),(.,.),0) = 'south', ((.,.),(.,.),1) = 'middle', (.,.),(.,.),2) = 'north'
    """
    skeleton_bridge_nodes = []
    for edge in edges:
        bridge_nodes = []
        for i in range(3):
            bridge_nodes.append(edge + (i,))
        skeleton_bridge_nodes.extend(bridge_nodes)
    return skeleton_bridge_nodes


def get_skeleton_maze_nodes(edges):
    """Generates all the nodes for the skeleton maze from a list of edge coordinates"""
    skeleton_tower_nodes = _get_skeleton_tower_nodes()
    skeleton_bridge_nodes = _get_skeleton_bridge_nodes(edges)
    skeleton_maze_nodes = skeleton_tower_nodes + skeleton_bridge_nodes
    return skeleton_maze_nodes


def _get_skeleton_tower_edges():
    """Makes a list of edges on the nodes in the tower postions on the skeleton maze. In a tower, the center node is connected
    to the outer nodes in a diagonal pattern, and the outer nodes are connected to each other in a square pattern."""
    simple_maze_nodes = nx.grid_2d_graph(*MAZE_NODE_DIMENSIONS).nodes()
    skeleton_tower_edges = []
    for node in simple_maze_nodes:
        tower_edges = []
        # define diagonal connections:
        for i in np.arange(1, 5):
            center_node = node + (0,)
            outer_node = node + (i,)
            edge = (center_node, outer_node)
            tower_edges.append(edge)
        # define outer connections:
        for i, j in [(1, 2), (3, 4), (3, 1), (4, 2)]:
            tower_edges.append((node + (i,), node + (j,)))
        skeleton_tower_edges.extend(tower_edges)
    return skeleton_tower_edges


def _get_skeleton_bridge_edges(edges):
    """Makes a list of edges on the nodes in the bridge postions on the skeleton maze, defined differently for horizontal and
    vertical bridges. Horizontal node 0s are connected on the the 2nd and 4th nodes of towers to the west and horizontal node
    2s are connected to the 1st and 3rd towers to the east. The middle horixontal node (1) is connected to horizontal nodes 0
    and 2. The same logical applies for connections that make up vertical edges on the maze."""
    skeleton_bridge_edges = []
    horizontal_edges, vertical_edges = _split_edges_into_horizontal_vertical(edges)
    bridge_edges = []
    for edge in horizontal_edges:
        west_tower = edge[0]
        east_tower = edge[1]
        # west edges
        bridge_edges.append((west_tower + (0,), edge + (0,)))
        bridge_edges.append((west_tower + (2,), edge + (0,)))
        bridge_edges.append((west_tower + (4,), edge + (0,)))
        # middle edges
        bridge_edges.append((edge + (0,), edge + (1,)))
        bridge_edges.append((edge + (1,), edge + (2,)))
        # east edges
        bridge_edges.append((edge + (2,), east_tower + (0,)))
        bridge_edges.append((edge + (2,), east_tower + (1,)))
        bridge_edges.append((edge + (2,), east_tower + (3,)))
    skeleton_bridge_edges.extend(bridge_edges)
    bridge_edges = []
    for edge in vertical_edges:
        south_tower = edge[0]
        north_tower = edge[1]
        # south edges
        bridge_edges.append((south_tower + (0,), edge + (0,)))
        bridge_edges.append((south_tower + (1,), edge + (0,)))
        bridge_edges.append((south_tower + (2,), edge + (0,)))
        # middle edges
        bridge_edges.append((edge + (0,), edge + (1,)))
        bridge_edges.append((edge + (1,), edge + (2,)))
        # north edges
        bridge_edges.append((edge + (2,), north_tower + (0,)))
        bridge_edges.append((edge + (2,), north_tower + (3,)))
        bridge_edges.append((edge + (2,), north_tower + (4,)))
    skeleton_bridge_edges.extend(bridge_edges)
    return skeleton_bridge_edges


def get_skeleton_maze_edges(edges):
    """Generates all the edges for the skeleton maze from a list of edge coordinates"""
    skeleton_tower_edges = _get_skeleton_tower_edges()
    skeleton_bridge_edges = _get_skeleton_bridge_edges(edges)
    skeleton_maze_edges = skeleton_tower_edges + skeleton_bridge_edges
    return skeleton_maze_edges


# %%
def tower_apothem():
    """Calculates the apothem of a regular octagon given TOWER_WIDTH"""
    # see https://en.wikipedia.org/wiki/Apothem
    return TOWER_WIDTH / 2


def _get_skeleton_tower_positions_dict():
    """Makes a dictionary of the positions of the nodes in the tower postions on the skeleton maze. In a tower, the center node
    lies at the center of the tower, and the outer nodes lie on the corners of a square inscribed with the the octagonal towers.
    """
    apothem = tower_apothem()  # define distance beween center and outer tower nodes as apothem
    skeleton_tower_positions = {}
    tower2position = _get_node_positions_dict()
    for node in tower2position.keys():
        x, y = tower2position[node]
        skeleton_tower_positions[node + (0,)] = (x, y)
        skeleton_tower_positions[node + (1,)] = ((x - apothem / np.sqrt(2)), y + apothem / np.sqrt(2))
        skeleton_tower_positions[node + (2,)] = ((x + apothem / np.sqrt(2)), y + apothem / np.sqrt(2))
        skeleton_tower_positions[node + (3,)] = ((x - apothem / np.sqrt(2)), y - apothem / np.sqrt(2))
        skeleton_tower_positions[node + (4,)] = ((x + apothem / np.sqrt(2)), y - apothem / np.sqrt(2))
    return skeleton_tower_positions


def _get_skeleton_bridge_positions_dict():
    """Makes a dictionary of the positions of the nodes in the bridge postions on the skeleton maze, defined differently for horizontal and
    vertical bridges. North, south, east and west bridge nodes lie on the edge of the octagonal towers, and the middle bridge nodes lie
    half way between the center of adjacent towers."""
    skeleton_bridge_positions = {}
    bridge_node_spacing = DISTNACE_BETWEEN_TOWERS / 2 - tower_apothem()
    all_edges = nx.grid_2d_graph(*MAZE_NODE_DIMENSIONS).edges()
    horizontal_edges, vertical_edges = _split_edges_into_horizontal_vertical(all_edges)
    edge2position = _get_center_edge_positions_dict()
    for edge in horizontal_edges:
        x, y = edge2position[edge]
        skeleton_bridge_positions[edge + (0,)] = (x - bridge_node_spacing, y)
        skeleton_bridge_positions[edge + (1,)] = (x, y)
        skeleton_bridge_positions[edge + (2,)] = (x + bridge_node_spacing, y)
    for edge in vertical_edges:
        x, y = edge2position[edge]
        skeleton_bridge_positions[edge + (0,)] = (x, y - bridge_node_spacing)
        skeleton_bridge_positions[edge + (1,)] = (x, y)
        skeleton_bridge_positions[edge + (2,)] = (x, y + bridge_node_spacing)
    return skeleton_bridge_positions


def get_skeleton_maze_node_positions_dict():
    """Makes a dictionary of the positions of the nodes in the skeleton maze."""
    skeleton_tower_positions = _get_skeleton_tower_positions_dict()
    skeleton_bridge_positions = _get_skeleton_bridge_positions_dict()
    skeleton_maze_positions = {**skeleton_tower_positions, **skeleton_bridge_positions}
    return skeleton_maze_positions


def _get_skeleton_edge_distances_dict():
    """Makes a dictionary of the distances between the nodes in the skeleton maze."""
    # define different distances between skeleton nodes
    apothem = tower_apothem()
    tower_center_edge_distance = apothem
    bridge_bridge_distance = DISTNACE_BETWEEN_TOWERS / 2 - apothem
    tower_outer_edge_distance = apothem * np.sqrt(2)
    tower_to_edge_distance = apothem * np.sqrt(2 - np.sqrt(2))  # using cosine rule
    # attribute distances to edges
    all_edges = nx.grid_2d_graph(*MAZE_NODE_DIMENSIONS).edges()
    skeleton_maze_edges = get_skeleton_maze_edges(all_edges)
    skeleton_edge_distances = {}
    for edge in skeleton_maze_edges:
        node1, node2 = edge
        if isinstance(node1[0], int) and isinstance(node2[0], int):  # defines tower edges
            if node1[-1] == 0 or node2[-1] == 0:  # define inner tower edge
                skeleton_edge_distances[edge] = tower_center_edge_distance
            else:  # define outer tower edge
                skeleton_edge_distances[edge] = tower_outer_edge_distance
        elif (isinstance(node1[0], int) and isinstance(node2[0], tuple)) or (
            isinstance(node1[0], tuple) and isinstance(node2[0], int)
        ):  # define tower to bridge edges
            if ((node1[-1] == 0 or node1[-1] == 2) and node2[-1] == 0) or (
                node1[-1] == 0 and (node2[-1] == 0 or node2[-1] == 2)
            ):  # define bridge edge nodes to tower center nodes
                skeleton_edge_distances[edge] = tower_center_edge_distance
            else:
                skeleton_edge_distances[edge] = tower_to_edge_distance
        elif isinstance(node1[0], tuple) and isinstance(node2[0], tuple):  # define bridge edges
            skeleton_edge_distances[edge] = bridge_bridge_distance
    return skeleton_edge_distances


def _get_skeleton_maze_weighted_edges(edges):
    """Makes a list of the weighted edges in the skeleton maze."""
    skeleton_edge2distance = _get_skeleton_edge_distances_dict()
    skeleton_maze_edges = get_skeleton_maze_edges(edges)
    weighted_skeleton_maze_edges = []
    for edge in skeleton_maze_edges:
        weighted_skeleton_maze_edges.append(edge + (skeleton_edge2distance[edge],))
    return weighted_skeleton_maze_edges


def _get_skeleton_maze_tower_labels_dict():
    """Makes a dictionary of the labels of the nodes in the skeleton maze. The labels are the alphaneumeric code of the base node ('A1'),
    followed by the subpoint label; either: 'center: C', 'north-west: NW', 'north-east: NE', 'south-west: SW' or 'south-east: SE'.
    """
    tower_node2label = {}
    node_subplot2label = {0: "C", 1: "NW", 2: "NE", 3: "SW", 4: "SE"}
    simple_node2alphaneumeric = {v: k for k, v in get_simple_nodes_dict().items()}
    for node in _get_skeleton_tower_nodes():
        base_node_code = simple_node2alphaneumeric[(node[0], node[1])]
        subnode_code = node_subplot2label[node[2]]
        tower_node2label[node] = base_node_code + "_" + subnode_code
    return tower_node2label


def _get_skeleton_maze_bridge_labels_dict():
    """Makes a dictionary of the labels of the nodes in the skeleton maze. The labels are the alphaneumeric code of the base edge('A1-A2'),
    followed by the subpoint label; either: 'west: W', 'center: C' or 'east:E' for horizontal edges, or 'south: S', 'center: C' or 'north: N' for vertical edges.
    """
    bridge_node2label = {}
    horizontal_node_subplot2letter = {0: "W", 1: "C", 2: "E"}
    vertical_node_subplot2letter = {0: "S", 1: "C", 2: "N"}
    simple_node2alphaneumeric = {v: k for k, v in get_simple_nodes_dict().items()}
    all_skeleton_bridge_nodes = _get_skeleton_bridge_nodes(nx.grid_2d_graph(*MAZE_NODE_DIMENSIONS).edges())
    horizontal_skeleton_nodes, vertical_skeleton_nodes = _split_edges_into_horizontal_vertical(
        all_skeleton_bridge_nodes
    )
    for node in horizontal_skeleton_nodes:
        simple_node1_code = simple_node2alphaneumeric[node[0]]
        simple_node2_code = simple_node2alphaneumeric[node[1]]
        subnode_code = horizontal_node_subplot2letter[node[2]]
        bridge_node2label[node] = simple_node1_code + "-" + simple_node2_code + "_" + subnode_code
    for node in vertical_skeleton_nodes:
        simple_node1_code = simple_node2alphaneumeric[node[0]]
        simple_node2_code = simple_node2alphaneumeric[node[1]]
        subnode_code = vertical_node_subplot2letter[node[2]]
        bridge_node2label[node] = simple_node1_code + "-" + simple_node2_code + "_" + subnode_code
    return bridge_node2label


def get_skeleton_maze_node_labels_dict():
    """Makes a dictionary of the labels of the nodes in the skeleton maze."""
    return {**_get_skeleton_maze_tower_labels_dict(), **_get_skeleton_maze_bridge_labels_dict()}


# %%
