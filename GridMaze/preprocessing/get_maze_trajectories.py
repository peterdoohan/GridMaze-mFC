"""This module translates subject centroid tracking data into simple maze trajectories of nodes and edges."""
#%% Imports
from scipy.spatial import KDTree
import networkx as nx
#%% Global variables

#%% Main functions

def get_valid_simple_maze_trajectory(centroid_positions, simple_maze):
    """Get the trajectory of the animal through the maze, if the trajectory is valid. 
    Returns: list of simple maze labels (nodes & edges)corresponding to the session trajectory.
    Note: test session have had valid trajectories. No current solutions for invalid trajectories."""
    coords_trajectory = get_nearest_simple_coord(centroid_positions, simple_maze)
    if trajectory_qc(coords_trajectory, simple_maze):
        maze_coord2label = get_simple_coord2label_dict(simple_maze)
        return [maze_coord2label[coord] for coord in coords_trajectory]
    else:
        #in some cases, nodes that transition between edges are missed potentially because animals
        #cut corners on the maze. Correct these instances by replacing the missing nodes in the trajectory
        #this also fixes instances in two sessions where tracking made big jumps (potentialy when rescuing
        # the mice from a tangled wire)
        corrected_coords_trajectory = correct_invalid_trajectory(coords_trajectory, simple_maze)
        if trajectory_qc(corrected_coords_trajectory, simple_maze):
            maze_coord2label = get_simple_coord2label_dict(simple_maze)
            return [maze_coord2label[coord] for coord in corrected_coords_trajectory]
        else:
            print('Invalid trajectory!')
    
def get_skeleton_maze_trajectory(centroid_positions, skeleton_maze):
    """Get the trajectory of the animal through the maze, if the trajectory is valid.
    Returns: list of simple maze labels (nodes only) corresponding to the session trajectory."""
    coords_trajectory = get_nearest_skeleton_node(centroid_positions, skeleton_maze)
    skeleton_coord2label = get_skeleton_coord2label_dict(skeleton_maze)
    return [skeleton_coord2label[coord] for coord in coords_trajectory]

def get_nearest_simple_coord(centroid_positions, maze):
    """Get the nearest maze node for each position in the centroid_positions array."""
    position2maze_coord = get_position2simple_coord_dict(maze)
    coord_positions = list(position2maze_coord.keys())
    kd_tree = KDTree(coord_positions) #Create a KDTree from the maze coord positions
    _, nearest_node_indices = kd_tree.query(centroid_positions) # Query the KDTree to get the nearest maze node for each position
    nearest_nodes = [position2maze_coord[tuple(coord_positions[i])] for i in nearest_node_indices]
    return nearest_nodes

def get_nearest_skeleton_node(centroid_positions, skeleton_maze):
    """Get the nearest maze node for each position in the centroid_positions array."""
    position2maze_coord = get_position2skeleton_coord_dict(skeleton_maze)
    coord_positions = list(position2maze_coord.keys())
    kd_tree = KDTree(coord_positions) #Create a KDTree from the maze coord positions
    _, nearest_node_indices = kd_tree.query(centroid_positions) # Query the KDTree to get the nearest maze node for each position
    nearest_nodes = [position2maze_coord[tuple(coord_positions[i])] for i in nearest_node_indices]
    return nearest_nodes

def distill_trajectory(positions):
    """Remove consecutive duplicate positions from a trajectory"""
    distilled_positions = []
    last_position = None
    for position in positions:
        if position != last_position:
            distilled_positions.append(position)
            last_position = position
    return distilled_positions

def trajectory_qc(trajectory_coords, maze):
    """Checks that maze state transitions are valid over a session trajectory"""
    distilled_coords = distill_trajectory(trajectory_coords)
    is_valid = True
    for i in range(len(distilled_coords)-1):
        coord_1 = distilled_coords[i]
        coord_2 = distilled_coords[i+1]
        coord_1_type = 'node' if isinstance(coord_1[0], int) else 'edge'
        coord_2_type = 'node' if isinstance(coord_2[0], int) else 'edge'
        if not coord_1_type == coord_2_type:
            if coord_1_type == 'node':
                if not coord_2 in maze.edges(coord_1):
                    #print(f'Invalid trajectory: {coord_1} to {coord_2}. Index: {i}')
                    is_valid = False
                else: pass
            elif coord_1_type == 'edge':
                if not coord_1 in maze.edges(coord_2):
                    #print(f'Invalid trajectory: {coord_1} to {coord_2}. Index: {i}')
                    is_valid = False
                else: pass
        else: 
            #print(f'Invalid trajectory: {coord_1} to {coord_2}. Index: {i}')
            is_valid = False
    if is_valid:
        #print('All trajectories are valid!')
        pass
    return is_valid

def invalid_transition_indices(trajectory, simple_maze):
    invalid_indicies = []
    coords_trajectory = trajectory.copy()
    for i in range(len(coords_trajectory)-1):
        error = False
        coord_1 = coords_trajectory[i]
        coord_2 = coords_trajectory[i+1]
        if coord_1 != coord_2:
            coord_1_type = 'node' if isinstance(coord_1[0], int) else 'edge'
            coord_2_type = 'node' if isinstance(coord_2[0], int) else 'edge'
            if not coord_1_type == coord_2_type:
                if coord_1_type == 'node':
                    if not coord_2 in simple_maze.edges(coord_1):
                        error = True
                elif coord_1_type == 'edge':
                    if not coord_1 in simple_maze.edges(coord_2):
                        error = True
            else: error = True
        if error:
            invalid_indicies.append(i+1)
    if len(invalid_indicies)==0:
        return None
    else:
        return invalid_indicies

def correct_invalid_transition(trajectory, invalid_indice, simple_maze):
    #instance where coord jumps randomly
    coords_trajectory = trajectory.copy()
    if coords_trajectory[invalid_indice-1] == coords_trajectory[invalid_indice+1]:
        coords_trajectory[invalid_indice] = coords_trajectory[invalid_indice-1]
        return coords_trajectory
    else:
        #instances where a node/edge is skipped
        coord_1 = coords_trajectory[invalid_indice-1]
        coord_2 = coords_trajectory[invalid_indice]
        coord_1_type = 'node' if isinstance(coord_1[0], int) else 'edge'
        coord_2_type = 'node' if isinstance(coord_2[0], int) else 'edge'
        missing_shortest_path = find_missing_shortest_path(simple_maze, coord_1, coord_1_type, coord_2, coord_2_type)
        for j, missing_coord in enumerate(missing_shortest_path):
            coords_trajectory[invalid_indice+j] = missing_coord
        return coords_trajectory

def correct_invalid_trajectory(trajectory, simple_maze):
    coords_trajectory = trajectory.copy()
    invalid_indicies = invalid_transition_indices(coords_trajectory, simple_maze)
    while invalid_indicies:
        i = invalid_indicies[0]
        new_trajectory = correct_invalid_transition(coords_trajectory, i, simple_maze)
        coords_trajectory = new_trajectory
        new_invalid_indicies = invalid_transition_indices(new_trajectory, simple_maze)
        invalid_indicies = new_invalid_indicies
    return coords_trajectory

def find_missing_shortest_path(simple_maze, coord_1, coord_1_type, coord_2, coord_2_type):
    """Find the shortest path between two simple maze coordinates given if the coords are nodes or edges"""
    if coord_1_type == 'node' and coord_2_type == 'edge':
        path1 = nx.shortest_path(simple_maze, coord_1, coord_2[0])
        path2 = nx.shortest_path(simple_maze, coord_1, coord_2[1])
        missing_shortest_path = path1 if len(path1) < len(path2) else path2
    elif coord_1_type == 'edge' and coord_2_type == 'node':
        path1 = nx.shortest_path(simple_maze, coord_1[0], coord_2)
        path2 = nx.shortest_path(simple_maze, coord_1[1], coord_2)
        missing_shortest_path = path1 if len(path1) < len(path2) else path2
    elif coord_1_type == 'edge' and coord_2_type == 'edge':
        paths = [nx.shortest_path(simple_maze, node1, node2) for node1 in coord_1 for node2 in coord_2]
        missing_shortest_path = min(paths, key=len)
    elif coord_1_type == 'node' and coord_2_type == 'node':
        missing_shortest_path = nx.shortest_path(simple_maze, coord_1, coord_2)
        if len(missing_shortest_path) == 2: #deal with case of adjacent nodes
            missing_shortest_path = (coord_1, coord_2)
    if len(missing_shortest_path)>1:
        return connect_shortest_path_nodes(missing_shortest_path)
    else: 
        return missing_shortest_path

def connect_shortest_path_nodes(missing_shortest_path):
    missing_edges = [(missing_shortest_path[i], missing_shortest_path[i+1]) for i in range(len(missing_shortest_path) - 1)]
    missing_edges = correct_edge_order(missing_edges)
    path_with_edges = []
    for i in range(len(missing_edges)):
        path_with_edges.append(missing_shortest_path[i])
        path_with_edges.append(missing_edges[i])
    path_with_edges.append(missing_shortest_path[-1])
    return path_with_edges

def correct_edge_order(edges):
    corrected_edges = []
    for edge in edges:
        node1, node2 = edge
        if node1 > node2:
            corrected_edges.append((node2, node1))
        else:
            corrected_edges.append(edge)
    return corrected_edges

def get_position2simple_coord_dict(simple_maze):
    """Get a dictionary mapping maze node/edge positions to maze node/edge coordinates"""
    position2simple_node = {tuple(position):node for node, position in simple_maze.nodes(data='position')}
    position2simple_edge={tuple(position):(node1, node2) for node1,node2,position in simple_maze.edges(data='position')}
    position2simple_coord = position2simple_node.copy(); position2simple_coord.update(position2simple_edge)
    return position2simple_coord

def get_position2skeleton_coord_dict(skeleton_maze):
    position2skeleton_node = {tuple(position):node for node, position in skeleton_maze.nodes(data='position')}
    return position2skeleton_node

def get_simple_coord2label_dict(simple_maze):
    """Get a dictionary mapping maze node/edge coordinates to maze node/edge labels"""
    simple_node2label = {k:v for k,v in simple_maze.nodes(data='label')}
    simple_edge2label = {(k1,k2):v for k1,k2,v in simple_maze.edges(data='label')}
    simple_coord2label = simple_node2label.copy(); simple_coord2label.update(simple_edge2label)
    return simple_coord2label

def get_skeleton_coord2label_dict(skeleton_maze):
    skeleton_node2label = {k:v for k,v in skeleton_maze.nodes(data='label')}
    return skeleton_node2label
