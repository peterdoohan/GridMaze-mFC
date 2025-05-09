"""
Library for partitioning mazes into sections. Eg, for space generalisation decoding
analyses.
@peterdoohan
"""

# %% Imports
from cProfile import label
import json
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from regex import B

from GridMaze.maze import representations as mr
from GridMaze.maze import plotting as mp


# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "maze_measurements.json", "r") as input_file:
    MAZE_MEASUREMENTS = json.load(input_file)

D = MAZE_MEASUREMENTS["maze_node_dimensions"][0]
TOWER_DIST = MAZE_MEASUREMENTS["distance_between_node_centers"]
TOWER_WIDTH = MAZE_MEASUREMENTS["tower_width"]
_MIN = MAZE_MEASUREMENTS["lower_left_node_cartesian_center"][0] - TOWER_WIDTH / 2
_MAX = D * TOWER_DIST + 0.025
# %% Functions


def get_AB_split(simple_maze, s=3, plot=False):
    """ """

    x_edges = y_edges = np.linspace(_MIN, _MAX, s + 1)
    grid_limits = {}
    for row in range(s):
        y_min = y_edges[s - row - 1]
        y_max = y_edges[s - row]
        for col in range(s):
            x_min = x_edges[col]
            x_max = x_edges[col + 1]
            grid_limits[(row, col)] = {"xlim": (x_min, x_max), "ylim": (y_min, y_max)}
    return grid_limits

    A_cells, B_cells = _get_AB_cells(s, by="diag")
    A_cell2lims = {idx: lim for idx, lim in grid_limits.items() if idx in A_cells}
    B_cell2lims = {idx: lim for idx, lim in grid_limits.items() if idx in B_cells}

    label2pos = _get_label2pos(simple_maze)
    A, B = [], []
    for label, pos in label2pos.items():
        x, y = pos
        for idx, lim in A_cell2lims.items():
            xlim, ylim = lim["xlim"], lim["ylim"]
            if xlim[0] <= x < xlim[1] and ylim[0] <= y < ylim[1]:
                A.append(label)
                break
        else:
            for idx, lim in B_cell2lims.items():
                xlim, ylim = lim["xlim"], lim["ylim"]
                if xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]:
                    B.append(label)
                    break
    if plot:
        plot_simple_maze_split(simple_maze, A, B, s)
    return A, B


def _get_checker_board_split(n):
    A = []
    B = []
    for i in range(n + 1):
        for j in range(n + 1):
            if (i + j) % 2 == 0:
                A.append((i, j))
            else:
                B.append((i, j))
    return A, B


def _get_AB_cells(s, by="diag"):
    """ """
    grid = [[i * s + j for j in range(s)] for i in range(s)]
    n = len(grid)
    if by == "diag":
        diag = []
        for i in range(n):
            diag.append(grid[i][i])
            diag.append(grid[i][n - 1 - i])
        diag = list(set(diag))
        A = diag
        B = [x for x in range(s * s) if x not in diag]
    elif by == "row":
        A = [cell for i, row in enumerate(grid) if i % 2 == 0 for cell in row]
        B = [cell for i, row in enumerate(grid) if i % 2 != 0 for cell in row]
    elif by == "col":
        A = [cell for row in grid for idx, cell in enumerate(row) if idx % 2 == 0]
        B = [cell for row in grid for idx, cell in enumerate(row) if idx % 2 != 0]
    return A, B


def _get_label2pos(simple_maze):
    """ """
    coord2label = {**nx.get_node_attributes(simple_maze, "label"), **nx.get_edge_attributes(simple_maze, "label")}
    coord2pos = {**nx.get_node_attributes(simple_maze, "position"), **nx.get_edge_attributes(simple_maze, "position")}
    label2pos = {}
    for coord, label in coord2label.items():
        pos = coord2pos[coord]
        label2pos[label] = pos
    return label2pos


def plot_simple_maze_split(simple_maze, A, B, s, ax=None, A_color="green", B_color="silver"):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(5, 5))
    # plot maze with location colored by A, B split
    label2color = {
        **{label: A_color for label in A},
        **{label: B_color for label in B},
    }
    mp.plot_simple_maze_silhouette(
        simple_maze,
        ax=ax,
        color="silver",
        special_location2color=label2color,
    )
    # plot grid edges
    x_edges = y_edges = np.linspace(_MIN, _MAX, s + 1)
    for i in x_edges:
        ax.axvline(i, color="black", linestyle="--", alpha=0.5)
        ax.axhline(i, color="black", linestyle="--", alpha=0.5)
