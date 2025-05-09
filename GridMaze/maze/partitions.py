"""
Library for partitioning mazes into sections. Eg, for space generalisation decoding
analyses.
@peterdoohan
"""

# %% Imports
import json
import numpy as np
from matplotlib import pyplot as plt

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


def get_AB_split(simple_maze, s=3):
    """ """
    return


def plot_check(simple_maze, s=3):
    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    mp.plot_simple_maze_silhouette(simple_maze, ax=ax, color="silver")

    x = y = np.linspace(_MIN, _MAX, s + 1)

    for _x in x:
        ax.axvline(_x, color="black", linestyle="--", alpha=0.5)
    for _y in y:
        ax.axhline(_y, color="black", linestyle="--", alpha=0.5)

    return
