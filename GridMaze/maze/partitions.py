"""
Library for partitioning mazes into sections. Eg, for space generalisation decoding
analyses.
@peterdoohan
"""

# %% Imports
import json
import numpy as np

from GridMaze.maze import representations as mr


# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "maze_configs.json", "r") as input_file:
    MAZE_DAY2DATE = json.load(input_file)


# %% Functions
