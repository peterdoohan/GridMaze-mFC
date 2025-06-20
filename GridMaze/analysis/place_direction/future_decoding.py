""" """

# %% Imports
import os
import json
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import convert

from GridMaze.maze import representations as mr

# %% Global Variables

from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

RESULTS_DIR = RESULTS_PATH / "place_direction" / "future_decoding"

# %% Functions
