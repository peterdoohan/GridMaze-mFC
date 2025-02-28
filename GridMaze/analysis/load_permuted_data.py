"""This script contains functions that load specific permuted datasets for various analyses"""
# %% Imports
import os
import itertools
import pandas as pd

from . import get_sessions as gs

# %% Global variables
PERMUTED_DATA_PATH = "../data/permuted_data"
DATASET2FOLDER_NAME = {
    "permuted_place_rate_dfs": "cluster.placeHeatmaps",
    "permuted_place_direction_rate_dfs": "cluster.placeDirectionHeatmaps",
}

# %% Functions


def load_permuted_dataset(dataset_name, maze_number, n_permuted="all"):
    """Currently all permuted datastructures have different number of columns (different maze structures)
    so can only load one maze at a time."""
    save_path = os.path.join(PERMUTED_DATA_PATH, DATASET2FOLDER_NAME[dataset_name])
    filenames = os.listdir(save_path)
    filenames = [f for f in filenames if eval(f.split("_")[-1].split(".")[0][-1]) == maze_number]
    if not n_permuted == "all":
        filenames = filenames[:n_permuted]
    filepaths = [os.path.join(save_path, f) for f in filenames]
    permuted_dfs = [gs.load_file(f, "permuted_df", ["permuted_df"]) for f in filepaths]
    return permuted_dfs


# add something that puts permutations of the same number together if more than one maze is recalled
