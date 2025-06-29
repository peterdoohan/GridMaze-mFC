"""
Library for anlysing distance-to-goal single cell tuning patterns across mazes.
"""

# %% Imports
import json
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from GridMaze.analysis.unit_match import get_across_maze_matches as mm

# %% Global Variables

from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

from GridMaze.analysis.unit_match.get_across_maze_matches import MAZE_PAIR2VALID_DAYS

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

RESULTS_DIR = RESULTS_PATH / "unit_match" / "distance_to_goal"
# %% Functions


# %% Misc
def plot_all_subject_matched_clusters(subject="m2", maze_pair=("maze_1", "maze_2")):
    """
    Search through matches to find some nice pairs!
    """
    colors = ["royalblue", "darkviolet"]
    matched_clusters = mm.get_cross_maze_matches(
        subject,
        maze_pair,
        single_units=True,
        tuning_metric="distance_to_goal",
        min_split_half_corr=0.3,
        return_as="cluster_objects",
        verbose=True,
    )
    for i, pair in enumerate(matched_clusters):
        f, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.spines[["top", "right"]].set_visible(False)
        for Clust, color in zip(pair, colors):
            Clust.plot_tuning(
                feature="distance_to_goal",
                ax=ax,
                feature_kwargs={"color": color},
            )
        ax.set_title(f"Match {i}")
        plt.show()
