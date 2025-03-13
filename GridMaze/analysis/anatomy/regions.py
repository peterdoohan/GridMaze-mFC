"""
Library for looking at the distribution of brain regions sampled in this experiment
@peterdoohan
"""

# %% Imports
import json
import re
import pandas as pd
import numpy as np
from collections import Counter
from GridMaze.analysis.core import get_sessions as gs

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

# %% Functions


def plot_subject_cell_counts():
    """ """
    return


def get_subject_cell_counts(ignore_layers=True, min_cells=10):
    """ """
    # count how many single units we recorded in each region throughout the experiment
    subject_region_counts = []
    for subject_ID in SUBJECT_IDS:
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names="all",
            days_on_maze="all",
            with_data=["cluster_metrics"],
            must_have_data=True,
        )
        cell_region_counts = []
        for session in sessions:
            cluster_metrics = session.cluster_metrics
            region_counts = list(cluster_metrics[cluster_metrics.single_unit].region.acronym.values)
            if ignore_layers:  # just keep main region not layer (just "PL", not "PL5")
                region_counts = [re.match(r"([A-Za-z]+)(.*)", s).groups()[0] for s in region_counts]
            cell_region_counts.extend(region_counts)
        total_counts = dict(Counter(cell_region_counts))
        if min_cells:
            total_counts = {k: v for k, v in total_counts.items() if v >= min_cells}
        subject_region_counts.append(total_counts)

    # find total unique regions recorded from
    all_regions = np.unique([j for i in subject_region_counts for j in i.keys()])

    # organise into dataframe
    results_df = pd.DataFrame(index=SUBJECT_IDS, columns=all_regions)
    for subject_ID, region_counts in zip(SUBJECT_IDS, subject_region_counts):
        for region, count in region_counts.items():
            results_df.loc[subject_ID, region] = count
    results_df.fillna(0, inplace=True)
    return results_df
