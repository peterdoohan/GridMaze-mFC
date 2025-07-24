"""
Looks for anaomtical gradients in egocentric action tuning
"""

# %% imports
import numpy as np
import pandas as pd
import json

from GridMaze.analysis.core import get_sessions as gs

# %% global variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)
# %% functions


def get_population_anatomy_df(subject_IDs="all", verbose=True):
    """ """
    subject_IDs = SUBJECT_IDS if subject_IDs == "all" else subject_IDs
    if verbose:
        print("Loading sessions...")
    sessions = gs.get_maze_sessions(
        subject_IDs=subject_IDs,
        maze_names="all",
        days_on_maze="all",
        with_data=["cluster_metrics", "cluster_egocentric_action_tuning_metrics"],
    )
    anat_dfs = []
    for session in sessions:
        if verbose:
            print(session.name)
        anat_df = get_session_egoaction_anat_df(session)
        anat_dfs.append(anat_df)
    results_df = pd.concat(anat_dfs, axis=0)
    return results_df


def get_session_egoaction_anat_df(
    session,
    actions=["turn_left", "turn_right"],
    min_split_half_corr=0.3,
    min_pref_action_factor=2,
    min_pref_action_frac=0.5,
):
    """ """
    # load data
    ego_metrics_df = session.cluster_egocentric_action_tuning_metrics
    ego_metrics_df = ego_metrics_df[ego_metrics_df.single_unit]
    cluster_metrics = session.cluster_metrics
    cluster_metrics = cluster_metrics[cluster_metrics.single_unit]
    # combine ant and tuning info
    _output_df = pd.concat(
        [
            ego_metrics_df.xs("all_action", axis=1, level=1)
            .xs("pref_action", axis=1, level=0, drop_level=False)
            .reset_index(drop=True),
            cluster_metrics.xs("voxel", axis=1, level=0, drop_level=False).reset_index(drop=True),
            cluster_metrics.xs("region", axis=1, level=0, drop_level=False).reset_index(drop=True),
        ],
        axis=1,
    )
    _output_df[("subject_ID", "")] = session.subject_ID
    return _output_df
