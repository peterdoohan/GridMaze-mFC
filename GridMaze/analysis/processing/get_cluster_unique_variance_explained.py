"""
Leverage neGLM results to save out which cells/clusters have unique variance expalined by
main data features (place-direction, distance-to-goal)
"""

# %%  Imports
import pandas as pd
import numpy as np

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import convert

from GridMaze.analysis.neGLM import load_model_sets as lms
from GridMaze.analysis.neGLM import variance_explained as ve

# %% Global Variables

from GridMaze.paths import ANALYSIS_INFO_PATH

if not ANALYSIS_INFO_PATH.exists():
    ANALYSIS_INFO_PATH.mkdir(parents=True)

# %% Functions


def load_unique_variance_explained_df(late_sessions=True, full_features=False):
    _fn = "cluster_unique_variance_explained"
    if full_features:
        _fn = _fn + "_full"
    if not late_sessions:
        _fn = _fn + "_all_sessions"
    filename = _fn + ".parquet"
    filepath = ANALYSIS_INFO_PATH / filename
    cve_df = pd.read_parquet(filepath)
    return cve_df


# see get_analysis_info.py for where these datastructures get saved out


def get_cluster_unique_variance_explained(late_sessions=True, full_features=False):
    """ """
    # get clusters with unique variance explained by main features
    ms = "variance_explained"
    reduced_models = ["remove_place_direction", "remove_distance_to_goal"]
    if full_features:
        ms = ms + "_full"
        reduced_models.extend(["remove_egocentric_action", "remove_velocity"])
    if not late_sessions:
        ms = ms + "_all_sessions"
    results_df = lms.load_model_set_cv_scores(ms, maze_names=["maze_1", "maze_2", "rooms_maze"], all_completed=True)
    feature_tuning_df = ve.get_feature_tuned_df(
        results_df, reduced_models, multiple_comparisons_corrected=False, alpha=0.01
    )
    # get all cluster IDs and set missing to False (no variance explained)
    _days_on_maze = "late" if late_sessions else "all"
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names="all",
        days_on_maze=_days_on_maze,
        with_data=["cluster_metrics"],
        must_have_data=True,
    )
    all_cluster_unique_IDs = []
    for session in sessions:
        all_cluster_unique_IDs.extend(_get_session_cluster_unique_IDs(session))
    missing_clusters = np.setdiff1d(all_cluster_unique_IDs, feature_tuning_df.index.get_level_values(1))
    missing_df = pd.DataFrame(
        data=False,
        index=pd.MultiIndex.from_tuples(list(zip([m.split(".")[0] for m in missing_clusters], missing_clusters))),
        columns=feature_tuning_df.columns,
    )
    # combine and sort
    uve_df = pd.concat([feature_tuning_df, missing_df])
    uve_df = uve_df.sort_index()
    return uve_df


def _get_session_cluster_unique_IDs(session):
    session_info = session.session_info
    cluster_metrics = session.cluster_metrics
    cluster_IDs = cluster_metrics[cluster_metrics.single_unit].cluster_ID.values
    return convert.cluster_IDs2scluster_unique_IDs(session_info, cluster_IDs)
