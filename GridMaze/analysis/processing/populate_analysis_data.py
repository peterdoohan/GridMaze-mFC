"""This module generates analysis data representations from processed data stored in the analysis_data folder."""

# %% Imports
import json
import pandas as pd

from .get_navigation_df import get_navigation_df
from .get_navigation_spike_dfs import get_navigation_spike_rates_df, get_navigation_spike_counts_df
from .get_basic_action_aligned_rates_dfs import get_basic_action_aligned_rates_df
from .get_time_aligned_rates_dfs import get_trial_aligned_rates_df, get_event_aligned_rates_df
from .get_navigation_strategies_dfs import get_navigation_strategies_df
from .get_trajectory_decisions_dfs import get_trajectory_decisions_df

from .get_distance_to_goal_aligned_rates_dfs import get_distance_to_goal_aligned_rates_df
from .get_cluster_heatmap_dfs import get_place_df, get_place_direction_df

from .get_angle_to_goal_dfs import (
    get_head_direction_tuning_df,
    get_allocentric_angle_to_goal_tuning_df,
    get_egocentric_angle_to_goal_tuning_df,
)

# %% Global variables

from ...paths import PROCESSED_DATA_PATH, ANALYSIS_DATA_PATH, EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)


ANALYSIS_DATA_STRUCTURES_DF = pd.DataFrame(
    [
        {"filename": "frames.navigation.parquet", "function": get_navigation_df, "session_types": ["maze"]},
        {"filename": "frames.spikeRates.parquet", "function": get_navigation_spike_rates_df, "session_types": ["maze"]},
        {
            "filename": "frames.spikeCounts.parquet",
            "function": get_navigation_spike_counts_df,
            "session_types": ["maze"],
        },
        {"filename": "trial_aligned_rates.parquet", "function": get_trial_aligned_rates_df, "session_types": ["maze"]},
        {"filename": "event_aligned_rates.parquet", "function": get_event_aligned_rates_df, "session_types": ["maze"]},
        {
            "filename": "action_aligned_rates.parquet",
            "function": get_basic_action_aligned_rates_df,
            "session_types": ["maze"],
        },
        {
            "filename": "navigation_strategies.parquet",
            "function": get_navigation_strategies_df,
            "session_types": ["maze"],
        },
        {
            "filename": "trajectory_decisions.parquet",
            "function": get_trajectory_decisions_df,
            "session_types": ["maze"],
        },
        {
            "filename": "distance_to_goal_aligned_rates.parquet",
            "function": get_distance_to_goal_aligned_rates_df,
            "session_types": ["maze"],
        },
        {"filename": "clusters.placeTuning.parquet", "function": get_place_df, "session_types": ["maze"]},
        {
            "filename": "clusters.placeDirectionTuning.parquet",
            "function": get_place_direction_df,
            "session_types": ["maze"],
        },
        {
            "filename": "head_direction_tuning.parquet",
            "function": get_head_direction_tuning_df,
            "session_types": ["maze"],
        },
        {
            "filename": "allocentric_angle_to_goal_tuning.parquet",
            "function": get_allocentric_angle_to_goal_tuning_df,
            "session_types": ["maze"],
        },
        {
            "filename": "egocentric_angle_to_goal_tuning.parquet",
            "function": get_egocentric_angle_to_goal_tuning_df,
            "session_types": ["maze"],
        },
    ]
)
# %% Process analysis data single process


def populate_analysis_data(data_structures="all", overwrite=False, subject_IDs="all"):
    """ """
    subject_IDs = SUBJECT_IDS if subject_IDs == "all" else subject_IDs
    data_strucutres_df = (
        ANALYSIS_DATA_STRUCTURES_DF
        if data_structures == "all"
        else ANALYSIS_DATA_STRUCTURES_DF[ANALYSIS_DATA_STRUCTURES_DF.filename.isin(data_structures)]
    )
    for subject in subject_IDs:
        processed_data_paths = [f for f in (PROCESSED_DATA_PATH / subject).iterdir() if f.is_dir()]
        analysis_data_paths = [ANALYSIS_DATA_PATH / subject / p.name for p in processed_data_paths]
        for processed_data_path, analysis_data_path in zip(processed_data_paths, analysis_data_paths):
            if not analysis_data_path.exists():
                analysis_data_path.mkdir(parents=True)
            print(f"Saving analysis data for {processed_data_path}")
            for _, row in data_strucutres_df.iterrows():
                try:
                    save_analysis_data(
                        row.filename,
                        row.function,
                        row.session_types,
                        processed_data_path,
                        analysis_data_path,
                        overwrite,
                    )
                except FileNotFoundError:
                    print(f"FileNotFoundError: {row.function.__name__} failed for {processed_data_path}")
                    pass


def populate_analysis_data_single_session(
    processed_data_path, analysis_data_path, data_structures="all", overwrite=False
):
    data_strucutres_df = (
        ANALYSIS_DATA_STRUCTURES_DF
        if data_structures == "all"
        else ANALYSIS_DATA_STRUCTURES_DF[ANALYSIS_DATA_STRUCTURES_DF.filename.isin(data_structures)]
    )
    if not analysis_data_path.exists():
        analysis_data_path.mkdir(parents=True)
    print(f"Saving analysis data for {processed_data_path}")
    for _, row in data_strucutres_df.iterrows():
        try:
            save_analysis_data(
                row.filename, row.function, row.session_types, processed_data_path, analysis_data_path, overwrite
            )
        except FileNotFoundError:
            print(f"FileNotFoundError: {row.function.__name__} failed for {processed_data_path}")
            pass
    return


def save_analysis_data(filename, function, session_types, processed_data_path, analysis_data_path, overwrite):
    """
    Saves a single analysis data structure to the analysis_data folder
    """
    if processed_data_path.name.split(".")[-1] not in session_types:
        # navigation data not relevant to rest sessions.
        return
    if overwrite or not (analysis_data_path / filename).exists():
        data = function(processed_data_path, analysis_data_path)
        if data is None:
            pass
        else:
            if filename.endswith(".parquet"):
                data.columns = data.columns.map(lambda x: str(x))
                data.to_parquet(analysis_data_path / filename, compression="gzip")
            elif filename.endswith(".json"):
                with open(analysis_data_path / filename, "w") as outfile:
                    json.dump(data, outfile, indent=4)
    return


# %%
