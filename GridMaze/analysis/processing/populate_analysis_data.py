"""This module generates analysis data representations from processed data stored in the analysis_data folder."""

# %% Imports
import json
import pandas as pd

# from concurrent.futures import ProcessPoolExecutor

from .get_navigation_df import get_navigation_df
from .get_navigation_spike_dfs import get_navigation_spike_rates_df, get_navigation_spike_counts_df
from .get_basic_action_aligned_rates_dfs import get_basic_action_aligned_rates_df
from .get_time_aligned_rates_dfs import get_trial_aligned_rates_df, get_event_aligned_rates_df
from .get_navigation_strategies_dfs import get_navigation_strategies_df
from .get_trajectory_decisions_dfs import get_trajectory_decisions_df
from .get_distance_to_goal_aligned_rates_dfs import get_distance_to_goal_aligned_rates_df
from .get_cluster_heatmap_dfs import get_place_df, get_place_direction_df
from .get_spatial_periodicity_dfs import get_spatial_periodicity_df
from .get_angle_to_goal_dfs import (
    get_head_direction_tuning_df,
    get_allocentric_angle_to_goal_tuning_df,
    get_egocentric_angle_to_goal_tuning_df,
)
from .get_inferred_routes import get_routes_df, get_navigation_routes_df, get_routes_prior
from .get_route_change_aligned_rates import get_route_change_aligned_rates, get_route_aligned_rates_df

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
            "filename": "clusters.spatialPeriodicity.parquet",
            "function": get_spatial_periodicity_df,
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
        {"filename": "routes.parquet", "function": get_routes_df, "session_types": ["maze"]},
        {"filename": "frames.routes.parquet", "function": get_navigation_routes_df, "session_types": ["maze"]},
        {"filename": "routes_prior.json", "function": get_routes_prior, "session_types": ["maze"]},
        {
            "filename": "route_change_aligned_rates.parquet",
            "function": get_route_change_aligned_rates,
            "session_types": ["maze"],
        },
        {"filename": "route_aligned_rates.parquet", "function": get_route_aligned_rates_df, "session_types": ["maze"]},
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


# %% Multioprocess version of main fuction for cluster

# def save_analysis_data_multiprocess(data_streams=["navigation_dfs"], n_processes=4, overwrite=False):
#     """Currently not running effiently on my local machein, use save_analysis_data instead """
#     data_stream2save_functions = {
#         "navigation_dfs": save_navigation_dfs,
#     }
#     if data_streams == "all":
#         save_functions = data_stream2save_functions.values()
#     else:
#         save_functions = [data_stream2save_functions[stream] for stream in data_streams]
#     filepaths = _get_all_processed_and_analysis_data_filepaths()  # tuple(processed_data_path, analysis_data_path)
#     n_sessions = len(filepaths)
#     for _, analysis_folder in filepaths:
#         if not analysis_folder.exists():
#             analysis_folder.mkdir(parents=True)
#         with ProcessPoolExecutor(max_workers=n_processes) as executor:
#             print("saving analysis data")
#             executor.map(
#                 _save_all_data_structures,
#                 [save_functions] * n_sessions,
#                 filepaths,
#                 [overwrite] * n_sessions,
#             )


# def _save_all_data_structures(
#     save_functions,
#     filepaths,
#     overwrite,
# ):
#     processed_data_path, analysis_data_path = filepaths
#     for fn in save_functions:
#         try:
#             fn(processed_data_path, analysis_data_path, overwrite)
#         except FileNotFoundError:  # if no processes ephys data available
#             pass
#         except AttributeError:
#             pass


# def _get_all_processed_and_analysis_data_filepaths():
#     filepaths = []
#     for s in SUBJECT_IDS:
#         processed_data_paths = [f for f in (PROCESSED_DATA_PATH / s).iterdir() if f.is_dir()]
#         # analysis data folder structure is a mirror of procesed data folder structure
#         analysis_data_paths = [ANALYSIS_DATA_PATH / s / p.name for p in processed_data_paths]
#         filepaths.extend([(p, a) for p, a in zip(processed_data_paths, analysis_data_paths)])
#     return filepaths
