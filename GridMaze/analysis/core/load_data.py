"""
Library for loading processed and analysis data from disk so it is appropriately stored in memory
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from datetime import date

# %% Global Variables
from ...paths import PROCESSED_DATA_PATH, ANALYSIS_DATA_PATH, EXPERIMENT_INFO_PATH

PROBE_DEPTHS_DF = pd.read_csv(EXPERIMENT_INFO_PATH / "probe_depths.htsv", sep="\t")

# %% Functions


def load_probe(subject_ID, tissue_sample=None, _date=None):
    """
    Loads probe data from disk into memory. Filtering for a particular tissue sample or date
    if provided.
    """
    probe_path = PROCESSED_DATA_PATH / subject_ID / "probe.htsv"
    probe_df = pd.read_csv(probe_path, sep="\t")
    probe_df = _unflatten_df_columns(probe_df)
    if _date is not None:
        tissue_sample = _get_tissue_sample(subject_ID, _date)
    if tissue_sample is not None:
        probe_df = probe_df[probe_df["tissue_sample"] == tissue_sample]
    return probe_df


def load(filepath):
    """Loads processed/analysis data from disk into memory. Making adjustments to the saved data as necessary."""
    # check filepath exists
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} does not exist")
    data_set = filepath.parts[-4]  # processed or analysis data
    if data_set == PROCESSED_DATA_PATH.name:
        data = _processed_data(filepath)
    elif data_set == ANALYSIS_DATA_PATH.name:
        data = _analysis_data(filepath)
    else:
        raise ValueError(f"Data set {data_set} not recognised, must  in folder processed_data or analysis_data")
    if data is None:
        raise FileNotFoundError(f"Data structure {filepath.name} not loaded")
    else:
        return data


def _processed_data(filepath):
    """Loads processed data from disk into memory. Making adjustments to the saved data as necessary."""
    processed_data_structure = filepath.name
    if processed_data_structure == "session_info.json":
        with open(filepath, "r") as infile:
            data = json.load(infile)
    elif processed_data_structure == "trials.htsv":
        df = pd.read_csv(filepath, sep="\t")
        data = _unflatten_df_columns(df)
    elif processed_data_structure == "events.htsv":
        data = pd.read_csv(filepath, sep="\t")
    elif processed_data_structure == "frames.tracking.htsv":
        df = pd.read_csv(filepath, sep="\t")
        data = _unflatten_df_columns(df)
    elif processed_data_structure == "frames.trajectories.htsv":
        df = pd.read_csv(filepath, sep="\t")
        data = _unflatten_df_columns(df)
    elif processed_data_structure == "frames.trialInfo.htsv":
        data = pd.read_csv(filepath, sep="\t")
    elif processed_data_structure == "spikes.times.npy":
        data = np.load(filepath)
    elif processed_data_structure == "spikes.clusters.npy":
        data = np.load(filepath)
    elif processed_data_structure == "clusters.metrics.htsv":
        data = pd.read_csv(filepath, sep="\t")
        data = _unflatten_df_columns(data)
    elif processed_data_structure == "lfp.times.npy":
        data = np.load(filepath)
    elif processed_data_structure == "lfp.signal.npy":
        data = np.load(filepath)
        data = data.astype(np.float32)
    elif processed_data_structure == "lfp.metrics.htsv":
        data = pd.read_csv(filepath, sep="\t")
        data = _unflatten_df_columns(data)
    else:
        raise ValueError(f"Processed data structure {processed_data_structure} not recognised")
    if data is None:
        raise FileNotFoundError(f"Processed data structure {processed_data_structure} not loaded")
    else:
        return data


def _analysis_data(filepath):
    """Loads analysis data from disk into memory. Making adjustments to the saved data as necessary."""
    analysis_data_structure = filepath.name
    if analysis_data_structure == "frames.navigation.parquet":
        data = _load_multiindex_parquet(filepath)
    elif analysis_data_structure == "frames.spikeRates.parquet":
        data = _load_multiindex_parquet(filepath)
    elif analysis_data_structure == "frames.spikeCounts.parquet":
        data = _load_multiindex_parquet(filepath)
    elif analysis_data_structure == "clusters.placeTuning.parquet":
        data = pd.read_parquet(filepath)
    elif analysis_data_structure == "clusters.placeDirectionTuning.parquet":
        data = _load_multiindex_parquet(filepath)
    elif analysis_data_structure == "trial_aligned_rates.parquet":
        data = _load_multiindex_parquet(filepath)
    elif analysis_data_structure == "event_aligned_rates.parquet":
        data = _load_multiindex_parquet(filepath)
    elif analysis_data_structure == "action_aligned_rates.parquet":
        data = _load_multiindex_parquet(filepath)
    elif analysis_data_structure == "navigation_strategies.parquet":
        data = _load_multiindex_parquet(filepath)
        # fix columns with multiple data types
        for d in ["N", "S", "E", "W"]:
            data[("available", d)] = data[("available", d)].map(_rebool)
    elif analysis_data_structure == "trajectory_decisions.parquet":
        data = _load_multiindex_parquet(filepath)
    elif analysis_data_structure == "distance_to_goal_aligned_rates.parquet":
        data = _load_multiindex_parquet(filepath)
    elif analysis_data_structure == "head_direction_tuning.parquet":
        data = _load_multiindex_parquet(filepath)
    elif analysis_data_structure == "allocentric_angle_to_goal_tuning.parquet":
        data = _load_multiindex_parquet(filepath)
    elif analysis_data_structure == "egocentric_angle_to_goal_tuning.parquet":
        data = _load_multiindex_parquet(filepath)
    else:
        raise ValueError(f"Analysis data structure {analysis_data_structure} not recognised")
    if data is None:
        raise FileNotFoundError(f"Analysis data structure {analysis_data_structure} not loaded")
    else:
        return data


def _rebool(x):
    if x == 0:
        return False
    elif x == 1:
        return True
    else:
        return x


def _unflatten_df_columns(df):
    """
    Unflattens columns when loading a multi-index columns from disk, see
    populate_processed_data.get_flattered_multiindex_columns to see how the columns were flattened
    and saved.
    """
    df.columns = pd.MultiIndex.from_tuples([tuple(col.split(".")) if "." in col else (col, "") for col in df.columns])
    return df


def _load_multiindex_parquet(filepath):
    df = pd.read_parquet(filepath, engine="fastparquet")
    if not np.all([isinstance(i, tuple) for i in df.columns]):
        if all([len(i.split(",")) > 1 for i in df.columns]):  # deal with multiindex
            df.columns = pd.MultiIndex.from_tuples([eval(col) for col in df.columns])
        else:
            pass
    df[df.isna()] = np.nan  # convert None values to np.nan
    return df


def _get_tissue_sample(subject_ID, _date):
    df = PROBE_DEPTHS_DF.copy()
    df["date"] = df.date.apply(date.fromisoformat)
    subject_df = df[(df["subject"] == subject_ID) & (df["date"] <= _date)]
    # Get the row with the latest date (i.e. the most recent measurement)
    latest_row = subject_df.sort_values("date", ascending=False).iloc[0]
    return latest_row.tissue_sample
