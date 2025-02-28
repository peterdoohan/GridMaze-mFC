"""Library for managing data paths. Used for importing global path variables into other modules."""

# %% Imports
from pathlib import Path

# %% Define paths

MOUNTED_DATA_PATH = Path("/Volumes/behrens/peter_doohan/goalNav_mFC/experiment/data")
RELATIVE_DATA_PATH = Path("../data")
if MOUNTED_DATA_PATH.exists():
    DATA_PATH = MOUNTED_DATA_PATH
else:
    if RELATIVE_DATA_PATH.exists():
        DATA_PATH = RELATIVE_DATA_PATH
    else:
        raise FileNotFoundError(
            "Raw data directory not found. Check Ceph drive is mounted if working locally. Check relative path is available if working on HPC."
        )

EXPERIMENT_INFO_PATH = Path("../data/experiment_info")

PROCESSED_DATA_PATH = Path("../data/processed_data")

ANALYSIS_DATA_PATH = Path("../data/analysis_data")

ANALYSIS_INFO_PATH = Path("../data/analysis_data/analysis_info")

RESULTS_PATH = Path("../results")
