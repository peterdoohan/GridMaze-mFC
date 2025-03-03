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

RAW_DATA_PATH = Path("../data/raw_data")

EXPERIMENT_INFO_PATH = Path("../data/experiment_info")

PROCESSED_DATA_PATH = Path("../data/processed_data")

PREPROCESSED_DATA_PATH = Path("../data/preprocessed_data")

ANALYSIS_DATA_PATH = Path("../data/analysis_data")

ANALYSIS_INFO_PATH = Path("../data/analysis_data/analysis_info")

RESULTS_PATH = Path("../results")

#%% Subpaths

PYCONTROL_PATH = RAW_DATA_PATH / "pycontrol"

EPHYS_PATH = RAW_DATA_PATH / "ephys"

VIDEO_PATH = RAW_DATA_PATH / "video"

DLC_PATH = PREPROCESSED_DATA_PATH / "DeepLabCut"

SPIKESORTING_PATH = PREPROCESSED_DATA_PATH / "spikesorting"
