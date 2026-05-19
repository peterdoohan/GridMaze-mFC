"""Library for managing data paths. Used for importing global path variables into other modules."""

# %% Imports
from pathlib import Path

# %% Define paths

DATA_PATH = Path("../data")

RAW_DATA_PATH = DATA_PATH / "raw_data"

EXPERIMENT_INFO_PATH = DATA_PATH / "experiment_info"

PROCESSED_DATA_PATH = DATA_PATH / "processed_data"

PREPROCESSED_DATA_PATH = DATA_PATH / "preprocessed_data"

ANALYSIS_DATA_PATH = DATA_PATH / "analysis_data"

ANALYSIS_INFO_PATH = ANALYSIS_DATA_PATH / "analysis_info"

RESULTS_PATH = Path("../results")

# %% Subpaths

PYCONTROL_PATH = RAW_DATA_PATH / "pycontrol"

EPHYS_PATH = RAW_DATA_PATH / "ephys"

VIDEO_PATH = RAW_DATA_PATH / "video"

DLC_PATH = PREPROCESSED_DATA_PATH / "DeepLabCut"

SPIKESORTING_PATH = PREPROCESSED_DATA_PATH / "spikesorting"
