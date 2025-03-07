"""
Simple Library for copying relevant files from UnitMatch preprocessed data to processed data that are
needed for on the fly UnitMatch calculations
"""

# %% Imports
from pathlib import Path
import shutil

# %% Global Variables


# %% Functions


def get_unit_match_folder(session_dir):
    return Path(session_dir.spikesorting_path) / "UM_inputs"


def copy_unit_match_folder(source_folder, processed_data_path):
    if not source_folder.exists():
        raise FileNotFoundError(f"Source Unit Match folder {source_folder} does not exist.")
    else:
        target_folder = processed_data_path / "UnitMatch"
        shutil.copytree(source_folder, target_folder, dirs_exist_ok=True)
    return
