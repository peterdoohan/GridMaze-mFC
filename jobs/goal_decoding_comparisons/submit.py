"""
Script for submitting goal decoding analyses from
GridMaze/analysis/distance_to_goal/combined_decoding.py
to the HPC
@peterdoohan
"""

# %% Imports
import os
import json

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "distance_to_goal" / "goal_decoding_comparisons"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "maze_day2date.json", "r") as input_file:
    MAZE_DAY2DATE = json.load(input_file)


# %% Functions


def submit_all_jobs():
    for subject in SUBJECT_IDS:
        script_path_1 = get_SLURM_script(subject, permuted=False, n_repeats=1)
        script_path_2 = get_SLURM_script(subject, permuted=True, n_repeats=10)
        for sp in [script_path_1, script_path_2]:
            os.system(f"chmod +x {sp}")
            os.system(f"sbatch {sp}")


def get_SLURM_script(subject, permuted, n_repeats):
    """ """
    _permuted = "permuted" if permuted else "true"
    exp_name = f"goal_decoding_comparisons_{subject}_{_permuted}"
    script = f"""#!/bin/bash
#SBATCH --job-name={exp_name}
#SBATCH --output=jobs/goal_decoding_comparisons/out/{exp_name}.out
#SBATCH --error=jobs/goal_decoding_comparisons/err/{exp_name}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH -p a100
#SBATCH --mem=32GB
#SBATCH --time=48:00:00

module load miniconda
conda deactivate
conda deactivate
conda activate goalNav_mEC

python - <<'PYCODE'
from GridMaze.analysis.distance_to_goal import combined_decoding as cd
cd.populate_goal_decoding_comparisons('{subject}', {permuted}, {n_repeats})

PYCODE
"""
    script_path = f"jobs/goal_decoding_comparisons/slurm/{exp_name}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    return script_path
