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
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "maze_day2date.json", "r") as input_file:
    MAZE_DAY2DATE = json.load(input_file)


# %% Functions


def submit_all_jobs():
    for subject in SUBJECT_IDS:
        for maze_name in MAZE_DAY2DATE.keys():
            for day_on_maze in [int(d) for d in MAZE_DAY2DATE[maze_name].keys()]:
                script_path = get_SLURM_script(subject, maze_name, day_on_maze)
                os.system(f"chmod +x {script_path}")
                os.system(f"sbatch {script_path}")


def get_SLURM_script(subject, maze_name, day_on_maze):
    """ """
    exp_name = f"goal_decoding_comparisons_{subject}_{maze_name}_{day_on_maze}"
    script = f"""#!/bin/bash
#SBATCH --job-name={exp_name}
#SBATCH --output=jobs/goal_decoding_comparisons/out/{exp_name}.out
#SBATCH --error=jobs/goal_decoding_comparisons/err/{exp_name}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH -p gpu
#SBATCH --mem=16GB
#SBATCH --time=2:00:00

module load miniconda
conda deactivate
conda deactivate
conda activate goalNav_mEC
python -c \"from GridMaze.analysis.distance_to_goal import combined_decoding as cd; cd.run_goal_decoding_comparison(('{subject}', '{maze_name}', {day_on_maze}))\"
"""
    script_path = f"jobs/goal_decoding_comparisons/slurm/{exp_name}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    return script_path
