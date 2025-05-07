"""
Script for submitting place decoding analyses that require permutations
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
                for training_trial_phases in ["navigation", "all"]:
                    for input_type in ["spikes", "spikes_by_distance"]:
                        for output_type in ["place_direction", "place"]:
                            script_path = get_SLURM_script(
                                subject,
                                maze_name,
                                day_on_maze,
                                input_type,
                                output_type,
                                training_trial_phases,
                            )
                            os.system(f"chmod +x {script_path}")
                            os.system(f"sbatch {script_path}")


def get_SLURM_script(subject, maze_name, day_on_maze, input_type, output_type, training_trial_phases):
    """ """
    exp_name = (
        f"{input_type}_{output_type}_decoding_{subject}_{maze_name}_{day_on_maze}_training_{training_trial_phases}"
    )
    script = f"""#!/bin/bash
#SBATCH --job-name={exp_name}
#SBATCH --output=jobs/place_decoding/out/{exp_name}.out
#SBATCH --error=jobs/place_decoding/err/{exp_name}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH -p gpu
#SBATCH --mem=16GB
#SBATCH --time=1:00:00

module load miniconda
conda deactivate
conda deactivate
conda activate goalNav_mEC
python -c \"from GridMaze.analysis.distance_to_goal import place_decoding as dp; dp.run_session_place_decoding(('{subject}', '{maze_name}', {day_on_maze}), '{input_type}', '{output_type}', '{training_trial_phases}')\"
"""
    script_path = f"jobs/place_decoding/slurm/{exp_name}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    return script_path
