""" """

# %% imports
import os
import json

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)


# %% Fns
def submit_all_jobs():
    for subject in SUBJECT_IDS:
        script_path = get_SLURM_script(subject)
        os.system(f"chmod +x {script_path}")
        os.system(f"sbatch {script_path}")


def get_SLURM_script(subject):
    """ """
    exp_name = f"place_generalised_goal_decoding_{subject}"
    script = f"""#!/bin/bash
#SBATCH --job-name={exp_name}
#SBATCH --output=jobs/place_generalised_goal_decoding/out/{exp_name}.out
#SBATCH --error=jobs/place_generalised_goal_decoding/err/{exp_name}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH -p gpu
#SBATCH --mem=16GB
#SBATCH --time=24:00:00

module load miniconda
conda deactivate
conda deactivate
conda activate goalNav_mEC

python - <<'PYCODE'
from GridMaze.analysis.distance_to_goal import place_generalised_decoding as pgd
pgd.populate_decoding_results('{subject}')
PYCODE
"""
    script_path = f"jobs/place_generalised_goal_decoding/slurm/{exp_name}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    return script_path
