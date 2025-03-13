"""
Script for submitting permutation tests for allocentric goal decoding analyses to the cluster.
"""

# %% Imports
import os
from GridMaze.analysis.time_aligned import allocentric_goal_decoding as agd

# %% Global Variables

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]

# %% Functions


def get_SLURM_script(experiment_dict, conda_env_name):
    """"""
    exp_name = experiment_dict["exp_name"]
    script = f"""#!/bin/bash
#SBATCH --job-name=embedding_model_exp_{exp_name}
#SBATCH --output=jobs/embedding_model/out/{exp_name}.out
#SBATCH --error=jobs/embedding_model/err/{exp_name}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=72:00:00
"""

    if getpass.getuser() == "kjensen":  # sneaky
        script += f"""\nsource ~/.bashrc\n"""

    else:
        script += f"""\nmodule load miniconda
module load cuda/11.8
conda deactivate
conda deactivate
conda deactivate
conda deactivate
conda deactivate
"""

    script += f"""\n\nconda activate {conda_env_name}

python -c \"from GridMaze.analysis.embedding_model import run_experiment as re; re.run_embedding_model_experiment(**{experiment_dict})\"
"""
    script_path = f"jobs/embedding_model/slurm/{exp_name}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    return script_path
