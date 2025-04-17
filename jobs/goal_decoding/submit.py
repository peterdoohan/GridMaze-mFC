"""
Script for submitting permutation tests for allocentric goal decoding analyses to the cluster.
"""

# %% Imports
import os

# %% Global Variables

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]

GOAL_SETS = ["subset_1", "subset_2", "all"]

DECODERS = ["logreg", "mlp"]

ALIGNMENTS = ["trial", "event"]

# %% Functions


def submit_all_jobs():
    """ """
    for maze_name in MAZE_NAMES:
        for goal_set in GOAL_SETS:
            for decoder in DECODERS:
                for aligned_to in ALIGNMENTS:
                    script_path = get_SLURM_script(maze_name, goal_set, aligned_to, decoder)
                    os.system(f"chmod +x {script_path}")
                    os.system(f"sbatch {script_path}")
    return print(f"Submitted all allocentric goal decoding jobs to HPC.")


def get_SLURM_script(maze_name, goal_set, aligned_to, decoder, n_permutations=500, n_jobs=10):
    """"""
    exp_name = f"{decoder}_{maze_name}_{goal_set}_{aligned_to}"
    script = f"""#!/bin/bash
#SBATCH --job-name={exp_name}
#SBATCH --output=jobs/goal_decoding/out/{exp_name}.out
#SBATCH --error=jobs/goal_decoding/err/{exp_name}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32GB
#SBATCH --time=72:00:00

module load miniconda
conda deactivate
conda deactivate
conda deactivate
conda deactivate
conda activate goalNav_mEC
python -c \"from GridMaze.analysis.event_aligned import allocentric_goal_decoding as agd; agd.run_bootstrapped_allocentric_goal_deocding('{maze_name}', '{goal_set}', '{aligned_to}', '{decoder}', '{n_permutations}', '{n_jobs}')\"
"""
    script_path = f"jobs/embedding_model/slurm/{exp_name}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    return script_path
