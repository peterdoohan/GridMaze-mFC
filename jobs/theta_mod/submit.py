"""
Sumbit theta-modulation analyses to cluster :)
@peterdoohan
"""

# %% Imports
import os

# %% Global Variances
from GridMaze.paths import RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "theta_mod" / "trajectory_alignment"
# %% Functions


def submit_jobs():
    """ """
    for subspace in ["all_spikes", "place_direction_tuning", "distance_to_goal_tuning", "egocentric_action_tuning"]:
        script_path = get_theta_alignment_SLURM_script(pcs_from=subspace)
        os.system(f"chmod +x {script_path}")
        os.system(f"sbatch {script_path}")
    return print("all jobs submitted to hpc")


def get_theta_alignment_SLURM_script(
    pcs_from,
    other_fn_params={"smooth_SD": 2.5, "vector_window": 2.5, "n_pcs": 5, "verbose": True, "save": True},
):
    """Create SLURM script for running nbeGLM experiment."""
    _job_name = f"theta_alignment_{pcs_from}"
    # create SLURM script
    script = f"""#!/bin/bash
#SBATCH --job-name={_job_name}
#SBATCH --output=jobs/theta_mod//out/{_job_name}.out
#SBATCH --error=jobs/theta_mod/err/{_job_name}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -p cpu
#SBATCH --mem=32GB
#SBATCH --time=48:00:00

module load miniconda
module load cuda/11.8
conda deactivate
conda activate goalNav_mEC

python <<EOF
from GridMaze.analysis.theta_mod import alignment as ali
ali.get_theta_alignment_summary_df(pcs_from='{pcs_from}', **{other_fn_params})
EOF
"""
    script_path = f"jobs/theta_mod/slurm/{_job_name}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    return script_path
