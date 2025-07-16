"""
Script for populating analysis in GridMaze/analysis/distance_to_goal/distance_metrics.py (CPD and modle weights), over all session
@peterdoohan
"""

# %% Imports
import os
import json

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

ANALYSIS_TYPES = ["cpd", "weights"]

# %% Functions


def run_distance_metrics_analysis(subfolder="trials", progress_mon_decr=False):
    """ """
    for subject_ID in SUBJECT_IDS:
        for analysis_type in ANALYSIS_TYPES:
            script_path = get_SLURM_script(subject_ID, analysis_type, subfolder, progress_mon_decr)
            print(f"Submitting job for {subject_ID} {analysis_type}")
            os.system(f"chmod +x {script_path}")
            os.system(f"sbatch {script_path}")


def get_SLURM_script(subject_ID, analysis_type, subfolder, progress_mon_decr):
    job_name = f"{subject_ID}_{analysis_type}"
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=jobs/distance_metrics/out/{job_name}.out
#SBATCH --error=jobs/distance_metrics/err/{job_name}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH -p cpu
#SBATCH --mem=32GB
#SBATCH --time=96:00:00

module load miniconda
conda deactivate
conda activate goalNav_mEC

python <<EOF
from GridMaze.analysis.distance_to_goal import distance_metrics as dm
"""
    if analysis_type == "cpd":
        script += f"dm.populate_CPD_summary_dfs('{subject_ID}', {progress_mon_decr}, '{subfolder}')\n"
    elif analysis_type == "weights":
        script += f"dm.populate_weight_metric_summary_dfs('{subject_ID}', {progress_mon_decr}, '{subfolder}')\n"

    script += "EOF\n"

    script_path = f"jobs/distance_metrics/slurm/{job_name}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    return script_path
