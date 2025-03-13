"""
"""

# %% Imports
import os
import json
from pathlib import Path
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.time_aligned import distance_decoding as dd

# %% Globs
from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(Path(EXPERIMENT_INFO_PATH) / "maze_day2date.json", "r") as input_file:
    MAZE_DAY2DATE = json.load(input_file)

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]
# %% Run


def run_distance_error_analysis(
    subject_IDs=["m2"],
    maze_names=["maze_1"],
    days_on_maze="late",
    event="cue",
    window=(-2, 2),
    resolution=0.1,
    n_folds=10,
    n_perm=10,
    alpha=10,
    normalise_spikes=False,
    exclude_short_distance_trials=True,
):
    for subject in subject_IDs:
        for maze in maze_names:
            days = [int(d) for d in MAZE_DAY2DATE[maze].keys()]
            if days_on_maze == "all":
                days_on_maze = days
            elif days_on_maze == "late":
                days_on_maze = days[-7:]
            for day in days_on_maze:
                submit_distance_error_job(
                    subject,
                    maze,
                    day,
                    event,
                    window,
                    resolution,
                    n_folds,
                    n_perm,
                    alpha,
                    normalise_spikes,
                    exclude_short_distance_trials,
                )
    return


def run_real_vs_decoded_distance_analysis(
    subject_IDs=["m2"], maze_names=["maze_1"], days_on_maze="late", event="cue", window=(0, 4), resolution=0.1, alpha=10
):
    for subject in subject_IDs:
        for maze in maze_names:
            days = [int(d) for d in MAZE_DAY2DATE[maze].keys()]
            if days_on_maze == "all":
                days_on_maze = days
            elif days_on_maze == "late":
                days_on_maze = days[-7:]
            for day in days_on_maze:
                submit_real_vs_decoded_distance_job(subject, maze, day, event, window, resolution, alpha)
                return
    return


# %%


def submit_real_vs_decoded_distance_job(
    subject,
    maze,
    day,
    event,
    window,
    resolution,
    alpha,
    conda_env_name="goalNav_mEC",
):
    """ """
    job_name = f"{subject}.{maze}.{day}.{event}_real_vs_decoded_distance"
    job_dict = {
        "subject_ID": subject,
        "maze_name": maze,
        "day_on_maze": day,
        "event": event,
        "window": window,
        "resolution": resolution,
        "alpha": alpha,
    }
    script_path = get_SLURM_script(
        job_dict, job_name, analysis_type="real_vs_decoded_distance", conda_env_name=conda_env_name
    )
    os.system(f"chmod +x {script_path}")
    os.system(f"sbatch {script_path}")
    return print(f"Submitted real vs decoded distance {job_name} to HPC")


def submit_distance_error_job(
    subject,
    maze,
    day,
    event,
    window,
    resolution,
    n_folds,
    n_perm,
    alpha,
    normalise_spikes,
    exclude_short_distance_trials,
    conda_env_name="goalNav_mEC",
):
    """ """
    job_name = f"{subject}.{maze}.{day}.{event}_distance_decoding_error"
    job_dict = {
        "subject_ID": subject,
        "maze_name": maze,
        "day_on_maze": day,
        "event": event,
        "window": window,
        "resolution": resolution,
        "n_folds": n_folds,
        "n_perm": n_perm,
        "alpha": alpha,
        "normalise_spikes": normalise_spikes,
        "exclude_short_distance_trials": exclude_short_distance_trials,
    }
    script_path = get_SLURM_script(job_dict, job_name, analysis_type="distance_error", conda_env_name=conda_env_name)
    os.system(f"chmod +x {script_path}")
    os.system(f"sbatch {script_path}")
    return print(f"Submitted distance decoding {job_name} to HPC")


def get_SLURM_script(job_dict, job_name, analysis_type, conda_env_name):
    """"""
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=jobs/distance_decoding/out/{job_name}.out
#SBATCH --error=jobs/distance_decoding/err/{job_name}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00

source /etc/profile.d/modules.sh
module load miniconda
conda deactivate
"""

    script += f"""\n\nconda activate {conda_env_name}"""

    if analysis_type == "distance_error":
        script += f"""
python -c \"from GridMaze.analysis.time_aligned import distance_decoding as dd; dd.decoding_distance_to_goal(**{job_dict})\""""
    elif analysis_type == "real_vs_decoded_distance":
        script += f"""
python -c \"from GridMaze.analysis.time_aligned import distance_decoding as dd; dd.decoded_vs_real_distance(**{job_dict})\""""

    script_path = f"jobs/distance_decoding/slurm/{job_name}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    return script_path
