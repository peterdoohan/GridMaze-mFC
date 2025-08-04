""" """

# %% Imports
from pathlib import Path

# %% Global Variables
from GridMaze.paths import RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "nbeGLM"

JOBS_PATH = Path("./jobs/nbeGLM")

# %% Default Parameters

DEFAULT_INPUT_DATA_KWARGS = {
    "subject_IDs": "all",
    "maze_name": "maze_1",
    "days_on_maze": "late",
    "input_groups": ["place_direction", "distance_to_goal", "egocentric_action"],
    "input_group_kwargs": {},
    "resolution": 0.1,
    "max_steps_to_goal": 30,
    "min_spike_count": 300,
    "moving_only": False,
}

DEFAULT_MODEL_INIT_KWARGS = {
    "Nhid": [100, 50],
    "Nlat": 20,
    "beta_act": 1e-1,
    "beta_weight": 1e-1,
    "partition": None,
    "latent_nonlin": None,
}

DEFAULT_MODEL_TRAIN_KWARGS = {
    "device": None,
    "test_freq": 300,
    "lr": 1e-3,
    "nepochs": 3001,
    "eval_alpha": 1e-3,
    "n_jobs": 64,
    "verbose": True,
}

DEFAULT_SCORE_KWARGS = {
    "n_folds": 5,
    "optimal_alpha": True,
    "n_jobs": 64,
    "verbose": False,
}

DEFAULT_NBEGLM_PARAMS = {
    "input_data_kwargs": DEFAULT_INPUT_DATA_KWARGS,
    "model_init_kwargs": DEFAULT_MODEL_INIT_KWARGS,
    "model_train_kwargs": DEFAULT_MODEL_TRAIN_KWARGS,
    "score_kwargs": DEFAULT_SCORE_KWARGS,
    "seed": 0,
    "save_path": None,
    "verbose": True,
    "overwrite": False,
}

# %% Functions


def get_SLURM_script(model_name, subfolder, maze_name, model_params, run_fn="run_cv_nbeGLM"):
    """Create SLURM script for running nbeGLM experiment."""
    # check jobs and results output folders exist
    _job_name = ".".join([maze_name, model_name])
    jobs_output_path = JOBS_PATH / subfolder
    for folder in ["out", "err", "slurm"]:
        output_path = jobs_output_path / "jobs" / f"{folder}"
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
    results_output_path = RESULTS_DIR / subfolder / maze_name / model_name
    if not results_output_path.exists():
        results_output_path.mkdir(parents=True, exist_ok=True)

    # create SLURM script
    script = f"""#!/bin/bash
#SBATCH --job-name=nbeGLM_{_job_name}
#SBATCH --output=jobs/nbeGLM/{subfolder}/out/{_job_name}.out
#SBATCH --error=jobs/nbeGLM/{subfolder}/err/{_job_name}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=72:00:00

module load miniconda
module load cuda/11.8
conda deactivate
conda activate goalNav_mEC

python <<EOF
from GridMaze.analysis.nbeGLM import run_nbeGLM as rn
rn.{run_fn}(**{model_params})
EOF
"""
    script_path = f"jobs/nbeGLM/{subfolder}/slurm/{_job_name}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    return script_path
