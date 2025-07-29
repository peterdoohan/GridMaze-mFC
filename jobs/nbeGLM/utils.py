""" """

# %% Imports
from pathlib import Path

# %% Global Variables
from GridMaze.paths import RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "nbeGLM"

JOBS_PATH = Path("../jobs/nbeGLM")

# %% Default Parameters

DEFAULT_INPUT_DATA_KWARGS = {
    "subject_IDs": "all",
    "maze_name": "maze_1",
    "days_on_maze": "all",
    "input_features": ["place-direction", "distance_to_goal", "egocentric_action"],
    "input_feature_kwargs": {},
    "resolution": 0.1,
    "max_steps_to_goal": 30,
    "min_spike_count": 300,
    "moving_only": False,
}

DEFAULT_MODEL_INIT_KWARGS = {
    "with_embedding": True,
    "latent_inputs": None,
    "latent_nonlin": None,
    "partition": None,
    "Nhid": [100, 50],
    "Nlat": 20,
    "beta_act": 1e-1,
    "beta_weight": 1e-1,
    "inv_link": "exp",
    "noise_function": "Poisson",
    "sqrt_counts": False,
    "combine_frs": False,
}

DEFAULT_MODEL_TRAIN_KWARGS = {
    "lr": 5e-4,
    "nepochs": 3001,
    "test_freq": 1000,
    "eval_alpha": 1e-3,
}

DEFAULT_MODEL_EVAL_KWARGS = {
    "crossval_folds": 5,
    "crossval_alpha": 1e-3,
    "crossval_train_sessions": False,
}

DEFAULT_CV_NBEGLM_PARAMS = {
    "input_data_kwargs": DEFAULT_INPUT_DATA_KWARGS,
    "model_init_kwargs": DEFAULT_MODEL_INIT_KWARGS,
    "model_train_kwargs": DEFAULT_MODEL_TRAIN_KWARGS,
    "model_eval_kwargs": DEFAULT_MODEL_EVAL_KWARGS,
    "seed": 0,
    "overwrite": True,
    "verbose": True,
}

# %% Functions


def get_SLURM_script(exp_name, subfolder, model_params, run_fn="run_cv_nbeGLM"):
    """Create SLURM script for running nbeGLM experiment."""
    # check subfolder and exp_name folder exist in jobs/nbeGLM and results/nbeGLM
    for base_dir in [RESULTS_DIR, JOBS_PATH]:
        output_path = base_dir / subfolder
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

    # make sure jobs folder structure is in place
    for folder in ["out", "err", "slurm"]:
        output_path = JOBS_PATH / subfolder / "jobs" / f"{folder}"
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

    # create SLURM script
    script = f"""#!/bin/bash
#SBATCH --job-name=nbeGLM_{exp_name}
#SBATCH --output=jobs/nbeGLM/{subfolder}/out/{exp_name}.out
#SBATCH --error=jobs/nbeGLM/{subfolder}/err/{exp_name}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
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
    script_path = f"jobs/nbeGLM/{subfolder}/slurm/{exp_name}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    return script_path
