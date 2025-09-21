""" """

# %% Imports
from pathlib import Path
from importlib import import_module

# %% Global Variables
from GridMaze.paths import RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "neGLM"

# %% Default Parameters

DEFAULT_INPUT_DATA_KWARGS = {
    "subject_IDs": "all",
    "maze_name": "maze_1",
    "days_on_maze": "late",
    "input_groups": ["place_direction", "distance_to_goal", "egocentric_action"],
    "input_group_kwargs": {},
    "resolution": 0.2,
    "max_steps_to_goal": 30,
    "min_spike_count": 300,
    "moving_only": True,
}

DEFAULT_MODEL_INIT_KWARGS = {
    "Nhid": [100, 50],
    "Nlat": 15,
    "beta_act": 1e-1,
    "beta_weight": 1e-1,
    "partition": None,
    "latent_nonlin": None,
    "latent_split": None,
}

DEFAULT_MODEL_TRAIN_KWARGS = {
    "device": None,
    "test_freq": 500,
    "lr": 1e-3,
    "nepochs": 3001,
    "eval_alpha": 1e-3,
    "n_jobs": 24,
    "verbose": True,
}

DEFAULT_SCORE_KWARGS = {
    "n_folds": 5,
    "optimal_alpha": True,
    "n_jobs": 24,
    "verbose": False,
}

DEFAULT_NEGLM_PARAMS = {
    "input_data_kwargs": DEFAULT_INPUT_DATA_KWARGS,
    "model_init_kwargs": DEFAULT_MODEL_INIT_KWARGS,
    "model_train_kwargs": DEFAULT_MODEL_TRAIN_KWARGS,
    "score_kwargs": DEFAULT_SCORE_KWARGS,
    "seed": 0,
    "save_path": None,
    "verbose": True,
    "overwrite": False,
}


# %%


def submit_all_jobs():
    for subfolder in [
        "performance_validation",
        "interaction_validation",
        "variance_explained",
        "other_features",
        "variance_explained_full",
        "feature_interactions_full",
        "variance_explained_all_sessions",
        "variance_explained_full_all_sessions",
    ]:
        module_path = f"jobs.neGLM.{subfolder}.submit"
        # Import the module
        submit = import_module(module_path)
        submit.submit_jobs(seed=0, subfolder=subfolder)


# %% Functions


def find_missing(model_set_params):
    missing = []
    for model_params in model_set_params:
        save_path = Path(model_params["model_params"]["save_path"])
        if not (save_path / "DONE.txt").exists():
            missing.append(model_params)
    return missing


def get_SLURM_script(
    model_name, subfolder, maze_name, model_params, run_fn="run_cv_neGLM", resource_type="gpu", RAM="16G"
):
    """Create SLURM script for running neGLM experiment."""
    # check jobs and results output folders exist
    _job_name = ".".join([maze_name, model_name])
    results_output_path = RESULTS_DIR / subfolder / maze_name / model_name
    if not results_output_path.exists():
        results_output_path.mkdir(parents=True, exist_ok=True)

    # determine SLURM resource directives based on resource_type
    if resource_type == "gpu":
        partition = "gpu"
        gres_directive = "#SBATCH --gres=gpu:1"
    elif resource_type == "cpu":
        partition = "cpu"
        gres_directive = ""
    else:
        raise ValueError(f"Unknown resource_type: {resource_type}. Use 'gpu' or 'cpu'.")

    # create SLURM script
    script = f"""#!/bin/bash
#SBATCH --job-name=neGLM_{_job_name}
#SBATCH --output=jobs/neGLM/{subfolder}/out/{_job_name}.out
#SBATCH --error=jobs/neGLM/{subfolder}/err/{_job_name}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH -p {partition}
{gres_directive}
#SBATCH --mem={RAM}
#SBATCH --time=72:00:00

module load miniconda
module load cuda/11.8
conda deactivate
conda activate goalNav_mEC

python <<EOF
from GridMaze.analysis.neGLM import run_neGLM as rn
rn.{run_fn}(**{model_params})
EOF
"""
    script_path = f"jobs/neGLM/{subfolder}/slurm/{_job_name}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    return script_path
