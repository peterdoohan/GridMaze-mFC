"""
Updated script for submitting embedding model jobs to the HPC
"""

# %% Imports
import os
import copy
import json
from pathlib import Path

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

EMBEDDING_MODEL_RESULTS = RESULTS_PATH / "embedding_model" / "exps"

with open(Path(EXPERIMENT_INFO_PATH) / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]


# %% Functions
def submit_job(experiment_dict, conda_env_name="goalNav_mEC"):
    """ """
    # check if experiment has already been run
    save_dir = EMBEDDING_MODEL_RESULTS / experiment_dict["exp_name"]
    if (save_dir / "DONE.txt").exists() and not experiment_dict["overwrite"]:
        return print(
            f"Exp with name {experiment_dict['exp_name']} already completed: {save_dir}. To overwrite, set overwrite=True"
        )
    else:
        script_path = get_SLURM_script(experiment_dict, conda_env_name)
        os.system(f"chmod +x {script_path}")
        os.system(f"sbatch {script_path}")
        return print(f"Submitted embedding model experiment: {experiment_dict["exp_name"]} to HPC")


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

module load miniconda
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


# %% Default params

DEFAULT_EXPERIMENT = {
    "exp_name": "default_exp",
    "exp_set": None,
    "with_embedding": True,
    "run_crossvalidation": True,
    "train_full_model": False,
    "input_kwargs": {
        "subject_IDs": ["m2"],
        "maze_name": "maze_1",
        "days_on_maze": "late",
        "input_features": ["distance", "place_direction"],
        "distance_metrics": ("distance_to_goal", "geodesic"),
        "include_multi_unit": False,
        "navigation_only": True,
        "moving_only": True,
        "resolution": 0.1,
        "max_distance": None,
        "max_steps_to_goal": 30,
        "distance_bin_method": "uniform",
        "n_distance_bins": 20,
        "min_spike_count": 300,
    },
    "model_init_kwargs": {
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
    },
    "model_train_kwargs": {
        "lr": 5e-4,
        "nepochs": 3001,
        "test_freq": 1000,
    },
    "model_eval_kwargs": {
        "crossval_folds": 5,
        "crossval_alpha": 1e-3,
        "crossval_train_sessions": False,
    },
    "overwrite": True,
    "notes": None,
    "seed": 0,
}

# %% Run Experiment sets


def validate_embedding_approach(exp_set="validate_embedding_approach"):
    """ """
    all_jobs = []
    for maze in MAZE_NAMES:
        # linear-in-linear-out (lin), onehot inputs, NO embedding
        lin_onehots_no_embed = copy.deepcopy(DEFAULT_EXPERIMENT)
        lin_onehots_no_embed["exp_name"] = f"all_subjects.{maze}.onehot_inputs_no_embedding"
        lin_onehots_no_embed["exp_set"] = exp_set
        lin_onehots_no_embed["with_embedding"] = False
        lin_onehots_no_embed["input_kwargs"]["subject_IDs"] = "all"
        lin_onehots_no_embed["input_kwargs"]["maze_name"] = maze
        # lin, product-space input, NO embedding
        lin_prodspace_no_embed = copy.deepcopy(lin_onehots_no_embed)
        lin_prodspace_no_embed["exp_name"] = f"all_subjects.{maze}.productspace_input_no_embedding"
        lin_prodspace_no_embed["input_kwargs"]["input_features"] = ["place_direction_distance"]
        # lin, onehot inputs, with embedding
        lin_onehots_with_embed = copy.deepcopy(lin_onehots_no_embed)
        lin_onehots_with_embed["exp_name"] = f"all_subjects.{maze}.onehot_inputs_with_linear_embedding"
        lin_onehots_with_embed["with_embedding"] = True
        lin_onehots_with_embed["model_init_kwargs"]["Nhid"] = []  # linear input to latent
        # lin, product-space input, with embedding
        lin_prodspace_with_embed = copy.deepcopy(lin_onehots_with_embed)
        lin_prodspace_with_embed["exp_name"] = f"all_subjects.{maze}.productspace_input_with_linear_embedding"
        lin_prodspace_with_embed["input_kwargs"]["input_features"] = ["place_direction_distance"]
        # nonlinear-in-linear-out (nonlin), onehot inputs, with embedding
        nonlin_onehots_with_embed = copy.deepcopy(lin_onehots_with_embed)
        nonlin_onehots_with_embed["exp_name"] = f"all_subjects.{maze}.onehot_inputs_with_nonlinear_embedding"
        nonlin_onehots_with_embed["model_init_kwargs"]["Nhid"] = [100, 50]  # nonlinear input to latent
        # add to all jobs
        all_jobs.extend(
            [
                lin_onehots_no_embed,
                lin_prodspace_no_embed,
                lin_onehots_with_embed,
                lin_prodspace_with_embed,
                nonlin_onehots_with_embed,
            ]
        )
    # submit all jobs
    for job in all_jobs:
        submit_job(job)


def validate_extra(exp_set="validate_embedding_approach"):
    """Delete after running"""
    all_jobs = []
    for maze in MAZE_NAMES:
        for subject in SUBJECT_IDS:
            # linear-in-linear-out (lin), onehot inputs, NO embedding
            lin_onehots_no_embed = copy.deepcopy(DEFAULT_EXPERIMENT)
            lin_onehots_no_embed["exp_name"] = f"{subject}.{maze}.onehot_inputs_no_embedding"
            lin_onehots_no_embed["exp_set"] = exp_set
            lin_onehots_no_embed["with_embedding"] = False
            lin_onehots_no_embed["input_kwargs"]["subject_IDs"] = [subject]
            lin_onehots_no_embed["input_kwargs"]["maze_name"] = maze
            # lin, product-space input, NO embedding
            lin_prodspace_no_embed = copy.deepcopy(lin_onehots_no_embed)
            lin_prodspace_no_embed["exp_name"] = f"{subject}.{maze}.productspace_input_no_embedding"
            lin_prodspace_no_embed["input_kwargs"]["input_features"] = ["place_direction_distance"]
            all_jobs.extend([lin_onehots_no_embed, lin_prodspace_no_embed])
    # submit all jobs
    for job in all_jobs:
        submit_job(job)
    return


# %%


def save_example_models(exp_set="example_models"):
    """ """
    all_jobs = []
    maze = "maze_1"
    z = 10
    for latent_nonlin, label in zip([None, "relu"], ["linear", "nonlinear"]):
        # product_space input
        prodspace = copy.deepcopy(DEFAULT_EXPERIMENT)
        prodspace["exp_set"] = exp_set
        prodspace["exp_name"] = f"all_subjects.{maze}.productspace_input_{label}_{z}_latents"
        prodspace["run_crossvalidation"] = False
        prodspace["train_full_model"] = True
        prodspace["input_kwargs"]["subject_IDs"] = "all"
        prodspace["input_kwargs"]["maze_name"] = maze
        prodspace["input_kwargs"]["input_features"] = ["place_direction_distance"]
        prodspace["model_init_kwargs"]["Nlat"] = z
        prodspace["model_init_kwargs"]["latent_nonlin"] = latent_nonlin
        # onehots input
        onehots = copy.deepcopy(prodspace)
        onehots["exp_name"] = f"all_subjects.{maze}.onehot_input_{label}_{z}_latents"
        onehots["input_kwargs"]["input_features"] = ["distance", "place_direction"]
        # add to all jobs
        all_jobs.extend([onehots])
    # submit all
    for job in all_jobs:
        submit_job(job)


def save_more_example_models(exp_set="example_models"):
    """ """
    maze = "maze_1"
    z = 10
    exp_1 = copy.deepcopy(DEFAULT_EXPERIMENT)
    exp_1["exp_set"] = exp_set
    exp_1["exp_name"] = f"all_subjects.{maze}.productspace_input_nonlinear_{z}_latents"
    exp_1["run_crossvalidation"] = False
    exp_1["train_full_model"] = True
    exp_1["input_kwargs"]["subject_IDs"] = "all"
    exp_1["input_kwargs"]["maze_name"] = maze
    exp_1["input_kwargs"]["input_features"] = ["distance", "place_direction"]
    exp_1["model_init_kwargs"]["Nlat"] = z
    exp_1["model_init_kwargs"]["Nhid"] = []
    exp_1["model_init_kwargs"]["latent_nonlin"] = None
    #
    exp_2 = copy.deepcopy(exp_1)
    exp_2["exp_name"] = f"all_subjects.{maze}.productspace_input_linear_{z}_latents_with_hid"
    exp_2["model_init_kwargs"]["Nhid"] = [100, 50]
    for job in [exp_1, exp_2]:
        submit_job(job)

    return
