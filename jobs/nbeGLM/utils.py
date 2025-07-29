""" """

# %% Imports


# %% Global Variables

DEFAULT_CV_NBEGLM_PARAMS = {
    "input_data_kwargs": {
        "subject_IDs": "all",
        "maze_name": "maze_1",
        "days_on_maze": "all",
        "input_features": ["place-direction", "distance_to_goal", "egocentric_action"],
        "input_feature_kwargs": {},
        "resolution": 0.1,
        "max_steps_to_goal": 30,
        "min_spike_count": 300,
        "moving_only": False,
    },
    "model_init_kwargs": {
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
    },
    "model_train_kwargs": {
        "lr": 5e-4,
        "nepochs": 3001,
        "test_freq": 1000,
        "eval_alpha": 1e-3,
    },
    "model_eval_kwargs": {
        "crossval_folds": 5,
        "crossval_alpha": 1e-3,
        "crossval_train_sessions": False,
    },
    "seed": 0,
    "overwrite": True,
    "verbose": True,
}

# %% Functions


def get_SLURM_script(exp_name, subfolder, model_params, run_fn="run_cv_nbeGLM"):
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
