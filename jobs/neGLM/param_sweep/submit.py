""" """

# %% Imports
import os
import json
from copy import deepcopy
from jobs.neGLM import utils as ju

# %% Global variables

RESULTS_DIR = ju.RESULTS_DIR
# %% Functions


def submit_jobs(seed=0, subfolder="param_sweep"):
    model_set_params = get_model_set_params(seed, subfolder)
    # save model set params to json
    with open(RESULTS_DIR / subfolder / "model_set_params.json", "w") as f:
        json.dump(model_set_params, f, indent=4)

    # write slurm script for each job/model and submit to cluster
    for model_params in ju.find_missing(model_set_params):
        script_path = ju.get_SLURM_script(**model_params)
        os.system(f"chmod +x {script_path}")
        os.system(f"sbatch {script_path}")
    return print("all jobs submitted to hpc")


def get_model_set_params(seed=0, subfolder="param_sweep"):
    """
    hyperparameter sweep around defaults to justify chosen regime.
    maze_1 only, default input_groups (place_direction + distance_to_goal).
    one-at-a-time: baseline runs all defaults; each other model overrides a single kwarg.
    """
    maze_name = "maze_1"
    # (model_name, kind, value); kind in {None, "resolution", "Nhid", "Nlat", "beta_act", "beta_weight"}
    sweep_specs = [
        ("baseline", None, None),
        ("resolution_0.1", "resolution", 0.1),
        ("resolution_0.5", "resolution", 0.5),
        ("Nhid_50_25", "Nhid", [50, 25]),
        ("Nhid_200_100", "Nhid", [200, 100]),
        ("Nhid_150", "Nhid", [150]),
        ("Nhid_100", "Nhid", [100]),
        ("Nhid_50", "Nhid", [50]),
        ("Nlat_5", "Nlat", 5),
        ("Nlat_30", "Nlat", 30),
        ("beta_act_1e-3", "beta_act", 1e-3),
        ("beta_act_1e-2", "beta_act", 1e-2),
        ("beta_act_1e0", "beta_act", 1e0),
        ("beta_act_1e1", "beta_act", 1e1),
        ("beta_act_1e2", "beta_act", 1e2),
        ("beta_weight_1e-3", "beta_weight", 1e-3),
        ("beta_weight_1e-2", "beta_weight", 1e-2),
        ("beta_weight_1e0", "beta_weight", 1e0),
        ("beta_weight_1e1", "beta_weight", 1e1),
        ("beta_weight_1e2", "beta_weight", 1e2),
    ]

    model_set_params = []
    for model_name, kind, value in sweep_specs:
        # start from defaults
        input_data_kwargs = deepcopy(ju.DEFAULT_INPUT_DATA_KWARGS)
        input_data_kwargs["maze_name"] = maze_name
        model_init_kwargs = deepcopy(ju.DEFAULT_MODEL_INIT_KWARGS)
        model_train_kwargs = deepcopy(ju.DEFAULT_MODEL_TRAIN_KWARGS)
        score_kwargs = deepcopy(ju.DEFAULT_SCORE_KWARGS)

        # apply single override
        if kind == "resolution":
            input_data_kwargs["resolution"] = value
        elif kind in {"Nhid", "Nlat", "beta_act", "beta_weight"}:
            model_init_kwargs[kind] = value

        model_set_params.append(
            {
                "model_name": model_name,
                "subfolder": subfolder,
                "maze_name": maze_name,
                "model_params": {
                    "input_data_kwargs": input_data_kwargs,
                    "model_init_kwargs": model_init_kwargs,
                    "model_train_kwargs": model_train_kwargs,
                    "score_kwargs": score_kwargs,
                    "seed": seed,
                    "verbose": True,
                    "save_path": str(RESULTS_DIR / subfolder / maze_name / model_name),
                },
                "resource_type": "gpu",
                "run_fn": "run_cv_neGLM",
            }
        )

    return model_set_params
