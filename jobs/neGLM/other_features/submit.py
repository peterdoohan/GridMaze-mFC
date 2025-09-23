""" """

# %% Imports
import os
import json
from copy import deepcopy
from jobs.neGLM import utils as ju

# %% Global variables

RESULTS_DIR = ju.RESULTS_DIR
# %% Functions


def submit_jobs(seed=0, subfolder="other_features"):
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


def get_model_set_params(seed=0, subfolder="other_features", overwrite=False):
    """ """
    model_set_params = []
    feature_groups = ["place_direction", "distance_to_goal", "goal", "egocentric_action", "velocity"]
    for maze_name in ["maze_1", "maze_2", "rooms_maze"]:
        input_features = []
        for added_feat in feature_groups:
            input_features.append(added_feat)
            model_name = ".".join(input_features)
            # update defualt input data kwargs
            input_data_kwargs = deepcopy(ju.DEFAULT_INPUT_DATA_KWARGS)
            input_data_kwargs["maze_name"] = maze_name
            input_data_kwargs["input_groups"] = input_features.copy()
            input_data_kwargs["input_group_kwargs"] = {}
            # use defualt model init kwargs
            model_init_kwargs = deepcopy(ju.DEFAULT_MODEL_INIT_KWARGS)
            # use defualt model train kwargs
            model_train_kwargs = deepcopy(ju.DEFAULT_MODEL_TRAIN_KWARGS)
            # use defualt score kwargs
            score_kwargs = deepcopy(ju.DEFAULT_SCORE_KWARGS)
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
                        "overwrite": overwrite,
                    },
                    "resource_type": "gpu",
                    "run_fn": "run_cv_neGLM",
                }
            )
    # also run models upto the inclusion of goal, where goal is coded differently
    base_features = ["place_direction", "distance_to_goal"]
    for maze_name in ["maze_1", "maze_2", "rooms_maze"]:
        for goal_feat in ["egocentric_angle_to_goal", "allocentric_angle_to_goal"]:
            input_features = base_features + [goal_feat]
            model_name = ".".join(input_features)
            # update defualt input data kwargs
            input_data_kwargs = deepcopy(ju.DEFAULT_INPUT_DATA_KWARGS)
            input_data_kwargs["maze_name"] = maze_name
            input_data_kwargs["input_groups"] = input_features
            input_data_kwargs["input_group_kwargs"] = {}
            # use defualt model init kwargs
            model_init_kwargs = deepcopy(ju.DEFAULT_MODEL_INIT_KWARGS)
            # use defualt model train kwargs
            model_train_kwargs = deepcopy(ju.DEFAULT_MODEL_TRAIN_KWARGS)
            # use defualt score kwargs
            score_kwargs = deepcopy(ju.DEFAULT_SCORE_KWARGS)
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
                        "overwrite": overwrite,
                    },
                    "resource_type": "gpu",
                    "run_fn": "run_cv_neGLM",
                }
            )
    # also run a speed and velocity comparison
    feature_groups = ["place_direction", "distance_to_goal", "goal", "egocentric_action", "speed"]
    for maze_name in ["maze_1", "maze_2", "rooms_maze"]:
        model_name = ".".join(feature_groups)
        # update defualt input data kwargs
        input_data_kwargs = deepcopy(ju.DEFAULT_INPUT_DATA_KWARGS)
        input_data_kwargs["maze_name"] = maze_name
        input_data_kwargs["input_groups"] = feature_groups
        input_data_kwargs["input_group_kwargs"] = {}
        # use defualt model init kwargs
        model_init_kwargs = deepcopy(ju.DEFAULT_MODEL_INIT_KWARGS)
        # use defualt model train kwargs
        model_train_kwargs = deepcopy(ju.DEFAULT_MODEL_TRAIN_KWARGS)
        # use defualt score kwargs
        score_kwargs = deepcopy(ju.DEFAULT_SCORE_KWARGS)
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
                    "overwrite": overwrite,
                },
                "resource_type": "gpu",
                "run_fn": "run_cv_neGLM",
            }
        )
    return model_set_params
