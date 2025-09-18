""" """

# %% Imports
import os
import json
from copy import deepcopy
from jobs.nbeGLM import utils as ju

# %% Global variables
from GridMaze.paths import RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "nbeGLM"
# %% Functions


def submit_jobs(seed=0, subfolder="variance_explained_full2"):
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


def get_model_set_params(seed=0, subfolder="variance_explained_full2"):
    model_set_params = []
    all_input_groups = [
        "place_direction",
        "distance_to_goal",
        "egocentric_action",
        "goal",
        "egocentric_angle_to_goal",
        "allocentric_angle_to_goal",
        "speed",
        "acceleration",
        "head_direction",
    ]
    for maze_name in ["maze_1", "maze_2", "rooms_maze"]:
        for remove_groups, input_group_kwargs, model_name in [
            ([], {}, "full_model"),
            (["place_direction"], {}, "remove_place_direction"),
            (["distance_to_goal"], {}, "remove_distance_to_goal"),
            (["egocentric_action"], {}, "remove_egocentric_action_all"),
            (
                [],
                {"egocentric_action": {"components": ["action", "tower_bridge"]}},
                "remove_egocentric_action_free_forced",
            ),
            (
                [],
                {
                    "egocentric_action": {"components": ["free_forced", "tower_bridge"]},
                },
                "remove_egocentric_action_action",
            ),
            (["goal"], {}, "remove_goal"),
            (["egocentric_angle_to_goal"], {}, "remove_egocentric_angle_to_goal"),
            (["allocentric_angle_to_goal"], {}, "remove_allocentric_angle_to_goal"),
            (["speed"], {}, "remove_speed"),
            (["acceleration"], {}, "remove_acceleration"),
            (["head_direction"], {}, "remove_head_direction"),
            ### remove distance + other variables ###
            (["distance_to_goal", "place_direction"], {}, "remove_distance_to_goal_place_direction"),
            (["distance_to_goal", "egocentric_action"], {}, "remove_distance_to_goal_egocentric_action_all"),
            (
                ["distance_to_goal"],
                {"egocentric_action": {"components": ["action", "tower_bridge"]}},
                "remove_distance_to_goal_egocentric_action_free_forced",
            ),
            (
                ["distance_to_goal"],
                {
                    "egocentric_action": {"components": ["free_forced", "tower_bridge"]},
                },
                "remove_distance_to_goal_egocentric_action_action",
            ),
            (["distance_to_goal", "goal"], {}, "remove_distance_to_goal_goal"),
            (["distance_to_goal", "egocentric_angle_to_goal"], {}, "remove_distance_to_goal_egocentric_angle_to_goal"),
            (
                ["distance_to_goal", "allocentric_angle_to_goal"],
                {},
                "remove_distance_to_goal_allocentric_angle_to_goal",
            ),
            (["distance_to_goal", "speed"], {}, "remove_distance_to_goal_speed"),
            (["distance_to_goal", "acceleration"], {}, "remove_distance_to_goal_acceleration"),
            (["distance_to_goal", "head_direction"], {}, "remove_distance_to_goal_head_direction"),
        ]:
            # update defualt input data kwargs
            input_data_kwargs = deepcopy(ju.DEFAULT_INPUT_DATA_KWARGS)
            input_data_kwargs["maze_name"] = maze_name
            input_data_kwargs["input_groups"] = [group for group in all_input_groups if group not in remove_groups]
            input_data_kwargs["input_group_kwargs"] = input_group_kwargs
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
                    },
                    "resource_type": "gpu",
                    "run_fn": "run_cv_nbeGLM",
                }
            )

    return model_set_params
