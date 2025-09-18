""" """

# %% Imports
import os
import json
from copy import deepcopy
from jobs.nbeGLM import utils as ju

# %% Global variables

RESULTS_DIR = ju.RESULTS_DIR
# %% Functions


def submit_jobs(seed=0, subfolder="feature_interactions_full"):
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


def get_model_set_params(seed=0, subfolder="feature_interactions_full"):
    """ """
    model_set_params = []
    input_groups = ["place", "direction", "velocity", "distance_to_goal", "egocentric_action"]
    for maze_name in ["maze_1", "maze_2", "rooms_maze"]:
        for partition, model_name in [
            (None, "place.direction.velocity.distance_to_goal.egocentric_action"),  # full nonlinear
            (
                [("place",), ("direction",), ("velocity",), ("distance_to_goal",), ("egocentric_action",)],
                "place-direction-velocity-distance_to_goal-egocentric_action",
            ),  # fully factorised
            (
                [("place", "direction"), ("velocity",), ("distance_to_goal",), ("egocentric_action",)],
                "place.direction-velocity-distance_to_goal-egocentric_action",
            ),  # place-direction factorised
            (
                [
                    (
                        "place",
                        "direction",
                        "velocity",
                    ),
                    ("distance_to_goal",),
                    ("egocentric_action",),
                ],
                "place.direction.velocity-distance_to_goal-egocentric_action",
            ),  # place-direction-velocity factorised
        ]:
            # update defualt input data kwargs
            input_data_kwargs = deepcopy(ju.DEFAULT_INPUT_DATA_KWARGS)
            input_data_kwargs["maze_name"] = maze_name
            input_data_kwargs["input_groups"] = input_groups
            # update defualt model init kwargs
            model_init_kwargs = deepcopy(ju.DEFAULT_MODEL_INIT_KWARGS)
            model_init_kwargs["partition"] = partition
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
