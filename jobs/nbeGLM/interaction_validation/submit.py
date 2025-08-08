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


def submit_jobs(seed=0, subfolder="interaction_validation"):
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


def get_model_set_params(seed=0, subfolder="interaction_validation", overwrite=False):
    """ """
    model_set_params = []
    for maze_name in ["maze_1", "maze_2", "rooms_maze"]:
        for input_groups, partition, model_name in [
            (["place"], None, "place"),
            (["direction"], None, "direction"),
            (["place", "direction"], [("place",), ("direction",)], "place_direction_linear"),
            (["place", "direction"], None, "place_direction_nonlinear"),
            (["place_direction"], None, "place_direction_conjunction"),
            (
                ["place_direction", "distance_to_goal"],
                [("place_direction",), ("distance_to_goal",)],
                "place_direction_distance_to_goal_linear",
            ),
            (["distance_to_goal"], None, "distance_to_goal"),
            (
                ["place_direction", "distance_to_goal"],
                None,
                "place_direction_distance_to_goal_nonlinear",
            ),
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
                        "overwrite": overwrite,
                    },
                    "resource_type": "gpu",
                    "run_fn": "run_cv_nbeGLM",
                }
            )
    return model_set_params
