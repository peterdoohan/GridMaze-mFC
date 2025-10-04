""" """

# %% Imports
import os
import json
from copy import deepcopy
from jobs.neGLM import utils as ju

# %% Global variables

RESULTS_DIR = ju.RESULTS_DIR
# %% Functions


def submit_jobs(seed=0, subfolder="angle_to_goal"):
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


def get_model_set_params(seed=0, subfolder="angle_to_goal"):
    """ """
    model_set_params = []
    for maze_name in ["maze_1", "maze_2", "rooms_maze"]:
        for input_groups, partition, model_name in [
            (  # baseline
                ["place_direction", "distance_to_goal"],
                None,
                "place_direction.distance_to_goal",
            ),
            (  # should add no extra perf with just goal
                ["place_direction", "distance_to_goal", "goal"],
                None,
                "place_direction.distance_to_goal.goal",
            ),
            (  # some extra variance w/ head-direction?
                ["place_direction", "distance_to_goal", "head_direction"],
                None,
                "place_direction.distance_to_goal.head_direction",
            ),
            (  # is it just head-direction is is prev model making some interactions?
                ["place_direction", "distance_to_goal", "head_direction"],
                [("place_direction",), ("distance_to_goal",), ("head_direction",)],
                "place_direction-distance_to_goal-head_direction",  # note factorised interation naming with "-"
            ),
            (  # does adding goal back allow model to learn angle to goal reps
                ["place_direction", "distance_to_goal", "goal", "head_direction"],
                None,
                "place_direction.distance_to_goal.goal.head_direction",
            ),
            (  # can any performance increases be explained by egocentric angle to goal
                ["place_direction", "distance_to_goal", "egocentric_angle_to_goal"],
                None,
                "place_direction.distance_to_goal.egocentric_angle_to_goal",
            ),
            (  # is it just angle to goal or is the prev model learning a vector to goal rep?
                ["place_direction", "distance_to_goal", "egocentric_angle_to_goal"],
                [("place_direction",), ("distance_to_goal",), ("egocentric_angle_to_goal",)],
                "place_direction-distance_to_goal-egocentric_angle_to_goal",
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
                    },
                    "resource_type": "gpu",
                    "run_fn": "run_cv_neGLM",
                }
            )
    return model_set_params
