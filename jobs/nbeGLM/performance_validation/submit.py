""" """

# %% Imports
import os
import json
from pathlib import Path
from copy import deepcopy
from jobs.nbeGLM import utils as ju

# %% Global variables
from GridMaze.paths import RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "nbeGLM"
# %% Functions


def submit_jobs(seed=0, subfolder="performance_validation"):
    model_set_params = get_model_set_params(seed, subfolder)
    model_set_path = RESULTS_DIR / subfolder
    model_set_path.mkdir(parents=True, exist_ok=True)
    # save model set params to json
    with open(model_set_path / "model_set_params.json", "w") as f:
        json.dump(model_set_params, f, indent=4)
    # write slurm script for each job/model and submit to cluster
    for model_params in find_missing(model_set_params):
        script_path = ju.get_SLURM_script(**model_params)
        os.system(f"chmod +x {script_path}")
        os.system(f"sbatch {script_path}")
    return print("all jobs submitted to hpc")


def find_missing(model_set_params):
    missing = []
    for model_params in model_set_params:
        save_path = Path(model_params["model_params"]["save_path"])
        if not (save_path / "DONE.txt").exists():
            missing.append(model_params)
    return missing


def get_model_set_params(seed=0, subfolder="performance_validation", overwrite=False):
    """
    generate a list of dicts (.jsons) that define all the models to compare for the nbeGLM validation figure
    """
    model_set_params = []
    for maze_name in ["maze_1", "maze_2", "rooms_maze"]:
        # with embedding condition
        fn = "run_cv_nbeGLM"
        for input_groups, input_group_kwargs, model_name in [
            (
                ["direction"],
                {},
                "embedding_direction",
            ),
            (
                ["place"],
                {},
                "embedding_place",
            ),
            (
                ["direction", "place"],
                {},
                "embedding_place_direction",
            ),
            (
                ["direction", "place", "distance_to_goal"],
                {},
                "embedding_place_direction_distance_to_goal",
            ),
            (
                ["direction", "place", "distance_to_goal", "egocentric_action"],
                {},
                "embedding_place_direction_distance_to_goal_egocentric_action",
            ),
        ]:
            # update defualt input data kwargs
            input_data_kwargs = deepcopy(ju.DEFAULT_INPUT_DATA_KWARGS)
            input_data_kwargs["maze_name"] = maze_name
            input_data_kwargs["input_groups"] = input_groups
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
                        "overwrite": overwrite,
                    },
                    "resource_type": "gpu",
                    "run_fn": fn,
                }
            )
        # without embedding condition
        fn = "run_cv_baselineGLM"
        for input_groups, input_group_kwargs, model_name in [
            (
                ["direction"],
                {},
                "baseline_direction",
            ),
            (
                ["place"],
                {},
                "baseline_place",
            ),
            (
                ["place_direction"],
                {},
                "baseline_place_direction",
            ),
            (
                ["place_direction_distance_to_goal"],
                {"place_direction_distance_to_goal": {"keep_only_visited": True}},
                "baseline_place_direction_distance_to_goal",
            ),
            (
                ["place_direction_distance_to_goal_egocentric_action"],
                {"place_direction_distance_to_goal_egocentric_action": {"keep_only_visited": True}},
                "baseline_place_direction_distance_to_goal_egocentric_action",
            ),
        ]:
            # update defualt input data kwargs
            input_data_kwargs = deepcopy(ju.DEFAULT_INPUT_DATA_KWARGS)
            input_data_kwargs["maze_name"] = maze_name
            input_data_kwargs["input_groups"] = input_groups
            input_data_kwargs["input_group_kwargs"] = input_group_kwargs
            # use defualt score kwargs
            score_kwargs = deepcopy(ju.DEFAULT_SCORE_KWARGS)
            model_set_params.append(
                {
                    "model_name": model_name,
                    "subfolder": subfolder,
                    "maze_name": maze_name,
                    "model_params": {
                        "input_data_kwargs": input_data_kwargs,
                        "score_kwargs": score_kwargs,
                        "seed": seed,
                        "verbose": True,
                        "save_path": str(RESULTS_DIR / subfolder / maze_name / model_name),
                    },
                    "resource_type": "cpu",
                    "run_fn": fn,
                }
            )

    return model_set_params
