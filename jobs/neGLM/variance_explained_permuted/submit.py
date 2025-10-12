""" """

# %% Imports
import os
import json
from copy import deepcopy
from jobs.neGLM import utils as ju

# %% Global variables

RESULTS_DIR = ju.RESULTS_DIR
# %% Functions


def run_permutation_jobs(subfolder="variance_explained_permuted", n_permutations=1):
    """ """
    # check if any permutations have already been run
    permutation_results = RESULTS_DIR / subfolder
    existing_perms = [eval(d.name) for d in permutation_results.iterdir() if d.is_dir()]
    max_perm = max(existing_perms) if len(existing_perms) > 0 else -1
    start_perm = max_perm + 1
    perms = range(start_perm, start_perm + n_permutations)
    # ensure all permutation folders exist
    for perm in perms:
        perm_path = RESULTS_DIR / subfolder / str(perm)
        if not perm_path.exists():
            perm_path.mkdir(parents=True, exist_ok=True)
    # run a set of spike permuted models for n_permutations
    for permutation in perms:
        model_set_params = get_model_set_params(seed=permutation, subfolder=subfolder, permutation=permutation)
        # save model set params to json
        with open(RESULTS_DIR / subfolder / str(permutation) / f"model_set_params.json", "w") as f:
            json.dump(model_set_params, f, indent=4)
        # write slurm script for each job/model and submit to cluster
        for model_params in model_set_params:
            script_path = ju.get_permutation_SLURM_script(**model_params)
            os.system(f"chmod +x {script_path}")
            os.system(f"sbatch {script_path}")
    return


def get_model_set_params(seed=0, subfolder="variance_explained_permuted", permutation=0):
    model_set_params = []
    all_input_groups = ["place_direction", "distance_to_goal"]
    for maze_name in ["maze_1", "maze_2", "rooms_maze"]:
        for remove_groups, input_group_kwargs, model_name in [
            ([], {}, "full_model"),
            (["place_direction"], {}, "remove_place_direction"),
            (["distance_to_goal"], {}, "remove_distance_to_goal"),
        ]:
            # update defualt input data kwargs
            input_data_kwargs = deepcopy(ju.DEFAULT_INPUT_DATA_KWARGS)
            input_data_kwargs["maze_name"] = maze_name
            input_data_kwargs["input_groups"] = [group for group in all_input_groups if group not in remove_groups]
            input_data_kwargs["input_group_kwargs"] = input_group_kwargs
            input_data_kwargs["permute_spikes"] = True
            # use defualt model init kwargs
            model_init_kwargs = deepcopy(ju.DEFAULT_MODEL_INIT_KWARGS)
            # use defualt model train kwargs
            model_train_kwargs = deepcopy(ju.DEFAULT_MODEL_TRAIN_KWARGS)
            model_train_kwargs["test_freq"] = 3000  # don't bother with monitoring training
            # use defualt score kwargs
            score_kwargs = deepcopy(ju.DEFAULT_SCORE_KWARGS)
            model_set_params.append(
                {
                    "model_name": model_name,
                    "subfolder": subfolder,
                    "maze_name": maze_name,
                    "permutation": permutation,
                    "model_params": {
                        "input_data_kwargs": input_data_kwargs,
                        "model_init_kwargs": model_init_kwargs,
                        "model_train_kwargs": model_train_kwargs,
                        "score_kwargs": score_kwargs,
                        "seed": seed,
                        "verbose": True,
                        "save_path": str(RESULTS_DIR / subfolder / str(permutation) / maze_name / model_name),
                    },
                    "resource_type": "gpu",
                    "run_fn": "run_cv_neGLM",
                }
            )
    return model_set_params
