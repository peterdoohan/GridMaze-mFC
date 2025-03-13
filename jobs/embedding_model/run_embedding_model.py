"""
Script for submitting and jobs to the HPC that run experiments with the embedding model descirbed
in GridMaze.analysis.embedding_model 
"""

# %% Imports
import os
import copy
import json
from pathlib import Path
import getpass

# %% Global variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

EMBEDDING_MODEL_RESULTS = RESULTS_PATH / "embedding_model" / "exps"

with open(Path(EXPERIMENT_INFO_PATH) / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]

# can use for testing
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
"""

    if getpass.getuser() == "kjensen":  # sneaky
        script += f"""\nsource ~/.bashrc\n"""

    else:
        script += f"""\nmodule load miniconda
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


# %% OLD EXPS


def run_one_vs_all_subject_experiments():
    """
    TODO: Need to rerun with new model config
    """
    for maze in MAZE_NAMES:
        # single subject experiments
        single_sub_exp = DEFAULT_EXPERIMENT.copy()
        for subject in SUBJECT_IDS:
            single_sub_exp["exp_name"] = f"{subject}_{maze}_state_action_distance"
            single_sub_exp["input_kwargs"]["subject_IDs"] = [subject]
            single_sub_exp["input_kwargs"]["maze_name"] = maze
            submit_job(single_sub_exp)
        # all subject exp
        all_sub_exp = single_sub_exp.copy()
        all_sub_exp["exp_name"] = f"all_subjects_{maze}_state_action_distance"
        all_sub_exp["input_kwargs"]["subject_IDs"] = "all"
        submit_job(all_sub_exp)
    return


# %% EXPS


def full_state_action_distance(exp_set="example_models"):
    """ """
    all_jobs = []
    for maze in ["maze_1", "maze_2"]:
        exp = copy.deepcopy(DEFAULT_EXPERIMENT)
        exp["exp_set"] = exp_set
        exp["run_crossvalidation"] = False
        exp["train_full_model"] = True
        exp["input_kwargs"]["subject_IDs"] = "all"
        exp["input_kwargs"]["maze_name"] = maze
        exp["input_kwargs"]["input_features"] = ["place_direction", "distance"]
        for z in [5, 10, 15, 20]:
            _exp = copy.deepcopy(exp)
            _exp["exp_name"] = f"{maze}_state_action_distance_{z}_latents"
            _exp["model_init_kwargs"]["Nlat"] = z
            all_jobs.append(_exp)
    # submit all
    for job in all_jobs:
        submit_job(job)


# %%


def egocentric_action_vs_angle_to_goal(maze_name="maze_1", subject_IDs=["m2"], exp_set="egocentricaction_vs_angle"):
    """ """
    all_exps = []
    for subject in subject_IDs:
        # full model
        full = copy.deepcopy(DEFAULT_EXPERIMENT)
        full["exp_set"] = exp_set
        full["exp_name"] = f"{subject}_{maze_name}_full_model"
        full["input_kwargs"]["subject_IDs"] = [subject]
        full["input_kwargs"]["maze_name"] = maze_name
        full["input_kwargs"]["input_features"] = [
            "place_direction",
            "distance",
            "tower_bridge",
            "left_right_turns",
            "egocentric_angle_to_goal",
        ]
        # reduced: LR
        reduced_LR = copy.deepcopy(full)
        reduced_LR["exp_name"] = f"{subject}_{maze_name}_reduced_LR"
        reduced_LR["input_kwargs"]["input_features"] = [
            "place_direction",
            "distance",
            "tower_bridge",
            "egocentric_angle_to_goal",
        ]
        # reduced: angle
        reduced_angle = copy.deepcopy(full)
        reduced_angle["exp_name"] = f"{subject}_{maze_name}_reduced_angle"
        reduced_angle["input_kwargs"]["input_features"] = [
            "place_direction",
            "distance",
            "tower_bridge",
            "left_right_turns",
        ]
        # reduced: LR & angle
        reduced_LR_angle = copy.deepcopy(full)
        reduced_LR_angle["exp_name"] = f"{subject}_{maze_name}_reduced_LR_angle"
        reduced_LR_angle["input_kwargs"]["input_features"] = ["place_direction", "distance", "tower_bridge"]
        all_exps.extend([full, reduced_LR, reduced_angle, reduced_LR_angle])
    # submit all
    for exp in all_exps:
        submit_job(exp)
    return


# %%


def run_egocentric_action_experiments(maze_name="maze_1", subject_IDs=["m2"], exp_set="egocentric_action2"):
    """
    Regressors:
    - tower_bridge: nuissance regressor (control for turns only defined at towers)
    - left_right_turns: left/right turn regressors
    - free_forced_choice: free/forced choice regressors
    """
    all_exps = []
    for subject in subject_IDs:
        # egocentric-only full model
        full = copy.deepcopy(DEFAULT_EXPERIMENT)  # default nonlin (Poission Exp) w/ embedding & 20 latents (z)
        full["exp_set"] = exp_set
        full["exp_name"] = f"{subject}_{maze_name}_full_model"
        full["input_kwargs"]["subject_IDs"] = [subject]
        full["input_kwargs"]["maze_name"] = maze_name
        full["input_kwargs"]["input_features"] = ["tower_bridge", "left_right_turns", "free_forced_choice"]
        # reduced: left_right
        reduced_LR = copy.deepcopy(full)
        reduced_LR["exp_name"] = f"{subject}_{maze_name}_reduced_LR"
        reduced_LR["input_kwargs"]["input_features"] = ["tower_bridge", "free_forced_choice"]
        # reduced: free_forced
        reduced_choice = copy.deepcopy(full)
        reduced_choice["exp_name"] = f"{subject}_{maze_name}_reduced_choice"
        reduced_choice["input_kwargs"]["input_features"] = ["tower_bridge", "left_right_turns"]
        # reduced all
        reduced_all = copy.deepcopy(full)
        reduced_all["exp_name"] = f"{subject}_{maze_name}_reduced_all"
        reduced_all["input_kwargs"]["input_features"] = ["tower_bridge"]
        # SAD full model (does egocentric tuning explain variance above the state-action + distance model (SAD))
        SAD_full = copy.deepcopy(full)
        SAD_full["exp_name"] = f"{subject}_{maze_name}_SAD_full_model"
        SAD_full["input_kwargs"]["input_features"] = [
            "place_direction",
            "distance",
            "tower_bridge",
            "left_right_turns",
            "free_forced_choice",
        ]
        # SAD+ reduced: left_right
        SAD_reduced_LR = copy.deepcopy(SAD_full)
        SAD_reduced_LR["exp_name"] = f"{subject}_{maze_name}_SAD_reduced_LR"
        SAD_reduced_LR["input_kwargs"]["input_features"] = [
            "place_direction",
            "distance",
            "tower_bridge",
            "free_forced_choice",
        ]
        # SAD+ reduced: free_forced
        SAD_reduced_choice = copy.deepcopy(SAD_full)
        SAD_reduced_choice["exp_name"] = f"{subject}_{maze_name}_SAD_reduced_choice"
        SAD_reduced_choice["input_kwargs"]["input_features"] = [
            "place_direction",
            "distance",
            "tower_bridge",
            "left_right_turns",
        ]
        # SAD+ reduced: left_right & free_forced
        SAD_reduced_LR_choice = copy.deepcopy(SAD_full)
        SAD_reduced_LR_choice["exp_name"] = f"{subject}_{maze_name}_SAD_reduced_LR_choice"
        SAD_reduced_LR_choice["input_kwargs"]["input_features"] = ["place_direction", "distance", "tower_bridge"]
        # SAD+ reduced: sate-action
        SAD_reduced_SA = copy.deepcopy(SAD_full)
        SAD_reduced_SA["exp_name"] = f"{subject}_{maze_name}_SAD_reduced_SA"
        SAD_reduced_SA["input_kwargs"]["input_features"] = [
            "distance",
            "tower_bridge",
            "left_right_turns",
            "free_forced_choice",
        ]
        # SAD+ reduced: distane_to_goal
        SAD_reduced_D = copy.deepcopy(SAD_full)
        SAD_reduced_D["exp_name"] = f"{subject}_{maze_name}_SAD_reduced_D"
        SAD_reduced_D["input_kwargs"]["input_features"] = [
            "place_direction",
            "tower_bridge",
            "left_right_turns",
            "free_forced_choice",
        ]
        # add to jobs
        all_exps.extend(
            [SAD_full, SAD_reduced_LR, SAD_reduced_choice, SAD_reduced_LR_choice, SAD_reduced_SA, SAD_reduced_D]
        )
    # submit all
    for exp in all_exps:
        submit_job(exp)
    return


def run_full_linear_place_direction_distance(
    maze_name="maze_1", subject_IDs=["m2"], exp_set="state-action_distance_full_linear2"
):
    """ """
    # TODO: build exp set
    all_exps = []
    for subject in subject_IDs:
        # lin product space
        prodspace = copy.deepcopy(DEFAULT_EXPERIMENT)
        prodspace["exp_set"] = exp_set
        prodspace["exp_name"] = f"{subject}_{maze_name}_product-space_full_linear"
        prodspace["input_kwargs"]["subject_IDs"] = [subject]
        prodspace["input_kwargs"]["maze_name"] = maze_name
        prodspace["input_kwargs"]["input_features"] = ["place_direction_distance"]
        prodspace["model_init_kwargs"]["Nhid"] = []
        prodspace["model_init_kwargs"]["noise_function"] = "Gaussian"
        prodspace["model_init_kwargs"]["inv_link"] = "identity"
        prodspace["model_init_kwargs"]["sqrt_counts"] = True
        # lin one-hot features
        onehots = copy.deepcopy(prodspace)
        onehots["exp_name"] = f"{subject}_{maze_name}_onehots_full_linear"
        onehots["input_kwargs"]["input_features"] = ["place_direction", "distance"]
        # nonlin control
        nonlin = copy.deepcopy(onehots)
        nonlin["exp_name"] = f"{subject}_{maze_name}_nonlin"
        nonlin["model_init_kwargs"]["Nhid"] = [250, 250, 150, 50]
        all_exps.extend([prodspace, onehots, nonlin])
    # submit all
    for exp in all_exps:
        submit_job(exp)


# %% EXPSs


def run_state_action_experiments(maze_name, subject_IDs=SUBJECT_IDS, exp_set="state_action_interactions"):
    """
    Exps naming convention: subject_maze_inv-link_features_interaction-type (optional)
    """
    # define experiments
    all_exps = []
    for subject in subject_IDs:
        for inv_link in ["softplus", "exp"]:
            # state-action linear
            linear = copy.deepcopy(DEFAULT_EXPERIMENT)
            linear["exp_name"] = f"{subject}_{maze_name}_{inv_link}_state_action_linear"
            linear["exp_set"] = exp_set
            linear["input_kwargs"]["subject_IDs"] = [subject]
            linear["with_embedding"] = True
            linear["input_kwargs"]["maze_name"] = maze_name
            linear["input_kwargs"]["input_features"] = ["place", "direction"]
            linear["model_init_kwargs"]["noise_function"] = "Poisson"
            linear["model_init_kwargs"]["inv_link"] = inv_link
            linear["model_init_kwargs"]["partition"] = [(0,), (1,)]
            # state-action non-linear
            nonlinear = copy.deepcopy(linear)
            nonlinear["exp_name"] = f"{subject}_{maze_name}_{inv_link}_state_action_nonlinear"
            nonlinear["model_init_kwargs"]["partition"] = None
            # state-action conjunctive
            conjunctive = copy.deepcopy(nonlinear)
            conjunctive["exp_name"] = f"{subject}_{maze_name}_{inv_link}_state_action_conjunctive"
            conjunctive["input_kwargs"]["input_features"] = ["place_direction"]
            # just state
            state = copy.deepcopy(linear)
            state["exp_name"] = f"{subject}_{maze_name}_{inv_link}_state"
            state["input_kwargs"]["input_features"] = ["place"]
            state["model_init_kwargs"]["partition"] = None
            # just action exp
            action = copy.deepcopy(state)
            action["exp_name"] = f"{subject}_{maze_name}_{inv_link}_action"
            action["input_kwargs"]["input_features"] = ["direction"]
            all_exps.extend([linear, nonlinear, conjunctive, state, action])
        # no embedding controls
        state_action_no_embedding = copy.deepcopy(linear)
        state_action_no_embedding["with_embedding"] = False
        state_action_no_embedding["exp_name"] = f"{subject}_{maze_name}_state_action_no-embedding"
        state_no_embedding = copy.deepcopy(state)
        state_no_embedding["with_embedding"] = False
        state_no_embedding["exp_name"] = f"{subject}_{maze_name}_state_no-embedding"
        action_no_embedding = copy.deepcopy(action)
        action_no_embedding["with_embedding"] = False
        action_no_embedding["exp_name"] = f"{subject}_{maze_name}_action_no-embedding"
        all_exps.extend([state_action_no_embedding, state_no_embedding, action_no_embedding])
    # submit all
    for exp in all_exps:
        submit_job(exp)


def run_state_action_distance_experiments(
    maze_name, subject_IDs=SUBJECT_IDS, exp_set="state-action_distance_interactions"
):
    """
    Exps naming convention: subject_maze_inv-link_features_interaction-type (optional)
    """
    all_exps = []
    for subject in subject_IDs:
        for inv_link in ["softplus", "exp"]:
            # approx linear exp
            approx_linear = copy.deepcopy(DEFAULT_EXPERIMENT)
            approx_linear["exp_name"] = f"{subject}_{maze_name}_{inv_link}_state-action_distance_linear"
            approx_linear["exp_set"] = exp_set
            approx_linear["input_kwargs"]["subject_IDs"] = [subject]
            approx_linear["with_embedding"] = True
            approx_linear["input_kwargs"]["maze_name"] = maze_name
            approx_linear["input_kwargs"]["input_features"] = ["place_direction", "distance"]
            approx_linear["model_init_kwargs"]["noise_function"] = "Poisson"
            approx_linear["model_init_kwargs"]["inv_link"] = inv_link
            approx_linear["model_init_kwargs"]["partition"] = [(0,), (1,)]
            # nonlinear exp
            nonlinear = copy.deepcopy(approx_linear)
            nonlinear["exp_name"] = f"{subject}_{maze_name}_{inv_link}_state-action_distance_nonlinear"
            nonlinear["model_init_kwargs"]["partition"] = None
            # just state-action exp
            state_action = copy.deepcopy(approx_linear)
            state_action["exp_name"] = f"{subject}_{maze_name}_{inv_link}_state-action"
            state_action["input_kwargs"]["input_features"] = ["place_direction"]
            state_action["model_init_kwargs"]["partition"] = None
            # just distance exp
            distance = copy.deepcopy(state_action)
            distance["exp_name"] = f"{subject}_{maze_name}_{inv_link}_distance"
            distance["input_kwargs"]["input_features"] = ["distance"]
            # all_exps.extend([linear, nonlinear, state_action, distance])

        # no embedding controls
        state_action_no_embedding = copy.deepcopy(state_action)
        state_action_no_embedding["with_embedding"] = False
        state_action_no_embedding["exp_name"] = f"{subject}_{maze_name}_state-action_no-embedding"
        distance_no_embedding = copy.deepcopy(distance)
        distance_no_embedding["with_embedding"] = False
        distance_no_embedding["exp_name"] = f"{subject}_{maze_name}_distance_no-embedding"
        # full product space no embbedding control
        state_action_distance_no_embedding = copy.deepcopy(approx_linear)
        state_action_distance_no_embedding["with_embedding"] = False
        state_action_distance_no_embedding["exp_name"] = f"{subject}_{maze_name}_state-action-distance_no-embedding"
        state_action_distance_no_embedding["input_kwargs"]["input_features"] = ["place_direction_distance"]
        state_action_distance_no_embedding["model_init_kwargs"][
            "partition"
        ] = None  # need to specify to initi Encoder or will throw error
        # all_exps.extend([state_action_no_embedding, distance_no_embedding, state_action_distance_no_embedding])
        all_exps.extend([state_action_no_embedding])

        # linear & nonlinear gaussian exps
        linear_gaussian = copy.deepcopy(approx_linear)
        linear_gaussian["exp_name"] = f"{subject}_{maze_name}_state-action_distance_linear-gaussian"
        linear_gaussian["model_init_kwargs"]["noise_function"] = "Gaussian"
        linear_gaussian["model_init_kwargs"]["inv_link"] = "identity"
        nonlinear_gaussian = copy.deepcopy(linear_gaussian)
        nonlinear_gaussian["exp_name"] = f"{subject}_{maze_name}_state-action_distance_nonlinear-gaussian"
        nonlinear_gaussian["model_init_kwargs"]["partition"] = None
        state_action_gaussian = copy.deepcopy(nonlinear_gaussian)
        state_action_gaussian["exp_name"] = f"{subject}_{maze_name}_state-action_gaussian"
        state_action_gaussian["input_kwargs"]["input_features"] = ["place_direction"]
        distance_gaussian = copy.deepcopy(nonlinear_gaussian)
        distance_gaussian["exp_name"] = f"{subject}_{maze_name}_distance_gaussian"
        distance_gaussian["input_kwargs"]["input_features"] = ["distance"]
        all_exps.extend([linear_gaussian, nonlinear_gaussian, state_action_gaussian, distance_gaussian])
    # submit all
    for exp in all_exps:
        submit_job(exp)
    return


def run_all_inputs_experiments(maze_name, subject_IDs=SUBJECT_IDS, exp_set="all_inputs"):
    all_exps = []
    for subject in subject_IDs:
        subject_exp = copy.deepcopy(DEFAULT_EXPERIMENT)
        subject_exp["exp_set"] = exp_set
        subject_exp["exp_name"] = f"{subject}_all_inputs"
        subject_exp["input_kwargs"]["subject_IDs"] = [subject]
        subject_exp["input_kwargs"]["maze_name"] = maze_name
        subject_exp["with_embedding"] = True
        subject_exp["input_kwargs"]["input_features"] = [
            "place_direction",
            "distance",
            "goal",
            "speed",
            "acceleration",
            "head_direction",
        ]
        subject_exp["model_init_kwargs"]["noise_function"] = "Poisson"
        subject_exp["model_init_kwargs"]["inv_link"] = "exp"
        subject_exp["model_init_kwargs"]["partition"] = None
        all_exps.append(subject_exp)
    # submit all
    for exp in all_exps:
        submit_job(exp)


# %% Dev

# %% Pseudo R2 experiments


def run_var_explained_experiments(maze_name, subject_IDs=SUBJECT_IDS, exp_set="var_explained2"):
    all_exps = []
    for subject in subject_IDs:
        # set up full model
        full_model = copy.deepcopy(DEFAULT_EXPERIMENT)
        full_model["exp_set"] = exp_set
        full_model["exp_name"] = f"{subject}_{maze_name}_full_model"
        full_model["with_embedding"] = True
        full_model["input_kwargs"]["subject_IDs"] = [subject]
        full_model["input_kwargs"]["maze_name"] = maze_name
        full_model["input_kwargs"]["days_on_maze"] = [
            7,
            8,
            9,
            10,
            11,
            12,
            13,
        ]  # need to update "late" to be last 5 sessions
        full_model["input_kwargs"]["input_features"] = ["trial_phase", "place_direction", "distance", "goal"]
        full_model["input_kwargs"]["navigation_only"] = False
        full_model["input_kwargs"]["moving_only"] = False
        full_model["model_init_kwargs"]["noise_function"] = "Poisson"
        full_model["model_init_kwargs"]["inv_link"] = "exp"
        full_model["model_init_kwargs"]["partition"] = [(0, 1), (2, 3)]
        full_model["model_init_kwargs"]["Nlat"] = 20
        full_model["model_init_kwargs"]["Nhid"] = [250, 250, 150, 50]
        full_model["model_eval_kwargs"]["crossval_folds"] = 10  # more folds to do t-test per cluster
        # define reduced models
        reduced_trial_phase = copy.deepcopy(full_model)
        reduced_trial_phase["exp_name"] = f"{subject}_{maze_name}_reduced_trial_phase"
        reduced_trial_phase["input_kwargs"]["input_features"] = ["place_direction", "distance", "goal"]
        reduced_trial_phase["model_init_kwargs"]["partition"] = [(0,), (1, 2)]
        reduced_state_action = copy.deepcopy(full_model)
        reduced_state_action["exp_name"] = f"{subject}_{maze_name}_reduced_place_direction"
        reduced_state_action["input_kwargs"]["input_features"] = ["trial_phase", "distance", "goal"]
        reduced_state_action["model_init_kwargs"]["partition"] = [(0,), (1, 2)]
        reduced_distance = copy.deepcopy(full_model)
        reduced_distance["exp_name"] = f"{subject}_{maze_name}_reduced_distance"
        reduced_distance["input_kwargs"]["input_features"] = ["trial_phase", "place_direction", "goal"]
        reduced_distance["model_init_kwargs"]["partition"] = [(0, 1), (2,)]
        reduced_goal = copy.deepcopy(full_model)
        reduced_goal["exp_name"] = f"{subject}_{maze_name}_reduced_goal"
        reduced_goal["input_kwargs"]["input_features"] = ["trial_phase", "place_direction", "distance"]
        reduced_goal["model_init_kwargs"]["partition"] = [(0, 1), (2,)]
        all_exps.extend([full_model, reduced_trial_phase, reduced_state_action, reduced_distance, reduced_goal])
    # submit all
    for exp in all_exps:
        submit_job(exp)


# %% Distance metric comparison experiment


def run_distance_metrics_experiments(maze_name, subject_IDs=SUBJECT_IDS, exp_set="distance_metric_comparison"):
    """ """
    # define base model for comparing distance metrics
    base_model = copy.deepcopy(DEFAULT_EXPERIMENT)
    base_model["exp_set"] = exp_set
    base_model["with_embedding"] = True
    base_model["input_kwargs"]["maze_name"] = maze_name
    base_model["input_kwargs"]["days_on_maze"] = [7, 8, 9, 10, 11, 12, 13]
    base_model["input_kwargs"]["input_features"] = ["place_direction", "distance"]
    base_model["input_kwargs"]["max_distance"] = 1.8
    base_model["model_init_kwargs"]["partition"] = [(0,), (1,)]
    base_model["model_init_kwargs"]["noise_function"] = "Poisson"
    base_model["model_init_kwargs"]["inv_link"] = "exp"
    all_exps = []
    for subject in subject_IDs:
        for distance_metric in ["euclidean", "geodesic", "manhattan", "future"]:
            exp = copy.deepcopy(base_model)
            exp["exp_name"] = f"{subject}_{maze_name}_distance_to_goal_{distance_metric}"
            exp["input_kwargs"]["subject_IDs"] = [subject]
            exp["input_kwargs"]["distance_metrics"] = ("distance_to_goal", distance_metric)
            # all_exps.append(exp)
        for progress_metric in ["time", "path_length"]:
            exp = copy.deepcopy(base_model)
            exp["exp_name"] = f"{subject}_{maze_name}_progress_to_goal_{progress_metric}"
            exp["input_kwargs"]["subject_IDs"] = [subject]
            exp["input_kwargs"]["distance_metrics"] = ("progress_to_goal", progress_metric)
            all_exps.append(exp)
    # submit all
    for exp in all_exps:
        submit_job(exp)
    return


# %% Misc Experiments


def run_embedding_trouble_shoot_experiment(exp_set="embedding_trouble_shoot"):
    """
    Run extra tests
    """
    all_exps = []
    # set up single subject experiment
    single_sub_exp = copy.deepcopy(DEFAULT_EXPERIMENT)
    single_sub_exp["input_kwargs"]["subject_IDs"] = ["m2"]
    single_sub_exp["exp_set"] = exp_set
    single_sub_exp["input_kwargs"]["maze_name"] = "maze_1"
    single_sub_exp["input_kwargs"]["input_features"] = ["place_direction"]
    single_sub_exp["model_init_kwargs"]["noise_function"] = "Poisson"
    single_sub_exp["model_init_kwargs"]["inv_link"] = "exp"
    single_sub_exp["model_init_kwargs"]["partition"] = None
    # control exp without embedding
    no_embedding = copy.deepcopy(single_sub_exp)
    no_embedding["with_embedding"] = False
    no_embedding["exp_name"] = "state-action_no_embedding"
    no_embedding["input_kwargs"]["days_on_maze"] = [7, 8, 9, 10, 11, 12, 13]  # last 7 days
    all_exps.append(no_embedding)
    # try embedding with different number of late sessions going into learning embedding
    for days, label in zip(
        [
            [7, 8, 9, 10, 11, 12, 13],
            [9, 10, 11, 12, 13],
            [11, 12, 13],
            [7, 8, 9, 10, 11],
        ],
        ["last_7", "last_5", "last_3", "late_24_goals"],
    ):
        exp = copy.deepcopy(single_sub_exp)
        exp["exp_name"] = f"state-action_embedding_{label}"
        exp["input_kwargs"]["days_on_maze"] = days
        exp["with_embedding"] = True
        all_exps.append(exp)
    # try increasing the number of latent units
    for Nlat in [10, 20, 50]:
        exp = copy.deepcopy(single_sub_exp)
        exp["exp_name"] = f"state-action_embedding_Nlat_{Nlat}"
        exp["model_init_kwargs"]["Nlat"] = Nlat
        exp["with_embedding"] = True
        all_exps.append(exp)
    # submit all
    for exp in all_exps:
        submit_job(exp)


# %% Debugging
