# %% import some stuff

# os.chdir("/ceph/behrens/peter_doohan/goalNav_mFC/experiment/code")
from importlib import reload
import numpy as np
import torch
from GridMaze.analysis.nbeGLM.get_input_data import get_input_data

from nbeGLM import models
from nbeGLM import utils

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # run on GPU if possible

from jobs.nbeGLM.utils import (
    DEFAULT_INPUT_DATA_KWARGS,
    DEFAULT_MODEL_INIT_KWARGS,
    DEFAULT_MODEL_TRAIN_KWARGS,
    DEFAULT_SCORE_KWARGS,
)

from GridMaze.paths import RESULTS_PATH

TEST_RESULTS_DIR = RESULTS_PATH / "nbeGLM" / "test"

# %%
data = get_input_data()

# %%
reload(models)
reload(utils)

# %%
nbeGLM = models.nbeGLM()
i = 0
test_data = data[i]  # single session
train_data = data[:i] + data[i + 1 :]  # all other sessions

nbeGLM.train(train_data, test_data)


# %%

test_scores = nbeGLM.score(
    x=test_data["X"],
    y=test_data["spikes"],
    trials=test_data["trial_ids"],
    n_folds=5,
    optimal_alpha=True,
    n_jobs=24,
    verbose=True,
)

# %% baseline perf

baselineGLM = models.baselineGLM()
baseline_perf = baselineGLM.score(
    x=test_data["X"],
    y=test_data["spikes"],
    trials=test_data["trial_ids"],
    n_folds=5,
    optimal_alpha=True,
    n_jobs=24,
    verbose=True,
)

# %%

from GridMaze.analysis.nbeGLM import run_nbeGLM as rg
from GridMaze.analysis.nbeGLM import get_input_data as gid
from nbeGLM import models
from nbeGLM import utils
from importlib import reload

# %% test GridMaze code that uses nbeGLM package
reload(rg)
reload(gid)
reload(models)
reload(utils)

test_scores = rg.run_cv_nbeGLM(
    input_data_kwargs={
        "subject_IDs": ["m2"],
        "maze_name": "maze_1",
        "days_on_maze": "late",
        "input_groups": ["place", "direction", "distance_to_goal", "egocentric_action"],
        "input_group_kwargs": {},
        "resolution": 0.1,
        "max_steps_to_goal": 30,
        "min_spike_count": 300,
        "moving_only": False,
    },
    model_init_kwargs={
        "Nhid": [100, 50],
        "Nlat": 20,
        "beta_act": 1e-1,
        "beta_weight": 1e-1,
        "partition": None,
        "latent_nonlin": None,
    },
    model_train_kwargs={
        "device": None,
        "test_freq": 100,
        "lr": 1e-3,
        "nepochs": 101,
        "eval_alpha": 1e-3,
        "n_jobs": 64,
        "verbose": True,
    },
    score_kwargs={
        "n_folds": 5,
        "optimal_alpha": True,
        "n_jobs": 64,
        "verbose": True,
    },
    seed=0,
    verbose=True,
    save_path=TEST_RESULTS_DIR / "TEST_cv_nbeGLM",
)

# %%

test_scores = rg.run_cv_baselineGLM(
    input_data_kwargs={
        "subject_IDs": ["m2"],
        "maze_name": "maze_1",
        "days_on_maze": "late",
        "input_groups": ["place"],
        "input_group_kwargs": {},
        "resolution": 0.1,
        "max_steps_to_goal": 30,
        "min_spike_count": 300,
        "moving_only": False,
    },
    score_kwargs={
        "n_folds": 5,
        "optimal_alpha": True,
        "n_jobs": 24,
        "verbose": False,
    },
    seed=0,
    verbose=True,
    save_path=TEST_RESULTS_DIR / "TEST_cv_baselineGLM",
)

# %%

part_model2 = rg.train_nbeGLM(
    input_data_kwargs={
        "subject_IDs": "all",
        "maze_name": "maze_1",
        "days_on_maze": "late",
        "input_groups": ["place", "direction", "distance_to_goal", "egocentric_action"],
        "input_group_kwargs": {},
        "resolution": 0.1,
        "max_steps_to_goal": 30,
        "min_spike_count": 300,
        "moving_only": False,
    },
    model_init_kwargs={
        "Nhid": [100, 50],
        "Nlat": 15,
        "beta_act": 1e-1,
        "beta_weight": 1e-1,
        "partition": [("place",), ("direction",), ("distance_to_goal",), ("egocentric_action",)],
        "latent_nonlin": None,
    },
    model_train_kwargs={
        "device": None,
        "test_freq": 10,
        "lr": 1e-3,
        "nepochs": 101,
        "eval_alpha": 1e-3,
        "n_jobs": -1,
        "verbose": True,
    },
    seed=0,
    verbose=True,
    save_path=TEST_RESULTS_DIR / "TEST_full_model",
)

# %%
