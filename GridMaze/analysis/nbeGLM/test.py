# %% import some stuff

import os
from importlib import reload

# os.chdir("/ceph/behrens/peter_doohan/goalNav_mFC/experiment/code")
import numpy as np
import torch
from GridMaze.analysis.nbeGLM.get_input_data import get_input_data

from nbeGLM import models as glm
from nbeGLM import utils

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # run on GPU if possible

from jobs.nbeGLM.utils import (
    DEFAULT_INPUT_DATA_KWARGS,
    DEFAULT_MODEL_INIT_KWARGS,
    DEFAULT_MODEL_TRAIN_KWARGS,
    DEFAULT_MODEL_EVAL_KWARGS,
)


# %%
subject_ids = ["m2"]
data = get_input_data(subject_IDs=subject_ids)

# %%
reload(glm)
reload(utils)
nbeGLM = glm.Encoder()
i = 0
test_data = data[i]  # single session
train_data = data[:i] + data[i + 1 :]  # all other sessions

nbeGLM.train(train_data, test_data, device=DEVICE, **DEFAULT_MODEL_TRAIN_KWARGS)


# %%
reload(glm)
reload(utils)
test_scores = nbeGLM.score(
    x=test_data["X"],
    y=test_data["spikes"],
    trials=test_data["trial_ids"],
    n_folds=5,
    optimal_alpha=True,
    n_jobs=16,
)

# %%
