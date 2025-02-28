""" 
"""


import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import linear_model
from sklearn.metrics import mean_poisson_deviance
from scipy.stats import pearsonr
import os
import warnings 
warnings.filterwarnings("ignore")

from importlib import reload
import GridMaze

import GridMaze.analysis.core as core
from GridMaze.analysis.embedding_model.embedding_utils import Encoder, train_model
import GridMaze.analysis.core.get_sessions as gs

from GridMaze.analysis.embedding_model.get_input_data import get_input_data
from GridMaze.analysis.embedding_model.run_experiment import run_embedding_model_experiment

reload(GridMaze)
reload(GridMaze.analysis.embedding_model.get_input_data)
reload(GridMaze.analysis.embedding_model.run_experiment)
reload(GridMaze.analysis.embedding_model.embedding_utils)
from GridMaze.analysis.embedding_model.run_experiment import run_embedding_model_experiment
from GridMaze.analysis.embedding_model.embedding_utils import Encoder, train_model
from GridMaze.analysis.embedding_model.get_input_data import get_input_data

#%% set some parameters

seed = 1
np.random.seed(seed) # fix random seeds for now
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # run on GPU if possible

#%% preprocess data

# Input type must be in ['distance', 'place', 'direction', 'place_direction', 'current_route', 'next_route', 'tower_bridge ', 'speed', 'acceleration', 'trial_phase', 'head_direction']
#input_features=["distance", "place_direction"]
input_features=["place", "direction"]
partition = None
latent_inputs = None
latent_nonlin = None
Nhid = [150, 50] # 2 hidden layers
Nhid = [250, 250, 150, 50]
Nlat = 10


subjects = ["m2"]
sessions = get_input_data(subject_IDs=subjects,
    maze_name="maze_1",
    input_features=input_features,
    distance_metrics=("distance_to_goal", "geodesic"),
    include_multi_unit=False,
    navigation_only=True,
    moving_only=True,
    resolution=0.1,  # s
    max_distance=1.8,  # m
    n_distance_bins=20,
    min_spike_count = 300,
    start_ind = 3)

input_streams = sessions[0]["X_type_inds"]
input_stream_names = sessions[0]["input_feature_names"]

#%% train model
"""
let's start by training the model on S-1 sessions and testing on the held out session
"""

# define some parameters
lr = 5e-4
beta_act, beta_weight = 1e-1, 1e-1
nepochs = 3001
eval_alpha = 1e-3

Ntot = sum([sesh["spikes"].shape[0] for sesh in sessions]) # total number of neurons
all_test_perfs = []

for ntest in range(len(sessions)):

    train_sessions = sessions[:ntest] + sessions[ntest+1:] # list of training sessions
    test_session = sessions[ntest] # test session

    #train_sessions, test_session = sessions, None

    model_full = Encoder(input_streams, Nhid, Nlat, Ntot, input_stream_names = input_stream_names, beta_act = beta_act, beta_weight = beta_weight,
                        partition = partition, latent_inputs = latent_inputs, latent_nonlin = latent_nonlin, inv_link = "exp", noise_function = "Poisson", sqrt_counts = None) # instantiate model

    # train model
    model_full, train_losses, test_perfs, train_perfs = train_model(model_full, train_sessions, test_session = test_session,
                                                                    nepochs = nepochs, lr = lr, test_freq = 1000, device = device, eval_alpha = eval_alpha)


    model_full.eval_cv_function = model_full.eval_exp_poisson#_cv

    test_perf = model_full.eval_representation(test_session["X"].to(device), test_session["spikes"].to(device), cv = 5, alpha = eval_alpha, trials = test_session["trial_ids"])
    test_perf2 = model_full.eval_representation(test_session["X"].to(device), test_session["spikes"].to(device), cv = 5, alpha = eval_alpha, embed = False, trials = test_session["trial_ids"])
    print(ntest, test_perf.mean(), test_perf2.mean())
    all_test_perfs.append(np.array([test_perf, test_perf2]))


means = np.array([perf.mean((-1,-2)) for perf in all_test_perfs])
print(means.mean(0))
print(means.std(0)/np.sqrt(means.shape[0]))
