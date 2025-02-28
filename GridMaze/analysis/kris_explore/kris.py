""" 
"""


import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import pearsonr
import os
import warnings 
import copy
warnings.filterwarnings("ignore")

from importlib import reload
import GridMaze.analysis.embedding_model.get_input_data
import GridMaze.analysis.embedding_model.embedding_utils
from GridMaze.analysis.embedding_model.embedding_utils import Encoder, train_model, calc_poisson_deviance
from GridMaze.analysis.embedding_model.get_input_data import get_input_data
reload(GridMaze.analysis.embedding_model.get_input_data)
reload(GridMaze.analysis.embedding_model.embedding_utils)
from GridMaze.analysis.embedding_model.embedding_utils import Encoder, train_model, calc_poisson_deviance
from GridMaze.analysis.embedding_model.get_input_data import get_input_data

#%% set some parameters

seed = 2
np.random.seed(seed) # fix random seeds for now
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # run on GPU if possible

#%% preprocess data

# distance_metrics you can pass in ("distance_to_goal", "geodesic"),("distance_to_goal", "future") ,(“progress_to_goal”, “path_length”),(“progress_to_goal”, “time”). To try out the different distance metrics
# Geodesic is just shortest path from where they are, future is the distance left on the trajectory they actually take,
# progress path length is that normalized between 0 -> 1 (0 is at goal for convent comparison to distance metrics), progress time is also 0 -> 1 but normalized as progress in time.

# Input type must be in ['distance', 'place', 'direction', 'place_direction', 'current_route', 'next_route', 'tower_bridge ', 'speed', 'acceleration', 'trial_phase', 'head_direction']
input_features=["distance", "place_direction", "current_route", "speed", "head_direction", "acceleration", "goal", "tower_bridge"]
input_features=["distance", "place_direction"]#, "tower_bridge"]
partition = None
#partition = ((0,), (1,))
latent_inputs = ["tower_bridge"]
latent_inputs = None
Nhid = [150, 50] # 2 hidden layers
Nlat = 10
combine_frs = "multiplicative"
#Nhid = [250,250,150,50]
latent_nonlin = None
if not combine_frs: Nlat, Nhid = Nlat*2, [N*2 for N in Nhid]

# subject ids: ['m2', 'm3', 'm4', 'm6', 'm7', 'm8']
subjects = ['m2', 'm3', 'm4', 'm6', 'm7', 'm8']
#subjects = ["m2", "m3", "m6"]
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
    min_spike_count = 300,)

input_streams = sessions[0]["X_type_inds"]
input_stream_names = sessions[0]["input_feature_names"]
partition = [[0], [1]]

#%% train model
"""
let's start by training the model on S-1 sessions and testing on the held out session
"""

# define some parameters
lr = 5e-4
beta_act, beta_weight = 1e-1, 1e-1
nepochs = 3001
nepochs = 1001
eval_alpha = 1e-3

Ntot = sum([sesh["spikes"].shape[0] for sesh in sessions]) # total number of neurons
ntest = 4 # test session index
train_sessions = sessions[:ntest] + sessions[ntest+1:] # list of training sessions
test_session = sessions[ntest] # test session

#train_sessions, test_session = sessions, None

model_full = Encoder(input_streams, Nhid, Nlat, Ntot, input_stream_names = input_stream_names, beta_act = beta_act, beta_weight = beta_weight,
                     partition = partition, latent_inputs = latent_inputs, latent_nonlin = latent_nonlin, inv_link = "exp",
                     noise_function = "Poisson", sqrt_counts = None, combine_frs = combine_frs) # instantiate model

for p in model_full.parameters():
    print(p.shape)


# train model
model_full, train_losses, test_perfs, train_perfs = train_model(model_full, train_sessions, test_session = test_session,
                                                                nepochs = nepochs, lr = lr, test_freq = 200, device = device, eval_alpha = eval_alpha)


print()
# also compare to training session
test_session = sessions[ntest]

eval_sessions = [test_session] #+ train_sessions

test_perfs = np.zeros((len(eval_sessions), 2))
test_perf2 = np.zeros(1)
all_test_perfs = []

model_full.eval_cv_function = model_full.eval_exp_poisson
#model_full.eval_cv_function = model_full.eval_exp_poisson_cv

for isesh, session in enumerate(eval_sessions):
    test_perf = model_full.eval_representation(session["X"].to(device), session["spikes"].to(device), cv = 5, alpha = eval_alpha, trials = session["trial_ids"])
    #test_perf2 = model_full.eval_representation(session["X"].to(device), session["spikes"].to(device), cv = 5, alpha = eval_alpha, embed = False, trials = session["trial_ids"])
    print(test_perf.mean(), test_perf2.mean())
    test_perfs[isesh] = np.array([test_perf.mean(), test_perf2.mean()])
    all_test_perfs.append(test_perf)

print("perf:", np.mean(test_perfs, axis = 0))
    

# neuron_mean_cv_perfs = np.mean(all_test_perfs[0], -1)
# vmin = min(np.amin(neuron_mean_cv_perfs), -0.6)
# vmax = max(np.amax(neuron_mean_cv_perfs), 0.6)
# plt.figure(figsize = (4,3))
# plt.hist(neuron_mean_cv_perfs, bins = np.linspace(vmin, vmax, 12))
# plt.axvline(0, color = "k")
# plt.xlabel("accuracy")
# plt.ylabel("frequency")
# plt.savefig("/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/figs/perf_distribution.png", bbox_inches = "tight")
# plt.close()


