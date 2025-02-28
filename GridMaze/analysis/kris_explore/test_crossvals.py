

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

seed = 1
np.random.seed(seed) # fix random seeds for now
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # run on GPU if possible

#%% preprocess data

# Input type must be in ['distance', 'place', 'direction', 'place_direction', 'current_route', 'next_route', 'tower_bridge ', 'speed', 'acceleration', 'trial_phase', 'head_direction']
input_features=["distance", "place_direction"]
partition = None
latent_inputs = None
Nhid = [150, 50] # 2 hidden layers
Nlat = 10
latent_nonlin = None
cv = 4

subjects = ["m2"]
base_sessions = get_input_data(subject_IDs=subjects,
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
    start_ind = 0, end_ind = None)
#base_sessions = base_sessions[2:]

input_streams = base_sessions[0]["X_type_inds"]
input_stream_names = base_sessions[0]["input_feature_names"]

#%% train model
"""
let's start by training the model on S-1 sessions and testing on the held out session
"""

# define some parameters
lr = 5e-4
beta_act, beta_weight = 1e-1, 1e-1
nepochs = 2001
eval_alpha = 1e-3

Ntot = sum([sesh["spikes"].shape[0] for sesh in base_sessions]) # total number of neurons

### now we need to run 3 different models. ###
# 1. usual test on held-out session
# 2. embed on everything then crossval output weights on held-out session
# 3. embed on 9.9 sessions, test on 0.1, repeat

sessions_to_test = np.arange(1, len(base_sessions)-1)
sessions_to_test = [2,3,4,5]
sessions_to_test = [4,5,6,7]
print("sessions to test:", sessions_to_test)
all_ress = []

for itest, ntest in enumerate(sessions_to_test):
    print(f"\n\n\ntesting session {itest+1} of {len(sessions_to_test)}")
    ress = []
    #ntest = 4 # test session index
    for fold in range(cv):
        cv_modes = ["session", "none", "trials"] if fold == 0 else ["trials"]
        for imode, cv_mode in enumerate(cv_modes):
            sessions = [copy.deepcopy(sesh) for sesh in base_sessions]
            print(f"\n\nhold out {cv_mode}, fold: {fold}")

            if cv_mode == "session": # hold out a full session
                train_sessions = sessions[:ntest] + sessions[ntest+1:] # list of training sessions
                test_session = sessions[ntest] # test session
            elif cv_mode == "none": # train embedding model on everything
                train_sessions = sessions
                test_session = sessions[ntest]
            elif cv_mode == "trials": # hold out some trials from training the embedding
                
                train_sessions = sessions[:ntest] + sessions[ntest+1:] + copy.deepcopy([sessions[ntest]]) # list of training sessions
                test_session = copy.deepcopy(sessions[ntest]) # test session
                
                trials = test_session["trial_ids"]
                unique_trials = np.unique(trials)
                trial_splits = [[] for _ in range(cv)]
                for trial in unique_trials:
                    trial_splits[int(trial) % cv].append(trial)
                inds = [np.concatenate([np.where(trials == trial_id)[0] for trial_id in trial_split]) for trial_split in trial_splits]
                
                test, train = inds[fold], np.concatenate([inds[f] for f in range(cv) if f != fold])
                split = inds
                # train, test = inds[fold], np.concatenate([inds[f] for f in range(cv) if f != fold]) # try to just add a bit of train instead
                # split = [inds[f] for f in range(cv) if f != fold] # split for testing
                
                for key, value in copy.deepcopy(test_session).items():
                    if "shape" in dir(value):
                        if value.shape[-1] == len(test)+len(train):
                            test_session[key] = value[..., test]
                            train_sessions[-1][key] = value[..., train]
                        
                print(train_sessions[-1]["spikes"].shape, test_session["spikes"].shape)
                
                

            model_full = Encoder(input_streams, Nhid, Nlat, Ntot, input_stream_names = input_stream_names, beta_act = beta_act, beta_weight = beta_weight,
                                partition = partition, latent_inputs = latent_inputs, latent_nonlin = latent_nonlin, inv_link = "exp", noise_function = "Poisson", sqrt_counts = None) # instantiate model

            # train model
            model_full, train_losses, test_perfs, train_perfs = train_model(model_full, train_sessions, test_session = test_session,
                                                                        nepochs = nepochs, lr = lr, test_freq = 1000, device = device, eval_alpha = eval_alpha)


            ### now eval
            print()

            test_perf2 = np.zeros(1)

            model_full.eval_cv_function = model_full.eval_exp_poisson # just use simple CV for now
            #model_full.eval_cv_function = model_full.eval_exp_poisson_cv

            if cv_mode == "trials": # use the learned weights for this session, where we already embedded some of the data
                # ztest = model_full.encode(test_session["X"].to(device))  # T x D
                # yhat = model_full.decode(ztest, neuron_inds=test_session["cluster_inds"]).detach().cpu().numpy()
                # ytrue = test_session["spikes"].numpy()
                # test_perf = np.zeros((yhat.shape[0], 1))
                # for n in range(yhat.shape[0]):
                #     test_perf[n, 0] = calc_poisson_deviance(yhat[n], ytrue[n])
                sesh = sessions[ntest]
                # test_perf = model_full.eval_representation(sesh["X"].to(device), sesh["spikes"].to(device), cv = cv-1, alpha = eval_alpha, trials = sesh["trial_ids"], split = split)
                test_perf = model_full.eval_representation(sesh["X"].to(device), sesh["spikes"].to(device), cv = cv, alpha = eval_alpha, trials = sesh["trial_ids"], split = split)
                test_perf = test_perf[:, fold][:, None] # only for the held-out fold
                
            else: # just run standard crossval
                test_perf = model_full.eval_representation(test_session["X"].to(device), test_session["spikes"].to(device), cv = cv, alpha = eval_alpha, trials = test_session["trial_ids"])
                if cv_mode == "session":
                    test_perf2 = model_full.eval_representation(test_session["X"].to(device), test_session["spikes"].to(device), cv = cv, alpha = eval_alpha, embed = False, trials = test_session["trial_ids"])
            
            mean_test_perfs = np.array([test_perf.mean(), test_perf2.mean()])

            print("perf:", mean_test_perfs)
            ress.append({"summary": mean_test_perfs, "all": test_perf, "ctrl": test_perf2, "mode": cv_mode, "fold": fold})
    
    all_ress.append(ress)
    


vals = np.zeros((len(all_ress), 3, cv))
for ires, res in enumerate(all_ress):
    vals[ires, 0, :] = res[0]["all"].mean(0) # avg across neurons for each fold
    vals[ires, 1, :] = res[1]["all"].mean(0) # second method
    for icv in range(cv):
        vals[ires, 2, icv] = res[2+icv]["all"].mean(0)


xs = np.arange(3)
plt.figure(figsize = (3,2))
for val in vals:
    m, s = val.mean(1), val.std(1)/np.sqrt(val.shape[1])
    plt.errorbar(xs, m, yerr = s)
plt.xticks(xs, ["session", "none", "trials"], rotation = 45, ha = "right")
plt.ylabel("cv perf")
plt.savefig("/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/figs/compare_cvs.png", bbox_inches = "tight")
plt.close()


