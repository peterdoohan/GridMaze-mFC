
import GridMaze.analysis.core as core
import pickle
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

import GridMaze.analysis.core as core
from GridMaze.analysis.embedding_model.get_input_data import get_input_data
from GridMaze.analysis.embedding_model.embedding_utils import Encoder, train_model
import GridMaze.analysis.core.get_sessions as gs


base_dir = "/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/grid_search/"

Nhids = [[200, 150, 100, 50],
         [300, 200, 150, 100, 50],
         [150, 50]]
betas = [[1e-1, 1e-1], [1e-2, 1e-2], [1e-3, 1e-3]]
lrs = [1e-2, 1e-3, 1e-4]
Nlats = [2,4,6,10,15,20,25]

all_params = []
for Nhid in Nhids:
    for beta in betas:
        for lr in lrs:
            for Nlat in Nlats:
                all_params.append({"Nhid": Nhid, "Nlat": Nlat, "betas": beta, "lr": lr, "subject": "m2", "maze": "maze_1"})
                
                
#pickle.dump(all_params, open(f"{base_dir}hp_search_params.p", "wb"))

group_size = 30
groups = [np.arange(i*group_size, min(len(all_params), group_size*(i+1))) for i in range(int(np.ceil(len(all_params)/group_size)))]

experiment_index = 6

groups = [np.arange(107, 114), np.arange(114, 120), np.arange(141,150), [173,174,175,176,177,178,179, 89, 88]]
experiment_index = 3

print(f"running experiment group {experiment_index}")

subject = "m2"
maze_name  = "maze_1"

maze_sessions = gs.get_maze_sessions(
    subject_IDs=[subject],
    maze_names=[maze_name],
    days_on_maze="late",
    with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics"],
    must_have_data=True,
)
input_data = [get_model_input_data(s, resolution = 0.1) for s in maze_sessions]
# add neuron inds
ind = 0
thresh = 300
for data in input_data:
    data['spikes'] = data['spikes'][data['spikes'].sum(-1) > thresh, :]
    n_clusters = data["spikes"].shape[0]
    data["inds"] = torch.from_numpy(np.arange(ind, ind + n_clusters)).to(torch.int32)
    ind += n_clusters
sessions = input_data

Nin = sessions[0]["X"].shape[0] # input dimensionality
Ntot = sum([sesh["spikes"].shape[0] for sesh in sessions]) # total number of neurons

# run experiment

for igroup, group_ind in enumerate(groups[experiment_index]): # for each parameter set
    prms = all_params[group_ind]
    
    print(f"\n\nrunning hp set {igroup} of {len(groups[experiment_index])}: {group_ind}")
    print("params\n:", prms)
    
    np.random.seed(0) # fix random seeds for now
    torch.manual_seed(0)
    
    
    all_train_losses, all_test_perfs, all_train_perfs, all_cv_perfs = [[] for _ in range(4)]
    for ntest in range(len(sessions)): # hold out each session
        print(f"\nntest: {ntest}")
        train_sessions = sessions[:ntest] + sessions[ntest+1:] # list of training sessions
        test_session = sessions[ntest] # test session

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # run on GPU if possible
        model_full = Encoder(Nin, prms["Nhid"], prms["Nlat"], Ntot, beta_act = prms["betas"][0], beta_weight = prms["betas"][1]) # instantiate model

        # train model
        model_full, train_losses, test_perfs, train_perfs = train_model(model_full, train_sessions, test_session = test_session,
                                                                        nepochs = 2001, lr = prms["lr"], test_freq = 500, device = device)

        print()
        # also compare to training session
        test_cv_perfs = np.zeros(len(train_sessions)+1)
        for isesh, session in enumerate([test_session]+train_sessions):
            test_cv_perf = model_full.eval_representation(session["X"].to(device), session["spikes"].to(device), cv = 5, alpha = 1e-3).mean()
            print(test_cv_perf)
            test_cv_perfs[isesh] = test_cv_perf

        print("perf:", np.mean(test_cv_perfs))
        
        all_train_losses.append(train_losses)
        all_test_perfs.append(test_perfs)
        all_train_perfs.append(train_perfs)
        all_cv_perfs.append(test_cv_perfs)
        
    summary = {"train_losses": all_train_losses, "test_perfs": all_test_perfs, "train_perfs": all_train_perfs, "cv_perf": all_cv_perfs, "param": prms}
    
    pickle.dump(summary, open(f"{base_dir}hp_search_result_{group_ind}.p", "wb"))
        


                
                