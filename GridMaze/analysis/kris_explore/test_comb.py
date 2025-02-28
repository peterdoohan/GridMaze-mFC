
import pickle


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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # run on GPU if possible
ress = {}
for combine_frs in [False, "multiplicative", "additive"]:
    
    print("\n\n", combine_frs)
    ress[combine_frs] = []
    

    input_features=["distance", "place_direction"]#, "tower_bridge"]
    partition = [[0], [1]]

    latent_inputs = None
    Nhid = [150, 50] # 2 hidden layers
    Nlat = 10
    latent_nonlin = None
    if not combine_frs: Nlat, Nhid = Nlat*2, [N*2 for N in Nhid]

    subjects = ["m2"]
    sessions = get_input_data(subject_IDs=["m2"],
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

    lr = 5e-4
    beta_act, beta_weight = 1e-1, 1e-1
    nepochs = 2001
    eval_alpha = 1e-3

    Ntot = sum([sesh["spikes"].shape[0] for sesh in sessions]) # total number of neurons
    
    for ntest in range(len(sessions)): # test session index
        
        train_sessions = sessions[:ntest] + sessions[ntest+1:] # list of training sessions
        test_session = sessions[ntest] # test session

        model_full = Encoder(input_streams, Nhid, Nlat, Ntot, input_stream_names = input_stream_names, beta_act = beta_act, beta_weight = beta_weight,
                            partition = partition, latent_inputs = latent_inputs, latent_nonlin = latent_nonlin, inv_link = "exp",
                            noise_function = "Poisson", sqrt_counts = None, combine_frs = combine_frs) # instantiate model

        # train model
        model_full, train_losses, test_perfs, train_perfs = train_model(model_full, train_sessions, test_session = test_session,
                                                                        nepochs = nepochs, lr = lr, test_freq = 200, device = device, eval_alpha = eval_alpha)


        test_session = sessions[ntest]
        model_full.eval_cv_function = model_full.eval_exp_poisson

        test_perf = model_full.eval_representation(test_session["X"].to(device), test_session["spikes"].to(device), cv = 5, alpha = eval_alpha, trials = test_session["trial_ids"])

        print(np.mean(test_perf))

        ress[combine_frs].append(test_perf)
        
        

pickle.dump(ress, open(f"/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/add_vs_mul/test.p", "wb"))


ress = picke.load(open(f"/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/add_vs_mul/test.p", "rb"))

keys = list(ress.keys())

means = np.array([[arr.mean() for arr in ress[key]] for key in keys])

ms, ss = means.mean(-1), means.std(-1)/np.sqrt(means.shape[-1])
xs = np.arange(len(ms))

labels = ["nonlinear"] + keys[1:]

plt.figure(figsize = (3,3))
plt.bar(xs, ms, yerr = ss)
plt.xticks(xs, labels, ha = "right", rotation = 45)
plt.ylabel("accuracy")
plt.savefig(f"/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/add_vs_mul/test.png", bbox_inches = "tight")
plt.close()

