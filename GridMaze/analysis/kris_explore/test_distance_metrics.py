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
import pickle
warnings.filterwarnings("ignore")

from importlib import reload
import GridMaze

import GridMaze.analysis.core as core
from GridMaze.analysis.embedding_model.embedding_utils import Encoder, train_model
import GridMaze.analysis.core.get_sessions as gs



#%% preprocess data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # run on GPU if possible
base_dir = "/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/"
all_distance_metrics = [("distance_to_goal", "geodesic"), ("distance_to_goal", "future"), ("progress_to_goal", "path_length"), ("progress_to_goal", "time")]

subject = "m2"
maze_name  = "maze_1"
Nhid = [150, 50] # 2 hidden layers
Nlat = 10
lr = 5e-4
beta_act, beta_weight = 1e-1, 1e-1
nepochs = 4001

maze_sessions = gs.get_maze_sessions(
    subject_IDs=[subject],
    maze_names=[maze_name],
    days_on_maze="late",
    with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics"],
    must_have_data=True,
)

results = []
for idist, distance_metrics in enumerate(all_distance_metrics):
    
    print("\n\nrunning distance metrics:", distance_metrics)

    # get input data for this distance metric
    input_data = [get_model_input_data(s, resolution = 0.1, distance_metrics = distance_metrics) for s in maze_sessions]
    
    # add neuron ids, but only consider neurons with some threshold of spikes
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

    session_results = np.zeros((len(sessions), len(sessions), 2))
    
    for ntest in range(len(sessions)):

        print("\nntest:", ntest)
        train_sessions = sessions[:ntest] + sessions[ntest+1:] # list of training sessions
        test_session = sessions[ntest] # test session

        np.random.seed(ntest) # fix random seeds for now
        torch.manual_seed(ntest)


        model_full = Encoder(Nin, Nhid, Nlat, Ntot, beta_act = beta_act, beta_weight = beta_weight, partition = None) # instantiate model

        # train model
        model_full, train_losses, test_perfs, train_perfs = train_model(model_full, train_sessions, test_session = test_session,
                                                                        nepochs = nepochs, lr = lr, test_freq = 1000, device = device)


        print()
        # also compare to training session
        for isesh, session in enumerate([test_session]+train_sessions):
            test_perf = model_full.eval_representation(session["X"].to(device), session["spikes"].to(device), cv = 5, alpha = 1e-3).mean()
            test_perf2 = model_full.eval_representation(session["X"].to(device), session["spikes"].to(device), cv = 5, alpha = 1e-3, embed = False).mean()
            print(test_perf, test_perf2)
            session_results[ntest, isesh, :] = np.array([test_perf, test_perf2])
            
            ### also compare to performance from the pure input data!! ###

        print("perf:", np.mean(session_results[ntest, ...], axis = 0))
        
    results.append({"distance_metrics": distance_metrics, "result": session_results})
    
pickle.dump(results, open(f"{base_dir}/distance_metric_comparison.p", "wb"))

    
### now evaluate

raise NotImplementedError


plt.rcParams['font.size'] = 20
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

results = pickle.load(open(f"{base_dir}/distance_metric_comparison.p", "rb"))


summary = np.array([res["result"][:, 0, 0] for res in results]) # test fold cv accuracy for each fold

m, s = summary.mean(1), summary.std(1)/np.sqrt(summary.shape[1])
xs = np.arange(len(m))

plt.figure()
plt.bar(xs, m, yerr = s)
plt.xticks(xs, all_distance_metrics, rotation = 45, ha = "right")
plt.ylabel("cv performance")
plt.savefig(f"{base_dir}/figs/dist_metrics.png", bbox_inches = "tight")
plt.close()

    