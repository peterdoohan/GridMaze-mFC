

print("running")

import os
import sys
os.chdir("/ceph/behrens/peter_doohan/goalNav_mFC_refactor/code")
print(os.getcwd())

import GridMaze

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import linear_model
from sklearn.metrics import mean_poisson_deviance
from scipy.stats import pearsonr
import warnings 
import pickle
warnings.filterwarnings("ignore")

from importlib import reload

import GridMaze.analysis.core as core
from GridMaze.analysis.embedding_model.embedding_utils import Encoder, train_model
import GridMaze.analysis.core.get_sessions as gs
from GridMaze.maze import representations as mr
from GridMaze.analysis.embedding_model.get_input_data import get_input_data


plt.rcParams['font.size'] = 20
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False


#%% preprocess data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # run on GPU if possible
base_dir = "/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/test_variable_combinations/"

# list of possible variables
vars_ = ['distance', 'place', 'direction', 'place_direction', 'current_route', 'next_route', "trial_phase"]

# list of experiments to run

exps = [
    {"name": "linear_state_action",
     "input_types": ["place", "direction"],
     "partition": [(0,), (1,)]},
    
    {"name": "nonlinear_state_action",
     "input_types": ["place", "direction"],
     "partition": None},
    
    {"name": "conjunctive_state_action",
     "input_types": ["place_direction"],
     "partition": None},
    
    {"name": "linear_SA_distance",
     "input_types": ["place_direction", "distance"],
     "partition": [(0,), (1,)]},
    
    {"name": "nonlinear_SA_distance",
     "input_types": ["place_direction", "distance"],
     "partition": None},
    
    {"name": "nonlinear_state_distance",
     "input_types": ["place", "distance"],
     "partition": None},
    
    {"name": "linear_route_SAD",
     "input_types": ["place_direction", "distance", "current_route"],
     "partition": [(0,1,), (2,)]},
    
    {"name": "nonlinear_route_SAD",
     "input_types": ["place_direction", "distance", "current_route"],
     "partition": None},
    
    {"name": "linear_next_route_SADR",
     "input_types": ["place_direction", "distance", "current_route", "next_route"],
     "partition": [(0,1,2,), (3,)]},
    
    {"name": "nonlinear_next_route_SADR",
     "input_types": ["place_direction", "distance", "current_route", "next_route"],
     "partition": None},
    
    {"name": "trial_phase",
    "input_types": ["place_direction", "distance", "trial_phase"],
    "partition": None},
    
    {"name": "include_stationary",
     "input_types": ["place_direction", "distance"],
     "partition": None,
     "moving_only": False, "navigation_only": True},
    
    {"name": "all_timepoints",
     "input_types": ["place_direction", "distance"],
     "partition": None,
     "moving_only": False, "navigation_only": False},
    
    {"name": "pure_distance",
     "input_types": ["distance"],
     "partition": None},
]

inds_to_run = [10,11,12]

exps_to_run = [exps[ind] for ind in inds_to_run]
print("running:", exps_to_run)

# common parameters
distance_metrics = ("distance_to_goal", "geodesic")
subject = "m2"
maze_name  = "maze_1"
Nhid = [150, 50] # 2 hidden layers
Nlat = 10 # 10 latents
lr = 5e-4 # small learning rate
beta_act, beta_weight = 1e-1, 1e-1 # reasonable regularization
nepochs = 5001 # quite a few epochs
resolution = 0.1 # 100 ms resolution

# get experimental sessions
maze_sessions = gs.get_maze_sessions(
    subject_IDs=[subject],
    maze_names=[maze_name],
    days_on_maze="late",
    with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "navigation_routes_df"],
    must_have_data=True,
)


for exp in exps_to_run: # now run our experiments
    
    print("\n\nrunning experiment:", exp)
    moving_only, navigation_only = True, True
    
    
    name, input_types, partition = [exp[key] for key in ["name", "input_types", "partition"]]
    if "moving_only" in exp.keys(): moving_only = exp["moving_only"]
    if "navigation_only" in exp.keys(): navigation_only = exp["navigation_only"]

    # get input data for this distance metric
    sessions = [get_model_input_data(s, resolution = resolution, distance_metrics = distance_metrics, input_types = input_types, moving_only = moving_only, navigation_only = navigation_only) for s in maze_sessions]
    
    # find input indices corresponding to the partitions
    if partition is not None:
        X_type_inds = sessions[0]["X_type_inds"]
        partition = [np.concatenate([X_type_inds[inds[i]] for i in range(len(inds))]) for inds in partition]
    
    # add neuron ids, but only consider neurons with some threshold of spikes
    ind = 0
    thresh = 300
    for data in sessions:
        data['spikes'] = data['spikes'][data['spikes'].sum(-1) > thresh, :]
        n_clusters = data["spikes"].shape[0]
        data["inds"] = torch.from_numpy(np.arange(ind, ind + n_clusters)).to(torch.int32)
        ind += n_clusters

    Nin = sessions[0]["X"].shape[0] # input dimensionality
    Ntot = sum([sesh["spikes"].shape[0] for sesh in sessions]) # total number of neurons
    Ttot = sum([sesh["spikes"].shape[1] for sesh in sessions]) # total number of neurons
    
    print(f"Nin: {Nin}, Ntot: {Ntot}, Ttot: {Ttot}, partition: {partition}, navigation only: {navigation_only}, moving only: {moving_only}")

    # store our test metrics
    cv_results = np.zeros((len(sessions), len(sessions), 2)) # each 'training test session' by each 'test test session' by 'latent vs input' regularization
    all_train_losses, all_test_perfs, all_train_perfs, models = [], [], [], []
    
    for ntest in range(len(sessions)):

        print("\nntest:", ntest, "\n", exp)
        train_sessions = sessions[:ntest] + sessions[ntest+1:] # list of training sessions
        test_session = sessions[ntest] # test session

        # fix random seeds for now
        np.random.seed(ntest)
        torch.manual_seed(ntest)

        model_full = Encoder(Nin, Nhid, Nlat, Ntot, beta_act = beta_act, beta_weight = beta_weight, partition = partition) # instantiate model
        print(f"partition: {model_full.partition}")

        # train model
        model_full, train_losses, test_perfs, train_perfs = train_model(model_full, train_sessions, test_session = test_session,
                                                                        nepochs = nepochs, lr = lr, test_freq = 1000, device = device)
        all_train_losses.append(train_losses)
        all_test_perfs.append(test_perfs)
        all_train_perfs.append(train_perfs)
        models.append(model_full)

        print()
        # also compare to training session
        for isesh, session in enumerate([test_session]+train_sessions):
            test_perf = model_full.eval_representation(session["X"].to(device), session["spikes"].to(device), cv = 5, alpha = 1e-3).mean()
            test_perf2 = model_full.eval_representation(session["X"].to(device), session["spikes"].to(device), cv = 5, alpha = 1e-3, embed = False).mean() # regression directly from inputs
            print(test_perf, test_perf2)
            cv_results[ntest, isesh, :] = np.array([test_perf, test_perf2])
            

        print("perf:", np.mean(cv_results[ntest, ...], axis = 0))
        
    result = {"exp": exp, "cv_results": cv_results, "train_losses": all_train_losses, "test_perfs": all_test_perfs, "train_perfs": all_train_perfs, "models": models}
    
    pickle.dump(result, open(f"{base_dir}/result_{name}.p", "wb"))

    

raise NotImplementedError

# now write some analysis code

def plot_comparison(experiments, outfile = f"{base_dir}/figs/test.png", ticklabels = None):
    
    results = []
    for name in experiments:
        result = pickle.load(open(f"{base_dir}/result_{name}.p", "rb"))
        results.append(result["cv_results"])


    embed_result = np.array(results)[..., 0, 0] # test fold CV accuracy
    raw_result = np.array(results)[..., 0, 1] # regression directly on inputs
    
    me, se = embed_result.mean(1), embed_result.std(1)/np.sqrt(embed_result.shape[1])
    mr, sr = raw_result.mean(1), raw_result.std(1)/np.sqrt(raw_result.shape[1])
    xs = np.arange(len(me))

    print(me, se)
    print(mr, sr)
    if ticklabels is None: ticklabels = experiments
    
    offset, width = 0.2, 0.4
    plt.figure()
    plt.bar(xs-offset, me, yerr = se, width = width)
    plt.bar(xs+offset, mr, yerr = sr, width = width)
    plt.xticks(xs, ticklabels, rotation = 45, ha = "right")
    plt.ylabel("cv performance")
    plt.savefig(outfile, bbox_inches = "tight")
    plt.close()

    return

# state-action, linear vs non-linear vs. conjunctive

experiments = ["linear_state_action", "nonlinear_state_action", "conjunctive_state_action",]
#experiments = ["linear_state_action", "conjunctive_state_action",]
plot_comparison(experiments, outfile = f"{base_dir}/figs/state_action_embeddings.png")

# SA+distance, linear vs non-linear

experiments = ["conjunctive_state_action", "pure_distance", "linear_SA_distance", "nonlinear_SA_distance",]
plot_comparison(experiments, outfile = f"{base_dir}/figs/distance_embeddings.png")

# route vs non-route
experiments = ["nonlinear_SA_distance", "linear_route_SAD", "nonlinear_route_SAD"]
plot_comparison(experiments, outfile = f"{base_dir}/figs/route_embeddings.png")

# next route vs only current route
experiments = ["nonlinear_route_SAD", "linear_next_route_SADR", "nonlinear_next_route_SADR"]
plot_comparison(experiments, outfile = f"{base_dir}/figs/next_route_embeddings.png")

# different amounts of data
experiments = ["nonlinear_SA_distance", "include_stationary", "all_timepoints", "trial_phase"]
plot_comparison(experiments, outfile = f"{base_dir}/figs/different_data.png")


# try to plot some of the state-action-distance latents

name = "nonlinear_SA_distance"
SAD_result = pickle.load(open(f"{base_dir}/result_{name}.p", "rb"))
sample_model = SAD_result["models"][0].to(device)
exp = SAD_result["exp"]

moving_only, navigation_only = True, True
name, input_types, partition = [exp[key] for key in ["name", "input_types", "partition"]]
if "moving_only" in exp.keys(): moving_only = exp["moving_only"]
if "navigation_only" in exp.keys(): navigation_only = exp["navigation_only"]
sessions = [get_model_input_data(s, resolution = resolution, distance_metrics = distance_metrics, input_types = input_types, moving_only = moving_only, navigation_only = navigation_only) for s in maze_sessions]

num_states, num_dist_bins = [len(sessions[0]["X_type_inds"][i]) for i in range(2)]
all_locs = torch.arange(num_states)
all_dists = torch.arange(num_dist_bins)

all_X = torch.zeros(sample_model.Nin, num_states*num_dist_bins)
all_loc_dists = torch.zeros(all_X.shape[-1], 2)
for loc in all_locs:
  for dist in all_dists:
    ind = loc*num_dist_bins+dist
    all_X[loc, ind] = 1.
    all_X[num_states+dist, ind] = 1.
    all_loc_dists[ind, :] = torch.tensor([loc, dist])

print(all_X.shape)

all_z = sample_model.encode(all_X.to(sample_model.Wout.device)).detach().cpu().numpy()
print(all_z.shape)
print(all_z.min(), all_z.max(), all_z.std(-1).mean())


#
all_place_direction_pairs = mr.get_maze_place_direction_pairs(maze_sessions[0].simple_maze())
let_dict = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
ddict = {"N": 0, "W": 1, "S": 2, "E": 3}
def lets_to_inds(lets):
    return (let_dict[lets[0]], int(lets[1])-1)

acts = np.zeros((7,7,4, num_dist_bins, all_z.shape[0])) + np.nan
for d in range(num_dist_bins):
    zd = all_z[:, d::num_dist_bins]
    for ipd, pd in enumerate(all_place_direction_pairs):
        if len(pd[0]) == 2:
            ind = lets_to_inds(pd[0])
            acts[ind[0], ind[1], ddict[pd[1]], d, :] = zd[:, ipd]
acts = np.nanmean(acts, axis = 2)

#

for n in range(all_z.shape[0]): # for each latent
  fig, axs = plt.subplots(int(np.ceil(num_dist_bins/5)),5, figsize = (10, 8))
  for d in range(num_dist_bins):
    ax = axs[d//5, d%5]
    ax.imshow(acts[..., d, n], cmap = "viridis")
    ax.set_xticks([])
    ax.set_yticks([])
  plt.tight_layout()
  plt.savefig(f"{base_dir}/figs/example_latent{n}.png", bbox_inches = "tight")
  plt.close()
  print()



sims = np.zeros((Nlat, Nlat)) + np.nan
for i in range(Nlat):
  for j in range(i+1, Nlat):
    cor = pearsonr(all_z[i, :], all_z[j, :])[0]
    sims[i, j] = cor
    sims[j, i] = cor

plt.figure()
plt.imshow(sims, vmin = -1, vmax = 1)
plt.xticks([])
plt.yticks([])
plt.savefig(f"{base_dir}/figs/example_latent_correlations.png", bbox_inches = "tight")
plt.close()




