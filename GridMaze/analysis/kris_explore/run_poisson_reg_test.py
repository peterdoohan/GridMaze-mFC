

"""
for each neuron, fit a few different models:
    (i) state-action + distance
    (ii) state+action+distance
    (iii) distance
    (iv) state-action
    (v) state+action
    (vi) state
    (vii) action
    (iix) constant
Then do hierarchical comparison:
first compare: const vs. state vs. action vs. state+action vs. state-action. Classify as N/S/A/S+A/SA
then compare distance vs. best_SA vs. both and classify as N_SA vs. D_N vs D_SA
"""

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
import pickle

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
eval_alpha = 1e-3

#%% preprocess data

# Input type must be in ['distance', 'place', 'direction', 'place_direction', 'current_route', 'next_route', 'tower_bridge ', 'speed', 'acceleration', 'trial_phase', 'head_direction']
input_features=["distance", "place_direction", "current_route", "speed", "head_direction", "acceleration", "goal", "tower_bridge"]
input_features=["place", "direction", "place_direction", "distance"]#, "tower_bridge"]
inp_inds = {"place": 0, "direction": 1, "place_direction": 2, "distance": 3}

t0 = time.time()
all_subjects = ['m2', 'm3', 'm4', 'm6', 'm7', 'm8']
# subject ids: ['m2', 'm3', 'm4', 'm6', 'm7', 'm8']
results = {}

for subject in all_subjects:

    sessions = get_input_data(subject_IDs=[subject],
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
        )

    input_streams = sessions[0]["X_type_inds"]
    input_stream_names = sessions[0]["input_feature_names"]

    # define some parameters
    Ntot = sum([sesh["spikes"].shape[0] for sesh in sessions]) # total number of neurons

    model_full = Encoder(input_streams, [], 1, Ntot, inv_link = "exp", noise_function = "Poisson", sqrt_counts = None) # instantiate model
    #model_full.eval_cv_function = model_full.eval_exp_poisson
    model_full.eval_cv_function = model_full.eval_exp_poisson_cv

    combinations = [["constant"], ["place"], ["direction"], ["place", "direction"], ["place_direction"], ["distance"], ["place", "direction", "distance"], ["place_direction", "distance"]]
    #combinations = [["place", "direction"]]
    #combinations = [["distance"]]

    for isesh, session in enumerate(sessions):
        name = session["session_name"]
        if name not in results.keys(): results[name] = {}
        Xbase = session["X"].to(device)
        spikes = session["spikes"].to(device)
        trials = session["trial_ids"]
        for combination in combinations:
            if combination == ["constant"]:
                input_inds = []
                X = torch.ones(1, spikes.shape[-1])
            else:
                input_inds = np.concatenate([input_streams[inp_inds[input_]] for input_ in combination if input_ != "constant"])
                X = Xbase[input_inds, ...]
            
            test_perf = model_full.eval_representation(X, spikes, cv = 6, alpha = eval_alpha, embed = False, trials = trials)
            
            results[name][tuple(combination)] = test_perf
            
            print(name, combination, len(input_inds), test_perf.mean(), np.round((time.time() - t0)/60, 1))
        
        pickle.dump(results, open("/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/reg_comparisons_cv6_dist.p", "wb"))
        
    
raise NotImplementedError


ress = pickle.load(open("/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/reg_comparisons_cv6.p", "rb"))
print(len(ress))

vars_ = [["constant"], ["place"], ["direction"], ["place", "direction"], ["place_direction"], ["place", "direction", "distance"], ["place_direction", "distance"]]
cats = []
from scipy.stats import ttest_1samp
thresh = 0.05

all_cats = []
all_results = []
all_sessions = []
for name, session in ress.items():
    results = np.array([session[tuple(var_)] for var_ in vars_])
    all_results.append(results)
    cats = np.zeros((results.shape[1], 7))
    all_sessions.append(np.array([name for _ in range(results.shape[1])]))
    
    for neuron in range(results.shape[1]):
        result = results[:, neuron, :]

        # is anything significantly better than constant?
        pconsts = [ttest_1samp(result[i, :] - result[0, :], 0).pvalue for i in [1,2,3,4]]

        # compare state-action combinations vs. the two individual ones
        ind_inds, comb_inds, dist_inds = np.array([1,2]), np.array([3,4]), np.array([5,6])
        best_individual = ind_inds[np.argmax(result[ind_inds, :].mean(-1))] # is state or action better?
        best_combined = comb_inds[np.argmax(result[comb_inds, :].mean(-1))] # is state or action better?
        cats[neuron, 1], cats[neuron, 2] = best_individual, best_combined

        if np.amin(pconsts) > thresh: # constant
            cats[neuron, 0] = 0 # constant
        elif ttest_1samp(result[best_combined, :] - result[best_individual, :], 0).pvalue > thresh:
            # best combined no better than best individual
            cats[neuron, 0] = 1
        else:
            # best combined better than best individual
            cats[neuron, 0] = 2
        cats[neuron, 3] = np.argmax(result[:5, :].mean(-1))
        
        best_dist = dist_inds[np.argmax(result[dist_inds, :].mean(-1))] # is state or action better?
        cats[neuron, 4] = best_dist
        if ttest_1samp(result[best_dist, :] - result[best_combined, :], 0).pvalue <= thresh:
            cats[neuron, 5] = 1 # SA-distance significantly better than SA
        cats[neuron, 6] = np.argmax(result.mean(-1)) # best of everything
            
        # if this significant?
        
    all_cats.append(cats)

all_cats = np.concatenate(all_cats, 0)
all_results = np.concatenate(all_results, 1)
all_sessions = np.concatenate(all_sessions)
all_animals = np.array([name[:2] for name in all_sessions])
unique_animals = np.unique(all_animals)

best_fits = []
for name in unique_animals:
    cats = all_cats[all_animals == name]
    best_fits.append(np.array([np.mean(cats[:, 6] == i) for i in range(len(vars_))]))
    
best_fits = np.array(best_fits)

ms, ss = np.mean(best_fits, axis = 0), np.std(best_fits, axis = 0)/np.sqrt(best_fits.shape[0])
xvals = np.arange(len(ms))

# plot raw performance
plt.figure(figsize = (6,3))
plt.bar(xvals, ms, yerr = ss)
for x in xvals:
    vals = best_fits[:, x]
    xs = x + np.linspace(-0.3, 0.3, len(vals))
    plt.scatter(xs, vals, color = "k")
labels = [" & ".join(var_) for var_ in vars_]
plt.xticks(xvals, labels, rotation = 50, ha = "right")
plt.ylabel("cell fraction")
plt.savefig("/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/figs/reg_comparisons.png", bbox_inches = "tight")
plt.close()

# plot signifiance

best_fits = np.zeros((len(unique_animals), 4))
for iname, name in enumerate(unique_animals):
    cats = all_cats[all_animals == name]
    best_fits[iname, 0] = np.mean(cats[:, 0] == 0) # constant is best
    best_fits[iname, 1] = np.mean(cats[:, 0] == 1) # single variable is best
    best_fits[iname, 2] = np.mean((cats[:, 0] == 2) & (cats[:, 5] == 0)) # state-action is best and state-action-distance no better
    best_fits[iname, 3] = np.mean((cats[:, 0] == 2) & (cats[:, 5] == 1)) # state-action-distsance is best

ms, ss = np.mean(best_fits, axis = 0), np.std(best_fits, axis = 0)/np.sqrt(best_fits.shape[0])
xvals = np.arange(len(ms))
plt.figure(figsize = (4,3))
plt.bar(xvals, ms, yerr = ss)
for x in xvals:
    vals = best_fits[:, x]
    xs = x + np.linspace(-0.3, 0.3, len(vals))
    plt.scatter(xs, vals, color = "k")
labels = ["constant", "state OR action", "state AND action", "state, action, and distance"]
plt.xticks(xvals, labels, rotation = 50, ha = "right")
plt.ylabel("cell fraction")
plt.savefig("/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/figs/reg_comparisons_significance.png", bbox_inches = "tight")
plt.close()


#### do place_direction vs distance ####

vars_ = [["place_direction"], ["distance"]]
all_res = []
for name, session in ress.items():
    all_res.append(np.array([session[tuple(var_)] for var_ in vars_]))

cat_res = np.concatenate([res.mean(-1) for res in all_res], axis = -1)

diffs = cat_res[0] - cat_res[1]

plt.figure(figsize = (4,4))
plt.plot([-10, 10], [-10, 10], color = plt.get_cmap("tab10")(0))
plt.scatter(cat_res[0], cat_res[1], color = "k", marker = ".", s = 20)
plt.xlabel(vars_[0])
plt.ylabel(vars_[1])
plt.xlim(-0.2,0.6)
plt.ylim(-0.2,0.6)
plt.savefig("/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/figs/reg_dist_v_SA.png", bbox_inches = "tight")
plt.close()

plt.figure(figsize = (5,3))
plt.hist(diffs, bins = np.linspace(-0.5, 0.5, 21))
plt.axvline(0, color = "k", label = "zero")
plt.axvline(np.mean(diffs), color = plt.get_cmap("tab10")(1), label = "mean")
for q in [0.05, 0.5, 0.95]:
    plt.axvline(np.quantile(diffs, q), color = plt.get_cmap("tab10")(2), label = f"q{str(q)[2:]}")
plt.xlabel("SA-minus-dist")
plt.ylabel("frequency")
plt.legend()
plt.savefig("/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/figs/reg_dist_SA_diff.png", bbox_inches = "tight")
plt.close()

