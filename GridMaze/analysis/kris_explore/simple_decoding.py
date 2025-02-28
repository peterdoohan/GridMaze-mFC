
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
import GridMaze.analysis.core.get_sessions as gs

from GridMaze.analysis.embedding_model.embedding_utils import Encoder, train_model
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
input_features=["place_direction", "goal", "current_route", "next_route"]#, "tower_bridge"]

subjects = ["m3"]
sessions = get_input_data(subject_IDs=subjects,
    maze_name="maze_1",
    input_features=input_features,
    distance_metrics=("distance_to_goal", "geodesic"),
    include_multi_unit=False,
    navigation_only=True,
    moving_only=True,
    resolution=0.2,  # s
    max_distance=1.8,  # m
    n_distance_bins=20,
    min_spike_count = 300,)


# first try to predict current route just from place-direction+goal

X_vars = ["place_direction", "goal"] # also include current route when predicting next
y_var = "current_route"
#y_var = "next_route"

s0 = sessions[0]
X_inds = np.concatenate([s0["X_type_inds"][input_features.index(feat)] for feat in X_vars])
y_inds = s0["X_type_inds"][input_features.index(y_var)]

Xs = [s["X"][X_inds, :] for s in sessions]

ys = [s["X"][y_inds] for s in sessions]


# just try a simple logistic regression

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

possible_Xlabels = ["within-session", "across-session", "spikes", "within_spikes", "across_spikes"]
cv = 5
all_perfs = np.zeros((len(sessions), len(possible_Xlabels), cv))

for test_ind in range(len(sessions)):
    test_inds = [test_ind]
    train_inds = [ind for ind in np.arange(len(Xs)) if ind not in test_inds]

    Xtrain = np.concatenate([Xs[ind].T for ind in train_inds], axis = 0)
    Xtest = np.concatenate([Xs[ind].T for ind in test_inds], axis = 0)
    ytrain = np.concatenate([ys[ind].argmax(0) for ind in train_inds])
    ytest = np.concatenate([ys[ind].argmax(0) for ind in test_inds])

    clf = LogisticRegression(random_state=0, C = 100).fit(Xtrain, ytrain)
    #clf = MLPClassifier(hidden_layer_sizes=(100,50), alpha = 0.001, batch_size = 500).fit(Xtrain, ytrain)
    print(f"\n{test_ind}: {clf.score(Xtest, ytest)}")
    
    # neurons = ...
    preds_classes = clf.predict(Xtest)[:, None]
    preds = clf.predict_proba(Xtest)
    spikes = np.sqrt(np.concatenate([sessions[ind]["spikes"].T for ind in test_inds], axis = 0))
    # normalize
    spikes = (spikes - spikes.mean(0, keepdims = True))
    #spikes = spikes / spikes.std(0, keepdims = True)
    
    # try some PCA
    pca = PCA(n_components = min(8, spikes.shape[1]))
    pcs = pca.fit_transform(spikes)
    print(pca.explained_variance_ratio_.sum())
    #spikes = pcs
    
    spikes = spikes*0.01 # effectively increase regularization for neural data
    
    trials = np.concatenate([sessions[ind]["trial_ids"] for ind in test_inds])
    
    unique_trials = np.unique(trials)
    trial_splits = [[] for _ in range(cv)]
    for trial in unique_trials:
        trial_splits[int(trial) % cv].append(trial)
    inds = [np.concatenate([np.where(trials == trial_id)[0] for trial_id in trial_split]) for trial_split in trial_splits]
    
    possible_Xvals = [Xtest, preds, spikes, np.concatenate([Xtest, spikes], axis = 1), np.concatenate([preds, spikes], axis = 1)]
    
    import sklearn
  
    yvals = ytest
    for iX, Xvals in enumerate(possible_Xvals):
        for fold in range(cv):
            test, train = inds[fold], np.concatenate([inds[f] for f in range(cv) if f != fold])
            
            clf = LogisticRegression(C = 100).fit(Xvals[train], yvals[train])
            
            # clf = sklearn.linear_model.LogisticRegressionCV(Cs = 8, cv = sklearn.model_selection.KFold(n_splits = 4))
            # clf.fit(Xvals[train], yvals[train])
            
            
            all_perfs[test_ind, iX, fold] = clf.score(Xvals[test], yvals[test])
            
            if iX == 10:
                print(len(np.unique(yvals[train])), len(np.unique(yvals[test])), len(np.intersect1d(np.unique(yvals[train]), np.unique(yvals[test]))))
        try:
            print(iX, clf.C_.mean())
        except:
            None
            
    print(all_perfs[test_ind, ...].mean(-1))
 

perfs = all_perfs.mean(-1)
m, s = perfs.mean(0), perfs.std(0)/np.sqrt(len(perfs))
xs = np.arange(len(possible_Xlabels))

print(m)

plt.figure()
plt.bar(xs, m, yerr = s)
plt.xticks(xs, possible_Xlabels, rotation = 45, ha = "right")
plt.ylabel("route prediction")
plt.savefig("/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/figs/route_prediction.png", bbox_inches = "tight")
plt.close()

