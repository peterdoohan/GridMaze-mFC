
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
from GridMaze.analysis.embedding_model.get_input_data import get_input_data, _downsample_navigation_spike_counts
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import convert
from GridMaze.maze import representations as mr
    
#%% set some parameters

seed = 1
np.random.seed(seed) # fix random seeds for now
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # run on GPU if possible
resolution = 0.1
min_spike_count = 300
maze_name = "maze_1"
simple_maze = mr.get_simple_maze(maze_name)

subjects = ['m2', 'm3', 'm4', 'm6', 'm7', 'm8']
subjects = ["m8"]
t0 = time.time()

coord2label = mr.get_maze_coord2label(simple_maze)


deltas = {(1,0): "E", (-1,0): "W", (0,1): "N", (0,-1): "S"}
decision_points = set()
for node1 in simple_maze.nodes:
    for node2 in simple_maze.neighbors(node1):
        if len(list(simple_maze.neighbors(node2))) >= 3: # only degree >= 3
            dir_ = deltas[tuple(np.array(node2) - np.array(node1))]
            decision_points.add(coord2label[node1]+"_"+dir_) # going in direction of node2 from node1 yields a decision point
            # the intermediate bridge is also fine
            try:
                decision_points.add(coord2label[(node1, node2)]+"_"+dir_)
            except:
                decision_points.add(coord2label[(node2, node1)]+"_"+dir_)
        
SAs = mr.get_maze_place_direction_pairs(simple_maze)
all_SAs = set([SA[0]+"_"+SA[1] for SA in SAs])

for subject in subjects:
    print(f"\n\nrunning for subject {subject}")
    sessions = gs.get_maze_sessions(
        subject_IDs=[subject],
        maze_names=[maze_name],
        days_on_maze="late",
        with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "navigation_routes_df"],
        must_have_data=True,
    )

    all_ress = []

    for isesh, session in enumerate(sessions):

        # load data from session obj
        navigation_spikes_df = session.get_navigation_activity_df(
            type="spikes",
            cluster_kwargs={"single_units": True, "multi_units": False},
            with_routes=False,
        )
        # downsample data from frame-by-frame to specified resolution
        if resolution:
            navigation_spikes_df = _downsample_navigation_spike_counts(navigation_spikes_df, resolution)

        times = navigation_spikes_df.time.values
        diff = times[1:] - times[:-1]
        #assert np.std(diff) / np.mean(diff) < 0.01 # should have continuous values
        print(np.std(diff) / np.mean(diff))

        shift_type = "states" # either shift based on either number of states or time elapsed
        all_states = navigation_spikes_df.maze_position.simple.values
        phases = navigation_spikes_df.trial_phase.values
        all_states[phases != "navigation"] = None
        for trial in navigation_spikes_df.trial.unique():
            if not np.isnan(trial):
                trial_inds = np.where(navigation_spikes_df.trial == trial)[0]
                goal_inds = trial_inds[(navigation_spikes_df.maze_position.simple.values[trial_inds] == navigation_spikes_df.goal.values[trial_inds[0]] )]
                all_states[goal_inds] = None
        
        boundaries = np.concatenate([np.zeros(1).astype(int), np.where(all_states[1:] != all_states[:-1])[0]+1]) # first index of each state
        offsets = np.arange(1,7)
        
        offset_array = np.array([np.vstack([copy.deepcopy(all_states)] + [np.array([None for _ in range(len(all_states))]) for _ in range(len(offsets))]) for _ in range(2)]) # future and past
        N_offsets = offset_array.shape[1]
    
        for itype, type_ in enumerate(["future", "past"]):
            sign = +1 if type_ == "future" else -1
            #print()
            dir_boundaries = boundaries if (type_ == "future") else np.flip(boundaries)-1 # go either forwards or backwards
            for i_off, offset in enumerate(offsets):
                ref = offset_array[itype, offset-1, :]
                next_ = offset_array[itype, offset, :]
                #print(ref[500:515])
                #print(ref[200:210], next_[200:210])
                for i_b, b in enumerate(dir_boundaries[:-1]):
                    inds = np.arange(b, dir_boundaries[i_b+1], sign) # indices for this state
                    cur_state = ref[inds[0]]
                    next_state = ref[inds[-1]+sign]
                    if None in [cur_state, next_state]: # if we're at a trial boundary
                        next_[inds] = None
                    else:
                        assert cur_state != next_state
                        next_[inds] = next_state
        
        for offset in [0]+list(offsets):
            navigation_spikes_df["future", str(offset)] = offset_array[0, offset, :]
            navigation_spikes_df["past", str(offset)] = offset_array[1, offset, :]
            
        # df2 = filt.filter_navigation_rates_df(navigation_spikes_df, navigation_only=False, moving_only=False, exclude_time_at_goal=True)
        # df3 = filt.filter_navigation_rates_df(navigation_spikes_df, navigation_only=True, moving_only=False, exclude_time_at_goal=True)

        navigation_spikes_df = filt.filter_navigation_rates_df(
            navigation_spikes_df, navigation_only=True, moving_only=True, exclude_time_at_goal=True
        )
            
        # filter data based on shortest path distance to goal (remove off task behaviour)
        max_distance = 2.0
        if max_distance:
            navigation_spikes_df = navigation_spikes_df[~navigation_spikes_df.distance_to_goal.geodesic.gt(max_distance)]
        # filter based on number of spikes (after all the other filters)

        raw_spike_counts = navigation_spikes_df["spike_count"].values
        enough_spikes = np.where(raw_spike_counts.sum(0) >= min_spike_count)[0]
        raw_spike_counts = raw_spike_counts[:, enough_spikes]


        ### now get neural activity and behaviour ####
        spike_counts = np.sqrt(raw_spike_counts)

        perfs = []

        # also include current state-action as a regressor
        df_SA = navigation_spikes_df.maze_position.simple+ "_" + navigation_spikes_df.cardinal_movement_direction
        SA_1hot = convert.place_direction2onehot(df_SA.values, simple_maze)
        
        # and goal
        goal_1hot = convert.goal2onehot(navigation_spikes_df.goal.values)
        SAG_1hot = np.concatenate([SA_1hot, goal_1hot], axis = -1) # combine with SA

        Ys = np.array([navigation_spikes_df.future.values, navigation_spikes_df.past.values])
        Ys_1hot = np.array([[convert.place2onehot(Y[:, i], simple_maze) for i in range(Y.shape[-1])] for Y in Ys])
        
        print(Ys_1hot.sum((-1, -2)))
        
        future_and_past = Ys_1hot.sum(-1).mean((0,1)) == 1 # data for all future and past
        at_DP = np.array([sa in decision_points for sa in df_SA]) # animal is going to a decision point
        
        keep_inds = np.where(future_and_past)[0] 
        keep_inds = np.where(future_and_past & at_DP)[0] 
        
        print(len(keep_inds))
        
        Xspikes, Ys_ind = spike_counts[keep_inds, :], Ys_1hot[..., keep_inds, :].argmax(-1)
        #XSA = SAG_1hot[keep_inds, :]
        XSA = SA_1hot[keep_inds, :]
        trials = navigation_spikes_df.trial.values[keep_inds]

        ### now run logistic regression ###
        from sklearn.linear_model import LogisticRegression

        cv = 4
        unique_trials = np.unique(trials)
        trial_splits = [[] for _ in range(cv)]
        for trial in unique_trials:
            trial_splits[int(trial) % cv].append(trial)
        inds = [np.concatenate([np.where(trials == trial_id)[0] for trial_id in trial_split]) for trial_split in trial_splits]

        #possible_Xs = [Xspikes, XSA, np.concatenate([0.02*Xspikes, XSA], axis = -1)] # for future_decoding2.p
        possible_Xs = [Xspikes, XSA, np.concatenate([0.1*Xspikes, XSA], axis = -1)] # for first one (this worked)
        N_models = len(possible_Xs)
        print([X.shape for X in possible_Xs])
        
        scores = np.zeros((N_models, 2, N_offsets, N_offsets, cv)) # train decoder and apply to everything
        for itype in range(2): # decode future vs past
            for imodel, X in enumerate(possible_Xs):
                for ishift in range(N_offsets):
                    y = Ys_ind[itype, ishift, :]
                    for fold in range(cv):
                        test, train = inds[fold], np.concatenate([inds[f] for f in range(cv) if f != fold])
                        
                        clf = LogisticRegression(C = 1e-0)
                        clf.fit(X[train, :], y[train])
                        
                        # apply to each of the different offsets
                        for ishift_test in range(N_offsets):
                            ytest = Ys_ind[itype, ishift_test, :] # use this decoder to predict at different times
                            score = clf.score(X[test, :], ytest[test])
                            #print(itype, ishift, fold, score)
                            scores[imodel, itype, ishift, ishift_test, fold] = score
                            #scores[n, fold] = self.eval_cv_function(z[train], y_n[train], z[test], y_n[test], alpha = alpha, cv = cv)
                print(isesh, "of", len(sessions), ":", imodel, np.round(np.diag(np.mean(scores[imodel, itype, ...], axis = (-1))), 3), "  t =", np.round((time.time()-t0)/60, 1))
    
        all_ress.append(scores)


    pickle.dump(all_ress, open(f"/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/future_past_decoding/DP_{subject}.p", "wb"))

#all_ress = pickle.load(open("/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/future_past_decoding.p", "rb"))

subjects = ['m2', 'm3', 'm4', 'm6', 'm7', 'm8']
base_dir = f"/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/future_past_decoding/DP_"
all_all_ress = np.array([pickle.load(open(f"{base_dir}{subject}.p", "rb")) for subject in subjects])
all_ress = np.mean(all_all_ress, axis = 1) # average over sessions for each animal

### first do the analyses from just neural data ###

means = np.array(all_ress)[:, 0, ...].mean(-1) # sessions by fwd/bwd by train offset by test offset
m_all, s_all = means.mean(0), means.std(0)/np.sqrt(means.shape[0]) # mean and std across sessions (fwd/bwd by train offset by test offset)

m, s = [np.diagonal(arr, axis1 = -1, axis2 = -2) for arr in [m_all, s_all]]

offsets = np.arange(m.shape[1])
plotinds = offsets
plotinds = offsets[::2]

cols = [plt.get_cmap("tab10")(i) for i in range(2)]
plt.figure()
for itype, type_ in enumerate(["future", "past"]):
    c = cols[itype]
    xs = offsets if itype == 0 else -offsets
    plt.plot(xs[plotinds], m[itype][plotinds], label = type_, color = c)
    plt.fill_between(xs[plotinds], (m-s)[itype][plotinds], (m+s)[itype][plotinds], alpha = 0.2, color = c)
    for itrain in offsets[plotinds]:
        plt.plot(xs[plotinds], m_all[itype, itrain, :][plotinds], color = c, alpha = 1-(itrain+5)/(len(offsets)+5))
        
plt.xlim(-offsets[plotinds][-1], offsets[plotinds][-1])
plt.ylim(0, np.amax(m+s))
plt.axhline(1/(len(simple_maze.nodes)+len(simple_maze.edges)), color = "k")
plt.legend()
plt.xlabel("offset")
plt.ylabel("accuracy")
plt.savefig("/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/figs/future_past_decoding_temp.png", bbox_inches = "tight")
plt.close()

print(m)
print(m[0] - m[1])


### now do the analysis with/without model (and also the difference) ###

means = np.diagonal(np.array(all_ress)[:, 1:, ...].mean(-1), axis1 = -1, axis2 = -2) # sessions by model type by fwd/bwd by offset
m, s = means.mean(0), means.std(0)/np.sqrt(means.shape[0]) # mean and std across sessions (model type by fwd/bwd by offset)

offsets = np.arange(m.shape[-1])

cols = [plt.get_cmap("tab10")(i) for i in range(2)]
plt.figure()
for itype, type_ in enumerate(["future", "past"]):
    c = cols[itype]
    for imodel in range(2):
        ls = "-" if imodel == 1 else "--"
        xs = offsets if itype == 0 else -offsets
        plt.plot(xs, m[imodel, itype, :], label = type_, color = c, ls = ls)
        plt.fill_between(xs, (m-s)[imodel, itype, :], (m+s)[imodel, itype, :], alpha = 0.2, color = c)
  
plt.xlim(-offsets[-1], offsets[-1])
plt.legend()
plt.xlabel("offset")
plt.ylabel("accuracy")
plt.savefig("/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/figs/future_past_decoding_with_behav.png", bbox_inches = "tight")
plt.close()

# now plot difference

diff = means[:, 1, ...] - means[:, 0, ...] # with spikes minus without

# normalize by unexplained variance
#diff = diff / (1-means[:, 0, ...])


m, s = diff.mean(0), diff.std(0)/np.sqrt(diff.shape[0]) # mean and std across sessions (fwd/bwd by offset)

plt.figure()
for itype, type_ in enumerate(["future", "past"]):
    c = cols[itype]
    xs = offsets if itype == 0 else -offsets
    plt.plot(xs, m[itype, :], label = type_, color = c, ls = ls)
    plt.fill_between(xs, (m-s)[itype], (m+s)[itype], alpha = 0.2, color = c)
  
plt.xlim(-offsets[-1], offsets[-1])
plt.legend()
plt.xlabel("offset")
plt.ylabel("accuracy")
plt.savefig("/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/figs/future_past_decoding_with_behav_diff.png", bbox_inches = "tight")
plt.close()




