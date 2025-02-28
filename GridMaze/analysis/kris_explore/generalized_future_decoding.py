

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
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

from importlib import reload
import GridMaze.analysis.embedding_model.get_input_data
import GridMaze.analysis.embedding_model.embedding_utils
from GridMaze.analysis.embedding_model.embedding_utils import Encoder, train_model, calc_poisson_deviance
from GridMaze.analysis.embedding_model.get_input_data import get_input_data, _downsample_navigation_spike_counts
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import convert
from GridMaze.maze import representations as mr

from GridMaze.analysis.kris_explore.utils import res_dir
from scipy.stats import ttest_1samp

#%% set gobal some parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # run on GPU if possible
resolution = 0.1 # seconds per bin
subjects = ['m2', 'm3', 'm4', 'm6', 'm7', 'm8']
future_delta = 1
def round(num, digits = 2):
    return np.round(num, digits)

  
#%% run decoding analysis

Cs = 10**np.linspace(-4,4,10) # possible inverse reg strengths
shuffle = False
seed = 1
max_bins = 10 # maximally 1 second of neural activity
acc_byclass = True
average_activity = True # determines whether to average activity at each preceding tower or treat bins as independent samples

for seed in [1]:

    if shuffle: print("SHUFFLING!", seed)

    t0 = time.time()    
    for maze_name in ["maze_1", "maze_2"]:
        
        maze_data = {}
        np.random.seed(seed) # fix random seeds for now
        torch.manual_seed(seed)
        
        # get some maze utuils
        simple_maze = mr.get_simple_maze(maze_name)
        coord2label = mr.get_maze_coord2label(simple_maze)
        label2coord = mr.get_maze_label2coord(simple_maze)

        # node degree for all nodes
        node_degrees = {}
        for node in simple_maze.nodes:
            name = coord2label[node]
            node_degrees[name] = len(list(simple_maze.neighbors(node)))

        # all unique locations on the maze
        maze_locs = np.sort(np.array(list(node_degrees.keys())))
        maze_degrees = np.array([node_degrees[loc] for loc in maze_locs]) # corresponding node degrees
        degree4s = maze_locs[np.where(maze_degrees == 4)[0]] # all locations with node degree 4

        # load data

        for subject in subjects:
            maze_data[subject] = {"accs": [], "days": [], "locs": [], "n_data": []}

            print(f"\nrunning for subject {subject}")
            sessions = gs.get_maze_sessions(
                subject_IDs=[subject],
                maze_names=[maze_name],
                days_on_maze="late",
                with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "navigation_routes_df"],
                must_have_data=True,
            )

            for session in sessions:
                # load data from session obj
                navigation_spikes_df = session.get_navigation_activity_df(
                    type="spikes",
                    cluster_kwargs={"single_units": True, "multi_units": False},
                    with_routes=False,
                )

                # only consider navigation periods
                df = filt.filter_navigation_rates_df(
                            navigation_spikes_df, navigation_only=True, moving_only=True, exclude_time_at_goal=True
                        )
                # where is the animal at every point in time?
                all_locs = df.maze_position.simple.values
                # how many spikes at every point in time for every neuron?
                spike_counts = df["spike_count"].values

                # now iterate through degree 4 nodes
                for loc4 in degree4s:
                    # what are the neighbors of this node?
                    neighbors = [coord2label[n] for n in simple_maze.neighbors(label2coord[loc4])]
                    
                    # df indices where the animal moves between states
                    trans = np.where(all_locs[1:] != all_locs[:-1])[0]+1 # first index at each state
                    loc_seq = all_locs[trans] # sequence of locations visited

                    loc4_inds = np.where(loc_seq == loc4)[0] # indices in sequence where the animal visits degree 4 node
                    loc4_inds = loc4_inds[(loc4_inds > future_delta) & (loc4_inds < len(loc_seq)-2)] # not early or last node visited (need past and future)
                    currs = loc_seq[loc4_inds] # these should all be loc4 by definition
                    
                    prevs, nexts = loc_seq[loc4_inds-2], loc_seq[loc4_inds+2] # previous and subsequent location

                    eff_prevs, eff_nexts = copy.deepcopy(prevs), copy.deepcopy(nexts) # no shuffle
                    # optionally decorrelate behaviour and neural activity as a control
                    if shuffle:
                        perm_inds = np.random.permutation(np.arange(len(currs)))
                        eff_prevs, eff_nexts = eff_prevs[perm_inds], eff_nexts[perm_inds] # same shuffle for future and past

                    # array of neural activity for every past/future location pair
                    neural_acts = {loc: {} for loc in neighbors}
                    Nneuron = spike_counts.shape[-1]
                    for prev in neighbors: # for each past
                        for next_ in neighbors: # for each future
                            
                            # at which traverses did I go from prev to next_? (indexing into 'loc_seq' and 'trans')
                            type_inds = loc4_inds[np.where((eff_prevs == prev) & (eff_nexts == next_))[0]]
                            if len(type_inds) == 0: # no data for this combination
                                neural_acts[next_][prev] = np.zeros((0, Nneuron)) # store empty array
                            else:
                                # what are the indices of the full df where I was future_delta towers before the junction?
                                inds_at_prev_loc = [np.arange(trans[ind-future_delta*2], trans[ind-future_delta*2+1])[-max_bins:] for ind in type_inds] # up to 10
                                if average_activity:
                                    # what is the average activity at this tower for each traversal?
                                    neural_acts[next_][prev] = np.array([spike_counts[inds, :].mean(0) for inds in inds_at_prev_loc]) # traversals x N_neurons
                                else:
                                    neural_acts[next_][prev] = np.concatenate([spike_counts[inds, :] for inds in inds_at_prev_loc]) # traversals x N_neurons

                    session_tower_data = [] # data for this session/tower
                    ST_Cs, pred_majs, true_majs = [], [], []
                    for test_next in neighbors: # for each target location
                        for test_prev in [n for n in neighbors if n!=test_next]: # for each source location
                            
                            # prev locations used for training data
                            train_prev = [neigh for neigh in neighbors if neigh not in [test_prev, test_next]]

                            # train trials where we did go to test_next
                            train_true = np.concatenate([neural_acts[test_next][prev] for prev in train_prev])
                            
                            # train trials where we went from a train loc to NOT-test_next
                            train_false = [[neural_acts[next_][prev] for next_ in neighbors if (next_ not in [prev, test_next])] for prev in train_prev]
                            # concatenate across non-test future locations for each training source and across training sources
                            train_false = np.concatenate([np.concatenate(arr) for arr in train_false])

                            # test trials where we did go to test_next
                            test_true = neural_acts[test_next][test_prev]
                            # test trials where we went to NOT-test_next
                            test_false = np.concatenate([neural_acts[next_][test_prev] for next_ in neighbors if (next_ not in [test_prev, test_next])])

                            # make sure we have some training data and at least one test datapoint!
                            if acc_byclass:
                                conds = (min(len(train_true), len(train_false))) >= 3 and (min(len(test_true), len(test_false)) >= 1)
                            else:
                                conds = (min(len(train_true), len(train_false))) >= 3 and (max(len(test_true), len(test_false)) >= 1)
                                
                            #print(test_true.shape, test_false.shape, conds)
                            
                            if conds:
                                
                                # training data across positive and negative samples (neural activity and labels)
                                train_joint = np.concatenate([train_true, train_false])
                                train_labels = np.concatenate([np.ones(len(train_true)), np.zeros(len(train_false))]).astype(int)
                                # test data
                                test_joint = np.concatenate([test_true, test_false])
                                test_labels = np.concatenate([np.ones(len(test_true)), np.zeros(len(test_false))]).astype(int)

                                # if shuffle: # shuffle labels as a control
                                #     train_labels = np.random.permutation(train_labels)
                                #     test_labels = np.random.permutation(test_labels)

                                # now train a logistic regression model
                                # balance class weights for 50% chance level
                                # do hold-one-out crossvalidation
                                accs = np.zeros((len(Cs), train_joint.shape[0])) # accuracy for each combination of reg and crossval test ind
                                for iC, C in enumerate(Cs): # for each reg strength
                                    clf = LogisticRegression(class_weight = "balanced", C = C) # instantiate logistic regression
                                    for test_ind in range(train_joint.shape[0]): # for each held out traversal
                                        # crossval training inds are anything that is not the test ind
                                        train_inds = np.array([ind for ind in range(train_joint.shape[0]) if ind != test_ind])
                                        # fit model
                                        clf.fit(train_joint[train_inds, :], train_labels[train_inds])
                                        # correct or incorrect or our single held-out datapoint?
                                        accs[iC, test_ind] = float(clf.predict(train_joint[test_ind, :][None, :]) == train_labels[test_ind])

                                # accuracy for each training class (target or non-target)
                                accs_per_class = np.array([accs[:, train_labels == class_].mean(-1) for class_ in [0,1]])
                                # mean accuracy across classes for each reg strength
                                mean_accs = accs_per_class.mean(0)
                                # we pick one smaller than the smallest of best Cs (strongest regularization) since we'll generalize even less to held-out past
                                best_C_ind = np.amin(np.where(mean_accs == np.amax(mean_accs))[0]) 
                                best_C_ind = max(0, best_C_ind-1)
                                best_C = Cs[best_C_ind]

                                # now fit our final model on all the training data
                                clf = LogisticRegression(class_weight = "balanced", C = best_C).fit(train_joint, train_labels)
                                # evaluate on our test data -> compute mean accuracy
                                preds = clf.predict(test_joint)
                                if acc_byclass: # compute accuracy for each class, then average
                                    test_acc = np.mean([np.mean((preds == test_labels)[test_labels == label]) for label in np.unique(test_labels)])
                                else: # compute mean accuracy across all data
                                    test_acc = np.mean(preds == test_labels)
                                    
                                # append mean accuracy for this source/target combination
                                session_tower_data.append(test_acc)
                                ST_Cs.append(np.log10(best_C))
                                pred_majs.append(np.amax([np.mean(preds == i) for i in np.unique(preds)]))
                                true_majs.append(np.amax([np.mean(test_labels == i) for i in np.unique(test_labels)]))
                                
                                #print(test_next, test_prev, test_acc, "(", train_joint.shape, test_joint.shape, train_true.shape, train_false.shape, best_C,")")
                    
                    # compute mean and sem across source/targets for this session+tpwer
                    if len(session_tower_data) >= 1:
                        m, s = np.mean(session_tower_data)-0.5, np.std(session_tower_data)/np.sqrt(len(session_tower_data))
                        maze_data[subject]["accs"].append(m) # append mean accuracy across source/target locations
                        maze_data[subject]["days"].append(session.day_on_maze) # append mean across source/target locations
                        maze_data[subject]["locs"].append(loc4) # append mean across source/target locations
                        maze_data[subject]["n_data"].append(len(loc4_inds)) # append mean across source/target locations
                        print(f"day={session.day_on_maze},  loc={loc4},  m={round(m)},  s={round(s)},  C={round(np.mean(ST_Cs))},  pred_maj={round(np.mean(pred_majs))},  true_maj={round(np.mean(true_majs))}") # subtract baseline of 0.5
                    else:
                        print(f"day={session.day_on_maze},  loc={loc4}, no data")

            all_accs = maze_data[subject]["accs"]
            print(f"mean for mouse {subject} {maze_name}: {np.mean(all_accs)}, sem = {np.std(all_accs)/np.sqrt(len(all_accs))}, t = {np.round((time.time() - t0)/60, 1)}")

        pickle.dump(maze_data, open(f"{res_dir}generalized_decoding/data/{maze_name}_shuffle{int(shuffle)}_delta{future_delta}_seed{seed}.p", "wb"))
        # maze_data = pickle.load(open(f"{res_dir}generalized_decoding/data/{maze_name}_shuffle{int(shuffle)}_delta{future_delta}_seed{seed}.p", "rb"))

        # plot result for this maze
        mouse_accs = [maze_data[subject]["accs"] for subject in subjects]
        # mean and sem across session_towers for each mouse
        m, s = [np.mean(accs) for accs in mouse_accs], [np.std(accs)/np.sqrt(len(accs)) for accs in mouse_accs]
        mtot, stot = np.mean(m), np.std(m)/np.sqrt(len(m))
        
        xs = np.arange(len(m))
        plt.figure(figsize = (5,3))
        plt.axhline(0, color = "k", lw = 1)
        plt.bar(xs, m, yerr = s)
        plt.xticks(xs, subjects)
        plt.axhline(mtot, color = np.ones(3)*0.5, lw = 1.5)
        plt.fill_between([xs[0]-0.75, xs[-1]+0.75], [mtot-2*stot], [mtot+2*stot], color = np.ones(3)*0.5, alpha = 0.2)
        plt.ylabel("accuracy")
        plt.xlabel("subject")
        plt.xlim(xs[0]-0.6, xs[-1]+0.6)
        plt.title(f"{maze_name} d{future_delta}{' shuffle' if shuffle else ''}")
        plt.savefig(f"{res_dir}generalized_decoding/figs/generalization_{maze_name}_shuffle{int(shuffle)}_delta{future_delta}_seed{seed}.png", bbox_inches = "tight")
        plt.close()
        
        # print mean, sem, and one-sided t-test across animals
        print(f"{maze_name}: {mtot}, {stot}")
        print(ttest_1samp(m, 0, alternative = "greater"))

raise NotImplementedError

cat_accs = np.concatenate(mouse_accs)
cat_days = np.concatenate(mouse_days)
unique_days = np.unique(cat_days)
acc_by_day = [cat_accs[cat_days == day].mean() for day in unique_days]
print(unique_days)
print(acc_by_day)
print(pearsonr(unique_days, acc_by_day))


### write code to aggregate shuffled data across seeds and check that the analysis is not biased ###

future_delta = 1
all_ms = []
all_ps = []
for seed in range(1,15):
    for maze_name in ["maze_1", "maze_2"]:
        maze_data = pickle.load(open(f"{res_dir}generalized_decoding/data/{maze_name}_shuffle{int(shuffle)}_delta{future_delta}_seed{seed}.p", "rb"))
        mouse_accs = [maze_data[subject]["accs"] for subject in subjects]
        m, s = [np.mean(accs) for accs in mouse_accs], [np.std(accs)/np.sqrt(len(accs)) for accs in mouse_accs]
        mtot, stot = np.mean(m), np.std(m)/np.sqrt(len(m))
        all_ms.append(mtot)
        all_ps.append(ttest_1samp(m, 0, alternative = "greater").pvalue)

all_ms, all_ps = np.array(all_ms), np.array(all_ps)
print(np.mean(all_ms), np.std(all_ms), np.std(all_ms)/np.sqrt(len(all_ms)))
print(np.mean(all_ps < 0.05), np.quantile(all_ps, [0.05, 0.5, 0.95]))


### legacy code ###

            
# traj_freq = np.zeros((len(neighbors), len(neighbors)))
# for i1, n1 in enumerate(neighbors): # where did I come from?
#     for i2, n2 in enumerate(neighbors): # where did I go to?
#         traj_freq[i1, i2] = np.sum((prevs == n1) & (currs == loc4) & (nexts == n2))

# plt.figure(figsize = (3,3))
# plt.imshow(traj_freq, vmin = 0, vmax = traj_freq.max())
# plt.colorbar()
# plt.xticks(np.arange(len(neighbors)), neighbors)
# plt.yticks(np.arange(len(neighbors)), neighbors)
# plt.xlabel("next")
# plt.ylabel("previous")
# plt.savefig(f"{res_dir}generalized_decoding/maze1_example_trans.png", bbox_inches = "tight")
# plt.close()

# degrees = [1,2,3,4]
# freq = [np.sum(all_degrees == d) for d in degrees]
# plt.figure()
# plt.bar(degrees, freq)
# plt.xlabel("node degree")
# plt.ylabel("frequency")
# plt.savefig(f"{res_dir}generalized_decoding/maze1_freq.png", bbox_inches = "tight")
# plt.close()


# if shuffle: # shuffle labels as a control
#     train_labels = np.random.permutation(train_labels)
#     test_labels = np.random.permutation(test_labels)


# train_false = [[neural_acts[next_][prev] for next_ in neighbors
#     if (next_ not in [prev, test_next]) and (len(neural_acts[next_][prev]) > 0)]
#                 for prev in train_prev]



#%% try sampling pseudo-sessions

shuffle = False

all_test_accs = []
for maze_name in ["maze_1", "maze_2"]:
    simple_maze = mr.get_simple_maze(maze_name)
    coord2label = mr.get_maze_coord2label(simple_maze)
    label2coord = mr.get_maze_label2coord(simple_maze)

    # node degree for all nodes
    node_degrees = {}
    for node in simple_maze.nodes:
        name = coord2label[node]
        node_degrees[name] = len(list(simple_maze.neighbors(node)))

    # all unique locations on the maze
    maze_locs = np.sort(np.array(list(node_degrees.keys())))
    maze_degrees = np.array([node_degrees[loc] for loc in maze_locs]) # corresponding node degrees
    degree4s = maze_locs[np.where(maze_degrees == 4)[0]] # all locations with node degree 4


    all_sessions = gs.get_maze_sessions(
                    subject_IDs=subjects,
                    maze_names=[maze_name],
                    days_on_maze="late",
                    with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "navigation_routes_df"],
                    must_have_data=True,
                )


    maze_test_accs = []
    for loc4 in degree4s:
        
        neighbors = [coord2label[n] for n in simple_maze.neighbors(label2coord[loc4])]
        loc_test_accs = []

        for destination in neighbors:
            for source in [n for n in neighbors if n!= destination]:

                all_session_data = []
                for session in all_sessions:
                    # load data from session obj
                    navigation_spikes_df = session.get_navigation_activity_df(
                        type="spikes",
                        cluster_kwargs={"single_units": True, "multi_units": False},
                        with_routes=False,
                    )

                    # only consider navigation periods
                    df = filt.filter_navigation_rates_df(
                                navigation_spikes_df, navigation_only=True, moving_only=True, exclude_time_at_goal=True
                            )
                    # where is the animal at every point in time?
                    all_locs = df.maze_position.simple.values
                    # how many spikes at every point in time for every neuron?
                    spike_counts = df["spike_count"].values

                    # df indices where the animal moves between states
                    trans = np.where(all_locs[1:] != all_locs[:-1])[0]+1 # first index at each state
                    loc_seq = all_locs[trans] # sequence of locations visited

                    loc4_inds = np.where(loc_seq == loc4)[0] # indices in sequence where the animal visits degree 4 node
                    loc4_inds = loc4_inds[(loc4_inds > future_delta) & (loc4_inds < len(loc_seq)-2)] # not early or last node visited (need past and future)
                    currs = loc_seq[loc4_inds] # these should all be loc4 by definition

                    prevs, nexts = loc_seq[loc4_inds-2], loc_seq[loc4_inds+2] # previous and subsequent tower

                    eff_prevs, eff_nexts = copy.deepcopy(prevs), copy.deepcopy(nexts) # no shuffle
                    # optionally decorrelate behaviour and neural activity as a control
                    if shuffle:
                        perm_inds = np.random.permutation(np.arange(len(currs)))
                        eff_prevs, eff_nexts = eff_prevs[perm_inds], eff_nexts[perm_inds] # same shuffle for future and past

                    # array of neural activity for every past/future location pair
                    keys = ["train_true", "train_false", "test_true", "test_false"]
                    session_data = {key: [] for key in keys}
                    Nneuron = spike_counts.shape[-1]
                    #print(session)
                    for prev in neighbors: # for each past
                        for next_ in neighbors: # for each future
                            
                            # at which traverses did I go from prev to next_? (indexing into 'loc_seq' and 'trans')
                            type_inds = loc4_inds[np.where((eff_prevs == prev) & (eff_nexts == next_))[0]]
                            if len(type_inds) == 0: # no data for this combination
                                neural_activity = np.zeros((0, Nneuron)) # store empty array
                            else:
                                # what are the indices of the full df where I was future_delta towers before the junction?
                                inds_at_prev_loc = [np.arange(trans[ind-future_delta*2], trans[ind-future_delta*2+1])[-max_bins:] for ind in type_inds] # up to 10
                                # what is the average activity at this tower for each traversal?
                                neural_activity = np.array([spike_counts[inds, :].mean(0) for inds in inds_at_prev_loc]) # traversals x N_neurons
                                
                            key = ["train", "test"][int(prev == source)]+"_"+["false", "true"][int(next_ == destination)]
                            session_data[key].append(neural_activity)

                    data_type_lengths = [np.sum([len(arr) for arr in session_data[key]]) for key in keys]
                    if min(data_type_lengths) >= 1: # need to have data for all categories
                        #print("data woo", data_type_lengths)
                        all_session_data.append({key: np.concatenate(session_data[key]) for key in keys})
                    else:
                        None
                        #print("No data :(", source, destination, data_type_lengths)

                print(source, destination, len(all_session_data))

                print([len(sesh["test_true"]) for sesh in all_session_data])

                rng = np.random.default_rng()
                pseudo_session = {}
                Nsamps = 1000
                for key in keys:
                    pseudo_session[key] = np.array([np.concatenate([rng.choice(sesh[key]) for sesh in all_session_data]) for _ in range(Nsamps)])


                # training data across positive and negative samples (neural activity and labels)
                train_joint = np.concatenate([pseudo_session["train_true"], pseudo_session["train_false"]])
                train_labels = np.concatenate([np.ones(Nsamps), np.zeros(Nsamps)]).astype(int)
                # test data
                test_joint = np.concatenate([pseudo_session["test_true"], pseudo_session["test_false"]])
                test_labels = np.concatenate([np.ones(Nsamps), np.zeros(Nsamps)]).astype(int)

                pseudo_Cs = 10**np.linspace(-10,-2,5)
                test_accs = np.zeros(len(pseudo_Cs))
                for iC, C in enumerate(pseudo_Cs):
                    # now fit our model on all the training data
                    clf = LogisticRegression(class_weight = "balanced", C = C).fit(train_joint, train_labels)
                    # evaluate on our test data -> compute mean accuracy
                    preds = clf.predict(test_joint)
                    test_acc = np.mean(preds == test_labels)
                    test_accs[iC] = test_acc                  

                print(test_accs)

                loc_test_accs.append(test_accs)

        maze_test_accs.append(np.array(loc_test_accs))
        
    all_test_accs.append(np.array(maze_test_accs))


bymaze_accs = np.concatenate([np.array(accs) for accs in all_test_accs])
print(bymaze_accs.mean(1))
print(bymaze_accs.mean(1).mean(0))


#%% try to compare neural similarity when the future is the same or different ###


shuffle = False
seed = 1
acc_byclass = True
max_bins = 10

t0 = time.time()    
for maze_name in ["maze_1", "maze_2"]:
    
    maze_data = {}
    np.random.seed(seed) # fix random seeds for now
    torch.manual_seed(seed)
    
    # get some maze utuils
    simple_maze = mr.get_simple_maze(maze_name)
    coord2label = mr.get_maze_coord2label(simple_maze)
    label2coord = mr.get_maze_label2coord(simple_maze)

    # node degree for all nodes
    node_degrees = {}
    for node in simple_maze.nodes:
        name = coord2label[node]
        node_degrees[name] = len(list(simple_maze.neighbors(node)))

    # all unique locations on the maze
    maze_locs = np.sort(np.array(list(node_degrees.keys())))
    maze_degrees = np.array([node_degrees[loc] for loc in maze_locs]) # corresponding node degrees
    degree4s = maze_locs[np.where(maze_degrees == 4)[0]] # all locations with node degree 4

    # load data
    for subject in subjects:
        maze_data[subject] = {"accs": [], "days": [], "locs": [], "n_data": []}

        print(f"\nrunning for subject {subject}")
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject],
            maze_names=[maze_name],
            days_on_maze="late",
            with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "navigation_routes_df"],
            must_have_data=True,
        )

        for session in sessions:
            # load data from session obj
            navigation_spikes_df = session.get_navigation_activity_df(
                type="spikes",
                cluster_kwargs={"single_units": True, "multi_units": False},
                with_routes=False,
            )

            # only consider navigation periods
            df = filt.filter_navigation_rates_df(
                        navigation_spikes_df, navigation_only=True, moving_only=True, exclude_time_at_goal=True
                    )
            # where is the animal at every point in time?
            all_locs = df.maze_position.simple.values
            # how many spikes at every point in time for every neuron?
            spike_counts = df["spike_count"].values
            
            # filter and normalize spikes
            count_thresh = 500
            spike_counts = spike_counts[:, spike_counts.sum(0) > count_thresh]
            spike_counts = (spike_counts - np.mean(spike_counts, axis = 0, keepdims = True)) / np.std(spike_counts, axis = 0, keepdims = True)
            print(spike_counts.shape)

            # now iterate through degree 4 nodes
            for loc4 in degree4s:
                # what are the neighbors of this node?
                neighbors = [coord2label[n] for n in simple_maze.neighbors(label2coord[loc4])]
                
                # df indices where the animal moves between states
                trans = np.where(all_locs[1:] != all_locs[:-1])[0]+1 # first index at each state
                loc_seq = all_locs[trans] # sequence of locations visited

                loc4_inds = np.where(loc_seq == loc4)[0] # indices in sequence where the animal visits degree 4 node
                loc4_inds = loc4_inds[(loc4_inds > future_delta) & (loc4_inds < len(loc_seq)-2)] # not early or last node visited (need past and future)
                currs = loc_seq[loc4_inds] # these should all be loc4 by definition
                
                prevs, nexts = loc_seq[loc4_inds-2], loc_seq[loc4_inds+2] # previous and subsequent location

                eff_prevs, eff_nexts = copy.deepcopy(prevs), copy.deepcopy(nexts) # no shuffle
                # optionally decorrelate behaviour and neural activity as a control
                if shuffle:
                    perm_inds = np.random.permutation(np.arange(len(currs)))
                    eff_prevs, eff_nexts = eff_prevs[perm_inds], eff_nexts[perm_inds] # same shuffle for future and past

                # array of neural activity for every past/future location pair
                neural_acts = {loc: {} for loc in neighbors}
                Nneuron = spike_counts.shape[-1]
                for prev in neighbors: # for each past
                    for next_ in neighbors: # for each future
                        
                        # at which traverses did I go from prev to next_? (indexing into 'loc_seq' and 'trans')
                        type_inds = loc4_inds[np.where((eff_prevs == prev) & (eff_nexts == next_))[0]]
                        if len(type_inds) == 0: # no data for this combination
                            neural_acts[next_][prev] = np.zeros((0, Nneuron)) # store empty array
                        else:
                            # what are the indices of the full df where I was future_delta towers before the junction?
                            inds_at_prev_loc = [np.arange(trans[ind-future_delta*2], trans[ind-future_delta*2+1])[-max_bins:] for ind in type_inds] # up to 10
                            # what is the average activity at this tower for each traversal?
                            neural_acts[next_][prev] = np.array([spike_counts[inds, :].mean(0) for inds in inds_at_prev_loc]) # traversals x N_neurons

                session_tower_data = [] # data for this session/tower
                for test_next in neighbors: # for each target location
                    for test_prev in [n for n in neighbors if n!=test_next]: # for each source location
                        
                        # prev locations used for training data
                        train_prev = [neigh for neigh in neighbors if neigh not in [test_prev, test_next]]

                        # train trials where we did go to test_next
                        train_trues = [neural_acts[test_next][prev] for prev in train_prev]
                        
                        # train trials where we went from a train loc to NOT-test_next
                        train_falses = [[neural_acts[next_][prev] for next_ in neighbors if (next_ not in [prev, test_next])] for prev in train_prev]
                        # concatenate across non-test future locations for each training source and across training sources
                        train_falses = [np.concatenate(arr) for arr in train_falses]

                        # test trials where we did go to test_next
                        test_true = neural_acts[test_next][test_prev]
                        # test trials where we went to NOT-test_next
                        test_false = np.concatenate([neural_acts[next_][test_prev] for next_ in neighbors if (next_ not in [test_prev, test_next])])

                        train_lens = [[len(arr) for arr in arrs] for arrs in [train_trues, train_falses]]
                        test_lens = [len(test_true), len(test_false)]

                        # make sure we have some training data and at least one test datapoint!

                        if np.amin(np.concatenate(train_lens)) >= 1 and np.amin(test_lens) >= 1:
                            
                            def sim(x1, x2):
                                #return np.sum(x1*x2)/np.sqrt((x1**2).sum() * (x2**2).sum())
                                return np.mean((x1-x2)**2)
                            
                            # pairwise sim between 'true' and 'true' trials (train x test)
                            sim_true_true = [np.array([[sim(te_tr, tr_tr) for te_tr in test_true] for tr_tr in train_true]) for train_true in train_trues] 
                            sim_true_false = [np.array([[sim(te_tr, tr_fa) for te_tr in test_true] for tr_fa in train_false]) for train_false in train_falses]
                            sim_false_false = [np.array([[sim(te_fa, tr_fa) for te_fa in test_false] for tr_fa in train_false]) for train_false in train_falses]
                            sim_false_true = [np.array([[sim(te_fa, tr_tr) for te_fa in test_false] for tr_tr in train_true]) for train_true in train_trues]
                            
                            sim_pos = np.mean([np.mean([np.mean(arr) for arr in sims]) for sims in [sim_true_true, sim_false_false]])
                            sim_neg = np.mean([np.mean([np.mean(arr) for arr in sims]) for sims in [sim_true_false, sim_false_true]])
                            
                            # append mean accuracy for this source/target combination
                            session_tower_data.append(100*(sim_neg - sim_pos)) # these are actually distances so predict sim_neg _bigger_ than sim_pos
                            
                            #print(test_next, test_prev, test_acc, "(", train_joint.shape, test_joint.shape, train_true.shape, train_false.shape, best_C,")")
                
                # compute mean and sem across source/targets for this session+tower
                if len(session_tower_data) >= 1:
                    m, s = np.mean(session_tower_data), np.std(session_tower_data)/np.sqrt(len(session_tower_data))
                    maze_data[subject]["accs"].append(m) # append mean accuracy across source/target locations
                    maze_data[subject]["days"].append(session.day_on_maze) # append mean across source/target locations
                    maze_data[subject]["locs"].append(loc4) # append mean across source/target locations
                    maze_data[subject]["n_data"].append(len(loc4_inds)) # append mean across source/target locations
                    print(f"day={session.day_on_maze},  loc={loc4},  m={round(m)},  s={round(s)}") # subtract baseline of 0.5
                else:
                    print(f"day={session.day_on_maze},  loc={loc4}, no data")

        all_accs = maze_data[subject]["accs"]
        print(f"mean for mouse {subject} {maze_name}: {np.mean(all_accs)}, sem = {np.std(all_accs)/np.sqrt(len(all_accs))}, t = {np.round((time.time() - t0)/60, 1)}")

    # plot result for this maze
    mouse_accs = [maze_data[subject]["accs"] for subject in subjects]
    # mean and sem across session_towers for each mouse
    m, s = [np.mean(accs) for accs in mouse_accs], [np.std(accs)/np.sqrt(len(accs)) for accs in mouse_accs]
    mtot, stot = np.mean(m), np.std(m)/np.sqrt(len(m))
    
    # print mean, sem, and one-sided t-test across animals
    print(f"{maze_name}: {mtot}, {stot}")
    print(ttest_1samp(m, 0, alternative = "greater"))


