# %% Imports
import json
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_1samp
from scipy.stats import linregress
import os
os.chdir("/ceph/behrens/peter_doohan/goalNav_mFC/experiment/code")

from GridMaze.maze import representations as mr
from GridMaze.analysis.core import folds
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import convert
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.paths import EXPERIMENT_INFO_PATH
from GridMaze.analysis.neGLM import load_model_sets as lms
from GridMaze.analysis.neGLM import variance_explained as ve
from GridMaze.analysis.place_direction import future_decoding as fd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr

#%%

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

FRAME_RATE = 60

feature_tuned_df = ve.get_feature_tuned_df(
    lms.load_model_set_cv_scores("variance_explained_multiunit"),
    reduced_models=["remove_distance_to_goal", "remove_place_direction"],
)

distance_tuned = (
    feature_tuned_df[(feature_tuned_df.distance_to_goal & ~feature_tuned_df.place_direction)]
    .index.get_level_values(1)
    .values
)
place_tuned = (
    feature_tuned_df[(~feature_tuned_df.distance_to_goal & feature_tuned_df.place_direction)]
    .index.get_level_values(1)
    .values
)


#%%

resolution=0.2
mindist, maxdist = 0.5, 7.5 # minimum and maximum distance from goal to include data
mincount = 15.5 # number of unique timebins where a location is visited to include it
trial_counts = 2.5 # number of data points in a trial to include it
max_steps_to_goal = 16 # maximum number of true steps to goal to include data
min_rate = 0.5 # minimum average firing rate to include neuron in analysis (Hz)
permute = False
place_offset = 2

all_dat = []
for tower_or_bridge in ["tower", "bridge"]:

    all_mice = []
    all_mouse_accs = []
    mouse_neuron_counts = []
    mouse_locdist_counts = []
    mouse_true_dists = []
    mouse_dirs = []

    for subject in SUBJECT_IDS:

        print(f"\n\nsubject={subject}")
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject],
            maze_names=["maze_1"],
            days_on_maze="late",
            with_data=[
                "navigation_df",
                "cluster_metrics",
                "trials_df",
                "navigation_spike_counts_df",
            ],
            must_have_data=True,
        )


        # %%

        all_norms = []
        all_accs = []
        neuron_counts = []
        locdist_counts = []
        all_true_dists = []
        all_dirs = []

        for session in sessions:

            navigation_df = session.navigation_df.copy()
            spike_counts_df = session.navigation_spike_counts_df  # [frames, clusters]
            spike_counts_df.reset_index(inplace=True, drop=True)
            spike_counts_df.columns = pd.MultiIndex.from_tuples([(*c, "") for c in spike_counts_df.columns])

            # filter clusters
            keep_clusters = gc.filter_clusters(
                session.cluster_metrics,
                session.session_info,
                return_unique_IDs=True,
                single_units=True,
                multi_units=True,
            )
            spike_counts_df = spike_counts_df[
                spike_counts_df.columns[spike_counts_df.columns.get_level_values(1).isin(keep_clusters)]
            ]

            metric=("distance_to_goal", "geodesic")
            ds_nav_df, ds_spike_counts_df = ds.downsample_nav_spikes_data(
                navigation_df,
                spike_counts_df,
                resolution=resolution,
                distance_metrics=[("steps_to_goal", "future"), metric],
            )

            ds_nav_df.columns = pd.MultiIndex.from_tuples([(*c, "") for c in ds_nav_df.columns])
            input_df = pd.concat([ds_nav_df, ds_spike_counts_df], axis=1).reset_index(drop=True)

            # add future, past state (place) information
            input_df[("place_direction", "", "")] = input_df.maze_position.simple + "_" + input_df.cardinal_movement_direction
            offset_df = fd.get_past_and_future_states(
                input_df, state_type="place", past_offset=place_offset, future_offset=place_offset
            )
            offset_df.columns = pd.MultiIndex.from_tuples([(*col, "") for col in offset_df.columns])
            input_df = pd.concat([input_df, offset_df], axis=1)
            input_df = input_df.sort_index(axis=1)  # sort columns for easier indexing later

            # filter data
            input_df = filt.filter_navigation_rates_df(
                input_df,
                navigation_only=True,
                moving_only=True,
                exclude_time_at_goal=True,
                max_steps_to_goal=max_steps_to_goal,
            )

            input_df = input_df.droplevel(2, axis=1)

            # %% now train decoders

            maze = session.simple_maze()
            nodes = {node[1]["label"]:node[0] for node in maze.nodes.items()}
            base_dist = lambda loc, goal: nx.shortest_path_length(maze, nodes[loc], nodes[goal], weight=None)
            
            locs = np.array(input_df["future"][0])
            if tower_or_bridge == "tower":
                towers = np.array([len(loc) == 2 for loc in locs])
                train_df = input_df.loc[towers, :]
                dist_func = lambda loc, goal: base_dist(loc, goal) if (len(str(loc)) == 2) else np.nan
                bias = 0.0
            else:
                bridges = np.array([len(loc) == 5 for loc in locs])
                train_df = input_df.loc[bridges, :]
                dist_func = lambda loc, goal: np.amin([base_dist(adj, goal) for adj in loc.split("-")]) if (len(str(loc)) == 5) else np.nan
                bias = 0.5
                
            locs = np.array(train_df["future"][0])
            next = np.array(train_df["future"][2])
            prev = np.array(train_df["past"][2])
            trial = np.array(train_df["trial"]).astype(int)
            new_true_dists = np.array(train_df["distance_to_goal"]["geodesic"])
            
            spikes = np.array(train_df["spike_count"])
            units = np.array(train_df["spike_count"].columns)
            keep_units = np.where(spikes.mean(0)/resolution > min_rate)[0]
            spikes, units = np.sqrt(spikes[:, keep_units]), units[keep_units]

            dist_tuned = np.array([unit in distance_tuned for unit in units])
            loc_tuned = np.array([unit in place_tuned for unit in units])
            neuron_counts.append([loc_tuned.sum(), dist_tuned.sum()])

            goals = np.array(train_df["goal"])


            
            dists = np.array([dist_func(locs[i], goals[i]) for i in range(len(goals))]).astype(float)
            dists_next = np.array([dist_func(next[i], goals[i]) for i in range(len(goals))]).astype(float)
            dists_prev = np.array([dist_func(prev[i], goals[i]) for i in range(len(goals))]).astype(float)
            dirs = np.sign(dists - dists_next) # am I going towards the goal (+1) or away (-1)?
            print(np.unique((dists - dists_next), return_counts=True), np.nanmean(dirs))

            # %%

            cond_dist = (dists < maxdist) & (dists > 0.5)
            cond_dist_next = (dists_next < maxdist) & (dists_next > mindist)
            cond_dist_prev = (dists_prev < maxdist) & (dists_prev > mindist)
            unique_locs, counts = np.unique(locs, return_counts=True)
            keep_locs = unique_locs[counts > mincount]
            cond_loc = np.array([loc in keep_locs for loc in locs])
            cond_next = np.array([n in keep_locs for n in next])
            cond_prev = np.array([p in keep_locs for p in prev])

            cond_all = (cond_dist & cond_dist_prev & cond_dist_next) & (cond_loc & cond_next & cond_prev)
            
            # if permute:
            #     spikes[cond_all, :] = np.random.permutation(spikes[cond_all, :])

            Xloc, yloc, new_trials_loc = spikes[cond_loc, :][:, loc_tuned], locs[cond_loc], trial[cond_loc]
            Xdist, ydist, new_trials_dist = spikes[cond_dist, :][:, dist_tuned], dists[cond_dist], trial[cond_dist]
                
            ulocs, udists = np.unique(yloc), np.unique(ydist)
            yloc_1h = (yloc[:, None] == ulocs[None, :]).astype(float).argmax(-1)
            ydist_1h = (ydist[:, None] == udists[None, :]).astype(float).argmax(-1)

            print(Xloc.shape, yloc.shape, new_trials_loc.shape)
            print(Xdist.shape, ydist.shape, new_trials_dist.shape)

            accs_loc, accs_dist = [], []
            trial_ids, trial_counts = np.unique(trial[cond_loc & cond_dist], return_counts=True)

            all_loc_dist = []
            session_true_dists = []
            session_dirs = []
            for test_trial in trial_ids[trial_counts >= 2.5]:
                train_loc = new_trials_loc != test_trial
                test_loc = ~train_loc
                Xtrain_loc, ytrain_loc = Xloc[train_loc], yloc_1h[train_loc]
                Xtest_loc, ytest_loc = Xloc[test_loc], yloc_1h[test_loc]
                clf_loc = LogisticRegression(random_state=0, class_weight = "balanced", C = 1e-1, max_iter=500).fit(Xtrain_loc, ytrain_loc)
                accs_loc.append(clf_loc.score(Xtest_loc, ytest_loc))
                
                train_dist = new_trials_dist != test_trial
                test_dist = ~train_dist
                Xtrain_dist, ytrain_dist = Xdist[train_dist], ydist_1h[train_dist]
                Xtest_dist, ytest_dist = Xdist[test_dist], ydist_1h[test_dist]
                clf_dist = LogisticRegression(random_state=0, class_weight = "balanced", C = 1e-1, max_iter=500).fit(Xtrain_dist, ytrain_dist)
                accs_dist.append(clf_dist.score(Xtest_dist, ytest_dist))
                
                #print(test_trial, accs[-1])
                
                # also try to look at errors if possible
                for index in np.where( (trial == test_trial) & cond_all)[0]: # for each test sample
                    loc_probs = clf_loc.predict_proba(spikes[index:index+1, :][:, loc_tuned])
                    preds_loc = loc_probs[0, (np.array([prev[index], locs[index], next[index]])[:, None] == ulocs[None, :]).astype(float).argmax(-1)]
                    if permute and np.random.rand() < 0.5:
                        preds_loc = preds_loc[np.array([2,1,0])]
                        

                    dist_probs = clf_dist.predict_proba(spikes[index:index+1, :][:, dist_tuned])
                    preds_dist = dist_probs[0, (np.array([dists_prev[index], dists[index], dists_next[index]])[:, None] == udists[None, :]).astype(float).argmax(-1)]

                    all_loc_dist.append([preds_loc, preds_dist])
                    
                    # compare to the true displacement!!
                    session_true_dists.append((dists[index]+bias)*0.18 - new_true_dists[index])
                    session_dirs.append(dirs[index])

            for accs in [accs_loc, accs_dist]:
                print(np.mean(accs), np.std(accs)/np.sqrt(len(accs)), 1/np.amax(yloc_1h), 1/np.amax(ydist_1h))


            # %%


            all_loc_dist = np.array(all_loc_dist)
            norm = all_loc_dist / np.sum(all_loc_dist, axis = -1, keepdims = True)

            delta = norm[..., 2] - norm[..., 0]

            #%%

            print(pearsonr(delta[:, 0], delta[:, 1]), neuron_counts[-1])
            all_norms.append(norm)
            all_accs.append([accs_loc, accs_dist])
            all_true_dists.append(session_true_dists)
            all_dirs.append(session_dirs)
            locdist_counts.append([np.amax(yloc_1h), np.amax(ydist_1h)])
            
        all_mice.append(all_norms)
        all_mouse_accs.append(all_accs)
        mouse_neuron_counts.append(neuron_counts)
        mouse_locdist_counts.append(locdist_counts)
        mouse_true_dists.append(all_true_dists)
        mouse_dirs.append(all_dirs)
        
    all_dat.append([all_mice, all_mouse_accs, mouse_neuron_counts, mouse_locdist_counts, mouse_true_dists, mouse_dirs])
    
    
# %%
plt.figure()
tb_corrs = []
for itower in range(2):
    all_mice, all_mouse_accs, mouse_neuron_counts, mouse_locdist_counts, mouse_true_dists, mouse_dirs = all_dat[itower]

    all_corrs = []
    all_acc_corrs = []
    all_accs = []
    all_counts = []
    big_corrs = []
    big_acc_corrs = []
    big_acc_corrected_corrs = []
    big_acc_ctrl_corrs = []
    for idata, data in enumerate(all_mice):
        print(f"\n\nMouse", SUBJECT_IDS[idata])
        mouse_corrs, mouse_accs, mouse_counts, mouse_acc_corrs = [], [], [], []
        goods = []
        for inorm, norm in enumerate(data):
            delta = norm[..., 2] - norm[..., 0]
            
            delta = (norm[..., 2] - norm[..., 0]) / (norm[..., 2] + norm[..., 0])
            corr = pearsonr(delta[:, 0], delta[:, 1])[0]
            
            session_accs = np.array(all_mouse_accs[idata][inorm])
            accs = session_accs.mean(-1)
            counts = mouse_neuron_counts[idata][inorm]
            mouse_accs.append(accs)
            mouse_counts.append(counts)
            mouse_corrs.append(corr)
            print(corr, norm.shape[0], mouse_locdist_counts[idata][inorm],
                accs)
            
            sems = session_accs.std(-1) / np.sqrt(session_accs.shape[-1])
            chance = 1/np.array(mouse_locdist_counts[idata][inorm])

            if accs[0] > 2*chance[0]:
                mouse_acc_corrs.append(corr)
                goods.append(True)
            else:
                print("nopes,", accs, sems, chance)
                goods.append(False)
        
        all_corrs.append(mouse_corrs)
        all_accs.append(mouse_accs)
        all_counts.append(mouse_counts)
        all_acc_corrs.append(mouse_acc_corrs)
        
        for idat, dat in enumerate([data, [data[i] for i in range(len(data)) if goods[i]]]):
            bignorm = np.concatenate(dat, axis = 0)
            bigdelta = bignorm[..., 2] - bignorm[..., 0]
            bigdelta = (bignorm[..., 2] - bignorm[..., 0]) / (bignorm[..., 2] + bignorm[..., 0])
            big_corr = pearsonr(bigdelta[:, 0], bigdelta[:, 1])[0]
            #big_corr = spearmanr(bigdelta[:, 0], bigdelta[:, 1])[0]
            [big_corrs, big_acc_corrs][idat].append(big_corr)

            if idat == 1:
                trues = np.concatenate([mouse_true_dists[idata][i] for i in range(len(data)) if goods[i]]) # true displacements
                dirs = np.concatenate([mouse_dirs[idata][i] for i in range(len(data)) if goods[i]]) # directions of travel
                #big_acc_ctrl_corrs.append(pearsonr(bigdelta[:, 0], trues)[0])
                big_acc_ctrl_corrs.append(pearsonr(bigdelta[:, 0], dirs)[0])
                
                regs_true = [linregress(trues, bigdelta[:, i]) for i in range(2)]
                debias_true = [bigdelta[:, i] - (regs_true[i].intercept + regs_true[i].slope*trues) for i in range(2)]
                
                regs_dir = [linregress(dirs, debias_true[i]) for i in range(2)]
                debias_dir = [debias_true[i] - (regs_dir[i].intercept + regs_dir[i].slope*dirs) for i in range(2)]

                big_acc_corrected_corrs.append(pearsonr(debias_dir[0], debias_dir[1])[0])
        
        print(np.mean(mouse_corrs), big_corr, big_acc_corrected_corrs[-1])
        
    all_corrs = np.array(all_corrs)
    all_accs = np.array(all_accs)
    all_counts = np.array(all_counts)

    print([pearsonr(all_corrs.flatten(), all_counts[..., i].flatten()) for i in range(2)])
    print([pearsonr(all_corrs.flatten(), all_accs[..., i].flatten()) for i in range(2)])
    print(pearsonr(all_corrs.flatten(), all_accs.mean(-1).flatten()))

    
    for i in range(len(all_corrs)):
        plt.scatter(all_accs[i, :, 0], all_corrs[i], label = (f"{SUBJECT_IDS[i]}" if itower == 0 else None),
                    marker = ["o", "x"][itower], s = 100, color = plt.get_cmap("tab10")(i))
        print(ttest_1samp(all_corrs[i], 0, alternative='greater'))

    tb_corrs.append(np.array([all_corrs.mean(-1), [np.mean(c) for c in all_acc_corrs], big_corrs, big_acc_corrs, big_acc_corrected_corrs])) #, big_acc_ctrl_corrs])

plt.axhline(0, color = "k")
plt.ylabel("correlation")
plt.xlabel("location decoding accuracy")
plt.legend(ncol = 6, loc = "upper center", bbox_to_anchor = (0.5, 1.1), handletextpad = 0.1)
plt.show()

#%%

corrs = np.array(tb_corrs).mean(0)
#corrs = np.array(tb_corrs)[1]

m, s = corrs.mean(-1), corrs.std(-1) / np.sqrt(corrs.shape[-1])
xs = np.arange(len(m))
plt.figure(figsize = (4,3))
plt.bar(xs, m, yerr=s)
for x in xs:
    corr = corrs[x]
    plt.scatter(x+np.linspace(-0.1, 0.1, len(corr)), corr, color = "k", marker = ".", s = 50)
    print(ttest_1samp(corr, 0, alternative='greater'), np.round(corr, 3))
plt.xticks(xs, ["avg", "avg-good", "cat", "cat-good", "cat-good-debias"], rotation = 45, ha = "right")
plt.show()

# %%
