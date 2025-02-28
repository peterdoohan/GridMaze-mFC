import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('/ceph/behrens/peter_doohan/goalNav_mFC_refactor/code')

import GridMaze
from GridMaze.analysis.core import get_sessions as gs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import argparse
from sklearn.cluster import KMeans
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--session', type=int, default=5)
    parser.add_argument('--subject', type=str, default='m4')
    parser.add_argument('--maze', type=int, default=1)
    
    args, _ = parser.parse_known_args()
    session_id = args.session
    subject = args.subject
    maze = args.maze
    
    print(f"subject {subject} maze {maze} session {session_id}")
    # select subject and maze and get data
    # ------------------------------------------------------------------------------------------------------------
    print("Getting sessions...")
    subject_IDs = [subject]
    maze_name = f"maze_{maze}"

    sessions = gs.get_maze_sessions(
        subject_IDs=subject_IDs,
        maze_names=[maze_name],
        days_on_maze="all",
        with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "navigation_routes_df"],
        must_have_data=True,
    )
    # get average firing rate/route probability for all good clusters by trials and get correlations between firing and routes
    # ------------------------------------------------------------------------------------------------------------
    print("Preprocessing session...")
    session = sessions[session_id]
    day_on_maze = session.day_on_maze
    print(f"day on maze = {day_on_maze}")
    mua_clusters = session.cluster_metrics.cluster_ID[session.cluster_metrics.KSLabel == 'mua']
    spk_df = session.navigation_spike_counts_df.merge(session.navigation_df[['time', 'trial_phase', 'trial_unique_ID', 'distance_to_goal', 'centroid_position', 'head_direction']], left_on='time', right_on='time')
    spk_df = spk_df.merge(session.navigation_routes_df[['route_probability']], left_on='time', right_index=True)
    spk_df = spk_df[spk_df.trial_phase == 'navigation']
    prefix = spk_df.spike_count.columns[0].split('.maze_cluster')[0]
    spk_df = spk_df.drop(columns=['trial_phase'], errors='ignore') # + [('spike_count', f'{prefix}.maze_cluster{i}') for i in mua_clusters
    # firing = spk_df.groupby('trial_unique_ID')['spike_count'].transform(lambda s: s.rolling(20, win_type='gaussian', center=True).mean(std=10))
    # spk_df['spike_count'] = firing['spike_count']
    spk_bins = (spk_df.time - spk_df.time.min()) // 0.5
    spk_df['spk_bin'] = spk_bins
    spk_df_binned = spk_df.groupby(['trial_unique_ID', 'spk_bin']).mean().reset_index()
    
    # ------------------------------------------------------------------------------------------------------------
    print("Getting umap representation...")
    firing = spk_df_binned.spike_count
    filter = firing.notna().all(axis=1)
    firing_filtered = firing[filter].to_numpy()
    
    # ------------------------------------------------------------------------------------------------------------
    # pca = PCA()
    scaler = StandardScaler()
    n_components = 3
    # umap_reducer = umap.UMAP(n_neighbors=80, n_components=n_components)
    
    firing_filtered_scaled = scaler.fit_transform(firing_filtered)
    # firing_filtered_pca = pca.fit_transform(firing_filtered_scaled)
    # firing_filtered_umap = umap_reducer.fit_transform(firing_filtered_pca)
    
    # ------------------------------------------------------------------------------------------------------------
    print("Clustering umap representation into 5 clusters...")
    n_clusters = 5
    clusterer = KMeans(n_clusters)
    # clusterer.fit(firing_filtered_umap)
    clusterer.fit(firing_filtered_scaled)    
    
    # ------------------------------------------------------------------------------------------------------------
    print("Adding umap representation and clustering to dataframe...")
    # firing_umap = np.ones((len(firing), n_components)) * np.nan
    # firing_umap[filter] = firing_filtered_umap
    # firing_umap = pd.DataFrame(firing_umap, columns=[('umap', f'umap_{i}') for i in range(n_components)])
    
    fr_labels = np.ones(len(spk_df_binned)) * np.nan
    fr_labels[filter] = clusterer.labels_
    spk_df_binned[('cluster', 'cluster_label')] = fr_labels
    
    # ------------------------------------------------------------------------------------------------------------
    cluster_fr_torch = F.one_hot(torch.tensor(spk_df_binned.cluster.cluster_label.to_numpy()).to(torch.int64))
    cluster_fr = pd.DataFrame(cluster_fr_torch, columns=[('cluster', f'cluster_{cluster}') for cluster in range(n_clusters)])
    # spk_df_binned = pd.concat([spk_df_binned, firing_umap, cluster_fr], axis=1)
    spk_df_binned = pd.concat([spk_df_binned, cluster_fr], axis=1)
    
    # ------------------------------------------------------------------------------------------------------------
    print("Looking at distances persisted by cluster...")
    distances = np.ones_like(cluster_fr_torch) * np.nan
    mean_distances, std_distances = np.ones(n_clusters) * np.nan, np.ones(n_clusters) * np.nan
    for i in range(n_clusters):
        a1, b1, c1 = cluster_fr_torch[:, i].unique_consecutive(return_inverse=True, return_counts=True)
        # a[(c <=2) & (a==0)] = 1 # remove gaps smaller than 2. 
        
        # new_seq = a[b]
        # a1, b1, c1 = new_seq.unique_consecutive(return_inverse=True, return_counts=True)
        end_index = torch.cumsum(c1, axis=0)-1
        begin_index = torch.concatenate([torch.zeros(1), torch.cumsum(c1, axis=0)])[:-1]
        distance_traversed = spk_df_binned.distance_to_goal.future.to_numpy()[begin_index.int()] - spk_df_binned.distance_to_goal.future.to_numpy()[end_index.int()]
        distance_traversed[spk_df_binned.trial_unique_ID.to_numpy()[begin_index.int()] != spk_df_binned.trial_unique_ID.to_numpy()[end_index.int()]] = np.nan
        distance_traversed[a1==0] = np.nan
        
        distances[:, i] = distance_traversed[b1]
        mean_distances[i] = np.nanmean(distance_traversed)
        std_distances[i] = np.nanstd(distance_traversed)
    
    print("Saving it all...")
    distances = pd.DataFrame(distances, columns=[('fr_distance_for_cluster', f'cluster_{i}') for i in range(n_clusters)])
    spk_df_binned = pd.concat([spk_df_binned, distances], axis=1)
    
    os.makedirs(f"/ceph/behrens/peter_doohan/goalNav_mFC_refactor/code/GridMaze/analysis/xiao_explore/processed_data_store/umaps/{subject}_maze{maze}", exist_ok=True)
    spk_df_binned.to_csv(f"/ceph/behrens/peter_doohan/goalNav_mFC_refactor/code/GridMaze/analysis/xiao_explore/processed_data_store/umaps/{subject}_maze{maze}/day{day_on_maze}.csv")
    
    np.save(f"/ceph/behrens/peter_doohan/goalNav_mFC_refactor/code/GridMaze/analysis/xiao_explore/processed_data_store/umaps/{subject}_maze{maze}/day{day_on_maze}_cluster_distance_mean.npy", mean_distances)
    np.save(f"/ceph/behrens/peter_doohan/goalNav_mFC_refactor/code/GridMaze/analysis/xiao_explore/processed_data_store/umaps/{subject}_maze{maze}/day{day_on_maze}_cluster_distance_std.npy", std_distances)
    return 


if __name__ == "__main__":
    main()
