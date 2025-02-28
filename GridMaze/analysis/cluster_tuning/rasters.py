"""
Library for plotting raster plots relevant to GridMaze experiments.
"""
#%% Imports
import numpy as np
from matplotlib import pyplot as plt
from ..core import get_clusters as gc
#%% Global Variables
FRAME_RATE = 60
#%% Functions

def plot_session_trial_rasters(session, max_trial_length=45):
    """
    Plot a raster for each trial (rows), columns = time (unwarped)
    """
    navigation_df = session.navigation_df
    navigation_routes_df = session.navigation_routes_df.reset_index(drop=True)
    navigation_spike_counts_df = session.navigation_spike_counts_df.reset_index(drop=True)
    keep_clusters = gc.filter_clusters(session.cluster_metrics, session.session_info, return_unique_IDs=True)
    for cluster in keep_clusters:
        spikes = navigation_spike_counts_df.xs(cluster, level=1, axis=1)
        trial_spikes = [] # no spikes per frame
        trial_route_changes = [] # if frame is a route change
        trial_ends = [] # zeros then 1 at end of trial frame
        for trial in navigation_df.trial.dropna().unique():
            trial_mask = (navigation_df.trial == trial) & (navigation_df.trial_phase == "navigation")
            if trial_mask.sum() / FRAME_RATE > max_trial_length:
                continue
            trial_spikes.append(spikes[trial_mask].values.reshape(-1))
            trial_route_changes.append(navigation_routes_df[trial_mask].route_change.values)
            end_trial =  np.zeros(trial_mask.sum()+1) 
            end_trial[-1] = 1
            trial_ends.append(end_trial)
        # pad all vectors to the same length
        max_length = max([len(arr) for arr in trial_ends])
        
        def pad_array(arr_list):
            pad_array = np.array([np.pad(arr, (0, max_length - len(arr)), constant_values=0) for arr in arr_list])
            pad_array = np.where(np.isnan(pad_array), 0, pad_array)
            return pad_array

        trial_spikes = pad_array(trial_spikes)
        trial_route_changes = pad_array(trial_route_changes)
        trial_ends = pad_array(trial_ends)
        plot_cluster_trial_raster(trial_spikes, trial_route_changes, trial_ends)


def plot_cluster_trial_raster(trial_spikes, trial_route_changes, trial_ends, ax=None):
    if ax is None:
        f, ax = plt.subplots(1,1, figsize=(5,8))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Trial")
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    times = np.arange(trial_spikes.shape[1]) / FRAME_RATE
    ax.set_xlim(-0.5, times[-1]+0.5)
    for i, (spikes, route_changes, end_trial) in enumerate(zip(trial_spikes, trial_route_changes, trial_ends)):
        for j, spike in enumerate(spikes):
            if spike:
                ax.plot(times[j], i, 'k|', markersize=5)
        for j, route_change in enumerate(route_changes):
            if route_change:
                ax.plot(times[j], i, color='green', marker='|', markersize=5, alpha=0.5)
        for j, end in enumerate(end_trial):
            if end:
                ax.plot(times[j], i, 'r|', markersize=5)
    return


