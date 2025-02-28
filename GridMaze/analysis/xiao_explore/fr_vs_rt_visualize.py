import numpy as np
import torch
import matplotlib.pyplot as plt
import GridMaze
from GridMaze.analysis.core import get_sessions as gs


def main():
    
    # select subject and maze and get data
    # ------------------------------------------------------------------------------------------------------------
    subject_IDs = ['m3']
    maze_name = "maze_1"

    sessions = gs.get_maze_sessions(
        subject_IDs=subject_IDs,
        maze_names=[maze_name],
        days_on_maze="late",
        with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "navigation_routes_df"],
        must_have_data=True,
    )

    # get average firing rate/route probability for all good clusters by trials and get correlations between firing and routes
    # ------------------------------------------------------------------------------------------------------------
    session = sessions[8]
    mua_clusters = session.cluster_metrics.cluster_ID[session.cluster_metrics.KSLabel == 'mua']
    spk_df = session.navigation_spike_counts_df.merge(session.navigation_df[['time', 'trial_unique_ID', 'trial_phase']], left_on='time', right_on='time')
    spk_df = spk_df.merge(session.navigation_routes_df[['route_probability']], left_on='time', right_index=True)
    spk_df = spk_df[spk_df.trial_phase == 'navigation']
    prefix = spk_df.spike_count.columns[0].split('.maze_cluster')[0]
    spk_df = spk_df.drop(columns=['trial_phase'] + [('spike_count', f'{prefix}.maze_cluster{i}') for i in mua_clusters], errors='ignore')
    spktrial_df = spk_df.groupby("trial_unique_ID").mean()
    spktrial_df = spktrial_df.dropna()
    spk_rt_corrcoef = np.corrcoef(spktrial_df.spike_count.to_numpy().T, spktrial_df.route_probability.to_numpy().T)[:-11, -11:] #assume 11 routes including non route


    # get correlations between spk and route and sort. For highest to lowest, plot what the neuron and routes look like within a trial
    # ------------------------------------------------------------------------------------------------------------
    spk_rt_top_corr_arg = spk_rt_corrcoef.flatten().argsort()[::-1]
    spk_rt_top_corr_arg = (spk_rt_top_corr_arg // 11, spk_rt_top_corr_arg % 11)
    spk_columns = spktrial_df.spike_count.columns
    
    for spk, rt in zip(spk_rt_top_corr_arg[0][:1], spk_rt_top_corr_arg[1][:1]):
        if np.isnan(spk_rt_corrcoef[spk, rt]):
            continue
        if rt == 10:
            continue
        spk_cluster = spk_columns[spk]
        top_trials = spktrial_df.spike_count[spk_cluster].sort_values(ascending=False).index
        if len(top_trials) < 30:
            continue
        fig, ax = plt.subplots(30, 1, figsize=(10, 60))
        for i, trial in enumerate(top_trials[:30]):
            firing = spk_df[spk_df.trial_unique_ID == trial].spike_count[spk_cluster].rolling(window=50, win_type='gaussian', center=True).mean(std=50)
            route_repr = spk_df[spk_df.trial_unique_ID == trial].route_probability[f'route_{rt}'].rolling(window=50, win_type='gaussian', center=True).mean(std=50)
            ax[i].scatter(firing.index, firing, label=spk_cluster, c='black')
            ax[i].twinx().scatter(route_repr.index, route_repr, label=f'route_{rt}', c='orange')
            ax[i].set_title(trial)
            ax[i].legend(loc="upper right")
            ax[i].spines[['top', 'right']].set_visible(False)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        plt.savefig(f'/ceph/behrens/Xiao/misc/peter_maze/{rt}_{trial}_{spk_cluster}.png')
        plt.clf()  
    return 


if __name__ == "__main__":
    main()
