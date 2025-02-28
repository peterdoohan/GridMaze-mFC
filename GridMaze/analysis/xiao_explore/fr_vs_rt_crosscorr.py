import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import ccf

import GridMaze
from GridMaze.analysis.core import get_sessions as gs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def corrcoef_lagged(x, y, min_n=0, max_lag=2500):
    assert len(x) == len(y)
    max_lag = min(len(x), max_lag)
    lag = max_lag - min_n
    lagged_corrcoef = [np.corrcoef(x, y)[0, 1]]
    lagged_corrcoef.extend([np.corrcoef(x[:-i], y[i:])[0, 1] for i in range(1, lag)])
    return np.array(lagged_corrcoef)


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
    
    firing = spk_df.groupby('trial_unique_ID')['spike_count'].transform(lambda s: s.rolling(120, win_type='gaussian', center=True).mean(std=60))
    spk_df['spike_count'] = firing['spike_count']
    # route_repr = spk_df.groupby('trial_unique_ID').route_probability.transform(lambda s: s.rolling(120, win_type='gaussian', center=True).mean(std=60))
    
    filter = firing.notna().all(axis=1)
    firing_filtered = firing[filter].to_numpy()
    pca = PCA(n_components=5)
    scaler = StandardScaler()
    
    firing_filtered_scaled = scaler.fit_transform(firing_filtered)
    firing_filtered_pca = pca.fit_transform(firing_filtered_scaled)
    firing_pca = np.ones((len(firing), 5)) * np.nan
    firing_pca[filter] = firing_filtered_pca
    firing_pca = pd.DataFrame(firing_pca, columns=[('PC', f'PC_{i}') for i in range(5)], index=firing.index)    
    
    filter = spk_df['route_probability'].notna().all(axis=1)
    route_filtered = spk_df['route_probability'][filter].to_numpy()
    route_filtered_scaled = scaler.fit_transform(route_filtered)
    route_filtered_pca = pca.fit_transform(route_filtered_scaled)
    route_pca = np.ones((len(firing), 5)) * np.nan
    route_pca[filter] = route_filtered_pca
    route_pca = pd.DataFrame(route_pca, columns=[('route_PC', f'route_PC_{i}') for i in range(5)], index=firing.index)

    spk_df = pd.concat([spk_df, firing_pca, route_pca], axis=1)
    
    # for spk_c in spktrial_df.spike_count.columns:
    for pc in range(5):
        cross_corr_tot = []
        for rt in range(5):
            x = spk_df.route_PC[f'route_PC_{rt}']
            y = spk_df.PC[f'PC_{pc}']
            # y = spk_df.spike_count[spk_c]
            filter = ~x.isna() & ~y.isna()
            
            spk_df_filtered = spk_df[filter].sort_index()
            if len(spk_df_filtered) == 0:
                continue
            # cross_corrs = spk_df_filtered.groupby('trial_unique_ID').apply(
            #     lambda s: ccf(s[('spike_count', spk_c)], s[('route_probability', f'route_{rt}')])[:3600], include_groups=False).to_list()
            # cross_corrs = spk_df_filtered.groupby('trial_unique_ID').apply(
            #     lambda s: corrcoef_lagged(s[('spike_count', spk_c)].to_numpy(), s[('route_probability', f'route_{rt}')].to_numpy(), min_n=50), include_groups=False
            # ).to_list()
            cross_corrs = spk_df_filtered.groupby('trial_unique_ID').apply(
                lambda s: corrcoef_lagged(s[('PC', f'PC_{pc}')].to_numpy(), s[('route_PC', f'route_PC_{rt}')].to_numpy(), min_n=50), include_groups=False
            ).to_list()
            max_len = max(len(i) for i in cross_corrs)
            cross_corrs_padded = [np.pad(i, (0, max_len-len(i)), constant_values=np.nan) for i in cross_corrs]
            cross_corrs_stacked = np.stack(cross_corrs_padded)
            
            cross_corrs_mean = np.nanmean(cross_corrs_stacked, axis=0)
            cross_corrs_std = np.nanstd(cross_corrs_stacked, axis=0) 
            cross_corrs_ste = cross_corrs_std / ((~np.isnan(cross_corrs_stacked)).sum(axis=0) ** 0.5)
            time = np.arange(len(cross_corrs_mean))
            # cross_corr = ccf(x[filter], y[filter])[:3600]
            line, = plt.plot(time, cross_corrs_mean, label=f'{rt}', alpha=0.3)
            plt.fill_between(time, cross_corrs_mean-cross_corrs_ste, cross_corrs_mean+cross_corrs_ste, alpha=0.1, color=line.get_color())
            cross_corr_tot.append(cross_corrs_mean)
        
        max_len = max(len(i) for i in cross_corr_tot)
        cross_corr_tot = np.stack([np.pad(i, (0, max_len-len(i)), constant_values=np.nan) for i in cross_corr_tot])
        plt.plot(cross_corr_tot.mean(axis=0), label='mean', color='black')
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
        plt.savefig(f'/ceph/behrens/Xiao/misc/PC_{pc}.png')
        plt.clf()
    return 


if __name__ == "__main__":
    main()
