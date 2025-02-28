"""
Library for analyses related to neural tuning of time-lagged representations of where you will be in the future
@peterdoohan
"""

# %% Imports
import numpy as np
import pandas as pd
from ..core import get_clusters as gc
import seaborn as sns

import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter


from ...maze import plotting as mp

# %% Global Variables

FRAME_RATE = 60

#%%

def get_population_average_lagged_correlations(session, plot=False):
    keep_clusters = gc.filter_clusters(session.cluster_metrics, session.session_info, return_unique_IDs=True)
    lag_corrs = []
    for cluster in keep_clusters:
        lagged_corrs_df = get_lagged_rates_correlation(session, cluster, n=2)
        lag_corrs.append(lagged_corrs_df.mean(axis=1))
    lag_corrs_df = pd.concat(lag_corrs, axis=1)
    if plot:
        plot_symetrical_lagged_correlations(lag_corrs_df)
    return lag_corrs_df

def plot_session_lagged_rates_correlations(session, smooth_SD=1, lag_range=(-15, 15), n=10):
    simple_maze = session.simple_maze()
    keep_clusters = gc.filter_clusters(session.cluster_metrics, session.session_info, return_unique_IDs=True)
    lag_corrs = []
    for cluster in keep_clusters:
        lagged_corrs_df = get_lagged_rates_correlation(session, cluster, smooth_SD, lag_range, n)
        lag_corrs.append(lagged_corrs_df.mean(axis=1))
        place_direction_lagged_rates_df = get_place_direction_lagged_rates_df(session, cluster, smooth_SD, lag_range)
        lag_max_corr = get_max_lagged_correlation(lagged_corrs_df)
        now_tuning = place_direction_lagged_rates_df[0]
        vmax = now_tuning.max()
        if lag_max_corr is not None:
            future_tuning = place_direction_lagged_rates_df[lag_max_corr]
            vmax = max(vmax, future_tuning.max())
        f, ax = plt.subplots(1,3, figsize=(15,4))
        plot_lagged_correlations(lagged_corrs_df, ax=ax[0])
        ax[0].set_title(cluster)
        mp.plot_directed_heatmap(simple_maze, now_tuning, ax=ax[1], fixed_vmax=vmax,
                                colormap="heat", title="Now")
        if lag_max_corr is not None:
            mp.plot_directed_heatmap(simple_maze, future_tuning, ax=ax[2], fixed_vmax=vmax,
                                    colormap="heat", title="Past/Future", value_label="Firing Rate (Hz)")
        else:
            ax[2].axis('off')
            

def get_place_direction_lagged_rates_df(session, cluster_unique_ID, smooth_SD=1, lag_range=(-15, 15)):
    navigation_df = session.navigation_df
    navigation_rates_df = session.navigation_spike_rates_df
    cluster_rates = navigation_rates_df.xs(cluster_unique_ID, level=1, axis=1)
    smoothed_rates = gaussian_filter(cluster_rates, smooth_SD * FRAME_RATE)
    smoothed_rates = pd.Series(smoothed_rates.reshape(-1), index=navigation_df.index)
    all_trials = navigation_df.trial.dropna().unique()
    navigation_lagged_rates_df = get_navigation_lagged_rates_df(navigation_df, smoothed_rates, all_trials, lag_range)
    place_direction_lagged_rates = navigation_lagged_rates_df.groupby(
                [("maze_position", "simple"), ("cardinal_movement_direction", "")]).lagged_firing_rate.mean().lagged_firing_rate
    return place_direction_lagged_rates


def get_lagged_rates_correlation(session, cluster_unique_ID, smooth_SD=1, lag_range=(-15, 15), n=20):
    navigation_df = session.navigation_df
    navigation_rates_df = session.navigation_spike_rates_df
    cluster_rates = navigation_rates_df.xs(cluster_unique_ID, level=1, axis=1)
    smoothed_rates = gaussian_filter(cluster_rates, smooth_SD * FRAME_RATE)
    smoothed_rates = pd.Series(smoothed_rates.reshape(-1), index=navigation_df.index)
    trial_splits = get_trial_splits(session, split=0.5, n_splits=n)
    trial_split_correlations = []
    for trial_split in trial_splits:
        place_direction_splits = []
        for trials in trial_split:
            navigation_lagged_rates_df = get_navigation_lagged_rates_df(navigation_df, smoothed_rates, trials, lag_range)
            place_direction_lagged_rates = navigation_lagged_rates_df.groupby(
                [("maze_position", "simple"), ("cardinal_movement_direction", "")]).lagged_firing_rate.mean().lagged_firing_rate
            place_direction_splits.append(place_direction_lagged_rates)
        df_1, df_2 = place_direction_splits
        trial_split_correlations.append(df_1.corrwith(df_2))
    lagged_corrs_df = pd.concat(trial_split_correlations, axis=1)
    return lagged_corrs_df

def plot_lagged_correlations(lagged_corrs_df, ax=None):
    """"""
    long_df = _melt_corrs_df(lagged_corrs_df)
    if ax is None:
        f, ax = plt.subplots(1,1, figsize=(3,3))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Time lag (s)")
    ax.set_ylabel("Correlation")
    ax.axvline(0, color='black', linestyle='--', lw=0.5, alpha=0.3)
    ax.axhline(0, color='black', linestyle='--', lw=0.5, alpha=0.3)
    sns.lineplot(data=long_df, x='time_lag', y='value', ax=ax, errorbar='sd')
    lag_max_corr = get_max_lagged_correlation(lagged_corrs_df)
    av_corrs = lagged_corrs_df.mean(axis=1)
    ax.scatter(0, av_corrs.loc[0], c='black', zorder=3, s=100)
    if lag_max_corr is not None:
        ax.scatter(lag_max_corr, av_corrs.loc[lag_max_corr], c='red', zorder=3, s=100)


def plot_symetrical_lagged_correlations(lagged_corrs_df, ax=None):
    if ax is None:
        f, ax = plt.subplots(1,1, figsize=(3,3))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Time lag (s)")
    ax.set_ylabel("Correlation")
    ax.axvline(0, color='black', linestyle='--', lw=0.5, alpha=0.3)
    ax.axhline(0, color='black', linestyle='--', lw=0.5, alpha=0.3)
    long_df = _melt_corrs_df(lagged_corrs_df)
    pos_lags_df = long_df[long_df.time_lag >= 0]
    neg_lags_df = long_df[long_df.time_lag <= 0]
    neg_lags_df.loc[:,'time_lag'] = neg_lags_df['time_lag'].abs()
    sns.lineplot(data=pos_lags_df, x='time_lag', y='value', ax=ax, color='purple', label='Future')
    sns.lineplot(data=neg_lags_df, x='time_lag', y='value', ax=ax, color='grey', label='Past')
    ax.legend()


def _melt_corrs_df(lagged_corrs_df):
    corrs_df = lagged_corrs_df.reset_index().rename(columns={'index': 'time_lag'})
    long_df = pd.melt(corrs_df, id_vars=['time_lag'], var_name='measurement', value_name='value')
    return long_df


def get_navigation_lagged_rates_df(navigation_df, smoothed_rates, trials, lag_range=(-15, 15)):
    """"""
    trial_rates_dfs = []
    for trial in trials:
        trial_mask = (navigation_df.trial == trial) & (navigation_df.trial_phase == "navigation")
        trial_df = navigation_df[trial_mask]
        trial_smoothed_rates = smoothed_rates[trial_mask]
        trial_lagged_rates_df = pd.DataFrame(index=trial_df.index)
        for lag in range(lag_range[0], lag_range[1] + 1):
            trial_lagged_rates_df[lag] = trial_smoothed_rates.shift(-lag*FRAME_RATE)
        trial_lagged_rates_df.columns = pd.MultiIndex.from_product([["lagged_firing_rate"], trial_lagged_rates_df.columns])
        trial_rates_df = pd.concat([trial_df, trial_lagged_rates_df], axis=1)
        trial_rates_dfs.append(trial_rates_df)
    navigation_lagged_rates_df = pd.concat(trial_rates_dfs).reset_index(drop=True)
    return  navigation_lagged_rates_df


def get_max_lagged_correlation(lagged_corrs_df):
    """"""
    av_corrs = lagged_corrs_df.mean(axis=1)
    lag_max_corr = av_corrs[av_corrs.index != 0].idxmax()
    if av_corrs.loc[0] > av_corrs[lag_max_corr]:
        return None
    else:
        return lag_max_corr


def get_trial_splits(session, goal_stratified=True, split=0.5, n_splits=10):
    """"""
    trials_df = session.trials_df
    goal2trials = trials_df.groupby(('goal','')).trial.apply(lambda x: np.unique(x))
    goal2trials_df = goal2trials.apply(pd.Series)
    #shuffle
    goal2trials_df = goal2trials_df.apply(_shuffle, axis=1)
    max_trials_per_goal = goal2trials_df.count(axis=1).max()
    if goal_stratified:
        n_split = int(max_trials_per_goal * split)
        splits = []
        for i in range(n_splits):
            goal2trials_df = goal2trials_df.apply(_shuffle, axis=1)
            split_1 = goal2trials_df.iloc[:, :n_split].unstack().to_numpy()
            split_2 = goal2trials_df.iloc[:, n_split:].unstack().to_numpy()
            splits.append((split_1, split_2))
    return splits

def _shuffle(x):
    return pd.Series(np.random.choice(x, size=len(x), replace=False))


