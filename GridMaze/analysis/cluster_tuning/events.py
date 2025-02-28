"""
Library for plotting neural repsonses relative to trial events
"""
#%% Imports
import json
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from ..core import get_clusters as gc
from ...maze import plotting as mp

#%% Global Varibales
from ...paths import ANALYSIS_INFO_PATH

with open(ANALYSIS_INFO_PATH / "intra_trial_interval_times.json", "r") as f:
    INTRA_TRIAL_INTERVAL_TIMES = json.load(f)



#%% Trial Aligned Plotting

def plot_session_trial_aligned_rates(session, goal_stratified=False):
    """ """
    trial_aligned_rates_df = session.trial_aligned_rates_df
    keep_clusters = gc.filter_clusters(session.cluster_metrics, #plot only single units
                                       session.session_info, 
                                       return_unique_IDs=True, 
                                       single_units=True)
    for cluster_unique_ID in keep_clusters:
        cluster_trial_aligned_rates = trial_aligned_rates_df[trial_aligned_rates_df.cluster_unique_ID == cluster_unique_ID]
        plot_trial_aligned_rates(cluster_trial_aligned_rates, goal_stratified=goal_stratified)
    return

def plot_trial_aligned_rates(trial_aligned_rates, goal_stratified=False, smooth_SD=10, ax=None, color="black"):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(6, 3), clear=True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Firing Rate (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(-5, INTRA_TRIAL_INTERVAL_TIMES["ITI_end"]+0.5)
    ax.set_xticks(list(INTRA_TRIAL_INTERVAL_TIMES.values()))
    ax.set_xticklabels(["cue", 'reward', 'erc', 'end'])
    for time in INTRA_TRIAL_INTERVAL_TIMES.values():
        ax.axvline(time, color="black", linestyle="--", alpha=0.2)
    if not goal_stratified:
        aligned_rates_mean = trial_aligned_rates.firing_rate.mean(axis=0)
        aligned_rates_sem = trial_aligned_rates.firing_rate.sem(axis=0)
        time = aligned_rates_mean.index.to_numpy().astype(float)
        mean = aligned_rates_mean.to_numpy()
        sem = aligned_rates_sem.to_numpy()
        if smooth_SD:
            mean = gaussian_filter1d(mean, smooth_SD)
            sem = gaussian_filter1d(sem, smooth_SD)
        _plot_trial_aligned_rates(mean, sem, time, ax, color)
    else:
        goal2color = mp.get_goal2standard_color()
        for goal in trial_aligned_rates.goal.unique():
            aligned_rates_mean = trial_aligned_rates[trial_aligned_rates.goal == goal].firing_rate.mean(axis=0)
            aligned_rates_sem = trial_aligned_rates[trial_aligned_rates.goal == goal].firing_rate.sem(axis=0)
            time = aligned_rates_mean.index.to_numpy().astype(float)
            mean = aligned_rates_mean.to_numpy()
            sem = aligned_rates_sem.to_numpy()
            if smooth_SD:
                mean = gaussian_filter1d(mean, smooth_SD)
                sem = gaussian_filter1d(sem, smooth_SD)
            _plot_trial_aligned_rates(mean, sem, time, ax, goal2color[goal])
    return

def _plot_trial_aligned_rates(mean, sem, time, ax, color):
    ax.plot(time, mean, color=color)
    ax.fill_between(time, mean-sem, mean+sem, color=color, alpha=0.2)
    return




#%% Event Aligned Plotting

def plot_session_event_aligned_rates(session, goal_stratified=False):
    """"""
    event_aligned_rates_df = session.event_aligned_rates_df
    keep_clusters = gc.filter_clusters(session.cluster_metrics, #plot only single units
                                       session.session_info, 
                                       return_unique_IDs=True, 
                                       single_units=True)
    for cluster_unique_ID in keep_clusters:
        cluster_event_aligned_rates = event_aligned_rates_df[event_aligned_rates_df.cluster_unique_ID == cluster_unique_ID]
        plot_event_aligned_rates(cluster_event_aligned_rates, goal_stratified=goal_stratified)
    return


def plot_event_aligned_rates(event_aligned_rates, goal_stratified=False, smooth_SD=10, axes=None, color="black"):
    """ """
    events = ["cue", "reward"]
    if axes is None:
        f, axes = plt.subplots(1, 2, figsize=(6, 3), clear=True, sharey=True)
    f.subplots_adjust(wspace=0.01) 
    axes[0].spines["right"].set_visible(False)
    axes[0].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["left"].set_visible(False)
    axes[1].yaxis.set_visible(False)
    axes[0].set_ylabel("Firing Rate (Hz)")
    axes[0].set_xlabel("Cue (s)")
    axes[1].set_xlabel("Reward (s)")
    for ax in axes:
        ax.axvline(0, color="black", linestyle="--", alpha=0.2)
        ax.set_xlim(-12, 12)
    if not goal_stratified:
        aligned_rates_mean = event_aligned_rates.firing_rate.mean(axis=0)
        aligned_rates_sem = event_aligned_rates.firing_rate.sem(axis=0)
        for ax, event in zip(axes, events):
            key = event+"_aligned"
            mean = aligned_rates_mean[key].to_numpy()
            sem = aligned_rates_sem[key].to_numpy()
            if smooth_SD:
                mean = gaussian_filter1d(mean, smooth_SD)
                sem = gaussian_filter1d(sem, smooth_SD)
            time = aligned_rates_mean[key].index.to_numpy().astype(float)
            _plot_event_aligned_rates(mean, sem, time, ax, event, color)
    else:
        goal2color = mp.get_goal2standard_color()
        for goal in event_aligned_rates.goal.unique():
            aligned_rates_mean = event_aligned_rates[event_aligned_rates.goal == goal].firing_rate.mean(axis=0)
            aligned_rates_sem = event_aligned_rates[event_aligned_rates.goal == goal].firing_rate.sem(axis=0)
            for ax, event in zip(axes, events):
                key = event+"_aligned"
                mean= aligned_rates_mean[key].to_numpy()
                sem = aligned_rates_sem[key].to_numpy()
                if smooth_SD:
                    mean = gaussian_filter1d(mean, smooth_SD)
                    sem = gaussian_filter1d(sem, smooth_SD)
                time = aligned_rates_mean[key].index.to_numpy().astype(float)
                _plot_event_aligned_rates(mean, sem, time, ax, event, goal2color[goal])
    return

def _plot_event_aligned_rates(mean, sem, time, ax, event, color):
    """ """
    ax.plot(time, mean, color=color, label=event)
    ax.fill_between(time, mean-sem, mean+sem, color=color, alpha=0.2)
    return