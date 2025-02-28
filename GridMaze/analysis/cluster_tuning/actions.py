"""
Library for plotting firing rates aligned to basic actions (turn left, turn right, go forward, go back)
"""
#%% Imports
import matplotlib.pyplot as plt

from ..core import get_clusters as gc
from scipy.ndimage import gaussian_filter1d

#%% Global Variables


#%% Functions



def plot_session_action_tuning(session):
    action_aligned_rates_df = session.action_aligned_rates_df
    keep_clusters = gc.filter_clusters(session.cluster_metrics, #plot only single units
                                        session.session_info, 
                                        return_unique_IDs=True, 
                                        single_units=True)
    for cluster in keep_clusters:
        action_aligned_rates = action_aligned_rates_df[action_aligned_rates_df.cluster_unique_ID == cluster]
        plot_action_tuning(action_aligned_rates)
    return


def plot_action_tuning(action_aligned_rates, axes=None, smooth_SD=5):
    # set up plot
    action_aligned_rates = action_aligned_rates.copy()
    if axes is None:
        f, axes = plt.subplots(1, 3, figsize=(9, 3), clear=True, sharex=True, sharey=True)
    for ax in axes:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_ylabel("Firing Rate (Hz)")
        ax.axvline(0, color="black", linestyle="--")
    # process data
    action_aligned_rates[('forced', '')] = action_aligned_rates.choice_degree.le(2).to_numpy()
    actions = ["turn_left", "turn_right", "go_forward"]
    grouped_action_rates = action_aligned_rates.groupby(['basic_action', 'forced'], observed=True).action_aligned_rates
    mean_action_rates = grouped_action_rates.mean()
    sem_action_rates = grouped_action_rates.sem()
    for action, color, ax in zip(actions, ["red", "blue", "green"], axes):
        ax.set_xlabel(f"{action} (s)")
        for forced in [True, False]:
            color = 'black' if not forced else color
            #check there are valid actions to plot
            if not (action, forced) in mean_action_rates.index:
                continue
            else:
                select_action_mean = mean_action_rates.loc[action, forced].action_aligned_rates
                select_action_sem = sem_action_rates.loc[action, forced].action_aligned_rates
                time = select_action_mean.index.to_numpy().astype(float)
                mean = select_action_mean.to_numpy()
                sem = select_action_sem.to_numpy()
                if smooth_SD:
                    mean = gaussian_filter1d(mean, smooth_SD)
                    sem = gaussian_filter1d(sem, smooth_SD)
                _plot_action_tuning(mean, sem, time, color, ax, label=f"{action} forced={forced}")


    return

def _plot_action_tuning(mean, sem, time, color, ax, label=None):
    ax.plot(time, mean, color=color, label=label)
    ax.fill_between(time, mean-sem, mean+sem, color=color, alpha=0.2)
    ax.legend()
    return