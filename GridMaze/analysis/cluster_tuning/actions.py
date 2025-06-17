"""
Library for plotting firing rates aligned to basic actions (turn left, turn right, go forward, go back)
"""

# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# %% Global Variables
FRAME_RATE = 60  # Hz

# %% Functions


def plot_session_action_tuning(session):
    navigation_rates_df = session.get_navigation_activity_df(type="rates", cluster_kwargs={"single_units": True})
    action_aligned_rates_df = _get_basic_action_tuning(navigation_rates_df)
    cluster_unique_IDs = navigation_rates_df.firing_rate.columns.to_numpy()
    for cluster in cluster_unique_IDs:
        action_aligned_rates = action_aligned_rates_df[action_aligned_rates_df.cluster_unique_ID == cluster]
        plot_action_tuning(action_aligned_rates)
    return


def plot_action_tuning(action_aligned_rates, axes=None, smooth_SD=5):
    # set up plot
    action_aligned_rates = action_aligned_rates.copy()
    times = action_aligned_rates.action_aligned_rates.columns.values.astype(float)
    if axes is None:
        f, axes = plt.subplots(1, 3, figsize=(9, 3), clear=True, sharex=True, sharey=True)
    for ax in axes:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_ylabel("Firing Rate (Hz)")
        ax.axvline(0, color="black", linestyle="--")
    ax.set_xlim(times[0], times[-1])
    # process data
    action_aligned_rates[("forced", "")] = action_aligned_rates.choice_degree.le(2).to_numpy()
    actions = ["turn_left", "turn_right", "go_forward"]
    grouped_action_rates = action_aligned_rates.groupby(["basic_action", "forced"], observed=True).action_aligned_rates
    mean_action_rates = grouped_action_rates.mean()
    sem_action_rates = grouped_action_rates.sem()
    for action, color, ax in zip(actions, ["red", "blue", "green"], axes):
        ax.set_xlabel(f"{action} (s)")
        for forced in [True, False]:
            color = "black" if not forced else color
            # check there are valid actions to plot
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


def plot_action_tunning_concise(action_aligned_rates, ax=None, smooth_SD=5, action_type="all"):
    """
    Plot only forced actions on one axes
    """
    # set up plot
    action_aligned_rates = action_aligned_rates.copy()
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3), clear=True)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylabel("Firing Rate (Hz)")
    ax.axvline(0, color="black", linestyle="--")
    ax.set_xlabel("Action aligned time (s)")
    # process data
    action_aligned_rates[("forced", "")] = action_aligned_rates.choice_degree.le(2).to_numpy()
    actions = ["turn_left", "turn_right", "go_forward"]
    grouped_action_rates = action_aligned_rates.groupby(["basic_action", "forced"], observed=True).action_aligned_rates
    mean_action_rates = grouped_action_rates.mean()
    sem_action_rates = grouped_action_rates.sem()
    # do some plotting
    actions = ["turn_left", "turn_right", "go_forward"]
    for action, color in zip(actions, ["darkred", "royalblue", "grey"]):
        # only plot tuning to forced actions
        if action_type == "forced":
            select_action_mean = mean_action_rates.loc[action, True].action_aligned_rates
            select_action_sem = sem_action_rates.loc[action, True].action_aligned_rates
        elif action_type == "free":
            select_action_mean = mean_action_rates.loc[action, False].action_aligned_rates
            select_action_sem = sem_action_rates.loc[action, False].action_aligned_rates
        elif action_type == "all":
            select_action_mean = mean_action_rates.loc[action].action_aligned_rates.mean()
            select_action_sem = sem_action_rates.loc[action].action_aligned_rates.mean()
        else:
            raise ValueError("action_type must be 'free', 'forced' or 'all'")
        time = select_action_mean.index.to_numpy().astype(float)
        mean = select_action_mean.to_numpy()
        sem = select_action_sem.to_numpy()
        if smooth_SD:
            mean = gaussian_filter1d(mean, smooth_SD)
            sem = gaussian_filter1d(sem, smooth_SD)
        _plot_action_tuning(mean, sem, time, color, ax, label=f"{action.split('_')[-1]}")
    ax.legend()


def _plot_action_tuning(mean, sem, time, color, ax, label=None):
    ax.plot(time, mean, color=color, label=label)
    ax.fill_between(time, mean - sem, mean + sem, color=color, alpha=0.2)
    return


# %%


def _get_basic_action_tuning(
    navigation_rates_df,
    actions=["turn_left", "go_forward", "turn_right", "go_back"],
    window=(-3, 3),
):
    """
    Returns a dataframe with firing rates aligned to basic actions (turn_left, go_forward, turn_right). Where rows are cluster x action -> rates within window
    INPUT:
        - subject_session_path in preprocessing/analysis_data folders: eg, 'm8/2022-07-05-135156'
        - actions: list of basic actions to align to (not including 'go_back' actions from now by default)
        - window: tuple of time window to align to (in seconds)
        - frame_rate: frame rate of video (default 60 Hz)
    """
    # process basic action aligned rates
    pre_win, post_win = [w * FRAME_RATE for w in window]
    all_action_aligned_rates_dfs = []
    for action in actions:
        action_rates_df = navigation_rates_df[navigation_rates_df.action.basic == action]
        action_inds = action_rates_df.index.to_numpy()
        choice_degrees = action_rates_df.action.choice_degree.to_numpy()
        aligned_timepoints = np.arange(window[0], window[1], 1 / FRAME_RATE)
        cluster_unique_IDs = action_rates_df.firing_rate.columns.to_numpy()
        # initialise action_aligned_rates_df
        columns = [
            ("cluster_unique_ID", ""),
            ("basic_action", ""),
            ("action_number", ""),
            ("choice_degree", ""),
        ]
        columns += [("action_aligned_rates", t) for t in aligned_timepoints]
        init_action_aligned_rates_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(columns))
        init_action_aligned_rates_df[("cluster_unique_ID", "")] = cluster_unique_IDs
        action_aligend_rates_dfs = []
        for i, action_ind in enumerate(action_inds):
            if navigation_rates_df.iloc[action_ind].trial_phase.to_numpy() != "navigation":
                continue  # skip actions that do not occur during navigation
            else:
                action_aligned_rates_df = init_action_aligned_rates_df.copy()
                cluster_action_aligned_rates = navigation_rates_df.iloc[
                    (action_ind + pre_win) : (action_ind + post_win)
                ].firing_rate.T.to_numpy()  # [n_clusters, n_timepoints]
                if action_aligned_rates_df.action_aligned_rates.shape != cluster_action_aligned_rates.shape:
                    continue  # skip actions where the window ends outside of session (usually last couple of actions in session)
                action_aligned_rates_df[("basic_action", "")] = action
                action_aligned_rates_df[("action_number", "")] = i + 1
                action_aligned_rates_df[("choice_degree", "")] = choice_degrees[i]
                action_aligned_rates_df.action_aligned_rates = cluster_action_aligned_rates
                action_aligend_rates_dfs.append(action_aligned_rates_df)
        all_action_aligned_rates_dfs.append(pd.concat(action_aligend_rates_dfs, axis=0))
    return pd.concat(all_action_aligned_rates_dfs, axis=0)
