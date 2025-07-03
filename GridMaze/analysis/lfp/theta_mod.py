"""
Characterise theta modulation across cells in mFC
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from GridMaze.analysis.core import convert
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import get_sessions as gs

from GridMaze.analysis.event_aligned import lfp_utils as lu
from GridMaze.analysis.processing import get_lfp_aligned_spike_counts as la

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with open(EXPERIMENT_INFO_PATH / "maze_configs.json", "r") as input_file:
    MAZE_CONFIGS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "maze_day2date.json", "r") as input_file:
    MAZE_DAY2DATE = json.load(input_file)

RESULTS_DIR = RESULTS_PATH / "lfp"

THETA_RANGE = (7, 11)

# %% Prop spikes/ population preference in each theta phase


def plot_population_theta_mod(population_theta_df, ax=None):
    """ """
    # average theta mod across subjects
    sub_mean_df = population_theta_df.groupby(level=0).mean()
    # plot mean and sem across subjects
    mean = sub_mean_df.mean() * 100  # convert to %
    sem = sub_mean_df.sem() * 100  # convert to %
    phases = mean.index.values.astype(float)
    # plotting
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3))
    _even_split = 100 / len(phases)
    ax.axhline(_even_split, color="k", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.errorbar(
        phases,
        mean.values,
        yerr=sem.values,
        fmt="o-",
        color="k",
        markersize=6,
        linewidth=2,
        capsize=None,
        elinewidth=2,
    )
    ax.set_xlabel("theta phase")
    ax.set_ylabel("% spikes")
    ax.set_ylim(_even_split * 0.95, _even_split * 1.05)
    ax.set_xticks(np.arange(-np.pi, np.pi + 0.1, np.pi / 2))
    ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])


def plot_population_theta_pref(population_theta_df, ax=None):
    """ """
    # get each cluster's preferred theta phase
    cluster_prefs = population_theta_df.idxmax(axis=1)
    # count preferences for each subject
    subject_counts = cluster_prefs.groupby(level=0).value_counts().unstack()
    # normalise to prop of clusters per subject
    subject_counts = subject_counts.div(subject_counts.sum(axis=1), axis=0)
    # plot
    mean = subject_counts.mean()
    sem = subject_counts.sem()
    phases = mean.index.values.astype(float)
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3))
    _even_split = 1 / len(phases)
    ax.axhline(_even_split, color="k", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.errorbar(
        phases,
        mean.values,
        yerr=sem.values,
        fmt="o-",
        color="k",
        markersize=6,
        linewidth=2,
        capsize=None,
        elinewidth=2,
    )
    ax.set_xlabel("theta phase")
    ax.set_ylabel("prop. population")
    # ax.set_ylim(_even_split * 0.95, _even_split * 1.05)
    ax.set_xticks(np.arange(-np.pi, np.pi + 0.1, np.pi / 2))
    ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])
    return


def get_population_theta_mod(verbose=True, save=False):
    """ """
    save_path = RESULTS_DIR / "population_theta_mod.csv"
    if save_path.exists() and not save:
        if verbose:
            print(f"Loading population theta modulation from {save_path}")
        return pd.read_csv(save_path, index_col=[0, 1])

    def _process_session(session):
        if verbose:
            print(session.name)
        subject_ID = session.subject_ID
        theta_mod = get_session_theta_mod(session)
        # add subject id to index
        theta_mod.index = pd.MultiIndex.from_tuples([(subject_ID, c) for c in theta_mod.index])
        return theta_mod

    dfs = []
    for subject_ID in SUBJECT_IDS:
        if verbose:
            print(f"Loading sessions for {subject_ID}")
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names="all",
            days_on_maze="late",
            with_data=[
                "navigation_df",
                "cluster_metrics",
                "navigation_theta_spike_counts_df",
                "trials_df",
            ],
            must_have_data=True,
        )
        for session in sessions:
            dfs.append(_process_session(session))

    pop_theta_mod = pd.concat(dfs, axis=0)
    if save:
        if verbose:
            print(f"Saving population theta modulation to {save_path}")
        pop_theta_mod.to_csv(save_path)
    return pop_theta_mod


def get_session_theta_mod(session, navigation_only=True, moving_only=True, max_steps_to_goal=30, min_spikes=300):
    """ """
    # load data
    session_info = session.session_info
    cluster_metrics = session.cluster_metrics
    navigation_df = session.navigation_df.copy()
    theta_spike_counts_df = session.navigation_theta_spike_counts_df.reset_index(drop=True)
    # filter for single units
    single_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID
    single_units = convert.cluster_IDs2scluster_unique_IDs(session_info, single_units)
    theta_spike_counts_df = theta_spike_counts_df[
        theta_spike_counts_df.columns[[c in single_units for c in theta_spike_counts_df.columns.get_level_values(1)]]
    ]
    # combine nav and spikes
    navigation_df.columns = pd.MultiIndex.from_tuples([(*c, "") for c in navigation_df.columns])
    nav_spike_counts_df = pd.concat([navigation_df, theta_spike_counts_df], axis=1)
    # filter for moving, navigation, on task etc.
    nav_spike_counts_df = filt.filter_navigation_rates_df(
        nav_spike_counts_df, navigation_only, moving_only, max_steps_to_goal=max_steps_to_goal
    )
    cluster_phase_spike_counts = nav_spike_counts_df.spike_count.sum().unstack()
    # filter for cluster with few spikes in filtered data (eg, non-navigation tuned)
    cluster_phase_spike_counts = cluster_phase_spike_counts[cluster_phase_spike_counts.sum(axis=1) > min_spikes]
    # normalise each to prop (sum =1) of spikes in each phase
    cluster_phase_prop = cluster_phase_spike_counts.div(cluster_phase_spike_counts.sum(axis=1), axis=0)
    return cluster_phase_prop


# %% get average theta aligend lfp


def get_theta_aligned_lfp_df(save=False, verbose=True):
    """
    Note get sessions one-by-one to avoid memory issues
    with massive LFP arrays.
    """
    save_path = RESULTS_DIR / "theta_aligned_lfp.csv"
    if save_path.exists() and not save:
        if verbose:
            print(f"Loading theta aligned lfp from {save_path}")
        return pd.read_csv(save_path, index_col=[0, 1])

    aligned_lfps = []
    for subject in SUBJECT_IDS:
        for maze in MAZE_CONFIGS.keys():
            days_on_maze = [int(d) for d in MAZE_DAY2DATE[maze].keys()]
            late_days = days_on_maze[-7:]
            for day in late_days:
                try:
                    session = gs.get_maze_sessions(
                        subject_IDs=[subject],
                        maze_names=[maze],
                        days_on_maze=[day],
                        with_data=["lfp_times", "lfp_signal", "lfp_metrics", "cluster_metrics"],
                        must_have_data=True,
                    )
                    if verbose:
                        print(session.name)
                    theta_aligned_lfp = get_session_theta_aligned_lfp(session)
                    aligned_lfps.append(theta_aligned_lfp)
                except FileNotFoundError:
                    pass  # minority of sessions missing data
    theta_alinged_df = pd.concat(aligned_lfps, axis=1)
    if save:
        if verbose:
            print(f"Saving theta aligned lfp to {save_path}")
        theta_alinged_df.to_csv(save_path)
    return theta_alinged_df


def get_session_theta_aligned_lfp(session, n_bins=32):
    """ """
    lfp_signal = lu.get_LFP(session)
    # get theta phase
    theta_phase = la.get_lfp_phase(lfp_signal, freq_range=THETA_RANGE, N=4)
    # bin phases finely
    bin_edges, theta_phase_bins = la.bin_lfp_phase(theta_phase, n_bins=n_bins)
    # average lfp signal in each phase bin
    theta_aligned_lfp = np.zeros(len(bin_edges) - 1)
    for i in range(n_bins):
        theta_aligned_lfp[i] = lfp_signal[theta_phase_bins == i].mean()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return pd.Series(index=bin_centers, data=theta_aligned_lfp, name=(session.subject_ID, session.name))
