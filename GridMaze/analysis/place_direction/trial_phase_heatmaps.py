""" This script is for visualising and doing dimensionality reduction on neural place-direction data in different trial phases"""
"""Script not under active development"""
# %% Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from . import dimensionality_reduction as dr
from . import get_neural_place_direction_df as npd
from . import plot_components as pc
from .. import get_sessions as gs

# %% Global variables


# %% Functions


def get_trial_phase_split_neural_place_direction_df(
    session,
    navigation_exclude_time_at_goal=True,
    navigation_moving=True,
    ITI_moving=True,
    min_occupany=1,
    normalisation_method="length",
):
    """"""
    simple_maze = session.simple_maze()
    navigation_rates_df = session.get_navigation_activity_df(activity_type="firing_rate", cluster_type="good")
    cluster_analysis_metrics_df = session.cluster_analysis_metrics_df
    # split KS 'good' clusters based on trial phase tuning
    cluster_trial_phase_tuning = cluster_analysis_metrics_df.trial_phase_tuning
    KS_good_mask = cluster_analysis_metrics_df.KS_label == "good"
    trial_phase2active_clusters = {
        p: cluster_analysis_metrics_df[np.logical_and(KS_good_mask, cluster_trial_phase_tuning[p])].index.to_numpy()
        for p in ["navigation", "reward_consumption", "ITI"]
    }
    # split navigation_rates_df by trial phase and active clusters
    trial_phase2navigation_rates_df = {}
    for trial_phase in ["navigation", "reward_consumption", "ITI"]:
        active_clusters = trial_phase2active_clusters[trial_phase]
        if len(active_clusters) == 0:
            trial_phase2navigation_rates_df[trial_phase] = None
            continue
        navigation_phase_rates_df = navigation_rates_df[navigation_rates_df.trial_phase == trial_phase].copy()
        if (trial_phase == "navigation" and navigation_moving) or (trial_phase == "ITI" and ITI_moving):
            navigation_phase_rates_df = navigation_phase_rates_df[navigation_phase_rates_df.moving]
        if trial_phase == "navigation" and navigation_exclude_time_at_goal:
            navigation_phase_rates_df = navigation_phase_rates_df[
                ~(navigation_phase_rates_df.maze_position.simple == navigation_phase_rates_df.goal)
            ]
        all_clusters = navigation_phase_rates_df.firing_rate.columns.to_numpy()
        drop_clusters = np.setdiff1d(all_clusters, active_clusters)
        navigation_phase_rates_df.drop(columns=drop_clusters, level=1, inplace=True)
        trial_phase2navigation_rates_df[trial_phase] = navigation_phase_rates_df
    place_direction_dfs = []
    for trial_phase, fill_nans in zip(["navigation", "reward_consumption", "ITI"], ["mean", False, "mean"]):
        navigation_rates_df = trial_phase2navigation_rates_df[trial_phase]
        if navigation_rates_df is None:
            place_direction_dfs.append(None)
        else:
            place_direction_df = npd._get_place_direction_df(
                navigation_rates_df,
                simple_maze,
                fill_nans=fill_nans,
                minimum_occupancy=min_occupany,
                normalisation_method=normalisation_method,
            )
            place_direction_dfs.append(place_direction_df)
    navigation_place_direction_df, reward_consumption_place_direction_df, ITI_place_direction_df = place_direction_dfs
    return navigation_place_direction_df, reward_consumption_place_direction_df.fillna(0), ITI_place_direction_df


def get_analysis_sessions(maze_number):
    sessions = gs.get_sessions(
        subject_IDs="all",
        maze_number=[maze_number],
        day_on_maze="late",
        with_data=[
            "navigation_df",
            "navigation_spike_counts_df",
            "navigation_spike_rates_df",
            "cluster_metrics",
            "cluster_analysis_metrics_df",
        ],
    )
    return sessions


def get_multisession_trial_phase_split_neural_place_direction_df(sessions):
    navigation_dfs, reward_consumption_dfs, ITI_dfs = [], [], []
    for session in sessions:
        (
            navigation_place_direction_df,
            reward_consumption_place_direction_df,
            ITI_place_direction_df,
        ) = get_trial_phase_split_neural_place_direction_df(session)
        navigation_dfs.append(navigation_place_direction_df)
        reward_consumption_dfs.append(reward_consumption_place_direction_df)
        ITI_dfs.append(ITI_place_direction_df)
    navigation_place_direction_df = pd.concat(navigation_dfs, axis=0)
    reward_consumption_place_direction_df = pd.concat(reward_consumption_dfs, axis=0)
    ITI_place_direction_df = pd.concat(ITI_dfs, axis=0)
    return navigation_place_direction_df, reward_consumption_place_direction_df, ITI_place_direction_df


def plot_nmf_place_direction_decomposition(trial_phase_place_direction_dfs, simple_maze, n_components=8):
    for place_direction_df, trial_phase in zip(
        trial_phase_place_direction_dfs, ["navigation", "reward_consumption", "ITI"]
    ):
        nmf_df = dr.get_nmf_df(place_direction_df, n_components=n_components)
        pc.plot_nmf_components(nmf_df, simple_maze, title=trial_phase, colormap="Reds")
    return
