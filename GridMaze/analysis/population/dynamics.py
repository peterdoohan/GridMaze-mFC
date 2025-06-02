"""
Library for population dynamics analysis on GridMaze data
"""

# %% Imports
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm


from GridMaze.maze import plotting as mp

# %% Global Variables
FRAME_RATE = 60


# %% New


# %% Functions


def get_population_trial_dynamics(session, stratified_by="goal", n_components=10, smooth_SD=0.5):
    """ """
    navigation_rates_df = session.get_navigation_activity_df(
        type="rates", with_routes=True, cluster_kwargs={"single_units": True, "multi_units": True}
    )
    population_rates = navigation_rates_df.firing_rate.values  # [time, clusters]
    # smooth rates
    if smooth_SD:
        population_rates = gaussian_filter1d(population_rates, sigma=smooth_SD * FRAME_RATE, axis=0)
    # standardise rates
    scaler = StandardScaler()
    population_rates = scaler.fit_transform(population_rates)
    # PCA
    pca = PCA(n_components=n_components)
    princple_components = pca.fit_transform(population_rates)  # [time, components]
    princple_components_df = pd.DataFrame(
        princple_components, columns=pd.MultiIndex.from_product([["principle_component"], range(1, n_components + 1)])
    )
    # combine PCs with trial info
    navigation_pcs_df = pd.concat([navigation_rates_df, princple_components_df], axis=1)
    # bin progress to goal & average PCs grouped by progress & goal
    navigation_pcs_df = navigation_pcs_df[navigation_rates_df.trial_phase == "navigation"]
    navigation_pcs_df.loc[:, ("progress_to_goal", "binned")] = pd.cut(
        navigation_rates_df.progress_to_goal.path_length, bins=30, include_lowest=True
    )
    if stratified_by == "goal":
        population_dynamics_df = navigation_pcs_df.groupby(
            [("goal", ""), ("progress_to_goal", "binned")], observed=True
        ).principle_component.mean()
    elif stratified_by == "route":
        population_dynamics_df = navigation_pcs_df.groupby(
            [("route", "r"), ("progress_to_goal", "binned")], observed=True
        ).principle_component.mean()
    plot_navigation_population_dynamics(population_dynamics_df)


def plot_navigation_population_dynamics(population_dynamics_df, ax=None, PCs=(1, 2, 3)):
    if ax is None:
        f, axes = plt.subplots(1, 3, figsize=(9, 3), clear=True)
    axes[0].set_xlabel(f"PC{PCs[0]}")
    axes[0].set_ylabel(f"PC{PCs[1]}")
    axes[1].set_xlabel(f"PC{PCs[0]}")
    axes[1].set_ylabel(f"PC{PCs[2]}")
    axes[2].set_xlabel(f"PC{PCs[1]}")
    axes[2].set_ylabel(f"PC{PCs[2]}")
    bins = population_dynamics_df.index.get_level_values(1).unique()
    goals = population_dynamics_df.index.get_level_values(0).unique()
    colormap = cm.get_cmap("brg", len(goals))
    for c, goal in enumerate(goals):
        cmap = LinearSegmentedColormap.from_list("custom", [colormap(c), "silver"], N=len(bins))
        bin2color = {bin: cmap(i) for i, bin in enumerate(bins)}
        for i, bin in enumerate(bins):
            color = bin2color[bin]
            try:
                bin_df = population_dynamics_df.loc[goal, bin].principle_component
            except KeyError:
                continue
            axes[0].scatter(bin_df[PCs[0]], bin_df[PCs[1]], color=color, s=5)
            axes[1].scatter(bin_df[PCs[0]], bin_df[PCs[2]], color=color, s=5)
            axes[2].scatter(bin_df[PCs[1]], bin_df[PCs[2]], color=color, s=5)
    return


def plot_population_dynamics_3D(population_dynamics_df, ax=None, PCs=(1, 2, 3)):
    """ """
    if ax is None:
        f = plt.figure(figsize=(8, 6))
        ax = f.add_subplot(111, projection="3d")
    bins = population_dynamics_df.index.get_level_values(1).unique()
    goals = population_dynamics_df.index.get_level_values(0).unique()
    colormap = cm.get_cmap("brg", len(goals))
    for c, goal in enumerate(goals):
        cmap = LinearSegmentedColormap.from_list("custom", ["silver", colormap(c)], N=len(bins))
        for i, bin in enumerate(bins):
            try:
                bin_df = population_dynamics_df.loc[goal, bin].principle_component
            except KeyError:
                continue
            ax.scatter(bin_df[PCs[0]], bin_df[PCs[1]], bin_df[PCs[2]], color=cmap(i), s=5)
    return
