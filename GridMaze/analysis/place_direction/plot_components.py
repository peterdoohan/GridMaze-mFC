"""This script contains functions for plotting the components produced by dimensionality reduction (NMF, PCA) on neuron place-direction tuning"""

# %% Imports
import numpy as np
import matplotlib.pyplot as plt

from ...maze import plotting as mp

# %% Global variables


# %% Functions


def plot_nmf_components(nmf_df, simple_maze, title=False, colormap="Reds"):
    nmf_plotting_dicts = _get_nmf_component_plotting_dicts(nmf_df)
    n_components = nmf_df.shape[-1]
    n_figs = (n_components + 7) // 8
    component_idx = 0  # New index to keep track of component number
    for _ in range(n_figs):
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.tight_layout()
        if title:
            fig.suptitle(title)
        for ax in axes.flatten():
            if component_idx >= n_components:  # Break if we have plotted all components
                break
            place_direction_values = nmf_df[component_idx]
            mp.plot_directed_heatmap(
                simple_maze,
                place_direction_values,
                ax=ax,
                colormap=colormap,
                title=f"Component {component_idx}",
                value_label="Component Loading",
                silhouette_color="silver",
                silhouette_node_size=250,
                silhouette_edge_size=8,
            )
            component_idx += 1  # Increment the component index
    return


def plot_pca_components(pca_df, simple_maze, title=False, pos_cmap="Reds", neg_cmap="Blues"):
    pca_plotting_dicts = _process_PCA_components_for_plotting(pca_df)
    n_components = pca_df.shape[-1]
    n_figs = (n_components + 7) // 8
    component_idx = 0  # New index to keep track of component number
    for _ in range(n_figs):
        fig, axes = plt.subplots(4, 4, figsize=(24, 24))
        fig.tight_layout()
        if title:
            fig.suptitle(title)
        for i in range(8):  # Loop over the number of components per figure
            if component_idx >= n_components:  # Break if we have plotted all components
                break
            row_idx = i // 2
            col_start_idx = (i * 2) % 4
            pc_axes = [axes[row_idx, col_start_idx], axes[row_idx, col_start_idx + 1]]
            plotting_dict = pca_plotting_dicts[component_idx]
            pos_plotting_dict, neg_plotting_dict = plotting_dict
            pos_location2value, pos_location2NSEW = pos_plotting_dict
            neg_location2value, neg_location2NSEW = neg_plotting_dict
            component_idx += 1  # Increment the component index
            mp.plot_simple_star_heatmap(
                simple_maze,
                pos_location2value,
                pos_location2NSEW,
                ax=pc_axes[0],
                colormap=pos_cmap,
                title=f"Component {component_idx} +",
                value_label="Component Loading",
                silhouette_color="silver",
                silhouette_node_size=150,
                silhouette_edge_size=5,
            )
            mp.plot_simple_star_heatmap(
                simple_maze,
                neg_location2value,
                neg_location2NSEW,
                ax=pc_axes[1],
                colormap=neg_cmap,
                title=f"Component {component_idx} -",
                value_label="Component Loading",
                silhouette_color="silver",
                silhouette_node_size=150,
                silhouette_edge_size=5,
            )
    return


# %% Supporting functions


def _get_nmf_component_plotting_dicts(NMF_df):
    locations = list(NMF_df.index.get_level_values(0).unique())
    NMF_cols = list(NMF_df.columns)
    NMF_plotting_dicts = []
    for NMF in NMF_cols:
        comp_NMF_df = NMF_df[NMF].reset_index()
        location2value = {}
        location2NSEW = {}
        for loc in locations:
            loc_NMF_df = comp_NMF_df[comp_NMF_df.maze_position == loc]
            values = loc_NMF_df[NMF].values
            values_sum = values.sum()
            location2value[loc] = values.max()
            directions = list(loc_NMF_df.direction)
            NSEW = {}
            for dir in directions:
                if values_sum == 0:
                    norm_value = 0
                else:
                    value = loc_NMF_df[loc_NMF_df.direction == dir][NMF].values[0]
                    norm_value = value / values_sum
                NSEW[dir] = {"value": norm_value, "valid": True}
            location2NSEW[loc] = NSEW
        NMF_plotting_dicts.append((location2value, location2NSEW))
    return NMF_plotting_dicts


def _process_PCA_components_for_plotting(PCA_df):
    locations = list(PCA_df.index.get_level_values(0).unique())
    pos_PCA_df = PCA_df.copy()
    pos_PCA_df[pos_PCA_df < 0] = 0
    neg_PCA_df = PCA_df.copy()
    neg_PCA_df[neg_PCA_df > 0] = 0
    neg_PCA_df = neg_PCA_df.abs()
    PC_cols = list(PCA_df.columns)
    PC_plotting_dicts = []
    for PC in PC_cols:
        plotting_dicts = []
        for signed_PCA_df in [pos_PCA_df, neg_PCA_df]:
            PC_df = signed_PCA_df[PC].reset_index()
            location2value = {}
            location2NSEW = {}
            for loc in locations:
                loc_PC_df = PC_df[PC_df.maze_position == loc]
                values = loc_PC_df[PC].values
                values_sum = values.sum()
                location2value[loc] = values.max()
                directions = list(loc_PC_df.direction)
                NSEW = {}
                for dir in directions:
                    if values_sum == 0:
                        norm_value = 0
                    else:
                        value = loc_PC_df[loc_PC_df.direction == dir][PC].values[0]
                        norm_value = value / values_sum
                    NSEW[dir] = {"value": norm_value, "valid": True}
                location2NSEW[loc] = NSEW
            plotting_dicts.append((location2value, location2NSEW))
        PC_plotting_dicts.append(plotting_dicts)
    return PC_plotting_dicts
