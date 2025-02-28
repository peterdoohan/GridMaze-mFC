# %%
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('/ceph/behrens/peter_doohan/goalNav_mFC_refactor/code')

import GridMaze
from GridMaze.analysis.core import get_sessions as gs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import argparse
from sklearn.cluster import KMeans
import torch.nn.functional as F
import plotly
import plotly.graph_objs as go
import colorsys
# %%
subject = 'm3'
maze = 1
session_id = 4
n_clusters = 10
n_components = 3

subject_IDs = [subject]
maze_name = f"maze_{maze}"

sessions = gs.get_maze_sessions(
    subject_IDs=subject_IDs,
    maze_names=[maze_name],
    days_on_maze="late",
    with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "navigation_routes_df"],
    must_have_data=True,
)
# %%
session = sessions[session_id]
mua_clusters = session.cluster_metrics.cluster_ID[session.cluster_metrics.KSLabel == 'mua']
spk_df = session.navigation_spike_counts_df.merge(session.navigation_df[['time', 'trial_phase', 'trial_unique_ID', 'distance_to_goal', 'centroid_position', 'head_direction', 'speed']], left_on='time', right_on='time')
spk_df = spk_df.merge(session.navigation_routes_df[['route_probability']], left_on='time', right_index=True)
spk_df = spk_df[spk_df.trial_phase == 'navigation']
prefix = spk_df.spike_count.columns[0].split('.maze_cluster')[0]
spk_df = spk_df.drop(columns=['trial_phase'], errors='ignore') # + [('spike_count', f'{prefix}.maze_cluster{i}') for i in mua_clusters
# firing = spk_df.groupby('trial_unique_ID')['spike_count'].transform(lambda s: s.rolling(20, win_type='gaussian', center=True).mean(std=10))
# spk_df['spike_count'] = firing['spike_count']
spk_bins = (spk_df.time - spk_df.time.min()) // 0.5
spk_df['spk_bin'] = spk_bins
spk_df_binned = spk_df.groupby(['trial_unique_ID', 'spk_bin']).mean().reset_index()
# %%
firing = spk_df_binned.spike_count
filter = firing.notna().all(axis=1)
firing_filtered = firing[filter].to_numpy()

pca = PCA()
scaler = StandardScaler()
clusterer = KMeans(n_clusters=n_clusters)
umap_reducer = umap.UMAP(n_neighbors=80, n_components=n_components)

firing_filtered_scaled = scaler.fit_transform(firing_filtered)
firing_filtered_pca = pca.fit_transform(firing_filtered_scaled)
clusterer.fit(firing_filtered_scaled)
firing_filtered_umap = umap_reducer.fit_transform(firing_filtered_pca)

# %%
filter = firing.notna().all(axis=1)
firing_umap = np.ones((len(firing), n_components)) * np.nan
firing_umap[filter] = firing_filtered_umap
# firing_umap = pd.DataFrame(firing_umap, columns=[('umap', f'umap_{i}') for i in range(n_components)])

fr_labels = np.ones(len(spk_df_binned)) * np.nan
fr_labels[filter] = clusterer.labels_
spk_df_binned[('cluster', 'cluster_label')] = fr_labels

# %%
filter = firing.notna().all(axis=1)
firing_pca = np.ones((len(firing), firing_filtered_pca.shape[1])) * np.nan
firing_pca[filter] = firing_filtered_pca

firing_pca_diff = firing_pca[1:] - firing_pca[:-1]
firing_pca_diff[(spk_df_binned.trial_unique_ID != spk_df_binned.trial_unique_ID.shift(-1))[:-1]] = np.nan
# %%
firing_filtered_pca_diff = firing_pca_diff[~np.any(np.isnan(firing_pca_diff), axis=-1)]
firing_filtered_umap_diff = umap_reducer.fit_transform(firing_filtered_pca_diff)

# %%
# filter = firing.notna().all(axis=1)[:-1]
# firing_umap_diff = np.ones((len(firing)-1, n_components)) * np.nan
# firing_umap_diff[filter] = firing
# %%
x = firing_filtered_umap_diff[:, 0]
y = firing_filtered_umap_diff[:, 1]
z = firing_filtered_umap_diff[:, 2]

trace =  go.Scatter3d(
        x=x,  # <-- Put your data instead
        y=y,  # <-- Put your data instead
        z=z,  # <-- Put your data instead
        mode='markers',
        marker={
            'size': 1,
            'opacity': 1,
            # 'color': spk_df_binned.speed.to_numpy(),
            # 'color': spk_df_binned.distance_to_goal.geodesic.to_numpy(),
            # 'color': spk_df_binned.route_probability.route_9.to_numpy(),
            # 'colorscale':  'rdbu'
        },
    )

# Configure the layout.
layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
)

data = [trace]  # List of traces to plot

plot_figure = go.Figure(data=data, layout=layout)

# Render the plot.
plotly.offline.iplot(plot_figure)

# %%
import plotly
import plotly.graph_objs as go
import colorsys
plotly.offline.init_notebook_mode()

# Configure the trace.

x = firing_umap[:, 0]
y = firing_umap[:, 1]
z = firing_umap[:, 2]

trace =  go.Scatter3d(
        x=x,  # <-- Put your data instead
        y=y,  # <-- Put your data instead
        z=z,  # <-- Put your data instead
        mode='markers',
        marker={
            'size': 1,
            'opacity': 1,
            # 'color': spk_df_binned.speed.to_numpy(),
            'color': spk_df_binned.distance_to_goal.geodesic.to_numpy(),
            # 'color': spk_df_binned.route_probability.route_9.to_numpy(),
            # 'colorscale':  'rdbu'
        },
    )

# Configure the layout.
layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
)

data = [trace]  # List of traces to plot

plot_figure = go.Figure(data=data, layout=layout)

# Render the plot.
plotly.offline.iplot(plot_figure)
# %%
discrete_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#f4a261", "#2a9d8f", "#e9c46a", "#264653", "#f1faee",
    "#e63946", "#a8dadc", "#f1faee", "#d4a5a5", "#ffb4a2"
]
# discrete_colors = plotly.colors.qualitative.Set1 
# Configure Plotly to be rendered inline in the notebook.
plotly.offline.init_notebook_mode()
all_cluster_labels = clusterer.labels_
# Configure the trace.
traces = []
for label in range(n_clusters):
    x = firing_filtered_umap[:, 0][all_cluster_labels == label]
    y = firing_filtered_umap[:, 1][all_cluster_labels == label]
    z = firing_filtered_umap[:, 2][all_cluster_labels == label]
    color = discrete_colors[label]
    
    traces.append(go.Scatter3d(
        x=x,  # <-- Put your data instead
        y=y,  # <-- Put your data instead
        z=z,  # <-- Put your data instead
        mode='markers',
        marker={
            'size': 1,
            'opacity': 1,
            'color': color,
        },
        name=str(label)
    ))

# Configure the layout.
layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    showlegend=True, 
    # legend=dict(
    #     marker=dict(
    #         size=10  # This increases the size of the legend markers
    #     )
    # )
)

data = traces  #, trace3]

plot_figure = go.Figure(data=data, layout=layout)

# Render the plot.
plotly.offline.iplot(plot_figure)

# %%
# Example data: 10 points with 3D vectors
x = firing_umap[:, 0]
y = firing_umap[:, 1]
z = firing_umap[:, 2]

traces = []

# Vectors (dx, dy, dz) for each point
for trial in spk_df_binned.trial_unique_ID.unique():
    filter = spk_df_binned.trial_unique_ID == trial
    traces.append(
        go.Scatter3d(
            x=x[filter],
            y=y[filter],
            z=z[filter],
            mode='markers+lines',
            marker={
            'size': 1,
            'opacity': 0.8,
            'color': spk_df_binned.distance_to_goal.future[filter],
        },
            opacity=1,
            line={
            'width': 1,
            'color': spk_df_binned.distance_to_goal.future[filter],
        },
        )
    )

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    # showlegend=True, 
)

# Create the figure
fig = go.Figure(data=traces, layout=layout)

# Show the figure
fig.show()
# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Create a color map and a corresponding color key for 10 clusters
cmap = plt.get_cmap('tab20', 10)
colour_key = {i: cmap(i) for i in range(10)}

# Create legend patches using the color key
colour_patches = [mpatches.Patch(color=colour_key[c], label=c) for c in range(10)]

# Create subplots with 2 rows and 5 columns, sharing axes
fig, ax = plt.subplots(2, 5, sharex=True, sharey=True)

# Iterate through each cluster
for c in np.arange(max(clusterer.labels_)+1):
    filter = spk_df_binned[('cluster', 'cluster_label')] == c
    x = spk_df_binned[('centroid_position', 'x')][filter]
    y = spk_df_binned[('centroid_position', 'y')][filter]
    
    # Uncomment these lines if you want to use head direction data for quiver plot
    # head_direction = np.radians(spk_df_binned[('head_direction', f'value_{shift}')][filter].to_numpy() + 90)
    # dx = np.cos(head_direction)
    # dy = np.sin(head_direction)
    
    # Scatter plot of x and y positions, colored by the current cluster
    ax[c // 5, c % 5].scatter(x, y, c=cmap(c), s=10, alpha=0.3)
    
    # Uncomment the following line to plot vectors (e.g., quiver)
    # ax[c // 5, c % 5].quiver(x, y, dx / 20, dy / 20, angles='xy', scale_units='xy', scale=1, alpha=0.5)
    
    # Set equal aspect ratio and remove axis labels
    ax[c // 5, c % 5].set_aspect(1)
    ax[c // 5, c % 5].axis('off')
    
    # Set the title of the subplot to the cluster index
    ax[c // 5, c % 5].set_title(c)

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0, wspace=0)

# Show the plot
plt.show()


# %%

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

# Set font family for plots
plt.rcParams['font.family'] = 'serif'

# Create a color map with 10 distinct colors from 'tab20' colormap
cmap = plt.get_cmap('tab20', 10)

# Create a dictionary to map cluster labels to colors
colour_key = {i: cmap(i) for i in range(10)}

# Create color patches for the legend (mapping colors to cluster labels)
colour_patches = [mpatches.Patch(color=colour_key[c], label=c) for c in range(10)]

# Create subplots: adjust the number of rows based on the number of unique trials
fig, ax = plt.subplots(
    len(spk_df_binned.trial_unique_ID.unique()) // 10 + 1, 
    10, 
    sharex=True, 
    sharey=True, 
    figsize=(20, 2 * (len(spk_df_binned.trial_unique_ID.unique()) // 10 + 1))
)

# Iterate through the trials
for i, trial in enumerate(spk_df_binned.trial_unique_ID.unique()):
    # Filter the dataframe for the current trial
    filter = spk_df_binned.trial_unique_ID == trial
    
    # Get the firing labels and replace NaN or -1 values with -2 (for invalid points)
    color = spk_df_binned[filter].cluster['cluster_label']
    color[np.isnan(color) | (color == -1)] = -2
    
    # If all values are -2, set the color to 'black'
    if len(color) == (color == -2).sum():
        color = 'black'
    
    # Extract x and y centroid positions for the current trial
    x = spk_df_binned[filter][('centroid_position', 'x')]
    y = spk_df_binned[filter][('centroid_position', 'y')]
    
    # Plot the scatter plot for the current trial, with colors according to firing label
    scatter = ax[i // 10, i % 10].scatter(x, y, c=color, cmap=cmap, s=15, alpha=1)
    
    # Set aspect ratio and remove axis ticks and spines
    ax[i // 10, i % 10].set_aspect(1)
    ax[i // 10, i % 10].axis('off')
    
    # Optional: uncomment to remove top and right spines (not shown in original code)
    # ax[i // 10, i % 10].spines[['top', 'right']].set_visible(False)

# Add the color legend to the plot
plt.legend(handles=colour_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust the layout to remove extra space between subplots
plt.subplots_adjust(hspace=0, wspace=0)

# Show the plot
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

plt.figure(figsize=(20, 10))
# Create a color map and a corresponding color key for 10 clusters
cmap = plt.get_cmap('tab20', 10)
colour_key = {i: cmap(i) for i in range(10)}

# Create legend patches using the color key
colour_patches = [mpatches.Patch(color=colour_key[c], label=c) for c in range(10)]

# Create subplots with 2 rows and 5 columns, sharing axes
fig, ax = plt.subplots(2, 5, sharex=True, sharey=True)

# Iterate through each cluster
for c in np.arange(max(clusterer.labels_)+1):
    filter = spk_df_binned[('cluster', 'cluster_label')] == c
    x = spk_df_binned[('centroid_position', 'x')][filter]
    y = spk_df_binned[('centroid_position', 'y')][filter]
    
    # Uncomment these lines if you want to use head direction data for quiver plot
    head_direction = np.radians(spk_df_binned[('head_direction', f'value')][filter].to_numpy() + 90)
    dx = np.cos(head_direction)
    dy = np.sin(head_direction)
    
    # Scatter plot of x and y positions, colored by the current cluster
    # ax[c // 5, c % 5].scatter(x, y, c=cmap(c), s=10, alpha=0.3)
    
    # Uncomment the following line to plot vectors (e.g., quiver)
    ax[c // 5, c % 5].quiver(x, y, dx / 20, dy / 20, head_direction, angles='xy', scale_units='xy', scale=1, alpha=0.5, cmap='viridis')
    
    # Set equal aspect ratio and remove axis labels
    ax[c // 5, c % 5].set_aspect(1)
    ax[c // 5, c % 5].axis('off')
    
    # Set the title of the subplot to the cluster index
    ax[c // 5, c % 5].set_title(c)

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0, wspace=0)

# Show the plot
plt.show()
# %%
# Find the transition matrix
T = np.ones((n_clusters, n_clusters)) * np.nan
for i in range(n_clusters):
    for j in range(n_clusters):
        filter = spk_df_binned.trial_unique_ID == spk_df_binned.trial_unique_ID.shift(-1)
        denom = ((spk_df_binned.cluster.cluster_label == i)[filter]).sum()
        num = (((spk_df_binned.cluster.cluster_label == i) &  (spk_df_binned.cluster.cluster_label.shift(-1) == j))[filter]).sum()
        if denom == 0:
            continue
        T[i, j] = num / denom
# %%
import numpy as np

def mean_head_direction(x, dx, max_x=None, min_x=None, delta=0.1):
    """
    Computes the mean head direction for position bins.

    Args:
        x (np.array): Position, given as an np.array of shape (n_samples, n_dims).
        dx (np.array): Head direction, given as an np.array of shape (n_samples, n_dims) [(dx, dy)].
        max_x (np.array, optional): Maximum position values. Defaults to None.
        min_x (np.array, optional): Minimum position values. Defaults to None.
        delta (float, optional): The quantization of position. Defaults to 0.1.
    
    Returns:
        tuple:
            - vector_field (np.array): The mean head direction vector field, shape (n_bins, n_dims).
            - unique_positions_quantized (np.array): Unique quantized positions, shape (n_bins, n_dims).
            - n_data (np.array): Number of data points in each bin, shape (n_bins,).
    """
    # Set max_x and min_x if not provided
    if max_x is None:
        max_x = x.max(axis=0)
    if min_x is None:
        min_x = x.min(axis=0)
    
    # Quantize the positions
    position_quantized = np.floor((x - min_x) / delta).astype(int)
    
    # Get unique quantized positions
    unique_positions_quantized = np.unique(position_quantized, axis=0)
    
    # Initialize lists to store results
    vector_field = []
    n_data = []
    
    # Compute vector field and data point counts for each unique position
    for position in unique_positions_quantized:
        filter = np.all(position_quantized == position, axis=1)
        vector_field.append(np.nanmean(dx[filter], axis=0))
        n_data.append(filter.sum())
    
    # Convert results to numpy arrays
    vector_field = np.stack(vector_field)
    n_data = np.array(n_data)
    
    return vector_field, unique_positions_quantized, n_data
# %%
firing_dx = firing_umap[1:] - firing_umap[:-1]
firing_dx = firing_dx / ((firing_dx ** 2).sum(axis=-1, keepdims=True) ** 0.5)
firing_dx[(spk_df_binned.trial_unique_ID != spk_df_binned.trial_unique_ID.shift(-1))[:-1]] = np.nan
vector_field, unique_positions, n_data = mean_head_direction(firing_umap[:-1, [0, 2]], firing_dx[:, [0, 2]], delta=0.3)
# %%
import plotly.graph_objs as go
import numpy as np

# Create a grid of points (for simplicity, use a small grid)

# Create the 3D cone plot
trace = go.Cone(
    x=unique_positions[:, 0] + np.random.randn(len(unique_positions[:, 0])) * 0.2,
    y=unique_positions[:, 1] + np.random.randn(len(unique_positions[:, 0])) * 0.2,
    z=unique_positions[:, 2] + np.random.randn(len(unique_positions[:, 0])) * 0.1,
    u=vector_field[:, 0],
    v=vector_field[:, 1],
    w=vector_field[:, 2],
    cmin = 1, 
    cmax = 1,
    colorscale = 'rdbu',
    # coloraxis='coloraxis2',
    # color= vector_field[:, 0] / ((vector_field ** 2).sum(axis=-1) ** 0.5),
    # color='#000000',  # Color the cones based on the vector magnitude
    showscale=True,  # Show a color scale
    # sizemode="scaled",  # Scale the cones by their magnitude
    sizeref=1.5,  # Controls the maximum size of the cones
    opacity=1  # Opacity of the cones
)

# Set up the layout
layout = go.Layout(
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z')
    ),
    title="3D Cone Plot"
)

# Create the figure and plot
fig = go.Figure(data=[trace], layout=layout)
fig.show()

# %%
plt.figure(figsize=(20, 10))
plt.quiver(unique_positions[:, 0],
    unique_positions[:, 1],
    vector_field[:, 0],
    vector_field[:, 1],
    scale=1, scale_units='xy', angles='xy', color='r', pivot='mid', 
    headwidth=3, headlength=5, headaxislength=4
    )
plt.show()
# %%
