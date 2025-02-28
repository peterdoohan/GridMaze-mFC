# %%
import os
os.chdir('/ceph/behrens/peter_doohan/goalNav_mFC_refactor/code')
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

import GridMaze
from GridMaze.analysis.core import get_sessions as gs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from GridMaze.analysis.xiao_explore.clustering_curves import *
from GridMaze.analysis.xiao_explore.xiao_utils import *

# %%

subject_IDs = ['m2', 'm3', 'm4', 'm6', 'm7', 'm8']
maze_names = ['maze_1', 'maze_2']

sessions = gs.get_maze_sessions(
    subject_IDs=subject_IDs,
    maze_names=maze_names,
    days_on_maze="all",
    with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "navigation_routes_df"],
    must_have_data=True,
)

# %%
import glob
import re
root_path = '/ceph/behrens/peter_doohan/goalNav_mFC_refactor/code/GridMaze/analysis/xiao_explore/processed_data_store/umaps'
mean_paths = glob.glob(f"{root_path}/**/day**_cluster_distance_mean.npy")
std_paths = glob.glob(f"{root_path}/**/day**_cluster_distance_std.npy")
n_mazes = len(maze_names)
n_subs = len(subject_IDs)
n_sessions = 14
n_clusters = 5
mean_cluster_distances_all = np.ones((n_mazes, n_subs, n_sessions, n_clusters)) * np.nan
std_cluster_distances_all = np.ones((n_mazes, n_subs, n_sessions, n_clusters)) * np.nan
regex_pattern = r'{}/([a-zA-Z0-9_]+)_maze(\d+)/day(\d+)_cluster_distance_mean\.npy'.format(re.escape(root_path))
for path in mean_paths:
    m = re.search(regex_pattern, path)
    if m:
        subject = m.group(1)  # First group is subject
        maze = int(m.group(2))  # Second group is maze (convert to integer)
        day_on_maze = int(m.group(3))  
        subject_index = subject_IDs.index(subject)
    else:
        continue
    mean_distance = np.load(f"{root_path}/{subject}_maze{maze}/day{day_on_maze}_cluster_distance_mean.npy")
    std_distance = np.load(f"{root_path}/{subject}_maze{maze}/day{day_on_maze}_cluster_distance_std.npy")
    mean_cluster_distances_all[maze-1, subject_index, day_on_maze] = mean_distance
    std_cluster_distances_all[maze-1, subject_index, day_on_maze] = std_distance

# %%
import statsmodels.api as sm

n_subs = len(subject_IDs)
day_on_maze = np.arange(n_sessions)[None, None, ...].repeat((n_mazes, n_subs, 1)).flatten()
subjects = np.arange(n_subs)[None, ..., None].repeat((n_mazes, 1, n_sessions)).flatten()
mazes = np.arange(n_mazes)[..., None, None].repeat((1, n_subs, n_sessions)).flatten()
# subjects = np.array([subject_IDs.index(session.subject_ID) for session in sessions])
# day_on_maze = np.array([session.day_on_maze for session in sessions])
# maze_id = np.array([int(session.maze_name[-1]) for session in sessions])
# subject_maze_id = list(zip(subjects, maze_id))
# speeds = np.array([session.navigation_df[session.navigation_df.trial_phase == 'navigation'].speed.mean() for session in sessions])
mean_cluster_size = mean_cluster_distances_all.mean(axis=-1).flatten()
df = pd.DataFrame({'subject_ID': subjects, 'day_on_maze': day_on_maze, 'mean_cluster_size': mean_cluster_size})
df_exog = df[['day_on_maze']]

exog = sm.add_constant((df_exog - df_exog.mean())/df_exog.std())
endog = df['mean_cluster_size']
model = sm.MixedLM(endog, exog, groups=df.subject_ID, exog_re=exog).fit()
model.summary()

# %%
plt.rcParams['font.family'] = 'serif'
day_on_maze = np.array([session.day_on_maze for session in sessions])
subjects = np.array([int(session.subject_ID[-1]) for session in sessions])
maze = np.array([int(session.maze_name[-1]) for session in sessions])
filter = (subjects == 8) & (maze == 2)
plt.scatter(day_on_maze[filter], mean_cluster_distances_all.mean(axis=-1).flatten()[filter], c=subjects[filter])
plt.gca().spines[['top', 'right']].set_visible(False)
# plt.ylim(0, 0.125)
plt.show()
# plt.scatter(day_on_maze[filter], mean_cluster_distances_all.max(axis=-1).flatten()[filter], c=subjects[filter])
# plt.gca().spines[['top', 'right']].set_visible(False)
# # plt.ylim(0, 0.125)
# plt.show()
# plt.scatter(day_on_maze[filter], mean_cluster_distances_all.mean(axis=-1).flatten()[filter], c=subjects[filter])
# plt.gca().spines[['top', 'right']].set_visible(False)
# plt.ylim(0, 0.125)
# plt.show()
speeds = np.array([session.navigation_df.speed.mean() for session in sessions])
plt.scatter(day_on_maze[filter], speeds[filter])
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()
# %%
