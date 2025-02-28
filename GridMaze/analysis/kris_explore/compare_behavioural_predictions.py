


import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import pearsonr
import os
import warnings 
import copy
warnings.filterwarnings("ignore")
import pickle

from importlib import reload
import GridMaze.analysis.embedding_model.get_input_data
import GridMaze.analysis.embedding_model.embedding_utils
from GridMaze.analysis.embedding_model.embedding_utils import Encoder, train_model, calc_poisson_deviance
from GridMaze.analysis.embedding_model.get_input_data import get_input_data
reload(GridMaze.analysis.embedding_model.get_input_data)
reload(GridMaze.analysis.embedding_model.embedding_utils)
from GridMaze.analysis.embedding_model.embedding_utils import Encoder, train_model, calc_poisson_deviance
from GridMaze.analysis.embedding_model.get_input_data import get_input_data, _downsample_navigation_spike_counts
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import convert
from GridMaze.maze import representations as mr
    
#%% set some parameters

seed = 1
np.random.seed(seed) # fix random seeds for now
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # run on GPU if possible
resolution = 0.1
min_spike_count = 300
maze_name = "maze_1"
simple_maze = mr.get_simple_maze(maze_name)

subjects = ['m2', 'm3', 'm4', 'm6', 'm7', 'm8']
t0 = time.time()
subject = "m2"

print(f"\n\nrunning for subject {subject}")
sessions = gs.get_maze_sessions(
    subject_IDs=[subject],
    maze_names=[maze_name],
    days_on_maze="late",
    with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "navigation_routes_df"],
    must_have_data=True,
)

session = sessions[1]


        # load data from session obj
        navigation_spikes_df = session.get_navigation_activity_df(
            type="spikes",
            cluster_kwargs={"single_units": True, "multi_units": False},
            with_routes=False,
        )
        # downsample data from frame-by-frame to specified resolution
        if resolution:
            navigation_spikes_df = _downsample_navigation_spike_counts(navigation_spikes_df, resolution)

        times = navigation_spikes_df.time.values
        diff = times[1:] - times[:-1]
        #assert np.std(diff) / np.mean(diff) < 0.01 # should have continuous values
        print(np.std(diff) / np.mean(diff))

        shift_type = "states" # either shift based on either number of states or time elapsed
        all_states = navigation_spikes_df.maze_position.simple.values
        phases = navigation_spikes_df.trial_phase.values
        all_states[phases != "navigation"] = None
        for trial in navigation_spikes_df.trial.unique():
            if not np.isnan(trial):
                trial_inds = np.where(navigation_spikes_df.trial == trial)[0]
                goal_inds = trial_inds[(navigation_spikes_df.maze_position.simple.values[trial_inds] == navigation_spikes_df.goal.values[trial_inds[0]] )]
                all_states[goal_inds] = None
        
        boundaries = np.concatenate([np.zeros(1).astype(int), np.where(all_states[1:] != all_states[:-1])[0]+1]) # first index of each state
        offsets = np.arange(1,7)

