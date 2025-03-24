# %% set up development workspace
from importlib import reload
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import os
import json
import seaborn as sns


from pathlib import Path

processed_data_path = Path("../data/processed_data/m2/2022-07-04.maze")
analysis_data_path = Path("../data/analysis_data/m2/2022-07-04.maze")
from GridMaze.analysis.core import get_sessions as gs

session = gs.get_maze_sessions(
    subject_IDs=["m2"],
    maze_names=["maze_1"],
    days_on_maze=[12],
    with_data="all",
    must_have_data=False
)

cluster_unique_ID = "m2.2022-07-17.maze_cluster141"
# %%

# srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 -p gpu --gres=gpu:1 --time=12:00:00 --mem=32G --pty bash -i
# srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=8  --time=48:00:00 --mem=32G --pty bash -i

# might have to load modules first
# source /etc/profile.d/modules.sh
# module load miniconda
# conda activate maze_ephys
# (## or source activate maze_ephys )

# cd /ceph/behrens/peter_doohan/goalNav_mFC/experiment/code

# open an environment with jupyter installed

# conda activate maze_ephys
# jupyter-notebook --no-browser --ip=0.0.0.0 --port 8888

# Copy the url starting with your node name and insert it under existing jupyter servers when selecting your kernel.

# %%
