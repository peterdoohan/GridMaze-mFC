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


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--session', type=int, default=8)
    
    args, _ = parser.parse_known_args()
    session_id = args.session
    
