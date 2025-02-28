import sys
sys.path.append('/ceph/behrens/Xiao/MazeModel')
sys.path.append(f'/ceph/behrens/Xiao/Grid_Worlds/GridWorldRL')
sys.path.append(f'/ceph/behrens/Xiao/RNN_model/RNNmodel')

import MazeModel
from MazeModel.preprocess.maze_transitions_preprocess_funcs import maze_transitions_mask

from MazeModel.plots.plot_policy import *
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