"""
Intpret output weights from input partition + latent split models
"""

# %% Imports
import numpy as np
from matplotlib import pyplot as plt
from GridMaze.analysis.nbeGLM import load_model_sets as lms

# %% Global variables


# %% Functions


def test(model):
    _Wout = model.Wout.detach().cpu().numpy()
    _input_group_names = model.input_group_names
    _latent_split_inds = [arr.astype(int) for arr in model.latent_split_inds]
    input_group_SSWs = np.array(
        [np.sum(_Wout[:, inds] ** 2, axis=1) for inds in _latent_split_inds]
    )  # n_input_groups, n_neurons
    norm_weights = input_group_SSWs / np.sum(input_group_SSWs, axis=0, keepdims=True)
    plt.scatter(norm_weights[0], norm_weights[1], alpha=0.1, s=5)
    plt.xlabel(_input_group_names[0])
    plt.ylabel(_input_group_names[1])
    plt.show()
