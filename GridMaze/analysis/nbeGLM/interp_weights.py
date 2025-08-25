"""
Intpret output weights from input partition + latent split models
"""

# %% Imports
import numpy as np
from matplotlib import pyplot as plt
from GridMaze.analysis.nbeGLM import load_model_sets as lms
from GridMaze.analysis.nbeGLM import variance_explained as ve
from GridMaze.analysis.nbeGLM import get_input_data as gid


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


def test2(model_name="medium", maze_name="maze_1"):
    # load model for output weights analysis
    weights_model, model_params = lms.load_model(
        model_set="interpretable_models", model_name=model_name, maze_name=maze_name, with_model_params=True
    )
    return test(weights_model)
    # get cluster_unique_IDs that have variance explained by our viarbales of interest
    variance_explained_results = lms.load_model_set_cv_scores(
        "variance_explained", maze_names=[maze_name], all_completed=True
    )
    feature_tuned_df = ve.get_feature_tuned_df(
        variance_explained_results,
        reduced_models=[
            "remove_distance_to_goal",
            "remove_place_direction",
            "remove_egocentric_action_all",
        ],
    )
    keep_clusters = feature_tuned_df.index.get_level_values(1).values
    return


# %%
