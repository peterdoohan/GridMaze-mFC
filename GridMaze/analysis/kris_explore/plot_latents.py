""""Script for visualising latents from the embedding model"""

# %% Imports
import torch
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from . import fit_experimental_data as fed

from ..core import get_sessions as gs

from ...maze import representations as mr
from ...maze import plotting as mp

# %% Global variables
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # run on GPU if possible

EMBEDDING_MODEL_RESULTS = "../results/nn_dim_red/test_variable_combinations/"

# %% Functions


def plot_trained_model_latents(experiment_name="nonlinear_SA_distance", maze_name="maze_1"):
    latents_tuning_dfs = _load_trained_model_latents(experiment_name=experiment_name)
    simple_maze = mr.get_simple_maze(maze_name)
    for df in latents_tuning_dfs:
        _plot_model_latent(df, simple_maze, force_pos=False, norm=False)
    return


def _load_trained_model_latents(
    experiment_name="nonlinear_SA_distance",
    subject="m2",
    maze_name="maze_1",
    resolution=0.5,
    distance_metrics=("distance_to_goal", "geodesic"),
    moving_only=True,
    navigation_only=True,
    max_distance=1.8,
    n_distance_bins=20,
):
    """
    Input: results path to trained model
    Output: list len(model latents) of pd.DataFrames, each with rows: place_direction, columns = distance_bins (values = latent activity)

    Note:
    - function will only work when input_types are distance_to_goal and place_direction
    - its a bit clunky to have to pass in all the arguments that werre used to train the model + regenerate the input data
    to reextract various info, we should instead save all of this out in the model training results easily extracted all from one place
    and we can avoid errors associated with forgetting various inputs etc.
    """
    SAD_result = pickle.load(open(f"{RESULTS_DIR}/result_{experiment_name}.p", "rb"))
    sample_model = SAD_result["models"][0].to(DEVICE)
    exp = SAD_result["exp"]
    name, input_types, partition = [exp[key] for key in ["name", "input_types", "partition"]]
    if "moving_only" in exp.keys():
        moving_only = exp["moving_only"]
    if "navigation_only" in exp.keys():
        navigation_only = exp["navigation_only"]
    maze_sessions = gs.get_maze_sessions(
        subject_IDs=[subject],
        maze_names=[maze_name],
        days_on_maze="late",
        with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "navigation_routes_df"],
        must_have_data=True,
    )
    model_sessions = [
        fed.get_model_input_data(
            s,
            resolution=resolution,
            distance_metrics=distance_metrics,
            input_types=input_types,
            moving_only=moving_only,
            navigation_only=navigation_only,
        )
        for s in maze_sessions
    ]
    n_place_directions, n_dist_bins = [len(model_sessions[0]["X_type_inds"][i]) for i in range(2)]
    all_locs = torch.arange(n_place_directions)
    all_dists = torch.arange(n_dist_bins)

    all_X = torch.zeros(sample_model.Nin, n_place_directions * n_dist_bins)
    all_loc_dists = torch.zeros(all_X.shape[-1], 2)
    for loc in all_locs:
        for dist in all_dists:
            ind = loc * n_dist_bins + dist
            all_X[loc, ind] = 1.0
            all_X[n_place_directions + dist, ind] = 1.0
            all_loc_dists[ind, :] = torch.tensor([loc, dist])
    all_z = (
        sample_model.encode(all_X.to(sample_model.Wout.device)).detach().cpu().numpy()
    )  # [n_latents, n_place_directions * n_distance_bins]
    all_z = all_z.reshape(
        all_z.shape[0], n_place_directions, n_dist_bins
    )  # [n_latents, n_place_directions (ordered by mr.get_maze_place_direction_pairs), n_distance_bins (close to goal -> far from goal)]
    all_place_directions = mr.get_maze_place_direction_pairs(maze_sessions[0].simple_maze())
    place_direction_columns = pd.MultiIndex.from_tuples(all_place_directions)
    distance_bins = pd.interval_range(start=0, end=max_distance, freq=max_distance / n_distance_bins, closed="left")
    distance_bin_midpoints = [b.mid for b in distance_bins]
    # conver to list of pd.DataFrames for plotting
    return [
        pd.DataFrame(z, index=place_direction_columns, columns=distance_bin_midpoints, dtype=np.float64) for z in all_z
    ]


def _plot_model_latent(latent_tuning_df, simple_maze, force_pos=False, norm=False, axes=None):
    """ """
    n_distance_bins = latent_tuning_df.shape[1]
    distances = latent_tuning_df.columns
    vmax = latent_tuning_df.max().max()
    if force_pos:
        vmin = latent_tuning_df.min().min()
        latent_tuning_df = latent_tuning_df + abs(vmin)
        vmax = latent_tuning_df.max().max()
    if norm == "max":
        latent_tuning_df = latent_tuning_df / vmax
    if axes is None:
        fig, axes = plt.subplots(4, n_distance_bins // 4, figsize=(20, 20))
        axes = axes.flatten()
    for i, dist in enumerate(distances):
        mp.plot_directed_heatmap(
            simple_maze,
            latent_tuning_df[dist],
            fixed_vmin=-1,
            fixed_vmax=1,
            allow_negative=True,
            title=f"Distance to goal: {dist:.2f}",
            colormap="coolwarm",
            silhouette_node_size=300,
            silhouette_edge_size=6,
            ax=axes[i],
        )
    return
