"""
Library for plotting latent units from the embedding models
"""

# %% Imports
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from GridMaze.analysis.core import convert
from GridMaze.analysis.embedding_model import load_experiment as le
from GridMaze.maze import representations as mr
from GridMaze.maze import plotting as mp
from GridMaze.analysis.embedding_model import place_direction_distance_occupancies as occ

from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import pdist

# %% Global Variables
from ...paths import RESULTS_PATH

EMBEDDING_MODEL_RESULTS = RESULTS_PATH / "embedding_model"


# %%


def plot_raw_latents():
    exp_name = "all_subjects.maze_1.productspace_input_nonlinear_20_latents"
    exp_set = "example_models"
    Encoder = le.load_encoder(exp_name, exp_set)
    kwargs = le.load_kwargs(exp_name, exp_set)
    z_df = get_latent_place_direction_distance_tuning(Encoder, kwargs, return_as="df")
    return z_df
    f, axes = plt.subplots(1, 20, figsize=(10, 5))
    for i in range(z_df.shape[1]):
        df = z_df[i]
        df = df / df.sum()
        z = df.unstack()
        axes[i].imshow(z, cmap="coolwarm", aspect="auto")
        axes[i].axis("off")
    return


def test():
    """ """
    exp_name = "all_subjects.maze_1.productspace_input_linear_20_latents"
    exp_set = "example_models"
    Encoder = le.load_encoder(exp_name, exp_set)
    kwargs = le.load_kwargs(exp_name, exp_set)
    z_df = get_latent_place_direction_distance_tuning(Encoder, kwargs, return_as="df")
    return z_df
    occ_mask = occ.get_occupancy_mask("maze_1", "all", min_occupancy=0.5)
    save_dir = RESULTS_PATH / "embedding_model" / "figs" / "latent_tuning_productspace"
    plot_latent_marginals(
        z_df,
        maze_name="maze_1",
        occupancy_mask=occ_mask,
        cmap="coolwarm",
        save_dir=save_dir,
        norm_length=False,
        maringal_opp="mean",
    )
    return


# %%


def plot_latent_reordered(z_df):
    """ """
    Ms = [z_df[i].unstack().values for i in range(z_df.shape[1])]
    Ms = [M[:, :-3] for M in Ms if M.sum() > 0]
    Ms = [M / M.sum() for M in Ms]
    features = np.hstack(Ms)
    distance_vector = pdist(features, metric="euclidean")
    Z = linkage(distance_vector, method="average")
    Z_opt = optimal_leaf_ordering(Z, distance_vector)
    order = leaves_list(Z_opt)
    ordered_Ms = [M[order] for M in Ms]
    # order latents by distance to goal tuning max
    ordord_Ms = [ordered_Ms[i] for i in np.argsort([np.argmax(M.sum(axis=0)) for M in ordered_Ms])]
    # plot
    f, axes = plt.subplots(1, len(ordord_Ms))
    for i, ax in enumerate(axes):
        ax.imshow(ordord_Ms[i], cmap="mako")
        ax.set_axis_off()


# %%


def get_latent_place_direction_distance_tuning(Encoder, kwargs, return_as="df"):
    """
    Note this assumes the Encoder input only has input features "distance" and "place_direction"
    """
    input_kwargs = kwargs["input"]
    input_features = input_kwargs["input_features"]
    if len(input_features) == 1:
        assert input_features[0] == "place_direction_distance"
        input_type = "product_space"
    elif len(input_features) == 2:
        assert set(input_features) == {"place_direction", "distance"}
        input_type = "onehots"
    else:
        raise ValueError(
            f"input_features must be ['place_direction_distance'] or ['place_direction', 'distance'], not {input_features}"
        )
    # get distance bins
    distance_bins = convert._get_distance_bins(
        input_kwargs["distance_bin_method"],
        input_kwargs["n_distance_bins"],
        input_kwargs["distance_metrics"],
        input_kwargs["max_distance"],
    )
    # get place-directions
    simple_maze = mr.get_simple_maze(kwargs["input"]["maze_name"])
    place_directions = mr.get_maze_place_direction_pairs(simple_maze)
    # get all pairs of place-direction-distance onthot inputs and their position in the product-space
    n_pd = len(place_directions)
    n_d = len(distance_bins)
    PD = torch.arange(n_pd)  # place-direction indices
    D = torch.arange(n_d)  # distance indices
    # get all pairs of place-direction-distance as inputs (produtspace or onehots)
    if input_type == "product_space":
        X = torch.eye(n_pd * n_d)
    else:  # onehots
        if input_features == ["place_direction", "distance"]:
            X = torch.zeros(n_pd + n_d, n_pd * n_d)  # init all paired inputs to their prod-space position
            for _pd in PD:
                for d in D:
                    ind = _pd * n_d + d
                    X[_pd, ind] = 1.0
                    X[n_pd + d, ind] = 1.0
        elif input_features == ["distance", "place_direction"]:
            X = torch.zeros(n_d + n_pd, n_d * n_pd)
            for d in D:
                for _pd in PD:
                    ind = d * n_pd + _pd
                    X[d, ind] = 1.0
                    X[n_d + _pd, ind] = 1.0
    # Encode all pairs of onehot inputs and read out latent activations
    Z = Encoder.encode(X.to(Encoder.Wout.device)).detach().cpu().numpy()  # [n_latents, n_pd * n_d]
    if input_features in [["place_direction_distance"], ["place_direction", "distance"]]:
        if return_as == "tensor":
            return Z.reshape(Encoder.Nlat, n_pd, n_d)  # [n_latents, n_pd, n_d]
        elif return_as == "df":
            return pd.DataFrame(
                Z,
                columns=pd.MultiIndex.from_product(
                    [
                        place_directions,
                        [d.mid for d in distance_bins],
                    ],
                    names=[
                        "place_direction",
                        "distance_to_goal",
                    ],
                ),
            ).T.astype(np.float64)
    elif input_features == ["distance", "place_direction"]:
        if return_as == "tensor":
            return Z.reshape(Encoder.Nlat, n_d, n_pd)  # [n_latents, n_pd, n_d]
        elif return_as == "df":
            return (
                pd.DataFrame(
                    Z,
                    columns=pd.MultiIndex.from_product(
                        [
                            [d.mid for d in distance_bins],
                            place_directions,
                        ],
                        names=[
                            "distance_to_goal",
                            "place_direction",
                        ],
                    ),
                )
                .T.astype(np.float64)
                .swaplevel(1, 0)
                .sort_index()
            )


# %%


def test():
    exp_name = "maze_1_state_action_distance_10_latents"
    exp_set = "example_models"
    Encoder = le.load_encoder(exp_name, exp_set)
    kwargs = le.load_kwargs(exp_name, exp_set)
    latent_SAD_tuning = get_latent_place_direction_distance_tuning(Encoder, kwargs, return_as="df")
    z_df = latent_SAD_tuning
    occ_mask = occ.get_occupancy_mask("maze_1", "all", min_occupancy=0.5)
    plot_latent_marginals(z_df, "maze_1", occ_mask)
    return


def plot_latent_marginals(
    z_df,
    maze_name,
    occupancy_mask=None,
    max_distance=1.8,
    norm_length=True,
    marginal_opp="sum",
    save_dir=None,
):
    """
    Plots state-action (as heatmap) and distance-to-goal (as lineplot) tuning for each latent unit
    (w/ activity defined over the product-space of place-direction and distance-to-goal, defined in
    input z_df).
    """
    if occupancy_mask is not None:
        z_df = z_df.loc[occupancy_mask.values]
    if max_distance:
        z_df = z_df[z_df.index.get_level_values("distance_to_goal") <= max_distance]
    simple_maze = mr.get_simple_maze(maze_name)
    n_latents = z_df.shape[1]
    for i in range(n_latents):
        df = z_df[i]  # can we norm to length 1 with +/- values?
        if df.sum() == 0:
            print(f"Latent unit {i} has no activity")
            continue
        pd_marginal, dist_marginal = _get_marginals(df, norm_length, marginal_opp)
        # save plot if requested
        if save_dir:
            save_path = save_dir / f"latent_{i}_marginals.pdf"
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_path = False
        plot_marginals(pd_marginal, dist_marginal, simple_maze, save_path=save_path)
    return


def _get_marginals(df, norm_length=False, marginal_opp="sum"):
    if norm_length:
        df = df / df.sum()
    if marginal_opp == "sum":
        pd_marginal = df.groupby("place_direction").sum()  # sum or mean
        dist_marginal = df.groupby("distance_to_goal").sum()
    elif marginal_opp == "mean":
        pd_marginal = df.groupby("place_direction").mean()
        dist_marginal = df.groupby("distance_to_goal").mean()
    else:
        NotImplementedError()
    return pd_marginal, dist_marginal


def plot_marginals(
    pd_marginal,
    dist_marginal,
    simple_maze,
    axes=None,
    save_path=False,
):
    """ """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), width_ratios=[2, 1])
        fig.subplots_adjust(wspace=0.5)
    # plot place-direction heatmap
    if pd_marginal.index.nlevels == 1:
        pd_marginal.index = pd.MultiIndex.from_tuples(pd_marginal.index)
    with_negative = True if pd_marginal.min() < 0 else False
    pd_min = pd_marginal.min()
    pd_max = pd_marginal.max()
    if pd_min < 0:
        _max = max(abs(pd_min), abs(pd_max))
        _min = -_max
        pd_cmap = "bwr"
    else:
        _max = pd_max
        _min = pd_min
        pd_cmap = "Reds"
    mp.plot_directed_heatmap(
        simple_maze,
        pd_marginal,
        ax=axes[0],
        colormap=pd_cmap,
        colorbar=True,
        allow_negative=with_negative,
        fixed_vmin=_min,
        fixed_vmax=_max,
    )
    # plot distance curve
    x = dist_marginal.index.values
    y = dist_marginal.values
    axes[1].plot(x, y, color="grey", lw=3)
    axes[1].set_xlabel("Distance to Goal")
    axes[1].set_ylabel("Norm. Loading")
    axes[1].spines[["top", "right"]].set_visible(False)
    if axes is None:
        fig.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")


# %% old


def _plot_model_latent(latent_tuning_df, simple_maze, force_pos=False, norm=False, axes=None):
    """
    Update to NaN out locations outside of distribution limits when at goal or max distance bin once
    we can load experiment data
    """
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
