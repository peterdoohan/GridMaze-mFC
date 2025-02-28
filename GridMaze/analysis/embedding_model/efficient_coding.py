"""
Question: Does mFC represent subjects behaviour in place-direction by distance-to-goal product space?

TODO:
- First get behavioural place-direction by distance to goal by trials tensor
- Load latent tuning from embedding models trained to predict neural activity
- Use latents to explain variance in behaviour
- Worry about controls, data sampling issues later :)
"""

# %% Imports
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, NMF
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

from GridMaze.analysis.core import convert
from GridMaze.maze import representations as mr

from GridMaze.analysis.embedding_model import load_experiment as le
from GridMaze.analysis.embedding_model import plot_latents as pl
from GridMaze.analysis.embedding_model import place_direction_distance_occupancies as occ

# %% Global Variables
from GridMaze.analysis.embedding_model.run_experiment import DEFAULT_INPUT_KWARGS

EXP_SET = "example_models"

# %% Get variance explained


def linalg_sanity_check(maze="maze_1", subjects="all"):
    """ """
    # get behavioural data (trials) represented in place-direction by distance-to-goal product space
    B_df = get_behavioural_data(maze, subjects, return_as="df")  # [place_directions * distance_bins, trials]
    # filter unvisted places in that product space
    occupancy_mask = B_df.sum(axis=1).gt(0)
    B_df = B_df.loc[occupancy_mask]
    # get variance explained by PCA components (upper bound)
    B = B_df.values
    pca = PCA(random_state=0)
    pca.fit(B.T)  # input as [n_samples(trials), n_features(place_directions * distance_bins)]
    pca_explained = np.cumsum(pca.explained_variance_ratio_)
    # Precompute total sum of squares for B
    TSS = np.sum((B - B.mean()) ** 2)
    # get variance explained by neural latents
    latent_explained = []
    latent_dims = [5, 10, 15, 20]  # np.arange(len(pca_explained))
    pc_components = pca.components_  # [n_components, n_features]
    for n_latents in latent_dims:
        C = pc_components[:n_latents]  # [n_latents, n_features]
        C = C.T  # [n_features, n_latents]
        W, residuals, rank, s = np.linalg.lstsq(C, B, rcond=None)
        B_hat = C.dot(W)
        SSE = np.sum((B - B_hat) ** 2)  # sum of squared errors
        R2 = 1.0 - SSE / TSS
        latent_explained.append(R2)
    return pca_explained, latent_dims, latent_explained
    _plot_var_exp(pca_explained, latent_dims, latent_explained)
    return latent_explained, pca_explained


def test(maze="maze_1", subjects="all"):
    """ """
    # get behavioural data (trials) represented in place-direction by distance-to-goal product space
    B_df = get_behavioural_data(maze, subjects, return_as="df")  # [place_directions * distance_bins, trials]
    # filter unvisted places in that product space
    occupancy_mask = B_df.sum(axis=1).gt(0)
    B_df = B_df.loc[occupancy_mask]
    # get variance explained by PCA components (upper bound)
    B = B_df.values
    pca = PCA(random_state=0)
    pca.fit(B.T)  # input as [n_samples(trials), n_features(place_directions * distance_bins)]
    pca_explained = np.cumsum(pca.explained_variance_ratio_)
    # Precompute total sum of squares for B
    TSS = np.sum((B - B.mean()) ** 2)
    # get variance explained by neural latents
    latent_explained = []
    latent_dims = [20]
    for n_latents in latent_dims:
        NL_df = get_latent_tuning_df(
            subject="all", maze_name="maze_1", input_type="onehot", latent_nonlin=None, n_latents=n_latents
        )
        NL_df = NL_df.loc[occupancy_mask]  # filter
        N = NL_df.values  # [n_features, n_latents]
        # Solve for W in B ~ N W
        # We want N (n_feat x n_latents) * W (n_latents x n_trials) = B (n_feat x n_trials)
        # => W = (N^T N)^{-1} N^T B in the least squares sense.
        W, residuals, rank, s = np.linalg.lstsq(N, B, rcond=None)
        B_hat = N.dot(W)
        SSE = np.sum((B - B_hat) ** 2)  # sum of squared errors
        R2 = 1.0 - np.sum(residuals) / TSS
        latent_explained.append(R2)
    _plot_var_exp(pca_explained, latent_dims, latent_explained)
    return latent_explained


def _plot_var_exp(pca_explained, latent_dims, latent_explained):
    """ """
    fig, ax = plt.subplots(figsize=(6, 4))

    # --- Plot the PCA line for the behavior data ---
    # X-axis for PCA can be [0, 1, 2, ..., len(pca_explained)-1]
    x_pca = np.arange(1, len(pca_explained) + 1)  # from 0 to # of components
    ax.plot(x_pca, pca_explained, label="Behavior PCA (cumulative)")

    # --- Plot single points for each neural-latent subspace dimension ---
    ax.scatter(latent_dims, latent_explained, color="red", zorder=3, label="Neural latents")
    for xd, yd in zip(latent_dims, latent_explained):
        ax.text(xd, yd, f"{yd:.2f}", ha="center", va="bottom", color="red")

    ax.set_xlim(0, latent_dims[-1] + 10)
    ax.set_ylim(0, pca_explained[latent_dims[-1] + 1])
    ax.set_xlabel("# Components")
    ax.set_ylabel("Fraction Variance Explained")
    ax.set_title("Variance in Behavior Explained by PCA vs Neural Latent Subspaces")
    ax.legend(loc="best")
    plt.show()

    return


def get_pca_variance_explained(A, B):  # A & B: [n_samples, n_features]
    """Calculates the cumulative variance of matrix B that's explained by the first i components of matrix A."""
    model = PCA(random_state=0)
    model.fit(A)
    M = model.transform(B)  # [n_samples, n_components]
    pc_exp_var = np.square(M).sum(axis=0)
    cumsum_exp_var = np.cumsum(pc_exp_var) / pc_exp_var.sum()
    return cumsum_exp_var


# %% plot behavioural PCA components


def plot_nmf_components(maze, n_components=10, norm_trials=False, binarize=False, occupancy_normalised=False):
    """ """
    simple_maze = mr.get_simple_maze(maze)
    B_df = get_behavioural_data(maze, "all", return_as="df")
    if norm_trials:
        B_df = B_df.div(B_df.sum(axis=0), axis=1)
    if binarize:
        B_df = B_df.gt(0).astype(int)
    occ_mean = B_df.mean(axis=1)
    occ_mean = occ_mean / occ_mean.sum()
    occ_mask = occ.get_occupancy_mask(maze, "all")
    B = B_df.values.T  # [trials, filtered: place_directions * distance_bins]
    nmf = NMF(random_state=0, n_components=n_components)
    T = nmf.fit(B)  # [trials, n_components]
    W = nmf.components_  # [n_components, filtered: place_directions * distance_bins]
    nmf_df = pd.DataFrame(W.T, index=B_df.index)
    for i in range(n_components):
        c = nmf_df[i]
        c = c / c.sum()
        if occupancy_normalised:
            c = c.mul(occ_mean)
        c = c.loc[occ_mask.values]
        dist_marginal = c.groupby(level="distance_to_goal").sum()
        pd_marginal = c.groupby(level="place_direction").sum()
        if occupancy_normalised:
            dist_marg_occ = occ_mean.groupby(level="distance_to_goal").sum()
            dist_marginal = dist_marginal.div(dist_marg_occ)
            pd_marg_occ = occ_mean.groupby(level="place_direction").sum()
            pd_marginal = pd_marginal.div(pd_marg_occ)
        pl.plot_marginals(pd_marginal, dist_marginal, simple_maze, place_direction_cmap="Blues")


def plot_behavioural_components(n_components=10):
    """ """
    B_df = get_behavioural_data("maze_1", ["m2"], return_as="df")
    B_df = B_df.gt(0).astype(int)  # binarize
    B = B_df.values.T  # [trials, filtered: place_directions * distance_bins]
    pca = PCA(random_state=0, n_components=n_components)
    T = pca.fit(B)  # [trials, n_components]
    W = pca.components_  # [n_components, filtered: place_directions * distance_bins]
    component_df = pd.DataFrame(W.T, index=B_df.index)
    pl.plot_latent_marginals(component_df, "maze_1")


# %% Get behavioural data


def get_behavioural_data(maze_name, subject_ID, input_kwargs=DEFAULT_INPUT_KWARGS, return_as="df"):
    """ """
    # get navigation data filtered by same get_input_data kwargs as embedding model training runs
    # see embedding_model/place_direction_distance_occupancies.py
    navigation_data = occ.get_filtered_navigation_data(maze_name, subject_ID)
    distance_bins = convert._get_distance_bins(
        input_kwargs["distance_bin_method"],
        input_kwargs["n_distance_bins"],
        input_kwargs["distance_metrics"],
        input_kwargs["max_distance"],
    )
    distnace_bin_midpoints = [d.mid for d in distance_bins]
    d2idx = {d: i for i, d in enumerate(distance_bins)}
    simple_maze = mr.get_simple_maze(maze_name)
    place_directions = mr.get_maze_place_direction_pairs(simple_maze)
    pd2idx = {pd: i for i, pd in enumerate(place_directions)}
    pds = pd.array(
        [pd2idx[pd] for pd in zip(navigation_data.maze_position.simple, navigation_data.cardinal_movement_direction)]
    )
    distance_metrics = input_kwargs["distance_metrics"]
    distance_bins_col = (distance_metrics[0], distance_metrics[1] + "_binned")
    ds = navigation_data[distance_bins_col].map(d2idx).to_numpy()
    # loop over trials and count place-direction x distance-to-goal states
    trial_PDD_counts = []
    trial_unique_IDs = navigation_data.trial_unique_ID.unique()
    for t in trial_unique_IDs:
        trial_mask = (navigation_data.trial_unique_ID == t).to_numpy()
        trial_pds = pds[trial_mask]
        trial_ds = ds[trial_mask]
        PDD = np.zeros((len(pd2idx), len(d2idx)))
        for _pd, d in zip(trial_pds, trial_ds):
            PDD[_pd, d] += 1
        trial_PDD_counts.append(PDD)
    B = np.array(trial_PDD_counts)  # [trials, place_directions, distance_bins]
    if return_as == "tensor":
        return B
    elif return_as == "matrix":
        return B.reshape(
            len(trial_unique_IDs), len(place_directions) * len(distance_bins)
        )  # [trials, place_directions * distance_bins]
    elif return_as == "df":
        return pd.DataFrame(
            B.reshape(len(trial_unique_IDs), len(place_directions) * len(distance_bins)),
            columns=pd.MultiIndex.from_product(
                [place_directions, distnace_bin_midpoints],
                names=["place_direction", "distance_to_goal"],
            ),
        ).T.astype(
            int
        )  # index = [place_directions * distance_bins], cols = trials


# %% Get neural latent data


def get_latent_tuning_df(
    subject="all", maze_name="maze_1", input_type="productspace", latent_nonlin=None, n_latents=20
):
    """
    Note hardcoded exp_dir & EXP_SET names
    """
    # update inputs & get exp_name
    subject = "all_subjects" if subject == "all" else subject
    if latent_nonlin is None:
        ln = "linear"
    elif latent_nonlin == "relu":
        ln = "nonlinear"
    exp_name = f"{subject}.{maze_name}.{input_type}_input_{ln}_{n_latents}_latents"  # hardcoded
    Encoder = le.load_encoder(exp_name, exp_set=EXP_SET)
    kwargs = le.load_kwargs(exp_name, exp_set=EXP_SET)
    latent_tuning_df = pl.get_latent_place_direction_distance_tuning(Encoder, kwargs, return_as="df")
    return latent_tuning_df
