"""
Analysis of latent units from the embedding model. Are they best captured as an addition
or mutliplication of two feature tuning curves or some constant tuning curve in one dimension
dimensions = place_direction x distance_to_goal
equivalent to 
dimensions = state-action x distance_to_goal
"""

# %% Imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from GridMaze.analysis.embedding_model import plot_latents as pl
from GridMaze.analysis.embedding_model import load_experiment as le
from GridMaze.analysis.embedding_model import place_direction_distance_occupancies as occ
from GridMaze.maze import representations as mr
from GridMaze.maze import plotting as mp

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import pdist
from scipy.stats import ttest_rel

# %% Global Variables

from GridMaze.paths import RESULTS_PATH


# %%
def test(exp_name="all_subjects.maze_1.onehot_input_linear_20_latents"):
    exp_set = "example_models"
    Encoder = le.load_encoder(exp_name, exp_set)
    kwargs = le.load_kwargs(exp_name, exp_set)
    results_df = compare_models(
        Encoder,
        kwargs,
        plot_reconstruction=True,
        plot_model_comparison=True,
        save=False,
    )
    return results_df


# %% Compare models


def compare_models(
    Encoder,
    kwargs,
    compare_models=[
        "AdditiveModel",
        "MultiplicativeModel",
    ],
    plot_reconstruction=True,
    plot_model_comparison=True,
    save=False,
):
    """ """

    latent_SAD_tuning = pl.get_latent_place_direction_distance_tuning(Encoder, kwargs, return_as="df")
    results_df = pd.DataFrame(index=range(kwargs["model_init"]["Nlat"]), columns=compare_models)
    cosine_sims = []
    for i in range(latent_SAD_tuning.shape[1]):
        print(f"Latent {i}")
        z_df = latent_SAD_tuning[i].unstack()
        if z_df.sum().sum() == 0:
            print("Latent has no activity, skip")
            results_df.loc[i, :] = np.nan
            continue
        model_results = {}
        for model_name in compare_models:
            Model = globals()[model_name]
            best_model, best_losses, best_r2 = train_model_multiple(Model, z_df, kwargs, verbose=False)
            print(f"{model_name}: R² = {best_r2:.4f}")
            results_df.loc[i, model_name] = best_r2
            model_results[model_name] = {"model": best_model, "r2": best_r2}
        # cosine sims
        cosine_sims.append(get_cosine_sims(model_results))
        if plot_reconstruction:
            if save:
                save_path = (
                    RESULTS_PATH
                    / "embedding_model"
                    / "latent_add_mull"
                    / kwargs["input"]["exp_name"]
                    / f"latent_{i}.pdf"
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                save_path = False
            plot_reconstruction_summary(*model_results.values(), save_path)
    if plot_model_comparison:
        if save:
            save_path = (
                RESULTS_PATH / "embedding_model" / "latent_add_mull" / kwargs["input"]["exp_name"] / "add_vs_mull.pdf"
            )
        else:
            save_path = False
        plot_add_vs_mul(results_df, save_path)
    return results_df, pd.DataFrame(cosine_sims)


def plot_reconstruction_summary(additive_results, multiplicative_results, save_path=False):
    """ """
    fig, axes = plt.subplots(3, 2, figsize=(9, 14), width_ratios=[2, 1])
    fig.tight_layout()
    # plot true latent marginals
    axes[0][0].set_title("True Marginals")
    additive_results["model"].plot_true_marginals(axes=(axes[0][0], axes[0][1]))
    additive_results["model"].plot_feature_tuning(axes=(axes[1][0], axes[1][1]))
    axes[1][0].set_title(f"Additive: R2={additive_results["r2"]:.4f}")
    multiplicative_results["model"].plot_feature_tuning(axes=(axes[2][0], axes[2][1]))
    axes[2][0].set_title(f"Multiplicative: R2={multiplicative_results["r2"]:.4f}")
    if save_path:
        fig.savefig(save_path)
    return


def plot_add_vs_mul(results_df, save_path=False):
    """ """
    df = results_df.dropna()
    delta_df = df["MultiplicativeModel"] - df["AdditiveModel"]
    delta_df = delta_df.reset_index()
    delta_df.columns = ["latent", "delta_r2"]
    f, ax = plt.subplots(1, 1, figsize=(1, 3))
    sns.stripplot(data=delta_df, y="delta_r2", ax=ax)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", ls="--")
    t_stat, p_value = ttest_rel(df["MultiplicativeModel"], df["AdditiveModel"])
    ax.set_title(f"p={p_value:.4f}")
    if save_path:
        f.savefig(save_path, bbox_inches="tight")


# %%


def get_add_mull_metric(model_results):
    """ """
    AddMod = model_results["AdditiveModel"]["model"]
    MulMod = model_results["MultiplicativeModel"]["model"]
    A = AddMod.forward().detach().cpu().numpy()
    M = MulMod.forward().detach().cpu().numpy()
    T = AddMod.z.detach().cpu().numpy()
    A_vec = A.flatten()
    M_vec = M.flatten()
    T_vec = T.flatten()
    # compute projection of T onto the lin from A to M
    v = M_vec - A_vec
    t_star = np.dot(T_vec - A_vec, v) / np.dot(v, v)
    return t_star


def get_cosine_sims(model_results):
    """ """
    AddMod = model_results["AdditiveModel"]["model"]
    MulMod = model_results["MultiplicativeModel"]["model"]
    A_vec = AddMod.forward()
    M_vec = MulMod.forward()
    Z_vec = AddMod.z.detach()
    return {
        "Add2Mul": torch.cosine_similarity(A_vec, M_vec).item(),
        "Add2True": torch.cosine_similarity(A_vec, Z_vec).item(),
        "Mul2True": torch.cosine_similarity(M_vec, Z_vec).item(),
    }


# %% training functions


def train_model_multiple(ModelClass, z_df, kwargs, n_repeats=2, n_epochs=30_000, lr=0.05, verbose=False):
    """
    Trains the model multiple times (with new random initializations) and returns the model
    with the best final R² (variance explained) over the valid entries.

    Parameters:
      ModelClass - the class of the model to instantiate (e.g. AdditiveModel)
      z_df       - the DataFrame of latent values (used to initialize the model)
      kwargs     - a dictionary of additional keyword arguments; for example, containing input parameters.
      n_repeats  - the number of training repeats (each with a fresh model)
      n_epochs   - number of training epochs per repeat.
      lr         - learning rate.
      verbose    - if True, prints progress during training.

    Returns:
      best_model  - the trained model instance with the highest final R².
      best_losses - the list of R² values (over epochs) for the best model.
      best_r2     - the final R² value for the best model.
    """
    best_r2 = -float("inf")
    best_model = None
    best_losses = None

    for i in range(n_repeats):
        if verbose:
            print(f"\nTraining repeat {i + 1} of {n_repeats}:")
        # Instantiate a fresh model for this repeat.
        model = ModelClass(z_df, kwargs)
        losses = train_model(model, n_epochs=n_epochs, lr=lr, verbose=verbose)
        current_r2 = final_loss(model)
        if verbose:
            print(f"Repeat {i + 1} final R² = {current_r2:.4f}")
        if current_r2 > best_r2:
            best_r2 = current_r2
            best_model = model
            best_losses = losses

    return best_model, best_losses, best_r2


def train_model(model, n_epochs=5_000, lr=0.005, verbose=False):
    """
    Trains the model using only the valid (masked) entries of the target.
    The loss used is the negative proportion of variance explained (i.e. -R²)
    so that the optimizer maximizes the variance explained.

    Parameters:
      model    - a PyTorch nn.Module instance.
      target   - a torch.Tensor of shape (n_sa, n_distance).
      occ_mask - a boolean tensor of the same shape as target. True for valid entries.
      n_epochs - number of training iterations.
      lr       - learning rate.
      verbose  - if True, prints R² every 500 epochs.

    Returns:
      A list of R² values (proportion of variance explained) over epochs.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    target = model.z
    mask = model.mask
    # Pre-compute the valid target values and TSS (total sum of squares) for the valid entries.
    valid_target = target[mask]
    mean_valid_target = valid_target.mean()
    eps = 1e-8  # to avoid division by zero
    TSS = torch.sum((valid_target - mean_valid_target) ** 2) + eps

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = model()
        valid_output = output[mask]

        # Compute residual sum of squares (RSS)
        RSS = torch.sum((valid_target - valid_output) ** 2)
        # Compute proportion of variance explained (R²)
        r2 = 1 - (RSS / TSS)

        # Define loss as the negative R² so that minimizing loss maximizes R².
        loss = -r2
        loss.backward()
        optimizer.step()

        losses.append(r2.item())
        if verbose and epoch % 500 == 0:
            print(f"Epoch {epoch}: Proportion of Variance Explained (R²) = {r2.item():.4f}")

    return losses


def final_loss(model):
    """
    Computes the final proportion of variance explained (R²) for the trained model,
    taking into account only the valid entries specified by occ_mask.

    Parameters:
      model    - a PyTorch nn.Module instance.
      target   - a torch.Tensor of shape (n_sa, n_distance).
      occ_mask - a boolean tensor of the same shape as target, where True indicates valid entries.

    Returns:
      The R² computed only over the valid entries.
    """
    with torch.no_grad():
        output = model()
        mask = model.mask
        valid_output = output[mask]
        valid_target = model.z[mask]
        mean_valid_target = valid_target.mean()
        eps = 1e-8
        TSS = torch.sum((valid_target - mean_valid_target) ** 2) + eps
        RSS = torch.sum((valid_target - valid_output) ** 2)
        r2 = 1 - (RSS / TSS)
    return r2.item()


# %%


class _Model(nn.Module):
    """
    Base class for the additive and multiplicative & constant models.
    """

    def __init__(self, z_df, kwargs):
        super(_Model, self).__init__()
        self.z_df = z_df
        n_state_action, n_distance = z_df.shape
        self.n_state_action = n_state_action
        self.distance_idx = z_df.columns
        self.state_action_idx = z_df.index
        self.z = torch.tensor(z_df.values).float()
        self.n_distance = n_distance
        for attr in kwargs["input"].keys():
            setattr(self, attr, kwargs["input"][attr])
        # get occupancy mask (mask out of distrubution states in the latent z)
        self.occ_mask = occ.get_occupancy_mask(self.maze_name, self.subject_IDs)
        self.mask = torch.tensor(self.occ_mask.unstack().values)
        # exclude distances visited in < 1/3rd of states (for plotting)
        self.dist_mask = self.occ_mask.groupby("distance_to_goal").sum().gt(self.n_state_action // 3).values
        self.simple_maze = mr.get_simple_maze(self.maze_name)
        # init with true marginal
        self.sa_param = nn.Parameter(self.z.mean(axis=1)[:, None])
        self.dist_param = nn.Parameter(self.z.mean(axis=0)[None, :])

    def plot_true_marginals(self, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4), width_ratios=[2, 1])
        sa_marginal, dist_marginal = pl._get_marginals(
            self.z_df.unstack(),
            norm_length=True,
            marginal_opp="sum",
        )
        pl.plot_marginals(sa_marginal, dist_marginal, self.simple_maze, axes=axes)
        return

    def get_feature_tuning(self, feature):
        if feature == "state_action":
            return self.sa_param.detach().cpu().numpy().squeeze()
        if feature == "distance":
            return self.dist_param.detach().cpu().numpy().squeeze()

    def plot_feature_tuning(self, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(9, 4), width_ratios=[2, 1])
            fig.subplots_adjust(wspace=0.5)
        sa_tuning = self.get_feature_tuning("state_action")
        dist_tuning = self.get_feature_tuning("distance")
        dist_tuning = dist_tuning[self.dist_mask]
        sa_min = sa_tuning.min()
        sa_max = sa_tuning.max()
        if sa_min < 0 and sa_max < 0:
            cmap = "Reds_r"
            _min = sa_min
            _max = sa_max
        elif sa_min > 0 and sa_max > 0:
            cmap = "Reds"
            _min = sa_min
            _max = sa_max
        else:
            cmap = "bwr"
            _min = min(sa_min, -sa_max)
            _max = max(sa_max, -sa_min)
        sa = pd.Series(index=pd.MultiIndex.from_tuples(self.state_action_idx), data=sa_tuning)
        neg = True if _min.min() < 0 else False
        mp.plot_directed_heatmap(
            self.simple_maze,
            sa,
            colormap=cmap,
            allow_negative=neg,
            fixed_vmin=_min,
            fixed_vmax=_max,
            ax=axes[0],
        )
        # only plot up distances with more with valid occupancy in over 1/3 of states
        x = self.distance_idx.values[self.dist_mask]
        y = dist_tuning
        axes[1].plot(x, y, color="grey", lw=3)
        axes[1].set_xlabel("Distance to Goal (m)")
        axes[1].set_ylabel("Activity")
        axes[1].spines[["top", "right"]].set_visible(False)
        return

    def vis_latent_reconstruction(self, reorder_sa=False):
        """ """
        fig = plt.figure(figsize=(1, 5))  # Tall aspect ratio
        # Define GridSpec with 3 rows and 3 columns (to allow flexible sizing)
        gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[2, 20], width_ratios=[1, 3])
        # Turn off the top-left subplot
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 1])  # Top spanning two columns
        ax3 = fig.add_subplot(gs[1, 0])  # Bottom-left (1/3 width)
        ax4 = fig.add_subplot(gs[0, 1])  # Bottom-right (full width)
        vmin = self.z.min().item()
        vmax = self.z.max().item()
        z_np = self.z_df.values
        z = self.z
        dist_mask = self.dist_mask
        z = z[:, dist_mask]
        sa_tuning = self.get_feature_tuning("state_action")[:, None]
        dist_tuning = self.get_feature_tuning("distance")[None, :]
        dist_tuning = dist_tuning[:, dist_mask]
        if (np.mean(sa_tuning) < 0) or (np.mean(dist_tuning) < 0):
            sa_tuning = -1 * (sa_tuning)
            dist_tuning = -1 * (dist_tuning)
        if reorder_sa:
            distance_vector = pdist(z_np, metric="euclidean")
            Z = linkage(distance_vector, method="average")
            Z_opt = optimal_leaf_ordering(Z, distance_vector)
            sa_order = leaves_list(Z_opt)
            z = z[sa_order, :]
            sa_tuning = sa_tuning[sa_order]
        ax2.imshow(z, aspect="auto", cmap="mako")
        ax3.imshow(
            sa_tuning,
            aspect="auto",
            cmap="mako",
            vmin=vmin,
            vmax=vmax,
        )
        ax3.set_ylabel("State-Action")
        ax4.imshow(
            dist_tuning,
            aspect="auto",
            cmap="mako",
            vmin=vmin,
            vmax=vmax,
        )
        ax4.set_xlabel("Distance to Goal")
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_axis_off()
        fig.tight_layout()
        return

    def get_residuals_df(self):
        """
        returns df similar to z_df
        """
        z_hat = self.forward()
        z = self.z
        residuals = z - z_hat
        return pd.DataFrame(
            index=self.state_action_idx, columns=self.distance_idx, data=residuals.detach().cpu().numpy()
        )

    def plot_marginals_of_residuals(self):
        """ """
        residuals_df = self.get_residuals_df()
        sa_maringal, dist_marginal = pl._get_marginals(residuals_df.stack())
        dist_marginal = dist_marginal[self.dist_mask]
        pl.plot_marginals(sa_maringal, dist_marginal, self.simple_maze, place_direction_cmap="Greys")
        return


class MultiplicativePlusAdditiveModel(_Model):
    """ """

    def __init__(self, z_df, kwargs):
        super(MultiplicativePlusAdditiveModel, self).__init__(z_df, kwargs)

    def forward(self):
        return (self.sa_param * self.dist_param) + (self.sa_param + self.dist_param)


class AdditiveModel(_Model):
    """
    Predicts latent as the sum of two tuning curves:
      L_hat(sa, d) = a[sa] + b[d]
    where "sa" indexes state-action and "d" indexes distance.
    """

    def __init__(self, z_df, kwargs):
        super(AdditiveModel, self).__init__(z_df, kwargs)
        # The inherited self.sa_param and self.dist_param are used as is.

    def forward(self):
        # Broadcasting: (n_state_action, 1) + (1, n_distance) produces (n_state_action, n_distance)
        return self.sa_param + self.dist_param


class MultiplicativeModel(_Model):
    """
    Predicts latent as the product of two tuning curves:
      L_hat(sa, d) = a[sa] * b[d] + c
    """

    def __init__(self, z_df, kwargs):
        super(MultiplicativeModel, self).__init__(z_df, kwargs)
        # init multiplicative constant
        self.multiplicative_constant = nn.Parameter(torch.zeros(1, 1))

    def forward(self):
        return self.sa_param * self.dist_param + self.multiplicative_constant  # Broadcasting multiplication


class ConstantModel_StateAction(_Model):
    """
    Assumes no state-action tuning: the latent only depends on distance.
      L_hat(sa, d) = c[d]
    In this model, the state-action tuning is replaced by a constant value (ones).
    """

    def __init__(self, z_df, kwargs):
        super(ConstantModel_StateAction, self).__init__(z_df, kwargs)
        # Override state-action tuning: set to ones (non-learnable)
        self.sa_param = nn.Parameter(torch.ones(self.n_state_action, 1))

    def forward(self):
        # Return the distance tuning expanded across the state-action dimension.
        return self.dist_param.expand(self.n_state_action, -1)


class ConstantModel_Distance(_Model):
    """
    Assumes no distance tuning: the latent only depends on state-action.
      L_hat(sa, d) = c[sa]
    In this model, the distance tuning is replaced by a constant value (ones).
    """

    def __init__(self, z_df, kwargs):
        super(ConstantModel_Distance, self).__init__(z_df, kwargs)
        # Override distance tuning: set to ones (non-learnable)
        self.dist_param = nn.Parameter(torch.ones(1, self.n_distance))

    def forward(self):
        # Return the state-action tuning expanded across the distance dimension.
        return self.sa_param.expand(-1, self.n_distance)


# %%
def permutation_test(model_class, target, occ_mask, n_permutations=100, n_epochs=1000, lr=0.01):
    """
    For a given model type, this function performs a permutation test by
    shuffling the rows (state-action bins) of the target data, refitting the model,
    and collecting the reconstruction errors.

    Parameters:
      model_class    - the class of the model (e.g. AdditiveModel).
      target         - the original target tensor.
      n_permutations - number of shuffles.
      model_args     - tuple of arguments (e.g., (n_sa, n_distance)) to pass when instantiating the model.
      n_epochs       - training epochs for each permutation.
      lr             - learning rate.

    Returns:
      A list of reconstruction errors from each permutation.
    """
    permuted_errors = []
    # Convert target to a NumPy array for shuffling.
    target_np = target.numpy()

    for i in range(n_permutations):
        print(i)
        # Permute along the state-action dimension (rows).
        permuted_target_np = target_np.copy()
        np.random.shuffle(permuted_target_np)  # shuffles rows in-place
        permuted_target = torch.tensor(permuted_target_np, dtype=torch.float32)

        # Instantiate and train a new model on the permuted target.
        model = model_class(*target.shape)
        train_model(model, permuted_target, occ_mask, n_epochs=n_epochs, lr=lr, verbose=False)
        error = final_loss(model, permuted_target, occ_mask)
        permuted_errors.append(error)

    # true error
    model = model_class(*target.shape)
    train_model(model, target, occ_mask, n_epochs=n_epochs, lr=lr, verbose=False)
    true_error = final_loss(model, target, occ_mask)

    return permuted_errors, true_error
