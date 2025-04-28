"""
Library for generating distance to goal basis functions for other analyses
@peterdoohan
"""

# %% Imports
from ast import NotIn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm

# %% Global variables

GAMMA_2P_SCALE = {"distance": 0.75, "steps": 1.5}

GAUSSIAN_2P_SCALE = {"distance": 0.75, "steps": 1.5}

# %% Baiss activation generator


def distance_basis_generator(
    basis_values,
    basis="gamma",
    btype="steps",
    normalise=True,
    max_distance=1.8,
    max_steps=20,
    plot=False,
):
    """
    Returns a callable f(x) -> activations of shape (len(x), len(basis_values))
    or, if x is scalar, shape (len(basis_values),).

    - basis_values: shape params for gamma, or centre params for gaussian
    - basis: "gamma" or "gaussian"
    - btype: "steps" (integer‐indexed) or "distance" (continuous)
    - normalise: divide each curve by its peak over the sampled range
    - max_distance, max_steps: for sampling/normalisation grid
    """
    if btype == "steps":
        x_sample = np.linspace(0, max_steps, max_steps + 1)
    elif btype == "distance":
        x_sample = np.linspace(0, max_distance, 1000)
    else:
        raise ValueError(f"Unknown btype {btype!r}")
    if basis == "gamma":
        scale = GAMMA_2P_SCALE[btype]
    elif basis == "gaussian":
        scale = GAUSSIAN_2P_SCALE[btype]
    else:
        NotImplementedError
    # plot
    if plot:
        if basis == "gamma":
            plot_gamma_basis_functions(basis_values, btype=btype, max_distance=max_distance, max_steps=max_steps)
        elif basis == "gaussian":
            plot_gaussian_basis_functions(basis_values, btype=btype, max_distance=max_distance, max_steps=max_steps)
    # precompute normalisation constants
    basis_values = np.array(basis_values)
    if normalise:
        if basis == "gamma":
            norms = np.array([np.nanmax(gamma.pdf(x_sample, a, loc=0, scale=scale)) for a in basis_values])
        else:  # gaussian
            norms = np.array([np.nanmax(norm.pdf(x_sample, loc=mu, scale=scale)) for mu in basis_values])
    else:
        norms = np.ones_like(basis_values, dtype=float)
    # pick the right pdf
    if basis == "gamma":

        def _pdf(x, param):
            return gamma.pdf(x, param, loc=0, scale=scale)

    else:

        def _pdf(x, param):
            return norm.pdf(x, loc=param, scale=scale)

    # build the generator
    def basis_fn(x):
        x_arr = np.atleast_1d(x)
        # compute (len(x_arr), len(basis_values))
        acts = np.stack([_pdf(x_arr, param) / norm_val for param, norm_val in zip(basis_values, norms)], axis=1)
        if np.isscalar(x):
            return acts[0]
        return acts

    return basis_fn


# %% Gamma basis functions


def plot_gamma_basis_functions(
    shape_values, btype="steps", normalise_max=True, max_distance=1.8, max_steps=20, ax=None
):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(5, 3), clear=True)
    if btype == "steps":
        x = np.linspace(0, max_steps, max_steps + 1)
        scale = GAMMA_2P_SCALE["steps"]
    elif btype == "distance":
        x = np.linspace(0, max_distance, 100)
        scale = GAMMA_2P_SCALE["distance"]
    else:
        NotImplementedError
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.2, 1, len(shape_values)))
    for shape, color in zip(shape_values, colors):
        a = gamma_function(x, shape, scale)
        if normalise_max:
            a[a == np.inf] = np.nan
            a = a / np.nanmax(a)
        ax.plot(x, a, label=f"shape={shape}", color=color, lw=3)
        if normalise_max:
            ax.set_ylim(0, 1.1)
        # ax.legend()
        ax.set_xlabel(f"{btype} to Goal")
        ax.set_ylabel("Activity")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def get_gamma_basis_shape_params(n, btype="steps"):
    """
    Add defualt for distance too!
    """
    if btype == "steps":
        return np.arange(1, n + 1, 1) ** 2
    else:
        NotImplementedError


def gamma_function(x, shape, scale=None):
    if scale is None:
        scale = GAMMA_2P_SCALE["distance"]
    return gamma.pdf(x, shape, loc=0, scale=scale)


# %% Gaussian basis functions


def get_gaussian_basis_centres(n, dtype="steps", max_distance=1.8, max_steps=20):
    """
    Return n centres evenly spaced over either step‐indices or real distances.
    """
    if dtype == "steps":
        # n centres between 0 and max_steps inclusive
        return np.linspace(0, max_steps, n)
    elif dtype == "distance":
        # n centres between 0 and max_distance
        return np.linspace(0, max_distance, n)
    else:
        raise NotImplementedError(f"Unknown dtype {dtype}")


def gaussian_function(x, centre, scale=None):
    """
    A 2-parameter Gaussian (normal) pdf at x, with mean=`centre` and stdev=`scale`.
    """
    if scale is None:
        scale = GAUSSIAN_2P_SCALE["distance"]
    return norm.pdf(x, loc=centre, scale=scale)


def plot_gaussian_basis_functions(
    centre_values,
    btype="steps",
    normalise_max=True,
    max_distance=1.8,
    max_steps=20,
    ax=None,
):
    """
    Plot one Gaussian pdf per `centre_values[i]`, sampling either in integer steps
    or in a dense real‐valued grid.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3), clear=True)

    if btype == "steps":
        x = np.linspace(0, max_steps, max_steps + 1)
        scale = GAUSSIAN_2P_SCALE["steps"]
    elif btype == "distance":
        x = np.linspace(0, max_distance, 1000)
        scale = GAUSSIAN_2P_SCALE["distance"]
    else:
        raise NotImplementedError(f"Unknown btype {btype}")

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.2, 1, len(centre_values)))
    for centre, color in zip(centre_values, colors):
        y = gaussian_function(x, centre, scale)
        if normalise_max:
            y[y == np.inf] = np.nan
            y = y / np.nanmax(y)
        ax.plot(x, y, label=f"μ={centre:.2f}", color=color, lw=3)

    ax.set_xlabel(f"{btype} to Goal")
    ax.set_ylabel("Activity")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if normalise_max:
        ax.set_ylim(0, 1.1)
    return ax
