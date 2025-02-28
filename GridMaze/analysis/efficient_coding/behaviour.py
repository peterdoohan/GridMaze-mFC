""" 
Library for generating behavioural data matricies/tensors for efficient coding analysis, relating neurins to behaviour
"""

# %% Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.distance_to_goal import distributions as dd
from GridMaze.maze import representations as mr
from GridMaze.maze import plotting as mp

from sklearn.decomposition import NMF, PCA

# %% Global Variables
NMF_KWARGS = {
    "init": "random",
    "random_state": 0,
    "solver": "mu",
    "beta_loss": "kullback-leibler",
    "max_iter": 1000,
}


# %% Variance explained
def get_pca_variance_explained(A, B):  # A & B: [n_samples, n_features]
    """Calculates the cumulative variance of matrix B that's explained by the first i components of matrix A."""
    model = PCA(random_state=0)
    model.fit(A)
    M = model.transform(B)  # [n_samples, n_components]
    pc_exp_var = np.square(M).sum(axis=0)
    cumsum_exp_var = np.cumsum(pc_exp_var) / pc_exp_var.sum()
    return cumsum_exp_var


# %% Dimensionality reduction


def _plot_place_direction_distance(pd_marginal, dist_marginal, maze, axes=None):
    """ """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(8, 5), width_ratios=[2, 1])
    # plot place-direction heatmap
    pd_tuning = pd_marginal.copy()
    pd_tuning.index = pd.MultiIndex.from_tuples(pd_tuning.index)
    simple_maze = mr.get_simple_maze(maze)
    mp.plot_directed_heatmap(simple_maze, pd_tuning, ax=axes[0], colormap="silver2blue", colorbar=False)
    # plot distance curve
    x = dist_marginal.index.values
    y = dist_marginal.values
    axes[1].plot(x, y, color="darkblue", lw=3)
    axes[1].set_xlabel("Distance to Goal")
    axes[1].set_ylabel("Norm. Loading")
    axes[1].spines[["top", "right"]].set_visible(False)
    if axes is None:
        fig.tight_layout()


def plot_component_marginals(maze, occupancy_normalised=True):
    """ """
    PDD_df = get_place_direction_distance_matrix(subject_IDs="all", maze=maze, return_as="df")
    mean_occupancy = PDD_df.mean(axis=1)
    mean_occupancy = mean_occupancy / mean_occupancy.sum()  # Normalise length 1
    NMF_df = get_place_direction_distance_NMF(PDD_df, n_components=10)
    n_components = NMF_df.shape[1]
    f, axes = plt.subplots(n_components, 2, figsize=(9, 4 * n_components), width_ratios=[2, 1])
    for i in range(n_components):
        c = NMF_df[i]
        c = c / c.sum()  # Normalize length 1
        if occupancy_normalised:
            c = c.mul(mean_occupancy)
        dist_marginal = c.groupby(level="distance_to_goal").sum()
        pd_marginal = c.groupby(level="place_direction").sum()
        if occupancy_normalised:
            dist_marg_occ = mean_occupancy.groupby(level="distance_to_goal").sum()
            dist_marginal = dist_marginal.div(dist_marg_occ)
            pd_marg_occ = mean_occupancy.groupby(level="place_direction").sum()
            pd_marginal = pd_marginal.div(pd_marg_occ)

        _plot_place_direction_distance(pd_marginal, dist_marginal, maze, axes[i])
    f.tight_layout()


def get_place_direction_distance_NMF(PDD_df, n_components=10):
    """ """
    # NMF
    model = NMF(
        n_components=n_components,
        # alpha_W=1e-4,
        # l1_ratio=1,
        max_iter=2_000,
    )  # **NMF_KWARGS)
    W = model.fit_transform(PDD_df.values)
    NMF_df = pd.DataFrame(W, index=PDD_df.index)
    return NMF_df


# %% Build data structures


def get_place_direction_distance_matrix(
    subject_IDs=["m2"],
    maze="maze_1",
    distance_metric="geodesic",
    distance_bin_method="non-uniform",
    n_distance_bins=10,
    max_steps_to_goal=70,  # defined over nodes and edges
    max_distance=None,
    return_as="df",
):
    """
    Inputs:
        subject_IDs (list): list of subject IDs
        maze (str): name of maze ("maze_1", "maze_2", "rooms_maze")
        distance_metric (str): either "steps" (still defined geodesic, not future) or "geodesic"
        distance_bin_method (str): either "non-uniform" or "uniform"
        n_distance_bins (int): number of distance bins
        max_distance (float): maximum distance to consider
    """
    #
    if distance_metric == "geodesic":
        _metric = "geodesic_distance_to_goal"
    elif distance_metric == "steps":
        _metric = "steps_to_goal"
    else:
        NotImplementedError()
    # load data
    sessions = gs.get_maze_sessions(
        subject_IDs=subject_IDs,
        maze_names=[maze],
        days_on_maze="late",
        with_data=["trajectory_decisions_df"],
        must_have_data=True,
    )
    # place-direction & distance bin mappings
    place_directions = mr.get_maze_place_direction_pairs(mr.get_simple_maze(maze))
    pd2ind = {pd: i for i, pd in enumerate(place_directions)}
    distance_bins = get_distance_bins(distance_metric, distance_bin_method, n_distance_bins, max_distance)
    dist2ind = {d: i for i, d in enumerate(distance_bins)}
    PDDs = []  # place-direction-distance arrays
    for session in sessions:
        nav_decisions_df = session.trajectory_decisions_df
        distance_bins = get_distance_bins(
            distance_metric,
            distance_bin_method,
            n_distance_bins,
            max_distance,
        )
        # return nav_decisions_df, _metric, distance_bins, max_distance, max_steps_to_goal, pd2ind, dist2ind
        PDDs.extend(
            _get_place_direction_distance_array(
                nav_decisions_df, _metric, distance_bins, max_steps_to_goal, pd2ind, dist2ind
            )
        )
    PDD = np.array(PDDs)  # [trials, place_direction, distance]
    n_trials, n_pd, n_dist = PDD.shape
    # return as requested
    if return_as == "tensor":
        return PDD.reshape(n_pd, n_dist, n_trials)  # [place_direction, distance, trials]
    if return_as in ["matrix", "df"]:
        _PDD = PDD.reshape(n_trials, n_pd * n_dist)
        if return_as == "matrix":
            return _PDD.T  # [place_direction*distance, trials]
        else:
            columns = pd.MultiIndex.from_product(
                [
                    place_directions,
                    [d.mid for d in distance_bins],
                ],
                names=[
                    "place_direction",
                    "distance_to_goal",
                ],
            )
            df = pd.DataFrame(_PDD, columns=columns)
            return df.T  # [place_direction*distance, trials]


def _get_place_direction_distance_array(
    nav_decisions_df, distance_metric, distance_bins, max_steps_to_goal, pd2ind, dist2ind
):
    """ """
    # filter for navigation
    nav_decisions_df = nav_decisions_df[nav_decisions_df.trial_phase == "navigation"].copy()
    nav_decisions_df = nav_decisions_df[nav_decisions_df.steps_to_goal.lt(max_steps_to_goal)]
    # add distance_bin & place_direction indices
    nav_decisions_df.loc[:, "distance_bin"] = pd.cut(nav_decisions_df[distance_metric], bins=distance_bins)
    nav_decisions_df.loc[:, "distance_bin_ind"] = nav_decisions_df.distance_bin.map(dist2ind)
    nav_decisions_df.loc[:, "place_direction_ind"] = (
        pd.Series(zip(nav_decisions_df.maze_position, nav_decisions_df.action)).map(pd2ind).astype(int).values
    )
    # build place_direction_distance arrays looping over trials
    PDDs = []
    for trial in nav_decisions_df.trial.unique():
        trial_df = nav_decisions_df[nav_decisions_df.trial == trial]
        # filter max distance
        PDD = np.zeros((len(pd2ind), len(dist2ind)))
        for i, row in trial_df.iterrows():
            if np.isnan(row.distance_bin_ind):
                continue
            PDD[int(row.place_direction_ind), int(row.distance_bin_ind)] += 1
        PDDs.append(PDD)
    return PDDs


# %% Supporting functions


def get_distance_bins(distance_metric, binning_method, n_bins, max_dist):
    """
    Returns pd.IntervalIndex for binning distances to goal
    """
    if distance_metric == "geodesic":
        if max_dist is None:
            max_dist = dd.get_distance_percentile(("distance_to_goal", "geodesic"), 1)
        if binning_method == "uniform":
            distance_bins = pd.interval_range(start=0, end=max_dist, freq=max_dist / n_bins, closed="left")
        elif binning_method == "non-uniform":
            bin_edges = dd.bin_distribution_evenly(("distance_to_goal", "geodesic"), n_bins, max_distance=max_dist)
            distance_bins = pd.IntervalIndex.from_breaks(bin_edges, closed="left")
    elif distance_metric == "steps":
        if max_dist is None:
            max_dist = int(dd.get_distance_percentile(("steps_to_goal", "future"), 1))
        if binning_method == "uniform":
            assert max_dist == n_bins, "n_bins must equal max_dist for uniform bins"
            distance_bins = pd.interval_range(start=0, end=max_dist, freq=1, closed="left")
        if binning_method == "non-uniform":
            bin_edges = dd.bin_distribution_evenly(("steps_to_goal", "future"), n_bins, max_distance=max_dist)
            distance_bins = pd.IntervalIndex.from_breaks(bin_edges, closed="left")
    return distance_bins
