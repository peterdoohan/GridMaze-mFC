"""This script if for testing if the spatail periodicity of some cells is greater than expected
from the autocorrelation inherant to neural firing rates and behaviour."""
# %% Imports
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..cluster_tuning import spatial_periodicity as sp
from . import modularity as spm
from .. import load_permuted_data as lpd
from ...maze import representations as mr
from ...maze import plotting as mp
from .. import get_sessions as gs
from sklearn.metrics import r2_score
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay

# %% Global varaibles
RESULTS_PATH = "../results/cluster_spatial_periodicity_summary_plots"


# %% Functions
def get_analysis_sessions(maze_number):
    sessions = gs.get_sessions(
        subject_IDs="all",
        maze_number=[maze_number],
        day_on_maze="late",
        with_data=["spatial_periodicity_df", "place_rates_df"],
    )
    return sessions


def get_spatialy_periodic_clusters(maze_number):
    maze_periodicity_df = spm.get_multisession_maze_periodicity_df(maze_number)
    filtered_df = spm.filter_maze_periodicity_df(
        maze_periodicity_df,
        fit_r2_cutoff=0.95,
        spatial_correlation_cutoff=0.4,
        freq_range=(0.4, 2),
        amp_range=(0.1, 0.5),
    )
    return filtered_df.index.to_numpy()


def get_cluster_spatial_periodicity_permutation_test_summary(
    maze_number, max_distance=20
):
    sessions = get_analysis_sessions(maze_number)
    tuned_clusters = get_spatialy_periodic_clusters(maze_number)
    spatial_correlation_df = pd.concat(
        [s.spatial_periodicity_df for s in sessions], axis=0
    )
    spatial_correlation_df = spatial_correlation_df.loc[tuned_clusters]
    place_rates_df = pd.concat([s.place_rates_df for s in sessions], axis=0)
    place_rates_df = place_rates_df.loc[tuned_clusters]
    simple_maze = mr.get_simple_maze(maze_number)
    permuted_place_rate_dfs = lpd.load_permuted_dataset(
        "permuted_place_rate_dfs", maze_number, n_permuted="all"
    )
    permuted_place_rate_dfs = [df.loc[tuned_clusters] for df in permuted_place_rate_dfs]
    # rearange such that we get a list of dfs where each df contains all permutations for a single cluster
    cluster2permutation_df = {
        c: pd.concat(
            [df.loc[c] for df in permuted_place_rate_dfs], axis=1
        ).T.reset_index(drop=True)
        for c in tuned_clusters
    }
    # loop over tuned clusters
    cluster2p_value = {}
    for cluster, df in cluster2permutation_df.items():
        cluster_spatial_periodicity = spatial_correlation_df.loc[cluster]
        cluster_place_rates = place_rates_df.loc[cluster]
        distance_correlations_df = pd.DataFrame(
            columns=pd.MultiIndex.from_product(
                [["distance_correlations"], np.arange(1, max_distance + 1)]
            )
        )
        fit_params_df = pd.DataFrame(
            columns=pd.MultiIndex.from_product(
                [
                    ["fit_params"],
                    [
                        "exp_scale",
                        "exp_length",
                        "sin_scale",
                        "freq",
                        "phase",
                        "offest",
                        "r2",
                    ],
                ]
            )
        )
        for permutation in range(len(df.index)):
            place2rate = df.iloc[permutation].to_dict()
            maze_distance_correlations = sp.get_maze_distance_correlations(
                place2rate, simple_maze
            )
            av_fit, fit_params = sp.fit_oscilating_exponential_decay(
                maze_distance_correlations, return_fit=True, return_params=True
            )
            r2 = r2_score(
                cluster_spatial_periodicity.distance_correlations.to_numpy(), av_fit
            )
            distance_correlations_df.loc[permutation] = maze_distance_correlations
            fit_params_df.loc[permutation] = np.append(fit_params, r2)
        cluster_permuted_spatial_periodicity_df = pd.concat(
            [distance_correlations_df, fit_params_df], axis=1
        )
        p_value = plot_permutation_fit_summary(
            cluster,
            simple_maze,
            cluster_place_rates,
            cluster_spatial_periodicity,
            cluster_permuted_spatial_periodicity_df,
            save_fig=True,
        )
        cluster2p_value[cluster] = p_value
        print(f"Finished saving {cluster}")
    return cluster2p_value


# %% Plot summary


def plot_permutation_fit_summary(
    cluster,
    simple_maze,
    cluster_place_rates,
    cluster_spatial_periodicity,
    cluster_permuted_spatial_periodicity_df,
    n_closest_permutations=3,
    save_fig=False,
):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))
    f.tight_layout()
    ax3.axis("off")
    f.tight_layout()
    f.subplots_adjust(wspace=0.4)
    # plot rates heatmap
    mp.plot_simple_heatmap(
        simple_maze,
        cluster_place_rates.to_dict(),
        ax1,
        title=cluster,
        value_label="Firing Rate (Hz)",
        node_size=400,
    )
    # plot top
    # plot closest 3 permutations to the real data
    true_params = cluster_spatial_periodicity.fit_params[
        ["freq", "sin_scale", "r2"]
    ].to_numpy()
    permuted_params = cluster_permuted_spatial_periodicity_df.fit_params[
        ["freq", "sin_scale", "r2"]
    ].to_numpy()
    cluster_permuted_spatial_periodicity_df[("distance_to_true", "")] = [
        euclidean(true_params, p) for p in permuted_params
    ]
    closest_permutations = (
        cluster_permuted_spatial_periodicity_df.sort_values(by="distance_to_true")
        .iloc[0:n_closest_permutations]
        .distance_correlations
    )
    distances = closest_permutations.columns.to_numpy(dtype=int)
    permuted_tuning_curves = closest_permutations.to_numpy()
    for p_curve in permuted_tuning_curves:
        ax2.plot(distances, p_curve, color="k", ls="-", lw=0.2, alpha=0.8)
    ax2.plot(
        cluster_spatial_periodicity.distance_correlations, color="blue", ls="-", lw=2
    )
    true_fit = sp.oscilatting_exponential_decay(
        distances, *cluster_spatial_periodicity.fit_params.to_numpy()[:-1]
    )
    ax2.plot(distances, true_fit, color="red", ls="-", lw=2)
    ax2.set_xlabel("Maze Distance")
    ax2.set_xlim(1, 20)
    ax2.set_ylim(-1, 1)
    ax2.set_ylabel("Correlation")
    ax2.axhline(0, color="k", ls="--", lw=1)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    # plot amp and freq
    ax3 = f.add_subplot(1, 3, 3, projection="3d")
    permuted_fit_params = cluster_permuted_spatial_periodicity_df.fit_params
    null_distribution = permuted_fit_params[
        [("freq"), ("sin_scale"), ("r2")]
    ].to_numpy()
    real_data = cluster_spatial_periodicity.fit_params[
        ["freq", "sin_scale", "r2"]
    ].to_numpy()
    p_value, above_last_front, below_last_front = get_pareto_p_value_3d(
        null_distribution, real_data, return_points=True
    )
    for points, color in zip([below_last_front, above_last_front], ["black", "red"]):
        if points.shape[0] > 1:
            ax3.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=color,
                alpha=0.05,
            )
    ax3.scatter(
        cluster_spatial_periodicity.fit_params.freq,
        cluster_spatial_periodicity.fit_params.sin_scale,
        cluster_spatial_periodicity.fit_params.r2,
        color="blue",
        s=50,
    )
    ax3.set_xlabel("Amp")
    ax3.set_ylabel("Freq")
    ax3.set_zlabel("r2")
    ax3.set_title(f"p = {p_value:.3f}")
    ax3.view_init(elev=20.0, azim=125)
    if save_fig:
        f.savefig(os.path.join(RESULTS_PATH, cluster + ".pdf"), format="pdf")
    return p_value


# %% modified for 3D pareto fronts
def get_pareto_p_value_3d(null_distribution, real_data, return_points=False):
    """Compute the p-value for a real 3D data point given a null distribution and
    return the points above and below the last Pareto front before the real data is crossed.
    """
    fronts = get_pareto_fronts_3d(null_distribution)
    total_points = len(null_distribution)
    points_incorporated = 0
    last_front_before_real_data = None
    points_above_last_front = []
    points_below_last_front = []

    # Check each front for dominance over the real data
    for front in fronts:
        dominated = False
        for point in front:
            # Checking dominance in 3D
            if (
                point[0] <= real_data[0]
                and point[1] <= real_data[1]
                and point[2] <= real_data[2]
            ):
                dominated = True
                break
        # If real_data is dominated, break out of the loop
        if dominated:
            last_front_before_real_data = front
            break
        points_incorporated += len(front)

    # Identify points above and below the last front
    if last_front_before_real_data is not None:
        for point in null_distribution:
            if all(point <= last_front_before_real_data.max(axis=0)):
                points_below_last_front.append(point)
            else:
                points_above_last_front.append(point)

    p_value = points_incorporated / total_points

    if not return_points:
        return p_value
    else:
        return (
            p_value,
            np.array(points_above_last_front),
            np.array(points_below_last_front),
        )


def get_pareto_fronts_3d(points):
    """Calculate the Pareto front of a set of 3D points."""
    fronts = []
    remaining_points = np.array(points, copy=True)
    while len(remaining_points) > 0:
        front = []
        to_remove = []
        for i, point_i in enumerate(remaining_points):
            dominated = False
            for j, point_j in enumerate(remaining_points):
                if i != j and (
                    point_j[0] >= point_i[0]
                    and point_j[1] >= point_i[1]
                    and point_j[2] >= point_i[2]
                ):
                    dominated = True
                    break
            if not dominated:
                front.append(point_i)
                to_remove.append(i)
        remaining_points = np.delete(remaining_points, to_remove, axis=0)
        fronts.append(np.array(front))
    return fronts


def sort_by_nearest_neighbors(points):
    if len(points) == 0:
        return points
    sorted_points = [points[0]]
    remaining_points = set(range(1, len(points)))
    while remaining_points:
        last_point = sorted_points[-1]
        distances = cdist([last_point], points[list(remaining_points)])
        nearest_point_index = list(remaining_points)[np.argmin(distances)]
        sorted_points.append(points[nearest_point_index])
        remaining_points.remove(nearest_point_index)
    return np.array(sorted_points)
