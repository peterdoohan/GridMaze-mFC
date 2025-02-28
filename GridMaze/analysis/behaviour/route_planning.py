"""
Library for analysing and plotting basic metrics from Xiao's latent routes discovery
"""

# %% Imports
from ..core import get_sessions as gs
from ...maze import plotting as mp

import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram

from scipy.ndimage import gaussian_filter1d


from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
import matplotlib as mpl

# %% Global Variables
from ...paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

CUSTOM_COLORMAPS = list(mp.CUSTOM_COLORMAPS.keys())

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

with open(EXPERIMENT_INFO_PATH / "maze_configs.json", "r") as f:
    MAZE_CONFIGS = json.load(f)

FRAME_RATE = 60
# %% Functions


def save_subject_routes_pdf():
    """ """
    save_path = RESULTS_PATH / "subject_routes.pdf"
    with PdfPages(save_path) as pdf:
        for subject_ID in SUBJECT_IDS:
            for maze_name in MAZE_CONFIGS.keys():
                f = plot_subject_routes(subject_ID, maze_name)
                pdf.savefig(f)
    return


def plot_subject_routes(subject_ID, maze_name, day=5):
    """ """
    session = gs.get_maze_sessions(
        subject_IDs=[subject_ID], maze_names=[maze_name], days_on_maze=[day], with_data=["routes_df"]
    )
    simple_maze = session.simple_maze()
    routes_df = session.routes_df
    f, axes = plt.subplots(3, 4, figsize=(20, 16))
    f.suptitle(f"Subject {subject_ID} - {maze_name}")
    f.tight_layout()
    axes = axes.flatten()
    for i, route in enumerate(routes_df.index):
        route_place_direction_tuning = routes_df.loc[route]
        ax = axes[i]
        mp.plot_directed_heatmap(
            simple_maze,
            route_place_direction_tuning,
            colormap=CUSTOM_COLORMAPS[i],
            colorbar=False,
            ax=ax,
            title=f"route_{i}",
            value_label="P(Place,Direction|Route)",
        )
    for ax in axes[len(routes_df) :]:
        ax.axis("off")
    return f


def get_route_similarity_matrix(routes_df, plot=True):
    """ """
    R = routes_df.to_numpy()  # [n_routes, n_place-directions]
    S = cosine_similarity(R)
    if plot:
        _plot_routes_matrix(S)
    return S


def get_route_transitions_matrix(subject, maze, plot=True):
    """"""
    sessions = gs.get_maze_sessions(
        subject_IDs=[subject],
        maze_names=[maze],
        days_on_maze="late",
        with_data=["navigation_routes_df", "trial_info_df"],
    )
    # initialise transition matrix
    T = np.zeros((11, 11))
    for session in sessions:
        navigation_routes_df = session.navigation_routes_df.reset_index(drop=True)
        route_names = sorted(navigation_routes_df.route.r.dropna().unique())
        trial_info_df = session.trial_info_df
        trials = trial_info_df.trial.dropna().unique()
        # define transitions within trials (importantly, not across trial - disjoint in time)
        for trial in trials:
            trial_nav_routes_df = navigation_routes_df[trial_info_df.trial == trial]
            current_route = trial_nav_routes_df.route.r
            route_sequence = current_route[trial_nav_routes_df.route_change == True].to_numpy()
            for i, route in enumerate(route_sequence):
                if i == 0:
                    continue
                previous_route = route_sequence[i - 1]
                T[route_names.index(previous_route), route_names.index(route)] += 1
    # ignore non_route transitions
    T = T[1:, 1:]
    if plot:
        _plot_routes_matrix(T)
    return T


def _plot_routes_matrix(M):
    f, ax = plt.subplots(1, 1, figsize=(3, 3))
    cax = ax.imshow(M, cmap="Greys")
    f.colorbar(cax, ax=ax)
    ax.set_xticks(range(M.shape[0]))
    ax.set_yticks(range(M.shape[1]))
    ax.set_xlabel("route_id")
    ax.set_ylabel("route_id")
    return


def _get_hierarchical_linkage(routes_df):
    route_features = routes_df.to_numpy()  # [n_routes, n_place_directions]
    similarities = cosine_similarity(route_features)
    linked = linkage(similarities, method="average")
    return linked


def plot_route_dendrogram(routes_df, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(1, 3))
    linked = _get_hierarchical_linkage(routes_df)
    dendrogram(
        linked,
        orientation="left",  # Changed orientation to left-to-right
        distance_sort="descending",
        show_leaf_counts=True,
        color_threshold=-1,  # Forces all lines to be in one color
    )
    for i in ax.collections:
        i.set_color("black")
    ax.set_axis_off()
    return


def _get_route_cluster_order(routes_df):
    linked = _get_hierarchical_linkage(routes_df)
    dendro = dendrogram(
        linked,
        no_plot=True,
    )
    return dendro["leaves"]


def get_n_routes_per_trial(session):
    """"""
    trial_info_df = session.trial_info_df
    navigation_routes_df = session.navigation_routes_df.reset_index(drop=True)
    trial2n_routes = {}
    for trial in trial_info_df.trial.dropna().unique():
        current_route = navigation_routes_df[trial_info_df.trial == trial].route.r.dropna()
        route_sequence = current_route[~(current_route == current_route.shift(1))]
        n_routes = len(route_sequence)
        trial2n_routes[trial] = n_routes
    return trial2n_routes


# %% Plotting Routes Trajectories


def _get_route2color(session, colormap="nipy_spectral"):
    """ """
    routes_df = session.routes_df
    route_ids = routes_df.index.to_list() + ["non_route"]
    cmap = plt.get_cmap(colormap, len(route_ids))
    return {route_id: cmap(i) for i, route_id in enumerate(route_ids)}


def add_categorical_colorbar(cat2color, ax, orientation="vertical", aspect_ratio=20, shrink_factor=0.8):
    """
    Creates a colorbar from a dictionary of categories and their associated colors.

    Parameters:
        cat2color (dict): Dictionary where keys are category names and values are RGB triplets.
        ax (matplotlib.axes.Axes): The axis to draw the colorbar on.
        orientation (str): Orientation of the colorbar, either 'horizontal' or 'vertical'.
        aspect_ratio (int): The aspect ratio of the colorbar (default: 20).
        shrink_factor (float): Factor to shrink the colorbar (default: 0.8).
    """
    colors = [rgb for rgb in cat2color.values()]
    categories = [category for category in cat2color.keys()]
    cmap = mpl.colors.ListedColormap(colors)
    bounds = np.arange(len(categories) + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # ScalarMappable needs an array, even if it's not used
    cbar = plt.colorbar(
        sm, ticks=np.arange(len(categories)), ax=ax, aspect=aspect_ratio, orientation=orientation, shrink=shrink_factor
    )
    cbar.set_ticklabels(categories)
    cbar.outline.set_visible(False)
    return cbar


def plot_session_route_colored_trajectories(session):
    """"""
    trials = session.trial_info_df.trial.dropna().unique()
    for t in trials:
        plot_route_colored_trajectory(session, t)


def plot_route_colored_trajectory(session, trial, smooth_SD=10, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(5, 5))
    navigation_df = session.navigation_df
    navigation_routes_df = session.navigation_routes_df
    navigation_routes_df = pd.concat([navigation_df, navigation_routes_df.reset_index(drop=True)], axis=1)
    trial_mask = (navigation_routes_df.trial == trial) & (navigation_routes_df.trial_phase == "navigation")
    trial_df = navigation_routes_df[trial_mask]
    if trial_df.route.r.isna().all():
        print(f"Trial {trial} involved no routes.")
        return
    x_pos = trial_df.centroid_position.x
    y_pos = trial_df.centroid_position.y
    start_pos = trial_df.maze_position.simple.iloc[0]
    goal = trial_df.goal.iloc[-1]
    if smooth_SD:
        x_pos = gaussian_filter1d(x_pos, sigma=smooth_SD)
        y_pos = gaussian_filter1d(y_pos, sigma=smooth_SD)
    route2color = _get_route2color(session)
    route_colors = trial_df.route.r.map(route2color).to_numpy()
    points = np.array([x_pos, y_pos]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    mp.plot_simple_maze_silhouette(
        session.simple_maze(),
        ax,
        color="silver",
        special_location2color={start_pos: "black", goal: "deepskyblue"},
        node_size=350,
        edge_size=8,
    )
    lc = LineCollection(segments, colors=route_colors, linewidth=4)
    ax.add_collection(lc)
    add_categorical_colorbar(route2color, ax)
    ax.text(0, 0, " -> ".join(trial_df.route.r.drop_duplicates().to_numpy()))
    return


# %% Route Change aligned speeds

def get_cross_subject_route_change_aligned_speeds(plot=True):
    """"""
    subject_mean_speeds = []
    for subject in SUBJECT_IDS:
        speeds = get_subject_route_change_aligned_speeds(subject, smooth_SD=0.1, plot=False)
        subject_mean_speeds.append(speeds.mean(axis=0))
    if plot:
        f, ax = plt.subplots(1, 1, figsize=(3, 3))
        mean = np.vstack(subject_mean_speeds).mean(axis=0)
        sem = np.vstack(subject_mean_speeds).std(axis=0) / np.sqrt(len(subject_mean_speeds))
        ax.plot(mean, color="black")
        ax.fill_between(np.arange(len(mean)), mean-sem, mean+sem, color="black", alpha=0.3)
        ax.set_ylabel("Speed (cm/s)")
        ax.set_xlabel("Route Change Aligned Time (s)")
        ax.axvline(len(mean)//2, color="black", linestyle="--", lw=1, alpha=0.8)
        ax.set_xticks([len(mean)//2])
        ax.set_xticklabels([0])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return speeds


def get_subject_route_change_aligned_speeds(subject, smooth_SD=0.1, plot=True):
    """ """
    sessions = gs.get_maze_sessions(subject_IDs=[subject], maze_names="all", days_on_maze="late", with_data=["navigation_df", "navigation_routes_df"])
    session_speeds = [get_session_route_change_aligned_speeds(session) for session in sessions]
    speeds = np.vstack(session_speeds)
    if smooth_SD:
        speeds = gaussian_filter1d(speeds, sigma=smooth_SD*FRAME_RATE, axis=1)
    if plot:
        mean = speeds.mean(axis=0)
        sem = speeds.std(axis=0) / np.sqrt(speeds.shape[0])
        f, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.plot(mean, color="black")
        ax.fill_between(np.arange(len(mean)), mean-sem, mean+sem, color="black", alpha=0.3)
        ax.set_ylabel("Speed (cm/s)")
        ax.set_xlabel("Route Change Aligned Time (s)")
        ax.axvline(len(mean)//2, color="black", linestyle="--", lw=1, alpha=0.8)
        ax.set_xticks([len(mean)//2])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return speeds


def get_session_route_change_aligned_speeds(session, window=(-5,5), optimal=True, max_from_goal=4):
    """ """
    navigation_df = session.navigation_df
    navigation_routes_df = session.navigation_routes_df
    nav_routes_df = pd.concat([navigation_df, navigation_routes_df.reset_index(drop=True)], axis=1)
    route_changes_df = nav_routes_df[nav_routes_df.route_change == 1]
    if optimal:
        route_changes_df = route_changes_df[route_changes_df.optimal_route==1]
    if max_from_goal:
        route_changes_df = route_changes_df[route_changes_df.route_order.from_goal.le(max_from_goal)]
    speeds = []
    for i, _ in route_changes_df.iterrows():
        start_frame = i + (window[0] * FRAME_RATE)
        end_frame = i + (window[1] * FRAME_RATE) - 1
        if start_frame < 0 or end_frame > len(nav_routes_df):
            continue
        route_speeds = nav_routes_df.loc[start_frame:end_frame].speed
        speeds.append(route_speeds.values)
    return np.vstack(speeds)
