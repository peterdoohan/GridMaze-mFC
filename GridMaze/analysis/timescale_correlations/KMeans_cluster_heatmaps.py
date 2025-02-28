"""
This script contains functions to use KMeans clustering to cluster patterns in neural activity at each
timepoint during navigation/session with different amounts of neural smoothing and visualise what these clusters
look like on a place-direction heatmap. 
"""

# %% Imports
import os
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy.ndimage import gaussian_filter1d
from ...maze import plotting as mp
from ...maze import representations as mr
from sklearn.decomposition import PCA
from .. import get_sessions as gs
from scipy.spatial import KDTree
from collections import Counter
import seaborn as sns
from scipy.stats import pearsonr, ttest_1samp, ttest_rel
from ..time_aligned.lagged_pc_correlations import get_frequency_decomposition
from matplotlib.collections import LineCollection
from ..place_direction import plot_components as pc

# %% Global variables
FRAME_RATE = 60
SAVE_PATH = "../results/KMeans_cluster_trajectories"
SMALL_CONSTANT = 1e-10
# %% Build KMeans data structures


def get_analysis_sessions(subject_ID):
    if not subject_ID == "all":
        subject_ID = [subject_ID]
    sessions = gs.get_sessions(
        subject_IDs=subject_ID,
        maze_number="all",
        day_on_maze="late",
        with_data=["navigation_df", "navigation_spike_rates_df", "cluster_metrics"],
    )
    return sessions


def get_navigation_KMcluster_df(
    session,
    n_clusters=10,
    timescale="both",
    long_navigation_only=True,
    long_moving=None,
    long_exclude_time_at_goal=True,
    long_smooth_SD=120,
    long_reward_consumption_control=False,
    short_navigation_only=False,
    short_moving=None,
    short_exclude_time_at_goal=True,
):
    """"""
    if not timescale in ["short", "long", "both"]:
        raise ValueError("timescale must be 'short', 'long or 'both'")
    navigation_rates_df = session.get_navigation_activity_df(activity_type="firing_rate", cluster_type="good")
    rates_df = navigation_rates_df.firing_rate
    smooth_rates_df = pd.DataFrame(  # add smoothed rates to nav df for long timescale clustering
        index=rates_df.index,
        columns=pd.MultiIndex.from_product([["smoothed_firing_rate"], rates_df.columns]),
        data=gaussian_filter1d(rates_df.to_numpy(), sigma=long_smooth_SD, axis=0),
    )
    navigation_rates_df = navigation_rates_df.join(smooth_rates_df)
    if not long_reward_consumption_control:
        long_rates_df = filter_navigation_rates_df(
            navigation_rates_df,
            long_navigation_only,
            long_moving,
            long_exclude_time_at_goal,
        ).copy()
    else:
        long_rates_df = navigation_rates_df[navigation_rates_df.trial_phase == "reward_consumption"].copy()
    long_rates = long_rates_df.smoothed_firing_rate.to_numpy()
    model = KMeans(n_clusters=n_clusters, random_state=0)
    model = model.fit(long_rates)  # train on long timescale data
    if timescale in ["long", "both"]:
        long_timescale_labels = model.predict(long_rates)
        long_rates_df[("KMeans_cluster", "long")] = long_timescale_labels
        long_rates_df = _drop_all_rates(long_rates_df)
    if timescale in ["short", "both"]:
        short_rates_df = filter_navigation_rates_df(
            navigation_rates_df,
            short_navigation_only,
            short_moving,
            short_exclude_time_at_goal,
        ).copy()
        short_rates = short_rates_df.firing_rate.to_numpy()
        short_timescale_labels = model.predict(short_rates)
        short_rates_df[("KMeans_cluster", "short")] = short_timescale_labels
        short_rates_df = _drop_all_rates(short_rates_df)
    if timescale == "long":
        return long_rates_df
    elif timescale == "short":
        return short_rates_df
    else:
        return long_rates_df, short_rates_df


def get_KMcluster_place_direction_df(KMeans_cluster_df, simple_maze, n_clusters, min_occupancy=0.5):
    """
    Update to use new nav_df function and plot joint distance to goal profile
    """
    place_direction_cols = [
        ("maze_position", "simple"),
        ("cardinal_movement_direction", ""),
    ]
    place_direction_KMcluster_cols = place_direction_cols + [("KMeans_cluster", "long")]
    place_direction_grouped_navigation_df = KMeans_cluster_df.set_index(place_direction_cols).groupby(
        place_direction_cols
    )
    place_direction_total_frames = place_direction_grouped_navigation_df.count().time
    place_direction_KMcluster_grouped_navigation_df = KMeans_cluster_df.set_index(
        place_direction_KMcluster_cols
    ).groupby(place_direction_KMcluster_cols)
    place_direction_KMcluster_counts_df = place_direction_KMcluster_grouped_navigation_df.count().reset_index(
        [("KMeans_cluster", "long")]
    )  # count proportion of time in a given location that a cluster is active
    place_direction_KMcluster_counts_df[("KMeans_cluster_occupancy", "")] = (
        place_direction_KMcluster_counts_df.time.div(place_direction_total_frames)
    )
    place_direction_KMcluster_occupancies = place_direction_KMcluster_counts_df[
        [("KMeans_cluster", "long"), ("KMeans_cluster_occupancy", "")]
    ]
    KMcluster_place_direction_df = pd.DataFrame(index=place_direction_total_frames.index, columns=range(n_clusters))
    KMcluster_place_direction_df.index.names = ["maze_position", "direction"]
    for KMc in range(n_clusters):
        cluster_occ = place_direction_KMcluster_occupancies[
            place_direction_KMcluster_occupancies.KMeans_cluster.long == KMc
        ].KMeans_cluster_occupancy
        KMcluster_place_direction_df[KMc].update(cluster_occ)
    # need to figure out why there are nans in the resulting df, occs sum to 1 so 0 -> nan somehow
    KMcluster_place_direction_df.fillna(0, inplace=True)
    KMcluster_place_direction_df[place_direction_total_frames < FRAME_RATE * min_occupancy] = np.nan
    # add nan values for unvisited locations (for all KMeans clusters)
    visited_place_directions = KMcluster_place_direction_df.index.to_numpy()
    all_place_directions = mr.get_maze_place_direction_pairs(simple_maze)
    unvistied_place_directions = list(set(all_place_directions) - set(visited_place_directions))
    if len(unvistied_place_directions) > 0:
        unvisited_place_direction_nans = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(unvistied_place_directions),
            columns=range(n_clusters),
            data=np.nan,
        )
        KMcluster_place_direction_df = pd.concat([KMcluster_place_direction_df, unvisited_place_direction_nans], axis=0)
    KMcluster_place_direction_df = KMcluster_place_direction_df.reindex(
        sorted(KMcluster_place_direction_df.index), axis=0
    )
    return KMcluster_place_direction_df


def get_KMeans_cluster_distance_to_goal_df(KMeans_cluster_df, n_clusters):
    KMeans_cluster_df["distance_to_goal", "geodesic_binned"] = pd.cut(
        KMeans_cluster_df.distance_to_goal.geodesic, bins=40, include_lowest=True
    )
    distance_bin_total_frames = KMeans_cluster_df.groupby([("distance_to_goal", "geodesic_binned")]).count().time
    cluster_distance_grouped_counts = (
        KMeans_cluster_df.groupby([("KMeans_cluster", "long"), ("distance_to_goal", "geodesic_binned")])
        .count()
        .reset_index([("KMeans_cluster", "long")])
    ).sort_index()
    cluster_distance_grouped_counts[("KMeans_cluster_occupancy", "")] = cluster_distance_grouped_counts.time.div(
        distance_bin_total_frames
    )
    KMcluster_distance_to_goal_df = pd.DataFrame(index=distance_bin_total_frames.index, columns=range(n_clusters))
    KMcluster_distance_to_goal_df.index = [r.mid for r in KMcluster_distance_to_goal_df.index]
    KMcluster_distance_to_goal_df.index.name = "distance_to_goal"
    for cluster in range(n_clusters):
        cluster_occ = cluster_distance_grouped_counts[cluster_distance_grouped_counts.KMeans_cluster.long == cluster][
            ("KMeans_cluster_occupancy", "")
        ]
        KMcluster_distance_to_goal_df[cluster].update(cluster_occ)
    return KMcluster_distance_to_goal_df


def filter_navigation_rates_df(navigation_rates_df, navigation_only, moving, exclude_time_at_goal):
    """"""
    conditions = []
    if navigation_only:
        conditions.append(navigation_rates_df.trial_phase == "navigation")
    if moving is not None:
        if moving:
            conditions.append(navigation_rates_df.moving)
        else:
            conditions.append(~navigation_rates_df.moving)
    if exclude_time_at_goal:
        conditions.append(navigation_rates_df.goal != navigation_rates_df.maze_position.simple)
    if len(conditions) == 0:
        return navigation_rates_df
    else:
        return navigation_rates_df[np.logical_and.reduce(conditions)]


# %% Visualise KMeans clusters (place-direction heatmaps & distance to goal tuning)


def plot_KMeans_cluster_tuning(session, n_clusters=10, rate_smooth_SD=120, distance_smooth_SD=2):
    """ """
    KMeans_cluster_df = get_navigation_KMcluster_df(
        session,
        n_clusters=n_clusters,
        timescale="long",
        long_navigation_only=True,
        long_moving=None,
        long_exclude_time_at_goal=True,
        long_smooth_SD=rate_smooth_SD,
    )
    simple_maze = session.simple_maze()
    KMcluster_place_direction_df = get_KMcluster_place_direction_df(KMeans_cluster_df, simple_maze, n_clusters)
    KMcluster_distance_to_goal_df = get_KMeans_cluster_distance_to_goal_df(KMeans_cluster_df, n_clusters)
    distances = KMcluster_distance_to_goal_df.index.to_numpy()
    place_direction_plotting_dicts = pc._get_nmf_component_plotting_dicts(KMcluster_place_direction_df)
    for i in range(n_clusters):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), clear=True)
        f.tight_layout()
        location2value, location2NSEW = place_direction_plotting_dicts[i]
        mp.plot_simple_star_heatmap(
            simple_maze,
            location2value,
            location2NSEW,
            ax=ax1,
            colormap="Reds",
            title=f"Component {i}",
            value_label="place-direction occupancy",
            silhouette_color="silver",
            silhouette_node_size=250,
            silhouette_edge_size=8,
        )
        distance_tuning = KMcluster_distance_to_goal_df[i].to_numpy(dtype=float)
        if distance_smooth_SD:
            distance_tuning = gaussian_filter1d(distance_tuning, sigma=distance_smooth_SD)
        ax2.plot(distances, distance_tuning, color="darkred", lw=2)
        ax2.set_xlabel("Geodesic Distance to goal (m)")
        ax2.set_ylabel("Cluster Occupancy")
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)

    return


# %% Visual analyses


def get_KMcluster2color(n_clusters, colormap="viridis"):
    cmap = plt.cm.get_cmap(colormap)
    colors = [cmap(i / n_clusters) for i in range(n_clusters)]
    return {i: colors[i] for i in range(n_clusters)}


# %% Node degree analyses


def cluster_transition_analysis(navigation_KMeans_cluster_df, simple_maze):
    """"""
    node_coord2position = nx.get_node_attributes(simple_maze, "position")
    position_xy = navigation_KMeans_cluster_df.centroid_position.to_numpy()
    node_xy = np.array(list(node_coord2position.values()))
    kd_tree = KDTree(node_xy)
    nearest_indices = kd_tree.query(position_xy)[1]
    nearest_nodes = [list(node_coord2position.keys())[index] for index in nearest_indices]
    navigation_KMeans_cluster_df[("nearest_node", "")] = nearest_nodes
    node_coord2degree = dict(simple_maze.degree())
    navigation_KMeans_cluster_df[("nearest_node_degree", "")] = navigation_KMeans_cluster_df.nearest_node.map(
        node_coord2degree
    )
    transition_degrees = []
    vistied_degrees = []
    for trial in navigation_KMeans_cluster_df.trial.unique():
        trial_df = navigation_KMeans_cluster_df[navigation_KMeans_cluster_df.trial == trial]
        cluster_transition_degrees = list(
            trial_df[trial_df.KMeans_cluster.ne(trial_df.KMeans_cluster.shift())][
                1:  # don't count start trial as transition
            ].nearest_node_degree
        )
        transition_degrees.extend(cluster_transition_degrees)
        visted_node_degrees = list(
            trial_df[trial_df.nearest_node.ne(trial_df.nearest_node.shift())].nearest_node_degree
        )
        vistied_degrees.extend(visted_node_degrees)
    transition_degree_counts = Counter(transition_degrees)
    visited_degree_counts = Counter(vistied_degrees)
    node_degrees = [1, 2, 3, 4]
    prob_transitions = [
        (
            transition_degree_counts[degree] / visited_degree_counts[degree]
            if not visited_degree_counts[degree] == 0
            else 0
        )
        for degree in node_degrees
    ]
    f, ax = plt.subplots(1, 1, figsize=(6, 6), clear=True)
    ax.bar(node_degrees, prob_transitions)
    ax.set_ylabel("Probability of transition")
    ax.set_xlabel("Node degree")
    ax.set_xticks(node_degrees)
    return prob_transitions


# make random effects version of this analysis


def get_transition_diffs_vector(M):
    differences = []
    for i in range(M.shape[0]):
        for j in range(i + 1, M.shape[1]):
            diff = M[i, j] - M[j, i]
            differences.append(diff)
    return differences


# %% Refactor correlations analysis function to work with timewindows around different trial times


def _drop_all_rates(navigation_rates_df):
    df = navigation_rates_df.drop(
        columns=pd.concat(
            [
                navigation_rates_df.filter(like="firing_rate", axis=1),
                navigation_rates_df.filter(like="smoothed_firing_rate", axis=1),
            ],
            axis=1,
        )
    )
    return df


def get_KMcluster_transition_timescale_correlation(
    session,
    n_clusters=10,
    window_size=1,
    window=(-10, 10),
    reward_consumption_control=False,
):
    n_windows = (window[1] - window[0]) // window_size
    frames_window_size = int(window_size * FRAME_RATE)
    frames_window = (int(window[0] * FRAME_RATE), int(window[1] * FRAME_RATE))

    long_df, short_df = get_navigation_KMcluster_df(
        session,
        n_clusters=n_clusters,
        timescale="both",
        long_navigation_only=True,  # train clusters on smoothed rates during nav
        long_moving=None,
        long_exclude_time_at_goal=True,
        long_smooth_SD=120,
        long_reward_consumption_control=reward_consumption_control,
        short_navigation_only=False,  # use KM model to predict cluster ids on fast timescale t/o all trial phases
        short_moving=None,
        short_exclude_time_at_goal=False,
    )
    KMclusters = np.arange(n_clusters)
    long_df["KMeans_cluster", "long_shift"] = long_df.KMeans_cluster.long.shift(-1)
    short_df["KMeans_cluster", "short_shift"] = short_df.KMeans_cluster.short.shift(-1)
    long_M = np.zeros((len(KMclusters), len(KMclusters)))
    trials = np.intersect1d(
        long_df.trial.unique(), short_df.trial.unique()
    )  # some trials missing due to filtering contraints
    for trial in trials:
        long_trial_df = long_df[long_df.trial == trial]
        long_transitions_mask = long_trial_df.KMeans_cluster.long.ne(long_trial_df.KMeans_cluster.long_shift)
        long_cluster_transitions = (
            long_trial_df[long_transitions_mask].KMeans_cluster[:-1].to_numpy(dtype=int)
        )  # last transition not valid
        for i, j in long_cluster_transitions:
            long_M[i, j] += 1
    u = (len(KMclusters), len(KMclusters))
    cue_short_Ms = [np.zeros(u) for _ in range(n_windows)]
    reward_short_Ms = [np.zeros((u)) for _ in range(n_windows)]
    for trial in trials:
        short_trial_df = short_df[short_df.trial == trial]
        try:
            cue_frame = short_trial_df[short_trial_df.trial_phase == "navigation"].time.index[0]
            reward_frame = short_trial_df[short_trial_df.trial_phase == "reward_consumption"].time.index[0]
        except IndexError:  # if no fames during navigation or reward consumption skip trial
            continue
        for event_frame, Ms in zip([cue_frame, reward_frame], [cue_short_Ms, reward_short_Ms]):
            windows = generate_windows(event_frame, frames_window_size, frames_window)  # (start_frame, end_frame)
            for i, window in enumerate(windows):
                window_short_df = short_df.loc[
                    window[0] : window[1] + 1
                ]  # indexing with frames so no need to use trial_df
                short_transitions_mask = window_short_df.KMeans_cluster.short.ne(
                    window_short_df.KMeans_cluster.short_shift
                )
                short_cluster_transitions = (
                    window_short_df[short_transitions_mask].KMeans_cluster[:-1].to_numpy(dtype=int)
                )
                for j, k in short_cluster_transitions:
                    Ms[i][j, k] += 1
    # normalise matrices (avoiding div by 0)
    long_M = long_M / corr_matrix_row_sum(long_M)
    cue_short_Ms = [M / corr_matrix_row_sum(M) for M in cue_short_Ms]
    reward_short_Ms = [M / corr_matrix_row_sum(M) for M in reward_short_Ms]
    # # calculate correlations
    # long_diff_vector = get_transition_diffs_vector(long_M)
    # cue_short_diff_vectors = [get_transition_diffs_vector(short_M) for short_M in cue_short_Ms]
    # reward_short_diff_vectors = [get_transition_diffs_vector(short_M) for short_M in reward_short_Ms]
    # cue_corrs = [pearsonr(long_diff_vector, v)[0] for v in cue_short_diff_vectors]
    # reward_corrs = [pearsonr(long_diff_vector, v)[0] for v in reward_short_diff_vectors]

    # new correlations calculation with forward and backward sequences
    cue_corrs, reward_corrs = [], []
    non_diagonal_mask = ~np.eye(long_M.shape[0], dtype=bool)
    nd_long_M = long_M[non_diagonal_mask]
    for short_Ms, corrs in zip([cue_short_Ms, reward_short_Ms], [cue_corrs, reward_corrs]):
        for short_M in short_Ms:
            forward_correlation = pearsonr(nd_long_M, short_M[non_diagonal_mask])[0]
            backward_correlation = pearsonr(nd_long_M, short_M.T[non_diagonal_mask])[0]
            corrs.append((forward_correlation, backward_correlation))
    cue_corrs = np.array(cue_corrs).T
    forward_cue_corrs, backward_cue_corrs = cue_corrs
    reward_corrs = np.array(reward_corrs).T
    forward_reward_corrs, backward_reward_corrs = reward_corrs
    return (
        forward_cue_corrs,
        backward_cue_corrs,
        forward_reward_corrs,
        backward_reward_corrs,
    )


def generate_windows(cue_frame, frame_window_size, frame_window):
    start_frame = cue_frame + frame_window[0]
    end_frame = cue_frame + frame_window[1]
    return [
        (start, min(start + frame_window_size - 1, end_frame))
        for start in range(start_frame, end_frame, frame_window_size)
    ]


def get_transition_diffs_vector(M):
    differences = []
    for i in range(M.shape[0]):
        for j in range(i + 1, M.shape[1]):
            diff = M[i, j] - M[j, i]
            differences.append(diff)
    return differences


def get_event_aligned_timescale_corrs(window_size=1, window=(-10, 10), plot=False, reward_consumption_control=False):
    sessions = get_analysis_sessions(subject_ID="all")
    start, end = window
    window_midpoints = [
        (start + window_size * i + start + window_size * (i + 1)) / 2 for i in range((end - start) // window_size)
    ]
    results_df = pd.DataFrame(
        columns=pd.MultiIndex.from_product((["subject_ID", "maze_number", "day_on_maze"], [""], [""])).append(
            pd.MultiIndex.from_product(
                (
                    ["forward", "backward"],
                    ["cue_aligned", "reward_aligned"],
                    window_midpoints,
                )
            )
        )
    )
    for i, session in enumerate(sessions):
        print(session)
        (
            forward_cue_corrs,
            backward_cue_corrs,
            forward_reward_corrs,
            backward_reward_corrs,
        ) = get_KMcluster_transition_timescale_correlation(
            session,
            window_size=window_size,
            window=window,
            reward_consumption_control=reward_consumption_control,
        )

        results_df.loc[i, ("subject_ID", "", "")] = session.subject_ID
        results_df.loc[i, ("maze_number", "", "")] = session.maze_number
        results_df.loc[i, ("day_on_maze", "", "")] = session.day_on_maze
        for j, midpoint in enumerate(window_midpoints):
            results_df.loc[i, ("forward", "cue_aligned", midpoint)] = forward_cue_corrs[j]
            results_df.loc[i, ("backward", "cue_aligned", midpoint)] = backward_cue_corrs[j]
            results_df.loc[i, ("forward", "reward_aligned", midpoint)] = forward_reward_corrs[j]
            results_df.loc[i, ("backward", "reward_aligned", midpoint)] = backward_reward_corrs[j]
    if plot:
        _plot_cluster_transition_corr_timeseries(results_df)
    return results_df


def _plot_cluster_transition_corr_timeseries(results_df):
    subject_grouped_av = results_df.set_index("subject_ID").groupby("subject_ID").mean()
    max_value = subject_grouped_av.drop(columns=["maze_number", "day_on_maze"], level=0).max().max()
    subject_av = subject_grouped_av.mean()
    subject_sem = subject_grouped_av.sem()
    times = results_df.cue_aligned.columns.to_numpy(dtype=float)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), clear=True, sharey=True)
    f.tight_layout()
    f.subplots_adjust(wspace=0.02)
    subject_av = subject_grouped_av.mean()
    subject_sem = subject_grouped_av.sem()
    for event, ax, color in zip(
        [
            "cue_aligned",
            "reward_aligned",
        ],
        [ax1, ax2],
        ["goldenrod", "deepskyblue"],
    ):
        ax.plot(
            times,
            subject_av[event],
            color=color,
            lw=2,
        )
        ax.fill_between(
            times,
            subject_av[event] - subject_sem[event],
            subject_av[event] + subject_sem[event],
            alpha=0.2,
            color=color,
        )
        event_df = subject_grouped_av[event]
        pre_event_av, post_event_av = [
            event_df[event_df.columns[np.logical_and.reduce([event_df.columns > r[0], event_df.columns < r[1]])]]
            .mean()
            .to_numpy()
            for r in [(-5, 0), (0, 5)]
        ]
        pre_event_mean = np.mean(pre_event_av)
        pre_event_error = np.std(pre_event_av, ddof=1) / np.sqrt(len(pre_event_av))
        post_event_mean = np.mean(post_event_av)
        post_event_error = np.std(post_event_av, ddof=1) / np.sqrt(len(post_event_av))
        for x, mean, error in zip(
            [-2.5, 2.5],
            [pre_event_mean, post_event_mean],
            [pre_event_error, post_event_error],
        ):
            ax.errorbar(
                x=x,
                y=mean,
                yerr=error,
                fmt="_",
                color=color,
                markersize=25,
                markeredgewidth=4,
                elinewidth=2,
            )
        _, p_value = ttest_rel(pre_event_av, post_event_av)
        stars = get_stats_stars(p_value)
        ax.plot([-2.5, 2.5], [max_value, max_value], lw=1.5, color="k")
        ax.text(0, max_value, stars, ha="center", va="bottom", fontsize=12)

    for ax in [ax1, ax2]:
        ax.set_xlim(times[0], times[-1])
        ax.axvline(0, color="k", linestyle="--", lw=1, alpha=0.5)
        # ax.axhline(0, color="k", linestyle="--", lw=1, alpha=0.5)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.legend(
            loc="upper right",
            frameon=False,
            bbox_to_anchor=(1.05, 1.1),
        )
    ax1.set_xlabel("Cue-aligned Time (s)")
    ax2.set_xlabel("Reward-aligned Time (s)")
    ax1.set_ylabel("Correlation of Cluster Transitions \n at short and long timescales")
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    return


def get_stats_stars(pvalue):
    if pvalue < 0.001:
        return "***"
    elif pvalue < 0.01:
        return "**"
    elif pvalue < 0.05:
        return "*"
    else:
        return "ns"


# %%


def corr_matrix_row_sum(M):
    """Row sum for div, avoiding div by 0"""
    return np.where(M.sum(axis=1) == 0, SMALL_CONSTANT, M.sum(axis=1))[:, np.newaxis]


def get_KMcluster_transtion_trial_phase_correlations(session, n_clusters=10, moving_stationary=False):
    long_df, short_df = get_navigation_KMcluster_df(
        session,
        n_clusters=n_clusters,
        timescale="both",
        long_navigation_only=True,
        long_moving=None,
        long_exclude_time_at_goal=True,
        long_smooth_SD=120,
        short_navigation_only=False,
        short_moving=None,
        short_exclude_time_at_goal=True,
    )
    KMclusters = np.arange(n_clusters)
    long_df["KMeans_cluster", "long_shift"] = long_df.KMeans_cluster.long.shift(-1)
    short_df["KMeans_cluster", "short_shift"] = short_df.KMeans_cluster.short.shift(-1)
    long_M = np.zeros((len(KMclusters), len(KMclusters)))
    trials = np.intersect1d(  # some trials missing due to filtering contraints
        long_df.trial.unique(), short_df.trial.unique()
    )
    # long timescale transitions
    for trial in trials:
        long_trial_df = long_df[long_df.trial == trial]
        long_transitions_mask = long_trial_df.KMeans_cluster.long.ne(long_trial_df.KMeans_cluster.long_shift)
        long_cluster_transitions = (
            long_trial_df[long_transitions_mask].KMeans_cluster[:-1].to_numpy(dtype=int)
        )  # last transition not valid
        for i, j in long_cluster_transitions:
            long_M[i, j] += 1
    # short timescale transitions
    u = (len(KMclusters), len(KMclusters))
    nav_short_M, rc_short_M, ITI_short_M = np.zeros(u), np.zeros(u), np.zeros(u)
    moving_nav_short_M, stationary_nav_short_M = np.zeros(u), np.zeros(u)  # reward consumption not split by moving
    moving_ITI_short_M, stationary_ITI_short_M = np.zeros(u), np.zeros(u)
    for trial_phase, short_M, moving_M, stationary_M in zip(
        ["navigation", "reward_consumption", "ITI"],
        [nav_short_M, rc_short_M, ITI_short_M],
        [moving_nav_short_M, None, moving_ITI_short_M],
        [stationary_nav_short_M, None, stationary_ITI_short_M],
    ):
        for trial in trials:
            short_trial_df = short_df[short_df.trial == trial]
            short_transitions_mask = short_trial_df.KMeans_cluster.short.ne(short_trial_df.KMeans_cluster.short_shift)
            trial_phase_mask = short_trial_df.trial_phase == trial_phase
            cluster_transitions = (
                short_trial_df[np.logical_and.reduce([short_transitions_mask, trial_phase_mask])]
                .KMeans_cluster[:-1]
                .to_numpy(dtype=int)
            )
            for i, j in cluster_transitions:
                short_M[i, j] += 1
            if moving_stationary:
                if trial_phase == "reward_consumption":
                    continue
                moving_transitions, stationary_transitions = [
                    short_trial_df[np.logical_and.reduce([short_transitions_mask, trial_phase_mask, mask])]
                    .KMeans_cluster[:-1]
                    .to_numpy(dtype=int)
                    for mask in [short_trial_df.moving, ~short_trial_df.moving]
                ]
                for transitions, M in zip(
                    [moving_transitions, stationary_transitions],
                    [moving_M, stationary_M],
                ):
                    for i, j in transitions:
                        M[i, j] += 1
    # normalise matrices (avoiding div by 0)
    long_M = long_M / corr_matrix_row_sum(long_M)
    nav_short_M = nav_short_M / corr_matrix_row_sum(nav_short_M)
    rc_short_M = rc_short_M / corr_matrix_row_sum(rc_short_M)
    ITI_short_M = ITI_short_M / corr_matrix_row_sum(ITI_short_M)
    if moving_stationary:
        moving_nav_short_M = moving_nav_short_M / corr_matrix_row_sum(moving_nav_short_M)
        stationary_nav_short_M = stationary_nav_short_M / corr_matrix_row_sum(stationary_nav_short_M)
        moving_ITI_short_M = moving_ITI_short_M / corr_matrix_row_sum(moving_ITI_short_M)
        stationary_ITI_short_M = stationary_ITI_short_M / corr_matrix_row_sum(stationary_ITI_short_M)
    # calculate correlations
    long_diff_vector = get_transition_diffs_vector(long_M)
    trial_phase_corrs = []
    for trial_phase_M in [nav_short_M, rc_short_M, ITI_short_M]:
        short_diff_vector = get_transition_diffs_vector(trial_phase_M)
        trial_phase_corrs.append(pearsonr(long_diff_vector, short_diff_vector)[0])
    if moving_stationary:
        ms_corrs = []
        for ms_M in [
            moving_nav_short_M,
            stationary_nav_short_M,
            moving_ITI_short_M,
            stationary_ITI_short_M,
        ]:
            short_diff_vector = get_transition_diffs_vector(ms_M)
            ms_corrs.append(pearsonr(long_diff_vector, short_diff_vector)[0])
        return tuple(
            trial_phase_corrs + ms_corrs
        )  # nav, rc, ITI, moving_nav, stationary_nav, moving_ITI, stationary_ITI
    else:
        return tuple(trial_phase_corrs)  # nav, rc, ITI


def get_trial_phase_corrs_df():
    sessions = get_analysis_sessions(subject_ID="all")
    results_df = pd.DataFrame(
        columns=pd.MultiIndex.from_product((["subject_ID", "maze_number", "day_on_maze"], [""])).append(
            pd.MultiIndex.from_product(
                (
                    ["navigation", "reward_consumption", "ITI"],
                    ["moving", "stationary", "all"],
                )
            )
        )
    )
    for i, session in enumerate(sessions):
        print(session)
        (
            nav,
            rc,
            ITI,
            moving_nav,
            stationary_nav,
            moving_ITI,
            stationary_ITI,
        ) = get_KMcluster_transtion_trial_phase_correlations(session, moving_stationary=True)
        result = [
            session.subject_ID,
            session.maze_number,
            session.day_on_maze,
            moving_nav,
            stationary_nav,
            nav,
            np.nan,
            np.nan,
            rc,
            moving_ITI,
            stationary_ITI,
            ITI,
        ]
        results_df.loc[i] = result
    return results_df


def plot_trial_phase_movement_stationary_corrs(results_df):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 6), clear=True, sharey=True)
    session_averaged_data = results_df.set_index("subject_ID").groupby("subject_ID").mean()
    # trial phase figure
    trial_phase_results = session_averaged_data[[("navigation", "all"), ("reward_consumption", "all"), ("ITI", "all")]]
    trial_phase_results.columns = ["Navigation", "Reward Consumption", "ITI"]
    max_value = trial_phase_results.max().max()
    min_value = trial_phase_results.min().min()
    sns.swarmplot(data=trial_phase_results, palette="plasma", size=8, alpha=0.5, ax=ax1)
    sns.pointplot(
        data=trial_phase_results,
        markers="_",
        palette="plasma",
        scale=2,
        linestyles="none",
        ci=95,
        ax=ax1,
    )
    ax1.set_ylabel("Correlation of Cluster Transitions \n at short and long timescales")
    # ax.set_ylim(min_value - 0.1, max_value + 0.1)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.axhline(0, color="k", linestyle="--", lw=1, alpha=0.5)
    ax1.set_xticklabels(["Nav", "RC", "ITI"])
    ax1.set_xlabel("Trial Phase")
    # moving stationary figure
    moving_stationary_results = session_averaged_data[
        [
            ("navigation", "moving"),
            ("navigation", "stationary"),
            ("ITI", "moving"),
            ("ITI", "stationary"),
        ]
    ]
    melted_moving_df = moving_stationary_results.reset_index().melt(
        id_vars="subject_ID", var_name=["trial_phase", "movement"], value_name="value"
    )
    sns.pointplot(
        ax=ax2,
        data=melted_moving_df,
        x="trial_phase",
        y="value",
        hue="movement",
        markers="_",
        dodge=0.4,
        join=False,
        ci=95,
        scale=2,
        palette="mako",
    )

    sns.swarmplot(
        ax=ax2,
        data=melted_moving_df,
        x="trial_phase",
        y="value",
        hue="movement",
        palette="mako",
        size=8,
        alpha=0.5,
        dodge=True,
    )

    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(
        handles=handles[:2],
        labels=labels[:2],
        title="Movement",
        bbox_to_anchor=(1.05, 1),
        loc=2,
    )
    ax2.set_xlabel("Trial Phase")
    ax2.axhline(0, color="k", linestyle="--", lw=1, alpha=0.5)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.set_xticklabels(["Nav", "ITI"])
    ax2.set_ylabel("")

    return


# %%


def plot_cluster_trial_trajectories(session, traj_smooth_SD=5, colormap="nipy_spectral"):
    long_df, short_df = get_navigation_KMcluster_df(
        session,
        n_clusters=10,
        timescale="both",
        long_navigation_only=True,
        long_moving=None,
        long_exclude_time_at_goal=True,
        long_smooth_SD=120,
        short_navigation_only=True,
        short_moving=None,
        short_exclude_time_at_goal=True,
    )
    simple_maze = session.simple_maze()

    KMeans_clusters = long_df.KMeans_cluster.long.unique()
    KMcluster2color = get_KMcluster2color(len(KMeans_clusters), colormap=colormap)
    trials = np.intersect1d(  # some trials missing due to filtering contraints
        long_df.trial.unique(), short_df.trial.unique()
    )
    for trial in trials:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), clear=True)
        f.subplots_adjust(wspace=-0.2)
        for df, label, ax in zip([long_df, short_df], ["long", "short"], [ax1, ax2]):
            trial_df = df[df.trial == trial]
            goal = trial_df.goal.unique()[0]
            cluster_color_traj = trial_df.KMeans_cluster[label].map(KMcluster2color).to_numpy()
            x_traj = trial_df.centroid_position.x
            y_traj = trial_df.centroid_position.y
            if traj_smooth_SD:
                x_traj = gaussian_filter1d(x_traj, sigma=traj_smooth_SD)
                y_traj = gaussian_filter1d(y_traj, sigma=traj_smooth_SD)
            # Create a list of endpoints for each line segment
            points = np.array([x_traj, y_traj]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # Create the line collection object, setting the colormapping parameters.
            lc = LineCollection(segments, colors=cluster_color_traj, linewidth=4)
            mp.plot_simple_maze_silhouette(
                simple_maze,
                ax,
                color="silver",
                node_size=400,
                edge_size=9,
                highlight_nodes=[goal],
                highlight_color="deepskyblue",
            )
            ax.add_collection(lc)
            plot_colormap(ax, len(KMeans_clusters), colormap=colormap)
            ax.set_title(f"{label} timescale")
            if label == "long":
                cluster_sequence = list(
                    trial_df[
                        trial_df.KMeans_cluster.long.ne(trial_df.KMeans_cluster.long.shift(-1))
                    ].KMeans_cluster.long
                )
        cluster_string = insert_newlines(cluster_sequence, every=6)
        ax1.text(0.5, -0.1, cluster_string, ha="center", transform=ax1.transAxes)
    return


def insert_newlines(sequence, every=6):
    return " → ".join(
        str(item) + ("\n" if (index + 1) % every == 0 and index + 1 != len(sequence) else "")
        for index, item in enumerate(sequence)
    )


def plot_colormap(ax, n_clusters, colormap="viridis"):
    cmap = plt.cm.get_cmap(colormap)
    bounds = np.arange(-0.5, n_clusters + 0.5, 1)
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Create the colorbar
    cbar = plt.colorbar(sm, ticks=np.arange(0, n_clusters), ax=ax, aspect=20)
    cbar.set_label("KMeans Cluster")

    # 1. Remove the border
    cbar.outline.set_visible(False)
    cbar.ax.set_position([0.85, 0.3, 0.03, 0.4])


# %% test for autocorrelation functions


def get_cluster_autocorrelation(session, max_lag=2, n_clusters=10, plot=True):
    long_df, short_df = get_navigation_KMcluster_df(
        session,
        n_clusters=n_clusters,
        timescale="both",
        long_navigation_only=True,
        long_moving=None,
        long_exclude_time_at_goal=True,
        long_smooth_SD=120,
        short_navigation_only=True,
        short_moving=None,
        short_exclude_time_at_goal=True,
    )
    cluster_pair_lagged_correlations = []
    trial_fds = []
    for trial in short_df.trial.unique():
        trial_df = short_df[short_df.trial == trial]
        clusters_timeseries = trial_df.KMeans_cluster.short
        if len(clusters_timeseries) < max_lag * FRAME_RATE + 2:
            continue
        lagged_correlations = get_paired_cluster_autocorrelation_array(clusters_timeseries, max_lag, n_clusters)
        cluster_pair_lagged_correlations.append(lagged_correlations)
        freq, fd = get_cluster_pair_frequency_decomposition(lagged_correlations)
        trial_fds.append(fd)
    av_fd = np.nanmean(np.stack(trial_fds), axis=0)
    cluster_pair_corrs = np.nanmean(
        np.stack(cluster_pair_lagged_correlations, axis=0), axis=0
    )  # av autocorr across trials
    if plot:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8), clear=True)
        f.tight_layout()
        sns.heatmap(
            ax=ax1,
            data=cluster_pair_corrs,
            cmap="coolwarm",
            center=0,
            cbar_kws={"label": "Pearson Correlation"},
        )
        ax1.set_xlabel("Time Lag (seconds)")
        ax1.set_ylabel("Cluster Pair")
        pc_labels = []
        row = 0
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                pc_labels.append(f"C{i} - C{j}")
                row += 1
        yticks = np.arange(len(pc_labels)) + 0.5
        ax1.set_yticks(yticks)
        ax1.set_yticklabels(pc_labels, rotation=0, ha="right", size="small")
        # Set xticks and xticklabels
        total_ticks = 11  # 5 on either side of 0, plus the 0
        tick_locations = np.linspace(0, cluster_pair_corrs.shape[1] - 1, total_ticks) + 0.5
        tick_labels = [str(round(val, 2)) for val in np.linspace(-max_lag, max_lag, total_ticks)]
        ax1.set_xticks(tick_locations)
        ax1.set_xticklabels(tick_labels, rotation=0, ha="center")
        ax1.axvline(
            x=cluster_pair_corrs.shape[1] // 2 + 0.5,
            color="black",
            linewidth=1,
            ls="--",
            alpha=0.5,
        )
        # frequency decomposition plot
        sns.heatmap(
            ax=ax2,
            data=av_fd,
            cmap="mako",
            vmax=0.01,
            vmin=0,
            cbar_kws={"label": "Amplitude"},
        )
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Cluster Pair")
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(pc_labels, rotation=0, ha="right", size="small")
        freq_tick_locations = np.linspace(0, len(freq) - 1, total_ticks, dtype=int)
        freq_tick_labels = [int(freq[i]) for i in freq_tick_locations]
        ax2.set_xticks(freq_tick_locations + 0.5)
        ax2.set_xticklabels(freq_tick_labels, rotation=0, ha="center")
        ax2.axvline(np.argmin(np.abs(freq - 4)), color="r", ls="--")
        ax2.axvline(np.argmin(np.abs(freq - 8)), color="r", ls="--")
    return


def get_paired_cluster_autocorrelation_array(clusters_timeseries, max_lag, n_clusters):
    cluster_timeseries_array = (  # split timesieres into binary cluster vectors for each cluster
        pd.concat([(clusters_timeseries == c).astype(int) for c in range(n_clusters)], axis=1)
        .to_numpy()  # [n_clusters, n_timepoints/frames]
        .T
    )
    # center around 0 before correlations
    cluster_timeseries_array = cluster_timeseries_array - cluster_timeseries_array.mean(axis=1, keepdims=True)
    # Get lagged autocorrelations
    max_lag_frames = int(max_lag * FRAME_RATE)
    n_pairs = n_clusters * (n_clusters - 1) // 2
    n_lags = 2 * max_lag_frames + 1  # Total number of lags
    autocorr_results = np.empty((n_pairs, n_lags))
    lags = np.arange(-max_lag_frames, max_lag_frames + 1)  # at frame rate resolution
    pair_index = 0
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            corrs = []
            for lag in lags:
                if lag < 0:
                    corrs.append(
                        pearsonr(
                            cluster_timeseries_array[i, :lag],
                            cluster_timeseries_array[j, -lag:],
                        )[0]
                    )
                elif lag > 0:
                    corrs.append(
                        pearsonr(
                            cluster_timeseries_array[i, lag:],
                            cluster_timeseries_array[j, :-lag],
                        )[0]
                    )
                else:
                    corrs.append(pearsonr(cluster_timeseries_array[i], cluster_timeseries_array[j])[0])
            autocorr_results[pair_index, :] = corrs
            pair_index += 1
    return autocorr_results


# %%


def get_cluster_pair_frequency_decomposition(lagged_correlations):
    fds = []
    for corr in lagged_correlations:
        freq, amplitude = get_frequency_decomposition(corr)
        fds.append(amplitude)
    return freq, np.stack(fds, axis=0)


def get_cluster_long_transition_matrix(long_df, n_clusters):
    long_df["KMeans_cluster", "long_shift"] = long_df.KMeans_cluster.long.shift(-1)
    long_M = np.zeros((n_clusters, n_clusters))
    # long timescale transitions
    for trial in long_df.trial.unique():
        long_trial_df = long_df[long_df.trial == trial]
        long_transitions_mask = long_trial_df.KMeans_cluster.long.ne(long_trial_df.KMeans_cluster.long_shift)
        long_cluster_transitions = (
            long_trial_df[long_transitions_mask].KMeans_cluster[:-1].to_numpy(dtype=int)
        )  # last transition not valid
        for i, j in long_cluster_transitions:
            long_M[i, j] += 1
    return long_M / corr_matrix_row_sum(long_M)
