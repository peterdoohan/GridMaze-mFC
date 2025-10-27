"""
Look at theta modulation characteristics across the population, including stratification
by cluster tuning
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc

from GridMaze.analysis.processing import get_cluster_unique_variance_explained as cve
from matplotlib.patches import Rectangle

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

VARIANCE_EXPLAINED_DF = cve.get_cluster_unique_variance_explained(
    late_sessions=True, full_features=False, add_missing_clusters=True
)


# %% Functions


def test(late_sessions=True):
    theta_df = get_theta_mod_clusters(late_sessions=late_sessions)
    theta_mod_clusters = theta_df.cluster_unique_ID.values
    ve_df = cve.get_cluster_unique_variance_explained(
        late_sessions=late_sessions, full_features=False, add_missing_clusters=True
    )
    theta_df[("unique_ve", "distance_to_goal")] = ve_df.loc[theta_mod_clusters].distance_to_goal.values
    theta_df[("unique_ve", "place_direction")] = ve_df.loc[theta_mod_clusters].place_direction.values
    return theta_df


def get_theta_mod_clusters(late_sessions=True):
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names="all",
        days_on_maze="late" if late_sessions else "all",
        with_data=["cluster_theta_modulation_metrics"],
        must_have_data=True,
    )
    dfs = []
    for session in sessions:
        theta_mod_metrics = session.cluster_theta_modulation_metrics.copy()
        theta_mod_metrics = theta_mod_metrics[theta_mod_metrics.single_unit]
        dfs.append(theta_mod_metrics)
    combined_df = pd.concat(dfs)
    return combined_df


# def get_session_theta_mod_clusters(session, single_units=True, min_firing_rate=1, min_split_half_corr=0.4, min_r2=0.75):
#     """ """
#     theta_mod_metrics = session.cluster_theta_modulation_metrics.copy()
#     theta_mod_masks = []
#     if single_units:
#         theta_mod_masks.append(theta_mod_metrics.single_unit)
#     if min_firing_rate is not None:
#         theta_mod_masks.append(theta_mod_metrics.mean_firing_rate >= min_firing_rate)
#     if min_split_half_corr is not None:
#         theta_mod_masks.append(theta_mod_metrics.split_half_corr >= min_split_half_corr)
#     if min_r2 is not None:
#         theta_mod_masks.append(theta_mod_metrics.vonmises.r2 >= min_r2)
#     if len(theta_mod_masks) > 0:
#         combined_mask = np.logical_and.reduce(theta_mod_masks)
#     df = theta_mod_metrics[combined_mask]
#     return df


def plot_tuning(cuID):
    Cluster = gc.get_cluster(cuID)
    Cluster.plot_tuning(feature="place_direction")
    Cluster.plot_tuning(feature="distance_to_goal")
    plt.show()


# %%


def plot_cluster_theta_mod(
    session,
    cuID="m6.2022-07-04.maze_cluster4",
    navigation_only=True,
    moving_only=True,
    max_steps_to_goal=30,
    n_perm=1_000,
):
    """ """
    # load data
    navigation_df = session.navigation_df
    df = session.navigation_theta_spike_counts_df.spike_count[cuID]
    # filter for moving, navigation, on task etc.
    mask = []
    if navigation_only:
        mask.append((navigation_df.trial_phase == "navigation").values)
    if moving_only:
        mask.append(navigation_df.moving.values)
    if max_steps_to_goal is not None:
        mask.append(navigation_df.steps_to_goal.future.le(max_steps_to_goal).values)
    if len(mask) > 0:
        combined_mask = np.logical_and.reduce(mask)
        df = df[combined_mask]

    # shuffle procedure
    T = df.values
    null = np.zeros((n_perm, T.shape[1]), dtype=np.float32)
    for i in range(n_perm):
        null[i] = np.apply_along_axis(np.random.permutation, 1, T).sum(axis=0)

    # rng = np.random.default_rng(seed=0)
    # n_rows, n_cols = T.shape
    # rand = rng.random(size=(n_perm, n_rows, n_cols), dtype=np.float32)
    # # Argsort to produce permutation indices along columns for each (perm, row)
    # idxs = np.argsort(rand, axis=2).astype(np.int32, copy=False)  # (n_perm, n_rows, n_cols)
    # T_expanded = T[None, :, :]  # (1, n_rows, n_cols)
    # shuffled = np.take_along_axis(T_expanded, idxs, axis=2)  # (n_perm, n_rows, n_cols)
    # null = shuffled.sum(axis=1)  # (n_perm, n_cols)

    # plot
    phases = df.columns.astype(float).values
    plt.plot(phases, T.sum(axis=0), color="k", linewidth=2)
    plt.fill_between(
        phases, np.percentile(null, 2.5, axis=0), np.percentile(null, 97.5, axis=0), color="gray", alpha=0.5
    )
    return null


# %% stratify theta-modulation by cluster representation (distance-to-goal, place-direction, etc.)


# %% define theta-modulated clusters


def get_pop_theta_mod_metrics(late_sessions=False):
    """ """
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names="all",
        days_on_maze="late" if late_sessions else "all",
        with_data=["cluster_theta_modulation_metrics"],
        must_have_data=True,
    )
    dfs = []
    for session in sessions:
        theta_mod_metrics = session.cluster_theta_modulation_metrics.copy()
        theta_mod_metrics = theta_mod_metrics[theta_mod_metrics.single_unit]
        dfs.append(theta_mod_metrics)
    combined_df = pd.concat(dfs)
    return combined_df.reset_index(drop=True)


def plot_theta_phase_by_mod_depth(metrics_df, corr_thres=0.45, r2_thres=0.7, axes=None):
    """ """
    # set up fig
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(3, 3), width_ratios=[1, 0.2], height_ratios=[0.2, 1])
    axes[0, 0].spines[["top", "right"]].set_visible(False)  # phase hist
    axes[0, 0].set_xticks([])
    axes[1, 0].spines[["top", "right"]].set_visible(False)  # depth vs phase
    axes[1, 1].spines[["top", "right"]].set_visible(False)  # depth hist
    axes[1, 1].set_yticks([])
    axes[0, 1].set_visible(False)

    # select for theta mod neurons
    df = _filter_for_theta_mod_neurons(metrics_df, corr_thres=corr_thres, r2_thres=r2_thres)

    palette = sns.color_palette("hls", n_colors=len(SUBJECT_IDS))
    for subject, color in zip(SUBJECT_IDS, palette):
        _df = df[df.subject_ID == subject]
        x, y = _df.vonmises.phase_max_rad, _df.vonmises.modulation_depth
        axes[1, 0].scatter(x=x, y=y, color=color, alpha=0.5, s=2)
        sns.histplot(
            x=x,
            bins=20,
            element="step",
            fill=False,
            stat="probability",
            color=color,
            alpha=0.8,
            ax=axes[0, 0],
        )
        sns.histplot(
            y=y,
            bins=20,
            element="step",
            fill=False,
            stat="probability",
            color=color,
            alpha=0.8,
            ax=axes[1, 1],
        )

    axes[0, 0].set_ylabel("Prop.")
    axes[1, 0].set_xlabel("Theta Phase (Rad)")
    axes[1, 0].set_ylabel("Modulation Depth")
    axes[1, 1].set_xlabel("Prop.")
    return


def _filter_for_theta_mod_neurons(metrics_df, corr_thres=0.45, r2_thres=0.7, fr_thres=1):
    df = metrics_df.copy()
    df = df[df.split_half_corr.gt(corr_thres) & df.vonmises.r2.gt(r2_thres) & df.mean_firing_rate.gt(fr_thres)]
    return df


def plot_prop_theta_mod_neurons(metrics_df, corr_thres=0.45, r2_thres=0.7, ax=None):
    """ """
    if ax is None:
        fig, ax = plt.subplots(figsize=(1, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", linewidth=0.5)

    df = metrics_df.copy()
    df[("theta_modulated", "")] = df.split_half_corr.gt(corr_thres) & df.vonmises.r2.gt(r2_thres)

    prop_df = df.groupby(["subject_ID", "theta_modulated"]).maze_name.count().unstack()
    prop_df = prop_df.div(prop_df.sum(axis=1), axis=0)

    prop_cross_subjects = prop_df[True]
    prop_cross_subjects = prop_cross_subjects.reset_index()
    prop_cross_subjects.columns = ["subject_ID", "score"]
    palette = sns.color_palette("hls", n_colors=len(SUBJECT_IDS))
    sns.pointplot(
        data=prop_cross_subjects,
        y="score",
        hue="subject_ID",
        dodge=False,
        errorbar=None,
        markers="o",
        markersize=7,
        markeredgewidth=0,
        linestyle="none",
        palette=palette,
        legend=False,
        alpha=0.8,
        ax=ax,
    )
    ax.set_ylabel("prop. Theta-mod. neurons")


def plot_theta_mod_cuttoffs(metrics_df, corr_thres=0.45, r2_thres=0.7, fr_thres=None, ax=None):
    """ """
    # set up fig
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 3))
    ax.spines[["top", "right"]].set_visible(False)

    df = metrics_df.copy()
    if fr_thres is not None:
        df = df[df.mean_firing_rate.gt(fr_thres)]
    df = df[[("split_half_corr", ""), ("vonmises", "r2")]].copy()
    df = df.dropna()
    x, y = df[("split_half_corr", "")], df[("vonmises", "r2")]

    sns.scatterplot(x=x, y=y, color="k", alpha=0.5, s=2, ax=ax)
    sns.histplot(
        x=x,
        y=y,
        bins=30,
        pthresh=0.01,
        cmap="gray",
        ax=ax,
        cbar=True,
        cbar_kws={"label": "neurons", "shrink": 0.5},
    )
    sns.kdeplot(x=x, y=y, levels=8, color="white", linewidths=0.5, ax=ax)

    ax.axvline(corr_thres, color="red", linestyle="--")
    ax.axhline(r2_thres, color="red", linestyle="--")
    x0, y0 = corr_thres, r2_thres
    x_max, y_max = ax.get_xlim()[1], ax.get_ylim()[1]
    width = max(0, x_max - x0)
    height = max(0, y_max - y0)

    rect = Rectangle(
        (x0, y0),
        width,
        height,
        linewidth=1,
        edgecolor="red",
        facecolor="red",
        alpha=0.12,
        linestyle="--",
        zorder=10,
    )
    ax.add_patch(rect)

    ax.set_xlim(-0.5, 1)
    ax.set_ylim(0, 1)
    return
