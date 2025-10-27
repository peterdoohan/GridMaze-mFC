"""
Look at theta modulation characteristics across the population, including stratification
by cluster tuning
@peterdoohan
"""

# %% Imports
import json
from os import error
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


# %% Functions


# %% stratify theta-modulation by cluster representation (distance-to-goal, place-direction, etc.)


def get_theta_mod_split_tuning(
    df_type="feature_tuned", late_sessions=False, corr_thres=0.45, r2_thres=0.7, fr_thres=1, ax=None
):
    """ """
    # load data unique var exp df or feature tuned df
    if df_type == "feature_tuned":
        ft_df = cve.get_cluster_feature_tuned_df(
            late_sessions=late_sessions, full_features=False, add_missing_clusters=False
        )
    elif df_type == "variance_explained":
        ft_df = cve.get_cluster_unique_variance_explained(
            late_sessions=late_sessions, full_features=False, add_missing_clusters=False
        )
        ft_df = ft_df.swaplevel(axis=0)
    else:
        raise ValueError("df_type must be 'feature_tuned' or 'variance_explained'")

    # load theta metrics
    metrics_df = get_pop_theta_mod_metrics(late_sessions=late_sessions)
    theta_mod_df = _filter_for_theta_mod_neurons(
        metrics_df, corr_thres=corr_thres, r2_thres=r2_thres, fr_thres=fr_thres
    )
    theta_mod_clusters = theta_mod_df.cluster_unique_ID.values

    ft_df["theta_modulated"] = ft_df.index.get_level_values(1).isin(theta_mod_clusters)

    df = (
        ft_df.reset_index("subject_ID")
        .groupby(["subject_ID", "theta_modulated"])[["distance_to_goal", "place_direction"]]
        .mean()
    )
    df = df.reset_index()

    df_long = df.melt(
        id_vars=["subject_ID", "theta_modulated"],
        value_vars=["distance_to_goal", "place_direction"],
        var_name="measure",
        value_name="value",
    )
    return df_long


def plot_theta_mod_split_tuning(df):
    fig, axes = plt.subplots(1, 2, figsize=(3, 2), sharey=True)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="k", ls="--", alpha=0.5)

    palette = sns.color_palette("hls", n_colors=len(SUBJECT_IDS))
    for (
        ax,
        measure,
    ) in zip(
        axes,
        ["distance_to_goal", "place_direction"],
    ):
        subdf = df[df["measure"] == measure]
        sns.pointplot(
            data=subdf,
            x="theta_modulated",
            y="value",
            hue="subject_ID",
            dodge=False,
            markers="o",
            linestyle="none",
            palette=palette,
            errorbar=None,
            legend=False,
            ax=ax,
        )
        sns.pointplot(
            data=subdf,
            x="theta_modulated",
            y="value",
            dodge=False,
            markers="_",
            linestyle="none",
            errorbar="se",
            color="k",
            ax=ax,
        )
        ax.set_ylabel("prop. neurons")
        ax.set_xlabel(measure)
    return


def plot_feature_rep_by_theta_phase_pref(late_sessions=False, corr_thres=0.45, r2_thres=0.7, fr_thres=1, axes=None):
    """ """
    # set up fig
    if axes is None:
        f, axes = plt.subplots(1, 2, figsize=(4, 2))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="k", ls="--", alpha=0.5)

    # load theta mod and variance explained data
    metrics_df = get_pop_theta_mod_metrics(late_sessions=late_sessions)
    filt_metrics_df = _filter_for_theta_mod_neurons(
        metrics_df, corr_thres=corr_thres, r2_thres=r2_thres, fr_thres=fr_thres
    )
    unique_ve_df = cve.get_cluster_unique_variance_explained(
        late_sessions=late_sessions, full_features=False, add_missing_clusters=True
    )
    theta_mod_clusters = filt_metrics_df.cluster_unique_ID.values
    filt_metrics_df[("unique_ve", "distance_to_goal")] = unique_ve_df.loc[theta_mod_clusters].distance_to_goal.values
    filt_metrics_df[("unique_ve", "place_direction")] = unique_ve_df.loc[theta_mod_clusters].place_direction.values

    # only include clusters that have unique ve data
    filt_metrics_df = filt_metrics_df[~filt_metrics_df.unique_ve.isna().any(axis=1)].copy()

    # bin phase pref
    bins = np.linspace(-np.pi, np.pi, 6)
    filt_metrics_df[("phase_max_bin", "")] = pd.cut(
        filt_metrics_df.loc[:, ("vonmises", "phase_max_rad")], bins=bins, include_lowest=True
    )

    subject_phase_by_ve = (
        filt_metrics_df.groupby(["subject_ID", "phase_max_bin"], observed=True).unique_ve.mean().unique_ve
    )

    # plot variance explained phase preference
    x = filt_metrics_df.vonmises.phase_max_rad
    for ft, ax, color1, color2 in zip(
        ["distance_to_goal", "place_direction"],
        axes,
        ["cornflowerblue", "palevioletred"],
        ["darkblue", "darkred"],
    ):
        # plot background scatter plot
        y = filt_metrics_df[("unique_ve", ft)]
        sns.scatterplot(x=x, y=y, ax=ax, color=color1, alpha=0.5, s=5)

        ve_df = subject_phase_by_ve[ft].unstack()
        _mean = ve_df.mean()
        _sem = ve_df.sem()
        _phase = [b.mid for b in _mean.index]
        ax.errorbar(
            x=_phase,
            y=_mean,
            yerr=_sem,
            marker="o",
            markersize=3,
            linestyle="none",
            color=color2,
            capsize=0,
        )
        # set log scale
        ax.set_ylim(-10, 60)
        ax.set_ylabel(ft)
        ax.set_xlabel("θ Phase")
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    return


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
    axes[1, 0].set_xlabel("θ Phase")
    axes[1, 0].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axes[1, 0].set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
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
