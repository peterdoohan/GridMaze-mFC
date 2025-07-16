"""
Library for distance to goal tuning analyses: curve fits, headmaps etc.
"""

# %% Imports
import re
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import convert

from GridMaze.analysis.cluster_tuning import distance_to_goal as dtg
from GridMaze.analysis.distance_to_goal import distributions as dd
from GridMaze.analysis.processing import get_distance_tuning_metrics_df as dtm
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)


CURVE_FITS = ["gamma_4p", "gaussian_4p", "polynomial_4p"]


def plot_distance_tunned_heatmap(
    population_tuning_df,
    metric="distance_to_goal",
    sign="pos",
    order_by="fit",
    smooth_SD=2,
    fit="gamma_4p",
    normalisation_method="zscore",
    cmap="coolwarm",
    v_range=(-1, 2),
    ax=None,
):
    """ """
    # include only "distance tuned" clusters (those with cv_r2 significanltly > 0)
    df = population_tuning_df[population_tuning_df["gamma_4p_cv"].sig]
    if sign == "pos":
        sign_mask = df[fit]["size"].gt(0)
    elif sign == "neg":
        sign_mask = df[fit]["size"].lt(0)
    else:
        raise ValueError(f"Unknown sign: {sign}")
    df = df[sign_mask]
    x = df[metric].columns.values.astype(float)
    if sign == "pos":
        if order_by == "fit":
            df[("idx_max", "")] = df.apply(lambda row: get_idx_order(row, x, fit=fit, op="max"), axis=1)
        elif order_by == "max":
            df[("idx_max", "")] = df[metric].idxmax(axis=1)
        df = df.sort_values(by=[("idx_max", "")], ascending=True)
    elif sign == "neg":
        if order_by == "fit":
            df[("idx_min", "")] = df.apply(lambda row: get_idx_order(row, x, fit=fit, op="min"), axis=1)
        elif order_by == "min":
            df[("idx_min", "")] = df[metric].idxmin(axis=1)
        df = df.sort_values(by=[("idx_min", "")], ascending=True)
    D = df[metric].values
    if smooth_SD:
        D = gaussian_filter1d(D, smooth_SD, axis=1)
    if normalisation_method == "max":
        D = D / np.max(D, axis=1)[:, None]
    elif normalisation_method == "zscore":
        D = zscore(D, axis=1)
    else:
        raise ValueError(f"Unknown normalisation method: {normalisation_method}")
    # plot
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 5))
    sns.heatmap(
        D,
        cmap=cmap,
        vmax=v_range[1],
        vmin=v_range[0],
        ax=ax,
        cbar_kws={"label": "Firing Rate (z-score)", "shrink": 0.5},
    )
    y_tick = len(D) // 10 * 10
    ax.set_yticks([y_tick])
    ax.set_yticklabels([f"{y_tick}"], rotation=90)
    ax.set_ylabel("Neurons", labelpad=-10)
    unit = "m" if metric == "distance_to_goal" else "%"
    ax.set_xlabel(f"{metric} \n (path-length) ({unit})")
    # _xticks = np.arange(0, max(x), 0.25)
    # ax.set_xticks(np.arange(0, len(x), len(_xticks) * 2))
    # ax.set_xticklabels(_xticks, rotation=0)


def get_idx_order(row, x, fit="gamma_4p", op="max"):
    """ """
    params = row[fit]
    curve_fit = dtm.gamma_4p(x, params["size"], params["shape"], params["scale"], params["shift"])
    if op == "max":
        return x[np.argmax(curve_fit)]
    elif op == "min":
        return x[np.argmin(curve_fit)]
    else:
        NotImplementedError


def get_population_tuning_df(
    late_sessions=True, metrics=("distance_to_goal", "geodesic"), min_split_half_corr=0.5, verbose=False
):
    """ """
    days_on_maze = "late" if late_sessions else "all"
    if verbose:
        print("Loading sessions...")
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names="all",
        days_on_maze=days_on_maze,
        with_data=["navigation_df", "navigation_spike_rates_df", "cluster_metrics", "cluster_distance_tuning_metrics"],
    )
    tuning_dfs = []
    for session in sessions:
        if verbose:
            print(session.name)
        distance_tuning_df = _get_session_distance_tuning(session, metrics=metrics)
        distance_tuning_df = distance_tuning_df.reset_index()
        distance_tuning_df = distance_tuning_df.rename(columns={"index": "cluster_unique_ID"}, level=0)
        distance_metrics_df = session.cluster_distance_tuning_metrics
        if distance_metrics_df.index.name == "cluster_unique_ID":
            distance_metrics_df = distance_metrics_df.reset_index()
        distance_metrics_df = distance_metrics_df[distance_metrics_df.single_unit & distance_metrics_df.distance_tuned]
        distance_tuning_df = distance_tuning_df[
            distance_tuning_df.cluster_unique_ID.isin(distance_metrics_df.cluster_unique_ID)
        ]
        distance_metrics_df.set_index("cluster_unique_ID", inplace=True)
        df = pd.merge(  # conbine tuning curves and (precomputed) tuning curve fit metrics
            distance_tuning_df,
            distance_metrics_df,
            on="cluster_unique_ID",
            how="inner",
        )
        # filter for distance tuned, split half corr, and r2
        df = df[(df.distance_tuned) & (df.split_half_corr.value.gt(min_split_half_corr))]
        tuning_dfs.append(df)
    population_tuning_df = pd.concat(tuning_dfs, axis=0).reset_index(drop=True)
    return population_tuning_df


def _get_session_distance_tuning(
    session,
    metrics=("distance_to_goal", "geodesic"),
    bin_spacing=0.05,
    max_steps_to_goal=30,
    progress_bins=30,  # if metrics[0] == "progress_to_goal"
    moving_only=False,
    return_as="tuning_curves",
):
    """ """
    navigation_rates_df = session.get_navigation_activity_df(type="rates", cluster_kwargs={"single_units": True})
    # deal with moving only
    if moving_only:
        navigation_rates_df = navigation_rates_df[navigation_rates_df.moving]
    if max_steps_to_goal is not None:
        navigation_rates_df = navigation_rates_df[navigation_rates_df.steps_to_goal.future < max_steps_to_goal]
    # remove frames where distance is above max (treat as outliers)
    if metrics[0] == "distance_to_goal":
        max_distance = dd.get_distance_percentile(metrics, 0.85)
        n_bins = int(max_distance / bin_spacing)
        navigation_rates_df = navigation_rates_df[navigation_rates_df[metrics] < max_distance]
        bins = convert._get_distance_bins(
            binning_method="uniform",
            n_distance_bins=n_bins,
            distance_metrics=metrics,
            max_distance=max_distance,
        )
    elif metrics[0] == "progress_to_goal":
        bins = convert._get_progress_bins(progress_bins)
    else:
        raise NotImplementedError
    # bin distances
    navigation_rates_df.loc[:, ("distance_bin", "")] = pd.cut(  # called distance but could be progress
        navigation_rates_df[metrics], bins=bins, include_lowest=True
    ).to_numpy()
    # average over frames in each bin over trials
    trial_av_rates = navigation_rates_df.groupby(["trial", "distance_bin"], observed=True).firing_rate.mean()
    if return_as == "trial_av_rates":
        return trial_av_rates
    else:  # return as tuning curves
        distance_tuning_df = (
            trial_av_rates.groupby(["distance_bin"]).mean().firing_rate.T
        )  # cluster x distance_bins (average over trials)
        distance_tuning_df.columns = pd.MultiIndex.from_product(
            [[metrics[0]], [b.mid for b in distance_tuning_df.columns]]
        )
        return distance_tuning_df


# %% Curve fit summary and plotting functions


def plot_curve_fit_distributions(summary_df, curve_fits=CURVE_FITS, ax=None):
    """ """
    df = summary_df[curve_fits]
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.set_xlabel("R2")
    ax.set_ylabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    for col in df.columns:
        sns.histplot(
            data=df[col],
            stat="count",
            kde=True,
            element="step",
            fill=False,
            ax=ax,
            label=col,
            alpha=0,
        )
    ax.legend(loc="upper left")


def plot_cross_subject_curve_fit_comparison(summary_df, curve_fits=CURVE_FITS, ax=None):
    """
    Make pretty later
    """
    # process data
    df = summary_df.groupby("subject_ID")[curve_fits].mean().unstack().reset_index()
    df.columns = ["fit", "subject_ID", "r2"]

    # plot
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("mean R2")
    ax.set_ylim(0.65, 0.85)
    sns.pointplot(
        data=df,
        x="fit",
        y="r2",
        hue="subject_ID",
        palette=sns.color_palette("hls", len(SUBJECT_IDS)),
        errorbar=None,
        dodge=False,
        markers="o",
        linestyles="-",
        legend=False,
        markersize=10,
        linewidth=4,
    )
    ax.tick_params(axis="x", which="both", top=False, bottom=True, labeltop=False, labelbottom=True, labelrotation=45)
    return


def get_tuning_fits_summary_df(late_sessions=True, min_split_half_corr=0.5):
    """
    Compare how well different distribution shapes (gamma, gaussian, polynomial) fit empirical
    distance to goal tuning curves.
    """
    days_on_maze = "late" if late_sessions else "all"
    subject_dfs = []
    for subject in SUBJECT_IDS:
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject],
            maze_names="all",
            days_on_maze=days_on_maze,
            with_data=["cluster_distance_tuning_metrics"],
        )
        r2_dfs = []
        for session in sessions:
            metrics_df = session.cluster_distance_tuning_metrics
            # select for "distance_tunned" clusters
            metrics_df = metrics_df[
                (metrics_df.distance_tuned) & (metrics_df.split_half_corr.value.gt(min_split_half_corr))
            ]
            # curve fit params and r2 values calculated in analysis/processing/get_distance_tuning_metrics_df.py
            r2_df = metrics_df[[(c, "r2") for c in CURVE_FITS]].droplevel(1, axis=1)
            r2_dfs.append(r2_df)
        subject_df = pd.concat(r2_dfs, axis=0).reset_index(drop=True)
        subject_df["subject_ID"] = subject
        subject_dfs.append(subject_df)
    # do cross subject states

    # combine all cells and plot distributions
    summary_df = pd.concat(subject_dfs, axis=0).reset_index(drop=True)

    return summary_df
