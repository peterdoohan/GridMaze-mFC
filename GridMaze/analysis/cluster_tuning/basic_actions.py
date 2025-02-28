"""This module is for plotting neurl tuning to basic actions (eg, left turn, right turn, straight)"""
# %% Imports
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .. import get_sessions as gs


# %% Functions
def plot_session_basic_action_aligned_rates(session):
    cluster_unique_IDs, plotting_variables = prepare_plot_basic_action_aligned_rates(session)
    for cluster_unique_ID in cluster_unique_IDs:
        f, ax = plt.subplots(1, 1, figsize=(6, 5), clear=True)
        plot_cluster_basic_action_aligned_rates(ax, cluster_unique_ID, *plotting_variables)


def prepare_plot_basic_action_aligned_rates(
    session, cluster_type="good", smoothed=True, smooth_SD=5, zoom_window=(-1.5, 1.5)
):
    data_needed = ["basic_action_aligned_rates_df"]
    if not gs.check_session_has_data(session, data_needed):
        pass
    # if necessary session attribute is not present raises error
    basic_action_rates_df = session.basic_action_aligned_rates_df
    if cluster_type == "good":
        basic_action_rates_df = basic_action_rates_df[basic_action_rates_df.cluster_type == "good"].reset_index(
            drop=True
        )
    if smoothed:
        basic_action_rates_df.loc[:, ("action_aligned_rates", slice(None))] = gaussian_filter1d(
            basic_action_rates_df.action_aligned_rates, sigma=smooth_SD, axis=1
        )
    basic_action_rates_df["forced_or_choice"] = basic_action_rates_df["choice_degree"].apply(
        lambda x: "choice" if x > 2 else "forced"
    )
    cluster_unique_IDs = basic_action_rates_df.cluster_unique_ID.unique()
    basic_actions = basic_action_rates_df.basic_action.unique()
    # data grouped by cluster and action
    cluster_action_grouped_rates = basic_action_rates_df.set_index(["cluster_unique_ID", "basic_action"]).groupby(
        ["cluster_unique_ID", "basic_action"]
    )
    av_cluster_action_rates_df = cluster_action_grouped_rates.mean().action_aligned_rates
    sem_cluster_action_rates_df = cluster_action_grouped_rates.sem().action_aligned_rates
    # data grouped by cluster and choice (or forced)
    cluster_choice_grouped_rates_df = basic_action_rates_df.set_index(
        ["cluster_unique_ID", "forced_or_choice"]
    ).groupby(["cluster_unique_ID", "forced_or_choice"])
    av_cluster_choice_rates_df = cluster_choice_grouped_rates_df.mean().action_aligned_rates
    sem_cluster_choice_rates_df = cluster_choice_grouped_rates_df.sem().action_aligned_rates
    # data grouped by cluster
    cluster_grouped_rates_df = basic_action_rates_df.set_index("cluster_unique_ID").groupby("cluster_unique_ID")
    av_cluster_rates_df = cluster_grouped_rates_df.mean().action_aligned_rates
    sem_cluster_rates_df = cluster_grouped_rates_df.sem().action_aligned_rates
    timepoints = av_cluster_rates_df.columns.to_numpy()
    zoom_timepoints_mask = np.logical_and(timepoints >= zoom_window[0], timepoints <= zoom_window[1])
    zoom_timepoints = timepoints[zoom_timepoints_mask]
    plotting_variables = (
        basic_actions,
        zoom_timepoints_mask,
        zoom_timepoints,
        av_cluster_action_rates_df,
        sem_cluster_action_rates_df,  # for action grouped plot
        av_cluster_choice_rates_df,
        sem_cluster_choice_rates_df,  # for choice grouped plot
        timepoints,
        av_cluster_rates_df,
        sem_cluster_rates_df,
    )  # for zoomed out plot
    return cluster_unique_IDs, plotting_variables


def plot_cluster_basic_action_aligned_rates(
    ax,
    cluster_ID,
    basic_actions,
    zoom_timepoints_mask,
    zoom_timepoints,
    av_cluster_action_rates_df,
    sem_cluster_action_rates_df,  # action grouped plot
    av_cluster_choice_rates_df,
    sem_cluster_choice_rates_df,  # choice grouped plot
    timepoints,
    av_cluster_rates_df,
    sem_cluster_rates_df,
):  # zoomed out plot
    ax.axis("off")
    ax.set_title(f"{cluster_ID}", fontdict={"family": "Courier", "size": 10}, loc="left", pad=10, x=0.02)
    # action grouped plot
    action_axis = inset_axes(
        ax, width="50%", height="95%", bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax.transAxes, loc="upper left"
    )
    action_colors = ["red", "green", "cornflowerblue"]
    actions_f_max = 0
    for action, color in zip(basic_actions, action_colors):
        action_label = action.split("_")[0].capitalize() + " " + action.split("_")[1].capitalize()
        cluster_action_av_rates_df = av_cluster_action_rates_df.loc[(cluster_ID, action)].loc[zoom_timepoints_mask]
        cluster_action_sem_rates_df = sem_cluster_action_rates_df.loc[(cluster_ID, action)].loc[zoom_timepoints_mask]
        action_f_max = np.nanmax(cluster_action_av_rates_df) + np.nanmax(cluster_action_sem_rates_df)
        if action_f_max > actions_f_max:
            actions_f_max = action_f_max
        action_axis.plot(zoom_timepoints, cluster_action_av_rates_df, color=color[0], label=action_label)
        action_axis.fill_between(
            zoom_timepoints.astype(float),
            cluster_action_av_rates_df - cluster_action_sem_rates_df,
            cluster_action_av_rates_df + cluster_action_sem_rates_df,
            color=color,
            alpha=0.2,
        )
    action_axis.axvline(0, color="k", linestyle="--", alpha=0.2)
    action_axis.spines["right"].set_visible(False)
    action_axis.spines["top"].set_visible(False)
    action_axis.set_ylim(0, actions_f_max)
    action_axis.set_xlim(-1.5, 1.5)
    action_axis.set_ylabel("Firing rate (Hz)", size=12)
    action_axis.set_xlabel("Action-aligned Time (s)", size=12)
    action_axis.legend(frameon=False, fancybox=False, shadow=False, fontsize=8, markerscale=10)
    # choice grouped plot
    choice_axis = inset_axes(
        ax, width="40%", height="40%", bbox_to_anchor=(0, 0, 1.1, 1), bbox_transform=ax.transAxes, loc="upper right"
    )
    for fc, color in zip(["forced", "choice"], ["mediumpurple", "indigo"]):
        cluster_choice_av_rates = av_cluster_choice_rates_df.loc[(cluster_ID, fc)].loc[zoom_timepoints_mask]
        cluster_choice_sem_rates = sem_cluster_choice_rates_df.loc[(cluster_ID, fc)].loc[zoom_timepoints_mask]
        choice_axis.plot(zoom_timepoints, cluster_choice_av_rates, color=color, label=fc.capitalize())
        choice_axis.fill_between(
            zoom_timepoints.astype(float),
            cluster_choice_av_rates - cluster_choice_sem_rates,
            cluster_choice_av_rates + cluster_choice_sem_rates,
            color=color,
            alpha=0.2,
        )
    choice_axis.axvline(0, color="k", linestyle="--", alpha=0.2)
    choice_axis.legend(frameon=False, fancybox=False, shadow=False, fontsize=8, markerscale=10)
    choice_axis.spines["right"].set_visible(False)
    choice_axis.spines["top"].set_visible(False)
    # zoomed out plot
    long_window_axis = inset_axes(
        ax, width="40%", height="40%", bbox_to_anchor=(0, 0, 1.1, 1), bbox_transform=ax.transAxes, loc="lower right"
    )

    cluster_av_rates_df = av_cluster_rates_df.loc[cluster_ID]
    cluster_sem_rates_df = sem_cluster_rates_df.loc[cluster_ID]
    long_window_axis.plot(timepoints, cluster_av_rates_df, color="black")
    long_window_axis.fill_between(
        timepoints.astype(float),
        cluster_av_rates_df - cluster_sem_rates_df,
        cluster_av_rates_df + cluster_sem_rates_df,
        color="black",
        alpha=0.2,
    )
    long_window_axis.axvline(0, color="k", linestyle="--", alpha=0.2)
    long_window_axis.spines["right"].set_visible(False)
    long_window_axis.spines["top"].set_visible(False)
    long_window_axis.set_xlabel("Action-aligned Time (s)", size=12)
    long_window_axis.text(0.6, 0.9, "Zoomed-out", transform=long_window_axis.transAxes)
