"""
Library for plotting tuning metric summaries
"""

# %% Imports
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages

from . import spatial
from . import angle_to_goal as atg
from . import events
from . import distance_to_goal as dtg
from . import actions
from ..core import get_clusters as gc


# %% Global Variables

TUNING_SUMMARIES_SAVE_PATH = Path("../results/tuning_summaries")

# %% Functions


def save_session_tuning_summaries(subject_ID, maze_name, day_on_maze):
    save_path = TUNING_SUMMARIES_SAVE_PATH / f"{subject_ID}_{maze_name}_{day_on_maze}.pdf"
    clusters = gc.get_clusters(subject_IDs=[subject_ID], maze_names=[maze_name], days_on_maze=[day_on_maze])
    with PdfPages(save_path) as pdf:
        for cluster in clusters:
            fig = get_tuning_summary(cluster)
            pdf.savefig(fig)
            plt.close(fig)
    return


def get_tuning_summary(cluster, return_fig=False):
    fig = plt.figure(figsize=(20, 25))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    gsc = GridSpec(22, 16, figure=fig)
    ax_space_all = fig.add_subplot(gsc[0:4, 0:4])  # heatmaps
    ax_space_nav = fig.add_subplot(gsc[0:4, 5:9])
    ax_pd = fig.add_subplot(gsc[0:4, 10:14])
    ax_trial = fig.add_subplot(gsc[5:7, 0:6])  # trial aligned rates
    ax_dist = fig.add_subplot(gsc[5:7, 7:13])  # distance to goal tuning
    ax_r1 = fig.add_subplot(gsc[8:10, 2:4])  # route tuning
    ax_r2 = fig.add_subplot(gsc[8:10, 5:13])
    ax_r3 = fig.add_subplot(gsc[11:13, 2:4])
    ax_r4 = fig.add_subplot(gsc[11:13, 5:13])
    ax_a1 = fig.add_subplot(gsc[14:17, 0:3])  # action tuning
    ax_a2 = fig.add_subplot(gsc[14:17, 4:7])
    ax_a3 = fig.add_subplot(gsc[14:17, 8:11])
    ax_route_rates = fig.add_subplot(gsc[18:21, 0:6])  # route aligned rates
    ax_ego = fig.add_subplot(gsc[18:21, 6:11], projection="polar")  # angle to goal tuning
    ax_allo = fig.add_subplot(gsc[18:21, 10:14], projection="polar")
    # change egocentric atg plot to polar coords
    for ax, label in zip([ax_ego, ax_allo], ["Egocentric", "Allocentric"]):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
        ax.set_xticklabels([int(i) for i in np.linspace(0, 360, 8, endpoint=False)])
        ax.set_xlabel(label + " Angle to Goal")
    # rate map plots
    rate_map_all = cluster._load_spatial_tuning(navigation_only=False, moving_only=False, exclude_time_at_goal=False)
    spatial.plot_spatial_heatmap(*rate_map_all, ax=ax_space_all)
    ax_space_all.set_title("Whole Session")
    rate_map_nav = cluster._load_spatial_tuning(navigation_only=True, moving_only=True, exclude_time_at_goal=True)
    spatial.plot_spatial_heatmap(*rate_map_nav, ax=ax_space_nav)
    ax_space_nav.set_title("Navigation Only")
    # plot place direction hm
    pd_data = cluster._load_place_direction_tuning()
    spatial.plot_place_direction_tuning(*pd_data, ax=ax_pd)
    # plot trial aligned rates
    trial_data = cluster._load_trial_aligned_tuning()
    events.plot_trial_aligned_rates(trial_data, ax=ax_trial)
    # plot distance to goal tuning
    dist_data = cluster._load_distance_to_goal_tuning(metrics=("distance_to_goal", "geodesic"))
    dtg.plot_distance_tuning(dist_data, ("distance_to_goal", "geodesic"), ax=ax_dist)
    # plot route tuning
    future_route_tuning = cluster._load_route_tuning(sequence="future")
    past_route_tuning = cluster._load_route_tuning(sequence="past")
    vmax = routes._get_route_tuning_vmax(future_route_tuning, past_route_tuning)
    routes.plot_routes_tuning(future_route_tuning, axes=[ax_r1, ax_r2], title="Future Route Tuning", vmax=vmax)
    routes.plot_routes_tuning(past_route_tuning, axes=[ax_r3, ax_r4], title="Past Route Tuning", vmax=vmax)
    # plot action tuning
    action_tuning = cluster._load_action_tuning()
    actions.plot_action_tuning(action_tuning, axes=[ax_a1, ax_a2, ax_a3])
    # load route aligned rates
    route_rates = cluster._load_route_aligned_rates()
    routes.plot_route_aligned_tuning(route_rates, ax=ax_route_rates)
    # plot angle to goal tuning
    eg_data = cluster._load_angle_to_goal_tuning("egocentric_angle_to_goal")
    atg.plot_angle_tuning(*eg_data, ax=ax_ego)
    allo_data = cluster._load_angle_to_goal_tuning("allocentric_angle_to_goal")
    atg.plot_angle_tuning(*allo_data, ax=ax_allo)
    if return_fig:
        return fig


# %% Short summaries


def plot_tuning_summary_full(Cluster):
    return


def plot_tuning_summary_concise(Cluster):
    fig = plt.figure(figsize=(18, 6))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    gsc = GridSpec(2, 5, figure=fig)
    # asign axes
    ax1 = fig.add_subplot(gsc[0:2, 0:2])  # place direction heatmap
    ax2 = fig.add_subplot(gsc[0:1, 2:4])  # distance to goal tuning
    ax3 = fig.add_subplot(gsc[1:2, 2:4])  # event tuning
    ax4 = fig.add_subplot(gsc[0:1, 4:5], projection="polar")  # angle to goal tuning
    ax5 = fig.add_subplot(gsc[1:2, 4:5])  # action tuning
    # use Cluser Obj to plot tuning to asigned axes
    Cluster.plot_tuning("place_direction", ax=ax1)
    Cluster.plot_tuning("distance_to_goal", ax=ax2)
    Cluster.plot_tuning("trial_events", ax=ax3)
    # Cluster.plot_tuning("angle_to_goal", feature_kwargs={}, ax=ax4)
    Cluster.plot_tuning("action", ax=ax5, feature_kwargs={"concise": True})
    return
