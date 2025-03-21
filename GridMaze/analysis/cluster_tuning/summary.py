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
from GridMaze.paths import RESULTS_PATH

TUNING_SUMMARIES_SAVE_PATH = RESULTS_PATH / "tuning_summaries"
if not TUNING_SUMMARIES_SAVE_PATH.exists():
    TUNING_SUMMARIES_SAVE_PATH.mkdir()

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 12,
        "figure.titlesize": 14,
        "pdf.fonttype": 42,
    }
)

# %% Functions


def save_session_tuning_summaries(subject_ID, maze_name, day_on_maze, type="concise"):
    save_path = TUNING_SUMMARIES_SAVE_PATH / f"{subject_ID}_{maze_name}_{day_on_maze}_{type}.pdf"
    clusters = gc.get_maze_clusters(
        subject_IDs=[subject_ID], maze_names=[maze_name], days_on_maze=[day_on_maze], single_units=True
    )
    with PdfPages(save_path) as pdf:
        for cluster in clusters:
            if type == "concise":
                fig = plot_tuning_summary_concise(cluster)
            else:
                NotImplementedError
            pdf.savefig(fig)
            plt.close(fig)
    return


# %% Short summaries


def plot_tuning_summary_full(Cluster):
    return


def plot_tuning_summary_concise(Cluster):
    fig = plt.figure(figsize=(12, 5), clear=True)
    gsc = GridSpec(2, 4, figure=fig)
    # asign axes
    ax1 = fig.add_subplot(gsc[0:2, 0:2])  # place direction heatmap
    ax2 = fig.add_subplot(gsc[0:1, 2:3])  # distance to goal tuning
    ax3 = fig.add_subplot(gsc[1:2, 2:3])  # event tuning
    ax4 = fig.add_subplot(gsc[0:1, 3:4], projection="polar")  # angle to goal tuning
    ax5 = fig.add_subplot(gsc[1:2, 3:4])  # action tuning
    # use Cluser Obj to plot tuning to asigned axes
    Cluster.plot_tuning("place_direction", ax=ax1)
    Cluster.plot_tuning("distance_to_goal", ax=ax2)
    Cluster.plot_tuning("trial_events", ax=ax3)
    Cluster.plot_tuning("angle_to_goal", feature_kwargs={"angle_metric": "summary"}, ax=ax4)
    # _adjust_polar_axis(ax4)
    Cluster.plot_tuning("actions", ax=ax5, feature_kwargs={"concise": True})
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    fig.suptitle(f"{Cluster.cluster_unique_ID}", fontsize=16)
    return fig


def _adjust_polar_axis(ax):
    ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    ax.set_xticklabels([int(i) for i in np.linspace(0, 360, 4, endpoint=False)])
    ax.spines["polar"].set_visible(False)
    rmax = ax.get_rmax()
    ax.plot([0, 0], [0, rmax], color="black", lw=1)  # positive x–axis (0°)
    ax.plot([np.pi, np.pi], [0, rmax], color="black", lw=1)  # negative x–axis (180°)
    ax.plot([np.pi / 2, np.pi / 2], [0, rmax], color="black", lw=1)  # positive y–axis (90°)
    ax.plot([3 * np.pi / 2, 3 * np.pi / 2], [0, rmax], color="black", lw=1)  # negative y–axis (270°)
    ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(-0.2, 1.2))
    return
