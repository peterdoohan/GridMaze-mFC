"""
Library for plotting tuning metric summaries
"""

# %% Imports
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages

from GridMaze.analysis.core import get_clusters as gc


# %% Global Variables
from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

TUNING_SUMMARIES_SAVE_PATH = RESULTS_PATH / "tuning_summaries"
if not TUNING_SUMMARIES_SAVE_PATH.exists():
    TUNING_SUMMARIES_SAVE_PATH.mkdir()

with open(Path(EXPERIMENT_INFO_PATH) / "maze_day2date.json", "r") as input_file:
    MAZE_DAY2DATE = json.load(input_file)

with open(Path(EXPERIMENT_INFO_PATH) / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 8,
        "figure.titlesize": 18,
        "pdf.fonttype": 42,
        # "font.family": "FreeMono",
    }
)

# %% Functions


def save_all_tuning_summaries(type="concise"):
    for maze in ["rooms_maze"]:
        for day in MAZE_DAY2DATE[maze].keys():
            for subject_ID in SUBJECT_IDS:
                try:
                    print(f"Saving {subject_ID}_{maze}_{day}_{type}.pdf")
                    save_session_tuning_summaries(subject_ID, maze, int(day), type)
                except:
                    print(f"Failed to save {subject_ID}_{maze}_{day}_{type}.pdf")
    return


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
    Cluster.plot_tuning("actions", ax=ax5, feature_kwargs={"concise": True, "action_type": "all"})
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    fig.suptitle(f"{Cluster.cluster_unique_ID}", fontsize=16)
    # make adjustments
    ax4.legend(loc="upper left", bbox_to_anchor=(-0.4, 1.2))
    ax5.legend(loc="upper right", bbox_to_anchor=(1.0, 1.3))
    return fig
