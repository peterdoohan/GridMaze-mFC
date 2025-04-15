"""
Library for looking at the distribution of brain regions sampled in this experiment
@peterdoohan
"""

# %% Imports
import json
import re
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import date
from collections import Counter
from GridMaze.analysis.core import get_sessions as gs
from matplotlib import pyplot as plt

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "maze_day2date.json", "r") as input_file:
    MAZE_DAY2DATE = json.load(input_file)

PROBE_DEPTHS_DF = pd.read_csv(EXPERIMENT_INFO_PATH / "probe_depths.htsv", sep="\t")

# %% Region cell counts


def get_subject_cell_counts(ignore_layers=True, min_cells=10, plot=False):
    """
    Get the number of single units recorded in each region for each subject
    Parameters:
        ignore_layers : bool, optional
            If True, disregards the specific cortical layer in region labels
            (e.g., converts "PL5" to "PL"). Default is True.
        min_cells : int, optional
            The minimum number of cells required for a region to be included in the results.
            Regions with fewer than this threshold are omitted. Default is 10.
    Returns:
        pd.DataFrame
            A DataFrame indexed by subject IDs with columns corresponding to each
            brain region. Each cell in the DataFrame contains the count of single
            units recorded in that region for the respective subject.
    """
    # count how many single units we recorded in each region throughout the experiment
    subject_region_counts = []
    for subject_ID in SUBJECT_IDS:
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names="all",
            days_on_maze="all",
            with_data=["cluster_metrics"],
            must_have_data=True,
        )
        cell_region_counts = []
        for session in sessions:
            cluster_metrics = session.cluster_metrics
            region_counts = list(cluster_metrics[cluster_metrics.single_unit].region.acronym.values)
            if ignore_layers:  # just keep main region not layer (just "PL", not "PL5")
                region_counts = [re.match(r"([A-Za-z]+)(.*)", s).groups()[0] for s in region_counts]
            cell_region_counts.extend(region_counts)
        total_counts = dict(Counter(cell_region_counts))
        if min_cells:
            total_counts = {k: v for k, v in total_counts.items() if v >= min_cells}
        subject_region_counts.append(total_counts)

    # find total unique regions recorded from
    all_regions = np.unique([j for i in subject_region_counts for j in i.keys()])

    # organise into dataframe
    results_df = pd.DataFrame(index=SUBJECT_IDS, columns=all_regions)
    for subject_ID, region_counts in zip(SUBJECT_IDS, subject_region_counts):
        for region, count in region_counts.items():
            results_df.loc[subject_ID, region] = count
    results_df.fillna(0, inplace=True)
    if plot:
        plot_subject_cell_counts(results_df)
    return results_df


def plot_subject_cell_counts(results_df, ax=None):
    """ """
    # set up figure
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2.5, 3))
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.set_ylabel("Single Unit • Days")
    ax.set_xlabel("Subjects")
    # set order of regions
    results_df = results_df[["PL", "ACAd", "ACAv", "MOs"]]
    colors = ["mediumvioletred", "royalblue", "cornflowerblue", "slategrey"]
    results_df.plot(kind="bar", stacked=True, ax=ax, color=colors, alpha=0.5, width=0.8)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="lower left", bbox_to_anchor=(1.05, 0.5), ncol=1, fontsize=10)


# %%  Unit stability plots


def get_single_unit_stability(plot=False):
    """ """
    unit_stability = []
    for subject_ID in SUBJECT_IDS:
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names="all",
            days_on_maze="all",
            with_data=["cluster_metrics"],
            must_have_data=True,
        )
        for session in sessions:
            cluster_metrics = session.cluster_metrics
            unit_stability.append(
                {
                    "subject_ID": subject_ID,
                    "experimental_day": session.experimental_day,
                    "n_single_units": cluster_metrics[cluster_metrics.single_unit].shape[0],
                }
            )
    unit_stability_df = pd.DataFrame(unit_stability)
    if plot:
        plot_single_unit_stability(unit_stability_df)
    return unit_stability_df


def plot_single_unit_stability(df, log_scale=True, ax=None):
    """ """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(5, 3))
    # set up figure
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.set_ylabel("Single Units")
    ax.set_xlabel("")
    maze_days = _get_maze_exp_days()
    probe_move_days = _get_probe_move_days()
    for day in probe_move_days:
        ax.axvline(day, 0, 10, color="black", linestyle="--", lw=1, alpha=0.2)
    mazes = ["Maze 1", "Maze 2", "Rooms Maze"]
    boundaries = [val for tup in maze_days for val in tup]
    midpoints = [(tup[0] + tup[1]) / 2 for tup in maze_days]
    ax.set_xticks(boundaries, minor=True)
    ax.set_xticks(midpoints)
    ax.set_xticklabels(mazes, fontsize=10)
    ax.tick_params(axis="x", which="minor", length=5)
    ax.tick_params(axis="x", which="major", length=0)
    if log_scale:
        ax.set_yscale("log")
    colors = sns.color_palette("blend:royalblue,salmon", len(SUBJECT_IDS))
    # plot data
    for subject, color in zip(SUBJECT_IDS, colors):
        subject_df = df[df.subject_ID == subject]
        ax.plot(subject_df.experimental_day, subject_df.n_single_units, label=subject, lw=2, alpha=0.8, color=color)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, fontsize=10, frameon=False)


# %% get key dates


def _get_probe_move_days():
    """Get experimental days when all subjects probes were moved.
    Ignore speciall instances where a single subjects probe was moved"""
    probe_move_dates = []
    for _date in PROBE_DEPTHS_DF.date.unique():
        if len(np.setdiff1d(SUBJECT_IDS, PROBE_DEPTHS_DF[PROBE_DEPTHS_DF.date == _date].subject.values)) == 0:
            probe_move_dates.append(_date)
    probe_move_dates = [date.fromisoformat(d) for d in probe_move_dates]
    # get date to experimental day
    start_date = date.fromisoformat(MAZE_DAY2DATE["maze_1"]["1"])
    date2exp_day = [(date - start_date).days + 1 for date in probe_move_dates]
    # ignore moves before first maze recording
    return [d for d in date2exp_day if d > 0]


def _get_maze_exp_days():
    """
    Return a tuple of the exp_days that correspond to the start and end of each maze
    """
    exp_start_date = date.fromisoformat(MAZE_DAY2DATE["maze_1"]["1"])
    maze_exp_days = []
    for maze in MAZE_DAY2DATE.keys():
        days = list(MAZE_DAY2DATE[maze].keys())
        start_date = date.fromisoformat(MAZE_DAY2DATE[maze][days[0]])
        start_day = (start_date - exp_start_date).days + 1
        end_date = date.fromisoformat(MAZE_DAY2DATE[maze][days[-1]])
        end_day = (end_date - exp_start_date).days + 1
        maze_exp_days.append((start_day, end_day))
    return maze_exp_days
