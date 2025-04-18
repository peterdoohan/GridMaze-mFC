"""
Library for analysis for similarrity of reward representations across goal locations across different
mazes, to highlight that maze structure dramatically effects neural representations.
@peterdoohan
"""

# %% Imports
import json
import pandas as pd
import numpy as np
from . import allocentric_goal_decoding as agd
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc

import seaborn as sns
from matplotlib import pyplot as plt

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "maze_configs.json", "r") as input_file:
    MAZE_CONFIGS = json.load(input_file)

MAZE_NAMES = list(MAZE_CONFIGS.keys())

# %% Functions


def plot_RDM_comparisons(comparisons_df, ax=None):
    """ """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("RDM comparison")
    ax.set_ylabel("Correlation")
    ax.set_ylim(0, 0.4)
    df = comparisons_df.groupby(["subject_1", "condition"])["corr"].mean().reset_index()
    sns.pointplot(data=df, x="condition", y="corr", hue="subject_1", ax=ax)


def get_within_across_maze_RDM_comparison(plot=True):
    """ """

    def _corr_RDMs(RDM1, RDM2):
        # make sure RDMs are the same size
        assert RDM1.shape == RDM2.shape
        # get upper triangle of each matrix
        mask = np.triu(np.ones_like(RDM1, dtype=bool), k=1)
        rdm1_upper = RDM1[mask]
        rdm2_upper = RDM2[mask]
        return np.corrcoef(rdm1_upper, rdm2_upper)[0, 1]

    comparisons = []
    # get RDMs for each subject and maze
    for goal_subset in ["subset_1", "subset_2", "all"]:
        subject2maze2RDM = {}
        for subject_ID in SUBJECT_IDS:
            maze2RDM = {}
            for maze_name in MAZE_NAMES:
                maze2RDM[maze_name] = get_RDM(subject_ID, maze_name, goal_subset, return_as="matrix")
            subject2maze2RDM[subject_ID] = maze2RDM

        # within maze comparisons
        for maze in MAZE_NAMES:
            for subject in SUBJECT_IDS:
                for other_subject in SUBJECT_IDS:
                    if subject != other_subject:
                        rdm1 = subject2maze2RDM[subject][maze]
                        rdm2 = subject2maze2RDM[other_subject][maze]
                        comparisons.append(
                            {
                                "goal_subset": goal_subset,
                                "maze_1": maze,
                                "maze_2": None,
                                "subject_1": subject,
                                "subject_2": other_subject,
                                "condition": "within",
                                "corr": _corr_RDMs(rdm1, rdm2),
                            }
                        )
        # across maze comparisons
        for maze1 in MAZE_NAMES:
            for maze2 in MAZE_NAMES:
                if maze1 != maze2:
                    for subject in SUBJECT_IDS:
                        rdm1 = subject2maze2RDM[subject][maze1]
                        rdm2 = subject2maze2RDM[subject][maze2]
                        comparisons.append(
                            {
                                "goal_subset": goal_subset,
                                "maze_1": maze1,
                                "maze_2": maze2,
                                "subject_1": subject,
                                "subject_2": None,
                                "condition": "across",
                                "corr": _corr_RDMs(rdm1, rdm2),
                            }
                        )
    comparisons_df = pd.DataFrame(comparisons)
    if plot:
        plot_RDM_comparisons(comparisons_df)
    return comparisons_df


def get_RDM(subject, maze_name, goal_subset, return_as="df"):
    """ """
    sessions = get_sessions_for_analysis(subject, maze_name, goal_subset)
    sessions = [sessions] if not isinstance(sessions, list) else sessions
    reward_rates_df = pd.concat(
        [get_reward_rates_df(s, window=(0, 0.5), include_multi_units=True) for s in sessions], axis=0
    )  # [n_neuron-sessions, n_goals]
    RDM = reward_rates_df.corr(method="pearson")
    if return_as == "matrix":
        return RDM.to_numpy()
    elif return_as == "df":
        return RDM
    else:
        raise NotImplementedError()


def get_reward_rates_df(session, window=(0, 0.5), include_multi_units=True):
    """ """
    aligned_rates_df = session.event_aligned_rates_df
    keep_clusters = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=include_multi_units,
    )
    aligned_rates_df = aligned_rates_df[aligned_rates_df.cluster_unique_ID.isin(keep_clusters)]
    # reduuce df to just firing rates at reward time
    _rates = aligned_rates_df.firing_rate.reward_aligned
    windowed_rates = _rates[_rates.columns[(_rates.columns >= window[0]) & (_rates.columns <= window[1])]]
    df = aligned_rates_df.drop(columns="firing_rate", level=0).droplevel([1, 2], axis=1)
    df["firing_rate"] = windowed_rates.mean(axis=1)
    # get cluster firing rates over trials at the same goal
    goal_rates_df = df.groupby(["goal", "cluster_unique_ID"]).firing_rate.mean().unstack()
    goal_rates_df = goal_rates_df.sort_index(axis=1).sort_index(axis=0).T
    return goal_rates_df  # [n_neurons, n_goals]


def get_sessions_for_analysis(subject, maze_name, goal_subset):
    days_on_maze = "late" if goal_subset == "all" else "all"
    return gs.get_maze_sessions(
        subject_IDs=[subject],
        maze_names=[maze_name],
        days_on_maze=days_on_maze,
        goal_subsets=[goal_subset],
        with_data=["event_aligned_rates_df", "cluster_metrics"],
        must_have_data=True,
    )
