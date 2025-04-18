"""
Library for analysis for similarrity of reward representations across goal locations across different
mazes, to highlight that maze structure dramatically effects neural representations.
@peterdoohan
"""

# %% Imports
import json
import pandas as pd
import numpy as np
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc

import seaborn as sns
from matplotlib import pyplot as plt

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "event_aligned" / "reward_time_repres"

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


def get_within_across_maze_RDM_comparison(plot=True, save=False, verbose=False):
    """ """
    # check if already run and load
    save_path = RESULTS_DIR / "within_across_maze_RDM_comparisons.csv"
    if not save and save_path.exists():
        comparisons_df = pd.read_csv(save_path, index_col=0)
        return comparisons_df

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
        if verbose:
            print(f"Getting RDMs for goal subset: {goal_subset}")
        for subject in SUBJECT_IDS:
            if verbose:
                print(subject)
            other_subjects = [s for s in SUBJECT_IDS if s != subject]
            for maze in MAZE_NAMES:
                other_mazes = [m for m in MAZE_NAMES if m != maze]
                # within maze comparisons
                rdm_subject_maze = get_RDM([subject], maze, goal_subset, return_as="matrix")
                rdm_other_subjects_same_maze = get_RDM(other_subjects, maze, goal_subset, return_as="matrix")
                comparisons.append(
                    {
                        "goal_subset": goal_subset,
                        "maze": maze,
                        "subject": subject,
                        "condition": "within",
                        "other_maze": None,
                        "corr": _corr_RDMs(rdm_subject_maze, rdm_other_subjects_same_maze),
                    }
                )
                # across maze comparisons
                for other_maze in other_mazes:
                    rdm_other_subject_other_maze = get_RDM(other_subjects, other_maze, goal_subset, return_as="matrix")
                    comparisons.append(
                        {
                            "goal_subset": goal_subset,
                            "maze": maze,
                            "subject": subject,
                            "condition": "across",
                            "other_maze": other_maze,
                            "corr": _corr_RDMs(rdm_subject_maze, rdm_other_subject_other_maze),
                        }
                    )
    comparisons_df = pd.DataFrame(comparisons)
    if save:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        comparisons_df.to_csv(save_path)
    if plot:
        plot_RDM_comparisons(comparisons_df)
    return comparisons_df


def get_RDM(subjects, maze_names, goal_subset, return_as="df"):
    """ """
    sessions = get_sessions_for_analysis(subjects, maze_names, goal_subset)
    sessions = [sessions] if not isinstance(sessions, list) else sessions
    reward_rates_df = pd.concat(
        [get_reward_rates_df(s, window=(-0.5, 0.5), include_multi_units=False) for s in sessions], axis=0
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


def get_sessions_for_analysis(subjects, maze_name, goal_subset):
    days_on_maze = "late" if goal_subset == "all" else "all"
    return gs.get_maze_sessions(
        subject_IDs=subjects,
        maze_names=[maze_name],
        days_on_maze=days_on_maze,
        goal_subsets=[goal_subset],
        with_data=["event_aligned_rates_df", "cluster_metrics"],
        must_have_data=True,
    )
