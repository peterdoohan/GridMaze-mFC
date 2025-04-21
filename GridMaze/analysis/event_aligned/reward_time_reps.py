"""
Library for analysis for similarrity of reward representations across goal locations across different
mazes, to highlight that maze structure dramatically effects neural representations.
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


# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH, ANALYSIS_INFO_PATH

RESULTS_DIR = RESULTS_PATH / "event_aligned" / "reward_time_repres"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "maze_configs.json", "r") as input_file:
    MAZE_CONFIGS = json.load(input_file)

MAZE_NAMES = list(MAZE_CONFIGS.keys())

with open(ANALYSIS_INFO_PATH / "intra_trial_interval_times.json", "r") as f:
    INTRA_TRIAL_INTERVAL_TIMES = json.load(f)

# %% Functions


def plot_RDM_comparisons(comparisons_df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(2, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Mazes")
    ax.set_ylabel("Rep. Similarity Corr.")
    ax.set_ylim(-0.1, 0.6)
    ax.axhline(0, color="k", linestyle="--", lw=0.5)
    df = comparisons_df.groupby(["subject", "condition"])["corr"].mean().reset_index()
    for _, grp in df.groupby("subject"):
        # we know each subject has exactly two rows, one 'within' and one 'across'
        y = grp.set_index("condition").loc[["within", "across"], "corr"].values
        ax.plot(
            ["within", "across"],
            y,
            color="grey",
            linewidth=2,
            zorder=1,
            alpha=0.5,
        )
    sns.scatterplot(
        data=df,
        x="condition",
        y="corr",
        ax=ax,
        hue="condition",
        palette={"within": "purple", "across": "grey"},
        legend=False,
        zorder=2,
        s=100,
    )
    ax.margins(x=0.4)


def get_within_across_maze_RSM_comparison(
    maze_names=["maze_1", "maze_2"],  # rooms_maze
    alignment="event",
    window=(-0.25, 0.25),
    plot=False,
    save_name=None,
    verbose=False,
):
    """
    Main analysis function
    """

    def _corr_RSMs(RDM1, RDM2):
        """Assumes array inputs"""
        # make sure RDMs are the same size
        assert RDM1.shape == RDM2.shape
        # get upper triangle of each matrix
        mask = np.triu(np.ones_like(RDM1, dtype=bool), k=1)
        rdm1_upper = RDM1[mask]
        rdm2_upper = RDM2[mask]
        return np.corrcoef(rdm1_upper, rdm2_upper)[0, 1]

    def _combine_sessions(preloaded_sessions, subjects, maze_name, goal_subset):
        """Much faster to preload sessions and combine them on the fly"""
        sessions = []
        for subject in subjects:
            s = preloaded_sessions[goal_subset][subject][maze_name]
            s = [s] if not isinstance(s, list) else s
            sessions.extend(s)
        return sessions

    # check if already run and load
    save_path = RESULTS_DIR / f"{save_name}.csv"
    if save_name is None and save_path.exists():
        comparisons_df = pd.read_csv(save_path, index_col=0)
    else:

        # preload sessions
        if verbose:
            print("Preloading sessions...")
        preloaded_sessions = {}
        for goal_subset in ["subset_1", "subset_2", "all"]:
            subject2sessions = {}
            for subject in SUBJECT_IDS:
                maze2sessions = {}
                for maze in maze_names:
                    maze2sessions[maze] = get_sessions_for_analysis(subject, maze, goal_subset, alignment)
                subject2sessions[subject] = maze2sessions
            preloaded_sessions[goal_subset] = subject2sessions

        # get RDMs for each subject and maze for with and across maze comparisons
        comparisons = []
        for goal_subset in ["subset_1", "subset_2", "all"]:
            if verbose:
                print(f"Getting RDMs for goal subset: {goal_subset}")
            for subject in SUBJECT_IDS:
                if verbose:
                    print(subject)
                other_subjects = [s for s in SUBJECT_IDS if s != subject]
                for maze in maze_names:
                    other_mazes = [m for m in maze_names if m != maze]
                    # within maze comparisons
                    rdm_subject_maze = get_RSM(
                        _combine_sessions(preloaded_sessions, [subject], maze, goal_subset),
                        alignment,
                        window,
                        return_as="matrix",
                    )
                    rdm_other_subjects_same_maze = get_RSM(
                        _combine_sessions(preloaded_sessions, other_subjects, maze, goal_subset),
                        alignment,
                        window,
                        return_as="matrix",
                    )
                    comparisons.append(
                        {
                            "goal_subset": goal_subset,
                            "maze": maze,
                            "subject": subject,
                            "condition": "within",
                            "other_maze": None,
                            "corr": _corr_RSMs(rdm_subject_maze, rdm_other_subjects_same_maze),
                        }
                    )
                    # across maze comparisons
                    for other_maze in other_mazes:
                        rdm_other_subjects_other_maze = get_RSM(
                            _combine_sessions(preloaded_sessions, other_subjects, other_maze, goal_subset),
                            return_as="matrix",
                        )
                        comparisons.append(
                            {
                                "goal_subset": goal_subset,
                                "maze": maze,
                                "subject": subject,
                                "condition": "across",
                                "other_maze": other_maze,
                                "corr": _corr_RSMs(rdm_subject_maze, rdm_other_subjects_other_maze),
                            }
                        )
        comparisons_df = pd.DataFrame(comparisons)
        if save_name is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            comparisons_df.to_csv(save_path)
    if plot:
        plot_RDM_comparisons(comparisons_df)
    return comparisons_df


def get_RSM(sessions, alignment="event", window=(-0.25, 0.25), return_as="matrix"):
    """
    Note input session should be of the same maze and goal subset
    """
    sessions = [sessions] if not isinstance(sessions, list) else sessions
    reward_rates_df = pd.concat(
        [get_reward_rates_df(s, alignment, window) for s in sessions], axis=0
    )  # [n_neuron-sessions, n_goals]
    RDM = reward_rates_df.corr(method="pearson")
    if return_as == "matrix":
        return RDM.to_numpy()
    elif return_as == "df":
        return RDM
    else:
        raise NotImplementedError()


def get_reward_rates_df(session, alignment="event", window=(-0.25, 0.25)):
    """ """
    aligned_rates_df = session.event_aligned_rates_df if alignment == "event" else session.trial_aligned_rates_df
    keep_clusters = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=False,
    )
    aligned_rates_df = aligned_rates_df[aligned_rates_df.cluster_unique_ID.isin(keep_clusters)]
    # reduuce df to just firing rates at reward time
    if alignment == "event":
        _rates = aligned_rates_df.firing_rate.reward_aligned
        windowed_rates = _rates[_rates.columns[(_rates.columns >= window[0]) & (_rates.columns <= window[1])]]
        df = aligned_rates_df.drop(columns="firing_rate", level=0).droplevel([1, 2], axis=1)
    else:  # alignment = "trial"
        r_time = INTRA_TRIAL_INTERVAL_TIMES["reward"]
        windowed_rates = _rates[  # get rates over just reward consumption period
            _rates.columns[(_rates.columns >= r_time + window[0]) & (_rates.columns <= (r_time + window[1]))]
        ]
        df = aligned_rates_df.drop(columns="firing_rate", level=0).droplevel([1], axis=1)
    df["firing_rate"] = windowed_rates.mean(axis=1)
    # get cluster firing rates over trials at the same goal
    goal_rates_df = df.groupby(["goal", "cluster_unique_ID"]).firing_rate.mean().unstack()
    goal_rates_df = goal_rates_df.sort_index(axis=1).sort_index(axis=0).T
    return goal_rates_df  # [n_neurons, n_goals]


def get_sessions_for_analysis(subject, maze_name, goal_subset, alignment="event"):
    days_on_maze = "late" if goal_subset == "all" else "all"
    data_stucture = "trial_aligned_rates_df" if alignment == "trial" else "event_aligned_rates_df"
    return gs.get_maze_sessions(
        subject_IDs=[subject],
        maze_names=[maze_name],
        days_on_maze=days_on_maze,
        goal_subsets=[goal_subset],
        with_data=[data_stucture, "cluster_metrics"],
        must_have_data=True,
    )
