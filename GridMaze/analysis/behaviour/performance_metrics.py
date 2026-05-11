"""
Library of fns to calculate basic performance metrics for maze navigation &
plot the results
@peterdoohan
"""

# %% Imports
import json
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from matplotlib import pyplot as plt
from GridMaze.analysis.core import get_sessions as gs

from scipy.stats import zscore
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.anova import AnovaRM

# %% Global variables
from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "maze_configs.json", "r") as input_file:
    MAZE_CONFIGS = json.load(input_file)

# %% Performance Metric Functions


def get_analysis_sessions():
    sessions = gs.get_maze_sessions(
        subject_IDs="all", maze_names="all", days_on_maze="all", with_data=["trials_df", "navigation_df"]
    )
    return sessions


def get_basic_behaviour_df():
    save_path = RESULTS_PATH / "behaviour" / "performance_metrics" / "basic_behaviour_df.htsv"
    if save_path.exists():
        return pd.read_csv(save_path, sep="\t")
    else:
        sessions = get_analysis_sessions()
        session_results = []
        for session in sessions:
            trials_df = session.trials_df
            trial_durations = (trials_df.time.reward - trials_df.time.cue).to_numpy()
            n_excess_steps = _get_n_excess_steps(session)
            results_df = pd.DataFrame(
                {
                    "subject_ID": session.subject_ID,
                    "maze_name": session.maze_name,
                    "day_on_maze": session.day_on_maze,
                    "trial": trials_df.trial,
                    "goal": trials_df.goal,
                    "errors": trials_df.errors,
                    "duration": trial_durations,
                    "n_excess_steps": n_excess_steps,
                }
            )
            session_results.append(results_df)
        combined_results = pd.concat(session_results, axis=0)
        # save
        save_path.parent.mkdir(parents=True, exist_ok=True)
        combined_results.to_csv(save_path, sep="\t", index=False)
        return combined_results


def plot_performance_metrics(basic_behaviour_df, cmap="plasma", colors=None):
    """ """
    f, axes = plt.subplots(1, 3, figsize=(6, 3), clear=True)
    _plot_trials(basic_behaviour_df, ax=axes[0], cmap=cmap, colors=colors)
    _plot_durations(basic_behaviour_df, ax=axes[1], cmap=cmap, colors=colors)
    _plot_n_excess_steps(basic_behaviour_df, ax=axes[2], cmap=cmap, colors=colors)
    f.tight_layout()


# %% subplots


def _plot_n_excess_steps(basic_behaviour_df, ax=None, legend=False, cmap="plasma", colors=None, print_stats=True):
    """ """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(1, 2), clear=True)
    ax.spines[["top", "right"]].set_visible(False)
    grouped_df = basic_behaviour_df.groupby(["maze_name", "subject_ID", "day_on_maze"])
    n_excess_steps_df = grouped_df["n_excess_steps"].median().reset_index()
    palette = cmap if colors is None else colors
    sns.lineplot(
        x="day_on_maze",
        y="n_excess_steps",
        hue="maze_name",
        data=n_excess_steps_df,
        errorbar="se",
        err_style="band",
        palette=palette,
        ax=ax,
        legend=legend,
    )
    ax.set_xlabel("Day on Maze")
    ax.set_ylabel("n Excess Steps")
    if print_stats:
        df = n_excess_steps_df.dropna()
        df["maze_order"] = df["maze_name"].map({"maze_1": 1, "maze_2": 2, "rooms_maze": 3})
        m = smf.mixedlm(
            "n_excess_steps ~ maze_order * day_on_maze",
            data=df,
            groups=df["subject_ID"],
        ).fit(reml=False)

        print(m.summary())


def _plot_errors(basic_behaviour_df, ax=None, legend=False, cmap="plasma", colors=None):
    """ """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(1, 2), clear=True)
    ax.spines[["top", "right"]].set_visible(False)
    grouped_df = basic_behaviour_df.groupby(["maze_name", "subject_ID", "day_on_maze"])
    errors_df = grouped_df["errors"].mean().reset_index()
    palette = cmap if colors is None else colors
    sns.lineplot(
        x="day_on_maze",
        y="errors",
        hue="maze_name",
        data=errors_df,
        errorbar="se",
        err_style="band",
        palette=palette,
        ax=ax,
        legend=legend,
    )
    ax.set_xlabel("Day on Maze")
    ax.set_ylabel("Errors")


def _plot_durations(basic_behaviour_df, ax=None, legend=False, cmap="plasma", colors=None):
    """ """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(1, 2), clear=True)
    ax.spines[["top", "right"]].set_visible(False)
    grouped_df = basic_behaviour_df.groupby(["maze_name", "subject_ID", "day_on_maze"])
    duration_df = grouped_df["duration"].median().reset_index()
    palette = cmap if colors is None else colors
    sns.lineplot(
        x="day_on_maze",
        y="duration",
        hue="maze_name",
        data=duration_df,
        errorbar="se",
        err_style="band",
        palette=palette,
        ax=ax,
        legend=legend,
    )
    ax.set_xlabel("Day on Maze")
    ax.set_ylabel("Duration (s)")


def _plot_trials(basic_behaviour_df, ax=None, legend=False, cmap="plasma", colors=None, print_stats=True):
    """ """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(1, 2), clear=True)
    ax.spines[["top", "right"]].set_visible(False)
    grouped_df = basic_behaviour_df.groupby(["maze_name", "subject_ID", "day_on_maze"])
    trial_counts_df = grouped_df["trial"].max().reset_index()
    palette = cmap if colors is None else colors
    sns.lineplot(
        x="day_on_maze",
        y="trial",
        hue="maze_name",
        data=trial_counts_df,
        errorbar="se",
        err_style="band",
        palette=palette,
        ax=ax,
        legend=legend,
    )
    ax.set_xlabel("Day on Maze")
    ax.set_ylabel("Trials")
    if print_stats:
        df = trial_counts_df.dropna()
        df["maze_order"] = df["maze_name"].map({"maze_1": 1, "maze_2": 2, "rooms_maze": 3})

        m = smf.mixedlm(
            "trial ~ maze_order * day_on_maze",
            data=df,
            groups=df["subject_ID"],
        ).fit(reml=False)

        print(m.summary())


# %%


def _get_n_excess_steps(session):
    """
    Loops over trials in navigation df to calculate the path length taken and compares it to
    the shortest path length possible. n_excess_steps = actual_path_length - optimal_path_length.
    Note: steps = node-to-node not distance in meters.
    """
    simple_maze = session.simple_maze()
    simple_node2coord = {v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()}
    navigation_df = session.navigation_df
    trials = [t for t in navigation_df.trial.unique() if not np.isnan(t)]
    navigation_df[("maze_position", "simple_shifted")] = navigation_df.maze_position.simple.shift(1)
    navigation_df[("maze_position", "simple_change")] = (
        navigation_df.maze_position.simple != navigation_df.maze_position.simple_shifted
    )
    navigation_df = navigation_df[navigation_df.trial_phase == "navigation"]
    navigation_df = navigation_df[navigation_df.maze_position.simple_change == True]
    trial_excess_steps = []
    for trial in trials:
        trial_df = navigation_df[navigation_df.trial == trial]
        if len(trial_df) == 0:
            trial_excess_steps.append(np.nan)
            continue  # NaN value if naviagtion contains no transitions between nodes
        goal = trial_df.goal.unique()[0]
        goal_coord = simple_node2coord[goal]
        full_path = trial_df.maze_position.simple
        node_path = full_path[full_path.apply(lambda x: len(x.split("-")) == 1)].to_numpy()  # remove edges
        node_path = np.array([key for key, _ in itertools.groupby(node_path)])  # remove duplicates
        if not len(node_path) < 2:
            path = [simple_node2coord[node] for node in node_path]  # convert to coordinates
            shortest_path = nx.shortest_path_length(simple_maze, path[0], goal_coord, weight=None)
            path_length = len(path) - 1
            n_excess_steps = path_length - shortest_path
            trial_excess_steps.append(n_excess_steps)
        else:  # NaN value is trials start right next to goal
            trial_excess_steps.append(np.nan)
    return trial_excess_steps
