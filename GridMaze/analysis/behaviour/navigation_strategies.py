"""
This moduel models choices during maze navigation as a function of vector based and shortest path based strategies.
And visualizes the results.
"""

# %% Imports
import os
import sys
import json
from matplotlib import lines
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from zict import File
from GridMaze.analysis.core import get_sessions as gs
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_1samp

# %% Global variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

INVALID_TRANSITION = -100
LOG_MAX_FLOAT = np.log(sys.float_info.max / 2.1)

with open(EXPERIMENT_INFO_PATH / "maze_day2date.json", "r") as input_file:
    MAZE_DAY2DATE = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)


# %% Modelling functions


def get_navigation_strategy_weights(sessions):
    """
    Calculates the optimal weights for the vector navigation weight, structure navigation weight, and penalty weight using maximum likelihood estimation.

    Parameters
    ----------
    sessions : list of Session
        A list of Session objects for which to calculate the optimal weights.

    Returns
    -------
    dict
        A dictionary containing the optimal weights for the vector navigation value, structure navigation value, and penalty value.
        The keys are 'weight_vector', 'weight_structure', and 'weight_penalty', respectively.
    """
    initial_weights = [0, 0, 0]
    result = minimize(get_neg_loglikelihood, initial_weights, args=(sessions,), method="BFGS")
    optimal_weights = result.x
    optimal_weight_vector, optimal_weight_structure, optimal_weight_penalty = optimal_weights
    return {
        "weight_vector": optimal_weight_vector,
        "weight_structure": optimal_weight_structure,
        "weight_penalty": optimal_weight_penalty,
    }


def get_neg_loglikelihood(weights, sessions):
    """
    Calculates the negative log likelihood of the data given the vector_navigation, structure_navigation and penalty_weights.

    Parameters
    ----------
    weights : tuple
        A tuple of three floats representing the weights for the vector navigation value, structure navigation value, and penalty value, respectively.
    sessions : list of Session
        A list of Session objects for which to calculate the negative log likelihood.

    Returns
    -------
    float
        The negative log likelihood of the data given the weights.
    """
    weight_vector, weight_structure, weight_penalty = weights
    session_navigation_strategies_dfs = [session.navigation_strategies_df for session in sessions]
    navigation_strategies_df = pd.concat(session_navigation_strategies_dfs, axis=0, ignore_index=True)
    navigation_strategies_df = navigation_strategies_df.dropna(axis=0)
    V_vector = navigation_strategies_df.vector_navigation_value.to_numpy()
    V_structure = navigation_strategies_df.structure_navigation_value.to_numpy()
    V_penalty = navigation_strategies_df.penalty_value.to_numpy()
    A_bool = navigation_strategies_df.available.to_numpy()
    A = np.where(A_bool, 0, INVALID_TRANSITION)
    choice_mask = navigation_strategies_df.choice_value.to_numpy().astype(bool)
    V = weight_vector * V_vector + weight_structure * V_structure + weight_penalty * V_penalty + A
    P = softmax(V, choice_mask)
    loglikelihood = np.log(P)
    if np.any(np.isnan(loglikelihood)):
        assert ValueError("Log likelihood contains NaN(s).")
    return -np.sum(np.log(P))


def softmax(V, choice_mask):
    """Calculates softmax probabilities for choices in a given state."""
    V[V > LOG_MAX_FLOAT] = LOG_MAX_FLOAT  # Protection against overflow in exponential.
    expV = np.exp(V)
    return expV[choice_mask] / np.sum(expV, axis=1)


# %% Plotting functions


def get_strategy_weights_across_subjects(plot=False):
    """ """
    navigation_strategy_weights = []
    for subject in SUBJECT_IDS:
        for maze_name in ["maze_1", "maze_2", "rooms_maze"]:
            late_sessions = gs.get_maze_sessions(
                subject_IDs=[subject],
                maze_names=[maze_name],
                days_on_maze="late",  # define late sessions as last 7 days on each maze
                with_data=["navigation_strategies_df"],
            )
            strategy_weights = get_navigation_strategy_weights(late_sessions)
            navigation_strategy_weights.append(
                {
                    "subject_ID": subject,
                    "maze_name": maze_name,
                    "weight_vector": strategy_weights["weight_vector"],
                    "weight_structure": strategy_weights["weight_structure"],
                    "weight_penalty": strategy_weights["weight_penalty"],
                }
            )
    navigation_strategy_weights_df = pd.DataFrame(navigation_strategy_weights)
    if plot:
        plot_strategy_weights_cross_subject(navigation_strategy_weights_df)
    return navigation_strategy_weights_df


def plot_strategy_weights_cross_subject(
    results_df, mazes=["maze_1", "maze_2", "rooms_maze"], colormap="mako", print_stats=False, ax=None
):
    """
    Plots the weights for vector navigation and structure navigation strategies for late sessions
    on each maze per subject and test if they are significantly different from zero.
    """
    if ax is None:
        f, ax = plt.subplots(figsize=(3, 4))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # filter data
    results_df = results_df[results_df.maze_name.isin(mazes)]
    # plotting
    df_long = results_df.melt(
        id_vars=["subject_ID", "maze_name"],
        value_vars=["weight_vector", "weight_structure"],
        var_name="variable",
        value_name="value",
    )
    df_long["x_shift"] = df_long["maze_name"].map(
        {"fully_connected": 1.0, "maze_1": 2.0, "maze_2": 3.0, "rooms_maze": 4.0}
    ) + df_long["variable"].map({"weight_vector": -0.01, "weight_structure": 0.01}).astype(float)

    sns.stripplot(
        data=df_long,
        x="x_shift",
        y="value",
        color="grey",
        size=4,
        alpha=0.5,
        dodge=False,
        zorder=1,
        legend=False,
    )
    sns.pointplot(
        data=df_long,
        x="x_shift",
        y="value",
        hue="variable",
        estimator=np.mean,
        linestyles="",
        dodge=False,
        palette=colormap,
        linestyle="none",
        markersize=5,
    )
    plt.ylim(0, 1)
    plt.xticks(
        ticks=[0.5, 2.5, 4.5],
        labels=["1", "2", "Rooms"],
    )
    plt.ylabel("Strategy Weighting")
    plt.xlabel("Maze Name")
    if print_stats:
        df_melted = pd.melt(
            results_df,
            id_vars=["subject_ID", "maze_name"],
            value_vars=["weight_vector", "weight_structure"],
            var_name="weight_type",
            value_name="weight_value",
        )
        model = ols("weight_value ~ C(maze_name) + C(weight_type) + C(maze_name):C(weight_type)", data=df_melted).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)
        # also do t-tests against zero for each weight type and maze
        for maze in mazes:
            for weight in ["weight_vector", "weight_structure"]:
                subset = df_melted[(df_melted.maze_name == maze) & (df_melted.weight_type == weight)]
                t_stat, p_value = ttest_1samp(subset.weight_value, 0)
                print(f"{maze} - {weight}: t({len(subset)-1})={t_stat:.3f}, p={p_value:.3f}")


# %% weights over sessions functions


def get_strategy_weights_across_sessions(n_itter=1000, plot=False):
    """
    X
    """
    save_path = (
        RESULTS_PATH
        / "behaviour"
        / "navigation_strategies_modelling"
        / "navigation_strategy_weights_over_sessions2.htsv"
    )
    if save_path.exists():
        navigation_strategy_weights_df = pd.read_csv(save_path, sep="\t")
    else:
        # First make a dict with maze, day, subject with session object keys
        maze2day_subject_sessions = get_maze2day_subject_session()
        navigation_strategy_weights = []
        for i in range(n_itter):
            sampled_subjects = np.random.choice(SUBJECT_IDS, size=len(SUBJECT_IDS), replace=True)
            for maze_name in MAZE_DAY2DATE.keys():
                for day in [int(i) for i in MAZE_DAY2DATE[maze_name].keys()]:
                    sessions = [maze2day_subject_sessions[maze_name][day][s] for s in sampled_subjects]
                    sessions = [s for s in sessions if s is not None]
                    strategy_weights = get_navigation_strategy_weights(sessions)
                    navigation_strategy_weights.append(
                        {
                            "itter": i,
                            "maze_name": maze_name,
                            "day_on_maze": day,
                            "weight_vector": strategy_weights["weight_vector"],
                            "weight_structure": strategy_weights["weight_structure"],
                            "weight_penalty": strategy_weights["weight_penalty"],
                        }
                    )
        navigation_strategy_weights_df = pd.DataFrame(navigation_strategy_weights)
        # save the results
        navigation_strategy_weights_df.to_csv(save_path, sep="\t", index=False)
    if plot:
        plot_nav_strategy_weights_over_sessions(navigation_strategy_weights_df)
    return navigation_strategy_weights_df


def plot_nav_strategy_weights_over_sessions(nav_strategy_weights_df, cmap="viridis", fig=None, axes=None):
    # firt calculate the upper and lower 95CIs for each weight over bootstrapped itters
    df = nav_strategy_weights_df.copy()
    if axes is None or fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for maze, marker, ax in zip(["maze_1", "maze_2", "rooms_maze"], ["o", "o", "o"], axes):
        maze_df = df[df.maze_name == maze]
        grouped_df = maze_df.groupby("day_on_maze")[["weight_vector", "weight_structure"]]
        mean = grouped_df.mean()
        std = grouped_df.std()
        cmap = plt.cm.get_cmap(cmap)
        colors = [cmap(i) for i in np.linspace(0.2, 0.95, len(mean))]
        x = mean.weight_vector.to_numpy()
        xerr = std.weight_vector.to_numpy()
        y = mean.weight_structure.to_numpy()
        yerr = std.weight_structure.to_numpy()
        plot_scatter_with_error_bars(x, y, xerr, yerr, colors, marker, ax)
        ax.set_xlim(-0.05, 1.0)
        ax.set_ylim(-0.05, 1.0)
        ax.set_xlabel("Vector Weight")
        ax.set_ylabel("Structure Weight")
        # ax.set_aspect("equal")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.axvline(0, color="silver", linestyle="--", alpha=1, lw=1.5)
        ax.axhline(0, color="silver", linestyle="--", alpha=1, lw=1.5)
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.2, vmax=0.95))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.5, pad=-0.1)
        cbar.outline.set_edgecolor("none")
        cbar.set_ticks([0.2, 0.95], labels=["First", "Last"])
    return


def plot_scatter_with_error_bars(x, y, xerr, yerr, colors, marker, ax):
    for xi, yi, xerri, yerri, color in zip(x, y, xerr, yerr, colors):
        ax.errorbar(
            xi,
            yi,
            xerr=xerri,
            yerr=yerri,
            color=color,
            capsize=0,
            marker=marker,
            markersize=5,
            alpha=1,
            elinewidth=1,
        )
    return


def get_maze2day_subject_session():
    maze2day_subject_sessions = {}
    for maze_name in MAZE_DAY2DATE.keys():
        day2subject_sessions = {}
        for day in [int(i) for i in MAZE_DAY2DATE[maze_name].keys()]:
            subject2sessions = {}
            for subject in SUBJECT_IDS:
                try:
                    session = gs.get_maze_sessions(
                        subject_IDs=[subject],
                        maze_names=[maze_name],
                        days_on_maze=[day],
                        with_data=["navigation_strategies_df"],
                        must_have_data=True,
                    )
                    subject2sessions[subject] = session
                except FileNotFoundError:
                    subject2sessions[subject] = None
            day2subject_sessions[day] = subject2sessions
        maze2day_subject_sessions[maze_name] = day2subject_sessions
    return maze2day_subject_sessions
