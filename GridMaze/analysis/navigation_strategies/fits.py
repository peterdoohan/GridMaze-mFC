""" """

# %% Imports
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp

from GridMaze.analysis.navigation_strategies import get_input_data as gid
from GridMaze.analysis.navigation_strategies import models

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]

# %% Functions


def get_strategy_weights(
    navigation_strategies_df,
    strategies=["vector", "structure", "habit", "backtracking_penalty"],
    late_sessions=True,
):
    """ """
    # filter data
    df = navigation_strategies_df.copy()
    if late_sessions:
        df = df[df.late_session]
    # fit nav strategy weights for decisions on each maze per subject
    results = []
    for maze in MAZE_NAMES:
        maze_df = df[df.maze_name == maze]
        for subject in SUBJECT_IDS:
            subj_df = maze_df[maze_df.subject_ID == subject]
            # fit strategy weights on select data
            strategy_weights = models.get_navigation_strategy_weights(subj_df, strategies=strategies)
            results.append(
                {
                    "subject_ID": subject,
                    "maze_name": maze,
                    **strategy_weights,
                }
            )
    results_df = pd.DataFrame(results)
    return results_df


def plot_strategy_weights(results_df, ax=None):
    """
    Plots fitted strategy weights per maze. X-axis = maze, y-axis = weight.
    Each strategy gets a distinct color via sns.pointplot (mean ± SE).
    Individual subject values shown in grey via sns.stripplot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    ax.spines[["top", "right"]].set_visible(False)

    # melt wide (subject, maze, strat1, strat2, ...) to long form
    strategies = [c for c in results_df.columns if c not in ("subject_ID", "maze_name")]
    long_df = results_df.melt(
        id_vars=["subject_ID", "maze_name"],
        value_vars=strategies,
        var_name="strategy",
        value_name="weight",
    )

    sns.stripplot(
        data=long_df,
        x="maze_name",
        y="weight",
        hue="strategy",
        order=MAZE_NAMES,
        palette={s: "grey" for s in strategies},
        dodge=True,
        alpha=0.4,
        size=3,
        jitter=True,
        legend=False,
        ax=ax,
    )
    sns.pointplot(
        data=long_df,
        x="maze_name",
        y="weight",
        hue="strategy",
        order=MAZE_NAMES,
        errorbar="se",
        capsize=0,
        linestyle="none",
        dodge=True,
        ax=ax,
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Maze")
    ax.set_ylabel("Strategy weight")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="Strategy", bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    return ax


# %%


def get_model_loglikelihood_comparisons(
    navigation_strategies_df,
    ordered_strategies=["backtracking_penalty", "vector", "habit", "structure"],
    late_sessions=True,
):
    """
    Fits a series of nested models (each adding one strategy in order) and returns
    per-subject fit statistics to test whether each additional strategy improves
    the model. All mazes are pooled together per subject.

    For each adjacent model pair, computes:
      - AIC  = 2k + 2*negLL
      - BIC  = k*ln(n_obs) + 2*negLL
      - LRT statistic = 2*(negLL_simpler - negLL_complex), chi2(df=1) under H0
      - LRT p-value

    Returns a long-form DataFrame with one row per (subject, model).
    """
    df = navigation_strategies_df.copy()
    if late_sessions:
        df = df[df.late_session]

    # nested model ladder: M1=[s0], M2=[s0,s1], ...
    nested_models = [ordered_strategies[: i + 1] for i in range(len(ordered_strategies))]

    results = []
    for subject in SUBJECT_IDS:
        subj_df = df[df.subject_ID == subject]
        n_obs = len(subj_df)

        prev_negll = None
        for strategies in nested_models:
            k = len(strategies)
            result = minimize(
                models.get_neg_loglikelihood,
                np.zeros(k),
                args=(strategies, subj_df),
                method="BFGS",
            )
            negll = result.fun
            aic = 2 * k + 2 * negll
            bic = k * np.log(n_obs) + 2 * negll

            if prev_negll is not None:
                lrt_stat = 2 * (prev_negll - negll)
                lrt_pval = 1 - chi2.cdf(lrt_stat, df=1)
            else:
                lrt_stat = np.nan
                lrt_pval = np.nan

            results.append(
                {
                    "subject_ID": subject,
                    "model": "+".join(strategies),
                    "n_params": k,
                    "n_obs": n_obs,
                    "negll": negll,
                    "aic": aic,
                    "bic": bic,
                    "lrt_stat": lrt_stat,
                    "lrt_pval": lrt_pval,
                    **{s: w for s, w in zip(strategies, result.x)},
                }
            )
            prev_negll = negll

    return pd.DataFrame(results)


def plot_incremental_bic(
    results_df,
    print_stats=True,
    ax=None,
):
    """
    Plots the incremental BIC improvement (ΔBIC) gained by adding each strategy
    to the nested model, pooled across mazes. Bars = mean ± SE across subjects;
    individual subject points overlaid. Wilcoxon significance markers above each bar.
    """
    # set up figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(2, 3))
    ax.spines[["top", "right"]].set_visible(False)

    # ordered model names by number of parameters
    model_order = results_df.groupby("model")["n_params"].first().sort_values().index.tolist()
    added_strategies = [m.split("+")[-1] for m in model_order[1:]]

    # pivot BIC to wide then compute delta for each adjacent model pair
    bic_wide = results_df.pivot_table(index="subject_ID", columns="model", values="bic")
    delta_rows = []
    for prev_model, curr_model in zip(model_order[:-1], model_order[1:]):
        added = curr_model.split("+")[-1]
        for subject, val in (bic_wide[prev_model] - bic_wide[curr_model]).items():
            delta_rows.append({"subject_ID": subject, "added_strategy": added, "delta_bic": val})
    delta_df = pd.DataFrame(delta_rows)

    sns.barplot(
        data=delta_df,
        x="added_strategy",
        y="delta_bic",
        order=added_strategies,
        errorbar="se",
        color="grey",
        alpha=1,
        ax=ax,
    )
    sns.stripplot(
        data=delta_df,
        x="added_strategy",
        y="delta_bic",
        order=added_strategies,
        color="black",
        alpha=0.5,
        size=4,
        jitter=True,
        ax=ax,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    for j, strategy in enumerate(added_strategies):
        vals = delta_df.loc[delta_df.added_strategy == strategy, "delta_bic"].dropna()
        if len(vals) > 1:
            t_stat, pval = ttest_1samp(vals, popmean=0)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            ax.text(j, vals.max() * 1.05, sig, ha="center", va="bottom", fontsize=11)
            if print_stats:
                print(f"+{strategy}: T({len(vals)-1})={t_stat:.3f}, p={pval:.3f}")

    ax.set_xlabel("Added strategy")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("ΔBIC (improvement)")
    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
