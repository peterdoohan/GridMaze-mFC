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
from joblib import delayed, Parallel

from GridMaze.analysis.navigation_strategies import get_input_data as gid
from GridMaze.analysis.navigation_strategies import models

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS2_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]
RESULTS_DIR = RESULTS2_PATH / "navigation_strategies"

# %% strategy weights over days (learning curve)


def get_strategy_weights_over_days(
    navigation_strategies_df,
    strategies=["backtracking_penalty", "vector", "habit", "structure"],
    n_iter=1_000,
    random_seed=0,
    zscore=True,
    n_jobs=-1,
    save=False,
    verbose=False,
):
    """
    Estimates navigation strategy weights as a function of day on maze using
    bootstrap resampling over subjects.

    For each iteration, subjects are resampled with replacement. For each
    maze x day combination, decisions from all resampled subjects are pooled
    (with a subject sampled k times contributing k copies of their decisions)
    and strategy weights are fit on the pooled data. The std of weights across
    iterations approximates the SEM across subjects.

    Set n_jobs to an integer or -1 to parallelise across iterations with joblib.
    Returns a long-form DataFrame with one row per (bootstrap_iter, maze, day).
    """
    save_path = RESULTS_DIR / f"strategy_weights_over_days_niter{n_iter}.parquet"
    if save_path.exists() and not save:
        if verbose:
            print(f"Loading from {save_path}")
        return pd.read_parquet(save_path)

    df = navigation_strategies_df.copy()

    # index of all (maze, day) pairs present in the data
    maze_day_pairs = (
        df[["maze_name", "day_on_maze"]].drop_duplicates().sort_values(["maze_name", "day_on_maze"]).values.tolist()
    )

    # pre-draw all subject samples so results are reproducible regardless of n_jobs
    rng = np.random.default_rng(random_seed)
    all_sampled_subjects = [rng.choice(SUBJECT_IDS, size=len(SUBJECT_IDS), replace=True) for _ in range(n_iter)]

    def _run_iter(i, sampled_subjects):
        if verbose:
            print(f"Bootstrap iteration {i + 1}/{n_iter}")
        iter_results = []
        for maze, day in maze_day_pairs:
            maze_day_df = df[(df.maze_name == maze) & (df.day_on_maze == day)]

            # concatenate per-subject slices so duplicated subjects contribute
            # duplicated decisions (preserve bootstrap weighting)
            subject_slices = [maze_day_df[maze_day_df.subject_ID == s] for s in sampled_subjects]
            pooled_df = pd.concat(subject_slices, ignore_index=True)

            if pooled_df.empty:
                continue

            weights = models.get_navigation_strategy_weights(pooled_df, strategies=strategies, zscore=zscore)
            iter_results.append({"bootstrap_iter": i, "maze_name": maze, "day_on_maze": day, **weights})
        return iter_results

    if n_jobs:
        nested = Parallel(n_jobs=n_jobs)(
            delayed(_run_iter)(i, sampled_subjects) for i, sampled_subjects in enumerate(all_sampled_subjects)
        )
        results = [row for iter_rows in nested for row in iter_rows]
    else:
        results = []
        for i, sampled_subjects in enumerate(all_sampled_subjects):
            results.extend(_run_iter(i, sampled_subjects))

    results_df = pd.DataFrame(results)

    if save:
        if verbose:
            print(f"Saving to {save_path}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(save_path)

    return results_df


def plot_strategy_weights_over_days(
    results_df,
    strategies=["habit", "vector", "structure"],
    cmap="plasma_r",
    colors=None,
    moving_avg=2,
    axes=None,
):
    """
    Plots strategy weights as a function of day on maze, one subplot per maze.
    Mean across bootstrap iterations shown as a solid line; std shown as shaded band.
    If moving_avg > 1, a rolling average of that window size is applied to both
    mean and std before plotting.
    """
    if axes is None:
        fig, axes = plt.subplots(1, len(MAZE_NAMES), figsize=(4, 2), sharey=True)

    if colors is None:
        strategy_colors = sns.color_palette(cmap, n_colors=len(strategies))
    else:
        strategy_colors = colors
    palette = dict(zip(strategies, strategy_colors))
    for ax, maze in zip(axes, MAZE_NAMES):
        maze_df = results_df[results_df.maze_name == maze]
        stats = maze_df.groupby("day_on_maze")[strategies].agg(["mean", "std"])

        for strategy in strategies:
            days = stats.index.values
            mean = stats[(strategy, "mean")].rolling(moving_avg, center=True, min_periods=1).mean().values
            std = stats[(strategy, "std")].rolling(moving_avg, center=True, min_periods=1).mean().values
            color = palette[strategy]
            ax.plot(days, mean, color=color, linewidth=1.5, label=strategy)
            ax.fill_between(days, mean - std, mean + std, color=color, alpha=0.2)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Day on maze")
        ax.set_title(maze.replace("_", " "))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax != axes[0]:
            ax.spines["left"].set_visible(False)
            ax.yaxis.set_visible(False)

    axes[0].set_ylabel("Strategy weight")

    handles = [plt.Line2D([0], [0], color=palette[s], linewidth=2, label=s) for s in strategies]
    axes[-1].legend(handles=handles, title="Strategy", bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)


# %% standard late session model fits


def get_strategy_weights(
    navigation_strategies_df,
    strategies=["vector", "structure", "habit", "backtracking_penalty"],
    late_sessions=True,
    zscore=True,
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
            strategy_weights = models.get_navigation_strategy_weights(
                subj_df,
                strategies=strategies,
                zscore=zscore,
            )
            results.append(
                {
                    "subject_ID": subject,
                    "maze_name": maze,
                    **strategy_weights,
                }
            )
    results_df = pd.DataFrame(results)
    return results_df


def plot_strategy_weights(
    results_df,
    strategies=["habit", "vector", "structure"],
    cmap="plasma_r",
    colors=None,
    print_stats=True,
    axes=None,
):
    """
    Plots fitted strategy weights with one panel per maze. X-axis = strategy,
    y-axis = weight. Pointplot shows mean ± SE; stripplot shows individual subjects in grey.
    Panels share the y-axis and are styled to look like a single figure.
    """
    if axes is None:
        fig, axes = plt.subplots(1, len(MAZE_NAMES), figsize=(2.5, 3), sharey=True)

    long_df = results_df.melt(
        id_vars=["subject_ID", "maze_name"],
        value_vars=strategies,
        var_name="strategy",
        value_name="weight",
    )

    if colors is None:
        strategy_colors = sns.color_palette(cmap, n_colors=len(strategies))
    else:
        strategy_colors = colors
    palette = dict(zip(strategies, strategy_colors))

    for ax, maze in zip(axes, MAZE_NAMES):
        maze_df = long_df[long_df.maze_name == maze]
        sns.stripplot(
            data=maze_df,
            x="strategy",
            y="weight",
            order=strategies,
            color="grey",
            alpha=0.4,
            size=4,
            jitter=True,
            ax=ax,
        )
        sns.pointplot(
            data=maze_df,
            x="strategy",
            y="weight",
            order=strategies,
            hue="strategy",
            hue_order=strategies,
            palette=palette,
            errorbar="se",
            capsize=0,
            linestyle="none",
            markersize=7,
            zorder=3,
            ax=ax,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title("")
        ax.set_xlabel(maze.replace("_", " "))
        ax.set_xticks([])
        if ax.get_legend():
            ax.get_legend().remove()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax != axes[0]:
            ax.spines["left"].set_visible(False)
            ax.yaxis.set_visible(False)

    axes[0].set_ylabel("Strategy weight")

    # build legend from pointplot colors on the last axis
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[s], markersize=7, label=s)
        for s in strategies
    ]
    axes[-1].legend(handles=handles, title="Strategy", bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)

    if print_stats:
        for maze in MAZE_NAMES:
            print(maze)
            maze_df = results_df[results_df.maze_name == maze]
            for strategy in strategies:
                vals = maze_df[strategy].dropna()
                t_stat, pval = ttest_1samp(vals, popmean=0)
                print(f"  {strategy}: T({len(vals)-1})={t_stat:.3f}, p={pval:.3f}")


# %% model justification analyese and plots


def get_cv_model_comparisons(
    navigation_strategies_df,
    strategies=["backtracking_penalty", "vector", "habit", "structure"],
    late_sessions=True,
    k=10,
    random_seed=0,
    verbose=True,
):
    """
    Leave-one-strategy-out cross-validated model comparison.

    Fits a full model and k reduced models (each with one strategy removed) using
    k-fold cross-validation. Folds are constructed by randomly partitioning
    trial_unique_IDs, so all decisions from a trial stay in the same fold.
    Weights are fit on k-1 folds (train) and negLL is evaluated on the held-out
    fold (test).

    Returns a long-form DataFrame with one row per (subject, fold, model).
    """
    rng = np.random.default_rng(random_seed)

    df = navigation_strategies_df.copy()
    if late_sessions:
        df = df[df.late_session]

    # full model + one reduced model per strategy
    model_specs = {"full": strategies}
    for s in strategies:
        model_specs[f"no_{s}"] = [x for x in strategies if x != s]

    results = []
    for subject in SUBJECT_IDS:
        if verbose:
            print(subject)
        subj_df = df[df.subject_ID == subject]

        # shuffle trial IDs and split into k folds
        trial_ids = subj_df.trial_unique_ID.unique()
        rng.shuffle(trial_ids)
        folds = np.array_split(trial_ids, k)

        for fold_idx, test_trials in enumerate(folds):
            if verbose:
                print(f"fold {fold_idx} of {k}")
            train_trials = np.concatenate([folds[i] for i in range(k) if i != fold_idx])
            train_df = subj_df[subj_df.trial_unique_ID.isin(train_trials)]
            test_df = subj_df[subj_df.trial_unique_ID.isin(test_trials)]

            for model_name, model_strategies in model_specs.items():
                n_params = len(model_strategies)
                fit = minimize(
                    models.get_neg_loglikelihood,
                    np.zeros(n_params),
                    args=(model_strategies, train_df),
                    method="BFGS",
                )
                test_negll = models.get_neg_loglikelihood(fit.x, model_strategies, test_df)
                results.append(
                    {
                        "subject_ID": subject,
                        "fold": fold_idx,
                        "model": model_name,
                        "removed_strategy": None if model_name == "full" else model_name[3:],
                        "test_negll": test_negll,
                        "n_train_trials": len(train_trials),
                        "n_test_decisions": len(test_df),
                    }
                )

    return pd.DataFrame(results)


def plot_strategy_delta_negLL(results_df, print_stats=True, ax=None):
    """
    Plots the cross-validated log-likelihood cost of removing each strategy.

    For each subject, averages test_negll across folds, then computes:
        delta_negll = mean_test_negll_reduced - mean_test_negll_full
    Positive values mean removing that strategy hurt generalisation (full model better).

    X-axis = removed strategy, y-axis = delta test negLL.
    Pointplot shows mean ± SE across subjects; stripplot shows individual subjects in grey.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Removed strategy")
    ax.set_ylabel("Δ cv negLL \n (reduced - full)")

    # average test negLL across folds per subject per model
    mean_negll = results_df.groupby(["subject_ID", "model"])["test_negll"].mean().reset_index()

    # pivot to wide: columns = model names, index = subject
    negll_wide = mean_negll.pivot(index="subject_ID", columns="model", values="test_negll")

    # compute delta per reduced model
    reduced_models = [m for m in negll_wide.columns if m != "full"]
    delta_rows = []
    for model in reduced_models:
        removed = model[3:]  # strip "no_"
        for subject, val in (negll_wide[model] - negll_wide["full"]).items():
            delta_rows.append({"subject_ID": subject, "removed_strategy": removed, "delta_negll": val})
    delta_df = pd.DataFrame(delta_rows)

    removed_strategies = [m[3:] for m in reduced_models]
    sns.stripplot(
        data=delta_df,
        x="removed_strategy",
        y="delta_negll",
        order=removed_strategies,
        color="grey",
        alpha=0.5,
        size=4,
        jitter=True,
        ax=ax,
    )
    sns.pointplot(
        data=delta_df,
        x="removed_strategy",
        y="delta_negll",
        order=removed_strategies,
        errorbar="se",
        capsize=0,
        linestyle="none",
        color="black",
        zorder=3,
        ax=ax,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    for j, strategy in enumerate(removed_strategies):
        vals = delta_df.loc[delta_df.removed_strategy == strategy, "delta_negll"].dropna()
        if len(vals) > 1:
            t_stat, pval = ttest_1samp(vals, popmean=0)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            ax.text(j, vals.max() * 1.05, sig, ha="center", va="bottom", fontsize=11)
            if print_stats:
                print(f"no_{strategy}: T({len(vals)-1})={t_stat:.3f}, p={pval:.3f}")

    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


def get_model_comparisons(
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


def plot_strategy_bic(
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
