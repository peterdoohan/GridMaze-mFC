"""
Strategy-agreement comparisons for the no-opto electrophysiology dataset.

Question: on decision points where named strategies (vector / structure / habit) agree
or disagree on the top action, how often do subjects pick each strategy's top action?

No between-group control here, so chance level is defined per row as
    chance(target) = |top_actions(target) ∩ available| / |available|
and averaged within each subset. This adapts to tied top-actions and to node degree.
"""

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp, ttest_rel

from GridMaze.analysis.navigation_strategies import get_input_data as gid

# %% Globals
NSEW = ["N", "S", "E", "W"]
DEFAULT_STRATEGIES = ["vector", "structure", "habit"]
DEFAULT_MAZE_NAMES = ["maze_1", "maze_2"]
CMAP = "plasma_r"

_PALETTE = dict(zip(DEFAULT_STRATEGIES, sns.color_palette(CMAP, n_colors=len(DEFAULT_STRATEGIES))))


# %% Public plotting functions


def plot_pairwise_disagree(
    navigation_strategies_df,
    strategy_pairs=(("vector", "structure"), ("vector", "habit"), ("structure", "habit")),
    maze_names=DEFAULT_MAZE_NAMES,
    last_n_days_on_maze=5,
    decision_points_only=True,
    print_stats=True,
    axes=None,
):
    """For each pair (A, B), restrict to decision points where A and B disagree on the
    top action, then plot per-subject P(match A) and P(match B) with chance lines.
    """
    df = _filter_df(navigation_strategies_df, maze_names, last_n_days_on_maze, decision_points_only)

    if axes is None:
        fig, axes = plt.subplots(1, len(strategy_pairs), figsize=(2.5 * len(strategy_pairs), 3), sharey=True)
    else:
        fig = axes[0].figure

    for ax, (a, b) in zip(axes, strategy_pairs):
        disagree = ~_strats_agree_mask(df, a, b)
        sub = df.loc[disagree]
        per_subj = _per_subject_means(sub, target_strategies=[a, b])

        if print_stats:
            n_total = len(disagree)
            n_kept = int(disagree.sum())
            print(f"\n--- {a} != {b}: {n_kept}/{n_total} ({100 * n_kept / max(n_total, 1):.1f}%) decisions kept ---")

        _plot_bars_with_chance(
            ax,
            per_subj,
            order=[a, b],
            palette=_PALETTE,
            print_stats=print_stats,
        )
        ax.set_xlabel(f"{a} ≠ {b}")

    axes[0].set_ylabel("P(match strategy)")
    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.spines["left"].set_visible(False)
        ax.yaxis.set_visible(False)

    fig.tight_layout()
    return fig


def plot_scenario_cell(
    navigation_strategies_df,
    constraints=("vector == habit", "vector != structure"),
    target_strategies=DEFAULT_STRATEGIES,
    maze_names=DEFAULT_MAZE_NAMES,
    last_n_days_on_maze=5,
    decision_points_only=True,
    print_stats=True,
    ax=None,
):
    """Restrict to a scenario subset (AND of constraints like 'vector == habit'),
    then plot per-subject P(match target) for each target in target_strategies."""
    df = _filter_df(navigation_strategies_df, maze_names, last_n_days_on_maze, decision_points_only)
    keep = _scenario_keep_mask(df, list(constraints))
    sub = df.loc[keep]
    per_subj = _per_subject_means(sub, target_strategies=list(target_strategies))

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    else:
        fig = ax.figure

    title = " & ".join(constraints)
    if print_stats:
        n_total = len(keep)
        n_kept = int(keep.sum())
        print(f"\n--- [{title}]: {n_kept}/{n_total} ({100 * n_kept / max(n_total, 1):.1f}%) decisions kept ---")

    _plot_bars_with_chance(
        ax,
        per_subj,
        order=list(target_strategies),
        palette=_PALETTE,
        print_stats=print_stats,
    )
    ax.set_xlabel("")
    ax.set_ylabel("P(match target)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()


def smoke_test(navigation_strategies_df=None, last_n_days_on_maze=5):
    """Generate the four headline figures with full stats printout."""
    if navigation_strategies_df is None:
        navigation_strategies_df = gid.get_navigation_strategies_df(verbose=True)

    figs = {}

    print("\n========== Plot 1: pairwise disagree triptych ==========")
    figs["pairwise"] = plot_pairwise_disagree(
        navigation_strategies_df,
        last_n_days_on_maze=last_n_days_on_maze,
    )

    scenarios = {
        "VHnotS": ("vector == habit", "vector != structure"),
        "VSnotH": ("vector == structure", "vector != habit"),
        "SHnotV": ("structure == habit", "structure != vector"),
        "all_disagree": ("habit != vector", "habit != structure", "structure != vector"),
    }
    for key, constraints in scenarios.items():
        print(f"\n========== Plot 2 ({key}): {' & '.join(constraints)} ==========")
        figs[key] = plot_scenario_cell(
            navigation_strategies_df,
            constraints=constraints,
            last_n_days_on_maze=last_n_days_on_maze,
        )


# %% Internals — filtering


def _filter_df(navigation_strategies_df, maze_names, last_n_days_on_maze, decision_points_only):
    df = navigation_strategies_df.copy()
    if maze_names is not None:
        df = df[df.maze_name.isin(maze_names)]
    if last_n_days_on_maze is not None:
        # per-maze max day_on_maze (e.g. 13 for maze_1, 11 for maze_2/rooms_maze) → keep last n days
        day = df.day_on_maze.values
        maze = df.maze_name.values
        max_per_maze = pd.Series(day, index=df.index).groupby(maze).transform("max")
        df = df[day > (max_per_maze.values - last_n_days_on_maze)]
    if decision_points_only:
        df = df[df.available.sum(axis=1).gt(2)]
    return df.reset_index(drop=True)


# %% Internals — strategy-agreement masks (ported from companion-repo strategy_comparisons.py)


def _tied_argmax_mask(values_df, available_df):
    """(n_rows x 4) bool mask of all tied argmax actions, restricted to available."""
    values = values_df.to_numpy(dtype=float)
    avail = available_df.values.astype(bool)
    values = np.where(avail, values, -np.inf)
    row_max = np.nanmax(values, axis=1)
    return np.isclose(values, row_max[:, None], rtol=0.0, atol=0.0, equal_nan=False)


def _strats_agree_mask(df, strat_a, strat_b):
    a = _tied_argmax_mask(df[strat_a], df.available)
    b = _tied_argmax_mask(df[strat_b], df.available)
    return pd.Series((a & b).any(axis=1), index=df.index)


def _parse_constraint(constraint):
    if "==" in constraint:
        a, b = constraint.split("==")
        return a.strip(), True, b.strip()
    if "!=" in constraint:
        a, b = constraint.split("!=")
        return a.strip(), False, b.strip()
    raise ValueError(f"Cannot parse constraint: {constraint!r}. Expected 'A == B' or 'A != B'.")


def _scenario_keep_mask(df, constraints):
    keep = pd.Series(True, index=df.index)
    for c in constraints:
        a, want_agree, b = _parse_constraint(c)
        agree = _strats_agree_mask(df, a, b)
        keep &= agree if want_agree else ~agree
    return keep


# %% Internals — observed and chance match rates


def _correct(df, target_strategy):
    chosen = (df.subject_choice == 1).values
    top = _tied_argmax_mask(df[target_strategy], df.available)
    return pd.Series((chosen & top).any(axis=1), index=df.index)


def _chance_match(df, target_strategy):
    """Per-row P(match target | uniform random over available) = |top ∩ avail| / |avail|."""
    top = _tied_argmax_mask(df[target_strategy], df.available)
    avail = df.available.values.astype(bool)
    n_top = top.sum(axis=1)
    n_avail = avail.sum(axis=1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        chance = np.where(n_avail > 0, n_top / n_avail, np.nan)
    return pd.Series(chance, index=df.index)


def _per_subject_means(df_subset, target_strategies):
    """Tidy long DataFrame: one row per (subject_ID, target_strategy) with p_observed,
    p_chance, n_decisions. Subjects with zero rows in `df_subset` are dropped."""
    if len(df_subset) == 0:
        return pd.DataFrame(columns=["subject_ID", "target_strategy", "p_observed", "p_chance", "n_decisions"])
    subject_ids = df_subset.subject_ID.values
    rows = []
    for target in target_strategies:
        correct = _correct(df_subset, target).values.astype(float)
        chance = _chance_match(df_subset, target).values
        for subj in pd.unique(subject_ids):
            mask = subject_ids == subj
            if not mask.any():
                continue
            rows.append(
                {
                    "subject_ID": subj,
                    "target_strategy": target,
                    "p_observed": float(np.nanmean(correct[mask])),
                    "p_chance": float(np.nanmean(chance[mask])),
                    "n_decisions": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


# %% Internals — plotting


def _plot_bars_with_chance(ax, per_subj, order, palette, print_stats=True):
    """Stripplot of subjects + pointplot of mean ± SE, with per-bar dashed chance ticks
    and one-sample-vs-chance asterisks above each bar. Also runs and prints pairwise
    paired t-tests between bars when `print_stats`."""
    if len(per_subj) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.spines[["top", "right"]].set_visible(False)
        return

    sns.stripplot(
        data=per_subj,
        x="target_strategy",
        y="p_observed",
        order=order,
        color="grey",
        alpha=0.4,
        size=4,
        jitter=True,
        ax=ax,
    )
    sns.pointplot(
        data=per_subj,
        x="target_strategy",
        y="p_observed",
        order=order,
        hue="target_strategy",
        hue_order=order,
        palette=palette,
        errorbar="se",
        capsize=0,
        linestyle="none",
        markersize=7,
        zorder=3,
        ax=ax,
    )
    if ax.get_legend():
        ax.get_legend().remove()
    ax.spines[["top", "right"]].set_visible(False)

    # per-bar chance tick + one-sample t-test annotation
    pivot_obs = per_subj.pivot(index="subject_ID", columns="target_strategy", values="p_observed")
    pivot_ch = per_subj.pivot(index="subject_ID", columns="target_strategy", values="p_chance")
    for j, target in enumerate(order):
        chance_mean = float(pivot_ch[target].mean())
        ax.hlines(chance_mean, j - 0.3, j + 0.3, color="black", linestyle="--", linewidth=0.8)

        diff = (pivot_obs[target] - pivot_ch[target]).dropna()
        if len(diff) > 1:
            t, p = ttest_1samp(diff, popmean=0.0)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            y_top = float(pivot_obs[target].dropna().max())
            ax.text(j, y_top + 0.03, sig, ha="center", va="bottom", fontsize=10)
            if print_stats:
                print(
                    f"  {target} vs chance ({chance_mean:.3f}): "
                    f"T({len(diff) - 1})={t:.3f}, p={p:.3g}, n={len(diff)}"
                )

    # pairwise paired t-tests between bars
    if print_stats and len(order) >= 2:
        for i in range(len(order)):
            for k in range(i + 1, len(order)):
                a_name, b_name = order[i], order[k]
                paired = pivot_obs[[a_name, b_name]].dropna()
                if len(paired) > 1:
                    t, p = ttest_rel(paired[a_name], paired[b_name])
                    print(
                        f"  P({a_name})={paired[a_name].mean():.3f} vs "
                        f"P({b_name})={paired[b_name].mean():.3f}: "
                        f"T({len(paired) - 1})={t:.3f}, p={p:.3g}, n={len(paired)}"
                    )

    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
