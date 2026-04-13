"""
Look at decoding performance of distance-to-goal aligned to navigational errors
Does the internal representation move with subject's internal estimate of distance even
when it is wrong?
@peterdoohan
"""

# %% Imports
import json
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import delayed, Parallel
from matplotlib import pyplot as plt
from pingouin import rm_anova
from tabulate import tabulate

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.distance_to_goal import logreg_decoder as lr

# %% Global Variables

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "distance_to_goal" / "errors"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)


# %% Run analysis


def run(window=(-1.0, 1.0), resolution=0.2, decision_window=0.8, decode_method="weighted", var_metric=None):
    decoding_df = get_distance_to_goal_decoding_df()
    print(f"Loaded decoding df: {decoding_df.shape}")

    aligned, distance_bins = align_probs_to_decisions(decoding_df, window=window, resolution=resolution)
    n_errors = (aligned.event_type == "error").sum() // len(aligned.time_relative.unique())
    n_correct = (aligned.event_type == "correct").sum() // len(aligned.time_relative.unique())
    print(f"Aligned: {len(aligned)} rows, {n_errors} error events, {n_correct} correct events")

    subject_heatmaps, group_heatmaps = build_conditional_heatmaps(
        aligned, distance_bins, decision_window=decision_window
    )
    print(f"Subject heatmaps: {subject_heatmaps.shape}, Group heatmaps: {group_heatmaps.shape}")

    plot_conditional_heatmaps(group_heatmaps, subject_heatmaps, distance_bins, decode_method=decode_method)
    fig, metric_df = plot_decoding_bias(
        subject_heatmaps,
        distance_bins,
        print_stats=True,
        decode_method=decode_method,
        var_metric=var_metric,
    )
    plt.show()

    return aligned, subject_heatmaps, group_heatmaps, metric_df


# %% Stage 1: Align decoded probabilities to decision events


def align_probs_to_decisions(df, window=(-1.0, 1.0), resolution=0.2):
    """
    Extract windows of decoded distance probabilities aligned to nav_error and nav_correct events.

    Parameters
    ----------
    df : pd.DataFrame
        Output of get_distance_to_goal_decoding_df(). MultiIndex columns with
        decoded_distance_prob, distance_bin_mid, nav_error, nav_correct, etc.
    window : tuple
        (start, end) in seconds relative to decision event.
    resolution : float
        Temporal resolution in seconds.

    Returns
    -------
    aligned_df : pd.DataFrame
        Long-format DataFrame with columns: time_relative, 31 decoded prob columns
        (float-named), true_distance, event_type, subject_ID, event_id.
    distance_bins : np.ndarray
        The 31 distance bin midpoints (float).
    """
    _df = df.reset_index(drop=True)

    # extract distance bins from decoded_distance_prob columns
    distance_bins = _df["decoded_distance_prob"].columns.values.astype(float)

    frames_before = int(-window[0] / resolution)
    frames_after = int(window[1] / resolution)
    expected_length = frames_before + frames_after + 1
    aligned_times = np.round(np.linspace(window[0], window[1], expected_length), 2)

    rows = []
    event_counter = 0

    for subject_ID in _df.subject_ID.unique():
        subj_df = _df[_df.subject_ID == subject_ID].reset_index(drop=True)

        for tuID in subj_df.trial_unique_ID.unique():
            trial_mask = subj_df.trial_unique_ID == tuID
            trial_df = subj_df[trial_mask]

            for event_col, event_label in [("nav_error", "error"), ("nav_correct", "correct")]:
                event_idxs = trial_df.index[trial_df[event_col]]
                if len(event_idxs) == 0:
                    continue

                for idx in event_idxs:
                    start_idx = idx - frames_before
                    end_idx = idx + frames_after

                    if start_idx < 0 or end_idx >= len(subj_df):
                        continue

                    chunk = subj_df.iloc[start_idx : end_idx + 1]
                    if len(chunk) != expected_length:
                        continue

                    # mask out-of-trial rows
                    same_trial = (chunk.trial_unique_ID == tuID).values

                    probs = chunk["decoded_distance_prob"].values.astype(float).copy()
                    true_dist = chunk["distance_bin_mid"].values.astype(float).copy()
                    probs[~same_trial] = np.nan
                    true_dist[~same_trial] = np.nan

                    for t_i, t in enumerate(aligned_times):
                        row = {
                            "time_relative": t,
                            "true_distance": true_dist[t_i],
                            "event_type": event_label,
                            "subject_ID": subject_ID,
                            "event_id": event_counter,
                        }
                        for b_i, b in enumerate(distance_bins):
                            row[b] = probs[t_i, b_i]
                        rows.append(row)

                    event_counter += 1

    aligned_df = pd.DataFrame(rows)
    return aligned_df, distance_bins


# %% Stage 2: Build conditional heatmaps


def build_conditional_heatmaps(aligned_df, distance_bins, decision_window=0.8, min_decisions=10):
    """
    Build P(decoded | true_distance, event_type, period) averaged within then across subjects.

    Timepoints are consolidated into pre-decision (-decision_window <= t < 0) and
    post-decision (0 < t <= decision_window) periods. The decision point (t=0) is excluded.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        Output of align_probs_to_decisions (first element of tuple).
    distance_bins : np.ndarray
        The 31 distance bin midpoints (from align_probs_to_decisions).
    decision_window : float
        Window size in seconds for pre/post periods (default 0.8).
    min_decisions : int
        Minimum number of unique decision events required per
        (subject_ID, event_type, true_distance, period) cell. Cells with fewer
        decisions are dropped to avoid noisy probability estimates.

    Returns
    -------
    subject_heatmaps : pd.DataFrame
        Per-subject conditional distributions, indexed by
        (subject_ID, event_type, true_distance, period).
    group_heatmaps : pd.DataFrame
        Group-averaged conditional distributions, indexed by
        (event_type, true_distance, period).
    """
    prob_cols = list(distance_bins)

    df = aligned_df.dropna(subset=["true_distance"]).copy()

    # map time_relative to pre/post period
    t = df["time_relative"]
    mask_pre = (t >= -decision_window) & (t < 0)
    mask_post = (t > 0) & (t <= decision_window)
    df["period"] = None
    df.loc[mask_pre, "period"] = "pre"
    df.loc[mask_post, "period"] = "post"
    df = df[df["period"].notna()]

    # filter by minimum number of unique decisions per cell
    group_keys = ["subject_ID", "event_type", "true_distance", "period"]
    decision_counts = df.groupby(group_keys)["event_id"].nunique()
    valid_cells = decision_counts[decision_counts >= min_decisions].index
    df = df.set_index(group_keys).loc[valid_cells].reset_index()

    # average within subject first
    subject_heatmaps = df.groupby(group_keys)[prob_cols].mean()

    # only keep (event_type, true_distance, period) cells defined for ALL subjects
    n_subjects = subject_heatmaps.index.get_level_values("subject_ID").nunique()
    subj_counts = subject_heatmaps.groupby(["event_type", "true_distance", "period"]).size()
    all_subjects_cells = subj_counts[subj_counts == n_subjects].index
    subject_heatmaps = subject_heatmaps.loc[subject_heatmaps.index.droplevel("subject_ID").isin(all_subjects_cells)]

    # then average across subjects
    group_heatmaps = subject_heatmaps.groupby(["event_type", "true_distance", "period"]).mean()

    return subject_heatmaps, group_heatmaps


# %% Helpers


def _decoded_distance(probs, distance_bins, method="weighted"):
    """Compute decoded distance from probability vectors.

    Parameters
    ----------
    probs : np.ndarray
        Probability array of shape (..., n_bins).
    distance_bins : np.ndarray
        Distance bin midpoints of shape (n_bins,).
    method : str
        "weighted" — probability-weighted mean (E[decoded] = probs @ bins).
        "max" — distance bin with highest probability (argmax).
    """
    if method == "weighted":
        return probs @ distance_bins
    elif method == "max":
        return distance_bins[np.argmax(probs, axis=-1)]
    else:
        raise ValueError(f"method must be 'weighted' or 'max', got '{method}'")


def _decoded_variability(probs, distance_bins, method="sd"):
    """Compute variability/uncertainty of decoded probability distributions.

    Parameters
    ----------
    probs : np.ndarray
        Probability array of shape (..., n_bins).
    distance_bins : np.ndarray
        Distance bin midpoints of shape (n_bins,).
    method : str
        "sd" — standard deviation of the distribution (in metres).
        "entropy" — Shannon entropy in bits.
        "max_prob" — peak probability (lower = flatter).
    """
    if method == "sd":
        mean = probs @ distance_bins  # (...,)
        variance = probs @ (distance_bins**2) - mean**2
        return np.sqrt(np.maximum(variance, 0))
    elif method == "entropy":
        p = np.clip(probs, 1e-12, None)
        return -np.sum(p * np.log2(p), axis=-1)
    elif method == "max_prob":
        return np.max(probs, axis=-1)
    else:
        raise ValueError(f"method must be 'sd', 'entropy', or 'max_prob', got '{method}'")


# %% Stage 3a: Plot conditional heatmaps


def plot_conditional_heatmaps(group_heatmaps, subject_heatmaps, distance_bins, vmax=0.15, decode_method="weighted"):
    """
    Plot 2x3 grid of conditional heatmaps: rows = pre/post decision, cols = correct/error/difference.
    Difference panels include E[decoded] lines (mean ± SEM across subjects) for both conditions.
    """
    all_true_dists = group_heatmaps.index.get_level_values("true_distance").unique().sort_values()
    tick_idx = np.arange(0, len(all_true_dists), 5)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for row_idx, period in enumerate(["pre", "post"]):
        heatmaps = {}

        # correct and error heatmaps
        for event_type, col_idx, title in zip(["correct", "error"], [0, 1], ["Correct", "Error"]):
            ax = axes[row_idx, col_idx]
            try:
                data = group_heatmaps.loc[event_type].xs(period, level="period")
            except KeyError:
                ax.set_title(f"{title} ({period})\n(no data)")
                continue

            matrix = data.reindex(all_true_dists).values
            heatmaps[event_type] = matrix

            sns.heatmap(
                matrix.T,
                ax=ax,
                cmap="Greys",
                vmin=0,
                vmax=vmax,
                cbar_kws={"shrink": 0.7},
            )
            ax.invert_yaxis()
            ax.set_xticks(tick_idx + 0.5)
            ax.set_xticklabels(np.round(all_true_dists[tick_idx], 2), rotation=0)
            ax.set_yticks(tick_idx + 0.5)
            ax.set_yticklabels(np.round(distance_bins[tick_idx], 2))
            ax.set_xlabel("True distance (m)")
            ax.set_ylabel("Decoded distance (m)")
            ax.set_title(f"{title} ({period}-decision)")
            ax.plot([0, len(all_true_dists)], [0, len(distance_bins)], "b--", alpha=0.5, lw=1)

        # difference heatmap with E[decoded] overlay
        diff_ax = axes[row_idx, 2]
        if "correct" in heatmaps and "error" in heatmaps:
            diff = heatmaps["error"] - heatmaps["correct"]
            vabs = np.nanmax(np.abs(diff))
            sns.heatmap(
                diff.T,
                ax=diff_ax,
                cmap="RdBu_r",
                center=0,
                vmin=-vabs,
                vmax=vabs,
                cbar_kws={"shrink": 0.7},
            )
            diff_ax.invert_yaxis()
            diff_ax.set_xticks(tick_idx + 0.5)
            diff_ax.set_xticklabels(np.round(all_true_dists[tick_idx], 2), rotation=0)
            diff_ax.set_yticks(tick_idx + 0.5)
            diff_ax.set_yticklabels(np.round(distance_bins[tick_idx], 2))
            diff_ax.set_xlabel("True distance (m)")
            diff_ax.set_ylabel("Decoded distance (m)")
            diff_ax.set_title(f"Error - Correct ({period}-decision)")
            diff_ax.plot([0, len(all_true_dists)], [0, len(distance_bins)], "k--", alpha=0.3, lw=1)

            # overlay E[decoded] lines for each condition
            for event_type, color in [("correct", "k"), ("error", "royalblue")]:
                try:
                    subj_data = subject_heatmaps.xs(event_type, level="event_type").xs(period, level="period")
                except KeyError:
                    continue

                # E[decoded] per (subject, true_distance)
                prob_cols = list(distance_bins)
                e_decoded = _decoded_distance(subj_data[prob_cols].values, distance_bins, method=decode_method)
                subj_df = subj_data.reset_index()[["subject_ID", "true_distance"]].copy()
                subj_df["e_decoded"] = e_decoded

                # mean and sem across subjects
                stats = subj_df.groupby("true_distance")["e_decoded"].agg(["mean", "sem"])
                stats = stats.reindex(all_true_dists)

                # map to heatmap bin-index space (0.5 offset for cell centers)
                true_x = np.arange(len(all_true_dists)) + 0.5
                decoded_y_mean = np.interp(stats["mean"].values, distance_bins, np.arange(len(distance_bins))) + 0.5
                decoded_y_lo = (
                    np.interp((stats["mean"] - stats["sem"]).values, distance_bins, np.arange(len(distance_bins))) + 0.5
                )
                decoded_y_hi = (
                    np.interp((stats["mean"] + stats["sem"]).values, distance_bins, np.arange(len(distance_bins))) + 0.5
                )

                valid = ~np.isnan(stats["mean"].values)
                diff_ax.plot(true_x[valid], decoded_y_mean[valid], color=color, lw=2, label=event_type)
                diff_ax.fill_between(true_x[valid], decoded_y_lo[valid], decoded_y_hi[valid], color=color, alpha=0.25)

            diff_ax.legend(frameon=False, fontsize=7, loc="upper left")

    fig.tight_layout()
    return fig


# %% Stage 3b: Plot decoding bias with RM ANOVA


def plot_decoding_bias(
    subject_heatmaps,
    distance_bins,
    ax=None,
    print_stats=False,
    decode_method="weighted",
    var_metric=None,
):
    """
    Plot pre vs post decision metric for error and correct decisions.

    Uses sns.pointplot with paired subject lines. Optionally runs a two-way repeated-measures
    ANOVA (event_type x period) and prints results on the plot.

    Parameters
    ----------
    decode_method : str
        "weighted" or "max" — how to derive decoded distance (used when var_metric is None).
    var_metric : str or None
        None — plot shift (decoded - true distance).
        "sd" — standard deviation of decoded distribution (metres).
        "entropy" — Shannon entropy of decoded distribution (bits).
        "max_prob" — peak probability of decoded distribution.
    """
    prob_cols = list(distance_bins)
    probs = subject_heatmaps[prob_cols].values
    df = subject_heatmaps.reset_index()

    # compute metric
    if var_metric is not None:
        df["metric"] = _decoded_variability(probs, distance_bins, method=var_metric)
        ylabel = {"sd": "Decoded SD (m)", "entropy": "Decoded entropy (bits)", "max_prob": "Max decoded prob"}[
            var_metric
        ]
        metric_name = var_metric
    else:
        expected_decoded = _decoded_distance(probs, distance_bins, method=decode_method)
        df["metric"] = expected_decoded - df["true_distance"]
        ylabel = "Decoded - True distance (m)"
        metric_name = "shift"

    # only average over true distance bins defined for BOTH correct and error
    # within each (subject, period) — ensures matched comparison
    matched_rows = []
    for _, grp in df.groupby(["subject_ID", "period"]):
        correct_dists = set(grp.loc[grp["event_type"] == "correct", "true_distance"])
        error_dists = set(grp.loc[grp["event_type"] == "error", "true_distance"])
        shared_dists = correct_dists & error_dists
        matched_rows.append(grp[grp["true_distance"].isin(shared_dists)])
    df = pd.concat(matched_rows, ignore_index=True)

    # average across matched true distances -> (subject, event_type, period)
    metric_df = df.groupby(["subject_ID", "event_type", "period"])["metric"].mean().reset_index()

    # set up figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    else:
        fig = ax.get_figure()

    event_types = ["correct", "error"]
    x_pos = {e: i for i, e in enumerate(event_types)}
    colors = {"pre": "grey", "post": "royalblue"}

    ax.spines[["top", "right"]].set_visible(False)
    if var_metric is None:
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xticks([x_pos[e] for e in event_types])
    ax.set_xticklabels(["Correct", "Error"])
    ax.set_xlim(-0.4, len(event_types) - 0.6)

    # paired subject lines within each event type (connecting pre to post)
    for event_type in event_types:
        et_df = metric_df[metric_df["event_type"] == event_type]
        pivot = et_df.pivot(index="subject_ID", columns="period", values="metric")
        x_pre = x_pos[event_type] - 0.15
        x_post = x_pos[event_type] + 0.15
        for _, row in pivot.iterrows():
            ax.plot(
                [x_pre, x_post],
                [row["pre"], row["post"]],
                "-",
                color="lightgrey",
                lw=1.5,
                alpha=0.8,
            )

    # group mean ± SEM via pointplot
    sns.pointplot(
        data=metric_df,
        x="event_type",
        order=event_types,
        y="metric",
        hue="period",
        hue_order=["pre", "post"],
        dodge=0.3,
        linestyle="none",
        errorbar="se",
        palette=[colors["pre"], colors["post"]],
        ax=ax,
    )
    sns.move_legend(
        ax,
        "lower center",
        ncol=2,
        title="period",
        frameon=True,
        fontsize="x-small",
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)

    if print_stats:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            stats_df = rm_anova(
                dv="metric",
                within=["event_type", "period"],
                subject="subject_ID",
                data=metric_df,
            )
        # rm_anova Source names: "event_type", "period", "event_type * period"
        source_map = dict(zip(stats_df["Source"], range(len(stats_df))))
        event = stats_df.iloc[source_map["event_type"]]
        period_row = stats_df.iloc[source_map["period"]]
        inter_key = [s for s in source_map if "*" in s][0]
        inter = stats_df.iloc[source_map[inter_key]]
        textstr = (
            f"Event: p={event['p-unc']:.3f}\n"
            f"Period: p={period_row['p-unc']:.3f}\n"
            f"Int:     p={inter['p-unc']:.3f}\n"
        )
        ax.text(0.05, 0.80, textstr, transform=ax.transAxes, fontsize=8)
        print(f"RM ANOVA: {metric_name}")
        print(tabulate(stats_df, headers="keys", tablefmt="psql", showindex=False))

    fig.tight_layout()
    return fig, metric_df


# %%


def get_distance_to_goal_decoding_df(sessions=None, resolution=0.2, verbose=True, save=False, n_jobs=-1):
    """
    slighly different params than logreg decoder
    """
    save_path = RESULTS_DIR / f"distance_to_goal_decoding_probs.parquet"
    if not save and save_path.exists():
        if verbose:
            print(f"Loading existing decoding df from {save_path}")
        return pd.read_parquet(save_path)

    if sessions is None:
        if verbose:
            print("Loading sessions...")
        sessions = gs.get_maze_sessions(
            subject_IDs="all",
            maze_names="all",
            days_on_maze="late",
            with_data=[
                "navigation_df",
                "navigation_spike_counts_df",
                "trajectory_decisions_df",
                "cluster_metrics",
                "trials_df",
                "events_df",
            ],
            must_have_data=True,
        )

    if n_jobs:
        results_dfs = Parallel(n_jobs=n_jobs)(
            delayed(lr.decode_session_distance_to_goal)(
                session,
                resolution=resolution,
                verbose=verbose,
            )
            for session in sessions
        )
    else:
        results_dfs = [
            lr.decode_session_distance_to_goal(
                session,
                resolution=resolution,
                verbose=verbose,
            )
            for session in sessions
        ]
    distance_to_goal_decoding_df = pd.concat(results_dfs, ignore_index=True)
    # save
    if save:
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
        distance_to_goal_decoding_df.to_parquet(save_path)
        if verbose:
            print(f"Saved decoding df to {save_path}")
    return distance_to_goal_decoding_df
