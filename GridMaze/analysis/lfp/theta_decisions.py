"""
Decision-time LFP analyses: relate behavioural strategy use (habit vs structure)
to LFP signals around the moment of each choice.

For each session, decision points are extracted and categorised by whether
structure and habit strategies agreed on the top action and -- when they
disagreed -- which strategy's top action the animal actually chose. Decision
times are absolute session-clock timestamps aligned with `session.lfp_times`,
so they can be used directly as event times for LFP wavelet analyses.
@peterdoohan
"""

# %% Imports
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_rel, zscore
from statsmodels.stats.multitest import multipletests

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.event_aligned import spectrograms as ea
from GridMaze.analysis.lfp import lfp_utils as lu
from GridMaze.analysis.navigation_strategies import comparisons as sc
from GridMaze.analysis.navigation_strategies import get_input_data as gid

# %% Global Variables
FRAME_RATE = 60  # navigation_df sampling rate (Hz)

# %% Functions


def get_session_decision_times(session, decision_points_only=True):
    """
    Tidy df of session decision points, with absolute session-clock times
    (aligned with `session.lfp_times`) and a strategy-comparison `category` per row.

    Categories (mutually exclusive, exhaustive over the filtered rows):
      - "agree":           structure and habit share a top action
      - "chose_structure": S != H and animal chose structure's top action
      - "chose_habit":     S != H and animal chose habit's top action
      - "chose_neither":   S != H and animal chose some other available action

    Note: the stored `time` is arrival at the choice node; `action` is the move
    out of it (i.e. the actual choice). Default `decision_points_only=True`
    restricts to nodes with >2 available actions; set False to include all rows.
    """
    df = gid.get_session_navigation_strategies_df(session)
    df = sc._filter_df(
        df,
        maze_names=None,
        last_n_days_on_maze=None,
        decision_point_filter="only" if decision_points_only else "all",
    )
    if df.empty:
        return None

    agree = sc._strats_agree_mask(df, "structure", "habit")
    chose_structure = sc._correct(df, "structure")
    chose_habit = sc._correct(df, "habit")

    category = pd.Series("chose_neither", index=df.index)
    category.loc[agree] = "agree"
    category.loc[~agree & chose_structure] = "chose_structure"
    category.loc[~agree & chose_habit] = "chose_habit"

    return pd.DataFrame(
        {
            "subject_ID": df[("subject_ID", "")].values,
            "maze_name": df[("maze_name", "")].values,
            "day_on_maze": df[("day_on_maze", "")].values,
            "trial": df[("trial", "")].values,
            "trial_unique_ID": df[("trial_unique_ID", "")].values,
            "time": df[("time", "")].values,
            "time_in_trial": df[("time_in_trial", "")].values,
            "location": df[("location", "")].values,
            "action": df[("action", "")].values,
            "steps_to_goal": df[("steps_to_goal", "")].values,
            "node_degree": df[("node_degree", "")].values,
            "category": category.values,
        }
    )


def get_session_decision_spectrograms(
    session,
    window=(-3, 3),
    freqs=np.geomspace(1, 250, 100),
    signal_type="LFP",
    decision_points_only=True,
):
    """
    Tidy df of decision-aligned spectrograms for a single session, one row per
    (category, frequency) with absolute time as `time` MultiIndex sub-columns.

    Processing mirrors `lu._get_spectrogram_df`: whole-session wavelet transform,
    power = |cwt|^2, z-score per frequency across session time, then ±`window`
    slices around each decision time, averaged within category. Output schema
    matches `_get_spectrogram_df` with `event` replaced by `category`, so the
    existing spectrogram plotters in `event_aligned.spectrograms` carry over.
    """
    decision_df = get_session_decision_times(session, decision_points_only=decision_points_only)
    if decision_df is None or decision_df.empty:
        return None

    # whole-session wavelet decomposition + per-freq zscore across session time
    signal = lu.get_LFP(session) if signal_type == "LFP" else lu.get_CSD(session)
    spec = zscore(np.abs(lu.compute_wavelet_transform_fft(signal, freqs, lu.FS)) ** 2, axis=1)

    # slice ±window around each decision time, average within category
    lfp_times = session.lfp_times
    samples_before, samples_after = int(window[0] * lu.FS), int(window[1] * lu.FS)
    n_samples = samples_after - samples_before
    t = np.linspace(*window, n_samples)

    dfs = []
    for category, cat_df in decision_df.groupby("category"):
        nearest = np.array([np.argmin(np.abs(lfp_times - dt)) for dt in cat_df.time.values])
        windows = [
            spec[:, s + samples_before : s + samples_after]
            for s in nearest
            if 0 <= s + samples_before and s + samples_after <= spec.shape[1]
        ]
        if not windows:
            continue
        av_df = pd.DataFrame(
            np.array(windows).mean(axis=0),
            columns=pd.MultiIndex.from_product([["time"], t]),
        )
        info_df = lu._get_info_df(session, av_df.index, signal_type=signal_type, single_channel=False)
        info_df[("category", "")] = category
        info_df[("frequency", "")] = freqs
        info_df[("n_decisions", "")] = len(windows)
        dfs.append(pd.concat([info_df, av_df], axis=1))
    return pd.concat(dfs, axis=0).reset_index(drop=True)


def plot_session_decision_spectrograms(
    session_spec_df,
    categories=("agree", "chose_structure", "chose_habit", "chose_neither"),
    window=None,
    axes=None,
    vmin=None,
    vmax=None,
):
    """
    Per-session decision-aligned spectrograms, one panel per category, sharing
    color scale so condition contrasts are comparable. Each panel title shows
    n_decisions in that category. Input is the output of
    `get_session_decision_spectrograms`.
    """
    available = [c for c in categories if c in session_spec_df.category.unique()]
    if axes is None:
        _, axes = plt.subplots(1, len(available), figsize=(2.5 * len(available), 3), sharey=True)
        if len(available) == 1:
            axes = [axes]

    cat_specs = {}
    for cat in available:
        cdf = session_spec_df[session_spec_df.category == cat]
        cat_specs[cat] = (
            cdf.time.values,
            cdf.time.columns.values.astype(np.float64),
            cdf[("frequency", "")].values,
            int(cdf[("n_decisions", "")].iloc[0]),
        )

    _vmin = min(s[0].min() for s in cat_specs.values()) if vmin is None else vmin
    _vmax = max(s[0].max() for s in cat_specs.values()) if vmax is None else vmax

    for i, (ax, cat) in enumerate(zip(axes, available)):
        spec, t, freqs, n_dec = cat_specs[cat]
        ea._plot_spectrogram(
            spec,
            t,
            freqs,
            ax=ax,
            _min=_vmin,
            _max=_vmax,
            colorbar=(i == len(available) - 1),
            y_label=(i == 0),
        )
        ax.set_title(f"{cat} (n={n_dec})", fontsize=9)
        if window is not None:
            ax.set_xlim(*window)
    return


def get_session_decision_traces(session, window=(-3, 3), smooth_sigma_s=0.1, decision_points_only=True):
    """
    Long-format df with one row per (decision, peri-decision timepoint), carrying
    speed and geodesic distance-to-goal values per frame. Use for flexible
    per-category plotting of either variable around the decision.

    `smooth_sigma_s` is the Gaussian smoothing sigma (in seconds) applied to x,
    y, and distance-to-goal at the session level before slicing, to suppress
    frame-rate-sampling noise. Set 0 to disable.

    Columns:
      - subject_ID, maze_name, day_on_maze, late_session
      - trial, trial_unique_ID, decision_index, category
      - time_from_decision (s)
      - speed             : from `navigation_df.centroid_position` (x, y) at frame rate
      - distance_to_goal  : `navigation_df.distance_to_goal.geodesic` (geodesic steps)
    """
    decision_df = get_session_decision_times(session, decision_points_only=decision_points_only)
    if decision_df is None or decision_df.empty:
        return None

    nav = session.navigation_df
    nav_times = nav[("time", "")].values
    x = nav[("centroid_position", "x")].values
    y = nav[("centroid_position", "y")].values
    dtg = nav[("distance_to_goal", "geodesic")].values.astype(float)
    if smooth_sigma_s:
        sigma_frames = smooth_sigma_s * FRAME_RATE
        x, y, dtg = (gaussian_filter1d(v, sigma_frames) for v in (x, y, dtg))
    speed = np.sqrt(np.gradient(x, nav_times) ** 2 + np.gradient(y, nav_times) ** 2)

    samples_before, samples_after = int(window[0] * FRAME_RATE), int(window[1] * FRAME_RATE)
    t = np.linspace(*window, samples_after - samples_before)

    records = []
    for di, drow in decision_df.iterrows():
        s = np.argmin(np.abs(nav_times - drow.time))
        start, end = s + samples_before, s + samples_after
        if start < 0 or end > len(speed):
            continue
        records.append(
            pd.DataFrame(
                {
                    "subject_ID": drow.subject_ID,
                    "maze_name": drow.maze_name,
                    "day_on_maze": drow.day_on_maze,
                    "late_session": session.late_session,
                    "trial": drow.trial,
                    "trial_unique_ID": drow.trial_unique_ID,
                    "decision_index": di,
                    "category": drow.category,
                    "time_from_decision": t,
                    "speed": speed[start:end],
                    "distance_to_goal": dtg[start:end],
                }
            )
        )
    return pd.concat(records, ignore_index=True) if records else None


def plot_session_decision_traces(
    traces_df,
    variable="speed",
    categories=("agree", "chose_structure", "chose_habit"),
    ax=None,
    palette=None,
):
    """
    Per-category mean ± SE peri-decision trace of `variable` ("speed" or
    "distance_to_goal") from a `get_session_decision_traces` long df. SE is
    computed across decisions within category at each timepoint.
    """
    if not isinstance(palette, dict):
        palette = dict(zip(categories, sns.color_palette(palette or "plasma_r", n_colors=len(categories))))
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    ax.spines[["top", "right"]].set_visible(False)
    for cat in [c for c in categories if c in traces_df.category.unique()]:
        sub = traces_df[traces_df.category == cat]
        agg = sub.groupby("time_from_decision")[variable].agg(["mean", "sem"])
        n_dec = sub.decision_index.nunique()
        ax.plot(agg.index, agg["mean"], color=palette[cat], lw=1.5, label=f"{cat} (n={n_dec})")
        ax.fill_between(agg.index, agg["mean"] - agg["sem"], agg["mean"] + agg["sem"], color=palette[cat], alpha=0.2)
    ax.axvline(0, color="k", ls="--", alpha=0.3)
    ax.set_xlabel("decision-aligned time (s)")
    ax.set_ylabel(variable.replace("_", " "))
    ax.legend(fontsize=7)
    return


def get_decision_traces_df(
    window=(-3, 3),
    smooth_sigma_s=0.1,
    decision_points_only=True,
    maze_names=("maze_1", "maze_2"),
    verbose=False,
):
    """
    Cross-session long df of decision traces: late sessions on the requested
    mazes (default maze_1 + maze_2) across all subjects, concatenating each
    session's `get_session_decision_traces` output.
    """
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names=list(maze_names),
        days_on_maze="late",
        with_data=["navigation_df", "trials_df"],
        must_have_data=True,
    )
    dfs = []
    for session in sessions:
        if verbose:
            print(f"processing {session.name}")
        traces = get_session_decision_traces(
            session,
            window=window,
            smooth_sigma_s=smooth_sigma_s,
            decision_points_only=decision_points_only,
        )
        if traces is not None:
            dfs.append(traces)
    return pd.concat(dfs, ignore_index=True) if dfs else None


def test_decision_traces_pairwise(
    traces_df,
    variable="speed",
    window=(-2, 0),
    step=0.2,
    categories=("agree", "chose_structure", "chose_habit"),
    alpha=0.05,
):
    """
    Sliding-bin pairwise paired t-tests of `variable` across categories. In each
    non-overlapping `step`-wide bin within `window`, per-subject means are taken
    per category, then `ttest_rel` is applied for every pair of categories,
    Holm-corrected across pairs within bin. Time is scanned, not corrected
    across (cluster-based permutation is the right tool for that).

    Returns a tidy df: bin_center, category_a, category_b, t, p_raw, p_corrected,
    n_subjects, significant.
    """
    n_bins = int(round((window[1] - window[0]) / step))
    edges = np.linspace(window[0], window[1], n_bins + 1)
    df = traces_df[traces_df.category.isin(categories)]
    pairs = list(combinations(categories, 2))
    rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        bin_df = df[(df.time_from_decision >= left) & (df.time_from_decision < right)]
        per_subj = bin_df.groupby(["subject_ID", "category"])[variable].mean().unstack("category")
        p_raws, t_stats, ns, valid = [], [], [], []
        for a, b in pairs:
            if a not in per_subj.columns or b not in per_subj.columns:
                continue
            paired = per_subj[[a, b]].dropna()
            if len(paired) < 2:
                continue
            t, p = ttest_rel(paired[a], paired[b])
            p_raws.append(p)
            t_stats.append(t)
            ns.append(len(paired))
            valid.append((a, b))
        if not p_raws:
            continue
        p_corrs = multipletests(p_raws, method="holm")[1]
        center = (left + right) / 2
        for (a, b), t, p, pc, n in zip(valid, t_stats, p_raws, p_corrs, ns):
            rows.append(
                {
                    "bin_center": float(center),
                    "category_a": a,
                    "category_b": b,
                    "t": float(t),
                    "p_raw": float(p),
                    "p_corrected": float(pc),
                    "n_subjects": int(n),
                    "significant": bool(pc < alpha),
                }
            )
    return pd.DataFrame(rows)


def plot_decision_traces(
    traces_df,
    variable="speed",
    categories=("agree", "chose_structure", "chose_habit"),
    test_window=(-2, 0),
    test_step=0.2,
    alpha=0.05,
    ax=None,
    palette="tab10",
):
    """
    Cross-subject mean ± SE peri-decision trace of `variable` ("speed" or
    "distance_to_goal") -- per-subject mean across decisions, then mean ± SE
    across subjects -- with pairwise paired-test significance overlaid. In each
    `test_step`-wide bin within `test_window`, per-subject mean `variable` is
    compared between every pair of categories via `ttest_rel`, Holm-corrected
    across pairs within bin (see `test_decision_traces_pairwise`). Significant
    pairs are drawn as colored horizontal segments above the traces, one row
    per category pair; segment color blends the two category colors. Returns
    the underlying stats df.
    """
    category_abbr = {"agree": "A", "chose_structure": "S", "chose_habit": "H", "chose_neither": "N"}
    if not isinstance(palette, dict):
        palette = dict(zip(categories, sns.color_palette(palette or "plasma_r", n_colors=len(categories))))
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    ax.spines[["top", "right"]].set_visible(False)

    # cross-subject mean ± SE traces per category
    for cat in [c for c in categories if c in traces_df.category.unique()]:
        sub = traces_df[traces_df.category == cat]
        per_subj = sub.groupby(["subject_ID", "time_from_decision"])[variable].mean().unstack("time_from_decision")
        mean, sem = per_subj.mean(axis=0), per_subj.sem(axis=0)
        ax.plot(mean.index, mean.values, color=palette[cat], lw=1.5, label=cat)
        ax.fill_between(mean.index, mean.values - sem.values, mean.values + sem.values, color=palette[cat], alpha=0.2)
    ax.axvline(0, color="k", ls="--", alpha=0.3)
    ax.set_xlabel("decision-aligned time (s)")
    ax.set_ylabel(variable.replace("_", " "))
    ax.legend(fontsize=7)

    # pairwise significance overlay
    stats_df = test_decision_traces_pairwise(
        traces_df,
        variable=variable,
        window=test_window,
        step=test_step,
        categories=categories,
        alpha=alpha,
    )
    ymin, ymax = ax.get_ylim()
    row_h = (ymax - ymin) * 0.05
    pairs = list(combinations(categories, 2))
    for i, (a, b) in enumerate(pairs):
        if a not in palette or b not in palette:
            continue
        pair_color = tuple(np.mean([mcolors.to_rgb(palette[a]), mcolors.to_rgb(palette[b])], axis=0))
        y = ymax + row_h * (i + 1)
        sig = stats_df[(stats_df.category_a == a) & (stats_df.category_b == b) & stats_df.significant]
        for _, row in sig.iterrows():
            ax.hlines(
                y,
                row.bin_center - test_step / 2,
                row.bin_center + test_step / 2,
                color=pair_color,
                lw=3,
            )
        ax.text(
            test_window[1] + 0.05,
            y,
            f"{category_abbr.get(a, a)}↔{category_abbr.get(b, b)}",
            va="center",
            fontsize=7,
            color=pair_color,
        )
    ax.set_ylim(ymin, ymax + row_h * (len(pairs) + 1))
