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
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import false_discovery_control, ttest_1samp, ttest_rel, zscore
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.event_aligned import spectrograms as ea
from GridMaze.analysis.lfp import lfp_utils as lu
from GridMaze.analysis.navigation_strategies import comparisons as sc
from GridMaze.analysis.navigation_strategies import get_input_data as gid

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "lfp"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as _f:
    SUBJECT_IDS = json.load(_f)

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


def get_session_decision_traces(
    session,
    window=(-3, 3),
    smooth_sigma_s=0.1,
    theta_band=lu.THETA_RANGE,
    filter_order=4,
    signal_type="LFP",
    decision_points_only=True,
):
    """
    Long-format df with one row per (decision, peri-decision timepoint), carrying
    speed, geodesic distance-to-goal, and LFP theta-band power per frame -- all
    on the navigation_df 60Hz time grid -- for flexible per-category plotting
    and covariate handling around the decision.

    Theta power is computed from the full session: `lu.get_LFP` (or `get_CSD`)
    -> 4th-order Butterworth bandpass in `theta_band` -> |Hilbert|^2 ->
    z-scored across session time -> downsampled to nav rate by nearest LFP
    sample per nav frame.

    `smooth_sigma_s` is the Gaussian smoothing sigma (in seconds) applied to x,
    y, and distance-to-goal at the session level before slicing, to suppress
    frame-rate-sampling noise. Set 0 to disable.

    Columns:
      - subject_ID, maze_name, day_on_maze, late_session
      - trial, trial_unique_ID, decision_index, category
      - time_from_decision (s)
      - speed             : from `navigation_df.centroid_position` (x, y) at frame rate
      - distance_to_goal  : `navigation_df.distance_to_goal.geodesic` (geodesic steps)
      - theta_power       : session-z-scored Hilbert envelope power in `theta_band`
    """
    decision_df = get_session_decision_times(session, decision_points_only=decision_points_only)
    if decision_df is None or decision_df.empty:
        return None

    # nav-rate variables (speed, distance_to_goal)
    nav = session.navigation_df
    nav_times = nav[("time", "")].values
    x = nav[("centroid_position", "x")].values
    y = nav[("centroid_position", "y")].values
    dtg = nav[("distance_to_goal", "geodesic")].values.astype(float)
    if smooth_sigma_s:
        sigma_frames = smooth_sigma_s * FRAME_RATE
        x, y, dtg = (gaussian_filter1d(v, sigma_frames) for v in (x, y, dtg))
    speed = np.sqrt(np.gradient(x, nav_times) ** 2 + np.gradient(y, nav_times) ** 2)

    # LFP-rate theta power via bandpass + Hilbert envelope, then downsampled to nav grid
    lfp_signal = lu.get_LFP(session) if signal_type == "LFP" else lu.get_CSD(session)
    nyq = lu.FS / 2
    b, a = butter(filter_order, [theta_band[0] / nyq, theta_band[1] / nyq], btype="bandpass")
    theta_power_lfp = zscore(np.abs(hilbert(filtfilt(b, a, lfp_signal))) ** 2)
    lfp_idx = np.clip(np.searchsorted(session.lfp_times, nav_times), 0, len(theta_power_lfp) - 1)
    theta_power = theta_power_lfp[lfp_idx]

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
                    "theta_power": theta_power[start:end],
                }
            )
        )
    return pd.concat(records, ignore_index=True) if records else None


def get_decision_traces_df(
    window=(-3, 3),
    smooth_sigma_s=0.1,
    theta_band=lu.THETA_RANGE,
    filter_order=4,
    signal_type="LFP",
    decision_points_only=True,
    maze_names=("maze_1", "maze_2"),
    n_jobs=False,
    save=False,
    verbose=False,
):
    """
    Cross-session long df of decision traces: late sessions on the requested
    mazes (default maze_1 + maze_2) across all subjects, concatenating each
    session's `get_session_decision_traces` output. The cache path includes
    `signal_type` so LFP and CSD runs save / load separately
    (`RESULTS_DIR/decision_traces_df_{signal_type}.parquet`). If the cached
    parquet exists and `save=False`, it is loaded from disk; otherwise the df
    is computed (and saved if `save=True`).

    Sessions are loaded and processed one (subject, maze) group at a time --
    each group's sessions are released before the next group loads. Within a
    group, sessions are processed either via joblib (`n_jobs` workers > 0) or
    serially one-at-a-time (`n_jobs=False`/None/0), bypassing joblib entirely.
    Use the serial path if joblib workers are leaking memory.
    """
    save_path = RESULTS_DIR / f"decision_traces_df_{signal_type}.parquet"
    if save_path.exists() and not save:
        if verbose:
            print(f"loading cached traces df from {save_path}")
        return pd.read_parquet(save_path)

    def _run(session):
        return get_session_decision_traces(
            session,
            window=window,
            smooth_sigma_s=smooth_sigma_s,
            theta_band=theta_band,
            filter_order=filter_order,
            signal_type=signal_type,
            decision_points_only=decision_points_only,
        )

    all_dfs = []
    for subject in SUBJECT_IDS:
        for maze in maze_names:
            if verbose:
                print(f"processing {subject} / {maze}")
            sessions = gs.get_maze_sessions(
                subject_IDs=[subject],
                maze_names=[maze],
                days_on_maze="late",
                with_data=[
                    "navigation_df",
                    "trials_df",
                    "lfp_times",
                    "lfp_signal",
                    "lfp_metrics",
                    "cluster_metrics",
                ],
                must_have_data=True,
            )
            if n_jobs:
                dfs = Parallel(n_jobs=n_jobs)(delayed(_run)(session) for session in sessions)
            else:
                dfs = [_run(session) for session in sessions]
            all_dfs.extend(d for d in dfs if d is not None)
    if not all_dfs:
        return None
    traces_df = pd.concat(all_dfs, ignore_index=True)
    if save:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"saving traces df to {save_path}")
        traces_df.to_parquet(save_path)
    return traces_df


def residualize_traces(traces_df, target="theta_power", covariates=("speed",), by_subject=True):
    """
    Linearly regress `target` on `covariates` at each `time_from_decision` (per
    subject if `by_subject=True`, else pooled across subjects) and return
    `traces_df` with a new `{target}_resid` column carrying the residuals. Use
    to remove movement covariates (default: `speed`) from neural signals
    (default: `theta_power`) before category contrasts.
    """
    df = traces_df.copy()
    new_col = f"{target}_resid"
    cov_list = list(covariates)
    group_keys = ["subject_ID", "time_from_decision"] if by_subject else ["time_from_decision"]

    def _residualize(g):
        y = g[target].values
        X = g[cov_list].values
        mask = ~(np.isnan(y) | np.isnan(X).any(axis=1))
        out = np.full(len(y), np.nan)
        if mask.sum() >= len(cov_list) + 2:
            X_design = np.column_stack([np.ones(mask.sum()), X[mask]])
            beta, *_ = np.linalg.lstsq(X_design, y[mask], rcond=None)
            out[mask] = y[mask] - X_design @ beta
        return pd.Series(out, index=g.index)

    df[new_col] = df.groupby(group_keys, sort=False, group_keys=False).apply(_residualize)
    return df


def regress_traces(
    traces_df,
    formula="theta_power ~ speed + distance_to_goal + choice_alignment",
    standardize=("speed", "distance_to_goal", "choice_alignment"),
    time_window=(-1, 1),
    downsample_to_hz=None,
    fdr_across_time=True,
    verbose=False,
):
    """ """
    traces_df = traces_df.copy()
    traces_df["choice_alignment"] = traces_df.category.map(
        {"chose_structure": 1, "chose_habit": -1, "chose_neither": np.nan, "agree": np.nan}
    )
    if standardize:
        for col in standardize:
            traces_df[col] = traces_df.groupby("subject_ID")[col].transform(lambda x: (x - x.mean()) / x.std())

    all_times = np.sort(traces_df.time_from_decision.unique())
    if time_window is not None:
        all_times = all_times[(all_times >= time_window[0]) & (all_times <= time_window[1])]
    if downsample_to_hz is not None:
        step = max(1, int(round(FRAME_RATE / downsample_to_hz)))
        times = all_times[::step]
    else:
        times = all_times

    coefs_rows = []
    for subject in traces_df.subject_ID.unique():
        if verbose:
            print(f"regressing {subject}")
        sub_all = traces_df[traces_df.subject_ID == subject]
        for t in times:
            sub_t = sub_all[sub_all.time_from_decision == t]
            if len(sub_t) < 5:
                continue
            try:
                fit = smf.ols(formula, data=sub_t).fit()
            except Exception as e:
                if verbose:
                    print(f"  fit failed at t={t}: {e}")
                continue
            for predictor, beta in fit.params.items():
                coefs_rows.append(
                    {
                        "subject_ID": subject,
                        "time_from_decision": float(t),
                        "predictor": predictor,
                        "beta": float(beta),
                    }
                )
    coefs_df = pd.DataFrame(coefs_rows)

    pvals_rows = []
    for predictor in coefs_df.predictor.unique():
        pivot = coefs_df[coefs_df.predictor == predictor].pivot(
            index="subject_ID", columns="time_from_decision", values="beta"
        )
        result = ttest_1samp(pivot.values, 0, axis=0, nan_policy="omit")
        ts, ps = np.asarray(result.statistic), np.asarray(result.pvalue)
        p_fdr = np.full_like(ps, np.nan, dtype=float)
        finite = np.isfinite(ps)
        if finite.any():
            p_fdr[finite] = false_discovery_control(ps[finite]) if fdr_across_time else ps[finite]
        n_subj = pivot.notna().sum(axis=0).values
        for tcol, t_, p_raw_, p_fdr_, n_ in zip(pivot.columns, ts, ps, p_fdr, n_subj):
            pvals_rows.append(
                {
                    "time_from_decision": float(tcol),
                    "predictor": predictor,
                    "t": float(t_) if np.isfinite(t_) else np.nan,
                    "p_raw": float(p_raw_) if np.isfinite(p_raw_) else np.nan,
                    "p_fdr": float(p_fdr_) if np.isfinite(p_fdr_) else np.nan,
                    "n_subjects": int(n_),
                    "significant": bool(np.isfinite(p_fdr_) and p_fdr_ < 0.05),
                }
            )
    pvals_df = pd.DataFrame(pvals_rows)
    return coefs_df, pvals_df


def plot_regression_betas(coefs_df, pvals_df, predictors=None, alpha=0.05, plot_intercept=False, ax=None, palette=None):
    """
    Mean ± SE beta across subjects per predictor over `time_from_decision`,
    one line per predictor, with FDR-corrected significance markers drawn as
    colored horizontal segments above the traces (one row per predictor).
    Input: `coefs_df` and `pvals_df` from `regress_traces`.

    `plot_intercept=False` (default) hides the Intercept row; set True to
    include it. Explicit `predictors=[...]` overrides both.
    """
    if predictors is None:
        predictors = list(coefs_df.predictor.unique())
        if not plot_intercept:
            predictors = [p for p in predictors if p != "Intercept"]
    if not isinstance(palette, dict):
        palette = dict(zip(predictors, sns.color_palette(palette or "plasma_r", n_colors=len(predictors))))
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.spines[["top", "right"]].set_visible(False)

    # mean ± SE beta over time, one line per predictor
    for pred in predictors:
        sub = coefs_df[coefs_df.predictor == pred]
        agg = sub.groupby("time_from_decision").beta.agg(["mean", "sem"])
        ax.plot(agg.index, agg["mean"], color=palette[pred], lw=1.5, label=pred)
        ax.fill_between(agg.index, agg["mean"] - agg["sem"], agg["mean"] + agg["sem"], color=palette[pred], alpha=0.2)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", ls="--", alpha=0.3)
    ax.set_xlabel("decision-aligned time (s)")
    ax.set_ylabel("beta")
    ax.legend(fontsize=7, loc="best")

    # FDR-significance overlay rows (one per predictor)
    times = np.sort(coefs_df.time_from_decision.unique())
    seg_w = float(np.median(np.diff(times))) if len(times) > 1 else 0.1
    ymin, ymax = ax.get_ylim()
    row_h = (ymax - ymin) * 0.05
    for i, pred in enumerate(predictors):
        sig = pvals_df[(pvals_df.predictor == pred) & (pvals_df.p_fdr < alpha)]
        y = ymax + row_h * (i + 1)
        for _, row in sig.iterrows():
            ax.hlines(
                y,
                row.time_from_decision - seg_w / 2,
                row.time_from_decision + seg_w / 2,
                color=palette[pred],
                lw=3,
            )
        label = pred[:14] + "…" if len(pred) > 14 else pred
        ax.text(
            times.max() + (times.max() - times.min()) * 0.02, y, label, va="center", fontsize=6, color=palette[pred]
        )
    ax.set_ylim(ymin, ymax + row_h * (len(predictors) + 1))
    return ax


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
