"""
Look at decoding performance of distance-to-goal aligned to navigational errors
Does the internal representation move with subject's internal estimate of distance even
when it is wrong?
@peterdoohan
"""

# %% Imports
import json
from tracemalloc import start
from idna import decode
import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from GridMaze.maze import representations as mr
from GridMaze.analysis.core import convert
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import folds
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.distance_to_goal import distributions as dd
from GridMaze.analysis.distance_to_goal import logreg_decoder as lr

# %% Global Variables

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "distance_to_goal" / "errors"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)


# %% plot decoding aligned to errors


def plot_error_aligned_distance(
    aligned_df,
    match_distance_sampling=True,
    random_state=None,
    color="royalblue",
    ax=None,
):
    """ """
    # set up figure
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("decision (s)")
    ax.set_ylabel("distance to goal (m) \n true or decoded")

    # optionally downsample correct decision to match the distances to goal in error trials
    if match_distance_sampling:
        _cols = [("subject_ID", ""), ("distance_to_goal", ""), ("error", "")]
        _df = aligned_df[_cols].droplevel(1, axis=1)
        error_df = _df[_df.error]
        counts_error = error_df.groupby(["subject_ID", "distance_to_goal"]).size().rename("n_error").reset_index()
        correct_df = _df[~_df.error].copy()
        correct_df["_orig_idx"] = correct_df.index  # preserve original indices
        merged = correct_df.merge(counts_error, on=["subject_ID", "distance_to_goal"], how="left")
        merged["n_error"] = merged["n_error"].fillna(0).astype(int)
        to_sample = merged[merged["n_error"] > 0]

        def sample_per_group(g):
            n = int(g["n_error"].iloc[0])
            # sample without replacement
            return g.sample(n=n, replace=False, random_state=random_state)

        sampled_correct = to_sample.groupby(["subject_ID", "distance_to_goal"]).apply(
            sample_per_group, include_groups=False
        )

        correct_indices = sampled_correct["_orig_idx"].tolist()
        error_indices = error_df.index.tolist()
        full_sample = correct_indices + error_indices
        _aligned_df = aligned_df.loc[full_sample].copy()
    else:
        _aligned_df = aligned_df.copy()

    subj_grouped = _aligned_df.groupby(["subject_ID", "error"])
    for dist, ls in zip(["true_distance", "decoded_distance"], ["-", "--"]):
        sub_avg = subj_grouped[dist].mean()[dist]
        times = sub_avg.columns.values.astype(float)
        _means = sub_avg.groupby(level=1).mean()
        _sems = sub_avg.groupby(level=1).sem()
        for error, color in zip(
            [False, True],
            ["gray", color],
        ):
            _mean = _means.loc[error].values
            _sem = _sems.loc[error].values
            # plot
            ax.plot(times, _mean, color=color, ls=ls, label=f"error:{error}, {dist}")
            ax.fill_between(times, _mean - _sem, _mean + _sem, color=color, alpha=0.2)
    ax.legend(frameon=False, fontsize=6)


def align_decoding_to_errors2(results_df, error_type="nav_error", window=(-5, 5), resolution=0.2):
    """Align true/decoded distance traces to decision points (error/correct)."""

    df = results_df.copy()

    # robust extraction of decoded prob matrix
    prob_df = df.decoded_distance_prob
    distance_bins = pd.to_numeric(prob_df.columns, errors="coerce").values
    P = prob_df.values.astype(float)
    df[("decoded_distance", "")] = np.dot(P, distance_bins)

    rows_before = int(round(-window[0] / resolution))
    rows_after = int(round(window[1] / resolution))
    expected_length = rows_before + rows_after + 1
    aligned_times = np.linspace(window[0], window[1], expected_length).round(2)

    dfs = []
    for subject_ID in df.subject_ID.unique():
        subj_df = df[df.subject_ID == subject_ID].copy().reset_index(drop=True)

        for tuID in subj_df.trial_unique_ID.unique():
            trial_df = subj_df[subj_df.trial_unique_ID == tuID]

            for err in [True, False]:
                idxs = trial_df[(trial_df[error_type] == err) & trial_df.node_degree.gt(2)].index
                if len(idxs) == 0:
                    continue

                aligned_true, aligned_decoded, distances_at_decision = [], [], []
                for i in idxs:
                    start_idx = i - rows_before
                    end_idx = i + rows_after
                    if start_idx < 0 or end_idx >= len(subj_df):
                        continue

                    w = subj_df.iloc[start_idx : end_idx + 1].copy()
                    if w.shape[0] != expected_length:
                        continue

                    diff_trial_mask = (w.trial_unique_ID != tuID).values
                    true_dist = w.distance_bin_mid.values.astype(float)
                    decoded_dist = w.decoded_distance.values.astype(float)
                    true_dist[diff_trial_mask] = np.nan
                    decoded_dist[diff_trial_mask] = np.nan

                    aligned_true.append(true_dist)
                    aligned_decoded.append(decoded_dist)
                    distances_at_decision.append(trial_df.loc[i, ("distance_bin_mid", "")])

                if len(aligned_true) == 0:
                    continue

                aligned_df = pd.concat(
                    [
                        pd.DataFrame(np.stack(arrs), columns=pd.MultiIndex.from_product([[label], aligned_times]))
                        for arrs, label in zip(
                            [aligned_true, aligned_decoded],
                            ["true_distance", "decoded_distance"],
                        )
                    ],
                    axis=1,
                )
                aligned_df[("error", "")] = err
                aligned_df[("subject_ID", "")] = subject_ID
                aligned_df[("distance_to_goal", "")] = distances_at_decision
                dfs.append(aligned_df)

    return pd.concat(dfs, ignore_index=True)


def align_decoding_to_errors(results_df, error_type="nav_error", window=(-5, 5), resolution=0.2):
    """ """
    df = results_df.copy()
    df.reset_index(inplace=True, drop=True)
    # add decoded distance column
    P = df.decoded_distance_prob.values
    distance_bins = df.decoded_distance_prob.columns.astype(float)
    weighted_dists = np.dot(P, distance_bins)  # weighted average of decoded distances
    df[("decoded_distance", "")] = weighted_dists
    # define error windows
    rows_before = int(-window[0] / resolution)
    rows_after = int(window[1] / resolution)
    expected_length = rows_before + rows_after + 1
    aligned_times = np.arange(window[0], window[1] + resolution, resolution).round(2)
    dfs = []
    for subject_ID in results_df.subject_ID.unique():
        print(subject_ID)
        # get decoding delta aliged each error
        subj_df = df[df.subject_ID == subject_ID].copy()
        for tuID in subj_df.trial_unique_ID.unique():
            trial_df = subj_df[subj_df.trial_unique_ID == tuID].copy()
            for err in [True, False]:
                aligned_true = []
                aligned_decoded = []
                distances_at_decision = []
                # filter for either errors or correct decisions at decision points
                idxs = trial_df[(trial_df[error_type] == err) & trial_df.node_degree.gt(2)].index
                if len(idxs) == 0:
                    continue
                for i in idxs:
                    start_idx = i - rows_before
                    end_idx = i + rows_after
                    try:
                        _df = subj_df.loc[start_idx:end_idx].copy()
                    except IndexError:
                        continue
                    if _df.shape[0] != expected_length:
                        continue
                    diff_trial_mask = (_df.trial_unique_ID != tuID).values.astype(bool)
                    true_dist = _df.distance_bin_mid.values.astype(float)
                    decoded_dist = _df.decoded_distance.values.astype(float)
                    true_dist[diff_trial_mask] = np.nan
                    decoded_dist[diff_trial_mask] = np.nan
                    aligned_true.append(true_dist)
                    aligned_decoded.append(decoded_dist)
                    distances_at_decision.append(trial_df.loc[i, ("distance_bin_mid", "")])
                if len(aligned_true) == 0:
                    continue
                # organise into dataframes
                aligned_df = pd.concat(
                    [
                        pd.DataFrame(data=np.stack(arrs), columns=pd.MultiIndex.from_product([[label], aligned_times]))
                        for arrs, label in zip([aligned_true, aligned_decoded], ["true_distance", "decoded_distance"])
                    ],
                    axis=1,
                )
                aligned_df[("error", "")] = err
                aligned_df[("subject_ID", "")] = subject_ID
                aligned_df[("distance_to_goal", "")] = distances_at_decision
            dfs.append(aligned_df)
    aligned_df = pd.concat(dfs, ignore_index=True)
    return aligned_df


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
