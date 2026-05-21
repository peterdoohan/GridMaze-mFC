"""
Library for distance-to-goal representation, theta-mod decoding.
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

from GridMaze.analysis.core import folds
from GridMaze.analysis.core import convert
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.distance_to_goal import distributions as dd
from GridMaze.analysis.theta_mod import theta_utils as tmu

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "theta_mod"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]
FRAME_RATE = 60


# %% Input data


def get_input_data(
    session,
    include_multi_units=True,
    max_steps_to_goal=30,
    resolution=0.1,
    sum_spike_window=0.4,
    max_distance=0.8,
    bin_spacing=0.04,
    bin_method="uniform",
    n_log_bins=30,
    balance_distances=False,
):
    """Build the per-(downsampled-)frame input dataframe for the distance-to-goal decoder.

    Returns a MultiIndex-columned df with behavioral state, distance bins, and
    theta-phase-resolved spike counts.
    """
    # load
    navigation_df = session.navigation_df.copy()
    spike_counts_df = session.navigation_theta_spike_counts_df.reset_index(drop=True).copy()

    # filter clusters
    keep_clusters = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=include_multi_units,
    )
    cluster_level = spike_counts_df.columns.get_level_values(1)
    spike_counts_df = spike_counts_df.loc[:, cluster_level.isin(keep_clusters)]

    # rolling sum over spike window
    sum_frames = int(sum_spike_window * FRAME_RATE)
    spike_counts_df = spike_counts_df.rolling(window=sum_frames, center=True).sum().fillna(0).astype(int)

    # align column depth and merge behavioral + spike dfs
    navigation_df.columns = pd.MultiIndex.from_tuples([(*col, "") for col in navigation_df.columns])
    input_df = pd.concat([navigation_df, spike_counts_df], axis=1)

    # temporal downsample by stride
    every_n_frames = int(resolution * FRAME_RATE)
    input_df = input_df.iloc[::every_n_frames].reset_index(drop=True)

    # behavioral filter
    input_df = filt.filter_navigation_rates_df(
        input_df,
        navigation_only=True,
        moving_only=True,
        exclude_time_at_goal=True,
        max_steps_to_goal=max_steps_to_goal,
    )

    # filter outlier distances + distance binning
    metric = ("distance_to_goal", "geodesic", "")
    if max_distance is None:
        max_distance = dd.get_distance_percentile(("distance_to_goal", "geodesic"), 0.85)
    input_df = input_df[input_df[metric] < max_distance]

    if bin_method == "uniform":
        n_bins = int(max_distance / bin_spacing)
    elif bin_method == "log":
        n_bins = n_log_bins
    else:
        raise ValueError(f"bin_method {bin_method!r} not recognised. Use 'uniform' or 'log'.")
    bins = convert._get_distance_bins(
        binning_method=bin_method,
        n_distance_bins=n_bins,
        distance_metrics=("distance_to_goal", "geodesic"),
        max_distance=max_distance,
    )
    input_df.loc[:, ("distance_bin", "", "")] = pd.cut(input_df[metric], bins=bins, include_lowest=True).to_numpy()
    input_df.loc[:, ("distance_bin_mid", "", "")] = input_df.distance_bin.apply(lambda x: x.mid).astype(float)
    # compact bin_id = position in sorted observed unique mids (so distance_bin_mids[bin_id] is well-defined
    # even if some bins from the IntervalIndex are empty in this session).
    observed_mids = sorted(input_df[("distance_bin_mid", "", "")].dropna().unique())
    mid_to_id = {m: i for i, m in enumerate(observed_mids)}
    input_df.loc[:, ("distance_bin_id", "", "")] = input_df[("distance_bin_mid", "", "")].map(mid_to_id).astype(int)

    if balance_distances:
        max_size = input_df.groupby(("distance_bin_id", "", "")).size().max()
        input_df = (
            input_df.groupby(("distance_bin_id", "", ""), group_keys=False)
            .sample(n=max_size, replace=True, random_state=42)
            .reset_index(drop=True)
        )
    return input_df


# %% Session-level decoder


def get_session_theta_mod_distance_error(
    session,
    n_folds=16,
    C=1,  # "opt"
    reg_search_folds=16,
    sqrt_spikes=True,
    normalise_X=True,
    output="weighted",
    include_multi_units=True,
    max_steps_to_goal=30,
    resolution=0.1,
    sum_spike_window=0.4,
    max_distance=0.8,
    bin_spacing=0.04,
    bin_method="uniform",
    balance_distances=False,
    verbose=False,
):
    """Per CV fold: train a phase-agnostic distance decoder on mean spike counts,
    then test on spikes at each theta phase. Returns a long df of per-sample
    decoding errors with (sample, theta_phase, fold) as the implicit row key.

    Sign convention: signed_error = y_pred_dist - true_dist
      +ve bias = decoder predicts location further from goal (past)
      -ve bias = decoder predicts location closer to goal (future)
    """
    # 1. load + bin
    input_df = get_input_data(
        session,
        include_multi_units=include_multi_units,
        max_steps_to_goal=max_steps_to_goal,
        resolution=resolution,
        sum_spike_window=sum_spike_window,
        max_distance=max_distance,
        bin_spacing=bin_spacing,
        bin_method=bin_method,
        balance_distances=balance_distances,
    )
    distance_bin_mids = np.array(sorted(input_df.distance_bin_mid.unique()))
    theta_phases = input_df.spike_count.columns.get_level_values(1).unique().astype(float)

    # 2. CV folds over valid trials
    folds_df = folds.get_folds_df(
        session,
        goal_stratified=False,
        valid_trials=input_df.trial.unique(),
        n_folds=n_folds,
        return_unique_IDs=False,
    )

    # 3. fold loop: train on phase-avg, test per phase
    results = []
    for fold in folds_df.columns.get_level_values(0).unique():
        if verbose:
            print(fold)
        fold_df = folds_df[fold]
        train_trials = fold_df["train"].unstack().dropna().values
        test_trials = fold_df["test"].unstack().dropna().values
        train_df = input_df[input_df.trial.isin(train_trials)]
        test_df = input_df[input_df.trial.isin(test_trials)]

        # nested CV for C
        _C = (
            _get_opt_C(
                input_df,
                fold_df,
                output=output,
                sqrt_spikes=sqrt_spikes,
                normalise_X=normalise_X,
                reg_search_folds=reg_search_folds,
                distance_bin_mids=distance_bin_mids,
                verbose=verbose,
            )
            if C == "opt"
            else C
        )
        if verbose:
            print(f"    optimal C = {_C}")

        # fit phase-agnostic decoder
        X_train = _prepare_X(train_df, phase=None, sqrt=sqrt_spikes)
        scaler = StandardScaler().fit(X_train) if normalise_X else None
        if scaler is not None:
            X_train = scaler.transform(X_train)
        y_train = train_df.distance_bin_id.values
        decoder = LogisticRegression(C=_C, random_state=0, max_iter=10_000, class_weight="balanced").fit(
            X_train, y_train
        )

        # phase-agnostic baseline on held-out samples
        X_test_mean = _prepare_X(test_df, phase=None, sqrt=sqrt_spikes)
        if scaler is not None:
            X_test_mean = scaler.transform(X_test_mean)
        y_test = test_df.distance_bin_id.values
        true_dist = distance_bin_mids[y_test]
        baseline_pred_dist = _decode_distance(decoder, X_test_mean, distance_bin_mids, output=output)
        fold_baseline_mae = float(np.mean(np.abs(baseline_pred_dist - true_dist)))

        # per-test-sample behavioral metadata
        res_base = pd.DataFrame(
            {
                "time": test_df[("time", "", "")].to_numpy(),
                "trial": test_df[("trial", "", "")].to_numpy(),
                "trial_unique_ID": test_df[("trial_unique_ID", "", "")].to_numpy(),
                "distance_to_goal": test_df[("distance_to_goal", "geodesic", "")].to_numpy(),
                "distance_bin_mid": test_df[("distance_bin_mid", "", "")].to_numpy().astype(float),
                "speed": test_df[("speed", "", "")].to_numpy(),
            },
            index=test_df.index,
        )
        res_base["fold_baseline_mae"] = fold_baseline_mae

        # test at each theta phase
        for phase in theta_phases:
            X_phase = _prepare_X(test_df, phase=phase, sqrt=sqrt_spikes)
            if scaler is not None:
                X_phase = scaler.transform(X_phase)
            y_pred_dist = _decode_distance(decoder, X_phase, distance_bin_mids, output=output)
            signed_error = y_pred_dist - true_dist  # +ve = further than truth (past)
            res = res_base.copy()
            res["theta_phase"] = float(phase)
            res["fold"] = fold
            res["signed_error"] = signed_error
            results.append(res)

    return pd.concat(results).reset_index(drop=True)


# %% Cross-session runner


def get_theta_mod_distance_error_df(verbose=True, C=1, save=False):
    """Run the session-level decoder across all subjects × mazes and concat results.

    Cached to parquet. Pass `save=True` to force rerun and overwrite the cache.
    """
    save_path = RESULTS_DIR / "theta_mod_distance_error_df.parquet"
    if save_path.exists() and not save:
        return pd.read_parquet(save_path)

    def _process_session(session):
        if verbose:
            print(session.name)
        try:
            res = get_session_theta_mod_distance_error(session, verbose=False, C=C)
            res["subject_ID"] = session.subject_ID
            res["maze_name"] = session.maze_name
            res["day_on_maze"] = session.day_on_maze
            res["late_session"] = session.late_session
            return res
        except Exception as e:
            if verbose:
                print(f"Error processing {session.name}: {e}")
            return None

    dfs = []
    for subject in SUBJECT_IDS:
        for maze_name in MAZE_NAMES:
            sessions = gs.get_maze_sessions(
                subject_IDs=[subject],
                maze_names=[maze_name],
                days_on_maze="all",
                with_data=["navigation_df", "navigation_theta_spike_counts_df", "cluster_metrics", "trials_df"],
                must_have_data=True,
            )
            if not sessions:
                continue
            session_dfs = Parallel(n_jobs=-1)(delayed(_process_session)(s) for s in sessions)
            dfs.extend([d for d in session_dfs if d is not None])

    summary_df = pd.concat(dfs).reset_index(drop=True)
    if save:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_parquet(save_path)
    return summary_df


# %% Plotting


def plot_theta_mod_distance_error(
    summary_df,
    late_sessions=False,
    distance_to_goal=None,
    speed_range=None,
    maze_names=None,
    max_baseline_mae=None,
    color="darkblue",
    label=None,
    print_stats=True,
    ax=None,
):
    """Per-subject sinusoid of distance-decoding bias along theta phase, in cm.

    Sign convention: +ve bias = decoder predicts a location further from the goal (past).
    """
    df = _filter_summary_df(
        summary_df,
        distance_to_goal=distance_to_goal,
        speed_range=speed_range,
        maze_names=maze_names,
        max_baseline_mae=max_baseline_mae,
        late_sessions=late_sessions,
    )
    bias = df.groupby(["subject_ID", "theta_phase"])["signed_error"].mean().unstack(0).T
    bias = bias.sub(bias.mean(axis=1), axis=0)
    bias = bias * 100  # m -> cm
    tmu.plot_decoding_bias(
        bias,
        color=color,
        label=label,
        ylabel="decoding bias (cm)\n(distance-to-goal)",
        print_stats=print_stats,
        ax=ax,
    )


# %% Filter helpers


SESSION_KEYS = ["subject_ID", "maze_name", "day_on_maze"]


def _filter_summary_df(
    summary_df,
    distance_to_goal=None,
    speed_range=None,
    maze_names=None,
    max_baseline_mae=None,
    late_sessions=False,
):
    """Apply plot-time filters to the cross-session decoding summary.

    - distance_to_goal: (lo, hi) in metres (geodesic), inclusive of `.between`.
    - speed_range: (lo, hi).
    - maze_names: iterable of maze names to keep.
    - max_baseline_mae: drop sessions whose mean per-fold phase-agnostic MAE exceeds
      this threshold (analog of place's min_chance_ratio; lower MAE = better decoder).
    """
    df = summary_df
    if maze_names is not None:
        df = df[df.maze_name.isin(maze_names)]
    if late_sessions:
        df = df[df.late_session]
    if distance_to_goal is not None:
        lo, hi = distance_to_goal
        df = df[df.distance_to_goal.between(lo, hi)]
    if speed_range is not None:
        df = df[df.speed.between(*speed_range)]
    if max_baseline_mae is not None:
        keep_keys = _sessions_below_baseline_mae(df, max_baseline_mae=max_baseline_mae)
        df = df.merge(keep_keys, on=SESSION_KEYS, how="inner")
    return df.reset_index(drop=True)


def _sessions_below_baseline_mae(df, max_baseline_mae):
    """Return SESSION_KEYS for sessions whose mean per-fold baseline MAE is at or
    below the threshold (lower MAE = better phase-agnostic decoder)."""
    fold_keys = SESSION_KEYS + ["fold"]
    per_fold = df.groupby(fold_keys)["fold_baseline_mae"].first()
    session_mae = per_fold.groupby(SESSION_KEYS).mean()
    keep = session_mae[session_mae <= max_baseline_mae].index.to_frame(index=False)
    return keep


# %% Decoder helpers


def _prepare_X(df, phase=None, sqrt=True):
    """Spike-count feature matrix (n_samples × n_clusters). phase=None → mean across theta phases."""
    if phase is None:
        X = df.spike_count.T.groupby(level=0).mean().T.values
    else:
        X = df.spike_count.xs(phase, level=1, axis=1).values
    if sqrt:
        X = np.sqrt(X)
    return X


def _decode_distance(decoder, X, distance_bin_mids, output="max"):
    """Return predicted distance (m) per sample under the chosen readout."""
    if output == "weighted":
        return decoder.predict_proba(X).dot(distance_bin_mids)
    elif output == "max":
        return distance_bin_mids[decoder.predict(X)]
    else:
        raise ValueError(f"output {output!r} not recognised. Use 'max' or 'weighted'.")


def _get_opt_C(
    input_data,
    fold_df,
    output="max",
    sqrt_spikes=True,
    normalise_X=True,
    reg_search_folds=8,
    C_range=None,
    distance_bin_mids=None,
    verbose=False,
):
    """Nested CV to pick LogisticRegression C by minimising MAE on decoded distance.

    Splits the fold's training trials into inner vfolds, evaluates MAE on
    validation samples over `C_range`, and returns the C with the lowest mean
    inner-vfold MAE.

    `reg_search_folds`: if int, evaluate on a random subset of that many inner vfolds
    (the train pool per iteration still spans all other vfolds). None → use all vfolds.
    """
    if C_range is None:
        C_range = np.logspace(-2, 2, 10)
    if distance_bin_mids is None:
        distance_bin_mids = np.array(sorted(input_data.distance_bin_mid.unique()))

    vfolds_df = fold_df.train
    all_vfolds = list(vfolds_df.columns)
    if reg_search_folds is not None and reg_search_folds < len(all_vfolds):
        rng = np.random.default_rng(0)
        eval_vfolds = list(rng.choice(all_vfolds, size=reg_search_folds, replace=False))
    else:
        eval_vfolds = all_vfolds

    results = np.full((len(eval_vfolds), len(C_range)), np.nan)
    for i, vfold in enumerate(eval_vfolds):
        if verbose:
            print(f"    vfold {i}")
        val_trials = vfolds_df[vfold].dropna().values
        train_trials = vfolds_df[[t for t in all_vfolds if t != vfold]].unstack().dropna().values
        _train_df = input_data[input_data.trial.isin(train_trials)]
        _val_df = input_data[input_data.trial.isin(val_trials)]
        if _train_df.shape[0] == 0 or _val_df.shape[0] == 0:
            continue
        X_train = _prepare_X(_train_df, phase=None, sqrt=sqrt_spikes)
        X_val = _prepare_X(_val_df, phase=None, sqrt=sqrt_spikes)
        if normalise_X:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
        y_train = _train_df.distance_bin_id.values
        y_val_true = distance_bin_mids[_val_df.distance_bin_id.values]
        for j, C in enumerate(C_range):
            decoder = LogisticRegression(C=C, random_state=0, max_iter=10_000, class_weight="balanced").fit(
                X_train, y_train
            )
            y_pred = _decode_distance(decoder, X_val, distance_bin_mids, output=output)
            results[i, j] = np.mean(np.abs(y_pred - y_val_true))
    return C_range[np.nanargmin(np.nanmean(results, axis=0))]


# %%
