"""
theta modulation of the place-direction representation.
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from GridMaze.analysis.place_direction import future_decoding as fd
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import folds
from GridMaze.analysis.theta_mod import theta_utils as tmu
from GridMaze.analysis.theta_mod import distance_to_goal_decoder as tdd
from GridMaze.maze import representations as mr

# %% Global Variables
from GridMaze.paths import RESULTS2_PATH, EXPERIMENT_INFO_PATH

RESULTS_DIR = RESULTS2_PATH / "theta_mod"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]

FRAME_RATE = 60
STEP_DISTANCE = 0.09  # m (9cm between tower->edge)


# %% Input data


def get_input_data(
    session,
    include_multi_units=True,
    max_steps_to_goal=30,
    resolution=0.1,
    sum_spike_window=0.4,
    envelope=2,
    state_type="place",
):
    """Build the per-(downsampled-)frame input dataframe for the trajectory decoder.

    Returns a MultiIndex-columned df with behavioral state, past/future envelope,
    and theta-phase-resolved spike counts.
    """
    # load
    navigation_df = session.navigation_df.copy()
    spike_counts_df = session.navigation_theta_spike_counts_df.reset_index(drop=True).copy()

    # filter to selected clusters (cluster_ID lives on level 1 of the spike-count columns)
    keep_clusters = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=include_multi_units,
    )
    cluster_level = spike_counts_df.columns.get_level_values(1)
    spike_counts_df = spike_counts_df.loc[:, cluster_level.isin(keep_clusters)]

    # rolling sum over spike_window
    sum_frames = int(sum_spike_window * FRAME_RATE)
    spike_counts_df = spike_counts_df.rolling(window=sum_frames, center=True).sum().fillna(0).astype(int)

    # align column depth and merge behavioral + spike dfs
    navigation_df.columns = pd.MultiIndex.from_tuples([(*col, "") for col in navigation_df.columns])
    input_df = pd.concat([navigation_df, spike_counts_df], axis=1)

    # temporal downsample
    if resolution is not None:
        every_n_frames = int(resolution * FRAME_RATE)
        input_df = input_df.iloc[::every_n_frames].reset_index(drop=True)

    # add place_direction state
    input_df[("place_direction", "", "")] = input_df.maze_position.simple + "_" + input_df.cardinal_movement_direction

    # attach past + future envelope
    envelope_df = fd.get_past_and_future_states(
        input_df, state_type=state_type, past_offset=envelope, future_offset=envelope
    )
    envelope_df = envelope_df[["past", "future"]]
    envelope_df.columns = pd.MultiIndex.from_tuples([(*col, "") for col in envelope_df.columns])
    input_df = pd.concat([input_df, envelope_df], axis=1)

    # final behavioral filter
    input_df = filt.filter_navigation_rates_df(
        input_df,
        navigation_only=True,
        moving_only=True,
        exclude_time_at_goal=True,
        max_steps_to_goal=max_steps_to_goal,
    )
    return input_df


# %% Session-level decoder


def get_session_theta_mod_trajectory_error(
    session,
    include_multi_units=True,
    max_steps_to_goal=30,
    sum_spike_window=0.4,
    resolution=0.1,
    envelope=2,
    sqrt_spikes=True,
    C="opt",
    normalise_X=True,
    n_folds=16,
    reg_search_folds=8,
    verbose=False,
):
    """Per CV fold: train a phase-agnostic place decoder on mean spike counts, then
    test on spikes at each theta phase. Returns a long df of per-sample decoding
    errors with (sample, theta_phase, fold) as the implicit row key.
    """
    # 1. load + keep only samples with full past+future envelope
    input_df = get_input_data(
        session,
        include_multi_units=include_multi_units,
        max_steps_to_goal=max_steps_to_goal,
        sum_spike_window=sum_spike_window,
        resolution=resolution,
        envelope=envelope,
    )
    input_df = input_df[input_df[["past", "future"]].notnull().all(axis=1)]
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

        # fit phase-agnostic decoder
        X_train = _prepare_X(train_df, phase=None, sqrt=sqrt_spikes)
        scaler = StandardScaler().fit(X_train) if normalise_X else None
        if scaler is not None:
            X_train = scaler.transform(X_train)
        Y_train = train_df.maze_position.simple.values
        _C = (
            _get_opt_C(
                fold_df,
                train_df,
                sqrt_spikes=sqrt_spikes,
                normalise_X=normalise_X,
                reg_search_folds=reg_search_folds,
            )
            if C == "opt"
            else C
        )
        decoder = LogisticRegression(C=_C, random_state=0, max_iter=10_000, class_weight="balanced").fit(
            X_train, Y_train
        )

        # phase-agnostic decoder quality on held-out samples (above-chance diagnostic)
        X_test_mean = _prepare_X(test_df, phase=None, sqrt=sqrt_spikes)
        Y_test = test_df.maze_position.simple.values
        if scaler is not None:
            X_test_mean = scaler.transform(X_test_mean)
        fold_accuracy = decoder.score(X_test_mean, Y_test)
        fold_n_classes = len(decoder.classes_)

        # per-test-sample behavioral metadata (explicit subset to avoid level-0 duplicates after flattening)
        res_base = pd.DataFrame(
            {
                "time": test_df[("time", "", "")].to_numpy(),
                "trial_unique_ID": test_df[("trial_unique_ID", "", "")].to_numpy(),
                "place_direction": test_df[("place_direction", "", "")].to_numpy(),
                "distance_to_goal": test_df[("distance_to_goal", "geodesic", "")].to_numpy(),
            },
            index=test_df.index,
        )
        res_base["fold_accuracy"] = fold_accuracy
        res_base["fold_n_classes"] = fold_n_classes

        # test at each theta phase
        for phase in theta_phases:
            X_phase = _prepare_X(test_df, phase=phase, sqrt=sqrt_spikes)
            if scaler is not None:
                X_phase = scaler.transform(X_phase)
            Yprob = decoder.predict_proba(X_phase)
            errors = _get_trajectory_error(Yprob, test_df, decoder.classes_)
            res = res_base.copy()
            res["theta_phase"] = phase
            res["fold"] = fold
            for k, v in errors.items():
                res[k] = v
            results.append(res)

    return pd.concat(results).reset_index(drop=True)


# %% Cross-session runner


def get_theta_mod_trajectory_error_df(verbose=True, save=False):
    """Run the session-level decoder across all subjects × mazes and concat results.

    Cached to parquet. Pass `save=True` to force rerun and overwrite the cache.
    """
    save_path = RESULTS_DIR / "theta_mod_trajectory_error_df.parquet"
    if save_path.exists() and not save:
        return pd.read_parquet(save_path)

    def _process_session(session):
        if verbose:
            print(session.name)
        res = get_session_theta_mod_trajectory_error(session, verbose=False)
        res["subject_ID"] = session.subject_ID
        res["maze_name"] = session.maze_name
        res["day_on_maze"] = session.day_on_maze
        return res

    dfs = []
    for subject in SUBJECT_IDS:
        for maze_name in MAZE_NAMES:
            sessions = gs.get_maze_sessions(
                subject_IDs=[subject],
                maze_names=[maze_name],
                days_on_maze="late",
                with_data=["navigation_df", "navigation_theta_spike_counts_df", "cluster_metrics", "trials_df"],
                must_have_data=True,
            )
            session_dfs = Parallel(n_jobs=-1)(delayed(_process_session)(s) for s in sessions)
            dfs.extend(session_dfs)

    summary_df = pd.concat(dfs).reset_index(drop=True)

    if save:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_parquet(save_path)
    return summary_df


# %% Plotting


def plot_theta_mod_trajectory_error(
    summary_df,
    distance_to_goal=None,
    decision_points=False,
    all_envelope_defined=True,
    min_chance_ratio=2,
    normalise=True,
    color="darkred",
    ref_color="darkblue",
    label=None,
    print_stats=True,
    plot_distance_ref=True,
    ax=None,
):
    """Per-subject sinusoid of place-decoding bias along theta phase, in mm.

    +ve bias = decoder predicts further along the future trajectory.
    """
    df = _filter_summary_df(
        summary_df,
        distance_to_goal=distance_to_goal,
        decision_points=decision_points,
        all_envelope_defined=all_envelope_defined,
        min_chance_ratio=min_chance_ratio,
    )
    place_bias = df.groupby(["subject_ID", "theta_phase"])["signed_error"].mean().unstack(0).T
    if normalise:
        place_bias = place_bias.sub(place_bias.mean(axis=1), axis=0)
    place_bias = place_bias * 100  # m → cm

    tmu.plot_decoding_bias(
        place_bias,
        color=color,
        label=label,
        ylabel="decoding bias (cm)",
        print_stats=print_stats,
        ax=ax,
    )

    if plot_distance_ref:
        if ax is None:
            ax = plt.gca()
        distance_mod_df = tdd.load_decoding_results(lfp_type="theta_mid")
        dist_bias = distance_mod_df.groupby(["subject_ID"]).lfp_phase.mean().lfp_phase
        dist_bias = dist_bias.sub(dist_bias.mean(axis=1), axis=0)
        phases = dist_bias.columns.values.astype(float)
        dist_fit = tmu.fit_sinusoid(phases, dist_bias.mean().values, fit_constant=True, return_as="params")
        place_fit = tmu.fit_sinusoid(phases, place_bias.mean().values, fit_constant=True, return_as="params")
        _x = np.linspace(-np.pi, np.pi, 100)
        scale = place_fit["A"] / dist_fit["A"] if dist_fit["A"] > 0 else 1.0
        _y = scale * dist_fit["A"] * np.sin(_x + dist_fit["phi"])
        ax.plot(_x, _y, color=ref_color, alpha=0.4, linewidth=1.5, label="distance (ref.)")
        if print_stats:
            print("place vs distance offset:")
            tmu.test_theta_offset(dist_bias, place_bias)


def plot_trough_phases(
    summary_df,
    distance_to_goal=None,
    decision_points=False,
    all_envelope_defined=True,
    min_chance_ratio=2.0,
    colors=("darkred", "darkblue"),
    orientation="horizontal",
    print_stats=True,
    ax=None,
):
    """Per-subject trough phase of sinusoid fit to decoding bias, place vs distance.

    orientation: "horizontal" (phase on x-axis) or "vertical" (phase on y-axis).
    colors: (place_color, distance_color).
    """
    if orientation not in ("horizontal", "vertical"):
        raise ValueError(f"orientation must be 'horizontal' or 'vertical'. Got {orientation!r}.")

    df = _filter_summary_df(
        summary_df,
        distance_to_goal=distance_to_goal,
        decision_points=decision_points,
        all_envelope_defined=all_envelope_defined,
        min_chance_ratio=min_chance_ratio,
    )
    place_bias = df.groupby(["subject_ID", "theta_phase"])["signed_error"].mean().unstack(0).T
    place_bias = place_bias.sub(place_bias.mean(axis=1), axis=0)

    distance_mod_df = tdd.load_decoding_results(lfp_type="theta_mid")
    dist_bias = distance_mod_df.groupby(["subject_ID"]).lfp_phase.mean().lfp_phase
    dist_bias = dist_bias.sub(dist_bias.mean(axis=1), axis=0)

    def _troughs(bias_df):
        phases = bias_df.columns.values.astype(float)
        out = {}
        for subject in bias_df.index:
            fit = tmu.fit_sinusoid(phases, bias_df.loc[subject].values, fit_constant=True, return_as="params")
            trough = (-np.pi / 2 - fit["phi"] + np.pi) % (2 * np.pi) - np.pi
            out[subject] = trough
        return out

    place_troughs = _troughs(place_bias)
    dist_troughs = _troughs(dist_bias)
    subjects = sorted(set(place_troughs) & set(dist_troughs))
    trough_df = pd.DataFrame(
        [
            {"subject_ID": s, "condition": c, "trough": t}
            for s in subjects
            for c, t in [("distance", dist_troughs[s]), ("place", place_troughs[s])]
        ]
    )

    if print_stats:

        tmu.test_theta_offset(dist_bias, place_bias)

    horizontal = orientation == "horizontal"

    if ax is None:
        figsize = (2, 0.5) if horizontal else (0.5, 2)
        _, ax = plt.subplots(1, 1, figsize=figsize)
    ax.spines[["top", "right"]].set_visible(False)
    for s in subjects:
        if horizontal:
            xs, ys = [place_troughs[s], dist_troughs[s]], [0, 1]
        else:
            xs, ys = [0, 1], [place_troughs[s], dist_troughs[s]]
        ax.plot(xs, ys, color="grey", alpha=0.3, linewidth=1, zorder=1)
    for cond, color in zip(["place", "distance"], colors):
        if horizontal:
            x_kw, y_kw, orient_kw = "trough", "condition", "h"
        else:
            x_kw, y_kw, orient_kw = "condition", "trough", "v"
        sns.pointplot(
            data=trough_df[trough_df.condition == cond],
            x=x_kw,
            y=y_kw,
            order=["place", "distance"],
            ax=ax,
            errorbar="se",
            markers="o",
            linestyles="",
            capsize=0,
            color=color,
            orient=orient_kw,
            zorder=3,
        )
    phase_ticks = np.arange(0, np.pi + 0.1, np.pi / 2)
    phase_labels = ["0", "π/2", "π"]
    if horizontal:
        ax.set_xlabel("theta phase (trough)")
        ax.set_ylabel("")
        ax.set_xticks(phase_ticks)
        ax.set_xticklabels(phase_labels)
        ax.set_xlim(0, np.pi)
        ax.set_ylim(1.3, -0.3)
    else:
        ax.set_ylabel("theta phase (trough)")
        ax.set_xlabel("")
        ax.set_yticks(phase_ticks)
        ax.set_yticklabels(phase_labels)
        ax.set_ylim(0, np.pi)
        ax.set_xlim(-0.3, 1.3)


# %% --- filter helpers ---


SESSION_KEYS = ["subject_ID", "maze_name", "day_on_maze"]


def _filter_summary_df(
    summary_df,
    distance_to_goal=None,
    decision_points=False,
    all_envelope_defined=True,
    min_chance_ratio=2.0,
):
    """Apply plot-time filters to the cross-session decoding summary.

    - distance_to_goal: (lo, hi) in metres (geodesic), inclusive.
    - decision_points: False | "future" | "past". "future" uses edges_only=True;
      "past" uses node_only=True. Drops `rooms_maze` rows (no decision points defined).
    - all_envelope_defined: keep only samples where the full ±envelope was in the training set.
    - min_chance_ratio: drop sessions whose mean fold accuracy is below
      min_chance_ratio × chance, where chance = 1 / fold_n_classes (per fold).
    """
    df = summary_df
    if all_envelope_defined:
        df = df[df.all_envelope_defined]
    if distance_to_goal is not None:
        lo, hi = distance_to_goal
        df = df[df.distance_to_goal.between(lo, hi)]
    if decision_points:
        df = _filter_decision_points(df, decision_points=decision_points)
    if min_chance_ratio is not None:
        keep_keys = _sessions_above_chance(df, min_chance_ratio=min_chance_ratio)
        df = df.merge(keep_keys, on=SESSION_KEYS, how="inner")
    return df.reset_index(drop=True)


def _filter_decision_points(summary_df, decision_points="future"):
    """Restrict to place_directions that are decision points. Drops rooms_maze."""
    dfs = []
    for maze_name in ["maze_1", "maze_2"]:
        maze_df = summary_df[summary_df.maze_name == maze_name]
        simple_maze = mr.get_simple_maze(maze_name)
        if decision_points == "future":
            _decision_points = fd.get_decision_points(
                simple_maze, mode="future", edges_only=True, node_only=False, return_as="strings", plot=False
            )
        elif decision_points == "past":
            _decision_points = fd.get_decision_points(
                simple_maze, mode="past", edges_only=False, node_only=True, return_as="strings", plot=False
            )
        else:
            raise ValueError(f"decision_points must be False, 'future', or 'past'. Got {decision_points!r}.")
        dfs.append(maze_df[maze_df.place_direction.isin(_decision_points)])
    return pd.concat(dfs, axis=0)


def _sessions_above_chance(df, min_chance_ratio=2.0):
    """Return the (subject_ID, maze_name, day_on_maze) keys whose mean fold
    accuracy clears `min_chance_ratio × chance`, where chance varies per fold
    as 1 / fold_n_classes.
    """
    fold_keys = SESSION_KEYS + ["fold"]
    per_fold = df.groupby(fold_keys)[["fold_accuracy", "fold_n_classes"]].first()
    per_fold["chance_ratio"] = per_fold.fold_accuracy * per_fold.fold_n_classes
    session_ratio = per_fold.groupby(SESSION_KEYS)["chance_ratio"].mean()
    keep = session_ratio[session_ratio >= min_chance_ratio].index.to_frame(index=False)
    return keep


# %% --- decoder helpers ---


def _prepare_X(df, phase=None, sqrt=True):
    """Spike-count feature matrix (n_samples × n_clusters). phase=None → mean across theta phases."""
    if phase is None:
        X = df.spike_count.T.groupby(level=0).mean().T.values
    else:
        X = df.spike_count.xs(phase, level=1, axis=1).values
    if sqrt:
        X = np.sqrt(X)
    return X


def _get_opt_C(fold_df, train_df, sqrt_spikes=True, normalise_X=True, C_range=None, reg_search_folds=8, verbose=False):
    """Nested CV to pick logistic regression regularization strength C.

    Splits the fold's training trials into inner vfolds, scores validation accuracy
    over `C_range`, and returns the C with the highest mean inner-vfold accuracy.

    `reg_search_folds`: if int, evaluate on a random subset of that many inner vfolds
    (the train pool per iteration still spans all other vfolds). None → use all vfolds.
    """
    if C_range is None:
        C_range = np.logspace(-2, 2, 10)
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
            print(f"vfold: {i}")
        val_trials = vfolds_df[vfold].dropna().values
        train_trials = vfolds_df[[t for t in all_vfolds if t != vfold]].unstack().dropna().values
        _train_df = train_df[train_df.trial.isin(train_trials)]
        _val_df = train_df[train_df.trial.isin(val_trials)]
        if _train_df.shape[0] == 0 or _val_df.shape[0] == 0:
            continue
        X_train = _prepare_X(_train_df, phase=None, sqrt=sqrt_spikes)
        X_val = _prepare_X(_val_df, phase=None, sqrt=sqrt_spikes)
        Y_train = _train_df.maze_position.simple.values
        Y_val = _val_df.maze_position.simple.values
        if normalise_X:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
        for j, C in enumerate(C_range):
            decoder = LogisticRegression(C=C, random_state=0, max_iter=10_000, class_weight="balanced")
            decoder.fit(X_train, Y_train)
            results[i, j] = decoder.score(X_val, Y_val)
    return C_range[np.nanargmax(np.nanmean(results, axis=0))]


def _get_trajectory_error(Yprob, test_df, decoder_classes):
    """Envelope-normalized, step-coordinate signed decoding bias per test sample.

    For each sample we treat each step of the past/future envelope (step 0 = true
    location) as a candidate position along the animal's actual trajectory. The
    decoder probability at each envelope location is summed over the full envelope
    and normalized to 1 per sample; the signed error is the resulting
    probability-weighted step coordinate, converted to meters.

    Returns a dict of per-sample arrays:
      - signed_error         (m)    center of mass along trajectory steps (+ve = future)
      - all_envelope_defined (bool) every envelope location appeared in training classes
    """
    # Build (n_samples × (2k+1)) envelope label matrix with step coords [-k, ..., 0, ..., +k]
    past = test_df["past"].droplevel(level=1, axis=1)
    future = test_df["future"].droplevel(level=1, axis=1)
    envelope = int(max(past.columns.max(), future.columns.max()))
    step_coords = np.arange(-envelope, envelope + 1)

    past_labels = np.stack([past[i].to_numpy() for i in range(envelope, 0, -1)], axis=1)
    center = test_df.maze_position.simple.to_numpy().reshape(-1, 1)
    future_labels = np.stack([future[i].to_numpy() for i in range(1, envelope + 1)], axis=1)
    envelope_labels = np.concatenate([past_labels, center, future_labels], axis=1)

    # Map each envelope cell to a column of Yprob; -1 marks labels not in training
    col_idx = np.full(envelope_labels.shape, -1, dtype=np.int64)
    for label, j in {c: j for j, c in enumerate(decoder_classes)}.items():
        col_idx[envelope_labels == label] = j
    all_envelope_defined = (col_idx >= 0).all(axis=1)

    # Sentinel 0-column so missing labels contribute zero probability
    n_samples = Yprob.shape[0]
    Yprob_ext = np.hstack([Yprob, np.zeros((n_samples, 1))])
    envelope_probs = Yprob_ext[np.arange(n_samples)[:, None], col_idx]

    envelope_mass = envelope_probs.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = envelope_probs / envelope_mass[:, None]
    signed_error = (norm * step_coords * STEP_DISTANCE).sum(axis=1)

    return {
        "signed_error": signed_error,
        "all_envelope_defined": all_envelope_defined,
    }
