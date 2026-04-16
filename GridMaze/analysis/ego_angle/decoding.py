"""
Population linear decoding of egocentric angle-to-goal via Ridge regression on sin/cos
of theta. Entry point: `decode_session_ego_angle_to_goal`.
"""

# %% Imports
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from GridMaze.analysis.core import convert
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import folds as folds_mod
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.distance_to_goal import distributions as dd
from GridMaze.analysis.goal_coding import decoding_utils as du

from GridMaze.paths import RESULTS_PATH

# %% Globals

FRAME_RATE = 60
RESULTS_DIR = RESULTS_PATH / "ego_angle" / "linreg_decoding"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REG_RANGE = np.logspace(-4, 4, 20)
FEATURE_TYPES = ("spikes", "place_direction_prob", "distance_prob")


# %% Population-level: populate / load


def get_ego_angle_decoding_df(
    goal_stratified=False,
    n_permutations=5,
    late_sessions=True,
    n_jobs=-1,
    resolution=0.25,
    verbose=True,
    save=False,
):
    """Run decode_session_ego_angle_to_goal across all sessions and return the concatenated df.

    If `save=False` and the cached parquet exists, load and return it.
    Otherwise run the decoder on every matching session, concat, save one parquet, and return.
    """
    _filename = "ego_angle_decoding_gs" if goal_stratified else "ego_angle_decoding"
    save_path = RESULTS_DIR / f"{_filename}.parquet"
    if not save and save_path.exists():
        if verbose:
            print(f"Loading cached results from {save_path}")
        return pd.read_parquet(save_path)

    if verbose:
        print("Loading sessions ...")
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names="all",
        days_on_maze="late" if late_sessions else "all",
        with_data=[
            "navigation_df",
            "navigation_spike_rates_df",
            "cluster_metrics",
            "trials_df",
        ],
        must_have_data=True,
    )

    def _process(session):
        if verbose:
            print(session.name)
        try:
            return decode_session_ego_angle_to_goal(
                session,
                goal_stratified=goal_stratified,
                resolution=resolution,
                n_permutations=n_permutations,
                verbose=False,
            )
        except Exception as e:
            print(f"[{session.name}] {type(e).__name__}: {e}")
            return None

    if n_jobs in (None, 1):
        dfs = [_process(s) for s in sessions]
    else:
        dfs = Parallel(n_jobs=n_jobs)(delayed(_process)(s) for s in sessions)
    dfs = [d for d in dfs if d is not None]
    if not dfs:
        return None
    results_df = pd.concat(dfs, axis=0, ignore_index=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(save_path)
    if verbose:
        print(f"Saved to {save_path}")
    return results_df


# %% Session-level decoder


def get_test_session():
    session = gs.get_maze_sessions(
        subject_IDs=["m6"],
        maze_names=["maze_1"],
        days_on_maze=[12],
        with_data=[
            "navigation_df",
            "navigation_spike_rates_df",
            "cluster_metrics",
            "trials_df",
        ],
        must_have_data=False,
    )
    return session


def decode_session_ego_angle_to_goal(
    session,
    resolution=0.25,
    include_multi_units=True,
    moving_only=False,
    max_steps_to_goal=30,
    goal_stratified=False,
    n_folds=5,
    sqrt_spikes=True,
    standardise_features=True,
    alpha="opt",
    n_permutations=5,
    training_trial_phases=("navigation",),
    verbose=False,
):
    """Run all three feature_types on one session and concatenate.

    - "spikes"               : real pass + n_permutations circular-shuffle passes
    - "place_direction_prob" : real pass only (serves as a control)
    - "distance_prob"        : real pass only (serves as a control)

    `permutation` column is NaN for real rows, integer 1..n_permutations for shuffles.

    Training fits only on samples whose trial_phase is in `training_trial_phases`;
    predictions are returned for all phases (ITI-of-next-trial, cue, navigation,
    reward_consumption) so decoding error can be aligned to cue/reward.
    """
    common = dict(
        resolution=resolution,
        include_multi_units=include_multi_units,
        moving_only=moving_only,
        max_steps_to_goal=max_steps_to_goal,
        goal_stratified=goal_stratified,
        n_folds=n_folds,
        sqrt_spikes=sqrt_spikes,
        standardise_features=standardise_features,
        alpha=alpha,
        training_trial_phases=tuple(training_trial_phases),
        verbose=verbose,
    )
    dfs = [
        _decode_session_ego_angle_to_goal(session, feature_type="spikes", n_permutations=n_permutations, **common),
        _decode_session_ego_angle_to_goal(session, feature_type="place_direction_prob", n_permutations=0, **common),
        _decode_session_ego_angle_to_goal(session, feature_type="distance_prob", n_permutations=0, **common),
    ]
    return pd.concat(dfs, axis=0, ignore_index=True)


def _decode_session_ego_angle_to_goal(
    session,
    resolution=0.25,
    feature_type="spikes",
    include_multi_units=True,
    moving_only=False,
    max_steps_to_goal=30,
    goal_stratified=False,
    n_folds=5,
    sqrt_spikes=True,
    standardise_features=True,
    alpha="opt",
    n_permutations=0,
    training_trial_phases=("navigation",),
    verbose=False,
):
    """Decode egocentric angle-to-goal from population activity (single feature_type).

    Ridge on sin(theta) and cos(theta); reconstruct theta_hat = arctan2(sin_hat, cos_hat).
    """
    if feature_type not in ("spikes", "place_direction_prob", "distance_prob"):
        raise ValueError(f"Unknown feature_type: {feature_type}")
    folds = _get_folds(session, goal_stratified=goal_stratified, n_folds=n_folds)
    # behavioural prep and control features are invariant across permutations — compute once
    input_data = _get_input_data(
        session,
        resolution=resolution,
        include_multi_units=include_multi_units,
        moving_only=moving_only,
        max_steps_to_goal=max_steps_to_goal,
    )
    if feature_type == "place_direction_prob":
        input_data = _attach_place_direction_probs(input_data, folds, verbose=verbose)
    elif feature_type == "distance_prob":
        input_data = _attach_distance_probs(input_data, folds, verbose=verbose)

    pass_dfs = []
    for perm_idx in range(n_permutations + 1):
        if perm_idx == 0:
            data = input_data
        else:
            data = _circular_permute_features(
                input_data, feature_type, np.random.default_rng(perm_idx), resolution=resolution
            )
        result = _run_cv(
            data,
            folds,
            feature_type=feature_type,
            sqrt_spikes=sqrt_spikes,
            standardise_features=standardise_features,
            alpha=alpha,
            training_trial_phases=training_trial_phases,
            verbose=verbose,
        )
        result["permutation"] = np.nan if perm_idx == 0 else perm_idx
        pass_dfs.append(result)
    results_df = pd.concat(pass_dfs, axis=0, ignore_index=True)
    results_df["subject_ID"] = session.subject_ID
    results_df["maze_name"] = session.maze_name
    results_df["day_on_maze"] = session.day_on_maze
    results_df["goal_subset"] = session.goal_subset
    results_df["feature_type"] = feature_type
    results_df["goal_stratified"] = goal_stratified
    return results_df


# %% Input data (load + optional circular permutation + downsample + filter)


def _get_input_data(
    session,
    resolution=0.25,
    include_multi_units=True,
    moving_only=False,
    max_steps_to_goal=30,
):
    """Frame-level trial/goal/angle recompute (ITI → next-trial's goal, RC → current-trial's goal),
    concat with activity, downsample, filter. All trial phases kept so test-time predictions can
    be aligned to cue/reward; the `max_steps_to_goal` filter only applies to navigation samples
    (non-nav samples have NaN steps_to_goal and are retained).
    """
    nav_df = _updated_navigation_df(session)
    activity_df = session._get_activity_df("rates", {"single_units": True, "multi_units": include_multi_units})
    ds_df = ds.downsample_navigation_activity_df(pd.concat([nav_df, activity_df], axis=1), resolution=resolution)
    # drop rows without a valid trial/goal/angle (session head/tail, post-last-trial ITI) before distance fills
    ds_df = ds_df.dropna(subset=[("trial_unique_ID", ""), ("goal", ""), ("angle_to_goal", "egocentric")]).reset_index(
        drop=True
    )
    # fill steps_to_goal.future (extended simple maze) and distance_to_goal.geodesic (skeleton maze)
    # for RC + ITI rows using the updated goals (raw nav_df only defines these for nav phase)
    ds_df[("steps_to_goal", "future")] = du.update_non_nav_distances(ds_df, session.simple_maze())
    ds_df[("distance_to_goal", "geodesic")] = _fill_non_nav_geodesic(ds_df, session.skeleton_maze())
    trials_df = session.trials_df
    ds_df[("event_aligned_time", "cue")] = _event_aligned_times(ds_df, trials_df, "cue")
    ds_df[("event_aligned_time", "reward")] = _event_aligned_times(ds_df, trials_df, "reward")
    if max_steps_to_goal is not None:
        nav_mask = ds_df.trial_phase == "navigation"
        steps_ok = ds_df.steps_to_goal.future < max_steps_to_goal
        ds_df = ds_df[~nav_mask | steps_ok]
    if moving_only:
        ds_df = ds_df[ds_df.moving]
    ds_df = ds_df.reset_index(drop=True)
    return ds_df


def _updated_navigation_df(session):
    """Reassign ITI samples to the *next* trial's goal (pre-first-cue ITI → trial 1), then recompute
    angle-to-goal per frame using the updated goal — so decoding extends outside the navigation phase."""
    nav_df = session.navigation_df.copy()
    trials_df = session.trials_df
    session_info = session.session_info
    skeleton_maze = session.skeleton_maze()

    # 1. ITI → next trial (with pre-first-cue patch to trial 1)
    updated_trial = du.update_trial_ID(nav_df, trials_df).values.astype(float)
    first_trial = trials_df.trial.min()
    first_cue_time = trials_df.set_index("trial").loc[first_trial, ("time", "cue")]
    pre_first_mask = (
        pd.isna(updated_trial)
        & (nav_df[("trial_phase", "")] == "ITI").values
        & (nav_df[("time", "")].values < first_cue_time)
    )
    updated_trial[pre_first_mask] = first_trial
    nav_df[("trial", "")] = updated_trial
    nav_df[("trial_unique_ID", "")] = convert.trial2trial_unique_ID(session_info, nav_df[("trial", "")])

    # 2. remap goal from updated trial
    trial2goal = trials_df.set_index("trial")["goal"].to_dict()
    nav_df[("goal", "")] = nav_df[("trial", "")].map(trial2goal)

    # 3. recompute allocentric + egocentric angle-to-goal per frame
    sk_label2sk_coord = {v: k for k, v in nx.get_node_attributes(skeleton_maze, "label").items()}
    sk_coord2pos = nx.get_node_attributes(skeleton_maze, "position")
    goal_to_pos = {g: sk_coord2pos[sk_label2sk_coord[f"{g}_C"]] for g in nav_df[("goal", "")].dropna().unique()}
    goal_x = nav_df[("goal", "")].map({g: p[0] for g, p in goal_to_pos.items()}).values.astype(float)
    goal_y = nav_df[("goal", "")].map({g: p[1] for g, p in goal_to_pos.items()}).values.astype(float)
    pos_x = nav_df[("centroid_position", "x")].values.astype(float)
    pos_y = nav_df[("centroid_position", "y")].values.astype(float)
    head_dir = nav_df[("head_direction", "value")].values.astype(float)
    allo = np.rad2deg(np.arctan2(goal_y - pos_y, goal_x - pos_x)) % 360
    ego = (allo - head_dir) % 360
    nav_df[("angle_to_goal", "allocentric")] = allo
    nav_df[("angle_to_goal", "egocentric")] = ego
    return nav_df


def _event_aligned_times(nav_info, trials_df, event):
    """Per-sample time relative to `event` (cue/reward) of that sample's (updated) trial."""
    trial2event_time = trials_df.set_index("trial")["time"][event]
    trial_ids = nav_info[("trial", "")].astype("Int64")
    event_times = trial_ids.map(trial2event_time)
    return nav_info[("time", "")] - event_times


def _fill_non_nav_geodesic(ds_df, skeleton_maze):
    """Geodesic distance (skeleton maze, weighted) from current position to updated goal, for RC + ITI rows.

    Navigation-phase rows keep their existing `distance_to_goal.geodesic` (computed in `get_navigation_df`).
    """
    sk_label2coord = {v: k for k, v in nx.get_node_attributes(skeleton_maze, "label").items()}
    shortest = dict(nx.all_pairs_dijkstra_path_length(skeleton_maze, weight="weight"))
    valid = ds_df.trial_phase.isin(["reward_consumption", "ITI"])
    src_labels = ds_df.loc[valid, ("maze_position", "skeleton")]
    dst_labels = ds_df.loc[valid, ("goal", "")].astype(str) + "_C"
    dists = []
    for s, d in zip(src_labels, dst_labels):
        sc, dc = sk_label2coord.get(s), sk_label2coord.get(d)
        if sc is None or dc is None:
            dists.append(np.nan)
        else:
            dists.append(shortest[sc].get(dc, np.nan))
    out = ds_df[("distance_to_goal", "geodesic")].copy()
    out.loc[valid.values] = dists
    return out


def _get_folds(session, goal_stratified, n_folds):
    """List of fold dicts: {'test', 'train', 'val_subfolds'} of trial_unique_IDs.

    Fold i holds out partition i as test; train = concat of other partitions;
    val_subfolds = other partitions (nested α CV runs leave-one-out over them).
    """
    partitions = _goal_partitions(session) if goal_stratified else _random_partitions(session, n_folds)
    folds = []
    for i, test in enumerate(partitions):
        others = [p for j, p in enumerate(partitions) if j != i]
        train = np.concatenate(others) if others else np.array([], dtype=object)
        folds.append({"test": test, "train": train, "val_subfolds": others})
    return folds


def _goal_partitions(session):
    """One partition per goal: all trial_unique_IDs belonging to that goal."""
    goals_df = folds_mod.get_goals_df(session, return_unique_IDs=True)
    return [row.dropna().values for _, row in goals_df.iterrows() if row.notna().any()]


def _random_partitions(session, n_folds):
    """Random K-way split of the session's trials."""
    session_info = session.session_info
    trials = session.trials_df.trial.dropna().unique()
    np.random.shuffle(trials)
    return [
        np.array(convert.trial2trial_unique_ID(session_info, list(chunk))) for chunk in np.array_split(trials, n_folds)
    ]


def _circular_permute_features(df, feature_type, rng, resolution=0.25, min_offset_seconds=5):
    """Circularly shift the feature block (rates or prob features) along the sample axis,
    breaking its alignment with behaviour for a chance baseline."""
    feature_level = {
        "spikes": "firing_rate",
        "place_direction_prob": "place_direction_prob",
        "distance_prob": "distance_prob",
    }[feature_type]
    feat_cols = df.xs(feature_level, level=0, axis=1, drop_level=False).columns
    n = len(df)
    min_offset = max(1, int(min_offset_seconds / resolution))
    if n <= 2 * min_offset:
        min_offset = max(1, n // 4)
    offset = int(rng.integers(min_offset, n - min_offset))
    df = df.copy()
    df[feat_cols] = np.roll(df[feat_cols].values, shift=offset, axis=0)
    return df


# %% Control feature generation (xvaled spikes -> predicted X probs)


def _attach_place_direction_probs(input_data, folds, C=1.0, verbose=False):
    """Per-sample cross-validated predicted P(place × direction) from firing rates."""
    input_data = input_data.copy()
    input_data[("place_direction", "")] = (
        input_data[("maze_position", "simple")].astype(str)
        + "_"
        + input_data[("cardinal_movement_direction", "")].astype(str)
    )
    all_labels = sorted(input_data[("place_direction", "")].dropna().unique())
    label_to_col = {lab: i for i, lab in enumerate(all_labels)}
    fr = input_data["firing_rate"].values.astype(float)
    y = input_data[("place_direction", "")].values
    phase_mask = input_data.trial_phase.isin(["navigation"]).values

    n = len(input_data)
    prob_matrix = np.zeros((n, len(all_labels)))
    for i, fold in enumerate(folds):
        if verbose:
            print(f"    place_direction xval fold_{i}")
        train_mask = input_data.trial_unique_ID.isin(fold["train"]).values & phase_mask
        test_mask = input_data.trial_unique_ID.isin(fold["test"]).values
        if not train_mask.any() or not test_mask.any():
            continue
        X_train = np.sqrt(np.maximum(fr[train_mask], 0))
        X_test = np.sqrt(np.maximum(fr[test_mask], 0))
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        model = LogisticRegression(penalty="l2", C=C, max_iter=10_000, random_state=0, class_weight="balanced")
        model.fit(X_train, y[train_mask])
        probs = model.predict_proba(X_test)
        test_idx = np.where(test_mask)[0]
        for k, cls in enumerate(model.classes_):
            prob_matrix[test_idx, label_to_col[cls]] = probs[:, k]
    cols = pd.MultiIndex.from_product([["place_direction_prob"], all_labels])
    prob_df = pd.DataFrame(prob_matrix, index=input_data.index, columns=cols)
    return pd.concat([input_data, prob_df], axis=1)


def _attach_distance_probs(input_data, folds, C=1.0, bin_spacing=0.05, verbose=False):
    """Per-sample cross-validated predicted P(distance-to-goal bin) from firing rates."""
    metric = ("distance_to_goal", "geodesic")
    max_distance = dd.get_distance_percentile(metric, 0.85)
    n_bins = int(max_distance / bin_spacing)
    bins = convert._get_distance_bins(
        binning_method="uniform",
        n_distance_bins=n_bins,
        distance_metrics=metric,
        max_distance=max_distance,
    )
    input_data = input_data.copy()
    input_data = input_data[input_data[metric] < max_distance].reset_index(drop=True)
    input_data.loc[:, ("distance_bin", "")] = pd.cut(input_data[metric], bins=bins, include_lowest=True).to_numpy()
    bin_ids = {b: i for i, b in enumerate(bins)}
    input_data.loc[:, ("distance_bin_id", "")] = input_data.distance_bin.map(bin_ids)

    fr = input_data["firing_rate"].values.astype(float)
    y = input_data[("distance_bin_id", "")].values
    phase_mask = input_data.trial_phase.isin(["navigation"]).values

    n = len(input_data)
    n_classes = len(bins)
    prob_matrix = np.zeros((n, n_classes))
    for i, fold in enumerate(folds):
        if verbose:
            print(f"    distance xval fold_{i}")
        train_mask = input_data.trial_unique_ID.isin(fold["train"]).values & phase_mask
        test_mask = input_data.trial_unique_ID.isin(fold["test"]).values
        if not train_mask.any() or not test_mask.any():
            continue
        X_train = np.sqrt(np.maximum(fr[train_mask], 0))
        X_test = np.sqrt(np.maximum(fr[test_mask], 0))
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        model = LogisticRegression(penalty="l2", C=C, max_iter=10_000, random_state=0, class_weight="balanced")
        model.fit(X_train, y[train_mask])
        probs = model.predict_proba(X_test)
        test_idx = np.where(test_mask)[0]
        for k, cls in enumerate(model.classes_):
            prob_matrix[test_idx, int(cls)] = probs[:, k]
    cols = pd.MultiIndex.from_product([["distance_prob"], list(range(n_classes))])
    prob_df = pd.DataFrame(prob_matrix, index=input_data.index, columns=cols)
    return pd.concat([input_data, prob_df], axis=1)


# %% CV loop (Ridge sin/cos per fold)


def _run_cv(
    input_data,
    folds,
    feature_type,
    sqrt_spikes,
    standardise_features,
    alpha,
    training_trial_phases=("navigation",),
    verbose=False,
):
    theta = np.deg2rad(input_data[("angle_to_goal", "egocentric")].values.astype(float))
    y_sin_all, y_cos_all = np.sin(theta), np.cos(theta)
    training_phase_mask = input_data.trial_phase.isin(list(training_trial_phases)).values
    row_dfs = []
    for i, fold in enumerate(folds):
        fold_name = f"fold_{i}"
        if verbose:
            print(f"  {fold_name}")
        # train on specified trial phases only; test on all phases
        train_mask = input_data.trial_unique_ID.isin(fold["train"]).values & training_phase_mask
        test_mask = input_data.trial_unique_ID.isin(fold["test"]).values
        if not test_mask.any() or not train_mask.any():
            continue
        X_train = _build_X(input_data, train_mask, feature_type, sqrt_spikes)
        X_test = _build_X(input_data, test_mask, feature_type, sqrt_spikes)
        scaler = None
        if standardise_features:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        ys_train, yc_train = y_sin_all[train_mask], y_cos_all[train_mask]
        if alpha == "opt":
            opt_alpha = _search_alpha(
                input_data,
                fold,
                feature_type,
                sqrt_spikes,
                standardise_features,
                training_trial_phases=training_trial_phases,
                verbose=verbose,
            )
        else:
            opt_alpha = float(alpha)
        m_sin = Ridge(alpha=opt_alpha, random_state=0).fit(X_train, ys_train)
        m_cos = Ridge(alpha=opt_alpha, random_state=0).fit(X_train, yc_train)
        ys_hat = m_sin.predict(X_test)
        yc_hat = m_cos.predict(X_test)
        theta_hat_deg = np.rad2deg(np.arctan2(ys_hat, yc_hat)) % 360
        true_angle = input_data.loc[test_mask, ("angle_to_goal", "egocentric")].values % 360
        d = np.abs(theta_hat_deg - true_angle)
        circ_err = np.minimum(d, 360 - d)
        test_df = input_data.loc[test_mask]
        rows = pd.DataFrame(
            {
                "trial_unique_ID": test_df[("trial_unique_ID", "")].values,
                "trial": test_df[("trial", "")].values,
                "goal": test_df[("goal", "")].values,
                "time": test_df[("time", "")].values,
                "trial_phase": test_df[("trial_phase", "")].values,
                "moving": test_df[("moving", "")].values,
                "maze_position": test_df[("maze_position", "simple")].values,
                "cardinal_movement_direction": test_df[("cardinal_movement_direction", "")].values,
                "distance_to_goal": test_df[("distance_to_goal", "geodesic")].values,
                "steps_to_goal": test_df[("steps_to_goal", "future")].values,
                "speed": test_df[("speed", "")].values,
                "cue_aligned_time": test_df[("event_aligned_time", "cue")].values,
                "reward_aligned_time": test_df[("event_aligned_time", "reward")].values,
                "true_angle": true_angle,
                "decoded_angle": theta_hat_deg,
                "circular_error": circ_err,
                "fold": fold_name,
                "opt_alpha": opt_alpha,
            }
        )
        row_dfs.append(rows)
    return pd.concat(row_dfs, axis=0, ignore_index=True)


def _build_X(input_data, mask, feature_type, sqrt_spikes):
    if feature_type == "spikes":
        X = input_data.loc[mask, "firing_rate"].values.astype(float)
        if sqrt_spikes:
            X = np.sqrt(np.maximum(X, 0))
    elif feature_type == "place_direction_prob":
        X = input_data.loc[mask, "place_direction_prob"].values.astype(float)
    elif feature_type == "distance_prob":
        X = input_data.loc[mask, "distance_prob"].values.astype(float)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")
    return X


def _search_alpha(
    input_data,
    fold,
    feature_type,
    sqrt_spikes,
    standardise_features,
    reg_range=None,
    training_trial_phases=("navigation",),
    verbose=False,
):
    """Nested CV over train-side validation subfolds; pick alpha minimising mean circular error.

    Fits on validation-train samples of the specified training_trial_phases;
    evaluates on validation-test samples of all phases (matches outer train/test pattern).
    """
    if reg_range is None:
        reg_range = REG_RANGE
    training_phase_mask = input_data.trial_phase.isin(list(training_trial_phases)).values
    val_subfolds = fold["val_subfolds"]
    if len(val_subfolds) < 2:
        return 1.0
    best_alphas = []
    for v, val_trials in enumerate(val_subfolds):
        vtrain_trials = np.concatenate([p for j, p in enumerate(val_subfolds) if j != v])
        if len(val_trials) == 0 or len(vtrain_trials) == 0:
            continue
        v_mask = input_data.trial_unique_ID.isin(val_trials).values
        vt_mask = input_data.trial_unique_ID.isin(vtrain_trials).values & training_phase_mask
        if not v_mask.any() or not vt_mask.any():
            continue
        X_vt = _build_X(input_data, vt_mask, feature_type, sqrt_spikes)
        X_v = _build_X(input_data, v_mask, feature_type, sqrt_spikes)
        if standardise_features:
            sc = StandardScaler().fit(X_vt)
            X_vt = sc.transform(X_vt)
            X_v = sc.transform(X_v)
        theta_vt = np.deg2rad(input_data.loc[vt_mask, ("angle_to_goal", "egocentric")].values.astype(float))
        ys_vt, yc_vt = np.sin(theta_vt), np.cos(theta_vt)
        theta_v_deg = input_data.loc[v_mask, ("angle_to_goal", "egocentric")].values % 360
        best_a, best_e = None, np.inf
        for a in reg_range:
            m_s = Ridge(alpha=a, random_state=0).fit(X_vt, ys_vt)
            m_c = Ridge(alpha=a, random_state=0).fit(X_vt, yc_vt)
            pred_deg = np.rad2deg(np.arctan2(m_s.predict(X_v), m_c.predict(X_v))) % 360
            d = np.abs(pred_deg - theta_v_deg)
            err = np.minimum(d, 360 - d).mean()
            if err < best_e:
                best_e, best_a = err, a
        if best_a is not None:
            best_alphas.append(best_a)
    if not best_alphas:
        return 1.0
    return float(np.median(best_alphas))


# %% Plotting


_CONDITIONS = [
    ("spikes", lambda d: (d.feature_type == "spikes") & d.permutation.isna(), "royalblue"),
    ("spikes (shuffle)", lambda d: (d.feature_type == "spikes") & d.permutation.notna(), "grey"),
    ("place+direction", lambda d: (d.feature_type == "place_direction_prob") & d.permutation.isna(), "darkorange"),
    ("distance", lambda d: (d.feature_type == "distance_prob") & d.permutation.isna(), "seagreen"),
]


def plot_session_decoding_summary(
    results_df,
    time_bin=0.5,
    event_window=(-5, 5),
    chance=90,
    figsize=(14, 6),
    axes=None,
    print_stats=True,
):
    """Session diagnostic (2 × 4 layout) for a `decode_session_ego_angle_to_goal` output.

    Row 1: cue-aligned error | reward-aligned error | nav-phase bar chart | error vs true angle.
    Row 2: 4 × true-vs-decoded 2D histograms (one per condition).
    """
    if axes is None:
        fig, axes_arr = plt.subplots(2, 4, figsize=figsize, constrained_layout=True)
        axes = {
            "cue": axes_arr[0, 0],
            "reward": axes_arr[0, 1],
            "bar": axes_arr[0, 2],
            "angle": axes_arr[0, 3],
            **{f"hist_{name}": axes_arr[1, i] for i, (name, _, _) in enumerate(_CONDITIONS)},
        }

    _plot_event_aligned(results_df, axes["cue"], "cue", time_bin, event_window, chance)
    _plot_event_aligned(results_df, axes["reward"], "reward", time_bin, event_window, chance)
    _plot_nav_bar(results_df, axes["bar"], chance)
    _plot_error_vs_true_angle(results_df, axes["angle"], chance)
    for name, mask_fn, _ in _CONDITIONS:
        _plot_confusion(results_df, axes[f"hist_{name}"], name, mask_fn)

    if print_stats:
        _print_nav_stats(results_df)


def _cond_bin_means(d, bin_col, is_shuffle):
    """Mean circular_error per bin; for shuffle, first average per permutation then across permutations."""
    if is_shuffle:
        per_perm = d.groupby(["permutation", bin_col], observed=True).circular_error.mean()
        mean = per_perm.groupby(bin_col).mean()
        sem = per_perm.groupby(bin_col).sem()
    else:
        grouped = d.groupby(bin_col, observed=True).circular_error
        mean = grouped.mean()
        sem = grouped.sem()
    return mean, sem


def _plot_event_aligned(results_df, ax, event, time_bin, window, chance):
    col = f"{event}_aligned_time"
    edges = np.arange(window[0], window[1] + time_bin, time_bin)
    centres = edges[:-1] + time_bin / 2
    in_window = results_df[col].between(window[0], window[1])
    df = results_df[in_window].copy()
    df["_tbin"] = pd.cut(df[col], bins=edges, labels=centres, include_lowest=True).astype(float)
    for name, mask_fn, color in _CONDITIONS:
        d = df[mask_fn(df)]
        if d.empty:
            continue
        mean, sem = _cond_bin_means(d, "_tbin", is_shuffle=(name == "spikes (shuffle)"))
        ax.plot(mean.index.values, mean.values, color=color, label=name, lw=1.5)
        ax.fill_between(mean.index.values, mean - sem, mean + sem, color=color, alpha=0.2)
    ax.axvline(0, color="k", ls="--", alpha=0.5)
    ax.axhline(chance, color="k", ls=":", alpha=0.5)
    ax.set_xlabel(f"time from {event} (s)")
    ax.set_ylabel("circular error (°)")
    ax.spines[["top", "right"]].set_visible(False)
    if event == "cue":
        ax.legend(fontsize=7, loc="best")


def _plot_nav_bar(results_df, ax, chance):
    df = results_df[results_df.trial_phase == "navigation"]
    labels, means, errs, colors = [], [], [], []
    for name, mask_fn, color in _CONDITIONS:
        d = df[mask_fn(df)]
        if d.empty:
            continue
        if name == "spikes (shuffle)":
            per_perm = d.groupby("permutation").circular_error.mean()
            m, e = per_perm.mean(), per_perm.sem()
        else:
            m, e = d.circular_error.mean(), d.circular_error.sem()
        labels.append(name)
        means.append(m)
        errs.append(e)
        colors.append(color)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errs, color=colors, capsize=3)
    ax.axhline(chance, color="k", ls=":", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean circular error (°) — nav phase")
    ax.spines[["top", "right"]].set_visible(False)


def _plot_error_vs_true_angle(results_df, ax, chance, n_bins=24):
    edges = np.linspace(0, 360, n_bins + 1)
    centres = edges[:-1] + (edges[1] - edges[0]) / 2
    df = results_df.copy()
    df["_abin"] = pd.cut(df.true_angle % 360, bins=edges, labels=centres, include_lowest=True).astype(float)
    for name, mask_fn, color in _CONDITIONS:
        d = df[mask_fn(df)]
        if d.empty:
            continue
        mean, sem = _cond_bin_means(d, "_abin", is_shuffle=(name == "spikes (shuffle)"))
        ax.plot(mean.index.values, mean.values, color=color, label=name, lw=1.5)
        ax.fill_between(mean.index.values, mean - sem, mean + sem, color=color, alpha=0.2)
    ax.axhline(chance, color="k", ls=":", alpha=0.5)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xticklabels(["front", "right", "behind", "left", "front"])
    ax.set_xlabel("true ego angle to goal")
    ax.set_ylabel("circular error (°)")
    ax.spines[["top", "right"]].set_visible(False)


def _plot_confusion(results_df, ax, name, mask_fn, n_bins=36):
    d = results_df[mask_fn(results_df)]
    if d.empty:
        ax.set_title(f"{name}\n(no data)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    true = d.true_angle.values % 360
    pred = d.decoded_angle.values % 360
    H, xe, ye = np.histogram2d(true, pred, bins=n_bins, range=[[0, 360], [0, 360]])
    ax.imshow(H.T, origin="lower", extent=(0, 360, 0, 360), aspect="equal", cmap="magma")
    ax.plot([0, 360], [0, 360], color="white", ls="--", lw=0.8, alpha=0.7)
    ax.set_xticks([0, 180, 360])
    ax.set_yticks([0, 180, 360])
    ax.set_xlabel("true (°)")
    ax.set_ylabel("decoded (°)")
    ax.set_title(name, fontsize=9)


def _print_nav_stats(results_df):
    df = results_df[results_df.trial_phase == "navigation"]
    print(f"{'condition':<18} {'mean':>10} {'median':>10} {'N':>10}")
    print("-" * 52)
    for name, mask_fn, _ in _CONDITIONS:
        d = df[mask_fn(df)]
        if d.empty:
            print(f"{name:<18} {'—':>10} {'—':>10} {0:>10}")
            continue
        if name == "spikes (shuffle)":
            per_perm = d.groupby("permutation").circular_error.mean()
            mean_str = f"{per_perm.mean():.1f} ± {per_perm.sem():.1f}°"
        else:
            mean_str = f"{d.circular_error.mean():.1f}°"
        print(f"{name:<18} {mean_str:>10} {d.circular_error.median():>9.1f}° {len(d):>10}")


# %% Population-level event-aligned plot


_SESSION_KEY = ["subject_ID", "maze_name", "day_on_maze", "goal_subset"]

def plot_event_aligned_decoding(
    results_df,
    baseline="place_direction_prob",
    time_bin=0.5,
    event_window=(-5, 5),
    figsize=(7, 3),
    color="royalblue",
    axes=None,
    return_data=False,
):
    """Population-level event-aligned spikes decoding advantage over a chosen baseline.

    Pipeline per event ∈ {cue, reward}:
      1. Per (session, time bin), compute mean circular error for spikes and for the baseline.
         Baselines: "place_direction_prob", "distance_prob", "shuffle" (permuted-spikes, averaged
         across permutations first).
      2. performance = baseline_error − spikes_error  (positive ⇒ spikes decodes better).
      3. Mean across sessions within subject; grand mean ± SEM across subjects.
      4. Plot one line + shaded SEM on each panel.
    """
    if axes is None:
        fig, axes_arr = plt.subplots(1, 2, figsize=figsize, sharey=True, constrained_layout=True)
        axes = {"cue": axes_arr[0], "reward": axes_arr[1]}
    data = {}
    for event, ax in (("cue", axes["cue"]), ("reward", axes["reward"])):
        curves, centres = _spikes_advantage_curves(results_df, event, baseline, time_bin, event_window)
        data[event] = curves
        mean = curves["mean"].reindex(centres)
        sem = curves["sem"].reindex(centres)
        ax.plot(centres, mean.values, color=color, lw=1.5)
        ax.fill_between(centres, (mean - sem).values, (mean + sem).values, color=color, alpha=0.25)
        ax.axvline(0, color="k", ls="--", alpha=0.5)
        ax.axhline(0, color="k", ls=":", alpha=0.5)
        ax.set_xlabel(f"time from {event} (s)")
        ax.spines[["top", "right"]].set_visible(False)
    axes["cue"].set_ylabel(f"{baseline} − spikes error (°)")
    if return_data:
        return data


def _baseline_error_per_session_bin(df, baseline):
    """Mean circular error per (session, _tbin) for the chosen baseline."""
    if baseline == "shuffle":
        mask = (df.feature_type == "spikes") & df.permutation.notna()
        return (
            df[mask]
            .groupby(_SESSION_KEY + ["permutation", "_tbin"], observed=True).circular_error.mean()
            .groupby(_SESSION_KEY + ["_tbin"], observed=True).mean()
        )
    if baseline in ("place_direction_prob", "distance_prob"):
        mask = (df.feature_type == baseline) & df.permutation.isna()
        return df[mask].groupby(_SESSION_KEY + ["_tbin"], observed=True).circular_error.mean()
    raise ValueError(f"Unknown baseline: {baseline!r}")


def _spikes_advantage_curves(results_df, event, baseline, time_bin, window):
    """Returns {'mean', 'sem', 'subjects'} (wide df) of (baseline - spikes) error per time bin."""
    col = f"{event}_aligned_time"
    edges = np.arange(window[0], window[1] + time_bin, time_bin)
    centres = edges[:-1] + time_bin / 2
    df = results_df[results_df[col].between(window[0], window[1])].copy()
    df["_tbin"] = pd.cut(df[col], bins=edges, labels=centres, include_lowest=True).astype(float)

    spikes_mask = (df.feature_type == "spikes") & df.permutation.isna()
    spikes_err = df[spikes_mask].groupby(_SESSION_KEY + ["_tbin"], observed=True).circular_error.mean()
    baseline_err = _baseline_error_per_session_bin(df, baseline)
    perf = baseline_err - spikes_err  # positive ⇒ spikes better

    subject_bin = perf.groupby(["subject_ID", "_tbin"], observed=True).mean()
    wide = subject_bin.unstack("subject_ID")  # rows=_tbin, cols=subjects
    return {"mean": wide.mean(axis=1), "sem": wide.sem(axis=1), "subjects": wide}, centres
