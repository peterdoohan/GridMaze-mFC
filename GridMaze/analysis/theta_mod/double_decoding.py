"""
Sanity-check decoder that runs place-direction and distance-to-goal theta-mod
decoding on identical fold splits per session. Single source of truth for the
underlying decoders: this module composes the existing `place_direction_decoding`
and `distance_to_goal_decoder2` subfunctions; no decoder logic is duplicated.
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from GridMaze.analysis.core import folds
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.theta_mod import theta_utils as tmu
from GridMaze.analysis.theta_mod import place_direction_decoding as pdd
from GridMaze.analysis.theta_mod import distance_to_goal_decoder2 as ddv2

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "theta_mod"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]


# %% Session-level matched-fold decoder


def get_session_double_decoding(
    session,
    n_folds=16,
    C=1,
    reg_search_folds=8,
    sqrt_spikes=True,
    normalise_X=True,
    output="weighted",
    include_multi_units=True,
    max_steps_to_goal=30,
    resolution=0.1,
    sum_spike_window=0.4,
    envelope=2,
    max_distance=0.8,
    bin_spacing=0.04,
    bin_method="uniform",
    verbose=False,
):
    """Run place and distance decoders on the same CV fold splits.

    Returns a single long df with `decoder_type` ∈ {"place", "distance"} on each row.
    Sign convention (matches both source modules): +ve `signed_error` = decoder
    predicts location further from goal (past); -ve = closer (future).
    """
    # 1. Build both input dfs (reusing each pipeline's own loader)
    place_df = pdd.get_input_data(
        session,
        include_multi_units=include_multi_units,
        max_steps_to_goal=max_steps_to_goal,
        sum_spike_window=sum_spike_window,
        resolution=resolution,
        envelope=envelope,
    )
    place_df = place_df[place_df[["past", "future"]].notnull().all(axis=1)]
    dist_df = ddv2.get_input_data(
        session,
        include_multi_units=include_multi_units,
        max_steps_to_goal=max_steps_to_goal,
        resolution=resolution,
        sum_spike_window=sum_spike_window,
        max_distance=max_distance,
        bin_spacing=bin_spacing,
        bin_method=bin_method,
    )

    # 2. Matched folds on trial intersection
    common_trials = np.intersect1d(place_df.trial.unique(), dist_df.trial.unique())
    folds_df = folds.get_folds_df(
        session,
        goal_stratified=False,
        valid_trials=common_trials,
        n_folds=n_folds,
        return_unique_IDs=False,
    )

    theta_phases = place_df.spike_count.columns.get_level_values(1).unique().astype(float)
    distance_bin_mids = np.array(sorted(dist_df.distance_bin_mid.unique()))

    # 3. Fold loop
    results = []
    for fold in folds_df.columns.get_level_values(0).unique():
        if verbose:
            print(fold)
        fold_df = folds_df[fold]
        train_trials = fold_df["train"].unstack().dropna().values
        test_trials = fold_df["test"].unstack().dropna().values

        # ----- PLACE -----
        train_p = place_df[place_df.trial.isin(train_trials)]
        test_p = place_df[place_df.trial.isin(test_trials)]
        place_C = (
            pdd._get_opt_C(
                fold_df,
                train_p,
                sqrt_spikes=sqrt_spikes,
                normalise_X=normalise_X,
                reg_search_folds=reg_search_folds,
            )
            if C == "opt"
            else C
        )
        Xtr_p = pdd._prepare_X(train_p, phase=None, sqrt=sqrt_spikes)
        scaler_p = StandardScaler().fit(Xtr_p) if normalise_X else None
        if scaler_p is not None:
            Xtr_p = scaler_p.transform(Xtr_p)
        Ytr_p = train_p.maze_position.simple.values
        place_decoder = LogisticRegression(
            C=place_C, max_iter=10_000, random_state=0, class_weight="balanced"
        ).fit(Xtr_p, Ytr_p)

        Xte_mean_p = pdd._prepare_X(test_p, phase=None, sqrt=sqrt_spikes)
        if scaler_p is not None:
            Xte_mean_p = scaler_p.transform(Xte_mean_p)
        Yte_p = test_p.maze_position.simple.values
        fold_accuracy = place_decoder.score(Xte_mean_p, Yte_p)
        fold_n_classes = len(place_decoder.classes_)

        place_meta = pd.DataFrame(
            {
                "time": test_p[("time", "", "")].to_numpy(),
                "trial": test_p[("trial", "", "")].to_numpy(),
                "trial_unique_ID": test_p[("trial_unique_ID", "", "")].to_numpy(),
                "place_direction": test_p[("place_direction", "", "")].to_numpy(),
                "distance_to_goal": test_p[("distance_to_goal", "geodesic", "")].to_numpy(),
            },
            index=test_p.index,
        )
        place_meta["fold_accuracy"] = fold_accuracy
        place_meta["fold_n_classes"] = fold_n_classes

        for phase in theta_phases:
            Xp = pdd._prepare_X(test_p, phase=phase, sqrt=sqrt_spikes)
            if scaler_p is not None:
                Xp = scaler_p.transform(Xp)
            Yprob = place_decoder.predict_proba(Xp)
            errs = pdd._get_trajectory_error(Yprob, test_p, place_decoder.classes_)
            res = place_meta.copy()
            res["theta_phase"] = float(phase)
            res["fold"] = fold
            res["decoder_type"] = "place"
            res["signed_error"] = errs["signed_error"]
            res["all_envelope_defined"] = errs["all_envelope_defined"]
            results.append(res)

        # ----- DISTANCE -----
        train_d = dist_df[dist_df.trial.isin(train_trials)]
        test_d = dist_df[dist_df.trial.isin(test_trials)]
        dist_C = (
            ddv2._get_opt_C(
                dist_df,
                fold_df,
                output=output,
                sqrt_spikes=sqrt_spikes,
                normalise_X=normalise_X,
                reg_search_folds=reg_search_folds,
                distance_bin_mids=distance_bin_mids,
            )
            if C == "opt"
            else C
        )
        Xtr_d = ddv2._prepare_X(train_d, phase=None, sqrt=sqrt_spikes)
        scaler_d = StandardScaler().fit(Xtr_d) if normalise_X else None
        if scaler_d is not None:
            Xtr_d = scaler_d.transform(Xtr_d)
        ytr_d = train_d.distance_bin_id.values
        dist_decoder = LogisticRegression(
            C=dist_C, max_iter=10_000, random_state=0, class_weight="balanced"
        ).fit(Xtr_d, ytr_d)

        Xte_mean_d = ddv2._prepare_X(test_d, phase=None, sqrt=sqrt_spikes)
        if scaler_d is not None:
            Xte_mean_d = scaler_d.transform(Xte_mean_d)
        true_dist = distance_bin_mids[test_d.distance_bin_id.values]
        baseline_pred = ddv2._decode_distance(dist_decoder, Xte_mean_d, distance_bin_mids, output=output)
        fold_baseline_mae = float(np.mean(np.abs(baseline_pred - true_dist)))

        dist_meta = pd.DataFrame(
            {
                "time": test_d[("time", "", "")].to_numpy(),
                "trial": test_d[("trial", "", "")].to_numpy(),
                "trial_unique_ID": test_d[("trial_unique_ID", "", "")].to_numpy(),
                "distance_to_goal": test_d[("distance_to_goal", "geodesic", "")].to_numpy(),
                "distance_bin_mid": test_d[("distance_bin_mid", "", "")].to_numpy().astype(float),
                "speed": test_d[("speed", "", "")].to_numpy(),
            },
            index=test_d.index,
        )
        dist_meta["fold_baseline_mae"] = fold_baseline_mae

        for phase in theta_phases:
            Xp = ddv2._prepare_X(test_d, phase=phase, sqrt=sqrt_spikes)
            if scaler_d is not None:
                Xp = scaler_d.transform(Xp)
            y_pred_dist = ddv2._decode_distance(dist_decoder, Xp, distance_bin_mids, output=output)
            res = dist_meta.copy()
            res["theta_phase"] = float(phase)
            res["fold"] = fold
            res["decoder_type"] = "distance"
            res["signed_error"] = y_pred_dist - true_dist  # +ve = further than truth
            results.append(res)

    return pd.concat(results, ignore_index=True)


# %% Cross-session runner


def get_theta_mod_double_decoding_df(verbose=True, C=1, days_on_maze="late", save=False):
    """Run the matched-fold decoder across all subjects × mazes and concat results.

    Cached to parquet. Pass `save=True` to force rerun and overwrite the cache.
    """
    save_path = RESULTS_DIR / "theta_mod_double_decoding_df.parquet"
    if save_path.exists() and not save:
        return pd.read_parquet(save_path)

    def _process_session(session):
        if verbose:
            print(session.name)
        try:
            res = get_session_double_decoding(session, C=C, verbose=False)
            res["subject_ID"] = session.subject_ID
            res["maze_name"] = session.maze_name
            res["day_on_maze"] = session.day_on_maze
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
                days_on_maze=days_on_maze,
                with_data=["navigation_df", "navigation_theta_spike_counts_df", "cluster_metrics", "trials_df"],
                must_have_data=True,
            )
            if not sessions:
                continue
            session_dfs = Parallel(n_jobs=-1)(delayed(_process_session)(s) for s in sessions)
            dfs.extend([d for d in session_dfs if d is not None])

    summary_df = pd.concat(dfs, ignore_index=True)
    if save:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_parquet(save_path)
    return summary_df


# %% Plotting


def plot_double_decoding_bias(
    summary_df,
    distance_to_goal=None,
    place_color="darkred",
    dist_color="darkblue",
    label_place="place",
    label_dist="distance",
    print_stats=True,
    ax=None,
):
    """Per-subject sinusoid of place and distance decoding bias overlaid on a
    single axis, both computed from the matched-fold double-decoding df.

    Sign convention: +ve bias = decoder predicts location further from goal (past).
    Both biases plotted in cm (per-subject mean-subtracted then × 100).
    """
    df = summary_df
    if distance_to_goal is not None:
        lo, hi = distance_to_goal
        df = df[df.distance_to_goal.between(lo, hi)]

    biases = {}
    for kind in ["place", "distance"]:
        sub = df[df.decoder_type == kind]
        b = sub.groupby(["subject_ID", "theta_phase"])["signed_error"].mean().unstack(0).T
        biases[kind] = b.sub(b.mean(axis=1), axis=0) * 100  # m -> cm

    tmu.plot_decoding_bias(
        biases["place"],
        color=place_color,
        label=label_place,
        ylabel="decoding bias (cm)",
        print_stats=print_stats,
        ax=ax,
    )
    tmu.plot_decoding_bias(
        biases["distance"],
        color=dist_color,
        label=label_dist,
        ylabel="decoding bias (cm)",
        print_stats=print_stats,
        ax=ax,
    )
