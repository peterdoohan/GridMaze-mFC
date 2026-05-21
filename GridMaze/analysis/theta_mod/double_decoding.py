"""
Sanity-check decoder that runs place-direction and distance-to-goal theta-mod
decoding on the SAME train/test rows per fold (leave-one-trial-out). Output is
a wide long-df with `signed_error_place` and `signed_error_distance` on every
row. Single source of truth for the underlying decoders: this module composes
the existing `place_direction_decoding` and `distance_to_goal_decoder2`
subfunctions; no decoder logic is duplicated.
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

from GridMaze.analysis.core import convert
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.theta_mod import theta_utils as tmu
from GridMaze.analysis.theta_mod import place_direction_decoding as pdd
from GridMaze.analysis.theta_mod import distance_to_goal_decoder2 as ddv2
from GridMaze.analysis.theta_mod import decoding_offsets as dox

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "theta_mod"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]


# %% Session-level matched-sample decoder


def get_session_double_decoding(
    session,
    C=1,
    reg_search_folds=4,
    random_state=0,
    sqrt_spikes=True,
    normalise_X=True,
    output="weighted",
    include_multi_units=True,
    max_steps_to_goal=16,
    resolution=0.1,
    sum_spike_window=0.4,
    envelope=2,
    bin_spacing=0.05,
    bin_method="uniform",
    verbose=False,
):
    """Run place and distance decoders on identical samples per leave-one-trial-out fold.

    Common filter: `steps_to_goal.future < max_steps_to_goal` (default 16, ~8 towers)
    AND `±envelope` defined (default 2 steps). Both decoders train and test on
    the SAME rows. Output is a long df with `signed_error_place` and
    `signed_error_distance` columns on every row.

    Sign convention (matches both source modules): +ve `signed_error_*` =
    decoder predicts location further from goal (past); -ve = closer (future).
    """
    # 1. Build common input (envelope + place_direction; max_steps_to_goal applied via filt.filter_navigation_rates_df)
    df = pdd.get_input_data(
        session,
        include_multi_units=include_multi_units,
        max_steps_to_goal=max_steps_to_goal,
        sum_spike_window=sum_spike_window,
        resolution=resolution,
        envelope=envelope,
    )
    # 2. Drop trial-boundary samples without full envelope
    df = df[df[["past", "future"]].notnull().all(axis=1)]

    # 3. Inline distance binning over the observed range
    metric = ("distance_to_goal", "geodesic", "")
    observed_max = float(df[metric].max())
    n_bins = int(np.ceil(observed_max / bin_spacing))
    binning_max = n_bins * bin_spacing  # next bin edge above observed max
    bins = convert._get_distance_bins(
        binning_method=bin_method,
        n_distance_bins=n_bins,
        distance_metrics=("distance_to_goal", "geodesic"),
        max_distance=binning_max,
    )
    df.loc[:, ("distance_bin", "", "")] = pd.cut(df[metric], bins=bins, include_lowest=True).to_numpy()
    df.loc[:, ("distance_bin_mid", "", "")] = df.distance_bin.apply(lambda x: x.mid).astype(float)
    observed_mids = sorted(df[("distance_bin_mid", "", "")].dropna().unique())
    mid_to_id = {m: i for i, m in enumerate(observed_mids)}
    df.loc[:, ("distance_bin_id", "", "")] = df[("distance_bin_mid", "", "")].map(mid_to_id).astype(int)

    # 4. Pre-loop constants
    theta_phases = df.spike_count.columns.get_level_values(1).unique().astype(float)
    distance_bin_mids = np.array(sorted(df.distance_bin_mid.unique()))
    common_trials = sorted(df.trial.unique().tolist())

    # 5. Leave-one-trial-out outer loop
    results = []
    for fold_i, test_trial in enumerate(common_trials):
        fold = f"fold_{fold_i}"
        if verbose:
            print(fold)
        train_trials = [t for t in common_trials if t != test_trial]
        train_df = df[df.trial.isin(train_trials)]
        test_df = df[df.trial == test_trial]
        inner_fold_df = (
            _make_inner_fold_df(train_trials, n_inner=reg_search_folds, seed=random_state) if C == "opt" else None
        )

        # --- PLACE (reuses pdd helpers) ---
        place_C = (
            pdd._get_opt_C(
                inner_fold_df,
                train_df,
                sqrt_spikes=sqrt_spikes,
                normalise_X=normalise_X,
                reg_search_folds=reg_search_folds,
            )
            if C == "opt"
            else C
        )
        Xtr_p = pdd._prepare_X(train_df, phase=None, sqrt=sqrt_spikes)
        scaler_p = StandardScaler().fit(Xtr_p) if normalise_X else None
        if scaler_p is not None:
            Xtr_p = scaler_p.transform(Xtr_p)
        place_decoder = LogisticRegression(C=place_C, max_iter=10_000, random_state=0, class_weight="balanced").fit(
            Xtr_p, train_df.maze_position.simple.values
        )
        Xte_mean_p = pdd._prepare_X(test_df, phase=None, sqrt=sqrt_spikes)
        if scaler_p is not None:
            Xte_mean_p = scaler_p.transform(Xte_mean_p)
        fold_accuracy = place_decoder.score(Xte_mean_p, test_df.maze_position.simple.values)
        fold_n_classes = len(place_decoder.classes_)

        # --- DISTANCE (reuses ddv2 helpers) ---
        dist_C = (
            ddv2._get_opt_C(
                train_df,
                inner_fold_df,
                output=output,
                sqrt_spikes=sqrt_spikes,
                normalise_X=normalise_X,
                reg_search_folds=reg_search_folds,
                distance_bin_mids=distance_bin_mids,
            )
            if C == "opt"
            else C
        )
        Xtr_d = ddv2._prepare_X(train_df, phase=None, sqrt=sqrt_spikes)
        scaler_d = StandardScaler().fit(Xtr_d) if normalise_X else None
        if scaler_d is not None:
            Xtr_d = scaler_d.transform(Xtr_d)
        dist_decoder = LogisticRegression(C=dist_C, max_iter=10_000, random_state=0, class_weight="balanced").fit(
            Xtr_d, train_df.distance_bin_id.values
        )
        Xte_mean_d = ddv2._prepare_X(test_df, phase=None, sqrt=sqrt_spikes)
        if scaler_d is not None:
            Xte_mean_d = scaler_d.transform(Xte_mean_d)
        # bin mids corresponding to the bin_ids actually seen in training (LOO may drop bins from train)
        trained_bin_mids = distance_bin_mids[dist_decoder.classes_]
        true_dist = distance_bin_mids[test_df.distance_bin_id.values]
        baseline_pred = ddv2._decode_distance(dist_decoder, Xte_mean_d, trained_bin_mids, output=output)
        fold_baseline_mae = float(np.mean(np.abs(baseline_pred - true_dist)))

        # --- Per-test-sample metadata (shared across phases) ---
        base = pd.DataFrame(
            {
                "time": test_df[("time", "", "")].to_numpy(),
                "trial": test_df[("trial", "", "")].to_numpy(),
                "trial_unique_ID": test_df[("trial_unique_ID", "", "")].to_numpy(),
                "place_direction": test_df[("place_direction", "", "")].to_numpy(),
                "distance_to_goal": test_df[("distance_to_goal", "geodesic", "")].to_numpy(),
                "distance_bin_mid": test_df[("distance_bin_mid", "", "")].to_numpy().astype(float),
                "speed": test_df[("speed", "", "")].to_numpy(),
            },
            index=test_df.index,
        )
        base["fold_accuracy"] = fold_accuracy
        base["fold_n_classes"] = fold_n_classes
        base["fold_baseline_mae"] = fold_baseline_mae

        # --- Per-phase test loop: compute BOTH errors on the same test rows ---
        for phase in theta_phases:
            X_p = pdd._prepare_X(test_df, phase=phase, sqrt=sqrt_spikes)
            if scaler_p is not None:
                X_p = scaler_p.transform(X_p)
            Yprob = place_decoder.predict_proba(X_p)
            place_errs = pdd._get_trajectory_error(Yprob, test_df, place_decoder.classes_)

            X_d = ddv2._prepare_X(test_df, phase=phase, sqrt=sqrt_spikes)
            if scaler_d is not None:
                X_d = scaler_d.transform(X_d)
            y_pred_dist = ddv2._decode_distance(dist_decoder, X_d, trained_bin_mids, output=output)
            dist_err = y_pred_dist - true_dist  # +ve = further than truth

            res = base.copy()
            res["theta_phase"] = float(phase)
            res["fold"] = fold
            res["signed_error_place"] = place_errs["signed_error"]
            res["signed_error_distance"] = dist_err
            results.append(res)

    return pd.concat(results, ignore_index=True)


def _make_inner_fold_df(train_trials, n_inner=4, seed=0):
    """Train-side-only fold_df for inner C-search CV (compatible with
    pdd._get_opt_C and ddv2._get_opt_C).

    Columns: ("train", 0), ..., ("train", n_inner-1). Rows: trial IDs
    (NaN-padded if not evenly divisible). Trials shuffled with `seed`.
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.choice(np.asarray(train_trials), size=len(train_trials), replace=False)
    trials_per_vfold = int(np.ceil(len(shuffled) / n_inner))
    pad = trials_per_vfold * n_inner - len(shuffled)
    padded = np.append(shuffled, np.array([np.nan] * pad, dtype=object))
    inner = pd.DataFrame(padded.reshape(trials_per_vfold, n_inner))
    inner.columns = pd.MultiIndex.from_product([["train"], inner.columns])
    return inner


# %% Cross-session runner


def get_theta_mod_double_decoding_df(verbose=True, C=1, save=False):
    """Run the matched-sample decoder across all subjects × mazes and concat results.

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

    summary_df = pd.concat(dfs, ignore_index=True)
    if save:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_parquet(save_path)
    return summary_df


# %% Plotting


def _filter_summary_df(summary_df, late_sessions=False, maze_names=None):
    """ """
    _df = summary_df.copy()
    if maze_names is not None:
        _df = _df[_df.maze_name.isin(maze_names)]
    if late_sessions:
        _df = _df[_df.late_session]
    return _df


def plot_double_decoding_bias(
    summary_df,
    late_sessions=False,
    maze_names=None,
    place_color="darkred",
    dist_color="darkblue",
    normalise=True,
    print_stats=True,
    axes=None,
):
    """Quick readout of the matched-sample double-decoding results.
    `normalise=True` divides each pipeline's per-subject bias df by the
    amplitude `A` of a sinusoid fit to its cross-subject mean curve, so both
    pipelines plot at the same vertical scale. Useful for visualising phase
    alignment when the raw bias magnitudes differ. Y-axis is unitless when
    normalised, cm otherwise.

    Sign convention: +ve bias = decoder predicts location further from goal (past).
    """
    df = _filter_summary_df(summary_df, late_sessions=late_sessions, maze_names=maze_names)

    # Split combined df into per-pipeline bias dfs (subjects × phases, in metres)
    biases_m = {}
    for kind, col in [("place", "signed_error_place"), ("distance", "signed_error_distance")]:
        b = df.groupby(["subject_ID", "theta_phase"])[col].mean().unstack(0).T
        biases_m[kind] = b.sub(b.mean(axis=1), axis=0)

    # Rescale each pipeline to unit amplitude (on cross-subject mean curve), else use cm.
    if normalise:
        plotted = {}
        for kind in ["place", "distance"]:
            phases = biases_m[kind].columns.values.astype(float)
            fit = tmu.fit_sinusoid(phases, biases_m[kind].mean().values, fit_constant=True, return_as="params")
            A = fit["A"]
            plotted[kind] = biases_m[kind] / A if A > 0 else biases_m[kind]
        ylabel = "decoding bias (normalised)"
    else:
        plotted = {kind: biases_m[kind] * 100 for kind in ["place", "distance"]}
        ylabel = "decoding bias (cm)"

    if axes is None:
        fig = plt.figure(figsize=(5, 2.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
        ax = fig.add_subplot(gs[0, 0])
        ax_polar = fig.add_subplot(gs[0, 1], projection="polar")
    else:
        ax, ax_polar = axes

    tmu.plot_decoding_bias(
        plotted["place"],
        color=place_color,
        label="place",
        ylabel=ylabel,
        print_stats=print_stats,
        ax=ax,
    )
    tmu.plot_decoding_bias(
        plotted["distance"],
        color=dist_color,
        label="distance",
        ylabel=ylabel,
        print_stats=print_stats,
        ax=ax,
    )
    dox.plot_phase_offset_polar(biases_m["place"], biases_m["distance"], ax=ax_polar)
