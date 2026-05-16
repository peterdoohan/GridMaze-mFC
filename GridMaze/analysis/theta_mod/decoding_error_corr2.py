"""
Single-sample correlation of place and distance-to-goal decoding errors (v2).

Same question as decoding_error_corr.py: are the place and distance decoders'
per-sample biases coordinated? Differs from v1 in three ways:
  - bias metric is the probability-weighted center-of-mass over the full
    ±envelope of past/future trajectory steps (mirrors
    trajectory_decoding2._get_trajectory_error), not the prev-vs-next triplet
    contrast.
  - distance label is steps_to_goal.future (integer node-to-adjacent-node steps
    along the trajectory), so place and distance bias share trajectory-step
    units and tower/bridge stratification is unnecessary.
  - regularisation strength is selected per outer LOO fold by a 5-split inner
    CV reg search, independent per decoder.

Place- and distance-tuned neurons are disjoint sets from the neGLM
variance-explained pipeline, so the two decoders are independent.
"""

# %% Imports
import json
import copy
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr, ttest_1samp

from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.place_direction import future_decoding as fd
from GridMaze.analysis.neGLM import load_model_sets as lms
from GridMaze.analysis.neGLM import variance_explained as ve

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "theta_mod" / "decoding_error_corr_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)


# %% plotting


def plot_decoding_error_corr(
    results_df,
    maze_names=("maze_1", "maze_2"),
    good_sessions_only=True,
    regress_out_displacement=False,
    regress_out_speed=False,
    regress_out_head_direction=False,
    min_neurons=None,
    min_confidence=None,
    corr_method="pearson",
    color="indigo",
    ax=None,
    ymax=0.05,
    print_stats=True,
):
    """Per-subject correlation between place and distance signed errors.

    Parameters:
      regress_out_displacement / regress_out_speed / regress_out_head_direction:
        if True, residualise each covariate out of both signed errors per subject
        before computing the correlation. Multiple flags are stacked into a
        single joint OLS fit. Head direction enters as (sin θ, cos θ) to respect
        its circularity.
      min_neurons: drop sessions with fewer tuned neurons than this (per decoder).
      min_confidence: drop samples whose max predict_proba (place AND distance)
        is below this threshold. Useful for filtering low-confidence predictions.
      corr_method: "pearson" or "spearman".
    """
    df = results_df.copy()
    if maze_names is not None:
        df = df[df.maze_name.isin(maze_names)]
    if good_sessions_only:
        df = df[df.good_session]
    if min_neurons is not None:
        df = df[df.n_place_neurons >= min_neurons]
        df = df[df.n_distance_neurons >= min_neurons]
    if min_confidence is not None and "max_p_loc" in df.columns:
        df = df[(df.max_p_loc >= min_confidence) & (df.max_p_dist >= min_confidence)]
    df = df[df.all_envelope_defined_place & df.all_envelope_defined_dist]
    df = df.dropna(subset=["signed_error_place", "signed_error_dist"])

    corr_fn = spearmanr if corr_method == "spearman" else pearsonr

    subject_corrs = []
    for subject_ID, sdf in df.groupby("subject_ID"):
        covariates = []
        if regress_out_displacement and "displacement" in sdf.columns:
            covariates.append(sdf.displacement.values)
        if regress_out_speed and "speed" in sdf.columns:
            covariates.append(sdf.speed.values)
        if regress_out_head_direction and "head_direction" in sdf.columns:
            hd_rad = np.deg2rad(sdf.head_direction.values)
            covariates.append(np.sin(hd_rad))
            covariates.append(np.cos(hd_rad))
        r_subj = _one_corr(sdf, corr_fn, covariates if covariates else None)
        if r_subj is None or not np.isfinite(r_subj):
            continue
        subject_corrs.append((subject_ID, r_subj))

    _, corrs = zip(*subject_corrs)
    corrs = np.array(corrs)
    print(corrs)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(0.5, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", ls="--", alpha=0.5)
    plot_df = pd.DataFrame({"x": np.zeros(len(corrs)), "corr": corrs})
    # individual subjects — drawn AFTER the pointplot so they paint on top
    sns.stripplot(
        data=plot_df,
        x="x",
        y="corr",
        color="grey",
        alpha=0.7,
        size=5,
        jitter=0.25,
        ax=ax,
    )
    # mean ± SEM
    sns.pointplot(
        data=plot_df,
        x="x",
        y="corr",
        errorbar="se",
        color=color,
        markersize=8,
        linestyle="none",
        zorder=10,
        capsize=0,
        ax=ax,
    )
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("correlation\n(place ↔ distance, signed)")
    ax.set_ylim(top=ymax)

    if print_stats:
        t, p = ttest_1samp(corrs, 0, alternative="greater")
        print(f"mean={corrs.mean():.3f}, sem={corrs.std()/np.sqrt(len(corrs)):.3f}, t={t:.3f}, p={p:.4f}")

    return ax


def _one_corr(sdf, corr_fn, covariates=None):
    """Compute one correlation (place vs distance signed errors) from a slice of
    the results df. Optionally residualises a list of covariates first. Returns
    None if fewer than 3 valid samples.
    """
    e_loc = sdf.signed_error_place.values
    e_dist = sdf.signed_error_dist.values
    if covariates:
        e_loc = _residualise(e_loc, covariates)
        e_dist = _residualise(e_dist, covariates)
        valid = np.isfinite(e_loc) & np.isfinite(e_dist)
        e_loc = e_loc[valid]
        e_dist = e_dist[valid]
    if len(e_loc) < 3:
        return None
    r = corr_fn(e_loc, e_dist)[0]
    return float(r) if np.isfinite(r) else None


def _residualise(y, x):
    """Regress one or more covariates out of y by OLS and return residuals.

    `x` can be a 1-D array (single covariate) or a list/tuple of 1-D arrays
    (multiple covariates, fit jointly). Samples where y or any covariate is NaN
    are excluded from the fit and returned as NaN in the residual.
    """
    y = np.asarray(y, dtype=float)
    if isinstance(x, (list, tuple)):
        X = np.column_stack([np.asarray(xi, dtype=float) for xi in x])
    else:
        X_arr = np.asarray(x, dtype=float)
        X = X_arr.reshape(-1, 1) if X_arr.ndim == 1 else X_arr
    if X.shape[1] == 0:
        return y.copy()
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if valid.sum() < X.shape[1] + 1:
        return y.copy()
    X_aug = np.column_stack([np.ones(valid.sum()), X[valid]])
    coefs, *_ = np.linalg.lstsq(X_aug, y[valid], rcond=None)
    pred = np.full_like(y, np.nan)
    pred[valid] = coefs[0] + X[valid] @ coefs[1:]
    return y - pred


# %% experiment-level populate + cache-load


def get_decoding_error_corr_df(
    maze_names=("maze_1", "maze_2", "rooms_maze"),
    days_on_maze="late",
    n_jobs=-1,
    save=False,
    save_label=None,
    verbose=True,
    **session_kwargs,
):
    """Populate (or load cached) per-sample results across all subjects.

    `save_label` (str, optional): when provided, results save under
    `RESULTS_DIR/tests/<save_label>.parquet` instead of the main file. Useful
    for parameter sweeps.

    `**session_kwargs`: forwarded to `get_session_decoding_error_corr_df` for
    per-session parameter overrides (envelope, mincount, sum_spike_window,
    multi_bin_K, etc.).
    """
    if save_label is not None:
        tests_dir = RESULTS_DIR / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        save_path = tests_dir / f"{save_label}.parquet"
    else:
        save_path = RESULTS_DIR / "decoding_error_corr_df.parquet"
    if save_path.exists() and not save:
        if verbose:
            print(f"Loading cached results from {save_path} ...")
        return pd.read_parquet(save_path)

    distance_tuned, place_tuned = get_tuned_neurons()

    results_dfs = []
    for subject_ID in SUBJECT_IDS:
        for maze_name in maze_names:
            if verbose:
                print(f"Loading sessions for {subject_ID} - {maze_name} ...")
            sessions = gs.get_maze_sessions(
                subject_IDs=[subject_ID],
                maze_names=[maze_name],
                days_on_maze=days_on_maze,
                with_data=[
                    "navigation_df",
                    "cluster_metrics",
                    "trials_df",
                    "navigation_spike_counts_df",
                ],
                must_have_data=True,
            )
            if n_jobs and n_jobs != 1:
                dfs = Parallel(n_jobs=n_jobs)(
                    delayed(get_session_decoding_error_corr_df)(
                        s,
                        distance_tuned=distance_tuned,
                        place_tuned=place_tuned,
                        verbose=False,
                        **session_kwargs,
                    )
                    for s in sessions
                )
            else:
                dfs = [
                    get_session_decoding_error_corr_df(
                        s,
                        distance_tuned=distance_tuned,
                        place_tuned=place_tuned,
                        verbose=verbose,
                        **session_kwargs,
                    )
                    for s in sessions
                ]
            results_dfs.extend([d for d in dfs if d is not None and len(d) > 0])

    if not results_dfs:
        return pd.DataFrame()
    results_df = pd.concat(results_dfs, axis=0, ignore_index=True)
    if save:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(save_path)
        if verbose:
            print(f"Saved results to {save_path}.")
    return results_df


# %% session-level


def get_session_decoding_error_corr_df(
    session,
    distance_tuned=None,
    place_tuned=None,
    envelope=1,
    resolution=0.2,
    max_steps_to_goal=16,
    min_rate=0.5,
    min_steps=1,
    mincount=15,
    min_neurons=5,
    min_trial_samples=2,
    n_inner_splits=4,
    C_range=None,
    C=1e-1,
    normalise_X=True,
    include_center=True,
    multi_bin_K=0,
    good_session_chance_mult=2.0,
    verbose=True,
):
    """LOO per-trial CV. Fits place + distance decoders on independent
    tuned-neuron subsets and reads out per-sample envelope-COM bias for each.
    Returns a long-form per-sample DataFrame, or None if the session has no
    usable data.

    `C`: regularisation strength for both decoders. If "opt", a 5-split inner-CV
    reg search runs per outer LOO fold (independent per decoder). Otherwise
    treated as a fixed float applied to all folds and both decoders — skips the
    reg search entirely.
    """
    if distance_tuned is None or place_tuned is None:
        distance_tuned, place_tuned = get_tuned_neurons()
    if C_range is None:
        C_range = np.logspace(-2, 2, 10)

    if verbose:
        print(f"{session.name}: loading input data ...")
    input_df = get_input_data(
        session,
        resolution=resolution,
        envelope=envelope,
        max_steps_to_goal=max_steps_to_goal,
    )
    if len(input_df) == 0:
        return None

    # core sample arrays (2-level columns after droplevel(2) in get_input_data)
    locs = input_df.maze_position.simple.to_numpy()
    dists_f = input_df.steps_to_goal.future.astype(float).to_numpy()
    trial = input_df[("trial", "")].to_numpy().astype(int)
    next_dist = input_df[("future_dist", 1)].astype(float).to_numpy()
    dirs = np.sign(dists_f - next_dist)
    displacement = input_df[("displacement", "")].astype(float).to_numpy()
    speed = (
        input_df[("speed", "")].astype(float).to_numpy()
        if ("speed", "") in input_df.columns
        else np.full(len(input_df), np.nan)
    )
    head_direction = (
        input_df[("head_direction", "")].astype(float).to_numpy()
        if ("head_direction", "") in input_df.columns
        else np.full(len(input_df), np.nan)
    )

    spikes_full = input_df.spike_count.to_numpy()
    units = np.array(input_df.spike_count.columns)
    keep_units = np.where(spikes_full.mean(0) / resolution > min_rate)[0]
    spikes = np.sqrt(spikes_full[:, keep_units])
    units = units[keep_units]

    loc_tuned_mask = np.array([u in place_tuned for u in units])
    dist_tuned_mask = np.array([u in distance_tuned for u in units])
    n_place_neurons = int(loc_tuned_mask.sum())
    n_dist_neurons = int(dist_tuned_mask.sum())
    if n_place_neurons < min_neurons or n_dist_neurons < min_neurons:
        if verbose:
            print(
                f"  skipping: {n_place_neurons} place-tuned, "
                f"{n_dist_neurons} distance-tuned units "
                f"(min_neurons={min_neurons})"
            )
        return None

    # optional multi-time-bin concat: replaces `spikes` with a (n, n_features*(2K+1))
    # matrix and tiles the tuned masks across offsets. multi_bin_valid marks samples
    # where every offset stayed within the same trial.
    if multi_bin_K and multi_bin_K > 0:
        spikes, multi_bin_valid = _multi_bin_concat(spikes, trial, int(multi_bin_K))
        loc_tuned_mask = np.tile(loc_tuned_mask, 2 * int(multi_bin_K) + 1)
        dist_tuned_mask = np.tile(dist_tuned_mask, 2 * int(multi_bin_K) + 1)
    else:
        multi_bin_valid = np.ones(len(input_df), dtype=bool)

    # per-sample inclusion masks
    valid_dist = ~np.isnan(dists_f)
    dists_int = np.where(valid_dist, dists_f, 0).astype(int)
    cond_steps = valid_dist & (dists_int >= min_steps) & (dists_int <= max_steps_to_goal) & multi_bin_valid

    unique_locs, loc_counts = np.unique(locs[cond_steps], return_counts=True)
    keep_locs = unique_locs[loc_counts > mincount]
    cond_loc_class = np.isin(locs, keep_locs)

    unique_dists, dist_counts = np.unique(dists_int[cond_steps], return_counts=True)
    keep_dists = unique_dists[dist_counts > mincount]
    cond_dist_class = np.isin(dists_int, keep_dists)

    place_train_pool = cond_steps & cond_loc_class
    dist_train_pool = cond_steps & cond_dist_class
    test_pool = cond_steps & cond_loc_class & cond_dist_class

    if place_train_pool.sum() == 0 or dist_train_pool.sum() == 0 or test_pool.sum() == 0:
        return None

    place_chance = 1.0 / len(np.unique(locs[place_train_pool]))
    dist_chance = 1.0 / len(np.unique(dists_int[dist_train_pool]))

    # LOO outer trial loop
    test_trial_ids, tcounts = np.unique(trial[test_pool], return_counts=True)
    test_trials = test_trial_ids[tcounts >= min_trial_samples]

    accs_loc, accs_dist = [], []
    sample_records = []
    X_loc_all = spikes[:, loc_tuned_mask]
    X_dist_all = spikes[:, dist_tuned_mask]

    for test_trial in test_trials:
        train_mask_loc = (trial != test_trial) & place_train_pool
        train_mask_dist = (trial != test_trial) & dist_train_pool
        test_mask = (trial == test_trial) & test_pool
        if train_mask_loc.sum() == 0 or train_mask_dist.sum() == 0 or test_mask.sum() == 0:
            continue
        if len(np.unique(locs[train_mask_loc])) < 2 or len(np.unique(dists_int[train_mask_dist])) < 2:
            continue

        # regularisation: nested-CV search per fold, or fixed scalar
        if C == "opt":
            C_loc = _get_opt_C_loo(
                X_loc_all[train_mask_loc],
                locs[train_mask_loc],
                trial[train_mask_loc],
                n_inner_splits=n_inner_splits,
                C_range=C_range,
                normalise_X=normalise_X,
            )
            C_dist = _get_opt_C_loo(
                X_dist_all[train_mask_dist],
                dists_int[train_mask_dist],
                trial[train_mask_dist],
                n_inner_splits=n_inner_splits,
                C_range=C_range,
                normalise_X=normalise_X,
            )
        else:
            C_loc = float(C)
            C_dist = float(C)

        # outer-fold scaler fit on outer-training samples per decoder
        scaler_loc = StandardScaler().fit(X_loc_all[train_mask_loc]) if normalise_X else None
        scaler_dist = StandardScaler().fit(X_dist_all[train_mask_dist]) if normalise_X else None
        X_loc_train = scaler_loc.transform(X_loc_all[train_mask_loc]) if normalise_X else X_loc_all[train_mask_loc]
        X_dist_train = (
            scaler_dist.transform(X_dist_all[train_mask_dist]) if normalise_X else X_dist_all[train_mask_dist]
        )

        clf_loc = LogisticRegression(C=C_loc, class_weight="balanced", max_iter=10_000, random_state=0).fit(
            X_loc_train, locs[train_mask_loc]
        )
        clf_dist = LogisticRegression(C=C_dist, class_weight="balanced", max_iter=10_000, random_state=0).fit(
            X_dist_train, dists_int[train_mask_dist]
        )

        # outer-fold accuracies (held-out trial, training-pool samples only)
        outer_loc_test = (trial == test_trial) & place_train_pool
        outer_dist_test = (trial == test_trial) & dist_train_pool
        if outer_loc_test.sum() > 0:
            Xo_loc = scaler_loc.transform(X_loc_all[outer_loc_test]) if normalise_X else X_loc_all[outer_loc_test]
            accs_loc.append(clf_loc.score(Xo_loc, locs[outer_loc_test]))
        if outer_dist_test.sum() > 0:
            Xo_dist = scaler_dist.transform(X_dist_all[outer_dist_test]) if normalise_X else X_dist_all[outer_dist_test]
            accs_dist.append(clf_dist.score(Xo_dist, dists_int[outer_dist_test]))

        # per-sample envelope COM on the test pool
        test_idx = np.where(test_mask)[0]
        test_df = input_df.iloc[test_idx]
        X_loc_test = scaler_loc.transform(X_loc_all[test_idx]) if normalise_X else X_loc_all[test_idx]
        X_dist_test = scaler_dist.transform(X_dist_all[test_idx]) if normalise_X else X_dist_all[test_idx]
        Yprob_loc = clf_loc.predict_proba(X_loc_test)
        Yprob_dist = clf_dist.predict_proba(X_dist_test)

        place_err = _get_trajectory_error_place(
            Yprob_loc, test_df, clf_loc.classes_, envelope, include_center=include_center
        )
        dist_err = _get_trajectory_error_dist(
            Yprob_dist, test_df, clf_dist.classes_, envelope, include_center=include_center
        )

        sample_records.append(
            pd.DataFrame(
                {
                    "trial": trial[test_idx].astype(int),
                    "loc": locs[test_idx],
                    "dist_cur": dists_int[test_idx],
                    "dirs": dirs[test_idx],
                    "displacement": displacement[test_idx],
                    "speed": speed[test_idx],
                    "head_direction": head_direction[test_idx],
                    "signed_error_place": place_err["signed_error_place"],
                    "all_envelope_defined_place": place_err["all_envelope_defined_place"],
                    "signed_error_dist": dist_err["signed_error_dist"],
                    "all_envelope_defined_dist": dist_err["all_envelope_defined_dist"],
                    "max_p_loc": Yprob_loc.max(axis=1),
                    "max_p_dist": Yprob_dist.max(axis=1),
                    "C_loc": C_loc,
                    "C_dist": C_dist,
                }
            )
        )

    if not sample_records:
        return None

    out = pd.concat(sample_records, axis=0, ignore_index=True)
    place_acc_mean = float(np.mean(accs_loc)) if accs_loc else np.nan
    dist_acc_mean = float(np.mean(accs_dist)) if accs_dist else np.nan
    out["place_acc_mean"] = place_acc_mean
    out["place_chance"] = place_chance
    out["dist_acc_mean"] = dist_acc_mean
    out["dist_chance"] = dist_chance
    out["n_place_neurons"] = n_place_neurons
    out["n_distance_neurons"] = n_dist_neurons
    out["good_session"] = place_acc_mean > good_session_chance_mult * place_chance
    out["subject_ID"] = session.subject_ID
    out["maze_name"] = session.maze_name
    out["day_on_maze"] = session.day_on_maze
    out["session_name"] = session.name
    return out


# %% input data


def get_input_data(
    session,
    resolution=0.2,
    envelope=2,
    max_steps_to_goal=24,
):
    """Downsampled nav + spike df with place AND distance ±envelope columns.

    Returned df has 2-level columns (the third "" placeholder is dropped at the
    end so the helpers can do `input_df.spike_count`, `test_df["past"][i]`, etc).
    """
    navigation_df = session.navigation_df.copy()
    spike_counts_df = session.navigation_spike_counts_df.copy()
    spike_counts_df.reset_index(inplace=True, drop=True)
    spike_counts_df.columns = pd.MultiIndex.from_tuples([(*c, "") for c in spike_counts_df.columns])

    keep_clusters = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=True,
    )
    spike_counts_df = spike_counts_df[
        spike_counts_df.columns[spike_counts_df.columns.get_level_values(1).isin(keep_clusters)]
    ]

    ds_nav_df, ds_spike_counts_df = ds.downsample_nav_spikes_data(
        navigation_df,
        spike_counts_df,
        resolution=resolution,
        distance_metrics=[("steps_to_goal", "future"), ("distance_to_goal", "geodesic")],
    )
    ds_nav_df.columns = pd.MultiIndex.from_tuples([(*c, "") for c in ds_nav_df.columns])
    input_df = pd.concat([ds_nav_df, ds_spike_counts_df], axis=1).reset_index(drop=True)

    # signed displacement (m) from discrete tower/bridge centre along direction of motion.
    # Computed on raw 60 Hz nav_df (centroid_position doesn't pass through downsample),
    # then bin-averaged to the analysis resolution.
    raw_disp = _get_aligned_displacement(navigation_df, session.simple_maze())
    input_df[("displacement", "", "")] = _downsample_displacement(raw_disp, resolution, len(input_df))

    # per-bin circular mean of head_direction (deg). Centroid-position-style:
    # computed from raw nav_df (head_direction.value isn't kept by downsample).
    if ("head_direction", "value") in navigation_df.columns:
        raw_hd = navigation_df[("head_direction", "value")].to_numpy().astype(float)
        input_df[("head_direction", "", "")] = _downsample_circular(raw_hd, resolution, len(input_df))
    else:
        input_df[("head_direction", "", "")] = np.full(len(input_df), np.nan)

    # place envelope (fd.get_past_and_future_states is place-only)
    place_env_df = fd.get_past_and_future_states(
        input_df, state_type="place", past_offset=envelope, future_offset=envelope
    )
    place_env_df.columns = pd.MultiIndex.from_tuples([(*col, "") for col in place_env_df.columns])
    input_df = pd.concat([input_df, place_env_df], axis=1).sort_index(axis=1)

    # distance envelope (built by mirroring fd's state-transition algorithm)
    dist_env_df = _get_distance_envelope(input_df, envelope=envelope)
    input_df = pd.concat([input_df, dist_env_df], axis=1).sort_index(axis=1)

    input_df = filt.filter_navigation_rates_df(
        input_df,
        navigation_only=True,
        moving_only=True,
        exclude_time_at_goal=True,
        max_steps_to_goal=max_steps_to_goal,
    )
    input_df = input_df.droplevel(2, axis=1)
    return input_df.reset_index(drop=True)


def _get_distance_envelope(input_df, envelope=2):
    """For each sample, return the steps_to_goal.future value at the first frame
    of the trajectory-state run that is i runs ahead (future) / behind (past).

    Mirrors the state-transition boundary logic of `fd.get_past_and_future_states`
    so the distance envelope and the place envelope are step-aligned (i.e. the
    label at offset +i for both decoders refers to the same trajectory step).
    """
    states = input_df[("maze_position", "simple", "")].to_numpy().copy()
    dist = input_df[("steps_to_goal", "future", "")].to_numpy().astype(float)
    phases = input_df[("trial_phase", "", "")].to_numpy()
    trials = input_df[("trial", "", "")].to_numpy()
    goals = input_df[("goal", "", "")].to_numpy()
    simples = input_df[("maze_position", "simple", "")].to_numpy()

    states_masked = states.astype(object)
    states_masked[phases != "navigation"] = None
    for tr in pd.unique(trials):
        if pd.isna(tr):
            continue
        trial_inds = np.where(trials == tr)[0]
        if len(trial_inds) == 0:
            continue
        goal_val = goals[trial_inds[0]]
        goal_inds = trial_inds[simples[trial_inds] == goal_val]
        states_masked[goal_inds] = None

    boundaries = np.concatenate([np.zeros(1, dtype=int), np.where(states_masked[1:] != states_masked[:-1])[0] + 1])

    n = len(states_masked)
    state_future = np.array(
        [copy.deepcopy(states_masked)] + [np.array([None] * n, dtype=object) for _ in range(envelope)]
    )
    state_past = np.array(
        [copy.deepcopy(states_masked)] + [np.array([None] * n, dtype=object) for _ in range(envelope)]
    )
    dist_future = np.full((envelope + 1, n), np.nan, dtype=float)
    dist_past = np.full((envelope + 1, n), np.nan, dtype=float)
    dist_future[0] = dist
    dist_past[0] = dist

    def _populate(state_arr, dist_arr, sign):
        dir_b = boundaries if sign == 1 else np.flip(boundaries) - 1
        for offset in range(1, envelope + 1):
            ref_state = state_arr[offset - 1]
            ref_dist = dist_arr[offset - 1]
            nxt_state = state_arr[offset]
            nxt_dist = dist_arr[offset]
            for i_b in range(len(dir_b) - 1):
                inds = np.arange(dir_b[i_b], dir_b[i_b + 1], sign)
                if len(inds) == 0:
                    continue
                cur_state = ref_state[inds[0]]
                nxt_idx = inds[-1] + sign
                if nxt_idx < 0 or nxt_idx >= n:
                    continue
                next_state = ref_state[nxt_idx]
                if cur_state is None or next_state is None:
                    continue
                nxt_state[inds] = next_state
                nxt_dist[inds] = ref_dist[nxt_idx]

    _populate(state_future, dist_future, +1)
    _populate(state_past, dist_past, -1)

    cols = {}
    for i in range(1, envelope + 1):
        cols[("past_dist", i, "")] = dist_past[i]
        cols[("future_dist", i, "")] = dist_future[i]
    out = pd.DataFrame(cols, index=input_df.index)
    out.columns = pd.MultiIndex.from_tuples(list(out.columns))
    return out


def _build_label_to_position(simple_maze):
    """Combine node + edge label→cartesian-position dicts from a simple_maze object.

    `get_maze_label2position` in `GridMaze.maze.representations` only covers nodes;
    we also need edge (bridge) centres here, so build the combined map locally.
    """
    node_coord2label = nx.get_node_attributes(simple_maze, "label")
    node_coord2pos = nx.get_node_attributes(simple_maze, "position")
    edge_coord2label = nx.get_edge_attributes(simple_maze, "label")
    edge_coord2pos = nx.get_edge_attributes(simple_maze, "position")
    label2pos = {}
    for coord, label in node_coord2label.items():
        label2pos[label] = node_coord2pos[coord]
    for coord, label in edge_coord2label.items():
        label2pos[label] = edge_coord2pos[coord]
    return label2pos


def _get_aligned_displacement(navigation_df, simple_maze):
    """Per-raw-frame signed displacement (m) of the animal's centroid from the
    discrete tower/bridge centre, projected onto the cardinal direction of motion.

    Operates on the raw 60 Hz navigation_df (centroid_position doesn't survive
    `ds.downsample_nav_spikes_data`). The caller bin-averages this to the
    analysis resolution.

    +ve = animal is ahead of the discrete centre along its direction of travel;
    −ve = behind. Returns NaN for samples with an unknown maze label, a missing
    centroid, or a non-NSEW movement direction (e.g. stationary).
    """
    label2pos = _build_label_to_position(simple_maze)
    labels = navigation_df[("maze_position", "simple")].to_numpy()
    centroid_x = navigation_df[("centroid_position", "x")].to_numpy().astype(float)
    centroid_y = navigation_df[("centroid_position", "y")].to_numpy().astype(float)
    dirs = navigation_df[("cardinal_movement_direction", "")].to_numpy()

    n = len(navigation_df)
    disc_x = np.full(n, np.nan)
    disc_y = np.full(n, np.nan)
    for i, lab in enumerate(labels):
        pos = label2pos.get(lab)
        if pos is not None:
            disc_x[i] = pos[0]
            disc_y[i] = pos[1]

    dx = centroid_x - disc_x
    dy = centroid_y - disc_y

    displacement = np.full(n, np.nan)
    east = dirs == "E"
    west = dirs == "W"
    north = dirs == "N"
    south = dirs == "S"
    displacement[east] = dx[east]
    displacement[west] = -dx[west]
    displacement[north] = dy[north]
    displacement[south] = -dy[south]
    return displacement


def _downsample_displacement(raw_disp, resolution, n_target, frame_rate=60):
    """Bin-mean raw-frame displacement to the analysis resolution, matching the
    `index // ds_frames` window scheme used in `downsample_nav_spikes_data`.
    """
    ds_frames = int(frame_rate * resolution)
    window_groups = np.arange(len(raw_disp)) // ds_frames
    ds_disp = pd.Series(raw_disp).groupby(window_groups).mean().to_numpy()
    if len(ds_disp) > n_target:
        ds_disp = ds_disp[:n_target]
    elif len(ds_disp) < n_target:
        ds_disp = np.concatenate([ds_disp, np.full(n_target - len(ds_disp), np.nan)])
    return ds_disp


def _downsample_circular(values_deg, resolution, n_target, frame_rate=60):
    """Per-bin circular mean of degrees, matching `_polar_downsample` in
    `analysis/core/downsample.py`. Returns a (n_target,) array of degrees in [0, 360).
    """
    ds_frames = int(frame_rate * resolution)
    window_groups = np.arange(len(values_deg)) // ds_frames
    rad = np.deg2rad(values_deg)
    sin_mean = pd.Series(np.sin(rad)).groupby(window_groups).mean().to_numpy()
    cos_mean = pd.Series(np.cos(rad)).groupby(window_groups).mean().to_numpy()
    ds_deg = np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360
    if len(ds_deg) > n_target:
        ds_deg = ds_deg[:n_target]
    elif len(ds_deg) < n_target:
        ds_deg = np.concatenate([ds_deg, np.full(n_target - len(ds_deg), np.nan)])
    return ds_deg


def _multi_bin_concat(spikes, trial, K):
    """Concatenate spike vectors at trajectory offsets [-K, …, 0, …, +K] into a
    wider feature matrix. Offsets are taken within the same trial; samples whose
    full ±K context spans a trial boundary are returned NaN and marked invalid.

    Returns:
      X_concat (n_samples, n_features * (2K+1))
      valid_mask (n_samples,) — True iff every offset stayed within the same trial.
    """
    n_samples, n_features = spikes.shape
    blocks = []
    valid = np.ones(n_samples, dtype=bool)
    for k in range(-K, K + 1):
        if k == 0:
            blocks.append(spikes.astype(float))
            continue
        shifted = np.full((n_samples, n_features), np.nan, dtype=float)
        src_idx = np.arange(n_samples) + k
        in_range = (src_idx >= 0) & (src_idx < n_samples)
        same_trial = np.zeros(n_samples, dtype=bool)
        if in_range.any():
            sub = src_idx[in_range]
            same_trial[in_range] = trial[sub] == trial[in_range]
        fill_idx = np.where(same_trial)[0]
        if fill_idx.size:
            shifted[fill_idx] = spikes[src_idx[fill_idx]]
        blocks.append(shifted)
        valid &= same_trial
    X_concat = np.hstack(blocks)
    # rows that aren't valid get NaN-filled; we mark them so the caller can exclude them.
    return X_concat, valid


# %% tuned neuron selection (duplicated from v1 so this file stands alone)


def get_tuned_neurons():
    """Cluster IDs of neurons selectively tuned (via neGLM variance-explained)
    to distance-to-goal or place-direction but not both.
    """
    feature_tuned_df = ve.get_feature_tuned_df(
        lms.load_model_set_cv_scores("variance_explained_multiunit"),
        reduced_models=["remove_distance_to_goal", "remove_place_direction"],
    )
    distance_tuned = (
        feature_tuned_df[feature_tuned_df.distance_to_goal & ~feature_tuned_df.place_direction]
        .index.get_level_values(1)
        .values
    )
    place_tuned = (
        feature_tuned_df[~feature_tuned_df.distance_to_goal & feature_tuned_df.place_direction]
        .index.get_level_values(1)
        .values
    )
    return distance_tuned, place_tuned


# %% decoder helpers


def _get_opt_C_loo(X, y, trials, n_inner_splits=5, C_range=None, normalise_X=True, random_state=0):
    """5-split inner-CV reg search. Randomly partitions the outer-training trials
    into n_inner_splits chunks; each chunk is the inner-validation set in turn,
    the remaining trials are the inner-training set. Returns the C with the
    highest mean inner-validation accuracy across the splits.

    If normalise_X, fits a StandardScaler on each inner-training fold and
    applies it to both inner-train and inner-val (per-fold scaling to avoid
    leakage).
    """
    if C_range is None:
        C_range = np.logspace(-2, 2, 10)
    unique_trials = np.unique(trials)
    if len(unique_trials) < 2:
        return float(np.median(C_range))
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(unique_trials)
    val_chunks = np.array_split(perm, min(n_inner_splits, len(perm)))

    accs = np.full((len(val_chunks), len(C_range)), np.nan)
    for i, val_trials in enumerate(val_chunks):
        if len(val_trials) == 0:
            continue
        val_mask = np.isin(trials, val_trials)
        train_mask = ~val_mask
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        if len(np.unique(y[train_mask])) < 2:
            continue
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        if normalise_X:
            scaler = StandardScaler().fit(X_tr)
            X_tr = scaler.transform(X_tr)
            X_val = scaler.transform(X_val)
        for j, C in enumerate(C_range):
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10_000, random_state=0).fit(X_tr, y_tr)
            accs[i, j] = clf.score(X_val, y_val)
    if np.all(np.isnan(accs)):
        return float(np.median(C_range))
    return float(C_range[np.nanargmax(np.nanmean(accs, axis=0))])


def _get_trajectory_error_place(Yprob, test_df, decoder_classes, envelope, include_center=True):
    """Envelope-COM signed bias for the place decoder, in trajectory-step units.

    +ve = decoder reads out a position further along the future trajectory.

    If `include_center` is True (default), the current location (coord 0) is
    included in both the envelope-probability sum (denominator) and the COM —
    its zero coord doesn't shift the bias numerator but a confident p_center
    shrinks |signed_error|. If False, the centre is excluded from both, so the
    bias is computed only over the 2k off-centre positions.
    """
    k = int(envelope)
    past = test_df["past"]
    future = test_df["future"]
    past_labels = np.stack([past[i].to_numpy() for i in range(k, 0, -1)], axis=1)
    future_labels = np.stack([future[i].to_numpy() for i in range(1, k + 1)], axis=1)
    if include_center:
        center = test_df[("maze_position", "simple")].to_numpy().reshape(-1, 1)
        envelope_labels = np.concatenate([past_labels, center, future_labels], axis=1)
        step_coords = np.arange(-k, k + 1)
    else:
        envelope_labels = np.concatenate([past_labels, future_labels], axis=1)
        step_coords = np.concatenate([np.arange(-k, 0), np.arange(1, k + 1)])

    col_idx = np.full(envelope_labels.shape, -1, dtype=np.int64)
    for label, j in {c: j for j, c in enumerate(decoder_classes)}.items():
        col_idx[envelope_labels == label] = j
    all_envelope_defined = (col_idx >= 0).all(axis=1)

    n_samples = Yprob.shape[0]
    Yprob_ext = np.hstack([Yprob, np.zeros((n_samples, 1))])
    envelope_probs = Yprob_ext[np.arange(n_samples)[:, None], col_idx]

    envelope_mass = envelope_probs.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = envelope_probs / envelope_mass[:, None]
    signed_error = (norm * step_coords).sum(axis=1)

    return {
        "signed_error_place": signed_error,
        "all_envelope_defined_place": all_envelope_defined,
    }


def _get_trajectory_error_dist(Yprob, test_df, decoder_classes, envelope, include_center=True):
    """Envelope-COM signed bias for the distance decoder, in trajectory-step units.

    +ve = decoder reads out the steps_to_goal value at the position one
    trajectory-step ahead of true. Same sign convention as the place decoder.

    See `_get_trajectory_error_place` for the `include_center` semantics.
    """
    k = int(envelope)
    past = test_df["past_dist"]
    future = test_df["future_dist"]
    past_labels = np.stack([past[i].to_numpy() for i in range(k, 0, -1)], axis=1)
    future_labels = np.stack([future[i].to_numpy() for i in range(1, k + 1)], axis=1)
    if include_center:
        center = test_df[("steps_to_goal", "future")].to_numpy().reshape(-1, 1)
        env_floats = np.concatenate([past_labels, center, future_labels], axis=1).astype(float)
        step_coords = np.arange(-k, k + 1)
    else:
        env_floats = np.concatenate([past_labels, future_labels], axis=1).astype(float)
        step_coords = np.concatenate([np.arange(-k, 0), np.arange(1, k + 1)])

    valid = ~np.isnan(env_floats)
    SENTINEL = np.iinfo(np.int64).min
    int_labels = np.where(valid, env_floats, SENTINEL).astype(np.int64)

    col_idx = np.full(int_labels.shape, -1, dtype=np.int64)
    for label, j in {int(c): j for j, c in enumerate(decoder_classes)}.items():
        col_idx[int_labels == label] = j
    col_idx[~valid] = -1
    all_envelope_defined = (col_idx >= 0).all(axis=1)

    n_samples = Yprob.shape[0]
    Yprob_ext = np.hstack([Yprob, np.zeros((n_samples, 1))])
    envelope_probs = Yprob_ext[np.arange(n_samples)[:, None], col_idx]

    envelope_mass = envelope_probs.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = envelope_probs / envelope_mass[:, None]
    signed_error = (norm * step_coords).sum(axis=1)

    return {
        "signed_error_dist": signed_error,
        "all_envelope_defined_dist": all_envelope_defined,
    }
