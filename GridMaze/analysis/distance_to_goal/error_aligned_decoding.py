"""
Look at decoding performance of distance-to-goal aligned to navigational errors
Does the internal representation move with subject's internal estimate of distance even
when it is wrong?
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from GridMaze.analysis.core import convert
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import folds
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.distance_to_goal import distributions as dd

# %% Global Variables

# %% Functions


# %% Imports


# %% Global Variables

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "distance_to_goal" / "logreg_decoding"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)


# %% plot decoding aligned to errors


def test(results_df, error_type="nav_error", window=(-2, 2), resolution=0.2):
    """ """
    df = results_df.copy()
    df.reset_index(inplace=True)
    df[("decoded_distance", "")] = results_df.decoded_distance_prob.idxmax(
        axis=1
    )  # could map this to fancier weighted decoded distance later
    df[("decoding_delta", "")] = results_df.distance_bin_mid - df.decoded_distance
    # get decoding delta aliged each error
    rows_before = int(-window[0] / resolution)
    rows_after = int(window[1] / resolution)
    expected_length = rows_before + rows_after + 1
    error_idxs = df[df[error_type]].index
    aligned_deltas = []
    for i in error_idxs:
        try:
            start_idx = i - rows_before
            end_idx = i + rows_after
            aligned_delta = df.loc[start_idx:end_idx, ("decoding_delta", "")].values.astype(float)
        except IndexError:
            continue
        if aligned_delta.shape[0] != expected_length:
            continue
        aligned_deltas.append(aligned_delta)
    # organise into dataframe
    aligned_times = np.arange(window[0], window[1] + resolution, resolution).round(2)
    aligned_df = pd.DataFrame(data=np.stack(aligned_deltas), columns=aligned_times)
    return aligned_df


# %% experiment level decoding (run over all sessions)


def get_distance_to_goal_decoding_df(sessions=None, resolution=0.2, verbose=True, save=False, n_jobs=-1):
    """ """
    save_path = RESULTS_DIR / "errors" / f"distance_to_goal_decoding_df_res{resolution}.parquet"
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
            delayed(decode_session_distance_to_goal)(session, resolution=resolution, verbose=verbose)
            for session in sessions
        )
    else:
        results_dfs = [
            decode_session_distance_to_goal(session, resolution=resolution, verbose=verbose) for session in sessions
        ]
    distance_to_goal_decoding_df = pd.concat(results_dfs, ignore_index=True)

    if save:
        distance_to_goal_decoding_df.to_parquet(save_path)
        if verbose:
            print(f"Saved decoding df to {save_path}")
    return distance_to_goal_decoding_df


# %% session level decoding


def decode_session_distance_to_goal(
    session,
    resolution=0.2,
    metric=("distance_to_goal", "geodesic"),
    include_multiunits=True,
    moving_only=False,
    max_steps_to_goal=30,
    bin_spacing=0.05,
    max_distance=None,
    bin_method="uniform",
    n_log_bins=30,
    balance_distances=False,
    n_folds=5,
    sqrt_spikes=True,
    standardise_spikes=True,
    alpha="opt",
    verbose=False,
):
    """ """
    if verbose:
        print(session.name)
    # get input data
    input_data = get_input_data(
        session,
        resolution,
        metric,
        include_multiunits,
        moving_only,
        max_steps_to_goal,
        bin_spacing,
        bin_method,
        max_distance,
        n_log_bins,
        balance_distances,
    )
    distance_bin_mids = sorted(input_data.distance_bin_mid.unique())
    # set up output df
    results_df = pd.concat(
        [
            input_data,
            pd.DataFrame(
                columns=pd.MultiIndex.from_product([["decoded_distance_prob"], distance_bin_mids]),
                index=input_data.index,
            ),
        ],
        axis=1,
    )
    # decode distance CV
    folds_df = folds.get_folds_df(session, goal_stratified=False, return_unique_IDs=True, n_folds=n_folds)
    _folds = folds_df.columns.get_level_values(0).unique()
    for fold in _folds:
        if verbose:
            print(fold)
        fold_df = folds_df[fold]
        # get optimal alpha, CV over training folds
        if alpha == "opt":
            opt_alpha = get_CV_alpha(
                input_data,
                fold_df,
                metric,
                sqrt_spikes=sqrt_spikes,
                standardise_spikes=standardise_spikes,
                return_as="best",
                verbose=verbose,
            )
        else:
            opt_alpha = alpha
        train_df, test_df = folds._get_test_train_dfs(input_data, fold_df)
        X_train, X_test = train_df.spike_count.values, test_df.spike_count.values
        if sqrt_spikes:
            X_train, X_test = np.sqrt(X_train), np.sqrt(X_test)
        if standardise_spikes:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        y_train, y_test = train_df.distance_bin_id.values, test_df.distance_bin_id.values
        # fit model
        model = LogisticRegression(penalty="l2", C=opt_alpha, max_iter=10_000, random_state=0, class_weight="balanced")
        model.fit(X_train, y_train)
        # predict
        y_prob = model.predict_proba(X_test)
        results_df.loc[test_df.index, "decoded_distance_prob"] = y_prob
    return results_df


def get_CV_alpha(
    input_data,
    fold_df,
    metric,
    output="max",
    sqrt_spikes=True,
    standardise_spikes=True,
    return_as="best",
    verbose=False,
):
    """ """
    distance_bin_mids = np.array(sorted([b.mid for b in input_data.distance_bin.unique()]))
    # split training data into folds
    val_df = fold_df["train"]
    _vfolds = val_df.columns.values
    val_results = []
    for i, vfold in enumerate(_vfolds):
        if verbose:
            print(f"    Validation fold {i}")
        # index input data for validation test and train
        test_df = input_data[input_data.trial_unique_ID.isin(val_df[vfold].values)]
        train_df = input_data[input_data.trial_unique_ID.isin(val_df.drop(columns=vfold).unstack().dropna().values)]
        # get X and y
        X_train, X_test = train_df.spike_count.values, test_df.spike_count.values
        if sqrt_spikes:
            X_train, X_test = np.sqrt(X_train), np.sqrt(X_test)
        if standardise_spikes:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        y_train, y_test = train_df.distance_bin_id.values, test_df.distance_bin_id.values
        y_true = test_df[metric].values
        # search over regularisation strengths
        best_alpha, best_MSE = search_reg(
            X_train, X_test, y_train, y_test, y_true, output=output, distances=distance_bin_mids
        )
        val_results.append(
            {
                "vfold": vfold,
                "best_alpha": best_alpha,
                "best_MSE": best_MSE,
            }
        )
    reg_df = pd.DataFrame(val_results)
    if return_as == "df":
        return reg_df
    elif return_as == "best":
        # median opt reg strength across folds
        opt_reg = reg_df.best_alpha.median()
        return opt_reg
    else:
        raise ValueError(f"Return as must be 'df' of 'best'. ")


def search_reg(
    X_train,
    X_test,
    y_train,
    y_test,
    y_true,
    distances,
    output="max",
    reg_range=np.logspace(-4, 4, 20),
    tol=1e-3,
    patience=None,
    return_as="best",
    verbose=False,
):
    """
    CV search for optimal regulaisation strength (in training data)
    """
    best_alpha = None
    best_SE = np.inf
    history = []
    no_improvement_count = 0
    for alpha in reg_range:
        model = LogisticRegression(penalty="l2", C=alpha, max_iter=10_000, random_state=0, class_weight="balanced")
        model.fit(X_train, y_train)
        if output == "weighted":
            y_prob = model.predict_proba(X_test)
            decoded_dist = np.dot(y_prob, distances)  # weighted average of decoded distances
            SE = np.mean(np.abs((decoded_dist - y_true)))
        elif output == "max":
            y_pred = model.predict(X_test)
            decoded_dist = distances[y_pred]
            test_dist = distances[y_test]
            SE = np.mean(np.abs((decoded_dist - test_dist)))
        if SE < best_SE - tol:
            best_SE = SE
            best_alpha = alpha
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if patience is not None and no_improvement_count >= patience:
                if verbose:
                    print(f"Stopping early at α = {alpha:.3e} with SE = {SE:.4f}")
                break
        if verbose:
            print(f" α = {alpha:.3e},  SE = {SE:.4f}")
        history.append((alpha, SE))
    if return_as == "best":
        return best_alpha, best_SE
    elif return_as == "history":
        return np.array(history).T
    else:
        raise ValueError(f"Unknown return_as: {return_as}. Must be 'best' or 'history'.")


def get_input_data(
    session,
    resolution=0.2,
    metric=("distance_to_goal", "geodesic"),
    include_multiunits=True,
    moving_only=False,
    max_steps_to_goal=30,
    bin_spacing=0.05,
    bin_method="uniform",
    max_distance=None,
    n_log_bins=25,
    balance_distances=False,
):
    """"""
    # load data
    navigation_df = session.navigation_df
    spike_counts_df = session.navigation_spike_counts_df.reset_index(drop=True)
    cluster_metrics = session.cluster_metrics
    session_info = session.session_info
    # filter for single units
    if not include_multiunits:
        single_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID
        single_units = convert.cluster_IDs2scluster_unique_IDs(session_info, single_units)
        spike_counts_df = spike_counts_df[[("spike_count", u) for u in single_units]]
    # downsample to specified resolution with sliding window
    ds_nav_df, ds_spike_counts_df = ds.downsample_nav_spikes_data(
        navigation_df,
        spike_counts_df,
        resolution=resolution,
        distance_metrics=[("steps_to_goal", "future"), metric],
    )
    input_df = pd.concat([ds_nav_df, ds_spike_counts_df], axis=1)
    # filter for valid trial times
    input_df = input_df[input_df.trial_phase == "navigation"]
    # add distance bins
    if moving_only:
        input_df = input_df[input_df.moving]
    if max_steps_to_goal is not None:
        input_df = input_df[input_df.steps_to_goal.future < max_steps_to_goal]
    # remove frames where distance is above max (treat as outliers)
    if metric[0] == "distance_to_goal":
        if max_distance is None:
            max_distance = dd.get_distance_percentile(metric, 0.85)
        if bin_method == "uniform":
            n_bins = int(max_distance / bin_spacing)
        elif bin_method == "log":
            n_bins = n_log_bins
        input_df = input_df[input_df[metric] < max_distance]
        bins = convert._get_distance_bins(
            binning_method=bin_method,
            n_distance_bins=n_bins,
            distance_metrics=metric,
            max_distance=max_distance,
        )
    else:
        NotImplementedError()
    # bin distances
    input_df.loc[:, "distance_bin"] = pd.cut(input_df[metric], bins=bins, include_lowest=True).to_numpy()
    input_df.loc[:, "distance_bin_mid"] = input_df.distance_bin.apply(lambda x: x.mid)
    input_df.loc[:, "distance_bin_id"] = input_df.distance_bin.map({b: i for i, b in enumerate(bins)})

    # add error times
    nav_error_times = get_nav_error_times(session)
    poke_error_times = get_poke_error_times(session)
    input_df.loc[:, "nav_error"] = nearest_time_mask(nav_error_times, input_df.time)
    input_df.loc[:, "poke_error"] = nearest_time_mask(poke_error_times, input_df.time)

    # add other info
    input_df[("subject_ID", "")] = session.subject_ID
    input_df[("maze_name", "")] = session.maze_name
    input_df[("day_on_maze", "")] = session.day_on_maze

    if not balance_distances:
        return input_df
    else:  # balance data across distance bins
        max_size = input_df.groupby("distance_bin_id").size().max()
        balanced_data = (
            input_df.groupby("distance_bin_id", group_keys=False)
            .sample(n=max_size, replace=True, random_state=42)
            .reset_index(drop=True)
        )
        return balanced_data


def nearest_time_mask(times_1, times_2):
    """
    Return a boolean mask (pd.Series) over times_2 marking nearest matches to times_1.
    """
    idx = pd.Index(times_2.values)
    nearest_pos = idx.get_indexer(times_1, method="nearest")
    mask = pd.Series(False, index=times_2.index)
    mask.iloc[nearest_pos[nearest_pos >= 0]] = True
    return mask


def get_nav_error_times(session, n=1):
    """
    define navigation errors as times where distance to goal is decreasing and then starts increasing
    for at least n towers.
    returned as a list of times errors occured in the session
    """
    df = session.trajectory_decisions_df  # node by node decisions
    trials = df.trial.dropna().unique()
    error_times = []
    for t in trials:
        trial_df = df[(df.trial == t) & (df.trial_phase == "navigation")]
        dist_to_goal = trial_df.geodesic_distance_to_goal
        diffs = dist_to_goal.diff() > 0
        increase = diffs > 0
        # previous n diffs were all negative
        prev_n_decreasing = diffs.shift(1).rolling(n, min_periods=n).max().fillna(0).astype(bool)
        # detect is True at the first increasing index i; shift it back to i-1
        error_mask = (increase & prev_n_decreasing).fillna(False)
        if len(error_mask) < 2:
            continue
        error_mask = error_mask.shift(-1)
        error_mask.iloc[-1] = False  # avoid NaN at end
        _error_times = trial_df.time[error_mask].to_list()
        error_times.extend(_error_times)
    return error_times


def get_poke_error_times(session):
    """
    defined as reward port pokes that were not at the goal location
    returned as a list of times errors occured in the session
    """
    trials_df = session.trials_df.copy()
    events_df = session.events_df

    error_times = []
    for _, t in trials_df.iterrows():
        _df = events_df[events_df.time.between(t.time.cue, t.time.reward)].copy()
        goal = t[("goal", "")]
        poked_mask = _df.name.str.contains("_in")
        poked_mask[poked_mask.isna()] = False
        goal_mask = _df.name.str.contains(goal)
        goal_mask[goal_mask.isna()] = False
        error_mask = poked_mask & ~goal_mask
        _error_times = _df.time[error_mask].to_list()
        error_times.extend(_error_times)
    return error_times
