"""
Refactoring goal_decoding.py / gd2.py to include a separate utils supporing lib as code bases was growing too large
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import polars as pl
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed


from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import convert
from GridMaze.analysis.core import permute
from GridMaze.maze import representations as mr


# %% Global Variables

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "distance_to_goal"


with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

FRAME_RATE = 60

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]
GOAL_SETS = ["subset_1", "subset_2", "all"]


# %% test train split functions


def _get_test_train_dfs(input_data, fold_df, training_trial_phases=["navigation"]):
    """ """
    test_trials = [t for t in fold_df.test.values.flatten() if isinstance(t, str)]
    train_trials = [t for t in fold_df.train.values.flatten() if isinstance(t, str)]
    train_df = input_data[input_data.trial_unique_ID.isin(train_trials)]
    # include only specified trial phases in training data
    train_df = train_df[train_df.trial_phase.isin(training_trial_phases)]
    test_df = input_data[input_data.trial_unique_ID.isin(test_trials)]
    return train_df, test_df


def _get_test_train_arrays(
    train_df,
    test_df,
    input_type="spikes",
    output_type="goal",
    whiten_features=True,
    basis_fn=None,
):
    """
    TODO: abstract the by_distance functionality
    """
    if "by_distance" in input_type:
        assert basis_fn is not None, "basis_fn must be provided for 'by_distance' input"
    # process input data (X)
    if input_type == "spikes":
        X_train, X_test = train_df.spike_count.values, test_df.spike_count.values
    elif input_type == "place_direction_prob":
        X_train, X_test = train_df.place_direction_prob.values, test_df.place_direction_prob.values
    elif input_type == "place_probs":
        X_train, X_test = train_df.place_probs.values, test_df.place_probs.values
    elif "by_distance" in input_type:
        Xs = []
        for df in [train_df, test_df]:
            basis_activations = basis_fn(df.steps_to_goal.future.values)
            if "spikes" in input_type:
                F = df.spike_count.values
            elif "place_probs" in input_type:
                F = df.place_probs.values
            elif "place_direction_prob" in input_type:
                F = df.place_direction_prob.values
            else:
                raise ValueError(f"Unknown input type {input_type!r}")
            F_by_distance = F[:, :, None] * basis_activations[:, None, :]  # [n_timepoints, n_neurons, n_bases]
            Xs.append(F_by_distance.reshape(F.shape[0], -1))
        X_train, X_test = Xs
    else:
        raise ValueError(f"Unknown input type {input_type!r}")

    # process output data (y)
    if output_type == "place_direction":
        y_train, y_test = train_df.place_direction.values, test_df.place_direction.values
    elif output_type == "goal":
        y_train, y_test = train_df.goal.values, test_df.goal.values
    elif output_type == "place":
        y_train, y_test = train_df.maze_position.simple.values, test_df.maze_position.simple.values
    else:
        raise ValueError(f"Unknown output type {output_type!r}")

    if whiten_features:
        scaler = StandardScaler()  # mean=0, std=1 per column
        scaler.fit(X_train)  # learn stats on train
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# %% auto opt regularisation


def get_opt_reg(
    input_data,
    fold_df,
    simple_maze=None,
    basis_fn=None,
    input_type="spikes",  # X
    output_type="place_direction",  # Y
    training_trial_phases=["navigation"],
    reg_range=[None, 1, 10, 50, 1e2, 5e2, 1e3, 5e3, 1e4],
    eval_metric="expected_distance_error",
    eval_kwargs={
        "op": "sum",
        "dist_metric": "geodesic",
        "cue_window": (0, 4),
        "reward_window": (-8, 0),
    },
    verbose=True,
):
    # prepare data exactly as before
    train_df, test_df = _get_test_train_dfs(input_data, fold_df, training_trial_phases)
    X_train, X_test, y_train, y_test = _get_test_train_arrays(
        train_df,
        test_df,
        input_type,
        output_type,
        whiten_features=True,
        basis_fn=basis_fn,
    )
    # now parallel evaluate
    if verbose:
        print("Evaluating reg_range in parallel")
    eval_metrics = Parallel(n_jobs=len(reg_range), verbose=False)(
        delayed(_evaluate_alpha)(
            inv_alpha, X_train, y_train, X_test, y_test, test_df, simple_maze, eval_metric, eval_kwargs, output_type
        )
        for inv_alpha in reg_range
    )
    eval_metrics = np.array(eval_metrics)
    # choose best
    if eval_metric == "expected_distance_error":
        opt_reg = reg_range[np.argmin(eval_metrics)]
    elif eval_metric == "decoding_accuracy":
        opt_reg = reg_range[np.argmax(eval_metrics)]
    else:
        raise ValueError(f"Unknown eval metric {eval_metric!r}")
    if verbose:
        print(f"reg_range: {reg_range}")
        print(f"eval_metrics: {eval_metrics}")
        print(f"opt_reg: {opt_reg}")
    return opt_reg


def _evaluate_alpha(
    inv_alpha, X_train, y_train, X_test, y_test, test_df, simple_maze, eval_metric, eval_kwargs, output_type
):
    # instantiate decoder
    if inv_alpha is None:
        decoder = LogisticRegression(penalty=None, max_iter=10_000, random_state=0, class_weight="balanced")
    else:
        cw = "balanced" if output_type in ["place", "place_direction"] else None
        decoder = LogisticRegression(penalty="l2", C=inv_alpha, max_iter=10_000, random_state=0, class_weight=cw)
    # fit & get probs
    decoder.fit(X_train, y_train)
    Yprobs = decoder.predict_proba(X_test)
    features = list(decoder.classes_)
    df = get_decoding_results_df(test_df, y_test, Yprobs, features, output_type, engine="polars")
    df = df.with_columns(pl.lit(0).alias("repeat"))
    decoding_metrics_df = get_decoding_metrics_df(df, simple_maze, output_type=output_type)
    if eval_metric == "expected_distance_error":
        metric = f"{eval_kwargs["dist_metric"]}_ede"
    elif eval_metric == "decoding_accuracy":
        metric = "test_acc"
    else:
        raise ValueError(f"Unknown eval metric {eval_metric!r}")
    values = []
    for event in ["cue", "reward"]:
        window = eval_kwargs[f"{event}_window"]
        _df = decoding_metrics_df[decoding_metrics_df[f"{event}_aligned_time"].between(*window)]
        mean_window_ede = (
            _df.groupby(["trial_unique_ID", f"{event}_aligned_time"])[metric].mean().unstack().mean().to_list()
        )
        values.extend(mean_window_ede)
    return np.mean(values)


# %% decoding metrics (polars)


def get_decoding_metrics_df(results_df, simple_maze, output_type="goal", ede_op="sum"):
    assert isinstance(results_df, pl.DataFrame), "results_df must be a Polars DataFrame"
    _check_decoding_type(results_df, output_type)
    metrics_df = _get_decoding_acc(results_df, output_type)
    EDEs = _get_expected_distance_error(results_df, simple_maze, output_type=output_type, op=ede_op)
    metrics_df = metrics_df.join(EDEs, on=["sample_index", "repeat"], how="inner")
    # return as pandas
    metrics_df = metrics_df.to_pandas()
    metrics_df.reset_index(drop=True, inplace=True)
    return metrics_df


def _get_decoding_acc(results_df, output_type):
    """"""
    # compute with polars
    acc_df = (
        results_df
        # sort each group so the highest-prob row comes first
        .sort(f"predicted_{output_type}_prob", descending=True)
        .group_by(["sample_index", "repeat"], maintain_order=True)
        .head(1)
        .with_columns(
            (pl.col(f"true_{output_type}") == pl.col(f"predicted_{output_type}")).cast(pl.Int8).alias("test_acc")
        )
    )
    acc_df = acc_df.sort(by=["sample_index", "repeat"])
    return acc_df


def _get_expected_distance_error(
    results_df,
    simple_maze,
    op="sum",
    output_type="goal",
):
    """
    input polars df (need speed from polars for processing large outputs from permuted decodings)
    output pandas df
    """
    # add colums for distance to goal (geo or euc) for every true and predicted place/goal pair
    df = results_df.with_columns(_get_distance_cols_pl(results_df, simple_maze, output_type, round_euc=False))
    # calc weighted distance error
    df = df.with_columns(
        [
            (pl.col(f"predicted_{output_type}_prob") * pl.col("geo_dist")).alias("geo_weight_prob"),
            (pl.col(f"predicted_{output_type}_prob") * pl.col("euc_dist")).alias("euc_weight_prob"),
        ]
    )
    group_cols = ["sample_index", "repeat"]
    # aggregate per‐sample (sum or max)
    if op == "sum":
        sample_EDE = (
            df.group_by(group_cols, maintain_order=True)
            .agg(
                [
                    pl.sum("geo_weight_prob").alias("geodesic_ede"),
                    pl.sum("euc_weight_prob").alias("euclidean_ede"),
                ]
            )
            .sort(group_cols)
        )
    elif op == "max":
        sample_EDE = (
            df.group_by(group_cols, maintain_order=True)
            .agg(
                [
                    pl.max("geo_weight_prob").alias("geodesic_ede"),
                    pl.max("euc_weight_prob").alias("euclidean_ede"),
                ]
            )
            .sort(group_cols)
        )
    else:
        raise ValueError(f"Unsupported op: {op!r}")
    return sample_EDE


def _get_distance_cols_pl(results_df, simple_maze, output_type, round_euc=False):
    """
    input must be a Polars DataFrame
    Vectorized version in Polars: builds NxN distance matrices once,
    then does batch lookups for all rows in results_df.
    """
    if output_type == "place_direction":
        # add true_place and predicted_place columns
        results_df = results_df.with_columns(
            [
                pl.col("true_place_direction").str.split("_").list.get(0).alias("true_place"),
                pl.col("predicted_place_direction").str.split("_").list.get(0).alias("predicted_place"),
            ]
        )
        output_type = "place"
    # Build label→coord and label→idx
    label2coord = mr.get_maze_label2coord(simple_maze)
    labels = list(label2coord.keys())
    label2idx = {lab: i for i, lab in enumerate(labels)}
    n_labels = len(labels)

    # Build geodesic distance matrix
    ext_maze = mr.get_extended_simple_maze(simple_maze)
    raw_geo = dict(nx.all_pairs_dijkstra_path_length(ext_maze, weight="weight"))
    geo_mat = np.empty((n_labels, n_labels), dtype=float)
    for i, lab_i in enumerate(labels):
        base_coord = label2coord[lab_i]
        row_dist = raw_geo[base_coord]
        for j, lab_j in enumerate(labels):
            geo_mat[i, j] = row_dist[label2coord[lab_j]]

    # Build “center” coords for Euclidean
    centers = np.vstack(
        [np.mean(c, axis=0) if isinstance(c[0], tuple) else np.array(c) for c in label2coord.values()]
    )  # shape (n_labels, 2)

    # Extract the integer indices from the Polars cols into NumPy arrays
    true_idxs = np.vectorize(label2idx.__getitem__)(results_df[f"true_{output_type}"].to_numpy())
    pred_idxs = np.vectorize(label2idx.__getitem__)(results_df[f"predicted_{output_type}"].to_numpy())

    # 5) Lookup geodesic and compute Euclidean
    geo_dist = geo_mat[true_idxs, pred_idxs]
    diffs = centers[true_idxs] - centers[pred_idxs]
    euc_dist = np.linalg.norm(diffs, axis=1) * 2
    if round_euc:
        euc_dist = np.rint(euc_dist).astype(int)

    return [pl.Series("geo_dist", geo_dist), pl.Series("euc_dist", euc_dist)]


def get_decoding_results_df(test_df, y_test, Yprobs, features, output_type, engine="polars"):
    """
    Organises output of decoding analyses to be read into get_decoding_metrics_df
    """
    # define df columns
    n_samples, n_features = Yprobs.shape
    sample_index = np.repeat(test_df.index.values, n_features)
    train_unique_IDs = np.repeat(test_df.trial_unique_ID.values, n_features)
    cue_aligned_times = np.repeat(test_df.event_aligned_bin["cue"].values, n_features)
    reward_aligned_times = np.repeat(test_df.event_aligned_bin["reward"].values, n_features)
    trial_phases = np.repeat(test_df.trial_phase.values, n_features)
    steps_to_goals = np.repeat(test_df.steps_to_goal.future.values, n_features)
    true = np.repeat(y_test, n_features)
    predicted = np.tile(features, n_samples)
    predicted_probs = Yprobs.ravel()
    # create df
    if engine == "polars":
        df = pl.DataFrame(  # note use of polars df (big output dfs need something faster than pandas)
            {
                "sample_index": sample_index,
                "trial_unique_ID": np.repeat(test_df.trial_unique_ID.values, n_features),
                "cue_aligned_time": np.repeat(test_df.event_aligned_bin["cue"].values, n_features),
                "reward_aligned_time": np.repeat(test_df.event_aligned_bin["reward"].values, n_features),
                "trial_phase": np.repeat(test_df.trial_phase.values, n_features),
                "steps_to_goal": np.repeat(test_df.steps_to_goal.future.values, n_features),
                f"true_{output_type}": np.repeat(y_test, n_features),
                f"predicted_{output_type}": np.tile(features, n_samples),
                f"predicted_{output_type}_prob": Yprobs.ravel(),
            }
        )
    elif engine == "pandas":
        df = pd.DataFrame(
            {
                "sample_index": sample_index,
                "trial_unique_ID": train_unique_IDs,
                "cue_aligned_time": cue_aligned_times,
                "reward_aligned_time": reward_aligned_times,
                "trial_phase": trial_phases,
                "steps_to_goal": steps_to_goals,
                f"true_{output_type}": true,
                f"predicted_{output_type}": predicted,
                f"predicted_{output_type}_prob": predicted_probs,
            }
        )
    else:
        raise ValueError(f"Unknown engine {engine!r}")
    return df


def _check_decoding_type(results_df, decoding_type):
    """ """
    if decoding_type == "goal":
        assert "predicted_goal" in results_df.columns, "results_df does not contain goal decoding"
    elif decoding_type == "place":
        assert "predicted_place" in results_df.columns, "results_df does not contain place decoding"
    elif decoding_type == "place_direction":
        assert "predicted_place_direction" in results_df.columns, "results_df does not contain place direction decoding"
    else:
        raise ValueError(f"Unknown decoding type {decoding_type}")


# %% get sessions


def get_sessions_for_analysis(subject_IDs, maze_names, goal_subsets):
    """ """
    days_on_maze = "late" if "all" in goal_subsets else "all"
    sessions = gs.get_maze_sessions(
        subject_IDs=subject_IDs,
        maze_names=maze_names,
        days_on_maze="all",
        goal_subsets=goal_subsets,
        with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "trials_df"],
        must_have_data=True,
    )
    sessions = [sessions] if isinstance(sessions, gs.MazeSession) else sessions
    # check sessions have at least 2 trials per goal
    keep_sessions = []
    for session in sessions:
        trials_df = session.trials_df
        if trials_df.groupby("goal").trial.count().ge(2).all():
            keep_sessions.append(session)
    if len(keep_sessions) == 0:
        raise FileNotFoundError(f"No valid session for: {subject_IDs}, {maze_names}, {goal_subsets}")
    return keep_sessions


# %% get distance aligned input data


def get_distance_aligned_input_data(
    session,
    resolution=0.5,
    include_multi_units=True,
    include_trial_phases=["navigation"],
    ignore_last_n=0,
    binning_method="uniform",
    n_bins=25,
    max_steps_to_goal=25,
    include_place_onehots=False,
):
    """
    Returns a dataframe with spike counts aligned to future path-distance to goal over all trials in a session.
    """
    # load data
    trials_df = session.trials_df
    navigation_df = session.navigation_df
    spike_counts_df = session.navigation_spike_counts_df.reset_index(drop=True)
    # filter clusters
    keep_clusters = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=include_multi_units,
    )
    spike_counts_df = spike_counts_df[spike_counts_df.columns[spike_counts_df.spike_count.columns.isin(keep_clusters)]]
    # downsample data
    ds_nav_info, ds_spike_counts_df = _downsample_data(navigation_df, spike_counts_df, resolution)
    # update distances in non_navigation trial periods
    if "ITI" in include_trial_phases or "reward_consumption" in include_trial_phases:
        ds_nav_info[("steps_to_goal", "future")] = _add_non_nav_distances(
            session, ds_nav_info, ignore_last_n=ignore_last_n
        )
    # add distance bins info
    bins = convert._get_distance_bins(
        binning_method=binning_method,
        n_distance_bins=n_bins,
        distance_metrics=("steps_to_goal", "future"),
        max_distance=max_steps_to_goal,
    )
    ds_nav_info[("steps_to_goal", "bin")] = pd.cut(ds_nav_info.steps_to_goal.future, bins=bins)
    ds_nav_info[("steps_to_goal", "bin_mid")] = ds_nav_info.steps_to_goal.bin.apply(lambda x: x.mid).astype(float)
    # add event aligned time info (for later cross-decoder comparisons)
    ds_nav_info[("event_aligned_time", "cue")] = _get_event_aligned_times(ds_nav_info, trials_df, "cue")
    ds_nav_info[("event_aligned_time", "reward")] = _get_event_aligned_times(ds_nav_info, trials_df, "reward")
    # combine and filter out points where distance is not defined
    ds_nav_rates_df = pd.concat([ds_nav_info, ds_spike_counts_df], axis=1)
    ds_nav_rates_df = ds_nav_rates_df[~ds_nav_rates_df.steps_to_goal.bin.isna()]
    ds_nav_rates_df = ds_nav_rates_df[ds_nav_rates_df.trial_phase.isin(include_trial_phases)]
    ds_nav_rates_df[("trial", "")] = ds_nav_rates_df[("trial", "")].astype(int)
    # include position inputs for decoder (optional)
    if include_place_onehots:
        place_onehots_df = _get_place_onehots_df(session, ds_nav_rates_df)
        ds_nav_rates_df = pd.concat([ds_nav_rates_df, place_onehots_df], axis=1)
    return ds_nav_rates_df


def _add_non_nav_distances(
    session,
    nav_info_df,
    ignore_last_n=0,
    fill_first_last_trial=True,
):
    """
    first/last trial can lack steps to goal bc/ no goal defined, fill_first_last_trial == True
    sets np.nans to 0
    """
    # load additional data
    trials_df = session.trials_df
    trials_df = trials_df.set_index(("trial", ""))
    last_trial = trials_df.index.max()
    next_goals = trials_df[("goal", "")].shift(-1)
    # precompute distances
    simple_maze = session.simple_maze()
    extended = mr.get_extended_simple_maze(simple_maze)
    raw_dist = dict(nx.all_pairs_dijkstra_path_length(extended, weight="weight"))
    label2coord = mr.get_maze_label2coord(simple_maze)
    # Build dense N×N matrix for fast look ups
    coords = list(label2coord.values())
    coord2idx = {coord: i for i, coord in enumerate(coords)}
    N = len(coords)
    dist_mat = np.zeros((N, N), float)
    for src_coord, targets in raw_dist.items():
        i = coord2idx[src_coord]
        for dst_coord, d in targets.items():
            dist_mat[i, coord2idx[dst_coord]] = d
    # rename cols for convience
    nav = nav_info_df.copy().sort_index(axis=1)
    nav["trial"] = nav[("trial", "")]
    nav["phase"] = nav[("trial_phase", "")]
    nav["future_dist"] = nav[("steps_to_goal", "future")]
    nav["pos_label"] = nav[("maze_position", "simple")]
    nav["next_goal"] = nav["trial"].map(next_goals)
    nav["window_idx"] = nav.groupby("trial").cumcount()
    nav["n_windows"] = nav.groupby("trial")["trial"].transform("count")
    # optionally ignore last n windows of each trial
    valid = (
        nav["future_dist"].isna()
        & nav["phase"].isin(["reward_consumption", "ITI"])
        & (nav["window_idx"] < nav["n_windows"] - ignore_last_n)
        & (nav["trial"] != last_trial)
    )
    # compute distances
    src_idxs = nav.loc[valid, "pos_label"].map(label2coord).map(coord2idx).to_numpy()
    dst_idxs = nav.loc[valid, "next_goal"].map(label2coord).map(coord2idx).to_numpy()
    computed = dist_mat[src_idxs, dst_idxs]
    # update distances and return
    out = nav_info_df[("steps_to_goal", "future")].copy()
    out.loc[valid.values] = computed
    if fill_first_last_trial:
        out.loc[out.isna()] = 0
    return out


def _get_place_onehots_df(session, nav_rates_df):
    """ """
    simple_maze = session.simple_maze()
    positions = mr.get_maze_locations(simple_maze)
    # convert position labels to one-hot encoding
    place_onehots = convert.place2onehot(nav_rates_df.maze_position.simple.values, session.simple_maze())
    place_onehots_df = pd.DataFrame(
        data=place_onehots.astype(int),
        index=nav_rates_df.index,
        columns=pd.MultiIndex.from_product([["place_onehot"], positions]),
    )
    return place_onehots_df


def _get_event_aligned_times(nav_info, trials_df, event):
    """
    Returns a series of time relative to event on every trial.
    """
    trials2event_time = trials_df.set_index("trial")["time"][event]
    trial_ids = nav_info["trial"].astype("Int64")
    event_times = trial_ids.map(trials2event_time)
    return nav_info["time"] - event_times


# %% place decoding input data


def get_place_decoding_input_data(
    session,
    resolution=0.5,
    include_multi_units=True,
    window=(-10, 10),
    permuted=False,
):
    """
    Simpler version of other input_data functions which just returns downsampled navigation
    data but with info relevant for place decoding analyses
    """
    # load data
    session_info = session.session_info
    trials_df = session.trials_df
    navigation_df = session.navigation_df
    spike_counts_df = session.navigation_spike_counts_df.reset_index(drop=True)
    if permuted:
        spike_counts_df = permute.random_circular_shift(spike_counts_df)
    # filter clusters
    keep_clusters = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=include_multi_units,
    )
    spike_counts_df = spike_counts_df[spike_counts_df.columns[spike_counts_df.spike_count.columns.isin(keep_clusters)]]
    # downsample data
    nav_info, spike_counts_df = _downsample_data(navigation_df, spike_counts_df, resolution)
    # update trial definitions to start in previous ITI
    nav_info[("trial", "")] = update_trial_ID(nav_info, trials_df)
    nav_info[("trial_unique_ID", "")] = convert.trial2trial_unique_ID(session_info, nav_info["trial"])
    # match goal to updated trial 1D
    trial2goal = trials_df.set_index("trial")[("goal", "")].to_dict()
    nav_info[("goal", "")] = nav_info.trial.map(trial2goal)
    # add event aligned time
    nav_info[("event_aligned_time", "cue")] = _get_event_aligned_times(nav_info, trials_df, "cue")
    nav_info[("event_aligned_time", "reward")] = _get_event_aligned_times(nav_info, trials_df, "reward")
    # bin event aligned time and report bin_mids for convience
    bins = pd.IntervalIndex.from_breaks(np.arange(window[0], window[1] + resolution, resolution), closed="right")
    cue_aligned_bins = pd.cut(nav_info[("event_aligned_time", "cue")], bins=bins)
    reward_aligned_bins = pd.cut(nav_info[("event_aligned_time", "reward")], bins=bins)
    nav_info[("event_aligned_bin", "cue")] = cue_aligned_bins.apply(lambda x: x.mid).astype(float)
    nav_info[("event_aligned_bin", "reward")] = reward_aligned_bins.apply(lambda x: x.mid).astype(float)
    # add non nav distances
    nav_info[("steps_to_goal", "future")] = _add_non_nav_distances(
        session, nav_info, ignore_last_n=0, fill_first_last_trial=True
    )
    # combine and remove out of trial times
    nav_rates_df = pd.concat([nav_info, spike_counts_df], axis=1)
    nav_rates_df = nav_rates_df[~nav_rates_df.trial.isna()]
    return nav_rates_df


def update_trial_ID(
    nav_info,
    trials_df,
):
    """
    Vectorized assignment of ITI-period rows to the *next* trial ID,
    with all other rows keeping their own trial ID (and non-string
    phases becoming NaN). Returns a NumPy array of the updated IDs.
    """
    trial = nav_info.trial
    trial_phase = nav_info.trial_phase

    # compute some masks
    is_def = trial_phase.map(lambda x: isinstance(x, str))
    is_iti = is_def & (trial_phase == "ITI")
    next_trial = trial + 1
    within_bounds = next_trial <= trials_df.trial.max()

    updated = np.where(~is_def, np.nan, np.where(is_iti, np.where(within_bounds, next_trial, np.nan), trial))

    return pd.Series(updated)


# %% event aligned input data


def get_event_aligned_input_data(
    session,
    event="cue",
    resolution=0.5,
    window=(-10, 10),
    include_multi_units=True,
    binning_method="uniform",
    n_bins=25,
    max_steps_to_goal=25,
    include_place_onehots=False,
):
    """
    Returns a dataframe with spike counts aligned to event (cue & reward) times.
    """
    # load data
    simple_maze = session.simple_maze()
    session_info = session.session_info
    trials_df = session.trials_df
    navigation_df = session.navigation_df
    spike_counts_df = session.navigation_spike_counts_df.reset_index(drop=True)
    # filter clusters
    keep_clusters = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=include_multi_units,
    )
    spike_counts_df = spike_counts_df[spike_counts_df.columns[spike_counts_df.spike_count.columns.isin(keep_clusters)]]
    # get rates aligned to event
    frames_before, frames_after = int(window[0] * FRAME_RATE), int(window[1] * FRAME_RATE)
    event_times = trials_df.set_index("trial").time[event]
    trial2goal = trials_df.set_index("trial").goal
    # precalculate distances to goal
    extended_simple_maze = mr.get_extended_simple_maze(simple_maze)
    path_distances = dict(nx.all_pairs_dijkstra_path_length(extended_simple_maze, weight="weight"))
    label2coord = mr.get_maze_label2coord(simple_maze)
    nav_info_dfs, spike_count_dfs = [], []
    for trial, event_time in event_times.items():
        event_frame = (navigation_df.time - event_time).abs().argmin()
        nav_aligned_df = navigation_df.iloc[event_frame + frames_before : event_frame + frames_after].reset_index(
            drop=True
        )
        spikes_aligned_df = spike_counts_df.iloc[event_frame + frames_before : event_frame + frames_after].reset_index(
            drop=True
        )
        # downsample to speficied resolution
        ds_nav_aligned_df, ds_spikes_aligned_df = _downsample_data(nav_aligned_df, spikes_aligned_df, resolution)
        # add event aligned time info
        timepoints = np.arange(window[0], window[1], resolution)
        if len(timepoints) > ds_nav_aligned_df.shape[0]:
            # can happen for last trial in session (no more frames)
            timepoints = timepoints[: ds_nav_aligned_df.shape[0]]
        ds_nav_aligned_df[("event_aligned_time", event)] = timepoints
        # update distnace outside navigation where they are not defined (use shortest path
        # upcoming goal (event=cue) or shortest path to just visted goal (event=reward))
        ds_nav_aligned_df[("goal", "")] = trial2goal[trial]
        outside_trial_mask = (ds_nav_aligned_df.trial != trial) | (ds_nav_aligned_df.trial_phase != "navigation")
        pos_coords = ds_nav_aligned_df.loc[outside_trial_mask, ("maze_position", "simple")].map(label2coord)
        goal_coords = ds_nav_aligned_df.loc[outside_trial_mask, ("goal", "")].map(label2coord)
        ds_nav_aligned_df.loc[outside_trial_mask, ("steps_to_goal", "future")] = [
            path_distances[src][dst] for src, dst in zip(pos_coords, goal_coords)
        ]
        # update trial info so it is consistent across all aligned times
        ds_nav_aligned_df[("trial", "")] = trial
        ds_nav_aligned_df[("trial_unique_ID", "")] = convert.trial2trial_unique_ID(session_info, trial)
        nav_info_dfs.append(ds_nav_aligned_df)
        spike_count_dfs.append(ds_spikes_aligned_df)
    # combine over trials
    nav_info_df = pd.concat(nav_info_dfs, axis=0).reset_index(drop=True)
    spike_count_df = pd.concat(spike_count_dfs, axis=0).reset_index(drop=True)
    # add distance bins info
    bins = convert._get_distance_bins(
        binning_method=binning_method,
        n_distance_bins=n_bins,
        distance_metrics=("steps_to_goal", "future"),
        max_distance=max_steps_to_goal,
    )
    nav_info_df[("steps_to_goal", "bin")] = pd.cut(nav_info_df.steps_to_goal.future, bins=bins)
    nav_info_df[("steps_to_goal", "bin_mid")] = nav_info_df.steps_to_goal.bin.apply(lambda x: x.mid).astype(float)
    # combine nav_info and spike counts
    event_aligned_nav_rates_df = pd.concat([nav_info_df, spike_count_df], axis=1)
    if include_place_onehots:
        place_onehots_df = _get_place_onehots_df(session, event_aligned_nav_rates_df)
        # combine with other data (nav_info and spike_counts)
        event_aligned_nav_rates_df = pd.concat([event_aligned_nav_rates_df, place_onehots_df], axis=1)
    return event_aligned_nav_rates_df


def _downsample_data(navigation_df, spike_counts_df, resolution=0.2):
    """ """
    # downsample spike counts by suming spikes within resolution window
    ds_frames = int(FRAME_RATE * resolution)
    ds_spike_counts_df = spike_counts_df.groupby(spike_counts_df.index // ds_frames).sum().reset_index(drop=True)
    # keep only relevant navigation info
    nav_info = navigation_df[
        [
            ("time", ""),
            ("trial_unique_ID", ""),
            ("trial", ""),
            ("goal", ""),
            ("trial_phase", ""),
            ("maze_position", "simple"),
            ("cardinal_movement_direction", ""),
            ("steps_to_goal", "future"),
        ]
    ]
    # downsample navigation info by taking values in mid window
    mid_window_inds = (spike_counts_df.index // ds_frames).unique() * ds_frames + (ds_frames // 2)
    mid_window_inds = mid_window_inds[mid_window_inds < len(nav_info)]
    nav_info = nav_info.iloc[mid_window_inds]
    # account for differences in ds methods
    nav_info.reset_index(drop=True, inplace=True)
    if nav_info.shape[0] < ds_spike_counts_df.shape[0]:
        ds_spike_counts_df = ds_spike_counts_df.iloc[:-1]
    return nav_info, ds_spike_counts_df


# %% Cross valdiation functions


def get_folds_df(session, goal_stratified=True, valid_trials=None, return_unique_IDs=True, n_test_trials=None):
    """ """
    n_trials = session.trials_df.trial.max()
    if goal_stratified:
        # check there are are enogh trials to stratify by goals if not split trials randomly
        # only applies to early sessions
        if n_trials < len(session.goals) * 2:
            folds_df = _get_folds_non_stratified(session, valid_trials, n_test_trials=(n_trials // 5))
        else:
            folds_df = _get_folds_goal_stratified(session, valid_trials, return_unique_IDs)
    else:
        folds_df = _get_folds_non_stratified(session, valid_trials, n_test_trials, return_unique_IDs)
    return folds_df


def _get_folds_goal_stratified(session, valid_trials=None, return_unique_IDs=True):
    """ """
    goals_df = get_goals_df(session, valid_trials, return_unique_IDs)
    # check there are at least 2 trials per goal (needed for test/train split)
    valid_goals_df = goals_df[goals_df.count(axis=1).ge(2)]
    # shuffle
    valid_goals_df = valid_goals_df.apply(lambda x: np.random.choice(x, size=len(x), replace=False), axis=1).apply(
        pd.Series
    )
    # split into test and train folds
    cols = valid_goals_df.columns
    fold_dfs = []
    for i in cols:
        fold = f"fold_{i}"
        test_df = pd.DataFrame(valid_goals_df[cols[i]])
        test_df.columns = pd.MultiIndex.from_product([[fold], ["test"], test_df.columns])
        train_df = valid_goals_df.drop(columns=cols[i])
        train_df.columns = pd.MultiIndex.from_product([[fold], ["train"], train_df.columns])
        fold_dfs.append(pd.concat([test_df, train_df], axis=1))
    # return as df
    folds_df = pd.concat(fold_dfs, axis=1)
    return folds_df


def get_goals_df(session, valid_trials=None, return_unique_IDs=True):
    """
    returns df with goals in index and corresponding session trials in columns
    """
    trials_df = session.trials_df
    assert trials_df.trial.max() > len(session.goals), "Session does not have enough trials to stratify by goals"
    if valid_trials is not None:
        trials_df = trials_df[trials_df.trial.isin(valid_trials)].reset_index(drop=True)
    goal2trials = {}
    for goal in session.goals:
        goal2trials[goal] = trials_df[trials_df.goal == goal].trial.to_list()
    goals_df = pd.DataFrame.from_dict(goal2trials, orient="index")
    if return_unique_IDs:
        session_info = session.session_info
        goals_df = goals_df.apply(lambda x: convert.trial2trial_unique_ID(session_info, x))
    return goals_df


def _get_folds_non_stratified(
    session,
    valid_trials=None,
    n_test_trials=5,
    return_unique_IDs=True,
):
    """
    No goal stratified validation folds. Instead specify how many trials per fold.
    Can be useful for all goals sessions where you want more training data per fold.
    """
    assert n_test_trials is not None, "n_test_trials must be specified for non goal-stratified validation folds"
    session_info = session.session_info
    trials_df = session.trials_df
    trials = trials_df.trial.values if valid_trials is None else valid_trials
    # shuffle trials
    trials = np.random.choice(trials, size=len(trials), replace=False)
    fold_dfs = []
    for fold, i in enumerate(range(0, len(trials), n_test_trials)):
        test_trials = trials[i : i + n_test_trials]
        train_trials = np.concatenate([trials[:i], trials[i + n_test_trials :]])
        fold_df = pd.DataFrame(
            {
                "test": pd.Series(test_trials),
                "train": pd.Series(train_trials),
            }
        )
        fold_df.columns = pd.MultiIndex.from_product([[f"fold_{fold}"], fold_df.columns])
        fold_dfs.append(fold_df)
    folds_df = pd.concat(fold_dfs, axis=1)
    if return_unique_IDs:
        folds_df = folds_df.apply(lambda x: convert.trial2trial_unique_ID(session_info, x))
    return folds_df


# %% Transform between distance and event aligned time


def _get_transformed_steps_to_goal(session, results_df, event, round_steps=False):
    """
    Translates between time aliged to cue or reward and steps to goal, based on that
    subjects average distance across all relevant maze and goal subset sessions.
    """
    assert "timepoint" in results_df.columns, "results_df does not contain timepoint column"
    window2step = get_step_time_transformation(session, event)
    transformed_steps_to_goal = results_df.timepoint.map(window2step).astype(int)
    if round_steps:
        transformed_steps_to_goal = transformed_steps_to_goal.round().astype(int)
    return transformed_steps_to_goal


def plot_steps_vs_time_curves(step_time_df, event="reward"):
    """ """
    f, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)
    for i, goal_subset in enumerate(GOAL_SETS):
        for j, maze_name in enumerate(MAZE_NAMES):
            ax = axes[i, j]
            df = step_time_df.query(f"goal_subset == '{goal_subset}' and maze == '{maze_name}' and event == '{event}'")
            grouped_df = df.groupby("event_aligned_time").steps_to_goal
            mean = grouped_df.mean()
            sem = grouped_df.sem()
            ax.plot(mean.index, mean)
            ax.fill_between(mean.index, mean - sem, mean + sem, alpha=0.2)
            ax.set_title(f"{goal_subset} {maze_name}")
            ax.set_xlabel("Steps to goal")
            ax.set_ylabel("Event-aligned time (s)")
    f.tight_layout()
    return


def get_step_time_transformation(session, event):
    """ """
    step_time_df = get_step_time_transformation_df()
    df = step_time_df.query(
        f"subject == '{session.subject_ID}' and goal_subset == '{session.goal_subset}' and maze == '{session.maze_name}' and event == '{event}'"
    )
    return df.set_index("event_aligned_time").steps_to_goal


def get_step_time_transformation_df(overwrite=False):
    """ """
    save_path = RESULTS_DIR / "step_time_transformation_df.csv"
    if save_path.exists() and not overwrite:
        return pd.read_csv(save_path, index_col=0)
    else:
        print("Generating step time transformation df")
        dfs = []
        for subject in SUBJECT_IDS:
            print(f"Processing subject {subject}")
            for maze in MAZE_NAMES:
                for goal_subset in GOAL_SETS:
                    for event in ["cue", "reward"]:
                        step_time_df = get_steps_vs_time_curve(subject, maze, goal_subset, event)
                        step_time_df["subject"] = subject
                        step_time_df["maze"] = maze
                        step_time_df["goal_subset"] = goal_subset
                        step_time_df["event"] = event
                        dfs.append(step_time_df)
        step_time_df = pd.concat(dfs).reset_index(drop=True)
        # save
        step_time_df.to_csv(save_path)
        return step_time_df


def get_steps_vs_time_curve(subject, maze, goal_subset, event, max_steps=30):
    sessions = get_sessions_for_analysis(subject_IDs=[subject], maze_names=[maze], goal_subsets=[goal_subset])
    dfs = []
    for session in sessions:
        df = get_event_aligned_input_data(session, event=event, resolution=0.5)
        df = df[[("event_aligned_time", event), ("steps_to_goal", "future")]]
        dfs.append(df)
    step_time_df = pd.concat(dfs).reset_index(drop=True).droplevel(1, axis=1)
    step_time_curve = step_time_df.groupby("event_aligned_time").steps_to_goal.mean()
    return step_time_curve[step_time_curve.index <= max_steps + 1].reset_index()
