"""
New Lib for combining decoding analyses to see if goal decoding at cue improve when using decoders that know
about distance to goal while controlling for place coding in the neuronal population.
@peterdoohan
"""

# %% Imports

import json
import numpy as np
import pandas as pd
import polars as pl
import networkx as nx
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from GridMaze.maze import representations as mr
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.distance_to_goal import place_decoding as dp
from GridMaze.analysis.distance_to_goal import decoding_utils as du
from GridMaze.analysis.distance_to_goal import goal_decoding as gd
from GridMaze.analysis.distance_to_goal import bases as db


# %% Global Variables


# %% Functions


def test(
    session,
    resolution=0.5,
    include_multi_units=True,
    window=(-10, 10),
    goal_stratified_validation=True,
    n_test_trials=None,
    n_bases=4,
    basis_type="gamma",
    training_trial_phases=["navigation"],
    training_steps_to_goal_range=None,
):
    """ """
    input_data = du.get_place_decoding_input_data(session, resolution, include_multi_units, window, permuted=False)
    folds_df = du.get_folds_df(session, goal_stratified_validation, return_unique_IDs=True, n_test_trials=n_test_trials)
    return input_data, folds_df
    # load basis functions

    results_dfs = []
    for fold in folds_df.columns.levels[0].unique():
        print(fold)
        fold_df = folds_df[fold]
        test_trials = [t for t in fold_df.test.values.flatten() if isinstance(t, str)]
        train_trials = [t for t in fold_df.train.values.flatten() if isinstance(t, str)]
        train_df = input_data[input_data.trial_unique_ID.isin(train_trials)]

    return


def get_predicted_place_directions(
    input_data, folds_df, training_trial_phases=["navigation"], whiten_features=True, verbose=False
):
    """
    From some input_data, and folds_df dataframes, preform cross-validated prediction
    of place_direction from spike counts (w/ Logisitic Rergression classifier).
    W/o stratification by distance to goal.

    Outputs the neural representation of place direction in the data as
    a probability distribution over the place directions.

    W/ automatic regularisation optimisation
    """
    input_data[("place_direction", "")] = input_data.apply(
        lambda x: f"{x[("maze_position", "simple")]}_{x[("cardinal_movement_direction", "")]}", axis=1
    )
    for fold in folds_df.columns.levels[0].unique():
        if verbose:
            print(fold)
        train_df, test_df = _get_test_train_dfs(input_data, folds_df[fold], training_trial_phases)
        X_train, y_train = train_df.spike_count.values, train_df.place_direction.values
        X_test, y_test = test_df.spike_count.values, test_df.maze_position.simple.values
        if whiten_features:
            scaler = StandardScaler()  # mean=0, std=1 per column
            scaler.fit(X_train)  # learn stats on train
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

    return


def get_opt_reg(
    input_data,
    fold_df,
    simple_maze=None,
    basis_fn=None,
    input_type="spikes",  # X
    output_type="place_direction",  # Y
    training_trial_phases=["navigation"],
    reg_range=[None, 10, 50, 1e2, 5e2, 1e3],
    eval_metric="expected_distance_error",
    eval_kwargs={
        "op": "sum",
        "dist_metric": "geodesic",
        "cue_window": (-2, 2),
        "reward_window": (-8, 0),
    },
):
    """
    Returns optimal regularisation parameter for a logistic regression decoder
    --- Notes ---
    input_type: ["spikes", "spikes_by_distance", "place_direction", "place"]
    output_type: ["place_direction", "goal"]
    """
    # process input data to X,y arrays to feed into decoder
    train_df, test_df = _get_test_train_dfs(input_data, fold_df, training_trial_phases)
    X_train, X_test, y_train, y_test = _get_test_train_arrays(train_df, test_df, input_type, output_type, basis_fn)
    eval_metrics = []
    for inv_alpha in reg_range:
        if inv_alpha is None:
            decoder = LogisticRegression(penalty=None, max_iter=10_000, random_state=0, class_weight="balanced")
        else:
            decoder = LogisticRegression(
                penalty="l2", C=inv_alpha, max_iter=10_000, random_state=0, class_weight="balanced"
            )
        decoder.fit(X_train, y_train)
        Yprobs = decoder.predict_proba(X_test)
        features = list(decoder.classes_)
        df = _get_decoding_results_df(test_df, y_test, Yprobs, features, output_type)
        if eval_metric == "expected_distance_error":
            cue_EDE_df, reward_EDE_edf = [
                get_expected_distance_error_pl(
                    df,
                    simple_maze,
                    op=eval_kwargs["op"],
                    decoding_type=output_type,
                    alignment=f"{event}_aligned_time",
                    permuted=False,
                    return_total_av=True,
                )[eval_kwargs["dist_metric"]]
                for event in ["cue", "reward"]
            ]
            M = np.concatenate(
                [
                    _df[(_df.index < window[1]) & (_df.index > window[0])].values
                    for _df, window in zip(
                        [cue_EDE_df, reward_EDE_edf], [eval_kwargs["cue_window"], eval_kwargs["reward_window"]]
                    )
                ]
            ).mean()

        elif eval_metric == "accuracy":
            NotImplementedError
        else:
            raise ValueError(f"Unknown eval metric {eval_metric!r}")

        eval_metrics.append(M)

    # return best regularisation parameter
    eval_metrics = np.array(eval_metrics)
    if eval_metric == "expected_distance_error":
        opt_reg = reg_range[np.argmin(eval_metrics)]
    elif eval_metric == "accuracy":
        opt_reg = reg_range[np.argmax(eval_metrics)]
    else:
        raise ValueError(f"Unknown eval metric {eval_metric!r}")
    return opt_reg


# %% new polars eval functions


def _get_decoding_results_df(test_df, y_test, Yprobs, features, output_type):
    n_samples, n_features = Yprobs.shape
    df = pl.DataFrame(  # note use of polars df (big output dfs need something faster than pandas)
        {
            "cue_aligned_time": np.repeat(test_df.event_aligned_bin["cue"].values, n_features),
            "reward_aligned_time": np.repeat(test_df.event_aligned_bin["reward"].values, n_features),
            "steps_to_goal": np.repeat(test_df.steps_to_goal.future.values, n_features),
            "trial_phase": np.repeat(test_df.trial_phase.values, n_features),
            f"true_{output_type}": np.repeat(y_test, n_features),
            "trial_unique_ID": np.repeat(test_df.trial_unique_ID.values, n_features),
            f"predicted_{output_type}": np.tile(features, n_samples),
            f"predicted_{output_type}_prob": Yprobs.ravel(),
        }
    )
    return df


def get_expected_distance_error_pl(
    results_df,
    simple_maze,
    op="sum",
    decoding_type="goal",
    alignment="timepoint",
    permuted=False,
    return_total_av=True,
):
    """
    input polars df (need speed from polars for processing large outputs from permuted decodings)
    output pandas df
    """
    # check decoding_type matches results df
    du._check_decoding_type(results_df, decoding_type)
    # add colums for distance to goal (geo or euc) for every true and predicted place/goal pair
    df = results_df.with_columns(_get_distance_cols_pl(results_df, simple_maze, decoding_type, round_euc=False))
    # calc weighted distance error
    df = df.with_columns(
        [
            (pl.col(f"predicted_{decoding_type}_prob") * pl.col("geo_dist")).alias("geo_weight_prob"),
            (pl.col(f"predicted_{decoding_type}_prob") * pl.col("euc_dist")).alias("euc_weight_prob"),
        ]
    )

    # pick your grouping keys
    group_cols = ["trial_unique_ID", "permutation", alignment] if permuted else ["trial_unique_ID", alignment]

    # aggregate per‐trial (sum or max)
    if op == "sum":
        trial_EDE = df.group_by(group_cols, maintain_order=True).agg(
            [
                pl.sum("geo_weight_prob").alias("geo_weight_prob"),
                pl.sum("euc_weight_prob").alias("euc_weight_prob"),
            ]
        )
    elif op == "max":
        trial_EDE = df.group_by(group_cols, maintain_order=True).agg(
            [
                pl.max("geo_weight_prob").alias("geo_weight_prob"),
                pl.max("euc_weight_prob").alias("euc_weight_prob"),
            ]
        )
    else:
        raise ValueError(f"Unsupported op: {op!r}")

    if not return_total_av:
        # pivot alignment values out to columns
        pivoted = trial_EDE.pivot(
            values=["geo_weight_prob", "euc_weight_prob"],
            index="trial_unique_ID",
            on=alignment,
        )
        # rename outer “geo_weight_prob → geodesic”, “euc_weight_prob → euclidean”
        rename_map = {
            c: c.replace("geo_weight_prob", "geodesic").replace("euc_weight_prob", "euclidean")
            for c in pivoted.columns
            if c != "trial_unique_ID"
        }

        pd_df = pivoted.rename(rename_map).to_pandas()
        pd_df.set_index("trial_unique_ID", inplace=True)
        # switch to multi-index
        pd_df.columns = pd.MultiIndex.from_tuples(
            [(c.split("_")[0], eval(c.split("_")[1])) if c.split("_")[1] != "NaN" else np.nan for c in pd_df.columns]
        )
        # remove NaN columns
        pd_df = pd_df[pd_df.columns[~pd_df.columns.get_level_values(0).isna()]]
        return pd_df

    # 4b) otherwise return the average EDE across trials, grouped by alignment
    av_EDE = trial_EDE.group_by(alignment, maintain_order=True).agg(
        [
            pl.mean("geo_weight_prob").alias("geodesic"),
            pl.mean("euc_weight_prob").alias("euclidean"),
        ]
    )
    pd_df = av_EDE.to_pandas()
    pd_df.set_index(alignment, inplace=True)
    # remove NaN columns
    pd_df = pd_df.loc[~pd_df.index.isna()]
    return pd_df


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


# %% pre-decoiding utils


def _get_test_train_dfs(input_data, fold_df, training_trial_phases=["navigation"]):
    """ """
    test_trials = [t for t in fold_df.test.values.flatten() if isinstance(t, str)]
    train_trials = [t for t in fold_df.train.values.flatten() if isinstance(t, str)]
    train_df = input_data[input_data.trial_unique_ID.isin(train_trials)]
    # include only specified trial phases in training data
    train_df = train_df[train_df.trial_phase.isin(training_trial_phases)]
    test_df = input_data[input_data.trial_unique_ID.isin(test_trials)]
    return train_df, test_df


def _get_test_train_arrays(train_df, test_df, input_type="spikes", output_type="goal", basis_fn=None):
    """ """
    # process input data (X)
    if input_type == "spikes":
        X_train, X_test = train_df.spike_count.values, test_df.spike_count.values
    elif input_type == "place_direction":
        NotImplementedError
    elif input_type == "place":
        NotImplementedError
    elif input_type == "spikes_by_distance":
        assert basis_fn is not None, "basis_fn must be provided for 'spikes_by_distance' input"
        Xs = []
        for df in [train_df, test_df]:
            basis_activations = basis_fn(df.steps_to_goal.future.values)
            spikes = df.spike_count.values
            spikes_by_distance = (
                spikes[:, :, None] * basis_activations[:, None, :]
            )  # [n_timepoints, n_neurons, n_bases]
            Xs.append(spikes_by_distance.reshape(spikes.shape[0], -1))
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

    return X_train, X_test, y_train, y_test
