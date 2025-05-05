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
from joblib import Parallel, delayed

from GridMaze.maze import representations as mr
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.distance_to_goal import decoding_utils as du
from GridMaze.analysis.distance_to_goal import goal_decoding as gd
from GridMaze.analysis.distance_to_goal import bases as db


# %% Global Variables

from GridMaze.paths import RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "distance_to_goal" / "goal_decoding_comparisons"

# %% Functions


def goal_decoding_comparison(
    session,
    resolution=0.5,
    include_multi_units=True,
    window=(-10, 10),
    goal_stratified_validation=True,
    n_test_trials=None,
    n_bases=8,
    basis_type="gamma",
    training_trial_phases=["navigation"],
    max_steps_to_goal=30,
    inv_alpha="auto",
    n_repeats=10,
    verbose=True,
):
    """
    CONDITION 1: spikes --(predict)--> goal
    CONDITION 2: spikes_by_distance --(predict)--> goal
    CONDITION 3: spikes --(predict)--> place_direction --(predict)--> goal (control)
    CONDITION 4: spikes --(predict)--> place_direction_by_distance --(predict)--> goal (control)


    Note deocders are trained on all data defined in training_trial_phases
    not separate decoders for each timepoint aligned to trial events
    """
    # get session object if strings input (when running jobs on HPC)
    if not isinstance(session, gs.MazeSession):
        if verbose:
            print(f"Getting session object for {session}")
        subject_ID, maze_name, day_on_maze = session
        session = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names=[maze_name],
            days_on_maze=[day_on_maze],
            with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "trials_df"],
            must_have_data=True,
        )
    # define conditions
    conditions = [
        "spikes",
        "spikes_by_distance",
        "place_direction",
        "place_direction_by_distance",
    ]
    # check if results already exist
    session_name = session.name
    save_paths = [RESULTS_DIR / session_name / f"{condition}.parquet" for condition in conditions]
    if all([path.exists() for path in save_paths]):
        print(f"Results already exist for {session_name}, skipping")
        return [pd.read_parquet(path) for path in save_paths]

    # else run analysis
    simple_maze = session.simple_maze()
    C_dfs = [[] for _ in conditions]  # store condition results here
    for n in range(n_repeats):
        # get downsampled input data containing behavioural info and spike data
        input_data = du.get_place_decoding_input_data(session, resolution, include_multi_units, window, permuted=False)
        # organise trials into test-train folds
        folds_df = du.get_folds_df(
            session, goal_stratified_validation, return_unique_IDs=True, n_test_trials=n_test_trials
        )
        # predict plce/place_direction probabilities from spike counts (for control conditions)
        if verbose:
            print(f"Predicting place_direction probabilities from spike counts")
        spatial_probs_df = get_predicted_spatial(
            input_data,
            folds_df,
            simple_maze,
            output_type="place_direction",
            inv_alpha=inv_alpha,
            training_trial_phases=training_trial_phases,
            verbose=verbose,
        )
        input_data = pd.concat([input_data, spatial_probs_df], axis=1)
        # get distance to goal basis functions (for spikes_by_distance condition)
        basis_fn = db.distance_basis_generator(
            n_bases=n_bases, basis=basis_type, max_steps=max_steps_to_goal, plot=False
        )
        if inv_alpha == "auto":
            # get optimal regularisation for each condition
            inv_alphas = []
            for condition, input_type in zip(
                conditions,
                [
                    "spikes",
                    "spikes_by_distance",
                    "place_direction_prob",
                    "place_direction_prob_by_distance",
                ],
            ):
                if verbose:
                    print(condition)
                inv_alphas.append(
                    get_opt_reg(
                        input_data,
                        folds_df["fold_0"],
                        simple_maze,
                        basis_fn,
                        input_type=input_type,
                        output_type="goal",
                        training_trial_phases=training_trial_phases,
                        eval_metric="expected_distance_error",
                    )
                )
        else:
            inv_alphas = [inv_alpha] * len(conditions)
        # run xvaled decoding for each condition aross folds
        folds = folds_df.columns.levels[0].unique()
        if verbose:
            print("Running condition decodings paralleised across folds")
        parallel_outputs = Parallel(n_jobs=len(folds), verbose=False)(
            delayed(_decode_fold_repeat)(
                fold,
                n,
                input_data,
                folds_df,
                conditions,
                basis_fn,
                inv_alphas,
                training_trial_phases,
                verbose,
            )
            for fold in folds
        )
        # parallel_outputs is a list of lists (fold × conditions), assign to C_dfs
        for cond_idx in range(len(conditions)):
            for fold_output in parallel_outputs:
                C_dfs[cond_idx].extend(fold_output[cond_idx])
        del parallel_outputs  # save memory
    # combine folds and repeats
    results_dfs = [pd.concat(_dfs, axis=0).reset_index(drop=True) for _dfs in C_dfs]
    # save results
    # for result_df, save_path in zip(results_dfs, save_paths):
    #     save_path.parent.mkdir(parents=True, exist_ok=True)
    #     result_df.to_parquet(save_path, index=False)
    #     if verbose:
    #         print(f"Saved results to {save_path}")
    return results_dfs


def _decode_fold_repeat(
    fold, repeat, input_data, folds_df, conditions, basis_fn, inv_alphas, training_trial_phases, verbose
):
    """
    Run decoding for a single fold & repeat. Returns a list of DataFrames,
    one per condition.
    """
    fold_df = folds_df[fold]
    C_dfs = [[] for _ in conditions]
    for cond_idx, (condition, input_type, inv_alpha) in enumerate(
        zip(
            conditions,
            ["spikes", "spikes_by_distance", "place_direction_prob", "place_direction_prob_by_distance"],
            inv_alphas,
        )
    ):
        if verbose:
            print(f"{fold}:{condition}")
        train_df, test_df = _get_test_train_dfs(input_data, fold_df, training_trial_phases=training_trial_phases)
        X_train, X_test, y_train, y_test = _get_test_train_arrays(
            train_df, test_df, input_type=input_type, output_type="goal", whiten_features=True, basis_fn=basis_fn
        )
        if inv_alpha is None:
            decoder = LogisticRegression(penalty=None, max_iter=10_000, random_state=0)
        else:
            decoder = LogisticRegression(penalty="l2", C=inv_alpha, max_iter=10_000, random_state=0)
        decoder.fit(X_train, y_train)
        Yprobs = decoder.predict_proba(X_test)
        C_df = _get_decoding_results_df(test_df, y_test, Yprobs, list(decoder.classes_), "goal", engine="pandas")
        C_df["fold"] = fold
        C_df["repeat"] = repeat
        C_dfs[cond_idx].append(C_df)
    return C_dfs


def quick_plot(results_dfs):
    acc_dfs = []
    for df in results_dfs:
        # predicted goal is max goal prob at each samle (ds window)
        acc_df = df.loc[df.groupby(["sample_index", "repeat"]).predicted_goal_prob.idxmax()]
        acc_df["test_acc"] = (df.true_goal == df.predicted_goal).astype(int)
        acc_dfs.append(acc_df)
    cue_dfs, reward_time_reps = [], []
    for df in acc_dfs:
        for event, dfs in zip(["cue", "reward"], [cue_dfs, reward_time_reps]):
            _df = df[~df[f"{event}_aligned_time"].isna()]
            trial_df = _df.groupby(["trial_unique_ID", f"{event}_aligned_time"]).test_acc.mean().unstack()
            dfs.append(trial_df.mean())
    cue_acc = pd.concat(cue_dfs, axis=1)
    cue_acc.plot()
    plt.show()
    reward_acc = pd.concat(reward_time_reps, axis=1)
    reward_acc.plot()
    plt.show()


def _get_decoding_acc_df_pl2(results_df, output_type):
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
    # return as pandas
    acc_df = acc_df.to_pandas()
    acc_df.sort_values(["sample_index", "repeat"], inplace=True)
    acc_df.reset_index(drop=True, inplace=True)
    return acc_df


# %%
def get_predicted_spatial(
    input_data,
    folds_df,
    simple_maze,
    output_type="place_direction",
    inv_alpha="auto",
    training_trial_phases=["navigation"],
    verbose=True,
):
    """
    From some input_data, and folds_df dataframes, preform cross-validated prediction
    of place_direction from spike counts (w/ Logisitic Rergression classifier).
    W/o stratification by distance to goal.

    Outputs the neural representation of place direction in the data as
    a probability distribution over the place directions.

    W/ automatic regularisation optimisation
    """
    if output_type == "place_direction":
        # precompute all place_directions ("A1_N")
        all_features = mr.get_maze_place_direction_pairs(simple_maze)
        all_features = ["_".join(x) for x in all_features]

        # add place_direction column to input_data
        input_data[("place_direction", "")] = input_data.apply(
            lambda x: f"{x[("maze_position", "simple")]}_{x[("cardinal_movement_direction", "")]}", axis=1
        )
    elif output_type == "place":
        all_features = mr.get_maze_locations(simple_maze)
    else:
        raise ValueError(f"Unknown output type {output_type!r}")

    # get x-val optimal regularisation
    if inv_alpha == "auto":
        if verbose:
            print("Auto-optimising regularisation")
        inv_alpha = get_opt_reg(
            input_data,
            folds_df["fold_0"],
            simple_maze,
            input_type="spikes",
            output_type=output_type,
            training_trial_phases=training_trial_phases,
            eval_metric="expected_distance_error",
        )
    # get x-valed place-direction prob from spikes on each input_data sample
    dfs = []
    for fold in folds_df.columns.levels[0].unique():
        if verbose:
            print(fold)
        train_df, test_df = _get_test_train_dfs(input_data, folds_df[fold], training_trial_phases)
        X_train, X_test, y_train, y_test = _get_test_train_arrays(
            train_df, test_df, input_type="spikes", output_type=output_type, whiten_features=True
        )
        if inv_alpha is None:
            decoder = LogisticRegression(penalty=None, max_iter=10_000, random_state=0, class_weight="balanced")
        else:
            decoder = LogisticRegression(
                penalty="l2", C=inv_alpha, max_iter=10_000, random_state=0, class_weight="balanced"
            )
        decoder.fit(X_train, y_train)
        Yprobs = decoder.predict_proba(X_test)
        features = list(decoder.classes_)
        probs_df = pd.DataFrame(
            index=test_df.index,
            columns=pd.MultiIndex.from_product([[f"{output_type}_prob"], features]),
            data=Yprobs,
        )
        # check for missing place_directions and add columns with value 0
        missing_features = set(all_features) - set(features)
        if len(missing_features) > 0:
            for missing_direction in missing_features:
                probs_df[(f"{output_type}_prob", missing_direction)] = 0
        dfs.append(probs_df.sort_index(axis=1))
    # combine folds and ensure index lines up with input_data
    probs_df = pd.concat(dfs, axis=0)
    probs_df.sort_index(axis=0, inplace=True)
    assert probs_df.index.equals(input_data.index)
    return probs_df


# %%


def get_opt_reg(
    input_data,
    fold_df,
    simple_maze=None,
    basis_fn=None,
    input_type="spikes",  # X
    output_type="place_direction",  # Y
    training_trial_phases=["navigation"],
    reg_range=[None, 1, 10, 50, 1e2, 5e2, 1e3],
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
    df = _get_decoding_results_df(test_df, y_test, Yprobs, features, output_type, engine="polars")

    if eval_metric == "expected_distance_error":
        values = []
        for event in ["cue", "reward"]:
            EDE_df = get_expected_distance_error_pl(
                df,
                simple_maze,
                op=eval_kwargs["op"],
                decoding_type=output_type,
                permuted=False,
                return_as="timeseries",
            )
            window = eval_kwargs[f"{event}_window"]
            dist_metric = eval_kwargs["dist_metric"]
            event_ede_df = EDE_df[~EDE_df[f"{event}_aligned_time"].isna()]
            av_ede = (
                event_ede_df.groupby(["trial_unique_ID", f"{event}_aligned_time"])[f"{dist_metric}_ede"]
                .mean()
                .unstack()
                .mean()
            )
            values.extend(av_ede[(av_ede.index > -2) & (av_ede.index < 2)].to_list())
        return np.mean(values)
    elif eval_metric == "decoding_accuracy":
        accs = []
        for event in ["cue", "reward"]:
            acc_df = decoding_accuracy_df_pl(
                df,
                decoding_type=output_type,
                alignment=f"{event}_aligned_time",
            )
            window = eval_kwargs[f"{event}_window"]
            accs.extend(acc_df[acc_df.cue_aligned_time.between(*window)].test_acc.to_list())
        return np.mean(accs)
    else:
        raise ValueError(f"Unknown eval metric {eval_metric!r}")


# %% new polars eval functions


def _get_decoding_results_df(test_df, y_test, Yprobs, features, output_type, engine="pandas"):
    """ """
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


def get_expected_distance_error_pl(
    results_df,
    simple_maze,
    op="sum",
    output_type="goal",
    permuted=False,
    return_as="timeseries",
):
    """
    input polars df (need speed from polars for processing large outputs from permuted decodings)
    output pandas df
    """
    # check output_type matches results df
    du._check_decoding_type(results_df, output_type)
    # add colums for distance to goal (geo or euc) for every true and predicted place/goal pair
    df = results_df.with_columns(_get_distance_cols_pl(results_df, simple_maze, output_type, round_euc=False))
    # calc weighted distance error
    df = df.with_columns(
        [
            (pl.col(f"predicted_{output_type}_prob") * pl.col("geo_dist")).alias("geo_weight_prob"),
            (pl.col(f"predicted_{output_type}_prob") * pl.col("euc_dist")).alias("euc_weight_prob"),
        ]
    )
    group_cols = ["sample_index", "permutation"] if permuted else ["sample_index"]
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
    info_df = (
        df.drop(
            [
                f"predicted_{output_type}",
                f"predicted_{output_type}_prob",
                "geo_dist",
                "euc_dist",
                f"geo_weight_prob",
                f"euc_weight_prob",
            ]
        )
        .unique()
        .sort(group_cols)
    )
    EDE_df = info_df.join(sample_EDE, on=group_cols, how="inner")
    if return_as == "timeseries":
        EDE_df = EDE_df.to_pandas()
        EDE_df.sort_values(["sample_index", "trial_unique_ID"], inplace=True)
        EDE_df.reset_index(drop=True, inplace=True)
        return EDE_df
    else:
        NotImplementedError


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
    elif input_type == "spikes_by_distance":
        Xs = []
        for df in [train_df, test_df]:
            basis_activations = basis_fn(df.steps_to_goal.future.values)
            spikes = df.spike_count.values
            spikes_by_distance = (
                spikes[:, :, None] * basis_activations[:, None, :]
            )  # [n_timepoints, n_neurons, n_bases]
            Xs.append(spikes_by_distance.reshape(spikes.shape[0], -1))
        X_train, X_test = Xs
    elif input_type == "place_prob_by_distance":
        Xs = []
        for df in [train_df, test_df]:
            basis_activations = basis_fn(df.steps_to_goal.future.values)
            place_probs = df.place_probs.values
            place_probs_by_distance = (
                place_probs[:, :, None] * basis_activations[:, None, :]
            )  # [n_timepoints, n_places, n_bases]
            Xs.append(place_probs_by_distance.reshape(place_direction_probs.shape[0], -1))
        X_train, X_test = Xs
    elif input_type == "place_direction_prob_by_distance":
        Xs = []
        for df in [train_df, test_df]:
            basis_activations = basis_fn(df.steps_to_goal.future.values)
            place_direction_probs = df.place_direction_prob.values
            place_direction_probs_by_distance = (
                place_direction_probs[:, :, None] * basis_activations[:, None, :]
            )  # [n_timepoints, n_place_directions, n_bases]
            Xs.append(place_direction_probs_by_distance.reshape(place_direction_probs.shape[0], -1))
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
