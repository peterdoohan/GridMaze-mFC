"""
New analysis to test generalisation of goal decoding from different inputs (spikes or spikes_by_distance)
from training data that only includes half of the places on the maze.
"""

# %% Imports
import numpy as np
import pandas as pd
import polars as pl
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed


from GridMaze.maze import partitions as mt
from GridMaze.analysis.distance_to_goal import decoding_utils as du
from GridMaze.analysis.distance_to_goal import bases as db

# %% Global Variables


# %% Main functions


def _process_fold(
    fold,
    input_data,
    folds_df,
    simple_maze,
    A_locs,
    B_locs,
    basis_fn,
    training_trial_phases,
    tol=1,
):
    # Extract fold-specific split
    fold_df = folds_df[fold]
    train_df, test_df = du._get_test_train_dfs(input_data, fold_df, training_trial_phases=training_trial_phases)

    # Partition into A and B regions
    train_A = train_df[train_df.maze_position.simple.isin(A_locs)]
    train_B = train_df[train_df.maze_position.simple.isin(B_locs)]
    test_A = test_df[test_df.maze_position.simple.isin(A_locs)]
    test_B = test_df[test_df.maze_position.simple.isin(B_locs)]

    # Ensure valid training for cross-space decoding
    test_A = has_training_data(train_B, test_A, tol=tol)
    test_B = has_training_data(train_A, test_B, tol=tol)

    # Build arrays for decoding
    Xa_t, Xa_e, ya_t, ya_e = du._get_test_train_arrays(
        train_A,
        test_A,
        input_type="spikes_by_distance",
        output_type="goal",
        whiten_features=True,
        basis_fn=basis_fn,
    )
    Xb_t, Xb_e, yb_t, yb_e = du._get_test_train_arrays(
        train_B,
        test_B,
        input_type="spikes_by_distance",
        output_type="goal",
        whiten_features=True,
        basis_fn=basis_fn,
    )

    # Fit decoders
    A_dec = LogisticRegression(penalty="l2", C=10, max_iter=10000, random_state=0)
    B_dec = LogisticRegression(penalty="l2", C=10, max_iter=10000, random_state=0)
    A_dec.fit(Xa_t, ya_t)
    B_dec.fit(Xb_t, yb_t)

    # Collect results
    fold_results = []
    for dec, dtype, test_df_loc, X_e, y_e in zip(
        [A_dec, B_dec],
        ["A", "B"],
        [test_B, test_A],  # cross-space: A-dec on B-test, B-dec on A-test
        [Xb_e, Xa_e],
        [yb_e, ya_e],
    ):
        probs = dec.predict_proba(X_e)
        df = du.get_decoding_results_df(
            test_df_loc, y_e, probs, classes=list(dec.classes_), output_type="goal", engine="polars"
        )
        df = df.with_columns(
            [
                pl.lit(fold).alias("fold"),
                pl.lit(dtype).alias("decoder_type"),
                pl.lit(0).alias("repeat"),
            ]
        )
        fold_results.append(df)

    return fold_results


def test(
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
    s_ab_split=3,
    verbose=True,
    n_jobs=-1,
):
    # Prepare shared resources
    input_data = du.get_place_decoding_input_data(session, resolution, include_multi_units, window, permuted=False)
    folds_df = du.get_folds_df(
        session,
        goal_stratified_validation,
        return_unique_IDs=True,
        n_test_trials=n_test_trials,
    )
    basis_fn = db.distance_basis_generator(
        n_bases=n_bases,
        basis=basis_type,
        max_steps=max_steps_to_goal,
    )
    maze = session.simple_maze()
    A_locs, B_locs = mt.get_AB_split(maze, s=s_ab_split)

    folds = folds_df.columns.get_level_values(0).unique()

    # Parallel processing across folds
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(_process_fold)(
            fold,
            input_data,
            folds_df,
            maze,
            A_locs,
            B_locs,
            basis_fn,
            training_trial_phases,
        )
        for fold in folds
    )

    # Flatten results and concatenate
    decoding_results = [df for fold_res in all_results for df in fold_res]
    results_df = pl.concat(decoding_results, how="vertical")

    # Compute metrics
    metrics_df = du.get_decoding_metrics_df(results_df, maze, output_type="goal")
    return metrics_df


def test(
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
    s_ab_split=3,
    verbose=True,
):
    """"""
    # get input data
    input_data = du.get_place_decoding_input_data(
        session,
        resolution,
        include_multi_units,
        window,
        permuted=False,
    )
    folds_df = du.get_folds_df(
        session,
        goal_stratified_validation,
        return_unique_IDs=True,
        n_test_trials=n_test_trials,
    )
    basis_fn = db.distance_basis_generator(
        n_bases=n_bases,
        basis=basis_type,
        max_steps=max_steps_to_goal,
    )
    simple_maze = session.simple_maze()
    A_locs, B_locs = mt.get_AB_split(
        simple_maze,
        s=s_ab_split,
    )
    # get xvaled and place generalised goal decodign across folds
    folds = folds_df.columns.get_level_values(0).unique()
    decoding_results_dfs = []
    for fold in folds:
        if verbose:
            print(fold)
        fold_df = folds_df[fold]
        # cross validate across trials
        train_df, test_df = du._get_test_train_dfs(
            input_data,
            fold_df,
            training_trial_phases=training_trial_phases,
        )

        # cross validate across space (A, B maze partitions)
        train_A_df, train_B_df = (
            train_df[train_df.maze_position.simple.isin(A_locs)],
            train_df[train_df.maze_position.simple.isin(B_locs)],
        )
        test_A_df, test_B_df = (
            test_df[test_df.maze_position.simple.isin(A_locs)],
            test_df[test_df.maze_position.simple.isin(B_locs)],
        )

        # check there is valid training data for the test data (obs of that goal at that distance with some tol)
        test_A_df = has_training_data(train_B_df, test_A_df, tol=1)
        test_B_df = has_training_data(train_A_df, test_B_df, tol=1)

        Xa_train, Xa_test, ya_train, ya_test = du._get_test_train_arrays(
            train_A_df,
            test_A_df,
            input_type="spikes_by_distance",
            output_type="goal",
            whiten_features=True,
            basis_fn=basis_fn,
        )
        Xb_train, Xb_test, yb_train, yb_test = du._get_test_train_arrays(
            train_B_df,
            test_B_df,
            input_type="spikes_by_distance",
            output_type="goal",
            whiten_features=True,
            basis_fn=basis_fn,
        )

        # init separate decoders (no reg for init testing)
        A_decoder = LogisticRegression(penalty="l2", C=10, max_iter=10_000, random_state=0)
        B_decoder = LogisticRegression(penalty="l2", C=10, max_iter=10_000, random_state=0)

        # fit the decoders
        A_decoder.fit(Xa_train, ya_train)
        B_decoder.fit(Xb_train, yb_train)

        for (
            decoder,
            decoder_type,
            test_df,
            X_test,
            y_test,
        ) in zip([A_decoder, B_decoder], ["A", "B"], [test_B_df, test_A_df], [Xb_test, Xa_test], [yb_test, ya_test]):
            Yprobs = decoder.predict_proba(X_test)
            df = du.get_decoding_results_df(
                test_df, y_test, Yprobs, list(decoder.classes_), output_type="goal", engine="polars"
            )
            df = df.with_columns(
                [pl.lit(fold).alias("fold"), pl.lit(decoder_type).alias("decoder_type"), pl.lit(0).alias("repeat")]
            )
            decoding_results_dfs.append(df)
    results_df = pl.concat(decoding_results_dfs, how="vertical")
    metrics_df = du.get_decoding_metrics_df(results_df, simple_maze, output_type="goal")
    return metrics_df


# %% supporting functions
def has_training_data(train_df, test_df, tol=1):
    """ """
    goal_dist_cols = [("goal", ""), ("steps_to_goal", "future")]
    training_obs = set(train_df[goal_dist_cols].apply(tuple, axis=1).values)
    test_obs = set(test_df[goal_dist_cols].apply(tuple, axis=1).values)
    expanded_training_obs = list(training_obs)
    for t in range(tol):
        expanded_training_obs.extend([(i[0], i[1] + (t + 1)) for i in training_obs])
        expanded_training_obs.extend([(i[0], i[1] - (t + 1)) for i in training_obs if i[1] - (t + 1) >= 0])
    expanded_training_obs = set(expanded_training_obs)
    no_training_obs = test_obs - expanded_training_obs
    # mask for test obs without training obs
    mask = test_df[[("goal", ""), ("steps_to_goal", "future")]].apply(tuple, axis=1).isin(no_training_obs)
    return test_df[~mask]


def quick_plot(df, axes=None, metric="test_acc", cue_window=(-5, 10), reward_window=(-10, 5)):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    for event, ax, window in zip(["cue", "reward"], axes, [cue_window, reward_window]):
        _df = df[df[f"{event}_aligned_time"].between(*window)]
        trial_df = _df.groupby(["trial_unique_ID", f"{event}_aligned_time"])[metric].mean().unstack()
        mean = trial_df.mean()
        ax.plot(mean.index, mean.values)


# %%
