"""
New analysis to test generalisation of goal decoding from different inputs (spikes or spikes_by_distance)
from training data that only includes half of the places on the maze.
"""

# %% Imports
import json
import pandas as pd
import polars as pl
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed

from GridMaze.maze import partitions as mt
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import folds
from GridMaze.analysis.distance_to_goal import decoding_utils as du
from GridMaze.analysis.distance_to_goal import bases as db

# %% Global Variables
from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

RESULTS_DIR = RESULTS_PATH / "distance_to_goal" / "place_generalised_goal_decoding"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

# %% Load results


def plot_place_generalised_goal_decoding(
    summary_df,
    metric="test_acc",
    cue_window=(-5, 10),
    reward_window=(-10, 5),
    chance=1 / 12,
    axes=None,
):
    # set up fig
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    for ax in axes:
        ax.axvline(0, color="k", linestyle="--", alpha=0.5)
        ax.axhline(chance, color="k", linestyle="--", alpha=0.5)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_ylabel(metric)
    # process
    for ax, event, window in zip(axes, ["cue", "reward"], [cue_window, reward_window]):
        df = summary_df[summary_df.event == event]
        subject_mean_df = df.groupby(["subject_ID", "aligned_time"])[["spikes", "spikes_by_distance"]].mean().unstack()
        mean_df = subject_mean_df.mean()
        sem_df = subject_mean_df.sem()
        for i in ["spikes", "spikes_by_distance"]:
            mean = mean_df.loc[i]
            sem = sem_df.loc[i]
            ax.plot(mean.index, mean.values, label=i)
            ax.fill_between(
                mean.index,
                mean.values - sem.values,
                mean.values + sem.values,
                alpha=0.2,
            )
        ax.set_xlim(window)
        ax.set_ylim(0, 0.2)
        ax.set_xlabel(f"{event}(s)")
    axes[0].legend(fontsize=8)


def get_decoding_summary_df(metric="test_acc", n_partitions=4, resolution=0.4):
    """ """
    all_dfs = []
    for subject_ID in SUBJECT_IDS:
        print(subject_ID)
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names="all",
            days_on_maze="late",
            goal_subsets=["subset_1", "subset_2"],
            with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "trials_df"],
            must_have_data=True,
        )
        cue_aligned_perf, reward_aligned_perf = [], []
        for s in sessions:
            try:
                decoding_df = run_place_generalised_goal_decoding(
                    s, n_partitions=n_partitions, resolution=resolution, verbose=False, load_only=True
                )
            except Exception as e:
                print(e)
                continue
            for event, dfs in zip(["cue", "reward"], [cue_aligned_perf, reward_aligned_perf]):
                _time = f"{event}_aligned_time"
                if event == "cue":
                    df = decoding_df[~decoding_df[_time].isna()]
                    df = df[  # only include ITI before cue and navigation time after cue
                        ((df[_time].le(0)) & (df.trial_phase == "ITI"))
                        | (df[_time].gt(0)) & (df.trial_phase == "navigation")
                    ]
                else:  # reward
                    df = decoding_df[~decoding_df[_time].isna()]
                    df = df[
                        ((df[_time].gt(0)) & (df.trial_phase == "reward_consumption"))
                        | (df[_time].le(0)) & (df.trial_phase == "navigation")
                    ]
                dfs.append(df.groupby(["input_type", "trial_unique_ID", _time])[metric].mean().unstack())
        for event, df in zip(["cue", "reward"], [cue_aligned_perf, reward_aligned_perf]):
            _df = pd.concat(df, axis=0).sort_index()
            mean_df = _df.groupby("input_type").mean().T.reset_index()
            mean_df = mean_df.rename(columns={f"{event}_aligned_time": "aligned_time"})
            mean_df["subject_ID"] = subject_ID
            mean_df["event"] = event
            all_dfs.append(mean_df)
    summary_df = pd.concat(all_dfs, axis=0)
    return summary_df


# %% Populate results


def populate_decoding_results(subject_ID):
    """ """
    sessions = gs.get_maze_sessions(
        subject_IDs=[subject_ID],
        maze_names="all",
        days_on_maze="late",
        goal_subsets=["subset_1", "subset_2"],
        with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "trials_df"],
        must_have_data=True,
    )

    def _run_with_exception(resolution, n_partitions):
        try:
            run_place_generalised_goal_decoding(session, resolution=resolution, n_partitions=n_partitions)
        except Exception as e:
            print(f"Error processing session {session}: {e}")
            pass

    for session in sessions:
        print(session)
        for resolution in [0.2, 0.4]:
            for n_partitions in [3, 4]:
                _run_with_exception(resolution, n_partitions)
    return


# %% Main functions


def run_place_generalised_goal_decoding(
    session,
    input_types=["spikes", "spikes_by_distance"],
    n_partitions=4,
    resolution=0.4,
    include_multi_units=True,
    window=(-10, 10),
    goal_stratified_validation=True,
    n_test_trials=None,
    n_bases=8,
    basis_type="gamma",
    training_trial_phases=["navigation"],
    training_steps_tol=1,
    max_steps_to_goal=30,
    inv_alpha="auto",
    verbose=True,
    n_repeats=10,
    max_jobs=10,
    load_only=False,
):
    """
    n_partitions: number of partitions to split the maze into for cross-space decoding
    (see GridMaze.maze.paritions.get_AB_split)
    """
    # check input types (if tuples convert to session obj)
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

    # check if results already saved on disk
    session_name = session.name
    save_path = RESULTS_DIR / f"{n_partitions}x{n_partitions}" / f"res_{resolution}" / f"{session_name}.parquet"
    if save_path.exists():
        if verbose:
            print(f"Loading results for {session_name} from disk")
        return pd.read_parquet(save_path)
    else:
        if load_only:
            raise FileNotFoundError(f"Results for {session_name} not found on disk")

    # MAIN ANALYSIS
    # Prepare shared resources
    input_data = du.get_place_decoding_input_data(session, resolution, include_multi_units, window, permuted=False)
    basis_fn = db.distance_basis_generator(
        n_bases=n_bases,
        basis=basis_type,
        max_steps=max_steps_to_goal,
    )
    simple_maze = session.simple_maze()
    A_locs, B_locs = mt.get_AB_split(simple_maze, s=n_partitions)

    all_results = []
    for r in range(n_repeats):
        if verbose:
            print(f"Repeat {r + 1} of {n_repeats}")
        folds_df = folds.get_folds_df(
            session,
            goal_stratified_validation,
            return_unique_IDs=True,
            n_test_trials=n_test_trials,
        )

        _folds = folds_df.columns.get_level_values(0).unique()

        if inv_alpha == "auto":
            if verbose:
                print("Calculating optimal regularisation parameter for each input type")
            inv_alphas = [
                du.get_opt_reg(
                    input_data,
                    fold_df=folds_df["fold_0"],
                    simple_maze=simple_maze,
                    basis_fn=basis_fn,
                    input_type=itype,
                    output_type="goal",
                    training_trial_phases=training_trial_phases,
                    eval_metric="expected_distance_error",
                    verbose=verbose,
                )
                for itype in input_types
            ]
        else:
            inv_alphas = [inv_alpha] * len(input_types)

        # Parallel processing across folds
        n_jobs = min([len(folds), max_jobs])
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_fold)(
                f,
                r,
                input_data,
                folds_df,
                A_locs,
                B_locs,
                input_types,
                inv_alphas,
                basis_fn,
                training_trial_phases,
                tol=training_steps_tol,
                verbose=verbose,
            )
            for f in _folds
        )
        all_results.extend(results)

    flat_results = [df for fold_res in all_results for df in fold_res]
    results_df = pl.concat(flat_results, how="vertical")
    metrics_df = du.get_decoding_metrics_df(results_df, simple_maze, output_type="goal")
    # save results to disk
    save_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_parquet(save_path, index=False)
    if verbose:
        print(f"Results saved to {save_path}")
    return metrics_df


def _process_fold(
    fold,
    repeat,
    input_data,
    folds_df,
    A_locs,
    B_locs,
    input_types,
    inv_alphas,
    basis_fn,
    training_trial_phases,
    tol=1,
    verbose=False,
):
    """
    For each input_type, fits separate decoders for data in A and B maze paritions to predict goal location.
    Decoders must are then tested on the opposite partition in held-out test trials.
    """
    if verbose:
        print(f"{repeat}-{fold}")
    # Extract fold-specific split
    fold_df = folds_df[fold]
    train_df, test_df = folds._get_test_train_dfs(input_data, fold_df, training_trial_phases=training_trial_phases)

    # Partition into A and B regions
    train_A = train_df[train_df.maze_position.simple.isin(A_locs)]
    train_B = train_df[train_df.maze_position.simple.isin(B_locs)]
    test_A = test_df[test_df.maze_position.simple.isin(A_locs)]
    test_B = test_df[test_df.maze_position.simple.isin(B_locs)]

    # Ensure valid training for cross-space decoding
    test_A = has_training_data(train_B, test_A, tol=tol)
    test_B = has_training_data(train_A, test_B, tol=tol)

    fold_results = []
    for itype, inv_alpha in zip(input_types, inv_alphas):
        # Build arrays for decoding
        Xa_t, Xa_e, ya_t, ya_e = du._get_test_train_arrays(
            train_A,
            test_A,
            input_type=itype,
            output_type="goal",
            whiten_features=True,
            basis_fn=basis_fn,
        )
        Xb_t, Xb_e, yb_t, yb_e = du._get_test_train_arrays(
            train_B,
            test_B,
            input_type=itype,
            output_type="goal",
            whiten_features=True,
            basis_fn=basis_fn,
        )

        # Fit decoders
        if inv_alpha is not None:
            A_dec = LogisticRegression(
                penalty="l2", C=inv_alpha, max_iter=10000, random_state=0, class_weight="balanced"
            )
            B_dec = LogisticRegression(
                penalty="l2", C=inv_alpha, max_iter=10000, random_state=0, class_weight="balanced"
            )
        else:
            A_dec = LogisticRegression(penalty=None, max_iter=10000, random_state=0, class_weight="balanced")
            B_dec = LogisticRegression(penalty=None, max_iter=10000, random_state=0, class_weight="balanced")
        A_dec.fit(Xa_t, ya_t)
        B_dec.fit(Xb_t, yb_t)

        # run place-generalised test decoding
        for dec, dtype, test_df_loc, X_e, y_e in zip(
            [A_dec, B_dec],
            ["A", "B"],
            [test_B, test_A],  # cross-space: A-dec on B-test, B-dec on A-test
            [Xb_e, Xa_e],
            [yb_e, ya_e],
        ):
            probs = dec.predict_proba(X_e)
            df = du.get_decoding_results_df(
                test_df_loc, y_e, probs, features=list(dec.classes_), output_type="goal", engine="polars"
            )
            df = df.with_columns(
                [
                    pl.lit(itype).alias("input_type"),
                    pl.lit(fold).alias("fold"),
                    pl.lit(dtype).alias("decoder_type"),
                    pl.lit(repeat).alias("repeat"),
                ]
            )
            fold_results.append(df)

    return fold_results


# %% supporting functions
def has_training_data(train_df, test_df, tol=1):
    """
    Remove observations from test data (test_df), if there are no examples of those observations (same goal and
    distance to goal (within some tolerance: tol) in the training data (train_df).
    """
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
    input_types = df.input_type.unique()
    for itype in input_types:
        itype_df = df[df.input_type == itype]
        for event, ax, window in zip(["cue", "reward"], axes, [cue_window, reward_window]):
            _df = itype_df[itype_df[f"{event}_aligned_time"].between(*window)]
            trial_df = _df.groupby(["trial_unique_ID", f"{event}_aligned_time"])[metric].mean().unstack()
            mean = trial_df.mean()
            ax.plot(mean.index, mean.values, label=itype)
    ax.legend(fontsize=8)


# %%
