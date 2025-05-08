"""
Library for decoding subject location (place) as a function of distance to goal of event
aligned time. Uses util functions in ./decoding_utils.py
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.distance_to_goal import decoding_utils as du
from GridMaze.analysis.distance_to_goal import bases as db

# %% Global Variables
from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

RESULTS_DIR = RESULTS_PATH / "place_decoding"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "maze_day2date.json", "r") as input_file:
    MAZE_DAY2DATE = json.load(input_file)

# %% Plot summary figures


def quick_plot(df, axes=None, metric="geodesic_ede", cue_window=(-5, 10), reward_window=(-10, 5)):
    """ """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for permuted in [True, False]:
        _df = df[df.permuted == permuted]
        for event, ax, window in zip(["cue", "reward"], axes, [cue_window, reward_window]):
            _df = _df[_df[f"{event}_aligned_time"].between(*window)]
            trial_df = _df.groupby(["trial_unique_ID", f"{event}_aligned_time"])[metric].mean().unstack()
            mean = trial_df.mean()
            ax.plot(mean.index, mean.values, label=f"permuted={permuted}")
    axes[0].legend()


# %% Summary plot


def plot_place_decoding(
    summary_df,
    metric="geodesic_ede",
    chance_subtracted=True,
    cue_window=(-5, 10),
    reward_window=(-10, 5),
    axes=None,
):
    """ """
    # set up fig
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    for ax in axes:
        ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_ylabel(metric)
    # process
    for ax, event, window in zip(axes, ["cue", "reward"], [cue_window, reward_window]):
        for input_type in ["spikes", "spikes_by_distance"]:
            df = summary_df[(summary_df.event == event) & (summary_df.input_type == input_type)]
            subject_mean_df = df.groupby(["subject_ID", "aligned_time"])[["true", "permuted"]].mean().unstack()
            if chance_subtracted:
                subject_norm_df = subject_mean_df["true"] - subject_mean_df["permuted"]
                mean = subject_norm_df.mean()
                sem = subject_norm_df.sem()
                ax.plot(mean.index, mean.values, label=f"{input_type}")
                ax.fill_between(
                    mean.index,
                    mean.values - sem.values,
                    mean.values + sem.values,
                    alpha=0.2,
                )
            else:
                mean_df = subject_mean_df.mean()
                sem_df = subject_mean_df.sem()
                for c in ["true", "permuted"]:
                    mean = mean_df.loc[c]
                    sem = sem_df.loc[c]
                    ax.plot(mean.index, mean.values, label=f"{input_type}_{c}")
                    ax.fill_between(
                        mean.index,
                        mean.values - sem.values,
                        mean.values + sem.values,
                        alpha=0.2,
                    )
        ax.set_xlim(window)
        ax.set_xlabel(f"{event}(s)")
        if chance_subtracted:
            ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=8)

    return


# %% Summary df


def get_place_decoding_summary_df(
    output_type="place_direction",
    training_trial_phases="navigation",
    metric="geodesic_ede",
):
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
        for input_type in ["spikes", "spikes_by_distance"]:
            cue_aligned_perf, reward_aligned_perf = [], []
            for session in sessions:
                try:
                    decoding_df = run_session_place_decoding(
                        session,
                        input_type=input_type,
                        output_type=output_type,
                        training_trial_phases=training_trial_phases,
                        load_only=True,
                    )
                except FileNotFoundError as e:
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
                    dfs.append(df.groupby(["permuted", "trial_unique_ID", _time])[metric].mean().unstack())
            for event, df in zip(["cue", "reward"], [cue_aligned_perf, reward_aligned_perf]):
                _df = pd.concat(df, axis=0).sort_index()  # always False, True in permuted col
                ede_df = _df.groupby("permuted").mean().T.reset_index()
                ede_df.columns = ["aligned_time", "true", "permuted"]
                ede_df["subject_ID"] = subject_ID
                ede_df["event"] = event
                ede_df["input_type"] = input_type
                all_dfs.append(ede_df)
    summary_df = pd.concat(all_dfs, axis=0)
    return summary_df


# %% Decoding


def run_session_place_decoding(
    session,
    input_type="spikes",
    output_type="place_direction",
    training_trial_phases="navigation",
    n_true=10,
    n_permuted=10,
    verbose=True,
    load_only=False,
):
    """ """
    # check if session is a MazeSession object
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
    # update inputs
    if training_trial_phases == "navigation":
        _training_trial_phases = ["navigation"]
    elif training_trial_phases == "all":
        _training_trial_phases = ["navigation", "reward_consumption", "ITI"]
    else:
        raise ValueError("training_trial_phases must be 'navigation' or 'all'")
    # check if save path exists
    save_path = RESULTS_DIR / input_type / output_type / training_trial_phases / f"{session.name}.parquet"
    if save_path.exists():
        results_df = pd.read_parquet(save_path, engine="pyarrow", use_threads=True)
        return results_df
    else:
        if load_only:
            raise FileNotFoundError(f"File {save_path} does not exist. Set load_only=False to run decoding.")
    # get expected distance error (EDE) for true and permuted data
    true_metrics_df = get_place_decoding(
        session,
        input_type=input_type,
        output_type=output_type,
        n_repeats=n_true,
        training_trial_phases=_training_trial_phases,
        permuted=False,
        verbose=verbose,
    )
    permuted_metrics_df = get_place_decoding(
        session,
        input_type=input_type,
        output_type=output_type,
        n_repeats=n_permuted,
        training_trial_phases=_training_trial_phases,
        permuted=True,
        verbose=verbose,
    )
    # combine into one df
    results_df = pd.concat([true_metrics_df, permuted_metrics_df], axis=0)
    results_df.reset_index(drop=True, inplace=True)
    # save results
    if verbose:
        print(f"Saving results to {save_path}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(save_path, index=False)
    return results_df


def get_place_decoding(
    session,
    input_type="spikes",
    output_type="place_direction",
    resolution=0.5,
    include_multi_units=True,
    window=(-10, 10),
    goal_stratified_validation=True,
    n_test_trials=None,
    training_trial_phases=["navigation"],
    n_bases=8,
    basis_type="gamma",
    max_steps_to_goal=30,
    inv_alpha="auto",
    permuted=False,
    n_repeats=10,
    verbose=True,
):
    """ """
    # check input_types
    assert input_type in ["spikes", "spikes_by_distance"]
    assert output_type in ["place_direction", "place"]
    # load input data
    simple_maze = session.simple_maze()
    basis_fn = db.distance_basis_generator(n_bases=n_bases, basis=basis_type, max_steps=max_steps_to_goal, plot=False)
    all_repeat_dfs = []
    for n in range(n_repeats):
        input_data = du.get_place_decoding_input_data(
            session, resolution, include_multi_units, window, permuted=permuted
        )
        if output_type == "place_direction":
            input_data[("place_direction", "")] = input_data.apply(
                lambda x: f"{x[("maze_position", "simple")]}_{x[("cardinal_movement_direction", "")]}", axis=1
            )
        folds_df = du.get_folds_df(
            session, goal_stratified_validation, return_unique_IDs=True, n_test_trials=n_test_trials
        )
        # find optimal regularisation
        if inv_alpha == "auto":
            _inv_alpha = du.get_opt_reg(
                input_data,
                folds_df["fold_0"],
                simple_maze,
                basis_fn=basis_fn,
                input_type=input_type,
                output_type=output_type,
                training_trial_phases=training_trial_phases,
                eval_metric="expected_distance_error",
            )

        # get cross validated decoding across folds
        folds = folds_df.columns.levels[0].unique()
        fold_dfs = Parallel(n_jobs=len(folds), verbose=False)(
            delayed(_decode_place_fold)(
                n,
                fold,
                input_data,
                folds_df,
                input_type,
                output_type,
                _inv_alpha,
                training_trial_phases,
                basis_fn,
                verbose,
            )
            for fold in folds
        )
        repeat_df = pl.concat(fold_dfs, how="vertical")
        all_repeat_dfs.append(repeat_df)
    # now combine **all** repeats
    decoding_df = pl.concat(all_repeat_dfs, how="vertical")
    metrics_df = du.get_decoding_metrics_df(decoding_df, simple_maze, output_type=output_type)
    metrics_df["permuted"] = permuted
    return metrics_df


# %%
def _decode_place_fold(
    repeat,
    fold,
    input_data,
    folds_df,
    input_type,
    output_type,
    inv_alpha,
    training_trial_phases,
    basis_fn,
    verbose,
):
    """
    Decode one fold of place (or place_direction) for get_place_decoding.
    Returns a single Polars DataFrame.
    """
    if verbose:
        print(f"Decoding {fold}")
    fold_df = folds_df[fold]
    train_df, test_df = du._get_test_train_dfs(input_data, fold_df, training_trial_phases)
    X_train, X_test, y_train, y_test = du._get_test_train_arrays(
        train_df,
        test_df,
        input_type=input_type,
        output_type=output_type,
        whiten_features=True,
        basis_fn=basis_fn,
    )
    if inv_alpha is None:
        decoder = LogisticRegression(
            penalty=None, max_iter=10_000, random_state=0, class_weight="balanced", verbose=False
        )
    else:
        decoder = LogisticRegression(
            penalty="l2", C=inv_alpha, max_iter=10_000, random_state=0, class_weight="balanced", verbose=False
        )
    decoder.fit(X_train, y_train)
    Yprobs = decoder.predict_proba(X_test)
    features = list(decoder.classes_)
    df = du.get_decoding_results_df(test_df, y_test, Yprobs, features, output_type, engine="polars")
    df = df.with_columns([pl.lit(fold).alias("fold"), pl.lit(repeat).alias("repeat")])
    return df
