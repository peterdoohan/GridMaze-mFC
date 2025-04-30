"""
Library for decoding subject location (place) as a function of distance to goal of event
aligned time. Uses util functions in ./decoding_utils.py
@peterdoohan
"""

# %% Imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler

from GridMaze.analysis.core import get_sessions as gs
from . import decoding_utils as du
from . import bases as db

# %% Global Variables
from GridMaze.paths import RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "place_decoding"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %% Plotting functions


def plot_session_place_decoding(
    results_df, simple_maze, dist_type="geodesic", axes=None, cue_window=(-4, 10), reward_window=(-10, 4), ymax=15
):
    """ """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(4, 2), sharey=True)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.axvline(0, color="k", ls="--", alpha=0.5)
    axes[0].set_ylabel("Expected distance error")
    # expected distance error (ede)
    true_results = results_df[results_df.permutation.isna()]
    permuted_results = results_df[~results_df.permutation.isna()]
    # plot
    for ax, event, window in zip(
        axes,
        ["cue", "reward"],
        [cue_window, reward_window],
    ):
        # plot true
        true_ede = du.get_expected_distance_error_df(
            true_results.copy(),
            simple_maze,
            decoding_type="place",
            alignment=f"{event}_aligned_time",
            permuted=False,
            return_total_av=True,
        )[dist_type]
        ax.plot(
            true_ede.index.values,
            true_ede.values,
            color="k",
            lw=2,
        )
        # plot chance
        perm_ede = du.get_expected_distance_error_df(
            permuted_results.copy(),
            simple_maze,
            decoding_type="place",
            alignment=f"{event}_aligned_time",
            permuted=True,
            return_total_av=False,
        )[dist_type]
        perm_av_ede = perm_ede.groupby("permutation").mean()
        p_mean = perm_av_ede.mean().values
        p_sem = perm_av_ede.sem().values

        ax.fill_between(
            true_ede.index.values,
            p_mean - p_sem,
            p_mean + p_sem,
            color="k",
            alpha=0.2,
        )
        ax.set_xlim(window)
        ax.set_xlabel(f"{event} (s)")
        ax.set_ylim(0, ymax)


# %% Decoding function


def run_session_place_decoding(
    session,
    n_chance=10,
    training_trial_phases=["navigation"],
    verbose=True,
):
    """
    Runs place decoding on a session on true data and on permuted data n_chance times where spikes are circularly shifted
    relative to subject's position/place.
    """
    if not isinstance(session, gs.MazeSession):  # optional input as tuple of strings for HPC
        subject_ID, maze_name, day_on_maze = session
        session = gs.get_maze_sessions(
            [subject_ID],
            [maze_name],
            [day_on_maze],
            with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "trials_df"],
            must_have_data=True,
        )
    # check if session has already been run
    save_path = RESULTS_DIR / ".".join(training_trial_phases) / session.name
    if save_path.exists():
        results_df = pd.read_parquet(save_path)
    else:
        # generate true results
        if verbose:
            print("Running non-permuted decoding")
        true_results_df = get_place_decoding(session, training_trial_phases=training_trial_phases, permuted=False)
        true_results_df["permutation"] = np.nan
        # generate permuted results
        if verbose:
            print("Running permuted decodings")
        permuted_dfs = []
        for i in range(n_chance):
            if verbose:
                print(i)
            permuted_results_df = get_place_decoding(
                session,
                training_trial_phases=training_trial_phases,
                permuted=True,
            )
            permuted_results_df["permutation"] = i
            permuted_dfs.append(permuted_results_df)
        # combine into one df
        results_df = pd.concat([true_results_df] + permuted_dfs, axis=0)
        results_df.reset_index(drop=True, inplace=True)
        # save results
        save_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(save_path, index=False, compression="gzip")
    return results_df


def get_place_decoding(
    session,
    resolution=0.5,
    include_multi_units=True,
    window=(-10, 10),
    goal_stratified_validation=True,
    n_test_trials=None,
    training_trial_phases=["navigation", "reward_consumption", "ITI"],
    training_steps_to_goal_range=None,
    whiten_features=True,
    permuted=False,
):
    """ """
    input_data = du.get_place_decoding_input_data(session, resolution, include_multi_units, window, permuted=permuted)
    folds_df = du.get_folds_df(session, goal_stratified_validation, return_unique_IDs=True, n_test_trials=n_test_trials)
    results_dfs = []
    for fold in folds_df.columns.levels[0].unique():
        fold_df = folds_df[fold]
        test_trials = [t for t in fold_df.test.values.flatten() if isinstance(t, str)]
        train_trials = [t for t in fold_df.train.values.flatten() if isinstance(t, str)]
        train_df = input_data[input_data.trial_unique_ID.isin(train_trials)]
        # include only specified trial phases in training data
        train_df = train_df[train_df.trial_phase.isin(training_trial_phases)]
        # include only specified steps to goal in training data (check how this works with NaNs in other trial phases)
        if training_steps_to_goal_range is not None:
            train_df = train_df[train_df.steps_to_goal.future.between(*training_steps_to_goal_range)]
        test_df = input_data[input_data.trial_unique_ID.isin(test_trials)]
        X_train, y_train = train_df.spike_count.values, train_df.maze_position.simple.values
        X_test, y_test = test_df.spike_count.values, test_df.maze_position.simple.values
        if whiten_features:
            scaler = StandardScaler()  # mean=0, std=1 per column
            scaler.fit(X_train)  # learn stats on train
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        decoder = LogisticRegression(
            penalty=None, max_iter=10_000, random_state=0, class_weight="balanced", verbose=False
        )
        decoder.fit(X_train, y_train)
        Pprobs = decoder.predict_proba(X_test)
        n_samples, n_places = Pprobs.shape
        places = list(decoder.classes_)
        df = pd.DataFrame(
            {
                "cue_aligned_time": np.repeat(test_df.event_aligned_bin["cue"].values, n_places),
                "reward_aligned_time": np.repeat(test_df.event_aligned_bin["reward"].values, n_places),
                "steps_to_goal": np.repeat(test_df.steps_to_goal.future.values, n_places),
                "trial_phase": np.repeat(test_df.trial_phase.values, n_places),
                "true_place": np.repeat(y_test, n_places),
                "trial_unique_ID": np.repeat(test_df.trial_unique_ID.values, n_places),
                "predicted_place": np.tile(places, n_samples),
                "predicted_place_prob": Pprobs.ravel(),
            }
        )
        df["fold"] = fold
        results_dfs.append(df)
    results_df = pd.concat(results_dfs, axis=0)
    results_df.reset_index(drop=True, inplace=True)
    return results_df


def get_event_aligned_place_deocoding(
    session,
    event="cue",
    resolution=0.5,
    window=(-10, 10),
    goal_stratified_validation=False,
    n_test_trials=6,
    include_multi_units=True,
    whiten_features=True,
):
    """ """
    input_data = du.get_event_aligned_input_data(session, event, resolution, window, include_multi_units)
    timepoints = sorted(input_data.event_aligned_time[event].unique())
    folds_df = du.get_folds_df(session, goal_stratified_validation, return_unique_IDs=True, n_test_trials=n_test_trials)
    results_dfs = []
    for fold in folds_df.columns.levels[0].unique():
        fold_df = folds_df[fold]
        test_trials = [t for t in fold_df.test.values.flatten() if isinstance(t, str)]
        train_trials = [t for t in fold_df.train.values.flatten() if isinstance(t, str)]
        train_df = input_data[input_data.trial_unique_ID.isin(train_trials)]
        test_df = input_data[input_data.trial_unique_ID.isin(test_trials)]
        decoder = LogisticRegression(penalty=None, max_iter=10000, random_state=0, class_weight="balanced")
        for t in timepoints:
            _train_df = train_df[train_df.event_aligned_time[event] == t]
            _test_df = test_df[test_df.event_aligned_time[event] == t]
            if _train_df.empty or _test_df.empty:
                continue  # rare cases when no trials for that timepoint (eg, end of session trial)
            X_train, y_train = _train_df.spike_count.values, _train_df.maze_position.simple.values
            X_test, y_test = _test_df.spike_count.values, _test_df.maze_position.simple.values
            if whiten_features:  # zscore features
                scaler = StandardScaler()  # mean=0, std=1 per column
                scaler.fit(X_train)  # learn stats on train
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
            decoder.fit(X_train, y_train)
            # out_df
            Gprobs = decoder.predict_proba(X_test)
            n_samples, n_places = Gprobs.shape
            places = list(decoder.classes_)
            df = pd.DataFrame(
                {
                    "timepoint": np.repeat(t, n_samples * n_places),
                    "true_place": np.repeat(y_test, n_places),
                    "trial_unique_ID": np.repeat(_test_df.trial_unique_ID.values, n_places),
                    "predicted_place": np.tile(places, n_samples),
                    "predicted_place_prob": Gprobs.ravel(),
                }
            )
            df["fold"] = fold
            results_dfs.append(df)
    results_df = pd.concat(results_dfs, axis=0)
    results_df.reset_index(drop=True, inplace=True)
    return results_df
