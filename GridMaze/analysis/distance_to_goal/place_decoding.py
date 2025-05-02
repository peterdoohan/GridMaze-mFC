"""
Library for decoding subject location (place) as a function of distance to goal of event
aligned time. Uses util functions in ./decoding_utils.py
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed

from GridMaze.analysis.core import get_sessions as gs
from . import decoding_utils as du
from . import bases as db

# %% Global Variables
from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

RESULTS_DIR = RESULTS_PATH / "place_decoding"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "maze_day2date.json", "r") as input_file:
    MAZE_DAY2DATE = json.load(input_file)

# %% Plot summary figures


def plot_place_decoding(results_df, distance_metric="geodesic", cue_window=(-8, 8), reward_window=(-8, 8), axes=None):
    """ """
    # set up fig
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(4, 2), sharey=True)
    for ax in axes:
        ax.axvline(0, color="k", ls="--", alpha=0.5)
        ax.axhline(0, color="k", ls="--", alpha=0.5)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_ylabel("Expected distance error")
    # plot cross subject expected distance error
    df = results_df[results_df.distance_metric == distance_metric]
    subject_means = df.groupby(["event", "timepoint", "permuted", "subject_ID"]).ede.mean()
    subject_grouped = subject_means.groupby(["event", "timepoint", "permuted"])
    mean = subject_grouped.mean().unstack()
    sem = subject_grouped.sem().unstack()
    timepoints = mean.index.get_level_values(1).unique().values
    for event, window, ax in zip(["cue", "reward"], [cue_window, reward_window], axes):
        for permuted, color in zip([True, False], ["k", "r"]):
            # plot mean and sem
            _mean = mean.loc[event, :][permuted]
            _sem = sem.loc[event, :][permuted]
            ax.plot(timepoints, _mean, color=color, lw=2)
            ax.fill_between(
                timepoints,
                _mean - _sem,
                _mean + _sem,
                color=color,
                alpha=0.2,
            )
        ax.set_xlim(window)
        ax.set_xlabel(f"{event} (s)")


# %%


def get_place_decoding_summary_df(
    maze_names="all",
    goal_subsets="all",
    days_on_maze="late",
    training_trial_phases="navigation",
    verbose=True,
    overwrite=False,
    n_jobs=5,
):
    save_path = RESULTS_DIR / f"place_decoding_summary_{training_trial_phases}.csv"
    if save_path.exists() and not overwrite:
        return pd.read_csv(save_path, index_col=0)

    # determine phases
    if training_trial_phases == "all":
        ttp = ["navigation", "reward_consumption", "ITI"]
    elif training_trial_phases == "navigation":
        ttp = ["navigation"]
    else:
        raise NotImplementedError(f"Unknown phase: {training_trial_phases}")

    # gather all sessions up front
    all_sessions = gs.get_maze_sessions(
        subject_IDs=SUBJECT_IDS,
        maze_names=maze_names,
        days_on_maze=days_on_maze,
        goal_subsets=goal_subsets,
        with_data=["cluster_metrics", "navigation_df"],
        must_have_data=True,
    )

    EDE_dfs = []
    # parallel dispatch
    with ProcessPoolExecutor(max_workers=n_jobs) as exe:
        futures = {exe.submit(_process_decoding_results, session, ttp): session for session in all_sessions}
        for fut in as_completed(futures):
            session = futures[fut]
            try:
                if verbose:
                    print(f"Finished: {session.name}")
                EDE_dfs.extend(fut.result())
            except Exception as e:
                print(f"Error processing {session.name}: {e}")

    # concat & save
    EDE_df = pd.concat(EDE_dfs, axis=0).reset_index(drop=True)
    EDE_df.to_csv(save_path)
    return EDE_df


def _process_decoding_results(session, ttp):
    """Compute EDE-dfs for a single session."""
    simple_maze = session.simple_maze()
    results_df = run_session_place_decoding(session, training_trial_phases=ttp)
    true_results = results_df[results_df.permutation.isna()]
    permuted_results = results_df[~results_df.permutation.isna()]
    session_dfs = []
    for event in ("cue", "reward"):
        valid_phases = ["ITI", "navigation"] if event == "cue" else ["navigation", "reward_consumption"]
        _true = true_results[true_results.trial_phase.isin(valid_phases)]
        _perm = permuted_results[permuted_results.trial_phase.isin(valid_phases)]
        for df, permuted in ((_true, False), (_perm, True)):
            ede = du.get_expected_distance_error_df(
                df.copy(),
                simple_maze,
                decoding_type="place",
                alignment=f"{event}_aligned_time",
                permuted=permuted,
                return_total_av=True,
                op="max",
            )
            ede_df = (
                ede.unstack()
                .reset_index()
                .rename(columns={"level_0": "distance_metric", "level_1": "timepoint", 0: "ede"})
            )
            ede_df["permuted"] = permuted
            ede_df["event"] = event
            ede_df["subject_ID"] = session.subject_ID
            ede_df["maze_name"] = session.maze_name
            ede_df["day_on_maze"] = session.day_on_maze
            ede_df["trial_subset"] = session.goal_subset
            session_dfs.append(ede_df)
    return session_dfs


# %%


# %% single session plotting functions


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
        session_name = f"{subject_ID}.{MAZE_DAY2DATE[maze_name][str(day_on_maze)]}.maze"
    else:
        session_name = session.name
    # check if session has already been run
    save_path = RESULTS_DIR / ".".join(training_trial_phases) / f"{session_name}.parquet.gzip"
    if save_path.exists():
        results_df = pd.read_parquet(save_path, engine="pyarrow", use_threads=True)
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


# %% Decoding


def get_place_decoding(
    session,
    resolution=0.5,
    include_multi_units=True,
    window=(-10, 10),
    goal_stratified_validation=True,
    n_test_trials=None,
    training_trial_phases=["navigation"],
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
