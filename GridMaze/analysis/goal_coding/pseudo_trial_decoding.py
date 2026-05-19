"""This module contains funtions to decoding navigational goals from time-aligned population activity"""

# %% Imports
from ensurepip import bootstrap
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed
from sklearn.neural_network import MLPClassifier
from GridMaze.analysis.goal_coding import mlp_utils as mu
from statsmodels.stats.multitest import multipletests
import seaborn as sns

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc

# %% Global variables

RATES_SAMPLE_RATE = 0.04

from GridMaze.paths import EXPERIMENT_INFO_PATH, ANALYSIS_INFO_PATH, RESULTS_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(ANALYSIS_INFO_PATH / "intra_trial_interval_times.json", "r") as f:
    INTRA_TRIAL_INTERVAL_TIMES = json.load(f)

RESULTS_DIR = RESULTS_PATH / "goal_coding"

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]

GOAL_SETS = ["subset_1", "subset_2", "all"]

DECODER2KWARGS = {
    "logreg": {"inv_alpha": 0.01},  # hyperparameters optimsed with fn: run_hyperparameter_search
    "mlp": {"alpha": 1, "Nhid": (50,), "solver": "adam"},
    "mlp_torch": {"alpha": 1e-2, "Nhid": (25, 25)},  # hyperparameters optimsed with fn: run_hyperparameter_search
}

WINDOW_SIZE = 0.2  # from HP search
SMOOTH_SD = 4

# %% Decoding heatmap summary plot


def plot_goal_decoding_heatmap_summary(alignment="trial", decoder="logreg", ax=None, cmap="Greens"):
    """ """
    decoding_accs, sig_df, columns = [], [], []
    for goal_subset in GOAL_SETS:
        for maze_name in MAZE_NAMES:
            # load results
            try:
                if goal_subset == "all":
                    chance = 1 / 24
                    df = _load_single_subject_results(decoder, maze_name, goal_subset, alignment)
                else:  # subset_1 and subset_2
                    chance = 1 / 12
                    df = _load_permuted_results(decoder, maze_name, goal_subset, alignment)
                # keep mean decoding acc
                decoding_accs.append(df.mean(axis=0))
                # keep timepoints significantly above chance
                timepoint_pvalues = 1 - df.gt(chance).mean(0)
                reject, pvals_corrected, _, _ = multipletests(timepoint_pvalues, alpha=0.05, method="fdr_bh", maxiter=1)
                sig_df.append(pd.Series(reject, index=df.columns))
                columns.append((goal_subset, maze_name))
            except Exception:
                print(f"Error loading results for: {goal_subset}, {maze_name}, {alignment}, {decoder}")
    summary_df = pd.concat(decoding_accs, axis=1).T
    summary_df.index = pd.MultiIndex.from_tuples(columns)
    above_chance_df = pd.concat(sig_df, axis=1).T
    above_chance_df.index = pd.MultiIndex.from_tuples(columns)
    # plotting
    if alignment == "trial":
        _plot_trial_aligned_heatmap_summary(summary_df, above_chance_df, ax, cmap)
    elif alignment == "event":
        _plot_event_aligned_heatmap_summary(summary_df, above_chance_df, ax, cmap)


def _plot_trial_aligned_heatmap_summary(summary_df, above_chance_df, ax=None, cmap="Greens"):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(4, 2), clear=True)
    sns.heatmap(
        summary_df,
        cmap=cmap,
        vmin=0,
        vmax=1,
        ax=ax,
        rasterized=True,
        cbar_kws={"label": "Decoding Acc."},
        mask=~above_chance_df,
    )
    event_times = list(INTRA_TRIAL_INTERVAL_TIMES.values())[:-1]
    timepoints = [float(col) for col in summary_df.columns]
    event_inds = [np.argmin(np.abs(np.array(timepoints) - time)) for time in event_times]
    for ind in event_inds:
        ax.axvline(ind, color="k", ls="--", alpha=0.5)
    ax.set_xticks(event_inds)
    ax.set_xticklabels(["Cue", "Reward", "ITI"], rotation=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(0.5)
    ax.set_ylabel("Dataset")


def _plot_event_aligned_heatmap_summary(summary_df, above_chance_df, axes=None, cmap="Greens"):
    """ """
    if axes is None:
        f, axes = plt.subplots(1, 2, figsize=(4, 2), clear=True, width_ratios=[0.8, 1])
    for ax, event in zip(axes, ["cue_aligned", "reward_aligned"]):
        df = summary_df[event]
        mask = ~above_chance_df[event]
        cbar = True if event == "reward_aligned" else False
        sns.heatmap(
            df,
            cmap=cmap,
            vmin=0,
            vmax=1,
            ax=ax,
            rasterized=True,
            cbar=cbar,
            cbar_kws={"label": "Decoding Acc."},
            mask=mask,
        )
        times = df.columns.values.astype(float)
        zero_point = np.argmin(np.abs(times))
        ax.axvline(zero_point, color="k", ls="--", alpha=0.5)
        tick_labels = [-5, 0, 5]
        tick_positions = [np.argmin(np.abs(times - tick)) for tick in tick_labels]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0)
        ax.set_xlabel(f"{event} time (s)")
        if event == "reward_aligned":
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ylabel("")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("black")
                spine.set_linewidth(0.5)
        else:
            ax.set_ylabel("Dataset")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("black")
                spine.set_linewidth(0.5)


#  %% top level results plotting functions


def plot_goal_decoding(
    datasets=[("maze_1", "subset_1")],
    alignment="trial",
    decoder="logreg",
    bootstrap_method="input",
    color="purple",
    sig_color="orchid",
    sig_pos=1.0,
    ax=None,
):
    """
    Plots allocentric goal decoding results
    Inputs:
        datasets: list of tuples, each tuple contains maze_name and goal_subset.
                  if multiple maze-goal pairs are provided, results will be averaged across datasets
        alignment: str, either "trial" or "event"
        decoder: str, either "logreg" or "mlp"
        bootstrap_method: str, either "input" or "output"
                if "input", bootstraps over subjects input to the full analysis (loaded from permtuted results)
                if "output", bootstraps over the output of the analysis (loaded from single subject results)
    """
    perm_dfs = []
    for dataset in datasets:
        maze_name, goal_subset = dataset
        if goal_subset == "all":
            assert bootstrap_method == "output", "results for goal_set 'all' only generated for single subject decoding"
        if bootstrap_method == "input":
            perm_dfs.append(_load_permuted_results(decoder, maze_name, goal_subset, alignment))
        elif bootstrap_method == "output":
            perm_dfs.append(_load_single_subject_results(decoder, maze_name, goal_subset, alignment))
        else:
            NotImplementedError
    # combine data across datasets
    perm_dfs = pd.concat(perm_dfs, axis=0).reset_index(drop=True)
    # plot results
    chance = 1 / 24 if goal_subset == "all" else 1 / 12
    if alignment == "trial":
        _plot_trial_aligned_decoding_acc(
            perm_dfs, ax=ax, chance=chance, color=color, sig_color=sig_color, sig_pos=sig_pos
        )
    elif alignment == "event":
        _plot_event_aligned_decoding_acc(
            perm_dfs, axes=ax, chance=chance, color=color, sig_color=sig_color, sig_pos=sig_pos
        )


def _load_permuted_results(
    decoder,
    maze_name,
    goal_subset,
    alignment,
    min_permutations=490,
):
    """
    Can take a bit to load all permutations from disk
    """
    perm_dir = RESULTS_DIR / "permutation_results" / alignment / decoder / maze_name / goal_subset
    perm_files = list(perm_dir.glob("*.csv"))
    # check that expected n permutations matches (i.e job has finished running)
    if len(perm_files) < min_permutations:
        raise FileNotFoundError(f"Expected {min_permutations} permutations, found {len(perm_files)}")
    # load all permutations into df
    accs = []
    for i, f in enumerate(perm_files):
        if alignment == "trial":
            df = pd.read_csv(f, index_col=[0, 1])
        else:
            df = pd.read_csv(f, index_col=[0, 1, 2])
        # ignore training accuracies
        df = df.xs("test", level=-1, axis=0)
        # mean acc across folds
        accs.append(df.mean(1))
    perm_df = pd.concat(accs, axis=1).T
    return perm_df


def _load_single_subject_results(
    decoder,
    maze_name,
    goal_subset,
    alignment,
    verbose=True,
    n_bootstraps=1000,
):
    """
    Loads the results of all single subject decoding across folds, averages across folds and
    bootstraps over subjects to get a distribution of accuracies for each timepoint.
    """
    subject_IDs = SUBJECT_IDS
    results_dir = RESULTS_DIR / "single_subject_decoding" / alignment / decoder / maze_name / goal_subset
    results_files = list(results_dir.glob("*.csv"))
    if verbose:  # missing results for a few subjects in some instances where no valid sessions
        subjects = set([r.name.split(".")[0] for r in results_files])
        if subjects != set(SUBJECT_IDS):
            print(f"Missing results for some subjects: {set(SUBJECT_IDS) - subjects}")
            subject_IDs = list(subjects)
    subject2acc = {}
    for f in results_files:
        subject = f.name.split(".")[0]
        if alignment == "trial":
            df = pd.read_csv(f, index_col=[0, 1])
        else:
            df = pd.read_csv(f, index_col=[0, 1, 2])
        # ignore training accuracies
        df = df.xs("test", level=-1, axis=0)
        # mean acc across folds
        subject2acc[subject] = df.mean(1)
    # bootstrap resample av subject accuracies
    rng = np.random.default_rng()
    subject_perms = rng.choice(
        subject_IDs,
        size=(n_bootstraps, len(subject_IDs)),
        replace=True,
    )
    accs = []
    for i in subject_perms:
        resampled_accs = pd.concat([subject2acc[subject] for subject in i], axis=1)
        accs.append(resampled_accs.mean(axis=1))
    acc_df = pd.concat(accs, axis=1).T
    return acc_df


def _plot_trial_aligned_decoding_acc(
    perm_df,
    ax=None,
    chance=1 / 12,
    color="purple",
    sig_color="orchid",
    sig_pos=1.00,
):
    """
    perm_df: pd.DataFrame, shape =[n_permutations, n_timepoints]
    """
    # set up fig
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(5, 3), clear=True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Acc.")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-5, INTRA_TRIAL_INTERVAL_TIMES["ITI_end"] + 0.5)
    ax.set_xticks(list(INTRA_TRIAL_INTERVAL_TIMES.values()))
    ax.set_xticklabels(["cue", "reward", "erc", "end"])
    for time in INTRA_TRIAL_INTERVAL_TIMES.values():
        ax.axvline(time, color="black", linestyle="--", alpha=0.5)
    ax.axhline(chance, color="black", linestyle="--", alpha=0.5)
    # plot acc
    time = perm_df.columns.values.astype(float)
    mean_acc = perm_df.mean(axis=0)
    std_acc = perm_df.std(axis=0).values
    ax.plot(time, mean_acc, color=color, lw=2)
    # plot std across permutations ~= sem across subjects
    ax.fill_between(time, mean_acc - std_acc, mean_acc + std_acc, color=color, alpha=0.2)
    # plot significance
    timepoint_pvalues = 1 - perm_df.gt(chance).mean(0)
    reject, pvals_corrected, _, _ = multipletests(timepoint_pvalues, alpha=0.05, method="fdr_bh", maxiter=1)
    sig_timepoints = time[reject]
    if len(sig_timepoints) > 1:
        ax.scatter(sig_timepoints, np.ones(len(sig_timepoints)) * sig_pos, marker="s", color=sig_color, s=5)


def _plot_event_aligned_decoding_acc(
    perm_df,
    axes=None,
    chance=1 / 12,
    color="purple",
    sig_color="orchid",
    sig_pos=1.00,
):
    """ """
    # set up fig
    if axes is None:
        f, axes = plt.subplots(1, 2, figsize=(6, 3), clear=True)
    for ax, label in zip(axes, ["Cue", "Reward"]):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel(label)
        ax.set_ylim(0, 1)
        ax.set_xlim(-10, 10)
        ax.axhline(chance, color="black", linestyle="--", alpha=0.5)
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Decoding Acc.")
    axes[1].spines["left"].set_visible(False)
    axes[1].set_yticks([])
    # plot acc
    mean_acc = perm_df.mean(axis=0)
    std_acc = perm_df.std(axis=0)
    for ax, event in zip(axes, ["cue_aligned", "reward_aligned"]):
        event_acc = mean_acc.loc[event]
        event_time = event_acc.index.values.astype(float)
        ax.plot(event_time, event_acc.values, color=color, lw=2)
        # plot error
        event_std = std_acc.loc[event].values
        ax.fill_between(event_time, event_acc - event_std, event_acc + event_std, color=color, alpha=0.2)
        # plot significance
        timepoint_pvalues = 1 - perm_df[event].gt(chance).mean(0)
        reject, pvals_corrected, _, _ = multipletests(timepoint_pvalues, alpha=0.05, method="fdr_bh", maxiter=1)
        sig_timepoints = event_time[reject]
        if len(sig_timepoints) > 1:
            ax.scatter(sig_timepoints, np.ones(len(sig_timepoints)) * sig_pos, marker="s", color=sig_color, s=5)


# %% run bootstrapped permutation tests


def run_bootstrapped_allocentric_goal_deocding(
    maze_name, goal_subset, alignment="trial", decoder="mlp_torch", n_permutations=4, n_jobs=4
):
    """ """
    # set up somewhere to save results bc. this is going to take a while
    save_dir = RESULTS_DIR / "permutation_results" / alignment / decoder / maze_name / goal_subset
    save_dir.mkdir(parents=True, exist_ok=True)
    # check how many permutations have already been run
    existing_files = list(save_dir.glob("*.csv"))
    completed_permutations = len(existing_files)
    remaining_permutations = n_permutations - completed_permutations
    # load sessions once
    subject2sessions = {}
    for subject in SUBJECT_IDS:
        subject2sessions[subject] = get_sessions_for_analysis([subject], maze_name, goal_subset, alignment)
    # generate session list permutations bootstrapping over subjects
    rng = np.random.default_rng()
    subject_perms = rng.choice(
        SUBJECT_IDS,
        size=(remaining_permutations, len(SUBJECT_IDS)),
        replace=True,
    )
    session_perms = [
        [session for subject in subjects for session in subject2sessions[subject]] for subjects in subject_perms
    ]
    # get save paths for each permutation
    save_paths = [save_dir / f"perm_{i}.csv" for i in range(completed_permutations, n_permutations)]
    Parallel(n_jobs=n_jobs)(
        delayed(_run_allocentric_goal_decoding)(
            session_perms[i],
            alignment=alignment,
            include_multi_units=True,
            decoder=decoder,
            window_size=WINDOW_SIZE,
            smooth_SD=SMOOTH_SD,
            save_path=save_paths[i],
        )
        for i in range(remaining_permutations)
    )


def _run_allocentric_goal_decoding(
    sessions,
    alignment="trial",
    include_multi_units=True,
    decoder="logreg",  # or "mlp"
    window_size=0.2,
    smooth_SD=4,
    save_path=None,
):
    decoder_kwargs = DECODER2KWARGS[decoder]
    activity_df = get_activity_df(sessions, alignment, include_multi_units, window_size, smooth_SD)
    validation_fold_df = get_validation_folds_df(sessions, n_training_trial_sets=15, alignment=alignment)
    results_df = _get_decoding_accurary(
        activity_df,
        validation_fold_df,
        classifier=decoder,
        decoder_kwargs=decoder_kwargs,
        verbose=True,
    )
    if save_path is not None:
        results_df.to_csv(save_path)
    else:
        return results_df


# %% allocentric goal decoding (non-bootstrapped)


def run_allocentric_goal_decoding(
    subject_IDs=["m2"],
    maze_name="maze_1",
    goal_subset="all",
    alignment="trial",
    include_multi_units=True,
    decoder="logreg",  # or "mlp"
    decoder_kwargs={"inv_alpha": 0.01},  # {"alpha": 1, "Nhid": (50,), "solver": "adam"} for mlp
    window_size=0.2,
    smooth_SD=4,
    plot=False,
):
    """ """
    sessions = get_sessions_for_analysis(subject_IDs, maze_name, goal_subset, alignment)
    if len(sessions) == 0:
        return print(f"No valid sessions found for specified input: {subject_IDs}, {maze_name}, {goal_subset}")
    activity_df = get_activity_df(sessions, alignment, include_multi_units, window_size, smooth_SD)
    validation_fold_df = get_validation_folds_df(sessions, n_training_trial_sets=50, alignment=alignment)
    results_df = _get_decoding_accurary(activity_df, validation_fold_df, decoder, decoder_kwargs, verbose=True)
    if plot:
        if alignment == "event":
            plot_event_aligned_decoding_results(results_df)
        elif alignment == "trial":
            plot_trial_aligned_decoding_results(results_df)
    return results_df


def get_sessions_for_analysis(subject_IDs, maze_name, goal_subset, alignment, verbose=False):
    """ """
    subject_IDs = SUBJECT_IDS if subject_IDs == "all" else subject_IDs
    data = ["event_aligned_rates_df", "trial_aligned_rates_df", "cluster_metrics"]
    sessions = []  # loop over subjects so we can add the same session twice when running bootstrapped permutation tests
    days_on_maze = "late" if goal_subset == "all" else "all"
    for subject in subject_IDs:
        s = gs.get_maze_sessions(
            subject_IDs=[subject],
            maze_names=[maze_name],
            days_on_maze=days_on_maze,
            goal_subsets=[goal_subset],
            with_data=data,
        )
        s = [s] if isinstance(s, gs.MazeSession) == 1 else s
        sessions.extend(s)
    # check sessions have enough trials (>=2) per goal for test-train split (cannot combine data across sessions)
    keep_sessions = []
    for session in sessions:
        trials_per_goal = get_trials_per_goal(session, alignment)
        if trials_per_goal.apply(len).lt(2).any():
            if verbose:
                print(f"{session} has some goals have less than 2 valid trials")
            continue
        else:
            keep_sessions.append(session)
    return keep_sessions


def get_activity_df(sessions, alignment="event", include_multi_units=True, window_size=0.2, smooth_SD=4):
    """"""
    data_struc = "event_aligned_rates_df" if alignment == "event" else "trial_aligned_rates_df"
    # combine data across sessions w/ select units
    session_count = {
        session.name: 0 for session in sessions
    }  # how many times has this session been added to activity df (relevant when bootstrap resampling)
    aligned_rates_dfs = []
    for session in sessions:
        session_name = session.name
        count = session_count[session_name]
        event_aligned_rates_df = getattr(session, data_struc)
        keep_clusters = gc.filter_clusters(
            session.cluster_metrics,
            session.session_info,
            return_unique_IDs=True,
            single_units=True,
            multi_units=include_multi_units,
        )
        session_df = event_aligned_rates_df[event_aligned_rates_df.cluster_unique_ID.isin(keep_clusters)].copy()
        # update cluster_unique_IDs with session counts (avoid non-uniquenss when boostrap resampling)
        cuID_loc = ("cluster_unique_ID", "", "") if alignment == "event" else ("cluster_unique_ID", "")
        session_df.loc[:, cuID_loc] = session_df.cluster_unique_ID.apply(lambda x: f"{x}_c{count}")
        # add trial_unique_ID column (with guarantees for uniqueness when bootstrap resampling)
        tuID_loc = ("trial_unique_ID", "", "") if alignment == "event" else ("trial_unique_ID", "")
        session_df.loc[:, tuID_loc] = (
            session_df[["subject_ID", "maze_name", "day_on_maze", "trial"]]
            .astype(str)
            .agg("_".join, axis=1)
            .apply(lambda x: f"{x}_c{count}")  # same IDs as in validation folds df
        )
        aligned_rates_dfs.append(session_df)
        session_count[session_name] += 1
    aligned_rates_df = pd.concat(aligned_rates_dfs, axis=0)

    # reduce resolution to specified window length
    window_length = int(window_size / RATES_SAMPLE_RATE)
    if alignment == "event":
        ds_rates_dfs = []
        for event in ["cue_aligned", "reward_aligned"]:
            rates_df = aligned_rates_df.firing_rate[event]
            ds_rates_df = rates_df.T.rolling(window=window_length, center=True).mean()[::window_length].dropna().T
            ds_rates_df.columns = pd.MultiIndex.from_tuples([("firing_rate", event, c) for c in ds_rates_df.columns])
            if smooth_SD:
                ds_rates = ds_rates_df.values
                smooted_rates = gaussian_filter1d(ds_rates, smooth_SD, axis=1)
                ds_rates_df = pd.DataFrame(smooted_rates, index=ds_rates_df.index, columns=ds_rates_df.columns)
            ds_rates_dfs.append(ds_rates_df)
        ds_rates_df = pd.concat(ds_rates_dfs, axis=1)
    elif alignment == "trial":
        rates_df = aligned_rates_df.firing_rate
        ds_rates_df = rates_df.T.rolling(window=window_length, center=True).mean()[::window_length].dropna().T
        ds_rates_df.columns = pd.MultiIndex.from_tuples([("firing_rate", c) for c in ds_rates_df.columns])
        if smooth_SD:
            ds_rates = ds_rates_df.values
            smooted_rates = gaussian_filter1d(ds_rates, smooth_SD, axis=1)
            ds_rates_df = pd.DataFrame(smooted_rates, index=ds_rates_df.index, columns=ds_rates_df.columns)
    # replace firing_rates columns with downsampled version
    aligned_rates_df = aligned_rates_df.drop("firing_rate", axis=1, level=0)
    activity_df = pd.concat([aligned_rates_df, ds_rates_df], axis=1)
    return activity_df


def _get_decoding_accurary(
    activity_df, validation_folds_df, classifier="logreg", decoder_kwargs={"inv_alpha": None}, verbose=False
):
    """Returns decoding accuracy for each timepoint and fold"""
    timepoints = activity_df.firing_rate.columns
    folds = validation_folds_df.columns.get_level_values(0).unique()
    if isinstance(timepoints[0], tuple):  # event-aligned case
        results_index = pd.MultiIndex.from_tuples(
            [(*col, new_level) for col in timepoints for new_level in ["test", "train"]]
        )
    else:  # trial-aligned case
        results_index = pd.MultiIndex.from_tuples(
            [(col, new_level) for col in timepoints for new_level in ["test", "train"]]
        )
    results_df = pd.DataFrame(index=results_index, columns=folds)
    for fold in folds:
        if verbose:
            print(fold)
        # get test, train data for logistic regression
        fold_df = validation_folds_df[fold]
        if len(fold_df.test.columns[0][0]) > 1:  # remove empty index when combining data across sessions
            test_df = fold_df.test.droplevel(0, axis=1)
        else:
            test_df = fold_df.test
        test_X, test_y = get_synthetic_activity_matrix(
            activity_df, test_df
        )  # [n_goals, n_session_clusters, n_timepoints], [n_goals]
        training_df = fold_df.training
        training_Xs, training_ys = [], []
        for training_set in training_df.columns.get_level_values(0).unique():
            training_set_df = training_df[training_set]
            X, y = get_synthetic_activity_matrix(activity_df, training_set_df)
            training_Xs.append(X)
            training_ys.append(y)
        training_X = np.vstack(training_Xs)  # [n_training_sets x n_goals, n_session_clusters, n_timepoints]
        training_y = np.hstack(training_ys)  # [n_training_sets x n_goals]

        # set up decoder based on speified inputs
        if classifier == "logreg":
            if decoder_kwargs["inv_alpha"] is None:
                decoder = LogisticRegression(penalty=None, max_iter=10000)
            else:
                decoder = LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    C=decoder_kwargs["inv_alpha"],
                    max_iter=10000,
                )
        elif classifier == "mlp":
            decoder = MLPClassifier(
                hidden_layer_sizes=decoder_kwargs["Nhid"],
                alpha=decoder_kwargs["alpha"],
                solver=decoder_kwargs["solver"],
                max_iter=500,
                verbose=False,
                tol=1e-4,
            )

        elif classifier == "mlp_torch":
            decoder = mu.MLPtorchClassifier(
                hidden_layer_sizes=decoder_kwargs["Nhid"],
                alpha=decoder_kwargs["alpha"],
                max_epochs=500,
                verbose=False,
                tol=1e-4,
            )
        else:
            raise NotImplementedError

        # run decoding at each timepoint
        for i in range(len(timepoints)):
            test_activity = test_X[:, :, i]  # [n_goals, n_session_clusters]
            training_activity = training_X[:, :, i]  # [n_training_sets x n_goals, n_session_clusters]
            decoder.fit(training_activity, training_y)
            test_predictions = decoder.predict(test_activity)
            test_accuracy = (test_predictions == test_y).mean()
            train_predictions = decoder.predict(training_activity)
            train_accuracy = (train_predictions == training_y).mean()
            rtest_loc = (*timepoints[i], "test") if isinstance(timepoints[i], tuple) else (timepoints[i], "test")
            rtrain_loc = (*timepoints[i], "train") if isinstance(timepoints[i], tuple) else (timepoints[i], "train")
            results_df.loc[rtest_loc, fold] = test_accuracy
            results_df.loc[rtrain_loc, fold] = train_accuracy
    return results_df


def get_synthetic_activity_matrix(activity_df, test_df):
    """ """
    activity_df = activity_df.set_index("trial_unique_ID")
    n_session_clusters = len(activity_df.cluster_unique_ID.unique())
    n_goals = test_df.shape[0]
    n_timepoints = activity_df.firing_rate.shape[1]
    synthetic_activity_matrix = np.full((n_goals, n_session_clusters, n_timepoints), np.nan)
    goal_vector = []
    for i, goal in enumerate(test_df.index):
        test_trials = test_df.loc[goal]
        test_trials_activity_df = activity_df.loc[test_trials]
        test_trials_activity_df = test_trials_activity_df[
            [c for c in test_trials_activity_df.columns if c[0] in ["cluster_unique_ID", "firing_rate"]]
        ].reset_index()  # [n_test_trials x n_neurons, n_timepoints + 2]
        synthetic_trial_activity_df = test_trials_activity_df.pivot(
            index="trial_unique_ID", columns="cluster_unique_ID"
        )  # [n_test_trials, n_timespoints x n_session-clusters], only has values for clusters recorded on trials from the same session
        synthetic_trial_activity = synthetic_trial_activity_df.sum(
            axis=0
        )  # collapse to [n_session-clusters, n_timespoints, 1] (pd.Series)
        # assign synthetic trial activity to synthetic activity matrix
        synthetic_activity_matrix[i, :, :] = (
            synthetic_trial_activity.unstack().values.T
        )  # [n_session-clusters, n_timespoints (cue and reward aligned)]
        goal_vector.append(goal)
    return synthetic_activity_matrix, np.array(goal_vector)


# %% Functions building validation_folds_df


def get_validation_folds_df(sessions, n_training_trial_sets, alignment="event"):
    """ """
    exp_max_trials_per_goal = get_max_trials_per_goal(sessions, alignment)
    session_fold_dfs = []
    session_counts = {session.name: 0 for session in sessions}
    for session in sessions:
        session_name = session.name
        count = session_counts[session_name]
        session_fold_df = get_session_validation_folds_df(
            session,
            exp_max_trials_per_goal,
            n_training_trial_sets,
            alignment,
            count,
        )
        # turn from trial float value to trial_unique_ID string
        session_fold_df = session_fold_df.map(
            lambda x: f"{session.subject_ID}_{session.maze_name}_{session.day_on_maze}_{int(x)}_c{count}",
        )
        session_fold_dfs.append(session_fold_df)
        session_counts[session_name] += 1
    validation_folds_df = pd.concat(session_fold_dfs, axis=1).sort_index(axis=1)
    return validation_folds_df


def get_max_trials_per_goal(sessions, alignment):
    """returns the maximum trials compelted for any one goal in a session or list of sessions"""
    sessions = [sessions] if isinstance(sessions, gs.MazeSession) else sessions
    session_maxes = []
    for session in sessions:
        trials_per_goal = get_trials_per_goal(session, alignment)
        max_trials_per_goal = np.max([len(t) for t in trials_per_goal])
        session_maxes.append(max_trials_per_goal)
    return np.max(session_maxes)


def get_trials_per_goal(session, alignment):
    if alignment == "event":  # has all trials even if very short etc.
        df = session.event_aligned_rates_df
    elif alignment == "trial":  # note some trials are missing in trial_aligned_rates_df bc/ erc after ITI
        df = session.trial_aligned_rates_df
    trials2goal_df = df.loc[:, (df.columns.get_level_values(0).isin(["trial", "goal"]))]
    trials_per_goal = trials2goal_df.dropna().groupby("goal")["trial"].apply(lambda x: np.unique(x))
    return trials_per_goal


def get_session_validation_folds_df(session, exp_max_trials_per_goal, n_training_trial_sets, alignment, count):
    """"""
    session_name = f"{session.name}_c{count}"
    if alignment == "event":  # has all trials even if very short etc.
        df = session.event_aligned_rates_df
    elif alignment == "trial":  # note some trials are missing in trial_aligned_rates_df bc/ erc after ITI
        df = session.trial_aligned_rates_df
    # trials2goal_df = df.loc[:, (df.columns.get_level_values(0).isin(["trial", "goal"]))]
    # trials_per_goal = trials2goal_df.dropna().groupby("goal")["trial"].apply(lambda x: np.unique(x))
    trials_per_goal = get_trials_per_goal(session, alignment)
    max_trials_per_goal = np.max([len(t) for t in trials_per_goal])
    # goals must have 2 or more valid trials (enough for test_train split)
    if trials_per_goal.apply(len).lt(2).any():
        raise ValueError(f"{session} has some goals have less than 2 valid trials")
    # test-train splits
    fold_dfs = []
    f = 0
    trials_per_goal = trials_per_goal.apply(lambda x: np.random.choice(x, size=len(x), replace=False))
    trials_per_goal_df = trials_per_goal.apply(pd.Series)
    for fold in range(exp_max_trials_per_goal):
        if f > max_trials_per_goal - 1:  # shuffle and reset fold counter
            trials_per_goal = trials_per_goal.apply(lambda x: np.random.choice(x, size=len(x), replace=False))
            trials_per_goal_df = trials_per_goal.apply(pd.Series)
            f = 0
        tpg_df = trials_per_goal_df.copy()
        test_trial_set = tpg_df[f]
        training_trial_sets = tpg_df.drop(f, axis=1)
        while test_trial_set.isna().any():  # choose another fold for test if includes NaNs
            replacement_fold = np.random.choice(training_trial_sets.columns)
            replacement_set = tpg_df[replacement_fold]
            if replacement_set.isna().any():
                continue
            test_trial_set = test_trial_set.fillna(replacement_set)
            training_trial_sets = training_trial_sets.drop(replacement_fold, axis=1)
        # extend training trial sets to desired length
        n_synthetic_trial_sets = n_training_trial_sets - training_trial_sets.shape[-1]
        extra_training_trial_sets = get_synthetic_trial_sets(
            training_trial_sets, n_synthetic_trial_sets, max_trials_per_goal
        )
        training_trial_sets = fill_training_trial_sets_df(training_trial_sets)
        training_trial_sets = pd.concat([training_trial_sets, extra_training_trial_sets], axis=1)
        training_trial_sets.columns = pd.MultiIndex.from_tuples(
            [(f"fold_{fold}", "training", c, session_name) for c in range(n_training_trial_sets)]
        )
        # add training sets to fold
        fold_df = training_trial_sets
        fold_df[(f"fold_{fold}", "test", "", session_name)] = test_trial_set
        fold_dfs.append(fold_df)
        f += 1
    session_fold_df = pd.concat(fold_dfs, axis=1)
    return session_fold_df


def fill_training_trial_sets_df(training_trial_sets):
    df = training_trial_sets.copy()
    for index, row in df.iterrows():
        nan_cols = row.index[row.isnull()]
        valid_cols = row.index[row.notnull()]
        if len(nan_cols) > len(valid_cols):
            sample = row.dropna().sample(n=len(nan_cols), replace=True)
        else:  # sample wo/ replacement possible
            sample = row.dropna().sample(n=len(nan_cols), replace=False)
        df.loc[index, nan_cols] = sample.values
    return df


def get_synthetic_trial_sets(training_trial_sets, n_extra_sets, max_trials_per_goal):
    synth_trials = []
    for index, row in training_trial_sets.iterrows():
        trial_generator = sample_without_replacement(list(row.dropna()))
        sampled_trials = []
        for i in range(n_extra_sets):
            sampled_trials.append(next(trial_generator))
        synth_trials.append(sampled_trials)
    return pd.DataFrame(
        np.array(synth_trials),
        columns=[f"s{max_trials_per_goal+i}" for i in range(n_extra_sets)],
        index=training_trial_sets.index,
    )


def sample_without_replacement(lst):
    assert len(lst) > 0, "list must have at least one item"
    working_copy = lst.copy()  # Create a copy of the list to work with
    while True:
        if not working_copy:  # If the working copy is empty, refresh it
            working_copy = lst.copy()
        choice = random.choice(working_copy)  # Randomly select an item
        working_copy.remove(choice)  # Remove the selected item to prevent future selection
        yield choice


# %% Validation folds quality control functions


def validation_fold_qc(validation_folds_df):
    """Checks test-train contamination, goal_stratification and similarity of training set folds"""
    if not _check_no_test_train_contamination(validation_folds_df):
        raise ValueError("test trials found in training sets")
    else:
        print("no test trials found in training sets")
    goal_stratification = _check_goal_stratification(validation_folds_df)
    print(f"goal stratification: {goal_stratification}")
    test_set_similarity = _check_similarity_of_training_sets(validation_folds_df)
    print(f"average similarity of training sets: {test_set_similarity}")
    return


def _check_goal_stratification(validation_folds_df):
    goals, counts = np.unique(validation_folds_df.index.to_numpy(), return_counts=True)
    return dict(zip(goals, counts))


def _check_similarity_of_training_sets(validation_folds_df):
    """Returns average similairy of training sets across folds. value between (0,1)"""
    similarities = []
    for fold in validation_folds_df.columns.get_level_values(0).unique():
        fold_df = validation_folds_df[fold]
        n_sets = fold_df.training.shape[-1]
        for i in range(n_sets):
            for j in range(i + 1, n_sets):
                set_i = fold_df.training[i]
                set_j = fold_df.training[j]
                similarity = (set_i == set_j).mean()
                similarities.append(similarity)
    return np.mean(similarities)


def _check_no_test_train_contamination(validation_folds_df):
    """Checks that there is no test trials in eany of the training data for each fold"""
    for fold in validation_folds_df.columns.get_level_values(0).unique():
        fold_df = validation_folds_df[fold]
        test_trials = fold_df.test.to_numpy().flatten()
        training_trials = fold_df.training.to_numpy().flatten()
        if np.in1d(test_trials, training_trials).any():
            return False
    return True


# %% Test regularisation param


def run_hyperparameter_search(maze_name="maze_1", goal_subset="subset_1", classifiers=["mlp_torch"], save=True):
    sessions = get_sessions_for_analysis(
        subject_IDs="all",
        maze_name=maze_name,
        goal_subset=goal_subset,
        alignment="event",
    )
    validation_folds_df = get_validation_folds_df(sessions, n_training_trial_sets=100, alignment="event")
    return_data = []
    for window_size in [0.2]:  # [0.1, 0.2, 0.5]:
        for smooth_SD in [4]:  # [False, 4, 8]:
            activity_df = get_activity_df(
                sessions, alignment="event", include_multi_units=True, window_size=window_size, smooth_SD=smooth_SD
            )
            # filter activity df to only include the window at reward where decoding is best
            _df = activity_df.firing_rate.reward_aligned
            reward_time = _df.columns[np.abs(_df.columns).argmin()]
            activity_df = activity_df[
                [
                    c
                    for c in activity_df.columns
                    if c[0] != "firing_rate" or (c[1] == "reward_aligned" and c[2] == reward_time)
                ]
            ]

            # logreg hyperparameter search
            if "logreg" in classifiers:
                logreg_hp_search_results = []
                for inv_alpha in [None, 1e-6, 1e-4, 1e-2, 1, 1e2]:
                    results_df = _get_decoding_accurary(
                        activity_df,
                        validation_folds_df,
                        classifier="logreg",
                        decoder_kwargs={"inv_alpha": inv_alpha},
                    )
                    mean_acc = results_df.droplevel([0, 1]).mean(axis=1)
                    hp_result = {
                        "decoder": "logreg",
                        "maze_name": maze_name,
                        "goal_subset": goal_subset,
                        "inv_alpha": inv_alpha,
                        "smooth_SD": smooth_SD,
                        "window_size": window_size,
                        "test_acc": mean_acc.test,
                        "train_acc": mean_acc.train,
                    }
                    print(hp_result)
                    logreg_hp_search_results.append(hp_result)
                logreg_hp_search_results = pd.DataFrame(logreg_hp_search_results)
                return_data.append(logreg_hp_search_results)
                logreg_save_path = RESULTS_DIR / "hyperparameter_search" / maze_name / goal_subset / "logreg.csv"
                if save:
                    if not logreg_save_path.parent.exists():
                        logreg_save_path.parent.mkdir(parents=True, exist_ok=True)
                    logreg_hp_search_results.to_csv(logreg_save_path, index=False)

            # mlp hyperparameter search
            if "mlp" in classifiers:
                mlp_hp_search_results = []
                for solver in ["adam"]:
                    for alpha in [1e-4]:  # [1, 10, 100]:
                        for Nhid in [(100,)]:  # [(50,), (50, 50)]:
                            results_df = _get_decoding_accurary(
                                activity_df,
                                validation_folds_df,
                                classifier="mlp",
                                decoder_kwargs={"alpha": alpha, "Nhid": Nhid, "solver": solver},
                            )
                            mean_acc = results_df.droplevel([0, 1]).mean(axis=1)
                            hp_result = {
                                "decoder": "mlp",
                                "maze_name": maze_name,
                                "goal_subset": goal_subset,
                                "solver": solver,
                                "alpha": alpha,
                                "Nhid": Nhid,
                                "smooth_SD": smooth_SD,
                                "window_size": window_size,
                                "test_acc": mean_acc.test,
                                "train_acc": mean_acc.train,
                            }
                            print(hp_result)
                            mlp_hp_search_results.append(hp_result)
                mlp_hp_search_results = pd.DataFrame(mlp_hp_search_results)
                return_data.append(mlp_hp_search_results)
                mlp_save_path = RESULTS_DIR / "hyperparameter_search" / maze_name / goal_subset / "mlp.csv"
                if save:
                    if not mlp_save_path.parent.exists():
                        mlp_save_path.parent.mkdir(parents=True, exist_ok=True)
                    mlp_hp_search_results.to_csv(mlp_save_path, index=False)

            if "mlp_torch" in classifiers:
                mlp_torch_hp_search_results = []
                for alpha in [1e-2]:
                    for Nhid in [
                        (25, 25),
                    ]:
                        results_df = _get_decoding_accurary(
                            activity_df,
                            validation_folds_df,
                            classifier="mlp_torch",
                            decoder_kwargs={"alpha": alpha, "Nhid": Nhid},
                        )
                        mean_acc = results_df.droplevel([0, 1]).mean(axis=1)
                        hp_result = {
                            "decoder": "mlp_torch",
                            "maze_name": maze_name,
                            "goal_subset": goal_subset,
                            "alpha": alpha,
                            "Nhid": Nhid,
                            "smooth_SD": smooth_SD,
                            "window_size": window_size,
                            "test_acc": mean_acc.test,
                            "train_acc": mean_acc.train,
                        }
                        print(hp_result)
                        mlp_torch_hp_search_results.append(hp_result)
                mlp_torch_hp_search_results = pd.DataFrame(mlp_torch_hp_search_results)
                return_data.append(mlp_torch_hp_search_results)
                mlp_torch_save_path = RESULTS_DIR / "hyperparameter_search" / maze_name / goal_subset / "mlp_torch.csv"
                if save:
                    if not mlp_torch_save_path.parent.exists():
                        mlp_torch_save_path.parent.mkdir(parents=True, exist_ok=True)
                    mlp_torch_hp_search_results.to_csv(mlp_torch_save_path, index=False)

    return tuple(return_data) if len(return_data) > 1 else return_data[0]


# %%
# %% Non bootstrapped plotting functions


def plot_trial_aligned_decoding_results(results_df, ax=None, sem=True, chance=1 / 12):
    """ """
    df = results_df.xs("test", level=1, axis=0)
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(5, 3), clear=True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Acc.")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-5, INTRA_TRIAL_INTERVAL_TIMES["ITI_end"] + 0.5)
    ax.set_xticks(list(INTRA_TRIAL_INTERVAL_TIMES.values()))
    ax.set_xticklabels(["cue", "reward", "erc", "end"])
    for time in INTRA_TRIAL_INTERVAL_TIMES.values():
        ax.axvline(time, color="black", linestyle="--", alpha=0.5)
    ax.axhline(chance, color="black", linestyle="--", alpha=0.5)
    # plot results
    mean_acc = df.mean(axis=1)  # across folds
    time = mean_acc.index.values.astype(float)
    av = mean_acc.values.astype(float)
    ax.plot(time, av, color="deepskyblue")
    if sem:
        sem = df.sem(axis=1).values.astype(float)
        ax.fill_between(time, av - sem, av + sem, color="deepskyblue", alpha=0.3)


def plot_event_aligned_decoding_results(results_df, axes=None, chance=1 / 12):
    """ """
    df = results_df.xs("test", level=2, axis=0)
    if axes is None:
        f, axes = plt.subplots(1, 2, figsize=(6, 3), clear=True, sharey=True)
    for ax, label in zip(axes, ["Cue", "Reward"]):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("Acc.")
        ax.set_xlabel(label)
        ax.set_ylim(0, 1)
        ax.set_xlim(-10, 10)
        ax.axhline(chance, color="black", linestyle="--", alpha=0.5)
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    # plot results
    mean_acc = df.mean(axis=1)  # across folds
    for ax, ind in zip(axes, ["cue_aligned", "reward_aligned"]):
        time = mean_acc[ind].index.values.astype(float)
        av = mean_acc[ind].values.astype(float)
        ax.plot(time, av, color="deepskyblue")


# %% run subset=="all" seperately per subject


def run_single_subject_decoding(decoder="logreg"):
    """
    Run with GPU if running mlp_torch decoder
    """
    for alignment in ["trial", "event"]:
        for maze in MAZE_NAMES:
            for goal_set in GOAL_SETS:
                for subject in SUBJECT_IDS:
                    save_path = (
                        RESULTS_DIR
                        / "single_subject_decoding"
                        / alignment
                        / decoder
                        / maze
                        / goal_set
                        / f"{subject}.csv"
                    )
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    if save_path.exists():
                        continue
                    print(f"Running {subject} {maze} {goal_set} {alignment} {decoder}")
                    results_df = run_allocentric_goal_decoding(
                        subject_IDs=[subject],
                        maze_name=maze,
                        goal_subset=goal_set,
                        alignment=alignment,
                        include_multi_units=True,
                        decoder=decoder,
                        decoder_kwargs=DECODER2KWARGS[decoder],
                        window_size=WINDOW_SIZE,
                        smooth_SD=SMOOTH_SD,
                        plot=False,
                    )
                    if results_df is None:
                        continue  # no valid sessions fn returns None
                    else:
                        results_df.to_csv(save_path)
    return
