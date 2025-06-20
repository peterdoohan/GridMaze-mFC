"""
Library for the analysis of neural tuning to place_direction explaining low dimension structure of subject's behaviour
"""

# %% Imports
import json
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from GridMaze.analysis.core import get_sessions as gs

from GridMaze.analysis.place_direction import dimensionality_reduction as pdr
from GridMaze.analysis.behaviour import dimensionality_reduction as bdr
from GridMaze.analysis.behaviour import synthetic_behaviour as sb


from GridMaze.maze import representations as mr
from GridMaze.analysis.core import filter as filt

from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit
from matplotlib import pyplot as plt
from scipy.stats import ttest_1samp
import seaborn as sns

# %% Global Variables

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "place_direction" / "efficient_coding"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)


# %% Within-Between subject control


def get_within_across_subject_neural_variance_explained_by_behaviour(
    maze_name,
    test_size=0.5,
    n_splits=5,
    late_sessions=False,
    max_steps_to_goal=30,
    demean=False,
    norm_length=True,
    verbose=True,
):
    """
    Null results

    With cross-validation takes the neurons from all but one subject, then takes the held out behaviour (from the
    held out subjects) or the within-subjects behaviour (same subjects from the neurons), downsampled to match the
    trials in held out subjects data and use this behaviour to explain variance in the neurons.
    """
    input_data = get_input_data(
        maze_name=maze_name,
        n_splits=n_splits,
        test_size=test_size,
        late=late_sessions,
        max_steps_to_goal=max_steps_to_goal,
        verbose=verbose,
    )
    results = []
    for i in range(n_splits):
        if verbose:
            print(f"split: {i}")
        for held_out_subject in SUBJECT_IDS:
            if verbose:
                print(f"held out subject: {held_out_subject}")
            held_out_subject_behaviour = pd.concat(
                [input_data[held_out_subject][i]["true_behaviour"][t] for t in ["train", "test"]], axis=0
            )  # combine train and test bc, these trials never went into making neural heatmaps in other subs
            n_trials = held_out_subject_behaviour.shape[0]
            other_subjects = [s for s in SUBJECT_IDS if s != held_out_subject]
            other_subject_behaviour = pd.concat(
                [input_data[subject][i]["true_behaviour"]["train"] for subject in other_subjects],
                axis=0,
            )
            # ensure same number of trials between held out and other subjects
            other_subject_behaviour = other_subject_behaviour.sample(n=n_trials, replace=False, random_state=0)
            other_subject_neurons = pd.concat(
                [input_data[subject][i]["neural_data"]["test"] for subject in other_subjects],
                axis=0,
            )
            # get variance expalin in neurons by within-subject or held-out subject behaviour
            arrays = [other_subject_neurons.values, other_subject_behaviour.values, held_out_subject_behaviour.values]
            if demean:
                arrays = [_demean(arr) for arr in arrays]
            if norm_length:
                arrays = [_norm_length(arr) for arr in arrays]
            other_neurons, other_behaviour, held_out_behaviour = arrays
            # calculate variance explained
            _results = {"split": i, "held_out_subject": held_out_subject}
            for label, B in zip(["held_out", "same"], [held_out_behaviour, other_behaviour]):
                cumsum_ve = get_pca_variance_explained(B, other_neurons)
                auc = np.trapz(cumsum_ve, dx=1 / len(cumsum_ve))
                _results[label] = auc
            results.append(_results)
    results_df = pd.DataFrame(results)
    return results_df


# %% Null vs subject behaviour 2


def plot_neural_variance_explained_by_behaviour(results_df, ax=None):
    """
    Need to do stats
    """
    df = results_df.groupby("resample").mean().drop(columns=["split"])  # average over splits(folds)
    conditions = df.columns.tolist()
    means = df.mean()
    lower = df.quantile(0.025)
    upper = df.quantile(0.975)
    err_lower = means - lower
    err_upper = upper - means
    colors = ["red", "blue", "grey", "grey", "grey", "grey"]
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 2), clear=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(means["neural_data"], color="red", ls="--", alpha=0.5)
    for i, (cond, color) in enumerate(zip(conditions, colors)):
        ax.errorbar(
            x=means[cond],
            y=i,
            xerr=[[err_lower[cond]], [err_upper[cond]]],
            fmt="o",
            markersize=5,
            color=color,
            capthick=1.5,
            elinewidth=1.5,
            capsize=3,
        )
    ax.set_yticks(range(len(conditions)), conditions)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 0.8)
    ax.set_xlabel("AUC of cumulative variance explained")
    return


def get_neural_variance_explained_by_behaviour(
    maze_name,
    late_sessions=False,  # need as much data as possible with CV approach
    n_splits=5,
    test_size=0.5,
    max_steps_to_goal=30,
    demean=False,
    norm_length=True,
    n_resamples=500,
    verbose=True,
    max_jobs=20,
):
    """
    Similar to original version but with bootstrap resample across subjects, still with X val
    just interested in differences between how well real behaviour explains neurons and how well synthetic
    behaviour explains neurons.
    """
    save_path = RESULTS_DIR / f"neurons_explained_by_behaviour_{maze_name}.csv"
    if save_path.exists():
        if verbose:
            print(f"Loading existing results from {save_path}")
        return pd.read_csv(save_path, index_col=0)
    # get input data
    if verbose:
        print("Loading input data...")
    subject2split_data = get_input_data(
        maze_name=maze_name,
        n_splits=n_splits,
        test_size=test_size,
        late=late_sessions,
        max_steps_to_goal=max_steps_to_goal,
        verbose=verbose,
    )
    data_types = [
        "neural_data",
        "true_behaviour",
        "random_diffusion",
        "forward_diffusion",
        "vector",
        "optimal",
    ]
    # proceses results across bootstrap resamples across subjects
    resampled_results = Parallel(n_jobs=max_jobs)(
        delayed(_process_resample)(subject2split_data, data_types, n, n_splits, demean, norm_length, verbose)
        for n in range(n_resamples)
    )
    results_df = pd.concat(resampled_results, axis=0)
    # save results
    results_df.to_csv(save_path)
    if verbose:
        print(f"Results saved to {save_path}")
    return results_df


def _process_resample(subject2split_data, data_types, n, n_splits, demean, norm_length, verbose):
    if verbose:
        print(f"resample: {n}")
    sampled_subjects = np.random.choice(SUBJECT_IDS, size=len(SUBJECT_IDS), replace=True)
    resample_results = []
    for i in range(n_splits):
        data_type2train_dfs = {data_type: [] for data_type in data_types}
        data_type2test_dfs = {data_type: [] for data_type in data_types}
        for subject in sampled_subjects:
            split_data = subject2split_data[subject][i]
            for data_type in data_types:
                data_type2train_dfs[data_type].append(split_data[data_type]["train"])
                data_type2test_dfs[data_type].append(split_data[data_type]["test"])
        train_data2df = {data_type: pd.concat(data_type2train_dfs[data_type], axis=0) for data_type in data_types}
        test_data2df = {data_type: pd.concat(data_type2test_dfs[data_type], axis=0) for data_type in data_types}
        # calculate variance explained
        split_results = {}
        # explain var in test neural data with...
        neural_test = test_data2df["neural_data"].values
        if demean:
            neural_test = _demean(neural_test)
        if norm_length:
            neural_test = _norm_length(neural_test)
        # each data type
        for data_type in data_types:
            d_train = train_data2df[data_type].values
            if demean:
                d_train = _demean(d_train)
            if norm_length:
                d_train = _norm_length(d_train)
            cumsum_ve = get_pca_variance_explained(d_train, neural_test)
            auc = np.trapz(cumsum_ve, dx=1 / len(cumsum_ve))
            split_results[data_type] = auc
        split_results["split"] = i
        resample_results.append(split_results)
    df = pd.DataFrame(resample_results)
    df["resample"] = n
    return df


def _demean(X):
    return X - X.mean(-1, keepdims=True)


def _norm_length(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def get_input_data(
    maze_name,
    with_synthetic_behaviour=True,
    n_splits=5,
    test_size=0.5,
    late=False,
    max_steps_to_goal=30,
    min_split_corr=0.5,
    verbose=False,
):
    """
    should avoid data regeneeration when making per subject Xval splits but not sure if this is overkill
    """
    days_on_maze = "late" if late == True else "all"
    all_data = {}
    subject2session_names = {}
    for subject in SUBJECT_IDS:
        if verbose:
            print(subject)
        sub_sessions = gs.get_maze_sessions(
            subject_IDs=[subject],
            maze_names=[maze_name],
            days_on_maze=days_on_maze,
            with_data=[
                "navigation_df",
                "navigation_spike_rates_df",
                "trajectory_decisions_df",
                "cluster_metrics",
                "cluster_place_direction_tuning_metrics",
            ],
        )
        session_names = []
        subject_data = {}
        for session in sub_sessions:
            session_data = {}
            session_data["neural_data"] = pdr.get_session_place_direction_tuning(
                session,
                fill_nans="mean",
                normalisation=False,
                min_split_corr=min_split_corr,
                max_steps_from_goal=max_steps_to_goal,
            )
            session_data["true_behaviour"] = bdr.get_session_behavioural_sequences(
                session, normalisation=False, max_steps_to_goal=max_steps_to_goal
            )
            if with_synthetic_behaviour:
                for policy in ["random_diffusion", "forward_diffusion", "vector", "optimal"]:
                    session_data[policy] = sb.get_session_synthetic_behavioural_sequences(
                        session,
                        policy=policy,
                        normalisation=False,
                        max_steps=max_steps_to_goal,
                    )
            session_names.append(session.name)
            subject_data[session.name] = session_data
        all_data[subject] = subject_data
        subject2session_names[subject] = session_names
    # remove session_names for the pool that have no data
    for subject in SUBJECT_IDS:
        session_names = subject2session_names[subject]
        session_names = [sn for sn in session_names if all_data[subject][sn] is not None]
        subject2session_names[subject] = session_names
    # combine data per subject across Xvaled splits
    if verbose:
        print("Sorting data into CV splits...")
    subject2split_data = {}
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
    data_types = ["neural_data", "true_behaviour"]
    if with_synthetic_behaviour:
        data_types += ["random_diffusion", "forward_diffusion", "vector", "optimal"]
    for subject in SUBJECT_IDS:
        _session_names = np.array(subject2session_names[subject])
        # Generate the splits (session names)
        split2data = {}
        for i, (train_index, test_index) in enumerate(ss.split(_session_names)):
            train, test = _session_names[train_index], _session_names[test_index]
            split_data = {}
            for data_type in data_types:
                train_data = [all_data[subject][session][data_type] for session in train]
                test_data = [all_data[subject][session][data_type] for session in test]
                split_data[data_type] = {
                    "train": pd.concat([df for df in train_data if df is not None], axis=0),
                    "test": pd.concat([df for df in test_data if df is not None], axis=0),
                }
            split2data[i] = split_data
        subject2split_data[subject] = split2data
    return subject2split_data


# %%


def plot_neural_behaviour_variance_explained(results_df, explaining="neurons", colors=["red", "blue"], ax=None):
    """ """
    # process data
    n_components = results_df.component.max()
    df = results_df.groupby(["resample", "component"]).mean()  # average over splits
    if explaining == "neurons":
        conditions = ["N_explains_N", "B_explains_N"]
    elif explaining == "behaviour":
        conditions = ["B_explains_B", "N_explains_B"]
    else:
        raise ValueError(f"explaining must be one of ['neurons', 'behaviour'], got {explaining}")
    # plot
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3), clear=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot([0, n_components], [0, 1], color="black", ls="--", alpha=0.5)  # baseline
    ax.set_xlabel("n components")
    ax.set_ylabel("Cum. var exp")
    for cond, color in zip(conditions, colors):
        _df = df[cond].unstack()
        c = _df.columns.values
        mean = _df.mean()
        lower = _df.quantile(0.025)
        upper = _df.quantile(0.975)
        ax.plot(c, mean, label=cond, color=color)
        ax.fill_between(c, lower, upper, color=color, alpha=0.3)
    ax.legend(fontsize="xx-small")


def get_neural_behaviour_variance_explained(
    maze_name,
    input_data=None,
    cv=True,
    test_size=0.1,
    n_splits=5,
    late_sessions=False,
    max_steps_to_goal=30,
    n_resamples=500,
    demean=False,
    norm_length=True,
    verbose=True,
    max_jobs=20,
    force_save=False,
):
    """ """
    save_path = RESULTS_DIR / f"neural_behaviour_variance_explained_{maze_name}.csv"
    if save_path.exists() and not force_save:
        if verbose:
            print(f"Loading existing results from {save_path}")
        return pd.read_csv(save_path, index_col=0)
    if verbose:
        print("Loading input data...")
    if input_data is None:
        input_data = get_input_data(
            maze_name=maze_name,
            with_synthetic_behaviour=False,
            n_splits=n_splits,
            test_size=test_size,
            late=late_sessions,
            max_steps_to_goal=max_steps_to_goal,
            verbose=verbose,
        )
    process_fn = _process_resample_cv if cv else _process_resample_no_cv
    all_results = joblib.Parallel(n_jobs=max_jobs)(
        delayed(process_fn)(input_data, n, n_splits, test_size, demean, norm_length, verbose)
        for n in range(n_resamples)
    )
    results_df = pd.concat(all_results, axis=0)
    # save results
    results_df.to_csv(save_path)
    if verbose:
        print(f"Results saved to {save_path}")
    return results_df


def _process_resample_no_cv(input_data, n, n_splits, test_size, demean, norm_length, verbose):
    """ """
    if verbose:
        print(f"resample: {n}")
    sampled_subjects = np.random.choice(SUBJECT_IDS, size=len(SUBJECT_IDS), replace=True)
    resample_results = []
    for i in range(n_splits):
        neurons = pd.concat(
            [input_data[subject][i]["neural_data"][t] for t in ["train", "test"] for subject in sampled_subjects],
            axis=0,
        )
        # subsample to match cv split
        neurons = neurons.sample(int(neurons.shape[0] * test_size), replace=False)
        neurons = neurons.values
        behaviour = pd.concat(
            [input_data[subject][i]["true_behaviour"][t] for t in ["train", "test"] for subject in sampled_subjects],
            axis=0,
        )
        # subsample to match cv split
        behaviour = behaviour.sample(int(behaviour.shape[0] * test_size), replace=False)
        behaviour = behaviour.values
        if demean:
            neurons, behaviour = [_demean(arr) for arr in [neurons, behaviour]]
        if norm_length:
            neurons, behaviour = [_norm_length(arr) for arr in [neurons, behaviour]]
        # calculate variance explained
        conditions = ["N_explains_N", "B_explains_N", "B_explains_B", "N_explains_B"]
        df = pd.DataFrame(index=range(neurons.shape[1] + 1), columns=conditions)
        for label, (A, B) in zip(
            conditions,
            [
                (neurons, neurons),
                (behaviour, neurons),
                (behaviour, behaviour),
                (neurons, behaviour),
            ],
        ):
            cum_ve = get_pca_variance_explained(A, B)
            df[label] = cum_ve
        df["split"] = i
        df["resample"] = n
        df.reset_index(inplace=True)
        df.rename(columns={"index": "component"}, inplace=True)
        resample_results.append(df)
    return pd.concat(resample_results, axis=0)


def _process_resample_cv(input_data, n, n_splits, test_size, demean, norm_length, verbose):
    """ """
    if verbose:
        print(f"resample: {n}")
    sampled_subjects = np.random.choice(SUBJECT_IDS, size=len(SUBJECT_IDS), replace=True)
    resample_results = []
    for i in range(n_splits):
        train_neurons, test_neurons = [
            pd.concat([input_data[subject][i]["neural_data"][t] for subject in sampled_subjects], axis=0)
            for t in ["train", "test"]
        ]
        train_neurons, test_neurons = train_neurons.values, test_neurons.values
        train_behaviour, test_behaviour = [
            pd.concat([input_data[subject][i]["true_behaviour"][t] for subject in sampled_subjects], axis=0)
            for t in ["train", "test"]
        ]
        train_behaviour, test_behaviour = train_behaviour.values, test_behaviour.values
        if demean:
            train_neurons, test_neurons = [_demean(arr) for arr in [train_neurons, test_neurons]]
            train_behaviour, test_behaviour = [_demean(arr) for arr in [train_behaviour, test_behaviour]]
        if norm_length:
            train_neurons, test_neurons = [_norm_length(arr) for arr in [train_neurons, test_neurons]]
            train_behaviour, test_behaviour = [_norm_length(arr) for arr in [train_behaviour, test_behaviour]]
        # calculate variance explained
        conditions = ["N_explains_N", "B_explains_N", "B_explains_B", "N_explains_B"]
        df = pd.DataFrame(index=range(test_neurons.shape[1] + 1), columns=conditions)
        for label, (A, B) in zip(
            conditions,
            [
                (train_neurons, test_neurons),
                (train_behaviour, test_neurons),
                (train_behaviour, test_behaviour),
                (train_neurons, test_behaviour),
            ],
        ):
            cum_ve = get_pca_variance_explained(A, B)
            if len(cum_ve) < test_neurons.shape[1] + 1:
                cum_ve = np.concatenate((cum_ve, np.ones(test_neurons.shape[1] + 1 - len(cum_ve))))
            df[label] = cum_ve
        df["split"] = i
        df["resample"] = n
        df.reset_index(inplace=True)
        df.rename(columns={"index": "component"}, inplace=True)
        resample_results.append(df)
    return pd.concat(resample_results, axis=0)


def get_svd_variance_explained(A, B, pad=False):  # A & B: [n_samples, n_features]
    """Calculates the cumulative variance of matrix B that's explained by the first i orthonormal bases of matrix A using SVD."""
    U, Sigma, Vt = np.linalg.svd(A, full_matrices=False)
    M = B @ Vt.T  # B projected onto the svd bases of A
    pc_exp_var = np.square(M).sum(axis=0)
    cumsum_exp_var = np.cumsum(pc_exp_var) / pc_exp_var.sum()
    # pad with 1s if n_samples < n_features
    result = np.concatenate(([0], cumsum_exp_var))
    if pad:
        if len(result) < A.shape[1]:
            result = np.concatenate((result, np.ones(A.shape[1] - len(cumsum_exp_var))))
    return result


def get_pca_variance_explained(A, B):  # A & B: [n_samples, n_features]
    """Calculates the cumulative variance of matrix B that's explained by the first i components of matrix A."""
    model = PCA(random_state=0)
    model.fit(A)
    M = model.transform(B)  # [n_samples, n_components]
    pc_exp_var = np.square(M).sum(axis=0)
    cumsum_exp_var = np.cumsum(pc_exp_var) / pc_exp_var.sum()
    return np.concatenate(([0], cumsum_exp_var))
