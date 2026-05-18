"""
Library for the analysis of neural tuning to place_direction explaining low dimension structure of subject's behaviour
"""

# %% Imports
from cProfile import label
import json
import joblib
from torch import fill
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
from scipy.stats import ttest_1samp, ttest_rel
import seaborn as sns

# %% Global Variables

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS2_PATH

RESULTS_DIR = RESULTS2_PATH / "place_direction" / "efficient_coding"

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


def _stats(
    auc_df,
    comparisons=[
        ("neural_data", ("true_behaviour", "optimal", "random_diffusion")),
        ("true_behaviour", ("optimal", "random_diffusion")),
    ],
):
    """ """
    xs = [c[0] for c in comparisons]
    ys = np.unique([y for c in comparisons for y in c[1]])
    stats_df = pd.DataFrame(index=xs, columns=ys, data=np.nan)
    auc = auc_df.groupby("resample").mean().drop(columns=["split"])
    for comparison in comparisons:
        x, ys = comparison
        for y in ys:
            stats_df.loc[x, y] = (auc[x] < auc[y]).mean()
    return stats_df


def _plot_auc_comparison(
    auc_df,
    conditions=[
        "neural_data",
        "true_behaviour",
        "vector",
        "forward_diffusion",
        "optimal",
        "random_diffusion",
    ],
    colors=["red", "blue", "grey", "grey", "grey", "grey"],
    print_stats=True,
    comparions=[
        ("true_behaviour", ("optimal", "random_diffusion", "forward_diffusion", "vector")),
    ],
    ax=None,
):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 1.5), clear=True)
    ax.spines[["top", "right"]].set_visible(False)
    # process AUC data (bottom axis)
    auc = auc_df.groupby("resample").mean().drop(columns=["split"])
    mean_auc = auc.mean()
    auc_lower = auc.quantile(0.025)
    auc_upper = auc.quantile(0.975)
    err_lower = mean_auc - auc_lower
    err_upper = auc_upper - mean_auc
    for i, (cond, color) in enumerate(zip(conditions, colors)):
        ax.errorbar(
            x=mean_auc[cond],
            y=i,
            xerr=[[err_lower[cond]], [err_upper[cond]]],
            fmt="none",
            color=color,
            label=cond,
            capthick=1.5,
            elinewidth=1.5,
            capsize=5,
        )
    ax.set_yticks(range(len(conditions)), conditions)
    ax.set_xlim(0.5, 0.85)
    ax.set_ylim(-0.5, len(conditions) - 0.5)
    ax.invert_yaxis()
    if print_stats:
        print("comparison p-values:")
        stats_df = _stats(auc_df, comparisons=comparions)
        print(stats_df)


def plot_neural_behavioural_ve_summary(
    auc_df,
    ve_df,
    conditions=["neural_data", "true_behaviour", "optimal", "random_diffusion"],
    colors=["red", "blue", "dimgrey", "grey"],
    print_stats=True,
    comparions=[
        ("neural_data", ("true_behaviour", "optimal", "random_diffusion")),
        ("true_behaviour", ("optimal", "random_diffusion")),
    ],
    axes=None,
):
    # set up figure with GridSpec for better layout control
    if axes is None:
        f = plt.figure(figsize=(2.5, 2))
        gs = f.add_gridspec(2, 2, width_ratios=(1, 0.4), height_ratios=(0.5, 1), hspace=0.3)
        axes = [f.add_subplot(gs[:, 0]), f.add_subplot(gs[1, 1])]
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].plot([0, ve_df.component.max()], [0, 1], color="k", linestyle="--", lw=1, alpha=1)
    axes[0].set_ylabel("Prop. \n variance explained")
    axes[0].set_xlabel("n components")
    axes[1].set_ylabel("AUC")

    # process cum ve curve data (top axis)
    components = ve_df.component.unique()
    ve = ve_df.groupby(["resample", "component"]).mean().drop(columns=["split"])
    for cond, color in zip(conditions[::-1], colors[::-1]):
        cond_ve = ve[cond].unstack()
        mean = cond_ve.mean()
        std = cond_ve.std()  # sem across subjects from bootstrap perms
        axes[0].plot(components, mean, color=color, linewidth=1.0, alpha=0.8, label=cond)
        axes[0].fill_between(
            components,
            mean - std,
            mean + std,
            color=color,
            alpha=0.2,
        )
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)

    # process AUC data (bottom axis) - inlined from _plot_auc_comparison
    auc = auc_df.groupby("resample").mean().drop(columns=["split"])
    mean_auc = auc.mean()
    auc_lower = auc.quantile(0.025)
    auc_upper = auc.quantile(0.975)
    err_lower = mean_auc - auc_lower
    err_upper = auc_upper - mean_auc
    for i, (cond, color) in enumerate(zip(conditions, colors)):
        axes[1].errorbar(
            x=i,
            y=mean_auc[cond],
            yerr=[[err_lower[cond]], [err_upper[cond]]],
            fmt="o",
            color=color,
            label=cond,
            capsize=0,
            elinewidth=1.5,
            markersize=4,
        )
    axes[1].set_xticks(range(len(conditions)), conditions)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_ylim(0.5, 0.85)
    axes[1].set_xlim(-0.5, len(conditions) - 0.5)
    axes[1].invert_xaxis()

    if print_stats:
        print("comparison p-values:")
        stat_df = _stats(auc_df, comparisons=comparions)
        print(stat_df)


def get_neural_variance_explained_by_synthetic_behaviour(
    maze_name,
    late_sessions=False,  # need as much data as possible with CV approach
    n_splits=5,
    test_size=0.1,
    max_steps_to_goal=30,
    demean=False,
    norm_length=True,
    n_resamples=500,
    verbose=False,
    max_jobs=20,
    save=False,
):
    """
    Similar to original version but with bootstrap resample across subjects, still with X val
    just interested in differences between how well real behaviour explains neurons and how well synthetic
    behaviour explains neurons.
    """
    auc_save_path = RESULTS_DIR / "synthetic_behaviour" / f"{maze_name}_auc_results.parquet"
    ve_save_path = RESULTS_DIR / "synthetic_behaviour" / f"{maze_name}_ve_results.parquet"
    if not save and auc_save_path.exists() and ve_save_path.exists():
        if verbose:
            print(f"Loading data from disk...")
        auc_df = pd.read_parquet(auc_save_path)
        ve_df = pd.read_parquet(ve_save_path)
        return auc_df, ve_df
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
    results_dfs = Parallel(n_jobs=max_jobs)(
        delayed(_process_resample)(subject2split_data, data_types, n, n_splits, demean, norm_length, verbose)
        for n in range(n_resamples)
    )
    auc_results_df = pd.concat([df[0] for df in results_dfs], axis=0)
    ve_results_df = pd.concat([df[1] for df in results_dfs], axis=0)
    if save:
        if verbose:
            print("saving results to disk...")
        auc_save_path.parent.mkdir(parents=True, exist_ok=True)
        ve_save_path.parent.mkdir(parents=True, exist_ok=True)
        auc_results_df.to_parquet(auc_save_path)
        ve_results_df.to_parquet(ve_save_path)
    return auc_results_df, ve_results_df


def _process_resample(subject2split_data, data_types, n, n_splits, demean, norm_length, verbose):
    if verbose:
        print(f"resample: {n}")
    sampled_subjects = np.random.choice(SUBJECT_IDS, size=len(SUBJECT_IDS), replace=True)
    auc_results, ve_results = [], []
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
        split_auc_results = {}
        # explain var in test neural data with...
        neural_test = test_data2df["neural_data"].values
        if demean:
            neural_test = _demean(neural_test)
        if norm_length:
            neural_test = _norm_length(neural_test)
        # each data type
        ve_df = pd.DataFrame(index=range(neural_test.shape[1] + 1), columns=data_types)
        ve_df["component"] = np.arange(neural_test.shape[1] + 1)
        for data_type in data_types:
            d_train = train_data2df[data_type].values
            if demean:
                d_train = _demean(d_train)
            if norm_length:
                d_train = _norm_length(d_train)
            cumsum_ve = get_pca_variance_explained(d_train, neural_test)
            if len(cumsum_ve) < neural_test.shape[1] + 1:
                cumsum_ve = np.concatenate((cumsum_ve, np.ones(neural_test.shape[1] + 1 - len(cumsum_ve))))
            ve_df[data_type] = cumsum_ve
            auc = np.trapz(cumsum_ve, dx=1 / len(cumsum_ve))
            split_auc_results[data_type] = auc
        split_auc_results["split"] = i
        split_auc_results["resample"] = n
        auc_results.append(split_auc_results)
        ve_df["split"] = i
        ve_df["resample"] = n
        ve_results.append(ve_df)
    # combine results
    auc_df = pd.DataFrame(auc_results)
    ve_df = pd.concat(ve_results, axis=0)
    return [auc_df, ve_df]


def _demean(X):
    return X - X.mean(-1, keepdims=True)


def _norm_length(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def get_input_data(
    maze_name,
    with_synthetic_behaviour=True,
    n_splits=5,
    test_size=0.1,
    late=False,
    max_steps_to_goal=30,
    min_split_corr=0.3,
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


def plot_cumulative_variance_explained(results_df, color="blue", ax=None):
    """ """
    # set up figure
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(1.5, 2), clear=True)
    ax.spines[["top", "right"]].set_visible(False)
    n_components = results_df.component.max()
    ax.plot([0, n_components], [0, 1], color="black", ls="--", alpha=0.5)  # baseline
    ax.set_xlabel("n components")
    ax.set_ylabel("prop. \n variance explained")
    # plot
    df = results_df.groupby(["resample", "component"]).mean()  # average over splits
    _df = df["B_explains_B"].unstack()
    c = _df.columns.values
    mean = _df.mean()
    lower = _df.quantile(0.025)
    upper = _df.quantile(0.975)
    ax.plot(c, mean, color=color)
    ax.fill_between(c, lower, upper, color=color, alpha=0.3)


def get_neural_behaviour_variance_explained(
    maze_name,
    input_data=None,
    cv=True,
    n_splits=5,
    test_size=0.1,
    late_sessions=False,
    max_steps_to_goal=30,
    n_resamples=500,
    demean=False,
    norm_length=True,
    verbose=False,
    max_jobs=20,
    save=False,
):
    """ """
    save_path = RESULTS_DIR / "behaviour_explains_neurons" / f"{maze_name}_ve_results.parquet"
    if not save and save_path.exists():
        if verbose:
            print(f"Loading existing results from {save_path}")
        return pd.read_parquet(save_path)
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
    if save:
        results_df.to_parquet(save_path)
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


# %% derive metric for similart of place_direction across subjects


def plot_self_vs_other_cve_curves(auc_df, cve_df, maze_name="maze_1", fill_cve=True, axes=None, print_stats=True):
    """
    Plot within- vs. across-subject VE curves and AUC summary.
    """
    if axes is None:
        f = plt.figure(figsize=(2, 2))
        gs_spec = f.add_gridspec(2, 2, width_ratios=(1, 0.2), height_ratios=(1, 1), hspace=0.3)
        axes = [f.add_subplot(gs_spec[:, 0]), f.add_subplot(gs_spec[1, 1])]
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Prop. \n variance explained")
    axes[0].set_xlabel("n components")
    axes[1].set_ylabel("AUC")
    # processing
    _cve_df = cve_df.copy()
    _auc_df = auc_df.copy()
    if fill_cve:
        _cve_df = fill_cve_df(_cve_df)
    _auc_df = _auc_df[_auc_df.maze_name == maze_name]
    _cve_df = _cve_df[_cve_df.maze_name == maze_name]
    conditions = ["subject_pcs", "other_pcs"]
    labels = ["self", "other"]
    colors = ["black", "gray"]
    components = np.sort(_cve_df["component"].unique())
    axes[0].plot([components.min(), components.max()], [0, 1], color="k", linestyle="--", alpha=0.5)

    # average across splits (and mazes) per subject, then mean ± SEM across subjects
    subject_mean_ve = _cve_df.groupby(["subject", "component"])[conditions].mean().reset_index()
    for cond, label, color in zip(conditions, labels, colors):
        cond_ve = subject_mean_ve.pivot(index="subject", columns="component", values=cond)
        mean = cond_ve.mean()
        sem = cond_ve.sem()
        axes[0].plot(components, mean, color=color, linewidth=1, label=label)
        axes[0].fill_between(components, mean - sem, mean + sem, color=color, alpha=0.2)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)

    # AUC: average across splits (and mazes) per subject, then mean ± SEM across subjects
    auc_cols = ["AUC_subject", "AUC_other"]
    subject_mean_auc = _auc_df.groupby("subject")[auc_cols].mean()
    mean_auc = subject_mean_auc.mean()
    sem_auc = subject_mean_auc.sem()
    for i, (cond, label, color) in enumerate(zip(auc_cols, labels, colors)):
        axes[1].errorbar(
            x=i,
            y=mean_auc[cond],
            yerr=[[sem_auc[cond]], [sem_auc[cond]]],
            fmt="o",
            color=color,
            capsize=0,
            elinewidth=1.5,
            markersize=4,
        )
    axes[1].set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axes[1].set_xlim(-0.5, len(labels) - 0.5)
    axes[1].set_ylim(0.5, 0.8)

    if print_stats:
        t, p = ttest_rel(subject_mean_auc["AUC_subject"], subject_mean_auc["AUC_other"])
        print(f"t={t:.2f}, p={p:.3f}")


def get_place_direction_cross_subject_similarity(
    test_size=0.5,
    n_splits=5,
    late_sessions=False,
    max_steps_to_goal=30,
    demean=False,
    norm_length=True,
    verbose=True,
    save=False,
):
    """ """
    auc_save_path = RESULTS_DIR / "cross_subject_similarity" / "cross_subject_similarity_auc.parquet"
    cve_save_path = RESULTS_DIR / "cross_subject_similarity" / "cross_subject_similarity_cve.parquet"
    if auc_save_path.exists() and not save:
        if verbose:
            print(f"Loading existing results from {auc_save_path}")
        auc_df = pd.read_parquet(auc_save_path)
        cve_df = pd.read_parquet(cve_save_path)
        return auc_df, cve_df
    all_auc_results, all_cve_results = [], []
    for maze_name in ["maze_1", "maze_2", "rooms_maze"]:
        if verbose:
            print(maze_name)
        input_data = get_input_data(
            maze_name=maze_name,
            n_splits=n_splits,
            test_size=test_size,
            late=late_sessions,
            max_steps_to_goal=max_steps_to_goal,
            verbose=verbose,
        )
        auc_results, cve_results = [], []
        for i in range(n_splits):
            if verbose:
                print(f"split: {i}")
            for subject in SUBJECT_IDS:
                if verbose:
                    print(subject)
                other_subjects = [s for s in SUBJECT_IDS if s != subject]
                # get
                subject_train = input_data[subject][i]["neural_data"]["train"]
                subject_test = input_data[subject][i]["neural_data"]["test"]
                other_subjects_test = pd.concat(
                    [input_data[s][i]["neural_data"][t] for t in ["test", "train"] for s in other_subjects], axis=0
                )
                n_test_subject_test_neurons = subject_test.shape[
                    0
                ]  # subsample other subject's data to match the comparison
                other_subjects_test = other_subjects_test.sample(
                    n=n_test_subject_test_neurons, replace=False, random_state=0
                )
                # define arrays
                subject_train_arr = subject_train.values
                subject_test_arr = subject_test.values
                other_subjects_test_arr = other_subjects_test.values
                if demean:
                    subject_train_arr, subject_test_arr, other_subjects_test_arr = [
                        _demean(arr) for arr in [subject_train_arr, subject_test_arr, other_subjects_test_arr]
                    ]
                if norm_length:
                    subject_train_arr, subject_test_arr, other_subjects_test_arr = [
                        _norm_length(arr) for arr in [subject_train_arr, subject_test_arr, other_subjects_test_arr]
                    ]
                # get variance expalin in neurons by within-subject
                _auc_results = {"split": i, "subject": subject}
                cve_df = pd.DataFrame(
                    index=range(min(subject_test_arr.shape) + 1),
                    columns=["subject", "split", "subject_pcs", "other_pcs", "component"],
                )
                for auc_label, cve_label, B in zip(
                    ["AUC_subject", "AUC_other"],
                    ["subject_pcs", "other_pcs"],
                    [subject_test_arr, other_subjects_test_arr],
                ):
                    cumsum_ve = get_pca_variance_explained(B, subject_train_arr)
                    cve_df[cve_label] = cumsum_ve
                    cve_df["component"] = np.arange(len(cumsum_ve))
                    cve_df["split"] = i
                    cve_df["subject"] = subject
                    auc = np.trapz(cumsum_ve, dx=1 / len(cumsum_ve))
                    _auc_results[auc_label] = auc
                auc_results.append(_auc_results)
                cve_results.append(cve_df)
        auc_results_df = pd.DataFrame(auc_results)
        auc_results_df["maze_name"] = maze_name
        cve_results_df = pd.concat(cve_results, axis=0)
        cve_results_df["maze_name"] = maze_name
        all_auc_results.append(auc_results_df)
        all_cve_results.append(cve_results_df)
    auc_results_df = pd.concat(all_auc_results, axis=0)
    cve_results_df = pd.concat(all_cve_results, axis=0)
    if save:
        auc_results_df.to_parquet(auc_save_path)
        cve_results_df.to_parquet(cve_save_path)
    return auc_results_df, cve_results_df


def fill_cve_df(cve_results_df):
    """
    Fill missing rows in cumulative variance explained (CVE) results dataframe.
    """
    # Get max component per maze
    max_components = cve_results_df.groupby("maze_name")["component"].max()

    # Create complete index for all subject/split/maze combinations
    idx = pd.MultiIndex.from_product(
        [
            cve_results_df["subject"].unique(),
            cve_results_df["split"].unique(),
            cve_results_df["maze_name"].unique(),
        ],
        names=["subject", "split", "maze_name"],
    )

    # Expand with components up to the max for each maze
    expanded_rows = []
    for subject, split, maze_name in idx:
        max_comp = max_components[maze_name]
        for component in range(int(max_comp) + 1):
            expanded_rows.append({"subject": subject, "split": split, "maze_name": maze_name, "component": component})

    cve_results_df = pd.DataFrame(expanded_rows).merge(
        cve_results_df, on=["subject", "split", "maze_name", "component"], how="left"
    )

    # Sort for forward fill
    cve_results_df = cve_results_df.sort_values(["subject", "split", "maze_name", "component"])

    # Forward fill values within each subject/split/maze group
    cve_results_df[["subject_pcs", "other_pcs"]] = cve_results_df.groupby(["subject", "split", "maze_name"])[
        ["subject_pcs", "other_pcs"]
    ].ffill()

    return cve_results_df
