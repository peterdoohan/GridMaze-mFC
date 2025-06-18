"""
Library for the analysis of neural tuning to place_direction explaining low dimension structure of subject's behaviour
"""

# %% Imports
import json
from cv2 import norm
import numpy as np
import pandas as pd

from GridMaze.analysis.core import get_sessions as gs

from GridMaze.analysis.behaviour import synthetic_behaviour as sb
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

from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

DATA_FILTER_KWARGS = {
    "navigation_only": True,
    "moving_only": True,
    "exclude_time_at_goal": False,
    "minimum_occupancy": 0.5,
    "max_steps_from_goal": None,
}

# %% Within-Between subject control


def test_within_across_subject_ve(maze="maze_2", demean=True, norm_length=True):
    """
    Null results
    """
    results = []
    for neurons_subject in SUBJECT_IDS:
        for behaviour_subject in SUBJECT_IDS:
            print(neurons_subject, behaviour_subject)
            neuron_sessions = get_analysis_sessions(neurons_subject, maze)
            behaviour_sessions = get_analysis_sessions(behaviour_subject, maze)
            neurons = _get_neural_tuning(neuron_sessions)
            # fill missing neural values with mean
            neurons = neurons.apply(lambda row: row.fillna(row.mean()), axis=1)
            N = neurons.values
            other_subject_behaviour = _get_behavioural_sequences(behaviour_sessions)
            B = other_subject_behaviour.values
            if demean:
                B, N = [arr - arr.mean(-1, keepdims=True) for arr in [B, N]]
            if norm_length:
                B, N = [arr / np.linalg.norm(arr, axis=1, keepdims=True) for arr in [B, N]]
            cum_ve = get_svd_variance_explained(N, B)
            auc = cum_ve.sum()  # area under the curve
            results.append({"neurons_subject": neurons_subject, "behaviour_subject": behaviour_subject, "auc": auc})
    # plotting
    results_df = pd.DataFrame(results)
    within_subject = results_df[results_df.neurons_subject == results_df.behaviour_subject].auc
    between_subject = (
        results_df[results_df.neurons_subject != results_df.behaviour_subject].groupby("neurons_subject").auc.mean()
    )
    diff = within_subject.values - between_subject.values
    t, p = ttest_1samp(diff, 0)
    print(f"t: {t}, p: {p}")
    return (results_df,)


# %% Null vs subject behaviour 2


def get_neural_variance_explained_by_behaviour(
    maze_name,
    n_splits=5,
    test_size=0.5,
    late=True,
    max_steps_to_goal=30,
    demean=False,
    norm_length=True,
    n_resamples=500,
    verobse=True,
):
    """
    Similar to original version but with bootstrap resample across subjects, still with X val
    just interested in differences between how well real behaviour explains neurons and how well synthetic
    behaviour explains neurons.
    """
    # get input data
    subject2split_data = get_input_data(
        maze_name="maze_2", n_splits=n_splits, test_size=0.5, late=True, max_steps_to_goal=30
    )

    # define helper functions
    def _demean(X):
        return X - X.mean(-1, keepdims=True)

    def _norm_length(X):
        return X / np.linalg.norm(X, axis=1, keepdims=True)

    # proceses results across bootstrap resamples across subjects
    all_results = []
    for n in range(n_resamples):
        print(n)
        sampled_subjects = np.random.choice(SUBJECT_IDS, size=len(SUBJECT_IDS), replace=True)
        resample_results = []
        for i in range(n_splits):
            data_types = [
                "neural_data",
                "true_behaviour",
                "random_diffusion",
                "forward_diffusion",
                "vector",
                "optimal",
            ]
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
        all_results.append(df)
    return pd.concat(all_results, axis=0)


def get_input_data(maze_name, n_splits=5, test_size=0.5, late=False, max_steps_to_goal=30):
    """
    should avoid data regeneeration when making per subject Xval splits but not sure if this is overkill
    """
    days_on_maze = "late" if late == True else "all"
    all_data = {}
    subject2session_names = {}
    for subject in SUBJECT_IDS:
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
                min_split_corr=0.3,
                max_steps_from_goal=max_steps_to_goal,
            )
            session_data["true_behaviour"] = bdr.get_session_behavioural_sequences(
                session, normalisation=False, max_steps_to_goal=max_steps_to_goal
            )
            for policy in ["random_diffusion", "forward_diffusion", "vector", "optimal"]:
                session_data[policy] = sb.get_session_synthetic_behavioural_sequences(
                    session, policy=policy, normalisation=False
                )
            session_names.append(session.name)
            subject_data[session.name] = session_data
        all_data[subject] = subject_data
        subject2session_names[subject] = session_names
    # combine data per subject across Xvaled splits
    print("recombining data")
    subject2split_data = {}
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
    for subject in SUBJECT_IDS:
        _session_names = np.array(subject2session_names[subject])
        # Generate the splits (session names)
        split2data = {}
        for i, (train_index, test_index) in enumerate(ss.split(_session_names)):
            train, test = _session_names[train_index], _session_names[test_index]
            split_data = {}
            for data_type in [
                "neural_data",
                "true_behaviour",
                "random_diffusion",
                "forward_diffusion",
                "vector",
                "optimal",
            ]:
                train_data = [all_data[subject][session][data_type] for session in train]
                test_data = [all_data[subject][session][data_type] for session in test]
                split_data[data_type] = {
                    "train": pd.concat([df for df in train_data if df is not None], axis=0),
                    "test": pd.concat([df for df in test_data if df is not None], axis=0),
                }
            split2data[i] = split_data
        subject2split_data[subject] = split2data
    return subject2split_data


# %% Null vs subject behaviour


def test(maze="maze_1", demean=False, norm_length=True):
    """
    Switch to behaviour -explains-> neurons so that neurons are constant and behaviour is changed.
    Do analysis sepeartely for each subject, with output metric auc BeN / auc NeN -> t-test across
    subjects for true vs synthetic behavioural policy.
    """
    NeNs = []  # [n_splits, n_components]
    BeBs, BeNs, NeBs = [], [], []  # [n_splits, n_policies, n_components]
    sessions = get_analysis_sessions("all", maze, late=True)
    split_sessions = _get_session_splits(sessions, n_splits=5, test_size=0.2)
    NeNs = []  # [n_splits]
    BeBs, BeNs, NeBs = [], [], []  # [n_splits, n_policies]
    for i, (train_sessions, test_sessions) in enumerate(split_sessions):
        print(f"Split {i+1} of {len(split_sessions)}")
        train_neural = pdr.get_population_place_direction_tuning(
            sessions=train_sessions, fill_nans="mean", normalisation=False
        ).values
        test_neural = pdr.get_population_place_direction_tuning(
            sessions=test_sessions, fill_nans="mean", normalisation=False
        ).values
        # get varaince explained under different behavioural policues (real data or synthetic)
        split_BeNs, split_BeBs, split_NeBs = [], [], []
        for policy in [None, "random_diffusion", "forward_diffusion", "vector", "optimal"]:
            print(f"Policy: {policy}")
            if policy is None:  # real data
                train_behaviour = bdr.get_maze_behavioural_sequences_df(
                    sessions=train_sessions, normalisation=False
                ).values
                test_behaviour = bdr.get_maze_behavioural_sequences_df(
                    sessions=test_sessions, normalisation=False
                ).values
            else:
                train_behaviour = sb.get_synthetic_maze_behavioural_sequences_df(
                    policy=policy, sessions=train_sessions, normalisation=False
                ).values
                test_behaviour = sb.get_synthetic_maze_behavioural_sequences_df(
                    policy=policy, sessions=test_sessions, normalisation=False
                ).values
            if demean:
                train_neural, test_neural, train_behaviour, test_behaviour = [
                    arr - arr.mean(-1, keepdims=True)
                    for arr in [train_neural, test_neural, train_behaviour, test_behaviour]
                ]
            if norm_length:
                train_neural, test_neural, train_behaviour, test_behaviour = [
                    arr / np.linalg.norm(arr, axis=1, keepdims=True)
                    for arr in [train_neural, test_neural, train_behaviour, test_behaviour]
                ]
            # variance explained
            split_BeNs.append(get_pca_variance_explained(train_behaviour, test_neural))
            split_BeBs.append(get_pca_variance_explained(train_behaviour, test_behaviour))
            split_NeBs.append(get_pca_variance_explained(train_neural, test_behaviour))
        BeNs.append(np.array(split_BeNs))
        BeBs.append(np.array(split_BeBs))
        NeBs.append(np.array(split_NeBs))
        NeN = get_pca_variance_explained(train_neural, test_neural)
        NeNs.append(NeN)
    # build results df
    auc_df = []
    for j in range(5):  # n_xval splits
        AUC_nen = NeNs[j].sum()
        auc_df.append({"policy": "Real", "fold": j, "auc": AUC_nen, "type": "NeN"})
        for k, policy in enumerate(["Real", "Random", "Forward", "Vector", "Optimal"]):
            AUC_ben = BeNs[j][k].sum()
            AUC_beb = BeBs[j][k].sum()
            AUC_neb = NeBs[j][k].sum()
            auc_df.append({"policy": policy, "fold": j, "auc": AUC_ben, "type": "BeN"})
            auc_df.append({"policy": policy, "fold": j, "auc": AUC_beb, "type": "BeB"})
            auc_df.append({"policy": policy, "fold": j, "auc": AUC_neb, "type": "NeB"})
    auc_df = pd.DataFrame(auc_df)
    return auc_df


def true_vs_random_plot(auc_df, ax=None):
    """ """
    # BeN - NeN test
    policies = ["Real", "Random"]
    r_hats = []
    NeN = auc_df[auc_df.type == "NeN"].set_index(["fold"]).auc
    for policy in policies:
        BeN = auc_df[(auc_df.type == "BeN") & (auc_df.policy == policy)].set_index(["fold"]).auc
        r_hat = BeN.div(NeN).groupby("fold").mean()
        r_hats.append(r_hat)
    r_hats = pd.concat(r_hats, axis=1)
    r_hats.columns = policies
    r1 = r_hats.reset_index().melt(id_vars="fold", var_name="policy", value_name="r")
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3), clear=True)
    ax.spines[["top", "right"]].set_visible(False)
    sns.pointplot(r1, x="policy", y="r", ax=ax, legend=False, color="grey")
    ax.set_ylim(0.7, 0.9)


def test_ve_diff(auc_df):
    """ """
    # BeN - NeN test
    policies = ["Real", "Random", "Forward", "Vector", "Optimal"]
    r_hats = []
    NeN = auc_df[auc_df.type == "NeN"].set_index(["fold"]).auc
    for policy in policies:
        BeN = auc_df[(auc_df.type == "BeN") & (auc_df.policy == policy)].set_index(["fold"]).auc
        r_hat = BeN.div(NeN).groupby("fold").mean()
        r_hats.append(r_hat)
    r_hats = pd.concat(r_hats, axis=1)
    r_hats.columns = policies
    r1 = r_hats.reset_index().melt(id_vars="fold", var_name="policy", value_name="r")
    f1, ax = plt.subplots()
    sns.swarmplot(r1, x="policy", y="r", hue="fold", ax=ax, legend=False)
    ax.set_ylim(0.5, 1)
    # BeB - NeB test
    r_hats = []
    for policy in policies:
        BeB = auc_df[(auc_df.type == "BeB") & (auc_df.policy == policy)].set_index(["fold"]).auc
        NeB = auc_df[(auc_df.type == "NeB") & (auc_df.policy == policy)].set_index(["fold"]).auc
        r_hat = NeB.div(BeB).groupby("fold").mean()
        r_hats.append(r_hat)
    r_hats = pd.concat(r_hats, axis=1)
    r_hats.columns = policies
    r2 = r_hats.reset_index().melt(id_vars="fold", var_name="policy", value_name="r")
    f2, ax = plt.subplots()
    sns.swarmplot(r2, x="policy", y="r", hue="fold", ax=ax, legend=False)
    ax.set_ylim(0.5, 1)
    return


def plot_subject_cum_ve(NeNs, BeBs, BeNs, NeBs, policies=["Real", "Random", "Forward", "Vector", "Optimal"]):
    """ """
    policy_colors = ["blue", "green", "purple", "orange", "brown"]
    mean_NeN, mean_BeBs, mean_BeNs, mean_NeBs = [arr.mean(axis=0) for arr in [NeNs, BeBs, BeNs, NeBs]]
    sem_NeN, sem_BeBs, sem_BeNs, sem_NeBs = [
        arr.std(axis=0) / np.sqrt(arr.shape[0]) for arr in [NeNs, BeBs, BeNs, NeBs]
    ]
    n_components = NeNs.shape[-1]

    ## Fig 1: Neurons explained by x
    f1, ax = plt.subplots(1, 1, figsize=(5, 5), clear=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot([0, n_components], [0, 1], color="black", ls="--")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cum. var exp")
    ax.set_title("Neurons explained by")
    # Neurons -explain-> Neurons
    ax.plot(mean_NeN, label="Neurons", color="red")
    ax.fill_between(range(len(mean_NeN)), mean_NeN - sem_NeN, mean_NeN + sem_NeN, color="red", alpha=0.3)
    # Behaviour -explain-> Behaviour
    for i, (policy, color) in enumerate(zip(policies, policy_colors)):
        mean_BeN = mean_BeNs[i, :]
        sem_BeN = sem_BeNs[i, :]
        ax.plot(mean_BeN, label=f"Behaviour {policy}", color=color)
        ax.fill_between(range(len(mean_NeN)), mean_BeN - sem_BeN, mean_BeN + sem_BeN, color=color, alpha=0.3)
    ax.legend(fontsize="xx-small")

    ## Fig 2: Behaviour explained by x
    f2, axes = plt.subplots(1, len(policies), figsize=(4 * len(policies), 4), clear=True, sharex=True, sharey=True)
    for i, (policy, ax) in enumerate(zip(policies, axes.flatten())):
        ax.plot([0, n_components], [0, 1], color="black", ls="--")
        mean_BeB = mean_BeBs[i, :]
        sem_BeB = sem_BeBs[i, :]
        ax.plot(mean_BeB, label="Behaviour", color="blue")
        ax.fill_between(range(len(mean_BeB)), mean_BeB - sem_BeB, mean_BeB + sem_BeB, color="blue", alpha=0.3)
        mean_NeB = mean_NeBs[i, :]
        sem_NeB = sem_NeBs[i, :]
        ax.plot(mean_NeB, label="Neurons", color="red")
        ax.fill_between(range(len(mean_NeB)), mean_NeB - sem_NeB, mean_NeB + sem_NeB, color="red", alpha=0.3)
        ax.set_ylabel("Cum. var exp")
        ax.set_xlabel("Number of components")
        ax.set_title(policy)
        ax.legend(fontsize="xx-small")

    return


# %% Variance explained analysis (SVD)


def run_neuron_to_behaviour_variance_explained_analysis(
    X, ve_method="pca", demean=(False, False), norm_length=(True, True), plot=True
):
    """
    X[0].keys = ["neurons, "behaviour]
    demean[0]: bool, demean neural data
    demean[1]: bool, demean behaviour data
    norm_length[0]: bool, normalise length of neural data
    norm_length[1]: bool, normalise length of behaviour data

    Note neural data in X already has nans filled with mean from fn: get_population_place_direction_tuning
    """
    if ve_method == "pca":
        ve_fn = get_pca_variance_explained
    elif ve_method == "svd":
        ve_fn = get_svd_variance_explained
    else:
        raise NotImplementedError(f"ve_method {ve_method} not recognised")
    n_components = X[0]["neurons"]["train"].shape[-1]
    results = np.zeros((len(X), 4, n_components + 1))  # [n_splits, 4, n_components]
    for i, data in enumerate(X):
        # neural and behavioural data
        train_neurons, test_neurons = data["neurons"]["train"].values, data["neurons"]["test"].values
        train_behaviour, test_behaviour = data["behaviour"]["train"].values, data["behaviour"]["test"].values
        # demean
        data = []
        for (test, train), _demean in zip([(test_neurons, train_neurons), (test_behaviour, train_behaviour)], demean):
            if _demean:
                test, train = [arr - arr.mean(-1, keepdims=True) for arr in [test, train]]
            data.append(test)
            data.append(train)
        test_neurons, train_neurons, test_behaviour, train_behaviour = data
        # normalise length
        data = []
        for (test, train), _norm_length in zip(
            [(test_neurons, train_neurons), (test_behaviour, train_behaviour)], norm_length
        ):
            if _norm_length:
                test, train = [arr / np.linalg.norm(arr, axis=1, keepdims=True) for arr in [test, train]]
            data.append(test)
            data.append(train)
        test_neurons, train_neurons, test_behaviour, train_behaviour = data
        # calculate variance explained
        beb = ve_fn(train_behaviour, test_behaviour)
        nen = ve_fn(train_neurons, test_neurons)
        ben = ve_fn(train_behaviour, test_neurons)
        neb = ve_fn(train_neurons, test_behaviour)
        results[i] = np.array([beb, nen, ben, neb])
    # plotting (make pretty later)
    if plot:
        f, axes = plt.subplots(1, 2, figsize=(5, 3), clear=True, sharex=True, sharey=True)
        for ax in axes.flatten():
            ax.spines[["top", "right"]].set_visible(False)
            ax.plot([0, n_components], [0, 1], color="black", ls="--")
        # behaviour explains plot
        beb_mean = results[:, 0].mean(axis=0)
        beb_sem = results[:, 0].std(axis=0) / np.sqrt(results.shape[0])
        axes[0].plot(beb_mean, label="Behaviour", color="blue")
        axes[0].fill_between(range(len(beb_mean)), beb_mean - beb_sem, beb_mean + beb_sem, color="blue", alpha=0.3)
        neb_mean = results[:, 3].mean(axis=0)
        neb_sem = results[:, 3].std(axis=0) / np.sqrt(results.shape[0])
        axes[0].plot(neb_mean, label="Neurons", color="red")
        axes[0].fill_between(range(len(neb_mean)), neb_mean - neb_sem, neb_mean + neb_sem, color="red", alpha=0.3)
        axes[0].set_xlabel("Number of components")
        axes[0].set_ylabel("Cum. var exp")
        axes[0].set_title("Behaviour explained by")
        axes[0].legend(fontsize="xx-small")
        # neurons explains plot
        nen_mean = results[:, 1].mean(axis=0)
        nen_sem = results[:, 1].std(axis=0) / np.sqrt(results.shape[0])
        axes[1].plot(nen_mean, label="Neurons", color="red")
        axes[1].fill_between(range(len(nen_mean)), nen_mean - nen_sem, nen_mean + nen_sem, color="red", alpha=0.3)
        ben_mean = results[:, 2].mean(axis=0)
        ben_sem = results[:, 2].std(axis=0) / np.sqrt(results.shape[0])
        axes[1].plot(ben_mean, label="Behaviour", color="blue")
        axes[1].fill_between(range(len(ben_mean)), ben_mean - ben_sem, ben_mean + ben_sem, color="blue", alpha=0.3)
        axes[1].legend(fontsize="xx-small")
        axes[1].set_xlabel("Number of components")
        axes[1].set_title("Neurons explained by")
        f.tight_layout()
        f.subplots_adjust(wspace=0.8)
    return results


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


# %% Main input data function


def get_joint_neural_behaviour_place_direction_dfs(sessions, n_splits=5, test_size=0.2, synthetic_behaviour=False):
    """
    synthetic_behaviour in [False, "random_diffusion", "forward_diffusion", "vector", "optimal"]
    """
    split_sessions = _get_session_splits(sessions, n_splits, test_size)
    X = []
    for train_sessions, test_sessions in split_sessions:
        split_data = {}
        split_data["neurons"] = {  # df [n_neurons, n_place_directions]
            "train": pdr.get_population_place_direction_tuning(
                sessions=train_sessions, fill_nans="mean", normalisation=False
            ),
            "test": pdr.get_population_place_direction_tuning(
                sessions=test_sessions, fill_nans="mean", normalisation=False
            ),
        }

        if not synthetic_behaviour:
            split_data["behaviour"] = {  # df [n_trials, n_place_directions]
                "train": bdr.get_maze_behavioural_sequences_df(sessions=train_sessions),
                "test": bdr.get_maze_behavioural_sequences_df(sessions=test_sessions),
            }

        else:
            policy = synthetic_behaviour
            split_data["behaviour"] = {  # df [n_trials, n_place_directions]
                "train": sb.get_synthetic_maze_behavioural_sequences_df(
                    policy=policy,
                    sessions=train_sessions,
                ),
                "test": sb.get_synthetic_maze_behavioural_sequences_df(
                    policy=policy,
                    sessions=test_sessions,
                ),
            }
        X.append(split_data)
    return X


def _get_session_splits(sessions, n_splits, test_size):
    """ """
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
    _sessions = np.array(sessions)
    # Generate the splits
    splits = []
    for train_index, test_index in ss.split(_sessions):
        train, test = _sessions[train_index], _sessions[test_index]
        splits.append((train, test))
    return splits


def get_analysis_sessions(subject, maze, late=True):
    """ """
    subject = [subject] if not subject == "all" else subject
    days_on_maze = "late" if late == True else "all"
    sessions = gs.get_maze_sessions(
        subject_IDs=subject,
        maze_names=[maze],
        days_on_maze=days_on_maze,
        with_data=[
            "navigation_df",
            "navigation_spike_rates_df",
            "trajectory_decisions_df",
            "cluster_metrics",
            "cluster_place_direction_tuning_metrics",
        ],
    )
    return sessions
