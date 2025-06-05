"""
Test if low dimensional structure of neural tuning to state-action can efficently reconstruct (explain variance)
in the subject's behavioural trajectories.
"""

# %% Imports
import json
import numpy as np
import pandas as pd

from GridMaze.analysis.core import get_sessions as gs

# from GridMaze.analysis.processing.get_cluster_heatmap_dfs import _get_place_direction_df
from GridMaze.analysis.efficient_coding import control_behaviour as cb
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


# %% Null vs subject behaviour


def test(maze="maze_1", demean=True, norm_length=True):
    """
    Switch to behaviour -explains-> neurons so that neurons are constant and behaviour is changed.
    Do analysis sepeartely for each subject, with output metric auc BeN / auc NeN -> t-test across
    subjects for true vs synthetic behavioural policy.
    """
    subject_NeNs = []  # [n_subjects, n_splits, n_components]
    subject_BeBs, subject_BeNs, subject_NeBs = [], [], []  # [n_subjects, n_splits, n_policies, n_components]
    for subject in SUBJECT_IDS:
        print(f"--- Loading {subject} data ---")
        subject_sessions = get_analysis_sessions(
            subject, maze, late=False
        )  # use all sessions to get as much data as possible
        split_sessions = _get_session_splits(subject_sessions, n_splits=5, test_size=0.2)
        NeNs = []  # [n_splits]
        BeBs, BeNs, NeBs = [], [], []  # [n_splits, n_policies]
        for i, (train_sessions, test_sessions) in enumerate(split_sessions):
            print(f"Split {i+1} of {len(split_sessions)}")
            train_neural = _get_neural_tuning(train_sessions)
            test_neural = _get_neural_tuning(test_sessions)
            # fill missing neural values with mean
            train_neural = train_neural.apply(lambda row: row.fillna(row.mean()), axis=1).values
            test_neural = test_neural.apply(lambda row: row.fillna(row.mean()), axis=1).values
            # get varaince explained under different behavioural policues (real data or synthetic)
            split_BeNs, split_BeBs, split_NeBs = [], [], []
            for policy in [False, "random_diffusion", "forward_diffusion", "vector", "optimal"]:
                print(f"Policy: {policy}")
                train_behaviour = _get_behavioural_sequences(train_sessions, synthetic=policy).values
                test_behaviour = _get_behavioural_sequences(test_sessions, synthetic=policy).values
                # demean and normalise
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
                split_BeNs.append(get_svd_variance_explained(train_behaviour, test_neural))
                split_BeBs.append(get_svd_variance_explained(train_behaviour, test_behaviour))
                split_NeBs.append(get_svd_variance_explained(train_neural, test_behaviour))
            BeNs.append(np.array(split_BeNs))
            BeBs.append(np.array(split_BeBs))
            NeBs.append(np.array(split_NeBs))
            NeN = get_svd_variance_explained(train_neural, test_neural)
            NeNs.append(NeN)
        subject_NeNs.append(np.array(NeNs))
        subject_BeBs.append(np.array(BeBs))
        subject_BeNs.append(np.array(BeNs))
        subject_NeBs.append(np.array(NeBs))
    # build results df
    auc_df = []
    for i, subject in enumerate(SUBJECT_IDS):
        for j in range(5):  # n_xval splits
            AUC_nen = subject_NeNs[i][j].sum()
            auc_df.append({"subject": subject, "policy": "Real", "fold": j, "auc": AUC_nen, "type": "NeN"})
            for k, policy in enumerate(["Real", "Random", "Forward", "Vector", "Optimal"]):
                AUC_ben = subject_BeNs[i][j][k].sum()
                AUC_beb = subject_BeBs[i][j][k].sum()
                AUC_neb = subject_NeBs[i][j][k].sum()
                auc_df.append({"subject": subject, "policy": policy, "fold": j, "auc": AUC_ben, "type": "BeN"})
                auc_df.append({"subject": subject, "policy": policy, "fold": j, "auc": AUC_beb, "type": "BeB"})
                auc_df.append({"subject": subject, "policy": policy, "fold": j, "auc": AUC_neb, "type": "NeB"})
    auc_df = pd.DataFrame(auc_df)

    return np.array(subject_NeNs), np.array(subject_BeBs), np.array(subject_BeNs), np.array(subject_NeBs)


def test_ve_diff(auc_df):
    """ """
    # BeN - NeN test
    policies = ["Real", "Random", "Forward", "Vector", "Optimal"]
    r_hats = []
    NeN = auc_df[auc_df.type == "NeN"].set_index(["subject", "fold"]).auc
    for policy in policies:
        BeN = auc_df[(auc_df.type == "BeN") & (auc_df.policy == policy)].set_index(["subject", "fold"]).auc
        r_hat = BeN.div(NeN).groupby("subject").mean()
        r_hats.append(r_hat)
    r_hats = pd.concat(r_hats, axis=1)
    r_hats.columns = policies
    r1 = r_hats.reset_index().melt(id_vars="subject", var_name="policy", value_name="r")
    f1, ax = plt.subplots()
    sns.swarmplot(r1, x="policy", y="r", hue="subject", ax=ax, legend=False)
    ax.set_ylim(0.5, 1)
    # BeB - NeB test
    r_hats = []
    for policy in policies:
        BeB = auc_df[(auc_df.type == "BeB") & (auc_df.policy == policy)].set_index(["subject", "fold"]).auc
        NeB = auc_df[(auc_df.type == "NeB") & (auc_df.policy == policy)].set_index(["subject", "fold"]).auc
        r_hat = NeB.div(BeB).groupby("subject").mean()
        r_hats.append(r_hat)
    r_hats = pd.concat(r_hats, axis=1)
    r_hats.columns = policies
    r2 = r_hats.reset_index().melt(id_vars="subject", var_name="policy", value_name="r")
    f2, ax = plt.subplots()
    sns.swarmplot(r2, x="policy", y="r", hue="subject", ax=ax, legend=False)
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


def run_main_analysis(X, ve_method="svd", demean=True, norm_length=True, plot=True):
    """"""
    if ve_method == "pca":
        ve_fn = get_pca_variance_explained
    elif ve_method == "svd":
        ve_fn = get_svd_variance_explained
    else:
        raise NotImplementedError(f"ve_method {ve_method} not recognised")
    n_components = X[0]["neurons"]["train"].shape[-1]
    results = np.zeros((len(X), 4, n_components + 1))  # [n_splits, 4, n_components]
    for i, data in enumerate(X):
        print(i)
        # neural data
        train_neurons, test_neurons = data["neurons"]["train"], data["neurons"]["test"]
        # fill nans in neural data with mean (unvistied place-directions)
        train_neurons = train_neurons.apply(lambda row: row.fillna(row.mean()), axis=1).values
        test_neurons = test_neurons.apply(lambda row: row.fillna(row.mean()), axis=1).values
        # behaviour data
        train_behaviour, test_behaviour = data["behaviour"]["train"].values, data["behaviour"]["test"].values
        if demean:
            train_neurons, test_neurons, train_behaviour, test_behaviour = [
                arr - arr.mean(-1, keepdims=True)
                for arr in [train_neurons, test_neurons, train_behaviour, test_behaviour]
            ]
        if norm_length:
            train_neurons, test_neurons, train_behaviour, test_behaviour = [
                arr / np.linalg.norm(arr, axis=1, keepdims=True)
                for arr in [train_neurons, test_neurons, train_behaviour, test_behaviour]
            ]

        beb = ve_fn(train_behaviour, test_behaviour)
        nen = ve_fn(train_neurons, test_neurons)
        ben = ve_fn(train_behaviour, test_neurons)
        neb = ve_fn(train_neurons, test_behaviour)
        print(beb.shape)
        print(nen.shape)
        print(ben.shape)
        print(neb.shape)
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
        # axes[0].set_xlim([0, 20])
        # axes[1].set_xlim([0, 20])
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


def get_joint_neural_behaviour_place_direction_dfs(sessions, n_splits=5, test_size=0.2):
    """ """
    split_sessions = _get_session_splits(sessions, n_splits, test_size)
    X = []
    for train_sessions, test_session in split_sessions:
        X.append(
            {
                "neurons": {
                    "train": _get_neural_tuning(train_sessions),  # df [n_neurons, n_place_directions]
                    "test": _get_neural_tuning(test_session),
                },
                "behaviour": {
                    "train": _get_behavioural_sequences(train_sessions),  # df [n_trials, n_place_directions]
                    "test": _get_behavioural_sequences(test_session),
                },
            }
        )
    return X


# %% Behaviour Sequences


def _get_behavioural_sequences(
    sessions,
    max_steps_from_goal=DATA_FILTER_KWARGS["max_steps_from_goal"],
    synthetic=False,
):
    """
    Behavioural sequences are binary vectors of place-direction pairs visited in each trial
    if synthetic in ["optimal", "random_diffusion", "forward_diffusion"] then the behaviour is generated
    synthetically (for control analyses)
    """
    trials_sequences = []
    place_direction2idx = {_pd: i for i, _pd in enumerate(mr.get_maze_place_direction_pairs(sessions[0].simple_maze()))}
    for session in sessions:
        # load
        if not synthetic:
            trajectories_df = session.trajectory_decisions_df
        else:
            trajectories_df = cb.get_synthetic_behaviour(session, policy=synthetic)
        # filter
        trajectories_df = trajectories_df[(trajectories_df.trial_phase == "navigation")]
        trajectories_df = trajectories_df[(trajectories_df.maze_position.notnull())]
        trajectories_df = trajectories_df[(trajectories_df.action.notnull())]
        if max_steps_from_goal is not None:
            trajectories_df = trajectories_df[(trajectories_df.steps_to_goal.lt(max_steps_from_goal))]
        # loop over trials to construct sequence vectors (binary in each place-direction, 1 if visited)
        trials = trajectories_df.trial.unique()
        session_sequences = np.zeros((len(trials), len(place_direction2idx)), dtype=int)
        for i, trial in enumerate(trials):
            trial_df = trajectories_df[trajectories_df.trial == trial]
            place_direction_sequence = list(zip(trial_df.maze_position, trial_df.action))
            for j in place_direction_sequence:
                session_sequences[i, place_direction2idx[j]] += 1
        trials_sequences.append(session_sequences)
    behaviour_df = pd.DataFrame(
        data=np.vstack(trials_sequences),
        columns=pd.MultiIndex.from_tuples(place_direction2idx.keys(), names=["maze_position", "direction"]),
    )
    return behaviour_df.sort_index(axis=1)


def _get_behavioural_sequences_ALT(sessions, filter_kwargs=DATA_FILTER_KWARGS):
    """
    Behavioural sequences defined over frames not node-egde-node transitions as in _get_behavioural_sequences

    Slighly slower var exp, although very slighly
    """
    trial_sequences = []
    place_direction2idx = {_pd: i for i, _pd in enumerate(mr.get_maze_place_direction_pairs(sessions[0].simple_maze()))}
    for session in sessions:
        navigation_df = session.navigation_df
        navigation_df = filt.filter_navigation_rates_df(
            navigation_df,
            navigation_only=filter_kwargs["navigation_only"],
            moving_only=filter_kwargs["moving_only"],
            exclude_time_at_goal=filter_kwargs["exclude_time_at_goal"],
            max_steps_to_goal=filter_kwargs["max_steps_from_goal"],
        )
        # filter edge cases that lack place-direction information
        navigation_df = navigation_df[navigation_df.maze_position.simple.notnull()]
        navigation_df = navigation_df[navigation_df.cardinal_movement_direction.notnull()]
        trials = navigation_df.trial.unique()
        # build binary reps of trails in place-direction space
        session_sequences = np.zeros((len(trials), len(place_direction2idx)), dtype=int)
        for i, trial in enumerate(trials):
            trial_df = navigation_df[navigation_df.trial == trial]
            place_direction_sequence = list(zip(trial_df.maze_position.simple, trial_df.cardinal_movement_direction))
            for j in place_direction_sequence:
                session_sequences[i, place_direction2idx[j]] += 1
        trial_sequences.append(session_sequences)
    behaviour_df = pd.DataFrame(
        data=np.vstack(trial_sequences),
        columns=pd.MultiIndex.from_tuples(place_direction2idx.keys(), names=["maze_position", "direction"]),
    )
    return behaviour_df.sort_index(axis=1)


# %% Neural Tuning Functions


def _get_neural_tuning(
    sessions,
    filter_kwargs=DATA_FILTER_KWARGS,
    min_firing_rate=1,  # Hz
):
    """
    min_firing_rate filters for neurons inactive during navigation
    """
    all_place_direction_rates = []
    for session in sessions:
        # load
        simple_maze = session.simple_maze()
        navigation_rates_df = session.get_navigation_activity_df(
            type="rates", cluster_kwargs={"single_units": True, "multi_units": False}
        )
        # filter & calculate average firing rates in each place-direction
        place_direction_rates = _get_place_direction_df(
            simple_maze, navigation_rates_df, **filter_kwargs  # [n_neurons, n_place_directions]
        )
        # exclude neurons with low firing rate during navigation
        all_place_direction_rates.append(place_direction_rates[place_direction_rates.mean(axis=1).gt(min_firing_rate)])
    return pd.concat(all_place_direction_rates, axis=0).sort_index(axis=1)


# %% load data


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
        ],
    )
    return sessions
