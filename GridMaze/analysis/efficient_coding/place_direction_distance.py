"""
Test if low dimensional structure of neural tuning to state-action, estimated via embedding mode can 
efficently reconstruct (explain variance) in the subject's behavioural trajectories, represented in the same
place-direction x distance to goal space.
"""

# %% Imports
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from matplotlib import pyplot as plt

from GridMaze.analysis.core import convert
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.maze import representations as mr

from GridMaze.analysis.embedding_model import load_experiment as le
from GridMaze.analysis.embedding_model import plot_latents as pl
from GridMaze.analysis.embedding_model import place_direction_distance_occupancies as occ

from GridMaze.analysis.efficient_coding.place_direction import _get_session_splits


# %% Globabl Variables
from GridMaze.analysis.embedding_model.run_experiment import DEFAULT_INPUT_KWARGS

EXP_SET = "example_models"

# %% Functions


def full_analysis_for_Kris():
    sessions = get_analysis_sessions("all", "maze_1")
    input_data = get_joint_neural_behaviour_place_direction_distance_dfs(sessions)
    test(input_data)  # working version of main analysis
    return


def test(X, demean=True):
    beb_results = []
    r2_results = []
    for data in X:
        behaviour_train = data["behaviour"]["train"]
        behaviour_test = data["behaviour"]["test"]
        # mask states not visited
        occ_mask = (
            pd.concat([behaviour_train, behaviour_test]).sum(axis=0).gt(0)
        )  # mask states not visited (maybe we should just mask based on train?)
        # occ_mask = behaviour_train.sum(axis=0).gt(0) # ??
        behaviour_train = behaviour_train.loc[:, occ_mask]
        behaviour_test = behaviour_test.loc[:, occ_mask]
        B_train = behaviour_train.values.T  # [n_features, n_trial]
        B_test = behaviour_test.values.T  # [n_features, n_trial]
        if demean:
            B_train = B_train - B_train.mean(
                axis=0, keepdims=True
            )  # for each trial, subtract the mean occupancy across state-direction-distances
            B_test = B_test - B_test.mean(axis=0, keepdims=True)
        # use behavioural basis to reconstruct behaviour (upper bound, xvaled)
        beb = get_svd_variance_explained(B_train.T, B_test.T)
        beb_results.append(np.array(beb))
        R2s = []
        for n_latents, neural_latent_tuning in data["neural_latents"].items():
            # use neural basis to reconstruct behaviour
            neural_latent_train = neural_latent_tuning.loc[:, occ_mask]
            neural_latent_test = neural_latent_tuning.loc[:, occ_mask]  # not cross validated yet
            N_train = neural_latent_train.values.T  # [n_features, n_components]
            N_test = neural_latent_test.values.T
            if demean:
                N_train = N_train - N_train.mean(
                    axis=0, keepdims=True
                )  # for each latent, subtract the mean occupancy across state-direction-distances
                N_test = N_test - N_test.mean(axis=0, keepdims=True)
            # neb = varexp(B_train, N_test)
            neb = varexp(N_train, B_test)
            R2s.append(neb)
        r2_results.append(R2s)
    # plotting
    beb_results = np.array(beb_results)
    r2_results = np.array(r2_results)
    f, ax = plt.subplots(1, 1, figsize=(3, 3), clear=True)
    mean_beb = np.mean(beb_results, axis=0)
    sem_beb = np.std(beb_results, axis=0) / np.sqrt(beb_results.shape[0])
    ax.plot(mean_beb, label="Behavioural")
    ax.fill_between(np.arange(mean_beb.size), mean_beb - sem_beb, mean_beb + sem_beb, alpha=0.5)
    ax.plot([0, mean_beb.size], [0, 1], color="black", ls="--")
    n_latents = np.array(list(data["neural_latents"].keys()))
    mean_r2 = np.mean(r2_results, axis=0)
    ax.scatter(n_latents + 1, mean_r2, color="red", marker="x", label="Neural")
    ax.set_xlim([0, n_latents[-1] + 5])
    ax.set_ylim([0, 0.4])
    ax.set_xlabel("Number of latents/components")
    ax.set_ylabel("Cum. var exp")
    ax.legend(fontsize="xx-small", loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    return beb_results, r2_results


# %%


def varexp(A, B):  # A & B: [n_features, n_samples]
    """ """
    W, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)  # lstsq(a, b) solves b = a @ w (W is dim x Ntest)
    B_hat = A @ W
    TSS = np.sum((B - B.mean(axis=0)) ** 2)  # total sum of squares
    SSE = np.sum((B - B_hat) ** 2)  # sum of squared errors
    R2 = 1.0 - SSE / TSS
    return R2


def test_varexp_lstsq(A, B, dims=np.arange(1, 101)):  # A & B: [n_features, n_samples]
    """Compares the cumulative variance of matrix B that's explained by the first i orthonormal bases of matrix A using SVD,
    with the approach that uses least squares."""
    U, Sigma, Vt = np.linalg.svd(A.T, full_matrices=True)  # rows of Vt are normalized eigenvectors

    r2s_lstsq = []
    for dim in dims:
        basis = Vt[:dim, :]  # basis (dim x features)
        W, residuals, rank, s = np.linalg.lstsq(
            basis.T, B, rcond=None
        )  # lstsq(a, b) solves b = a @ w (W is dim x Ntest)
        B_hat = basis.T @ W
        TSS = np.sum((B - B.mean(axis=0)) ** 2)  # total sum of squares
        SSE = np.sum((B - B_hat) ** 2)  # sum of squared errors
        r2s_lstsq.append(1.0 - SSE / TSS)

    M = (
        Vt @ B
    )  # B projected onto the svd bases of A (rowsof Vt); component on each eigvec (rows) for each data point (cols)
    pc_exp_var = np.square(M).mean(axis=1)  # average projection strength across data points for each eigenvector
    cumsum_exp_var = np.cumsum(pc_exp_var) / pc_exp_var.sum()  # normalize
    r2s_svd = cumsum_exp_var[dims - 1]

    assert np.all(np.abs(cumsum_exp_var[:100] - np.array(r2s)) < 1e-8)

    return np.concatenate(([0], cumsum_exp_var))


def varexp_xval(A_test, A_train, B_test, B_train):
    """
    Cross valdiated test for variance explained by the basis set B in matrix A:
        Solve for W in A_train ~ B_train . W
        Use W to predict A_test from B_test basis
        Calculate R2

    Inputs:
        - A_test, A_train: np.array [n_features, n_trials]
        - B_test, B_train: np.array [n_features, n_components]

    Outputs:
        - R2: float, variance explained
    """
    W, residuals, rank, s = np.linalg.lstsq(B_train, A_train, rcond=None)  # solve A_train ~ B_train . W
    A_hat = B_test.dot(W)
    TSS = np.sum((A_test - A_test.mean(axis=0)) ** 2)  # total sum of squares
    SSE = np.sum((A_test - A_hat) ** 2)  # sum of squared errors
    R2 = 1.0 - SSE / TSS
    return R2


def get_svd_variance_explained(A, B):  # A & B: [n_samples, n_features]
    """Calculates the cumulative variance of matrix B that's explained by the first i orthonormal bases of matrix A using SVD."""
    U, Sigma, Vt = np.linalg.svd(A, full_matrices=False)  # should we normalize this basis set?
    # Vt = (Vt - Vt.mean(-1, keepdims = True)) / Vt.std(-1, keepdims = True)
    M = B @ Vt.T  # B projected onto the svd bases of A
    pc_exp_var = np.square(M).sum(axis=0)
    cumsum_exp_var = np.cumsum(pc_exp_var) / pc_exp_var.sum()
    return np.concatenate(([0], cumsum_exp_var))


def get_joint_neural_behaviour_place_direction_distance_dfs(
    sessions, n_splits=5, test_size=0.2, n_latents=[5, 10, 15, 20]
):
    """
    Get behvioural data cross validated across sessions. Neural latent data is not cross validated.
    Think about how to match up Xval later
    """
    split_sessions = _get_session_splits(sessions, n_splits, test_size)
    maze_name = sessions[0].maze_name
    X = []
    for train_sessions, test_sessions in split_sessions:
        X.append(
            {
                "neural_latents": {Nlat: _get_neural_latent_tuning_df(maze_name, Nlat) for Nlat in n_latents},
                "behaviour": {
                    "train": _get_behavioural_sequences(train_sessions),
                    "test": _get_behavioural_sequences(test_sessions),
                },
            }
        )
    return X


# %% load and proecss data functions


def _get_neural_latent_tuning_df(maze_name="maze_1", n_latents=20):
    """
    In this case neural tuning is
    Rerun model training on models with some sessions held out
    TODO: come up with better system for storing xvaled models with different latents
    Note not cross validated, trained on all data.
    """
    # update inputs & get exp_name
    exp_name = f"{maze_name}_state_action_distance_{n_latents}_latents"  # hardcoded
    Encoder = le.load_encoder(exp_name, exp_set=EXP_SET)
    kwargs = le.load_kwargs(exp_name, exp_set=EXP_SET)
    latent_tuning_df = pl.get_latent_place_direction_distance_tuning(Encoder, kwargs, return_as="df")
    return latent_tuning_df.T.sort_index(axis=1)  # [latents, place_directions * distance_bins]


def _get_behavioural_sequences(sessions, input_kwargs=DEFAULT_INPUT_KWARGS):
    """ """
    assert input_kwargs["distance_metrics"] == ("distance_to_goal", "geodesic")
    # combine data across sessions
    trajectories_dfs = []
    for session in sessions:
        trajectories_df = session.trajectory_decisions_df
        # filter
        trajectories_df = trajectories_df[
            (trajectories_df.trial_phase == "navigation")
            & (trajectories_df.steps_to_goal.lt(DEFAULT_INPUT_KWARGS["max_steps_to_goal"]))
        ]
        trajectories_dfs.append(trajectories_df)
    trajectories_df = pd.concat(trajectories_dfs, axis=0).reset_index(drop=True)
    # bin distances to goal
    distance_bins = convert._get_distance_bins(
        input_kwargs["distance_bin_method"],
        input_kwargs["n_distance_bins"],
        input_kwargs["distance_metrics"],
        input_kwargs["max_distance"],
    )
    trajectories_df["binned_distance_to_goal"] = pd.cut(trajectories_df.geodesic_distance_to_goal, bins=distance_bins)
    distance_bin_midpoints = [d.mid for d in distance_bins]
    d2idx = {d: i for i, d in enumerate(distance_bins)}
    simple_maze = session.simple_maze()
    place_directions = mr.get_maze_place_direction_pairs(simple_maze)
    pd2idx = {pd: i for i, pd in enumerate(place_directions)}
    pds = pd.array([pd2idx[pd] for pd in zip(trajectories_df.maze_position, trajectories_df.action)])
    ds = trajectories_df["binned_distance_to_goal"].map(d2idx).to_numpy()
    # loop over trials and count place-direction x distance-to-goal states
    trial_PDD_counts = []
    trial_unique_IDs = trajectories_df.trial_unique_ID.unique()
    for t in trial_unique_IDs:
        trial_mask = (trajectories_df.trial_unique_ID == t).to_numpy()
        trial_pds = pds[trial_mask]
        trial_ds = ds[trial_mask]
        PDD = np.zeros((len(pd2idx), len(d2idx)))
        for _pd, d in zip(trial_pds, trial_ds):
            PDD[_pd, d] += 1
        trial_PDD_counts.append(PDD)
    B = np.array(trial_PDD_counts)  # [trials, place_directions, distance_bins]
    return (
        pd.DataFrame(
            B.reshape(len(trial_unique_IDs), len(place_directions) * len(distance_bins)),
            columns=pd.MultiIndex.from_product(
                [place_directions, distance_bin_midpoints],
                names=["place_direction", "distance_to_goal"],
            ),
        )
        .astype(int)
        .sort_index(axis=1)
    )  # cols = [place_directions * distance_bins], index = trials


def _get_behavioural_sequences_ALT(sessions, input_kwargs=DEFAULT_INPUT_KWARGS):
    """
    # get navigation data filtered by same get_input_data kwargs as embedding model training runs
    # see embedding_model/place_direction_distance_occupancies.py
    """
    navigation_data = []
    for session in sessions:
        navigation_df = occ.get_session_filtered_navigation_data(session, input_kwargs)
        navigation_data.append(navigation_df)
    navigation_data = pd.concat(navigation_data, axis=0).reset_index(drop=True)
    distance_bins = convert._get_distance_bins(
        input_kwargs["distance_bin_method"],
        input_kwargs["n_distance_bins"],
        input_kwargs["distance_metrics"],
        input_kwargs["max_distance"],
    )
    distnace_bin_midpoints = [d.mid for d in distance_bins]
    d2idx = {d: i for i, d in enumerate(distance_bins)}
    simple_maze = sessions[0].simple_maze()
    place_directions = mr.get_maze_place_direction_pairs(simple_maze)
    pd2idx = {pd: i for i, pd in enumerate(place_directions)}
    pds = pd.array(
        [pd2idx[pd] for pd in zip(navigation_data.maze_position.simple, navigation_data.cardinal_movement_direction)]
    )
    distance_metrics = input_kwargs["distance_metrics"]
    distance_bins_col = (distance_metrics[0], distance_metrics[1] + "_binned")
    ds = navigation_data[distance_bins_col].map(d2idx).to_numpy()
    # loop over trials and count place-direction x distance-to-goal states
    trial_PDD_counts = []
    trial_unique_IDs = navigation_data.trial_unique_ID.unique()
    for t in trial_unique_IDs:
        trial_mask = (navigation_data.trial_unique_ID == t).to_numpy()
        trial_pds = pds[trial_mask]
        trial_ds = ds[trial_mask]
        PDD = np.zeros((len(pd2idx), len(d2idx)))
        for _pd, d in zip(trial_pds, trial_ds):
            PDD[_pd, d] += 1
        trial_PDD_counts.append(PDD)
    B = np.array(trial_PDD_counts)  # [trials, place_directions, distance_bins]
    return pd.DataFrame(
        B.reshape(len(trial_unique_IDs), len(place_directions) * len(distance_bins)),
        columns=pd.MultiIndex.from_product(
            [place_directions, distnace_bin_midpoints],
            names=["place_direction", "distance_to_goal"],
        ),
    ).astype(
        int
    )  # index = [place_directions * distance_bins], cols = trials


# %%


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


# %%


def plot_behavioural_pcs(sessions):
    """ """
    behaviour_train = _get_behavioural_sequences(sessions)
    behaviour = np.clip(behaviour_train.values, 0, 1)
    model = NMF(random_state=0, n_components=10)
    model.fit(behaviour)
    pcs = model.components_  # [n_components, n_features]
    pcs_df = pd.DataFrame(pcs, columns=behaviour_train.columns)
    occ_mask = occ.get_occupancy_mask("maze_1", "all")
    pl.plot_latent_marginals(pcs_df.T, "maze_1", occ_mask)
    return pcs_df
