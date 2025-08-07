"""
Wouldn't it be cool if we could measure theta modulation over the abitraty neural representations
as behaviour unfolds?
"""

# %% Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from GridMaze.analysis.core import convert

from sklearn.decomposition import PCA
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import circmean

# %% Global Variables
FRAME_RATE = 60

# %% Functions


def test(session, smooth_SD=1):
    """ """
    df = get_neural_pc_df(
        session,
        spike_type="all",
        smooth_SD=smooth_SD,
        pc_kwargs={"sqrt_spikes": True, "zscore_spikes": True, "smooth_SD": smooth_SD, "frac_var_exp": 0.9},
    )
    trials = df.trial.unique()
    for trial in trials[-5:]:
        trial_df = df[df.trial == trial]
        # filter for navigation period and less than 30 steps from goal
        trial_df = trial_df[(trial_df.trial_phase == "navigation") & (trial_df.steps_to_goal.future.lt(20))]
        if trial_df.empty:
            continue
        # plot
        plot_neural_trajectory(trial_df, PCs=(0, 1, 2), dot_dt=0.5)


def test2(session, smooth_SD=1):
    """ """
    df = get_theta_peak_trough_df(
        session,
        smooth_SD=smooth_SD,
        pc_kwargs={"sqrt_spikes": True, "zscore_spikes": True, "smooth_SD": smooth_SD, "frac_var_exp": 0.9},
    )
    trials = df.trial.unique()
    for trial in trials:
        trial_df = df[df.trial == trial]
        # filter for navigation period and less than 20 steps from goal
        trial_df = trial_df[(trial_df.trial_phase == "navigation") & (trial_df.steps_to_goal.future.lt(20))]
        if trial_df.empty:
            continue
        plot_theta_peak_trough_trajectory(trial_df, PCs=(0, 1, 2), dot_dt=0.5, cmaps=("winter", "winter"))


def test3(
    session,
    smooth_SD=2,
    include_multi_unit=True,
    sqrt_spikes=False,
    zscore_spikes=False,
    dot_dt=0.5,
    dot_moving_only=False,
):
    window_frames = int(dot_dt * FRAME_RATE)
    pca, n_pcs = get_pcs(
        session,
        include_multi_unit=include_multi_unit,
        sqrt_spikes=sqrt_spikes,
        zscore_spikes=zscore_spikes,
        smooth_SD=smooth_SD,
        frac_var_exp=0.70,
    )
    print(n_pcs)
    neural_pc_df = get_neural_pc_df(
        session,
        include_multi_unit=include_multi_unit,
        sqrt_spikes=sqrt_spikes,
        zscore_spikes=zscore_spikes,
        smooth_SD=smooth_SD,
        pca=pca,
        n_pcs=n_pcs,
    )
    theta_pc_df = get_theta_peak_trough_df(
        session,
        include_multi_unit=include_multi_unit,
        sqrt_spikes=sqrt_spikes,
        zscore_spikes=zscore_spikes,
        smooth_SD=smooth_SD,
        pca=pca,
        n_pcs=n_pcs,
    )
    trials = neural_pc_df.trial.unique()
    angles = []
    for trial in trials:
        _masks = [
            (neural_pc_df.trial == trial),
            (neural_pc_df.trial_phase == "navigation"),
            (neural_pc_df.steps_to_goal.future <= 30),
        ]
        mask = np.logical_and.reduce(_masks)
        _neural_df = neural_pc_df[mask]
        _theta_df = theta_pc_df[mask]
        if _neural_df.empty or _theta_df.empty:
            continue
        # neural_vector, theta_vector, bins = _get_alignment_vectors(_neural_df, _theta_df, window_frames)
        # _angles = _get_alignment_angles(neural_vector, theta_vector)
        goal_vector, theta_vector, bins = _get_goal_vectors(_neural_df, _theta_df, window_frames)
        _angles = _get_alignment_angles(goal_vector, theta_vector)
        times = _neural_df.loc[bins].time.values
        moving_mask = _neural_df.loc[bins].moving
        if dot_moving_only:
            times = times[moving_mask]
            theta_vector = theta_vector[moving_mask, :]
            _angles = _angles[moving_mask]
        # plot
        plot_neural_trajectory(
            _neural_df,
            PCs=(0, 1, 2),
            t_targets=times,
            t_vectors=theta_vector,
            t_angles=_angles,
            dot_dt=dot_dt,
        )
        angles.append(_angles[moving_mask])
    all_angles = np.concatenate(angles)
    return all_angles


def _get_alignment_vectors(_neural_df, _theta_df, window_frames):
    """ """
    idx = _neural_df.index
    bin_edges = np.arange(idx[0], idx[-1], window_frames)
    bin_mids = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
    bin_mids = bin_mids.astype(int)  # convert to int for indexing
    # behavioural timescale vector
    neural_vector = _neural_df.loc[bin_mids[1:]].pc.values - _neural_df.loc[bin_mids[:-1]].pc.values
    # theta timescale vector
    __theta_df = _theta_df.loc[bin_mids[:-1]]
    theta_vector = __theta_df.peak.values - __theta_df.trough.values  # n_bins -1, n_pcs
    return neural_vector, theta_vector, bin_mids[:-1]


def _get_goal_vectors(_neural_df, _theta_df, window_frames):
    """cal vector from nerual data at time t point to goal times[-1]"""
    goal_pt = _neural_df.iloc[-1].pc.values.astype(float)
    idx = _neural_df.index
    bins = np.arange(idx[0], idx[-2], window_frames)
    # behavioural timescale goal vector
    goal_vector = goal_pt - _neural_df.loc[bins].pc.values
    # theta timescale vector
    __theta_df = _theta_df.loc[bins]
    theta_vector = __theta_df.peak.values - __theta_df.trough.values
    return goal_vector, theta_vector, bins


def _get_alignment_angles(neural_vector, theta_vector):
    # get angle between neural vector and theta vector (cosine similarity)
    dots = np.einsum("ij,ij->i", neural_vector, theta_vector)  # dot products per timepoint
    n_norm = np.linalg.norm(neural_vector, axis=1)  # norms per timepoint
    t_norm = np.linalg.norm(theta_vector, axis=1)
    den = n_norm * t_norm
    cos_sim = np.clip(dots / np.maximum(den, 1e-12), -1.0, 1.0)
    angles = np.arccos(cos_sim)  # shape (n_timepoints,), radian
    return angles


# %% make useful datastructures


def get_neural_pc_df(
    session,
    include_multi_unit=True,
    spike_type="all",
    sqrt_spikes=False,
    zscore_spikes=False,
    smooth_SD=0.5,
    pc_kwargs={"sqrt_spikes": False, "zscore_spikes": False, "smooth_SD": False, "frac_var_exp": 0.9},
    pca=None,
    n_pcs=None,
):
    """ """

    navigation_df = session.navigation_df
    if spike_type == "all":
        # load data
        spikes_df = session.navigation_spike_counts_df.spike_count
    elif spike_type == "theta_mean":
        theta_spikes_df = session.navigation_theta_spike_counts_df
        spikes_df = theta_spikes_df.T.groupby(level=1).mean().T
    else:
        raise ValueError(f"Unknown spike_type: {spike_type}")
    # filter for clusters to keep
    keep_clusters = _keep_clusters(session, include_multi_unit=include_multi_unit)
    spikes_df = spikes_df[keep_clusters]
    spikes = spikes_df.values.astype(float)  # n_samples (frame) x n_features (clusters)
    if sqrt_spikes:
        spikes = np.sqrt(spikes)
    if zscore_spikes:
        spikes = zscore(spikes, axis=0)
    if smooth_SD:
        # convert to n_frames
        spikes = gaussian_filter1d(spikes, sigma=int(smooth_SD * FRAME_RATE), axis=0)
    # get PC subspace for spike projection
    if pca is None or n_pcs is None:
        pca, n_pcs = get_pcs(session, include_multi_unit=include_multi_unit, **pc_kwargs)
    # project spikes to PCs
    spikes_pca = pca.transform(spikes)[:, :n_pcs]
    # output as navigation_df-style df
    pca_df = pd.DataFrame(
        index=navigation_df.index,
        data=spikes_pca,
        columns=pd.MultiIndex.from_product([["pc"], np.arange(n_pcs)]),
    )
    # add pca info to navigation_df
    df = pd.concat([navigation_df, pca_df], axis=1)
    return df


def get_theta_peak_trough_df(
    session,
    include_multi_unit=True,
    smooth_SD=0.5,
    sqrt_spikes=False,
    zscore_spikes=False,
    pc_kwargs={"sqrt_spikes": False, "zscore_spikes": False, "smooth_SD": False, "frac_var_exp": 0.75},
    theta_peak_inds=[4, 5, 6, 7],
    theta_trough_inds=[0, 1, 10, 11],
    pca=None,
    n_pcs=None,
):
    """ """
    # load data
    nav_df = session.navigation_df.copy()
    theta_spikes_df = session.navigation_theta_spike_counts_df
    # filter for clusters to keep
    keep_clusters = _keep_clusters(session, include_multi_unit=include_multi_unit)
    theta_spikes_df = theta_spikes_df[
        theta_spikes_df.columns[[c in keep_clusters for c in theta_spikes_df.columns.get_level_values(1)]]
    ]

    # get PC subspace for spike projection
    if pca is None or n_pcs is None:
        pca, n_pcs = get_pcs(session, include_multi_unit=include_multi_unit, **pc_kwargs)

    # separate theta phases into peak and trough
    cols = theta_spikes_df.columns
    phase_cols = cols.get_level_values(2)
    theta_phases = phase_cols.unique()
    theta_peak_df = theta_spikes_df[cols[phase_cols.isin(theta_phases[theta_peak_inds])]]
    theta_trough_df = theta_spikes_df[cols[phase_cols.isin(theta_phases[theta_trough_inds])]]

    # sum spikes over theta phases
    peak_spikes = (
        theta_peak_df.T.groupby(level=1).sum().T.values.astype(float)
    )  # n_samples (frames) x n_features (clusters)
    trough_spikes = theta_trough_df.T.groupby(level=1).sum().T.values.astype(float)

    # loop over peak and trough spikes
    dfs = []
    for spikes, label in zip([peak_spikes, trough_spikes], ["peak", "trough"]):
        if sqrt_spikes:
            spikes = np.sqrt(spikes)
        if zscore_spikes:
            spikes = zscore(spikes, axis=0)
        if smooth_SD:
            spikes = gaussian_filter1d(spikes, sigma=int(smooth_SD * FRAME_RATE), axis=0)
        # project theta phase spikes onto PCs
        spikes_pca = pca.transform(spikes)[:, :n_pcs]
        # output as navigation_df-style df
        pca_df = pd.DataFrame(
            index=nav_df.index,
            data=spikes_pca,
            columns=pd.MultiIndex.from_product([[label], np.arange(n_pcs)]),
        )
        dfs.append(pca_df)

    # combine all with navigation data
    return pd.concat([nav_df] + dfs, axis=1)


def get_theta_pc_df(
    session,
    include_multi_unit=True,
    smooth_SD=1,
    sqrt_spikes=False,
    zscore_spikes=False,
    pc_kwargs={"sqrt_spikes": False, "zscore_spikes": False, "smooth_SD": False, "frac_var_exp": 0.9},
    pca=None,
    n_pcs=None,
):
    """ """
    # load data
    nav_df = session.navigation_df.copy()
    theta_spikes_df = session.navigation_theta_spike_counts_df
    # filter for clusters to keep
    keep_clusters = _keep_clusters(session, include_multi_unit=include_multi_unit)
    theta_spikes_df = theta_spikes_df[
        theta_spikes_df.columns[[c in keep_clusters for c in theta_spikes_df.columns.get_level_values(1)]]
    ]

    # get PC subspace for spike projection
    if pca is None or n_pcs is None:
        pca, n_pcs = get_pcs(session, include_multi_unit=include_multi_unit, **pc_kwargs)

    # get sep df for each theta phase (and average across theta phases)
    theta_phases = list(theta_spikes_df.columns.get_level_values(2).unique())
    phase_dfs = [theta_spikes_df.xs(phase, level=2, axis=1) for phase in theta_phases] + [
        theta_spikes_df.T.groupby(level=1).mean().T
    ]
    theta_phases.append("theta_mean")  # add mean phase
    dfs = []
    for phase_df, phase in zip(phase_dfs, theta_phases):
        spikes = phase_df.values.astype(float)
        if sqrt_spikes:
            spikes = np.sqrt(spikes)
        if zscore_spikes:
            spikes = zscore(spikes, axis=0)
        if smooth_SD:
            spikes = gaussian_filter1d(spikes, sigma=int(smooth_SD * FRAME_RATE), axis=0)
        # project spikes to PCs
        spikes_pca = pca.transform(spikes)[:, :n_pcs]
        # output as navigation_df-style df
        pca_df = pd.DataFrame(
            index=nav_df.index,
            data=spikes_pca,
            columns=pd.MultiIndex.from_product([["pc"], np.arange(n_pcs)]),
        )
        pca_df.columns = pd.MultiIndex.from_tuples([(c[0], phase, c[1]) for c in pca_df.columns])
        dfs.append(pca_df)

    # combine all with navigation data
    nav_df.columns = pd.MultiIndex.from_tuples([(*c, "") for c in nav_df.columns])
    df = pd.concat([nav_df] + dfs, axis=1)
    return df


def get_pcs(
    session,
    include_multi_unit=True,
    max_steps_to_goal=30,
    sqrt_spikes=False,
    zscore_spikes=False,
    smooth_SD=0.5,
    n_pcs=None,
    frac_var_exp=0.9,
):
    """
    get the PCs that explain x frac_var_exp (default == 0.8) of the variance in the spike counts
    during navigation (& when animal is moving). Use these PCs to project the spike counts later.
    """
    navigation_df = session.navigation_df
    spike_counts_df = session.navigation_spike_counts_df
    keep_clusters = _keep_clusters(session, include_multi_unit=include_multi_unit)
    spikes = spike_counts_df.spike_count[keep_clusters].values.astype(
        float
    )  # n_samples (frame) x n_features (clusters)
    # smooth spikes counts
    if smooth_SD:
        spikes = gaussian_filter1d(spikes, sigma=int(smooth_SD * FRAME_RATE), axis=0)
    # filter for times during navigation & moving
    _masks = [(navigation_df.trial_phase == "navigation"), navigation_df.moving]
    if max_steps_to_goal is not None:
        _masks.append(navigation_df.steps_to_goal.future <= max_steps_to_goal)
    data_mask = np.logical_and.reduce(_masks)
    spikes = spikes[data_mask, :]
    # run PCA on spike counts over time
    if sqrt_spikes:
        spikes = np.sqrt(spikes)
    if zscore_spikes:
        spikes = zscore(spikes, axis=0)
    # run PCA
    pca = PCA(random_state=0)
    pca.fit(spikes)
    # get n_pcs to explain x pct_var
    if n_pcs is None:
        n_pcs = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= frac_var_exp)
    return pca, n_pcs


# %% other


def _keep_clusters(session, include_multi_unit=True):
    """
    get cluster_unique_IDs of clusters to keep, either single unit or multi-unit
    """
    cluster_metrics = session.cluster_metrics
    mask = [cluster_metrics.single_unit.values]
    if include_multi_unit:
        mask.append(cluster_metrics.multi_unit.values)
    mask = np.logical_or.reduce(mask)
    cluster_IDs = cluster_metrics.cluster_ID[mask]
    # convert to cluster_unique_IDs
    cluster_unique_IDs = convert.cluster_IDs2scluster_unique_IDs(session.session_info, cluster_IDs)
    return list(cluster_unique_IDs)
