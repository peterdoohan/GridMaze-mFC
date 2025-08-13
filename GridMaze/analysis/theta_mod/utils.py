"""
Wouldn't it be cool if we could measure theta modulation over the abitraty neural representations
as behaviour unfolds?
@peterdoohan
"""

# %% Imports
import numpy as np
import pandas as pd

from GridMaze.analysis.core import convert
from GridMaze.analysis.distance_to_goal.population_tuning import _get_session_distance_tuning
from GridMaze.analysis.place_direction.dimensionality_reduction import get_session_place_direction_tuning
from GridMaze.analysis.egocentric_action.population_tuning import get_session_egocentric_action_tuning

from sklearn.decomposition import PCA
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d

# %% Global Variables
FRAME_RATE = 60

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


# %% get pc functions


def get_pcs(
    session,
    include_multi_unit=True,
    max_steps_to_goal=30,
    sqrt_spikes=False,
    zscore_spikes=False,
    smooth_SD=0.5,
    n_pcs=None,
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
    pca = PCA(random_state=0, n_components=n_pcs)
    pca.fit(spikes)
    return pca


def get_distance_to_goal_pcs(
    session,
    include_multi_unit=True,
    bin_spacing=0.05,
    max_steps_to_goal=30,
    moving_only=True,
    n_pcs=5,
):
    """ """
    # get distance tuning curves
    distance_tuning = _get_session_distance_tuning(
        session,
        include_multi_unit=include_multi_unit,
        metrics=("distance_to_goal", "geodesic"),
        bin_spacing=bin_spacing,
        max_steps_to_goal=max_steps_to_goal,
        moving_only=moving_only,
        return_as="tuning_curves",
    )
    # do PCA on tuning curves to get loadings over neurons
    pca = PCA(random_state=0, n_components=n_pcs)
    pca.fit(distance_tuning.values.T)  # [samples = distances, features=neurons]
    return pca


def get_place_direction_pcs(
    session,
    include_multi_unit=True,
    min_occupancy=0.5,
    max_steps_to_goal=30,
    n_pcs=5,
):
    """ """
    # get place_direction tuning
    place_direction_tuning = get_session_place_direction_tuning(
        session,
        include_multi_unit=include_multi_unit,
        fill_nans="mean",
        normalisation=False,
        place_direction_tuned=False,
        min_split_corr=None,
        navigation_only=True,
        moving_only=True,
        exclude_time_at_goal=True,
        minimum_occupancy=min_occupancy,
        max_steps_from_goal=max_steps_to_goal,
    )
    # do PCA on tuning curves to get loadings over neurons
    pca = PCA(random_state=0, n_components=n_pcs)
    pca.fit(place_direction_tuning.values.T)  # [samples = place_directions, features=neurons]
    return pca


def get_egocentric_action_pcs(
    session,
    include_action_type=True,
    include_multi_unit=True,
    window=(-3, 3),
    n_pcs=5,
):
    """ """
    # get egocentric_action tuning
    ego_action_tuning = get_session_egocentric_action_tuning(
        session,
        actions=["turn_left", "turn_right", "go_forward", "go_back"],
        include_action_type=include_action_type,
        min_split_half_corr=None,
        window=window,
        include_multi_units=include_multi_unit,
    )
    # do PCA on tuning curves to get loadings over neurons
    pca = PCA(random_state=0, n_components=n_pcs)
    reshape_tuning = (
        ego_action_tuning.action_aligned_rates.T.stack(level=[1, 2], future_stack=True)
        .swaplevel(0, 2, axis=0)
        .sort_index()  # [samples = timepoints x n_actions x 2 (free, forced), features=neurons]
    )
    pca.fit(reshape_tuning.values)
    return pca


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
