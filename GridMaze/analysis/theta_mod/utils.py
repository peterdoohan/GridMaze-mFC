"""
Wouldn't it be cool if we could measure theta modulation over the abitraty neural representations
as behaviour unfolds?
"""

# %% Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy import cluster

from GridMaze.analysis.core import convert

from sklearn.decomposition import PCA
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d

# %% Global Variables
FRAME_RATE = 60

# %% Functions


def test(session):
    """ """
    df = get_neural_pc_df(
        session,
        spike_type="all",
        pc_kwargs={"sqrt_spikes": False, "zscore_spikes": False, "smooth_SD": False, "frac_var_exp": 0.9},
    )
    trials = df.trial.unique()
    for trial in trials[-5:]:
        trial_df = df[df.trial == trial]
        if trial_df.empty:
            continue
        plot_trial_activity(trial_df, PCs=(0, 1, 2), dot_dt=0.5)
        plt.show()


def test2(session):
    """ """
    df = get_theta_pc_df(
        session,
        include_multi_unit=True,
        pc_kwargs={"sqrt_spikes": False, "zscore_spikes": False, "smooth_SD": False, "frac_var_exp": 0.9},
    )
    trials = df.trial.unique()


# %% make useful datastructures


def get_neural_pc_df(
    session,
    include_multi_unit=True,
    spike_type="all",
    smooth_SD=0.5,
    pc_kwargs={"sqrt_spikes": False, "zscore_spikes": False, "smooth_SD": False, "frac_var_exp": 0.9},
):
    """ """
    # get PC subspace for spike projection
    pca, n_pcs = get_pcs(session, include_multi_unit=include_multi_unit, **pc_kwargs)
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
    if smooth_SD:
        # convert to n_frames
        spikes = gaussian_filter1d(spikes, sigma=int(smooth_SD * FRAME_RATE), axis=0)
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


def get_theta_pc_df(
    session,
    include_multi_unit=True,
    pc_kwargs={"sqrt_spikes": False, "zscore_spikes": False, "smooth_SD": False, "frac_var_exp": 0.9},
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
    pca, n_pcs = get_pcs(session, include_multi_unit=include_multi_unit, **pc_kwargs)

    # get theta phase spike projections
    theta_phases = list(theta_spikes_df.columns.get_level_values(2).unique())
    phase_dfs = [theta_spikes_df.xs(phase, level=2, axis=1) for phase in theta_phases] + [
        theta_spikes_df.T.groupby(level=1).mean().T
    ]
    theta_phases.append("theta_mean")  # add mean phase
    dfs = []
    for phase_df, phase in zip(phase_dfs, theta_phases):
        # project spikes to PCs
        spikes_pca = pca.transform(phase_df.values)[:, :n_pcs]
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


def get_theta_peak_trough_df(
    session,
    include_multi_unit=True,
    pc_kwargs={"sqrt_spikes": False, "zscore_spikes": False, "smooth_SD": False, "frac_var_exp": 0.9},
    theta_peak_inds=[5, 6],
    theta_trough_inds=[0, 11],
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
    pca, n_pcs = get_pcs(session, include_multi_unit=include_multi_unit, **pc_kwargs)

    # separate theta phases into peak and trough
    cols = theta_spikes_df.columns
    phase_cols = cols.get_level_values(2)
    theta_phases = phase_cols.unique()
    theta_peak_df = theta_spikes_df[cols[phase_cols.isin(theta_phases[theta_peak_inds])]]
    theta_trough_df = theta_spikes_df[cols[phase_cols.isin(theta_phases[theta_trough_inds])]]

    # sum spikes over theta phases
    theta_peak_df = theta_peak_df.T.groupby(level=1).sum().T
    theta_trough_df = theta_trough_df.T.groupby(level=1).sum().T

    # project spikes to PCs
    dfs = []
    for _df, label in zip([theta_peak_df, theta_trough_df], ["peak", "trough"]):
        spikes_pca = pca.transform(_df.values)[:, :n_pcs]
        # output as navigation_df-style df
        pca_df = pd.DataFrame(
            index=nav_df.index,
            data=spikes_pca,
            columns=pd.MultiIndex.from_product([[label], np.arange(n_pcs)]),
        )
        dfs.append(pca_df)

    # combine all with navigation data
    return pd.concat([nav_df] + dfs, axis=1)


def get_pcs(session, include_multi_unit=True, sqrt_spikes=False, zscore_spikes=False, smooth_SD=0.5, frac_var_exp=0.9):
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
    data_mask = np.logical_and.reduce([(navigation_df.trial_phase == "navigation"), navigation_df.moving])
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


# %% plotting


def plot_trial_activity(trial_df, PCs=(0, 1, 2), cmap="winter", dot_dt=0.5, ax=None):
    # set up fig
    if ax is None:
        f, ax = _init_3D_plot(PCs)

    # filter for navigation period
    _df = trial_df[trial_df.trial_phase == "navigation"]
    time = _df.time.values
    pcs = _df.pc[[*PCs]].values

    # plot
    # build segments between successive points
    P0 = pcs[:-1]
    P1 = pcs[1:]
    segments = np.stack([P0, P1], axis=1)

    # color mapping by time (use segment midpoints)
    t_mid = 0.5 * (time[:-1] + time[1:])
    norm = Normalize(vmin=time.min(), vmax=time.max())
    cmap_obj = plt.get_cmap(cmap)
    lc = Line3DCollection(segments, cmap=cmap_obj, norm=norm)
    lc.set_array(t_mid)
    lc.set_linewidth(2.0)
    ax.add_collection3d(lc)
    ax.auto_scale_xyz(pcs[:, 0], pcs[:, 1], pcs[:, 2])

    # plot time markers
    t0 = time[0]
    t_targets = np.arange(t0, time[-1] + 1e-9, dot_dt)  # include endpoint if aligned
    dots = np.vstack([np.interp(t_targets, time, pcs[:, d]) for d in range(pcs.shape[1])]).T
    dot_colors = cmap_obj(norm(t_targets))
    ax.scatter(
        dots[:, 0],
        dots[:, 1],
        dots[:, 2],
        s=30,
        c=dot_colors,
        alpha=0.9,
        depthshade=False,
        linewidths=0,
    )


def _init_3D_plot(PCs, figsize=(5, 5)):
    f = plt.figure(figsize=figsize)
    ax = f.add_subplot(111, projection="3d")
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel(f"PC{PCs[0]}")
    ax.set_ylabel(f"PC{PCs[1]}")
    ax.set_zlabel(f"PC{PCs[2]}")
    return f, ax
