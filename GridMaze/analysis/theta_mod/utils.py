"""
Wouldn't it be cool if we could measure theta modulation over the abitraty neural representations
as behaviour unfolds?
"""

# %% Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from sklearn.decomposition import PCA
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d

# %% Global Variables


# %% Functions


def test(session, sqrt_spikes=False, zscore_spikes=False, smooth_SD=1, pct_var=0.8, plot_eg_trials=False):
    """ """

    navigation_spikes_df = session.get_navigation_activity_df(
        type="spikes", cluster_kwargs={"single_units": True, "multi_units": True}
    )
    # filter for times during task
    navigation_spikes_df = navigation_spikes_df[navigation_spikes_df.trial.notnull()]
    # run PCA on spike counts over time
    spikes = navigation_spikes_df.spike_count.values.astype(float)  # n_samples (frame) x n_features (clusters)
    if sqrt_spikes:
        spikes = np.sqrt(spikes)
    if zscore_spikes:
        spikes = zscore(spikes, axis=0)
    if smooth_SD:
        spikes = gaussian_filter1d(spikes, sigma=smooth_SD, axis=0)
    # run PCA
    pca = PCA(random_state=0)
    pca.fit(spikes)
    # get n_pcs to explain x pct_var
    n_pcs = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= pct_var)
    print(n_pcs)
    # transform spikes to PCs
    spikes_pca = pca.transform(spikes)[:, :n_pcs]
    pca_df = pd.DataFrame(
        index=navigation_spikes_df.index,
        data=spikes_pca,
        columns=pd.MultiIndex.from_product([["pc"], np.arange(n_pcs)]),
    )
    # add pca info
    df = pd.concat([navigation_spikes_df, pca_df], axis=1)
    if plot_eg_trials:
        # choose random 4 trials
        trials = df.trial.unique()
        trials = np.random.choice(trials, size=4, replace=False)
        for trial in trials:
            trial_df = df[df.trial == trial]
            plot_trial_activity(trial_df, PCs=(2, 3, 4))
            plt.title(f"Trial {trial}")
            plt.show()
    return df


# %% plotting


def plot_trial_activity(trial_df, PCs=(0, 1, 2), cmap="winter", ax=None):
    # set up fig
    if ax is None:
        f, ax = _init_3D_plot(PCs)

    # filter for navigation period
    _df = trial_df[trial_df.trial_phase == "navigation"]
    time = _df.time.values
    pcs = _df.pc[[*PCs]].values
    # plot
    # build segments between successive points
    # shape (n-1, 2, dims)
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
    return


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
