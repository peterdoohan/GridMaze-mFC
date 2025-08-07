"""
Lib for plotting neural trajectories to illustrate theta mod analyses
"""

# %% Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from GridMaze.analysis.theta_mod import utils
from GridMaze.analysis.theta_mod import alignment as ali


# %% Global Variables

FRAME_RATE = 60

# %%


def get_trajectory_data(
    session,
    smooth_SD=2,
    include_multi_unit=True,
    sqrt_spikes=False,
    zscore_spikes=False,
    n_pcs=10,
    theta_peak_inds=[4, 5, 6, 7],
    theta_trough_inds=[0, 1, 10, 11],
    frac_var_exp=None,
    plot_trajectories=False,
    plot_angle_type="goal",
):
    common_kwargs = {
        "include_multi_unit": include_multi_unit,
        "sqrt_spikes": sqrt_spikes,
        "zscore_spikes": zscore_spikes,
        "smooth_SD": smooth_SD,
    }
    pca, n_pcs = utils.get_pcs(
        session,
        **common_kwargs,
        n_pcs=n_pcs,
        frac_var_exp=frac_var_exp,
    )
    neural_pc_df = utils.get_neural_pc_df(
        session,
        **common_kwargs,
        pca=pca,
        n_pcs=n_pcs,
    )
    theta_peak_trough_pc_df = utils.get_theta_peak_trough_df(
        session,
        **common_kwargs,
        theta_peak_inds=theta_peak_inds,
        theta_trough_inds=theta_trough_inds,
        pca=pca,
        n_pcs=n_pcs,
    )
    theta_pc_df = utils.get_theta_pc_df(
        session,
        **common_kwargs,
        pca=pca,
        n_pcs=n_pcs,
    )
    trials = neural_pc_df.trial.dropna().unique()
    if plot_trajectories:
        for trial in trials:
            plot_theta_vector_trial_trajectory(
                neural_pc_df,
                theta_peak_trough_pc_df,
                trial=trial,
                vector_window=smooth_SD,
                angle_type=plot_angle_type,
            )

    return neural_pc_df, theta_peak_trough_pc_df, theta_pc_df


# %%


def plot_all_theta_phase_trajectories(
    theta_pc_df,
    trial,
    max_steps_to_goal=None,
    PCs=(0, 1, 2),
    dot_dt=1,
    fig=None,
    ax=None,
):
    """ """
    phases = np.array(sorted([c for c in theta_pc_df.pc.columns.get_level_values(0).unique() if c != "theta_mean"]))

    # filter trial
    _masks = [
        (theta_pc_df.trial == trial),
        (theta_pc_df.trial_phase == "navigation"),
    ]
    if max_steps_to_goal is not None:
        _masks.append(theta_pc_df.steps_to_goal.future <= max_steps_to_goal)
    mask = np.logical_and.reduce(_masks)
    _theta_df = theta_pc_df[mask]
    if _theta_df.empty:
        raise ValueError(f"No data for trial: {trial}")
    times = _theta_df.time.values

    # plot
    if fig is None or ax is None:
        fig, ax = _init_3D_plot(PCs)
    # create colormaps for all phases
    colormaps = _get_theta_colormaps(len(phases))
    for phase, cmap in zip(phases, colormaps):
        phase_df = _theta_df.xs(phase, level=1, axis=1)
        pcs = phase_df.pc[[*PCs]].values
        _plot_neural_traj(
            pcs=pcs,
            time=times,
            dot_dt=dot_dt,
            cmap=cmap,
            show_colorbars=False,
            fig=fig,
            ax=ax,
        )
    fig.tight_layout()

    return


def plot_theta_vector_trial_trajectory(
    neural_pc_df,
    theta_pc_df,
    trial,
    max_steps_to_goal=None,
    PCs=(0, 1, 2),
    plot_vector_scale=2.0,
    cmap="winter",
    angle_type="goal",
    vector_window=1,
    fig=None,
    ax=None,
):
    # filter trial
    _masks = [
        (neural_pc_df.trial == trial),
        (neural_pc_df.trial_phase == "navigation"),
    ]
    if max_steps_to_goal is not None:
        _masks.append(neural_pc_df.steps_to_goal.future <= max_steps_to_goal)
    mask = np.logical_and.reduce(_masks)
    _neural_df = neural_pc_df[mask]
    _theta_df = theta_pc_df[mask]
    if _neural_df.empty or _theta_df.empty:
        raise ValueError(f"No data for trial: {trial}")

    # calculate angles
    idx = _neural_df.index
    window_edges = np.arange(idx[0], idx[-1], (vector_window * FRAME_RATE))
    window_mids = window_edges[:-1] + 0.5 * np.diff(window_edges)
    window_mids = window_mids.astype(int)  # convert to int for indexing
    dot_times = _neural_df.loc[window_mids].time.values

    theta_vectors = _get_theta_vectors(_theta_df.loc[window_mids])
    if angle_type == "goal":
        goal_vectors = ali._get_goal_vectors(_neural_df.loc[window_mids], at_goal=_neural_df.iloc[-1])
        angles = _get_theta_angles(theta_vectors, goal_vectors)

    elif angle_type == "trajectory":
        traj_vectors = ali._get_neural_trajectory_vectors(_neural_df.loc[window_edges])
        angles = _get_theta_angles(theta_vectors, traj_vectors)

    else:
        raise ValueError(f"Unknown angle_type: {angle_type}. Use 'goal' or 'trajectory'.")

    # plot!
    if fig is None or ax is None:
        fig, ax = _init_3D_plot(PCs)
    _plot_neural_traj(
        pcs=_neural_df.pc[[*PCs]].values,
        time=_neural_df.time.values,
        dot_times=dot_times,
        dot_vectors=theta_vectors[:, PCs],
        dot_angles=angles,
        dot_vector_scale=plot_vector_scale,
        dot_dt=vector_window,
        cmap=cmap,
        fig=fig,
        ax=ax,
    )


# %% Helper fns
def _get_theta_vectors(_theta_df):
    # theta timescale vector
    theta_vectors = _theta_df.peak.values - _theta_df.trough.values
    return theta_vectors


def _get_theta_angles(theta_vectors, base_vectors):
    dots = np.einsum("ij,ij->i", base_vectors, theta_vectors)  # dot products per timepoint
    n_norm = np.linalg.norm(base_vectors, axis=1)  # norms per timepoint
    t_norm = np.linalg.norm(theta_vectors, axis=1)
    den = n_norm * t_norm
    cos_sim = np.clip(dots / np.maximum(den, 1e-12), -1.0, 1.0)
    return np.arccos(cos_sim)  # shape (n_timepoints,), radian


def _get_theta_colormaps(n):
    """
    Generate n LinearSegmentedColormap objects that go from white
    to each of n bright colours sampled from Seaborn's HLS palette.

    Returns
    -------
    cmaps : list of matplotlib.colors.Colormap
        A list of n colormaps, named 'white_to_hls0', 'white_to_hls1', …
    """
    # get n colours from Seaborn's HLS palette (equally spaced hues)
    base_colors = sns.color_palette("hls", n)

    cmaps = []
    for i, colour in enumerate(base_colors):
        name = f"white_to_hls{i}"
        cmap = LinearSegmentedColormap.from_list(name, ["white", colour])
        cmaps.append(cmap)
    return cmaps


# %% plotting functions


def _plot_neural_traj(
    pcs,
    time,
    dot_times=None,
    dot_vectors=None,
    dot_angles=None,
    dot_vector_scale=1.0,
    t_cmap="RdGy",
    dot_dt=0.5,
    cmap="winter",
    show_colorbars=True,
    fig=None,
    ax=None,
):
    # build segments between successive points
    P0 = pcs[:-1]
    P1 = pcs[1:]
    segments = np.stack([P0, P1], axis=1)

    # color mapping by time (use segment midpoints)
    t_mid = 0.5 * (time[:-1] + time[1:])
    norm = Normalize(vmin=time.min(), vmax=time.max())
    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap
    lc = Line3DCollection(segments, cmap=cmap_obj, norm=norm)
    lc.set_array(t_mid)
    lc.set_linewidth(2.0)
    ax.add_collection3d(lc)

    # plot time markers
    t0 = time[0]
    if dot_times is None:
        dot_times = np.arange(t0, time[-1] + 1e-9, dot_dt)  # include endpoint if aligned
    dots = np.vstack([np.interp(dot_times, time, pcs[:, d]) for d in range(pcs.shape[1])]).T
    dot_colors = cmap_obj(norm(dot_times))
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

    # --- arrows at dots (direction+magnitude from dot_vectors) ---
    if dot_vectors is not None:
        dot_vectors = np.asarray(dot_vectors)
        if dot_vectors.ndim != 2 or dot_vectors.shape[1] != 3:
            raise ValueError("dot_vectors must have shape (n_targets, 3).")
        if dot_vectors.shape[0] != dots.shape[0]:
            raise ValueError("dot_vectors must have the same number of rows as dot_times/dots.")

        # subsample arrows if desired
        idx = np.arange(len(dots))
        U, V, W = (dot_vectors[idx] * dot_vector_scale).T

        # --- arrow colors based on dot_angles ---
        if dot_angles is not None:
            dot_angles = np.asarray(dot_angles)
            if dot_angles.shape[0] != dots.shape[0]:
                raise ValueError("dot_angles must have same length as dot_times/dots.")
            # Normalize angles from [-pi, pi] with center at 0
            angle_norm = TwoSlopeNorm(vmin=0, vcenter=np.pi / 2, vmax=np.pi)
            angle_cmap = plt.get_cmap(t_cmap)
            arrow_colors = angle_cmap(angle_norm(dot_angles))
        else:
            arrow_colors = "k"  # fallback if no dot_angles provided

        ax.quiver(
            dots[idx, 0],
            dots[idx, 1],
            dots[idx, 2],  # starts
            U,
            V,
            W,  # vectors
            normalize=False,
            arrow_length_ratio=0.2,
            linewidths=2,
            alpha=1,
            color=arrow_colors,
            length=1.0,  # leave at 1.0 to use raw U,V,W magnitudes
        )

    # --- add colorbars ---
    if show_colorbars:
        # Get figure size in normalized coordinates
        box = ax.get_position()
        fig_width = box.width
        fig_left = box.x0
        fig_right = box.x1

        # --- time colorbar (top-left, horizontal, no ticks or labels) ---
        sm_time = ScalarMappable(norm=norm, cmap=cmap_obj)
        cbar_time_ax = fig.add_axes([fig_left, box.y1 + 0.03, fig_width * 0.4, 0.015])
        cbar_time = fig.colorbar(sm_time, cax=cbar_time_ax, orientation="horizontal")
        cbar_time.outline.set_visible(False)
        cbar_time.set_ticks([time.min(), time.max()])
        cbar_time.set_ticklabels(["cue", "goal"])
        cbar_time.set_label("time", labelpad=4)

        # --- angle colorbar (top-right, horizontal, labeled) ---
        if dot_angles is not None:
            angle_norm = TwoSlopeNorm(vmin=0, vcenter=np.pi / 2, vmax=np.pi)
            angle_cmap = plt.get_cmap(t_cmap)
            sm_angle = ScalarMappable(norm=angle_norm, cmap=angle_cmap)

            # place on top-right
            cbar_angle_ax = fig.add_axes([fig_right - fig_width * 0.4, box.y1 + 0.03, fig_width * 0.4, 0.015])
            cbar_angle = fig.colorbar(sm_angle, cax=cbar_angle_ax, orientation="horizontal")
            cbar_angle.outline.set_visible(False)
            cbar_angle.set_ticks([0, np.pi / 2, np.pi])
            cbar_angle.set_ticklabels(["align.", "orthog.", "anti-align."])
            cbar_angle.set_label("angle (rad)", labelpad=4)


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
