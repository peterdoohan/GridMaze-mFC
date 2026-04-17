"""
Create analysis data structure with neuron spikes counted in per theta phase bin
"""

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from scipy.signal import butter, filtfilt, hilbert

from GridMaze.analysis.core import load_data
from GridMaze.analysis.core import convert
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.event_aligned import lfp_utils as lu

# %% Global Variables
FS_LFP = 1500

THETA_RANGE = (7, 11)

FRAME_RATE = 60  # Hz
# %% Functions


def get_navigation_theta_spike_counts_df(processed_data_path, analysis_data_path, n_bins=12):
    """
    see _get_navigation_lfp_phase_binned_spike_counts_df
    """
    return _get_navigation_lfp_phase_binned_spike_counts_df(processed_data_path, lfp_phase="theta", n_bins=n_bins)


def _get_navigation_lfp_phase_binned_spike_counts_df(processed_data_path, lfp_phase="theta", n_bins=12):
    """ """
    try:
        session_info = load_data.load(processed_data_path / "session_info.json")
        lfp_phase_binned_spike_counts = get_lfp_phase_binned_spike_counts(
            processed_data_path, lfp_phase=lfp_phase, n_bins=n_bins
        )
    except FileNotFoundError:
        print(
            f"Missing prerequist processed data to generate navigation_lfp_phase_binned_spike_counts_df for {processed_data_path}"
        )
        return None
    # convert to DataFrame
    cluster_phase_spike_counts = lfp_phase_binned_spike_counts[
        "aligned_spike_counts"
    ].T  # [n_frames, n_bins, n_clusters]
    cluster_phase_spike_counts = cluster_phase_spike_counts.reshape(
        cluster_phase_spike_counts.shape[0], -1
    )  # [n_frames, n_bins * n_clusters]
    cluster_IDs = lfp_phase_binned_spike_counts["cluster_IDs"]
    cluster_unique_IDs = convert.cluster_IDs2scluster_unique_IDs(session_info, cluster_IDs)
    phase_bin_edges = lfp_phase_binned_spike_counts["bin_edges"]
    phase_bin_centers = (phase_bin_edges[:-1] + phase_bin_edges[1:]) / 2
    time = lfp_phase_binned_spike_counts["t_out"]
    df = pd.DataFrame(
        data=cluster_phase_spike_counts,
        index=time,
        columns=pd.MultiIndex.from_product(
            [["spike_count"], phase_bin_centers, cluster_unique_IDs],
            names=["", f"{lfp_phase}_phase_bin", "cluster_unique_ID"],
        ),
    )
    df = df.swaplevel(2, 1, axis=1).sort_index(axis=1)
    return df


# %% combine spikes and lfp phase aligned to video frames


def get_lfp_phase_binned_spike_counts(processed_data_path, lfp_phase="theta", n_bins=12):
    """ """
    # load spike data
    spike_clusters = load_data.load(processed_data_path / "spikes.clusters.npy").reshape(-1)
    spike_times = load_data.load(processed_data_path / "spikes.times.npy").reshape(-1)
    cluster_IDs = np.sort(np.unique(spike_clusters)).astype(np.float64)

    # get lfp phase data
    lfp_signal = get_LFP(processed_data_path)
    lfp_times = load_data.load(processed_data_path / "lfp.times.npy")
    if lfp_phase == "theta":
        lfp_phase = get_lfp_phase(lfp_signal, freq_range=THETA_RANGE, N=4)
    else:
        raise ValueError("lfp_phase must be 'theta'")
    bin_edges, lfp_phase_bins = bin_lfp_phase(lfp_phase, n_bins=n_bins)

    # load frame times data
    trajectories_df = load_data.load(processed_data_path / "frames.trajectories.htsv")
    frame_times = trajectories_df.time.to_numpy()
    start_frame_times = frame_times - (  # offset by half a frame to get spike counts within each frame
        1 / (2 * FRAME_RATE)
    )
    # get spike counter per cluster x lfp_phase in each video frame
    aligned_spike_counts = np.zeros((len(cluster_IDs), n_bins, len(frame_times)), dtype=np.int32)
    for i, cluster_id in enumerate(cluster_IDs):
        cluster_spike_times = spike_times[spike_clusters == cluster_id]
        # get the lfp phase bin for each spike
        spike_lfp_idx = np.searchsorted(lfp_times, cluster_spike_times, side="left")
        spike_lfp_idx = np.clip(spike_lfp_idx - 1, 0, len(lfp_times) - 1)  # step back to the last sample ≤ spike time
        spike_phase_bins = lfp_phase_bins[spike_lfp_idx]
        # count spikes in each bin for each frame
        for bin_idx in range(n_bins):
            spike_times_in_bin = cluster_spike_times[spike_phase_bins == bin_idx]
            # count spikes in each frame
            cumsum_spike_counts = np.searchsorted(spike_times_in_bin, start_frame_times)
            aligned_spike_counts[i, bin_idx, :] = np.diff(cumsum_spike_counts, prepend=cumsum_spike_counts[0])
    return {
        "aligned_spike_counts": aligned_spike_counts,  # shape (n_clusters, n_bins, n_frames)
        "t_out": frame_times,  # time of each frame
        "bin_edges": bin_edges,  # edges of the phase bins
        "cluster_IDs": cluster_IDs,  # unique cluster IDs
    }


# %% LFP wrangling functions


def bin_lfp_phase(lfp_phase, n_bins=12):
    """bin osc phases from filtered lfp into n bins"""
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_indices = np.digitize(lfp_phase, bin_edges[1:-1])
    return bin_edges, bin_indices


def get_lfp_phase(
    lfp_signal,
    freq_range=(7, 11),
    N=4,
    return_filtered=False,
):
    """
    only for LFP (currently no CSD)
    """
    # filter for input frequency range
    nyq = FS_LFP / 2
    b, a = butter(N, [(freq_range[0] / nyq), (freq_range[1] / nyq)], btype="bandpass")
    filt_osc = filtfilt(b, a, lfp_signal)
    analytic = hilbert(filt_osc)
    phase_hilbert = np.angle(analytic)
    if return_filtered:
        return filt_osc, phase_hilbert
    return phase_hilbert


def get_LFP(processed_data, shank=3, single_channel=False, remove_artifacts=True):
    """
    copy from event_aligned/lfp_units.py
    to work straight from processed data
    """
    # load data
    lfp_metrics = load_data.load(processed_data / "lfp.metrics.htsv")
    cluster_metrics = load_data.load(processed_data / "clusters.metrics.htsv")
    lfp_signal = load_data.load(processed_data / "lfp.signal.npy")
    if single_channel:  # converting from channel_id to channel_index in lfp_signal to get the signal
        channel_id = lu._get_single_channel_for_LFP(lfp_metrics, cluster_metrics, shank)
        channel_index = lfp_metrics[lfp_metrics.contact.id == channel_id].index[0]
        lfp = lfp_signal[:, channel_index]
    else:  # average lfp over multiple channels
        channel_ids = lu._get_shank_channels_for_LFP(lfp_metrics, cluster_metrics, shank)
        channel_indices = lfp_metrics.contact.id.isin(channel_ids).index.values
        lfp = lfp_signal[:, channel_indices].mean(axis=1)
    if remove_artifacts:
        lfp = lu._remove_artifacts(lfp, thres=500)
    return lfp


# %% plotting


def _muted_hsv_colors(n, saturation=0.55, value=0.9):
    """Cyclic HSV palette with reduced saturation — rainbow ordering, softer look."""
    hues = np.linspace(0, 1, n, endpoint=False)
    return np.array([mcolors.hsv_to_rgb([h, saturation, value]) for h in hues])


def plot_theta_phase_stratified_spikes(
    session,
    time_window,
    n_bins=12,
    block_window=0.5,
    selected_phase=1,
    clusters=None,
    cmap=None,
    axes=None,
):
    """Visualise how spikes stratify by theta phase in a given time window.

    Panel 1: raw LFP
    Panel 2: theta-filtered LFP, line segments coloured by instantaneous phase bin
    Panel 3: spike raster, each spike coloured by the phase bin it falls in
    Panel 4: spike raster for `selected_phase` bin only (single-colour, sparse)
             — illustrates stratification by one theta phase.

    All panels share a time axis with faint grey bands alternating every
    `block_window` seconds (visual reference for spike-counting windows).

    Requires session to be loaded with: lfp_signal, lfp_times, lfp_metrics,
    cluster_metrics, spike_times, spike_clusters.
    """
    # load signals
    lfp = lu.get_LFP(session)
    lfp_times = session.lfp_times
    filt_osc, theta_phase = get_lfp_phase(lfp, freq_range=THETA_RANGE, N=4, return_filtered=True)
    _, lfp_phase_bins = bin_lfp_phase(theta_phase, n_bins=n_bins)

    # mask to time window
    t0, t1 = time_window
    mask = (lfp_times >= t0) & (lfp_times <= t1)
    t = lfp_times[mask]
    lfp_w = lfp[mask]
    theta_w = filt_osc[mask]
    phase_bins_w = lfp_phase_bins[mask]

    # prepare axes
    if axes is None:
        _, axes = plt.subplots(4, 1, figsize=(12, 7), sharex=True, height_ratios=[1, 1, 2, 2])

    # discrete colormap for phase bins (default: muted cyclic HSV)
    if cmap is None:
        bin_colors = _muted_hsv_colors(n_bins)
    else:
        cmap_obj = plt.get_cmap(cmap, n_bins)
        bin_colors = np.array([cmap_obj(i) for i in range(n_bins)])

    # faint grey alternating bands every block_window seconds
    block_edges = np.arange(t0, t1 + block_window, block_window)
    for ax in axes:
        for i in range(len(block_edges) - 1):
            if i % 2 == 0:
                ax.axvspan(block_edges[i], block_edges[i + 1], color="grey", alpha=0.1, linewidth=0)

    # Panel 1: raw LFP
    axes[0].plot(t, lfp_w, color="black", linewidth=0.8)
    axes[0].set_ylabel("LFP (uV)")
    axes[0].spines[["top", "right"]].set_visible(False)

    # Panel 2: theta-filtered, line coloured per-segment by phase bin
    points = np.array([t, theta_w]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_colors = bin_colors[phase_bins_w[:-1]]
    lc = LineCollection(segments, colors=seg_colors, linewidth=1.5)
    axes[1].add_collection(lc)
    axes[1].set_xlim(t[0], t[-1])
    theta_range = theta_w.max() - theta_w.min()
    axes[1].set_ylim(theta_w.min() - 0.1 * theta_range, theta_w.max() + 0.1 * theta_range)
    axes[1].set_ylabel(f"theta {THETA_RANGE[0]}-{THETA_RANGE[1]}Hz (uV)")
    axes[1].spines[["top", "right"]].set_visible(False)

    # Panel 3: spike raster, coloured by phase bin
    spike_times = session.spike_times.reshape(-1)
    spike_clusters = session.spike_clusters.reshape(-1)
    spike_mask = (spike_times >= t0) & (spike_times <= t1)
    st_w = spike_times[spike_mask]
    sc_w = spike_clusters[spike_mask]
    # phase bin for each spike
    spike_lfp_idx = np.clip(np.searchsorted(lfp_times, st_w, side="left") - 1, 0, len(lfp_times) - 1)
    spike_phase_bins = lfp_phase_bins[spike_lfp_idx]

    # pick clusters (default: all clusters in session)
    if clusters is None:
        clusters = np.unique(session.spike_clusters)
    clusters = np.asarray(clusters)

    # plot raster (one row per cluster, in given order)
    for row_i, c in enumerate(clusters):
        m = sc_w == c
        if not m.any():
            continue
        axes[2].scatter(
            st_w[m],
            np.full(m.sum(), row_i, dtype=float),
            c=bin_colors[spike_phase_bins[m]],
            s=30,
            marker="|",
            linewidths=1,
        )
    axes[2].set_ylabel("cluster")
    axes[2].set_ylim(-0.5, len(clusters) - 0.5)
    axes[2].spines[["top", "right"]].set_visible(False)

    # Panel 4: same raster but only spikes in selected_phase bin
    selected_color = bin_colors[selected_phase]
    sel_mask = spike_phase_bins == selected_phase
    for row_i, c in enumerate(clusters):
        m = (sc_w == c) & sel_mask
        if not m.any():
            continue
        axes[3].scatter(
            st_w[m],
            np.full(m.sum(), row_i, dtype=float),
            color=selected_color,
            s=30,
            marker="|",
            linewidths=1,
        )
    axes[3].set_ylabel(f"cluster\n(θ bin {selected_phase})")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylim(-0.5, len(clusters) - 0.5)
    axes[3].spines[["top", "right"]].set_visible(False)

    return axes


def plot_theta_stratification_schematic(n_bins=12, cmap=None, ax=None):
    """Schematic showing how spikes are stratified into `n_bins` theta-phase
    neuron × time matrices.

    Left:  source spike-count matrix (neurons × time), coloured in vertical
           stripes to indicate the continuously varying theta phase at each
           time point.
    Arrow: labelled "spike stratification by θ phase".
    Right: stack of `n_bins` overlapping rectangles, each a single phase
           colour, representing the per-phase-bin neurons × time matrices
           output by `get_navigation_theta_spike_counts_df`.
    """
    import matplotlib.patches as mpatches

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    if cmap is None:
        bin_colors = list(_muted_hsv_colors(n_bins))
    else:
        cmap_obj = plt.get_cmap(cmap, n_bins)
        bin_colors = [cmap_obj(i) for i in range(n_bins)]
    text_color = "#444444"
    accent = "#999999"

    # LEFT: source matrix as vertical coloured stripes (phase varies with time)
    lx, ly, lw, lh = 0.5, 1.0, 3.0, 2.5
    n_stripes = n_bins * 3  # ~3 theta cycles across the matrix
    sw = lw / n_stripes
    for i in range(n_stripes):
        ax.add_patch(
            mpatches.Rectangle(
                (lx + i * sw, ly),
                sw,
                lh,
                facecolor=bin_colors[i % n_bins],
                edgecolor="none",
            )
        )
    ax.text(lx + lw / 2, ly + lh + 0.2, "spikes", ha="center", va="bottom", fontsize=11, color=text_color)
    ax.text(lx + lw / 2, ly - 0.25, "time", ha="center", va="top", fontsize=9, color=accent)
    ax.text(lx - 0.25, ly + lh / 2, "neurons", ha="right", va="center", fontsize=9, color=accent, rotation=90)

    # ARROW with label
    arrow_y = ly + lh / 2
    ax.annotate(
        "",
        xy=(7.0, arrow_y),
        xytext=(4.0, arrow_y),
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color=accent, mutation_scale=18, shrinkA=0, shrinkB=0),
    )
    ax.text(
        5.5, arrow_y + 0.3, "spike stratification\nby θ phase", ha="center", va="bottom", fontsize=10, color=text_color
    )

    # RIGHT: overlapping stack of per-phase matrices (rounded corners, no borders)
    sx, sy, rw, rh = 7.5, 0.6, 2.5, 1.8
    dx, dy = 0.15, 0.15
    for i in range(n_bins):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (sx + i * dx, sy + i * dy),
                rw,
                rh,
                boxstyle="round,pad=0,rounding_size=0.08",
                facecolor=bin_colors[i],
                edgecolor="none",
            )
        )
    top_x = sx + (n_bins - 1) * dx
    top_y = sy + (n_bins - 1) * dy
    ax.text(
        top_x + rw / 2,
        top_y + rh + 0.2,
        "spike counts per θ-bin\n(downsampled)",
        ha="center",
        va="bottom",
        fontsize=11,
        color=text_color,
    )
    ax.text(sx - 0.25, sy + rh / 2, "neurons", ha="right", va="center", fontsize=9, color=accent, rotation=90)
    ax.text(sx + rw / 2, sy - 0.25, "time", ha="center", va="top", fontsize=9, color=accent)
    ax.text(sx + rw + 0.15, sy + rh / 2, "θ bin 1", ha="left", va="center", fontsize=9, color=text_color)
    ax.text(top_x + rw + 0.15, top_y + rh / 2, f"θ bin {n_bins}", ha="left", va="center", fontsize=9, color=text_color)

    return ax
