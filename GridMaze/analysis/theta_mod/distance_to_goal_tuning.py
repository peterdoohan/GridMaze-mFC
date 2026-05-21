"""
Is there a systematic shift in distance tuning curves across theta phases (peak vs trough)?
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.stats import ttest_1samp

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import convert
from GridMaze.analysis.distance_to_goal import distributions as dd
from GridMaze.analysis.processing import get_distance_tuning_metrics_df as dtm
from GridMaze.analysis.theta_mod import theta_utils as tmu
from GridMaze.analysis.theta_mod import distance_to_goal_decoder as tdd  # noqa: F401  (kept for ref)
from GridMaze.analysis.theta_mod import distance_to_goal_decoder2 as ddv2

# %% Global Variables

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "theta_mod" / "distance_tuning"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

FRAME_RATE = 60

# %% Different set of analyses looking averaging tuning across cells split by theta before quantifying shift


def plot_heatmap_slices(
    tuning_curves,
    tunning_metrics,
    sign="pos",
    neuron_groups=6,
    distance_groups=6,
    how="horizontal",
    cmap="plasma_r",
    ax=None,
):
    """ """
    # set up fig
    if ax is None:
        f, ax = plt.subplots(figsize=(3, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("norm. firing rate")
    # process heatmap
    df = get_theta_split_distance_heatmap(
        tuning_curves,
        tunning_metrics,
        sign=sign,
        downsample=True,
        neuron_groups=neuron_groups,
        distance_groups=distance_groups,
    )
    if how == "horizontal":
        # plot firing rate of neuron groups over distances
        cmap = sns.color_palette(cmap, neuron_groups)
        for i in range(neuron_groups):
            g = df.iloc[i]
            color = cmap[i]
            for phase, ls in zip(
                ["trough", "peak"],
                ["-", "--"],
            ):
                g_phase = g.loc[phase]
                x = g_phase.index.astype(float).values
                y = g_phase.values
                ax.plot(
                    x,
                    y,
                    color=color,
                    label=f"{i}: {phase}",
                    linestyle=ls,
                )
        ax.set_xlabel("distance to goal (m)")
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    elif how == "vertical":
        distances = df.columns.get_level_values(1).astype(float).unique().values
        cmap = sns.color_palette(cmap, len(distances))
        _df = df.unstack()
        for i, d in enumerate(distances):
            color = cmap[i]
            for phase, ls in zip(
                ["trough", "peak"],
                ["-", "--"],
            ):
                g_phase = _df.loc[(phase, d)]
                x = g_phase.index.astype(int).values
                y = g_phase.values
                ax.plot(
                    x,
                    y,
                    color=color,
                    label=f"{d:.2f} m: {phase}",
                    linestyle=ls,
                )
        ax.set_xlabel("neuron group")
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))


def get_theta_split_distance_heatmap(
    pop_tuning_curves,
    pop_tuning_metrics,
    sign="pos",
    smooth_SD=2,
    normalise="max",
    downsample=True,
    neuron_groups=6,
    distance_groups=6,
):
    """ """
    if sign == "pos":
        keep_clusters = pop_tuning_metrics[pop_tuning_metrics.gamma_4p["size"].gt(0)].cluster_unique_ID.values
    elif sign == "neg":
        keep_clusters = pop_tuning_metrics[pop_tuning_metrics.gamma_4p["size"].lt(0)].cluster_unique_ID.values
    df = pop_tuning_curves[keep_clusters].T
    x = df.columns.astype(float).values  # distance bin mids
    df = df.unstack(level=1).swaplevel(0, 1, axis=1).sort_index(axis=1)  # n_clusters, 2 (peak, trough) * n_distances
    # order clusters by distance peak
    if sign == "pos":
        df["idx_order"] = _get_idx_order(pop_tuning_metrics, df.index.values, x, fit="gamma_4p", op="max")
    elif sign == "neg":
        df["idx_order"] = _get_idx_order(pop_tuning_metrics, df.index.values, x, fit="gamma_4p", op="min")
    else:
        raise ValueError("sign must be 'pos' or 'neg'")
    df = df.sort_values(by=[("idx_order", "")], ascending=True)
    df.drop(columns=[("idx_order", "")], inplace=True)
    # smooth & normalise
    if smooth_SD:
        df.loc[:, "peak"] = gaussian_filter1d(df.peak.values, smooth_SD, axis=1)
        df.loc[:, "trough"] = gaussian_filter1d(df.trough.values, smooth_SD, axis=1)
    if normalise == "max":
        grand_max = df.max(axis=1)
        df.loc[:, "peak"] = df.peak.div(grand_max, axis=0).values
        df.loc[:, "trough"] = df.trough.div(grand_max, axis=0).values
    if not downsample:
        return df
    else:
        assert neuron_groups > 0, "neuron_groups must be greater than 0 for downsmapling"
        group_means_df = _downsample_neurons(df, neuron_groups)
        if distance_groups:
            group_means_df = _downsample_distances(group_means_df, distance_groups)
        return group_means_df


def _downsample_neurons(df, n_bins):
    """Downsample the neuron tuning DataFrame by averaging over n_bins."""
    n_neurons, n_distances = df.shape
    # chunk neurons into n groups and average
    neuron_group_size = n_neurons // n_bins
    _n_groups = np.minimum(np.arange(n_neurons) // neuron_group_size, n_bins - 1)
    return df.groupby(_n_groups).mean()


def _downsample_distances(df, n_bins):
    """
    Downsample the distance tuning DataFrame by averaging over n_bins.
    The DataFrame should have a MultiIndex with the first level being 'peak' or 'trough'
    and the second level being the distance bins.
    """
    parts = []
    for kind in ["peak", "trough"]:
        sub = df[kind]  # shape (n_neuron_groups, n_distances)
        distances = sub.columns.astype(float).values  # original distances
        n = len(distances)
        size = n // n_bins
        # assign each original column to an integer bin
        bins = np.arange(n) // size
        bins = np.minimum(bins, n_bins - 1)
        # collapse into bin‐means via transpose‐group‐transpose
        agg = sub.T.groupby(bins).mean().T
        # compute the midpoint of each bin
        mids = []
        for j in range(n_bins):
            idx = np.where(bins == j)[0]
            if idx.size:
                d0 = distances[idx].min()
                d1 = distances[idx].max()
                mids.append((d0 + d1) / 2.0)
            else:
                mids.append(np.nan)
        agg.columns = pd.MultiIndex.from_tuples([(kind, mid) for mid in mids], names=df.columns.names)
        parts.append(agg)
    return pd.concat(parts, axis=1).sort_index(axis=1)


def _get_idx_order(pop_tuning_metrics, cluster_unique_IDs, x, fit="gamma_4p", op="max"):
    metrics_df = pop_tuning_metrics.set_index("cluster_unique_ID")
    params = metrics_df.loc[cluster_unique_IDs][fit]
    curve_fits = dtm.gamma_4p(
        x[:, None],
        params["size"].values,
        params["shape"].values,
        params["scale"].values,
        params["shift"].values,
    )
    if op == "max":
        idx_order = np.argmax(curve_fits, axis=0)
    elif op == "min":
        idx_order = np.argmin(curve_fits, axis=0)
    x_orders = x[idx_order]
    return x_orders


def plot_subject_theta_x_shifts(ax=None, print_stats=True):
    """ """
    # set up fig
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_ylabel("opt. x-shift (cm) \n (theta peak - theta trough)")

    # load population x-shifts for each subject (dict)
    x_shifts = get_population_distance_tuning_theta_x_shifts()
    colors = sns.color_palette("hls", len(SUBJECT_IDS))
    offset = 0.02
    for i, subject in enumerate(SUBJECT_IDS):
        color = colors[i]
        shift = x_shifts[subject] * 100  # convert to cm
        ax.scatter(offset * i, shift, color=color, s=60)
    ax.set_xlim(-0.1, 0.15)
    ax.set_xticks([])
    ax.set_ylim(-5, 1)

    if print_stats:
        _shifts = np.array(list(x_shifts.values()))
        t_val, p_val = ttest_1samp(_shifts, 0, alternative="less")
        print(f"t-test: t = {t_val:.3f}, p = {p_val:.3f}")


def get_population_distance_tuning_theta_x_shifts(
    min_split_half_corr=0.7,
    shift=0.08,
    bin_spacing=0.04,
    smooth_SD=3,
    normalise=False,
    upsampled_spacing=0.001,
    save=False,
    verbose=False,
):
    """ """
    save_path = RESULTS_DIR / "population_theta_x_shifts.json"
    if not save and save_path.exists():
        if verbose:
            print(f"Loading existing results from {save_path}")
        with open(save_path, "r") as input_file:
            x_shifts = json.load(input_file)
        return x_shifts
    x_shifts = {}
    for subject in SUBJECT_IDS:
        tuning_df, _ = get_population_theta_split_distance_tuning(
            subject_ID=subject,
            verbose=verbose,
            min_split_half_corr=min_split_half_corr,
        )
        x_shifts[subject] = get_opt_heatmap_x_shift(
            tuning_df,
            shift,
            bin_spacing,
            smooth_SD,
            normalise,
            upsampled_spacing,
        )
    if save:
        if verbose:
            print(f"Saving results to {save_path}")
        with open(save_path, "w") as output_file:
            json.dump(x_shifts, output_file)
    return x_shifts


def get_opt_heatmap_x_shift(
    tuning_df, shift=0.08, bin_spacing=0.04, smooth_SD=3, normalise=False, upsampled_spacing=0.001
):
    """
    Find best x-shift moving across whole heatmaps from peak and trough
    """
    df = tuning_df.T.unstack(level=1).swaplevel(0, 1, axis=1).sort_index(axis=1)
    if smooth_SD:
        df.loc[:, "peak"] = gaussian_filter1d(df.peak.values, smooth_SD, axis=1)
        df.loc[:, "trough"] = gaussian_filter1d(df.trough.values, smooth_SD, axis=1)
    if normalise == "max":
        grand_max = df.max(axis=1)
        df.loc[:, "peak"] = df.peak.div(grand_max, axis=0).values
        df.loc[:, "trough"] = df.trough.div(grand_max, axis=0).values
    peak = df["peak"].values
    trough = df["trough"].values
    # upsample
    current_bins = peak.shape[1]
    upsampled_bins = int(current_bins * (bin_spacing / upsampled_spacing))
    x_new = np.linspace(0, current_bins - 1, upsampled_bins)
    peak_upsamp = interp1d(np.arange(current_bins), peak, axis=1, kind="quadratic")(x_new)
    trough_upsamp = interp1d(np.arange(current_bins), trough, axis=1, kind="quadratic")(x_new)
    # calculate MSE for all shifts
    n_shifts = int(shift / upsampled_spacing)
    # mask the edges of the upsampled data up to the max shift
    shift_mask = np.zeros_like(peak_upsamp, dtype=bool)
    shift_mask[:, :n_shifts] = True
    shift_mask[:, -n_shifts:] = True
    _peak_upsamp = peak_upsamp[~shift_mask]
    _shifts = np.arange(-n_shifts, n_shifts + 1, 1)
    mses = []
    for i, _shift in enumerate(_shifts):
        shifted_trough = np.roll(trough_upsamp, _shift, axis=1)
        _shifted_trough = shifted_trough[~shift_mask]
        mses.append(((_peak_upsamp - _shifted_trough) ** 2).mean())
    best_shift = _shifts[np.argmin(mses)]
    return best_shift * upsampled_spacing


def get_population_theta_split_distance_tuning(
    subject_ID="all",
    method="peak_trough",
    days_on_maze="all",
    peak_trough_inds=([4, 5, 6], [9, 10, 11]),
    verbose=True,
    min_split_half_corr=0.7,
):
    """ """
    all_tuning_curves = []
    all_metrics = []
    _subject_IDs = [subject_ID] if not subject_ID == "all" else SUBJECT_IDS
    for subject_ID in _subject_IDs:
        if verbose:
            print(subject_ID)
            print("loading sessions ...")
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names="all",
            days_on_maze=days_on_maze,
            with_data=[
                "navigation_df",
                "navigation_theta_spike_counts_df",
                "cluster_distance_tuning_metrics",
            ],
        )
        for session in sessions:
            if verbose:
                print(session.name)
            if method == "peak_trough":
                tuning_curves = get_session_theta_split_distance_tuning(
                    session,
                    min_split_half_corr=min_split_half_corr,
                    theta_peak_ind=peak_trough_inds[0],
                    theta_trough_ind=peak_trough_inds[1],
                )
            elif method == "mean_all_phases":
                tuning_curves = get_session_theta_split_distance_tuning_all_phases(
                    session, min_split_half_corr=min_split_half_corr
                )
            if tuning_curves is None:
                continue  # no distance tunned cells
            distance_metrics = session.cluster_distance_tuning_metrics
            distance_metrics = distance_metrics[distance_metrics.split_half_corr.value > min_split_half_corr]
            all_tuning_curves.append(tuning_curves)
            all_metrics.append(distance_metrics)
    pop_tuning_curves = pd.concat(all_tuning_curves, axis=1)
    pop_tuning_metrics = pd.concat(all_metrics, axis=0)
    return pop_tuning_curves, pop_tuning_metrics


def get_session_theta_split_distance_tuning(
    session,
    metrics=("distance_to_goal", "geodesic"),
    min_split_half_corr=0.7,
    theta_peak_ind=[3, 4, 5, 6],
    theta_trough_ind=[0, 9, 10, 11],
    bin_spacing=0.04,
    max_steps_to_goal=30,
    moving_only=True,
):
    """ """
    # load data
    navigation_df = session.navigation_df.copy()
    theta_spike_counts = session.navigation_theta_spike_counts_df.reset_index(drop=True).copy()
    distance_tuning_metrics = session.cluster_distance_tuning_metrics
    # filter for single units
    valid_units = distance_tuning_metrics[
        distance_tuning_metrics.single_unit & (distance_tuning_metrics.split_half_corr.value > min_split_half_corr)
    ].cluster_unique_ID.values
    if len(valid_units) == 0:
        # no clusters with sufficient distance tuning
        return None
    keep_cols = theta_spike_counts.columns.get_level_values(1).isin(valid_units)
    theta_spike_counts = theta_spike_counts[theta_spike_counts.columns[keep_cols]]
    # get theta phases
    phases = theta_spike_counts.columns.get_level_values(2).unique().astype(float)
    phase_cols = theta_spike_counts.columns.get_level_values(2)
    theta_peak_vals = phases[theta_peak_ind]
    theta_trough_vals = phases[theta_trough_ind]
    # sum spikes in theta peak and trough phases for each cluster
    peak_spike_counts = theta_spike_counts[theta_spike_counts.columns[phase_cols.isin(theta_peak_vals)]]
    peak_spike_counts = peak_spike_counts.T.groupby(level=1).sum().T
    trough_spike_counts = theta_spike_counts[theta_spike_counts.columns[phase_cols.isin(theta_trough_vals)]]
    trough_spike_counts = trough_spike_counts.T.groupby(level=1).sum().T
    # combine nav and spike
    navigation_df.columns = pd.MultiIndex.from_tuples([(*c, "") for c in navigation_df.columns])
    peak_spike_counts.columns = pd.MultiIndex.from_tuples(
        [("spike_count", c, "peak") for c in peak_spike_counts.columns]
    )
    trough_spike_counts.columns = pd.MultiIndex.from_tuples(
        [("spike_count", c, "trough") for c in trough_spike_counts.columns]
    )
    nav_spikes_df = pd.concat(
        [navigation_df, peak_spike_counts, trough_spike_counts],
        axis=1,
    )
    metrics = (*metrics, "")
    # now filter the data
    nav_spikes_df = filt.filter_navigation_rates_df(
        nav_spikes_df,
        navigation_only=True,
        moving_only=moving_only,
        exclude_time_at_goal=False,
        max_steps_to_goal=max_steps_to_goal,
    ).reset_index(drop=True)
    # bin distances to goal
    max_distance = dd.get_distance_percentile(metrics, 0.85)
    n_bins = int(max_distance / bin_spacing)
    nav_spikes_df = nav_spikes_df[nav_spikes_df[metrics] < max_distance]
    bins = convert._get_distance_bins(
        binning_method="uniform",
        n_distance_bins=n_bins,
        distance_metrics=metrics,
        max_distance=max_distance,
    )
    # bin distances
    nav_spikes_df.loc[:, ("distance_bin", "", "")] = pd.cut(
        nav_spikes_df[metrics], bins=bins, include_lowest=True
    ).to_numpy()
    # get average rates at each distance over trials
    grouped_df = nav_spikes_df.groupby(["trial", "distance_bin"]).spike_count
    distance_occ = grouped_df.count() * (1 / FRAME_RATE)  # convert to seconds
    distance_spikes = grouped_df.sum()
    distance_theta_rates = distance_spikes / distance_occ
    # average over trials
    distance_theta_tuning = distance_theta_rates.groupby("distance_bin").spike_count.mean().sort_index(axis=1)
    distance_theta_tuning.index = [c.mid for c in distance_theta_tuning.index]
    return distance_theta_tuning.spike_count


# %%


def get_session_theta_split_distance_tuning_all_phases(
    session,
    metrics=("distance_to_goal", "geodesic"),
    min_split_half_corr=0.7,
    bin_spacing=0.04,
    max_steps_to_goal=30,
    moving_only=True,
):
    """ """
    # load data
    navigation_df = session.navigation_df.copy()
    theta_spike_counts = session.navigation_theta_spike_counts_df.reset_index(drop=True).copy()
    distance_tuning_metrics = session.cluster_distance_tuning_metrics
    # filter for single units
    valid_units = distance_tuning_metrics[
        distance_tuning_metrics.single_unit & (distance_tuning_metrics.split_half_corr.value > min_split_half_corr)
    ].cluster_unique_ID.values
    if len(valid_units) == 0:
        # no clusters with sufficient distance tuning
        return None
    keep_cols = theta_spike_counts.columns.get_level_values(1).isin(valid_units)
    theta_spike_counts = theta_spike_counts[theta_spike_counts.columns[keep_cols]]
    # get distance tuning from mean spikes across theta phases
    mean_spikes = theta_spike_counts.spike_count.T.groupby(level=0).mean().T
    mean_spikes.columns = pd.MultiIndex.from_tuples([("spike_count", c, "mean") for c in mean_spikes.columns])
    # combine nav and spike
    navigation_df.columns = pd.MultiIndex.from_tuples([(*c, "") for c in navigation_df.columns])
    nav_spikes_df = pd.concat(
        [navigation_df, mean_spikes, theta_spike_counts],
        axis=1,
    )
    # now filter the data
    nav_spikes_df = filt.filter_navigation_rates_df(
        nav_spikes_df,
        navigation_only=True,
        moving_only=moving_only,
        exclude_time_at_goal=False,
        max_steps_to_goal=max_steps_to_goal,
    ).reset_index(drop=True)
    # bin distances to goal
    metrics = (*metrics, "")
    max_distance = dd.get_distance_percentile(metrics, 0.85)
    n_bins = int(max_distance / bin_spacing)
    nav_spikes_df = nav_spikes_df[nav_spikes_df[metrics] < max_distance]
    bins = convert._get_distance_bins(
        binning_method="uniform",
        n_distance_bins=n_bins,
        distance_metrics=metrics,
        max_distance=max_distance,
    )
    # bin distances
    nav_spikes_df.loc[:, ("distance_bin", "", "")] = pd.cut(
        nav_spikes_df[metrics], bins=bins, include_lowest=True
    ).to_numpy()
    # get average rates at each distance over trials
    grouped_df = nav_spikes_df.groupby(["trial", "distance_bin"]).spike_count
    distance_occ = grouped_df.count() * (1 / FRAME_RATE)  # convert to seconds
    distance_spikes = grouped_df.sum()
    distance_theta_rates = distance_spikes / distance_occ
    # average over trials
    distance_theta_tuning = distance_theta_rates.groupby("distance_bin").spike_count.mean().sort_index(axis=1)
    distance_theta_tuning.index = [c.mid for c in distance_theta_tuning.index]
    return distance_theta_tuning.spike_count


def get_opt_heatmap_x_shift_all_phases(
    tuning_df, shift=0.08, bin_spacing=0.04, smooth_SD=3, normalise=False, upsampled_spacing=0.001
):
    """
    Find best x-shift moving across whole heatmaps from peak and trough
    """
    # get theta phases
    phases = [c for c in tuning_df.columns.get_level_values(1).unique() if c != "mean"]
    df = tuning_df.T.swaplevel(0, 1, axis=0).sort_index(axis=0)
    if smooth_SD:
        df = pd.DataFrame(
            data=gaussian_filter1d(df.values, axis=1, sigma=smooth_SD), index=df.index, columns=df.columns
        )
    if normalise == "max":
        grand_max = df.max(axis=1)
        df = df.div(grand_max, axis=0)
    # upsample
    current_bins = df.shape[1]
    upsampled_bins = int(current_bins * (bin_spacing / upsampled_spacing))
    x_new = np.linspace(0, current_bins - 1, upsampled_bins)
    df_upsampled = pd.DataFrame(
        data=interp1d(np.arange(current_bins), df.values, axis=1, kind="quadratic")(x_new), index=df.index
    )
    # calculate MSE for all shifts
    n_shifts = int(shift / upsampled_spacing)
    mean_tuning = df_upsampled.loc["mean"].values  # n_neurons x n_upsampled_distance_bins
    # mask the edges of the upsampled data up to the max shift
    shift_mask = np.zeros_like(mean_tuning, dtype=bool)
    shift_mask[:, :n_shifts] = True
    shift_mask[:, -n_shifts:] = True
    _mean_tuning = mean_tuning[~shift_mask]
    _shifts = np.arange(-n_shifts, n_shifts + 1, 1)
    mses = np.zeros((len(phases), len(_shifts)))
    for i, phase in enumerate(phases):
        phase_tuning = df_upsampled.loc[phase].values  # n_neurons x n_upsampled_distance_bins
        for j, _shift in enumerate(_shifts):
            shifted_phase_tuning = np.roll(phase_tuning, _shift, axis=1)
            shifted_phase_tuning = shifted_phase_tuning[~shift_mask]
            mses[i, j] = ((_mean_tuning - shifted_phase_tuning) ** 2).mean()
    phase_opt_x_shifts = _shifts[np.argmin(mses, axis=1)] * upsampled_spacing
    return phase_opt_x_shifts, phases


def get_population_distance_tuning_theta_x_shifts_all_phases(
    days_on_maze="late",
    min_split_half_corr=0.7,
    shift=0.08,
    bin_spacing=0.04,
    smooth_SD=3,
    normalise=False,
    upsampled_spacing=0.001,
    save=False,
    verbose=False,
):
    """ """
    save_path = RESULTS_DIR / "population_theta_x_shifts_all_phases.csv"
    if not save and save_path.exists():
        if verbose:
            print(f"Loading existing results from {save_path}")
        results_df = pd.read_csv(save_path, index_col=0)
        return results_df

    subject_opt_x_shifts = []
    for subject in SUBJECT_IDS:
        tuning_df, _ = get_population_theta_split_distance_tuning(
            subject_ID=subject,
            method="mean_all_phases",
            verbose=verbose,
            days_on_maze=days_on_maze,
            min_split_half_corr=min_split_half_corr,
        )
        opt_x_shifts, phases = get_opt_heatmap_x_shift_all_phases(
            tuning_df,
            shift,
            bin_spacing,
            smooth_SD,
            normalise,
            upsampled_spacing,
        )
        subject_opt_x_shifts.append(opt_x_shifts)
    results_df = pd.DataFrame(data=np.array(subject_opt_x_shifts).T, index=phases, columns=SUBJECT_IDS)
    if save:
        if verbose:
            print(f"Saving results to {save_path}")
        results_df.to_csv(save_path)
    return results_df


def plot_theta_mod_x_shifts(
    results_df,
    late_sessions=False,
    color="dodgerblue",
    ref_color="darkblue",
    plot_decoding_ref=True,
    print_stats=True,
    ax=None,
):
    # set up fig
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2, 2))
    # convert to cm
    df = results_df.copy()
    df = df.T
    df = df.mul(100)
    # plot
    tmu.plot_decoding_bias(
        df, color=color, label=None, ax=ax, ylabel="tuning bias \n x-shift (cm)", print_stats=print_stats
    )
    if plot_decoding_ref:
        # load decoding results and det analagous modulation bias df
        decoding_mod_df = ddv2.get_theta_mod_distance_error_df()
        _decoding_mod_df = ddv2._filter_summary_df(decoding_mod_df, late_sessions=late_sessions)
        decoding_bias = _decoding_mod_df.groupby(["subject_ID", "theta_phase"])["signed_error"].mean().unstack(0).T
        decoding_bias_norm = decoding_bias.sub(decoding_bias.mean(axis=1), axis=0)
        decoding_bias_cm = decoding_bias_norm.mul(100)  # m -> cm
        # fit sinusoid to subject-averaged decoding results, rescale amplitude to match tuning fit
        phases = decoding_bias_cm.columns.values.astype(float)
        decoding_mean = decoding_bias_cm.mean(axis=0).values
        dec_params = tmu.fit_sinusoid(phases, decoding_mean, fit_constant=True, return_as="params")
        tuning_params = tmu.fit_sinusoid(phases, df.mean(axis=0).values, fit_constant=True, return_as="params")
        scale = tuning_params["A"] / dec_params["A"]
        _x = np.linspace(-np.pi, np.pi, 100)
        _y = scale * dec_params["A"] * np.sin(_x + dec_params["phi"])
        ax.plot(_x, _y, color=ref_color, alpha=0.3, linewidth=1.5)
        if print_stats:
            # test if decoding and tuning are sig offset
            print("tuning-bias vs decoding bias:")
            tmu.test_theta_offset(df, decoding_bias_norm)


def get_mod_max(bias_df):
    """ """
    subject_IDs = bias_df.index.values
    phases = bias_df.columns.values.astype(float)
    x_max = np.zeros(len(subject_IDs))
    for i, subject_ID in enumerate(subject_IDs):
        y = bias_df.loc[subject_ID].values
        x_fit, y_fit = tmu.fit_sinusoid(phases, y, return_as="curve")
        x_max[i] = x_fit[np.argmax(y_fit)]
    return x_max
