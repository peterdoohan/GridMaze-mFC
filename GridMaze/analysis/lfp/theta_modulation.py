"""
LFP theta-modulation analyses.

Two roles bundled here:
  1. Theta-aligned LFP characterisation — what the theta cycle looks like in the raw signal.
  2. Per-cell theta modulation — which neurons are theta-modulated, how strongly, and at
     what phase, optionally stratified by feature tuning (distance-to-goal, place-direction).

Distinct from `GridMaze.analysis.theta_mod/` (theta-phase-conditional decoding analyses).
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import delayed, Parallel
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from GridMaze.analysis.core import convert
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.lfp import lfp_utils as lu
from GridMaze.analysis.neGLM import load_model_sets as lms
from GridMaze.analysis.neGLM import variance_explained as ve
from GridMaze.analysis.processing import get_lfp_aligned_spike_counts as la
from GridMaze.analysis.theta_mod import double_decoding as tdd

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with open(EXPERIMENT_INFO_PATH / "maze_configs.json", "r") as input_file:
    MAZE_CONFIGS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "maze_day2date.json", "r") as input_file:
    MAZE_DAY2DATE = json.load(input_file)

RESULTS_DIR = RESULTS_PATH / "lfp"

THETA_RANGE = (7, 11)


# %% Theta-aligned LFP (what theta looks like in the raw signal)


def plot_theta_aligned_lfp(theta_aligned_df, ax=None, color="crimson"):
    """ """
    # average signal across sessions for each subject
    subject_means = theta_aligned_df.T.groupby(level=0).mean()
    # plot mean and sem across subjects
    mean = subject_means.mean()
    sem = subject_means.sem()
    phases = mean.index.values.astype(float)
    # plotting
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot(
        phases,
        mean.values,
        color=color,
        linewidth=2,
    )
    ax.fill_between(
        phases,
        mean.values - sem.values,
        mean.values + sem.values,
        color=color,
        alpha=0.3,
    )
    ax.set_xlabel("theta phase")
    ax.set_ylabel("LFP (uV)")
    ax.set_xticks(np.arange(-np.pi, np.pi + 0.1, np.pi / 2))
    ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])


def get_theta_aligned_lfp_df(save=False, verbose=False):
    """
    Note get sessions one-by-one to avoid memory issues
    with massive LFP arrays.
    """
    save_path = RESULTS_DIR / "theta_aligned_lfp.csv"
    if save_path.exists() and not save:
        if verbose:
            print(f"Loading theta aligned lfp from {save_path}")
        return pd.read_csv(save_path, index_col=[0], header=[0, 1])

    aligned_lfps = []
    for subject in SUBJECT_IDS:
        for maze in MAZE_CONFIGS.keys():
            days_on_maze = [int(d) for d in MAZE_DAY2DATE[maze].keys()]
            late_days = days_on_maze[-7:]
            for day in late_days:
                try:
                    session = gs.get_maze_sessions(
                        subject_IDs=[subject],
                        maze_names=[maze],
                        days_on_maze=[day],
                        with_data=["lfp_times", "lfp_signal", "lfp_metrics", "cluster_metrics"],
                        must_have_data=True,
                    )
                    if verbose:
                        print(session.name)
                    theta_aligned_lfp = get_session_theta_aligned_lfp(session)
                    aligned_lfps.append(theta_aligned_lfp)
                except FileNotFoundError:
                    pass  # minority of sessions missing data
    theta_alinged_df = pd.concat(aligned_lfps, axis=1)
    if save:
        if verbose:
            print(f"Saving theta aligned lfp to {save_path}")
        theta_alinged_df.to_csv(save_path)
    return theta_alinged_df


def get_session_theta_aligned_lfp(session, n_bins=32):
    """ """
    lfp_signal = lu.get_LFP(session)
    # get theta phase
    theta_phase = la.get_lfp_phase(lfp_signal, freq_range=THETA_RANGE, N=4)
    # bin phases finely
    bin_edges, theta_phase_bins = la.bin_lfp_phase(theta_phase, n_bins=n_bins)
    # average lfp signal in each phase bin
    theta_aligned_lfp = np.zeros(len(bin_edges) - 1)
    for i in range(n_bins):
        theta_aligned_lfp[i] = lfp_signal[theta_phase_bins == i].mean()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return pd.Series(index=bin_centers, data=theta_aligned_lfp, name=(session.subject_ID, session.name))


# %% Per-cell theta modulation (proportion of spikes / preferred phase across the population)


def plot_population_theta_mod(population_theta_df, ax=None):
    """ """
    # average theta mod across subjects
    sub_mean_df = population_theta_df.groupby("subject_ID").prop_spikes.mean().prop_spikes
    # normalise and conver to % normalised firing rate
    sub_mean_df = sub_mean_df.div(sub_mean_df.mean(axis=1), axis=0).mul(100)
    # plot mean and sem across subjects
    mean = sub_mean_df.mean()
    sem = sub_mean_df.sem()
    phases = mean.index.values.astype(float)
    # plotting
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.axhline(100, color="k", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.errorbar(
        phases,
        mean.values,
        yerr=sem.values,
        fmt="o-",
        color="k",
        markersize=6,
        linewidth=2,
        capsize=None,
        elinewidth=2,
    )
    ax.set_ylim(97, 103)
    ax.set_xlabel("theta phase")
    ax.set_ylabel("Norm. firing rate (%)")
    ax.set_xticks(np.arange(-np.pi, np.pi + 0.1, np.pi / 2))
    ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])


def plot_population_theta_pref(population_theta_df, ax=None):
    """ """
    # get each cluster's preferred theta phase
    cluster_prefs = population_theta_df.idxmax(axis=1)
    # count preferences for each subject
    subject_counts = cluster_prefs.groupby(level=0).value_counts().unstack()
    # normalise to prop of clusters per subject
    subject_counts = subject_counts.div(subject_counts.sum(axis=1), axis=0)
    subject_counts.columns = subject_counts.columns.astype(float)
    subject_counts = subject_counts.sort_index(axis=1)  # sort by phase
    # plot
    mean = subject_counts.mean()
    sem = subject_counts.sem()
    phases = mean.index.values.astype(float)
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3))
    _even_split = 1 / len(phases)
    ax.axhline(_even_split, color="k", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.errorbar(
        phases,
        mean.values,
        yerr=sem.values,
        fmt="o-",
        color="k",
        markersize=6,
        linewidth=2,
        capsize=None,
        elinewidth=2,
    )
    ax.set_xlabel("theta phase")
    ax.set_ylabel("prop. population")
    ax.set_xticks(np.arange(-np.pi, np.pi + 0.1, np.pi / 2))
    ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])
    return


def get_population_theta_mod(verbose=False, save=False):
    """ """
    save_path = RESULTS_DIR / "population_theta_mod2.csv"
    if save_path.exists() and not save:
        if verbose:
            print(f"Loading population theta modulation from {save_path}")
        df = pd.read_csv(save_path, index_col=[0], header=[0, 1])
        df.columns = pd.MultiIndex.from_tuples([(c if "Unnamed" not in c[1] else (c[0], "")) for c in df.columns])
        return df

    dfs = []
    for subject_ID in SUBJECT_IDS:
        if verbose:
            print(f"Loading sessions for {subject_ID}")
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names="all",
            days_on_maze="late",
            with_data=[
                "navigation_df",
                "cluster_metrics",
                "navigation_theta_spike_counts_df",
                "trials_df",
            ],
            must_have_data=True,
        )
        subject_dfs = Parallel(n_jobs=-1, verbose=verbose)(
            delayed(get_session_theta_mod)(session) for session in sessions
        )
        dfs.extend(subject_dfs)

    pop_theta_mod = pd.concat(dfs, axis=0)
    if save:
        if verbose:
            print(f"Saving population theta modulation to {save_path}")
        pop_theta_mod.to_csv(save_path)
    return pop_theta_mod


def get_session_theta_mod(
    session, navigation_only=True, include_multi_unit=True, moving_only=True, max_steps_to_goal=30, min_spikes=300
):
    """ """
    # load data
    session_info = session.session_info
    cluster_metrics = session.cluster_metrics
    navigation_df = session.navigation_df.copy()
    theta_spike_counts_df = session.navigation_theta_spike_counts_df.reset_index(drop=True)
    # filter for single units
    if not include_multi_unit:
        keep_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID
    else:
        keep_units = cluster_metrics[cluster_metrics.single_unit | cluster_metrics.multi_unit].cluster_ID
    keep_units = convert.cluster_IDs2scluster_unique_IDs(session_info, keep_units)
    theta_spike_counts_df = theta_spike_counts_df[
        theta_spike_counts_df.columns[[c in keep_units for c in theta_spike_counts_df.columns.get_level_values(1)]]
    ]
    # combine nav and spikes
    navigation_df.columns = pd.MultiIndex.from_tuples([(*c, "") for c in navigation_df.columns])
    nav_spike_counts_df = pd.concat([navigation_df, theta_spike_counts_df], axis=1)
    # filter for moving, navigation, on task etc.
    nav_spike_counts_df = filt.filter_navigation_rates_df(
        nav_spike_counts_df, navigation_only, moving_only, max_steps_to_goal=max_steps_to_goal
    )
    cluster_phase_spike_counts = nav_spike_counts_df.spike_count.sum().unstack()
    # filter for cluster with few spikes in filtered data (eg, non-navigation tuned)
    cluster_phase_spike_counts = cluster_phase_spike_counts[cluster_phase_spike_counts.sum(axis=1) > min_spikes]
    # normalise each to prop (sum =1) of spikes in each phase
    df = cluster_phase_spike_counts.div(cluster_phase_spike_counts.sum(axis=1), axis=0)
    df.columns = pd.MultiIndex.from_product([["prop_spikes"], df.columns])
    # add other info
    df[("subject_ID", "")] = session.subject_ID
    df[("maze_name", "")] = session.maze_name
    df[("day_on_maze", "")] = session.day_on_maze
    df[("tissue_sample", "")] = session.tissue_sample
    df[("probe_depth", "")] = session.probe_depth
    return df


# %% Population-level theta-modulation tuning curves (stratified by feature tuning)


def plot_subpopulation_theta_mod(
    theta_mod_df,
    populations=("all", "place_direction", "distance_to_goal"),
    colors=None,
    print_stats=True,
    ylim=(0.95, 1.05),
    ax=None,
):
    """
    Plots the average theta modulation of all neurons, and specific subpopulations of
    distance-to-goal tuned neurons and place-direction tuned neurons as identified in
    neGLM analyses. `populations` selects which subpopulations to plot and which
    stats to print (valid ids: "all", "place_direction", "distance_to_goal").
    `colors` is a sequence aligned with `populations`; if None, per-population defaults
    are used.
    """
    default_colors = {"all": "silver", "place_direction": "darkred", "distance_to_goal": "purple"}
    if colors is None:
        colors = [default_colors[p] for p in populations]
    elif len(colors) != len(populations):
        raise ValueError(f"colors (len {len(colors)}) must match populations (len {len(populations)})")
    # set up fig
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    ax.axhline(1, color="k", ls="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("theta phase")
    ax.set_ylabel("population \n theta modulation")
    ax.set_xticks(np.arange(-np.pi, np.pi + 0.1, np.pi / 2))
    ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])

    # process data
    df = theta_mod_df.copy()
    phases = df.spike_count.columns.values.astype(float)
    # norm neurons
    df["spike_count"] = df["spike_count"].astype(float)
    T = df.spike_count.values
    T = T / T.mean(axis=1, keepdims=True)
    df.loc[:, ("spike_count", slice(None))] = T

    all_neurons_mask = np.ones(len(df), dtype=bool)
    place_direction_mask = df.feature_tuning.place_direction & ~df.feature_tuning.distance_to_goal
    distance_mask = ~df.feature_tuning.place_direction & df.feature_tuning.distance_to_goal

    pop_specs = {
        "all": (all_neurons_mask, "all"),
        "place_direction": (place_direction_mask, "place-direction tuned"),
        "distance_to_goal": (distance_mask, "distance-to-goal tuned"),
    }
    unknown = [p for p in populations if p not in pop_specs]
    if unknown:
        raise ValueError(f"unknown populations: {unknown}. valid: {list(pop_specs)}")

    mod_dfs = {}
    for pop_id, color in zip(populations, colors):
        mask, label = pop_specs[pop_id]
        _df = df[mask]
        subj_avg = _df.groupby("subject_ID").spike_count.mean().spike_count
        _mean = subj_avg.mean()
        _sem = subj_avg.sem()
        ax.errorbar(
            phases,
            _mean,
            yerr=_sem,
            fmt="o",
            color=color,
            markersize=5,
            linewidth=None,
            capsize=None,
            elinewidth=1.5,
        )
        _x, _y = tdd.fit_sinusoid(phases, _mean, fit_constant=True, return_as="curve")
        ax.plot(_x, _y, color=color, linewidth=1.5, label=label)
        mod_dfs[pop_id] = subj_avg
    ax.legend(fontsize=6)
    ax.set_ylim(*ylim)
    if print_stats:
        pop_print_labels = {
            "all": "all neurons",
            "place_direction": "place-direction tuned neurons",
            "distance_to_goal": "distance-to-goal tuned neurons",
        }
        for pop_id in populations:
            print(pop_print_labels[pop_id])
            tdd._get_decoding_bias_stats(mod_dfs[pop_id])

        pair_print_labels = {
            ("all", "place_direction"): "all vs. place-direction",
            ("all", "distance_to_goal"): "all vs. distance",
            ("distance_to_goal", "place_direction"): "distance vs. place-direction",
        }
        for (a, b), label in pair_print_labels.items():
            if a in populations and b in populations:
                print(label)
                tdd.test_theta_offset(mod_dfs[a], mod_dfs[b])


def get_population_theta_mod_tuning(late_sessions=True, include_multi_units=True, save=False):
    """ """
    save_path = RESULTS_DIR / "theta_mod" / "population_theta_mod_tuning.parquet"
    if not save and save_path.exists():
        return pd.read_parquet(save_path)
    feature_tuned_df = get_tuned_neurons()
    dfs = []
    for subject in SUBJECT_IDS:
        for maze_name in ["maze_1", "maze_2", "rooms_maze"]:
            sessions = gs.get_maze_sessions(
                subject_IDs=[subject],
                maze_names=[maze_name],
                days_on_maze="late" if late_sessions else "all",
                with_data=["navigation_df", "navigation_theta_spike_counts_df", "cluster_metrics"],
                must_have_data=True,
            )
            for session in sessions:
                df = get_session_theta_mod_tuning(
                    session,
                    include_multi_units=include_multi_units,
                    navigation_only=True,
                    moving_only=True,
                    norm=False,
                    max_steps_to_goal=30,
                    min_spikes=300,
                    feature_tuned_df=feature_tuned_df,
                )
                dfs.append(df)
    combined_df = pd.concat(dfs)
    if save:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_parquet(save_path)
    return combined_df


def get_session_theta_mod_tuning(
    session,
    include_multi_units=True,
    navigation_only=True,
    moving_only=True,
    norm=False,
    max_steps_to_goal=30,
    min_spikes=300,
    include_feature_tuning=True,
    feature_tuned_df=None,
):
    """ """
    # load data
    session_info = session.session_info
    cluster_metrics = session.cluster_metrics
    navigation_df = session.navigation_df.copy()
    theta_spike_counts_df = session.navigation_theta_spike_counts_df.reset_index(drop=True)
    # filter for single units
    if not include_multi_units:
        keep_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID
    else:
        keep_units = cluster_metrics[cluster_metrics.single_unit | cluster_metrics.multi_unit].cluster_ID
    keep_units = convert.cluster_IDs2scluster_unique_IDs(session_info, keep_units)
    if include_feature_tuning:
        # further check that units have feature tuning ascribed (cells with few spikes don't get run in neGLM)
        keep_units = [u for u in keep_units if u in feature_tuned_df.index.get_level_values(1)]

    if len(keep_units) == 0:
        return None

    theta_spike_counts_df = theta_spike_counts_df[
        theta_spike_counts_df.columns[[c in keep_units for c in theta_spike_counts_df.columns.get_level_values(1)]]
    ]
    # combine nav and spikes
    navigation_df.columns = pd.MultiIndex.from_tuples([(*c, "") for c in navigation_df.columns])
    nav_spike_counts_df = pd.concat([navigation_df, theta_spike_counts_df], axis=1)
    # filter for moving, navigation, on task etc.
    nav_spike_counts_df = filt.filter_navigation_rates_df(
        nav_spike_counts_df, navigation_only, moving_only, max_steps_to_goal=max_steps_to_goal
    )
    cluster_phase_spike_counts = nav_spike_counts_df.spike_count.sum().unstack()
    # filter for cluster with few spikes in filtered data (eg, non-navigation tuned)
    cluster_phase_spike_counts = cluster_phase_spike_counts[cluster_phase_spike_counts.sum(axis=1) > min_spikes]
    # normalise each to avg 1
    if norm:
        df = cluster_phase_spike_counts.div(cluster_phase_spike_counts.mean(axis=1), axis=0)
    else:
        df = cluster_phase_spike_counts.copy()
    df.columns = pd.MultiIndex.from_product([["spike_count"], df.columns])
    # add feature tuning
    if include_feature_tuning:
        subject_ID = session.subject_ID
        if feature_tuned_df is None:
            feature_tuned_df = get_tuned_neurons()

        feature_tuning = feature_tuned_df.loc[subject_ID].loc[df.index]
        feature_tuning.columns = pd.MultiIndex.from_tuples([("feature_tuning", c) for c in feature_tuning.columns])
        df = pd.concat([df, feature_tuning], axis=1)
    # add other info
    df[("subject_ID", "")] = session.subject_ID
    df[("maze_name", "")] = session.maze_name
    df[("day_on_maze", "")] = session.day_on_maze
    return df


def get_tuned_neurons():

    feature_tuned_df = ve.get_feature_tuned_df(
        lms.load_model_set_cv_scores("variance_explained_multiunit"),
        reduced_models=["remove_distance_to_goal", "remove_place_direction"],
    )
    return feature_tuned_df


# %% --- Below: uncalled functions kept for potential supp / follow-up use ---
# %% Define theta-modulated clusters (cutoffs, prevalence, phase × modulation-depth)


def get_pop_theta_mod_metrics(late_sessions=False):
    """ """
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names="all",
        days_on_maze="late" if late_sessions else "all",
        with_data=["cluster_theta_modulation_metrics"],
        must_have_data=True,
    )
    dfs = []
    for session in sessions:
        theta_mod_metrics = session.cluster_theta_modulation_metrics.copy()
        theta_mod_metrics = theta_mod_metrics[theta_mod_metrics.single_unit]
        dfs.append(theta_mod_metrics)
    combined_df = pd.concat(dfs)
    return combined_df.reset_index(drop=True)


def _filter_for_theta_mod_neurons(metrics_df, corr_thres=0.45, r2_thres=0.7, fr_thres=1):
    df = metrics_df.copy()
    df = df[df.split_half_corr.gt(corr_thres) & df.vonmises.r2.gt(r2_thres) & df.mean_firing_rate.gt(fr_thres)]
    return df


def plot_theta_mod_cuttoffs(metrics_df, corr_thres=0.45, r2_thres=0.7, fr_thres=None, ax=None):
    """ """
    # set up fig
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 3))
    ax.spines[["top", "right"]].set_visible(False)

    df = metrics_df.copy()
    if fr_thres is not None:
        df = df[df.mean_firing_rate.gt(fr_thres)]
    df = df[[("split_half_corr", ""), ("vonmises", "r2")]].copy()
    df = df.dropna()
    x, y = df[("split_half_corr", "")], df[("vonmises", "r2")]

    sns.scatterplot(x=x, y=y, color="k", alpha=0.5, s=2, ax=ax)
    sns.histplot(
        x=x,
        y=y,
        bins=30,
        pthresh=0.01,
        cmap="gray",
        ax=ax,
        cbar=True,
        cbar_kws={"label": "neurons", "shrink": 0.5},
    )
    sns.kdeplot(x=x, y=y, levels=8, color="white", linewidths=0.5, ax=ax)

    ax.axvline(corr_thres, color="red", linestyle="--")
    ax.axhline(r2_thres, color="red", linestyle="--")
    x0, y0 = corr_thres, r2_thres
    x_max, y_max = ax.get_xlim()[1], ax.get_ylim()[1]
    width = max(0, x_max - x0)
    height = max(0, y_max - y0)

    rect = Rectangle(
        (x0, y0),
        width,
        height,
        linewidth=1,
        edgecolor="red",
        facecolor="red",
        alpha=0.12,
        linestyle="--",
        zorder=10,
    )
    ax.add_patch(rect)

    ax.set_xlim(-0.5, 1)
    ax.set_ylim(0, 1)
    return


def plot_prop_theta_mod_neurons(metrics_df, corr_thres=0.45, r2_thres=0.7, ax=None):
    """ """
    if ax is None:
        fig, ax = plt.subplots(figsize=(1, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", linewidth=0.5)

    df = metrics_df.copy()
    df[("theta_modulated", "")] = df.split_half_corr.gt(corr_thres) & df.vonmises.r2.gt(r2_thres)

    prop_df = df.groupby(["subject_ID", "theta_modulated"]).maze_name.count().unstack()
    prop_df = prop_df.div(prop_df.sum(axis=1), axis=0)

    prop_cross_subjects = prop_df[True]
    prop_cross_subjects = prop_cross_subjects.reset_index()
    prop_cross_subjects.columns = ["subject_ID", "score"]
    palette = sns.color_palette("hls", n_colors=len(SUBJECT_IDS))
    sns.pointplot(
        data=prop_cross_subjects,
        y="score",
        hue="subject_ID",
        dodge=False,
        errorbar=None,
        markers="o",
        markersize=7,
        markeredgewidth=0,
        linestyle="none",
        palette=palette,
        legend=False,
        alpha=0.8,
        ax=ax,
    )
    ax.set_ylabel("prop. Theta-mod. neurons")


def plot_theta_phase_by_mod_depth(metrics_df, corr_thres=0.45, r2_thres=0.7, axes=None):
    """ """
    # set up fig
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(3, 3), width_ratios=[1, 0.2], height_ratios=[0.2, 1])
    axes[0, 0].spines[["top", "right"]].set_visible(False)  # phase hist
    axes[0, 0].set_xticks([])
    axes[1, 0].spines[["top", "right"]].set_visible(False)  # depth vs phase
    axes[1, 1].spines[["top", "right"]].set_visible(False)  # depth hist
    axes[1, 1].set_yticks([])
    axes[0, 1].set_visible(False)

    # select for theta mod neurons
    df = _filter_for_theta_mod_neurons(metrics_df, corr_thres=corr_thres, r2_thres=r2_thres)

    palette = sns.color_palette("hls", n_colors=len(SUBJECT_IDS))
    for subject, color in zip(SUBJECT_IDS, palette):
        _df = df[df.subject_ID == subject]
        x, y = _df.vonmises.phase_max_rad, _df.vonmises.modulation_depth
        axes[1, 0].scatter(x=x, y=y, color=color, alpha=0.5, s=2)
        sns.histplot(
            x=x,
            bins=20,
            element="step",
            fill=False,
            stat="probability",
            color=color,
            alpha=0.8,
            ax=axes[0, 0],
        )
        sns.histplot(
            y=y,
            bins=20,
            element="step",
            fill=False,
            stat="probability",
            color=color,
            alpha=0.8,
            ax=axes[1, 1],
        )

    axes[0, 0].set_ylabel("Prop.")
    axes[1, 0].set_xlabel("θ Phase")
    axes[1, 0].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axes[1, 0].set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    axes[1, 0].set_ylabel("Modulation Depth")
    axes[1, 1].set_xlabel("Prop.")
    return
