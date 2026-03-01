"""
Look at theta modulation characteristics across the population, including stratification
by cluster tuning
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import convert
from GridMaze.analysis.core import filter as filt


from GridMaze.analysis.processing import get_cluster_unique_variance_explained as cve
from matplotlib.patches import Rectangle

from GridMaze.analysis.theta_mod import double_decoding as tdd

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

RESULTS_DIR = RESULTS_PATH / "lfp" / "theta_mod"

# %%
from GridMaze.analysis.neGLM import load_model_sets as lms
from GridMaze.analysis.neGLM import variance_explained as ve


def get_tuned_neurons():

    feature_tuned_df = ve.get_feature_tuned_df(
        lms.load_model_set_cv_scores("variance_explained_multiunit"),
        reduced_models=["remove_distance_to_goal", "remove_place_direction"],
    )
    return feature_tuned_df


# %% Functions


def plot_subpopulation_theta_mod(theta_mod_df, print_stats=True, ax=None):
    """
    Plots the average theta modulation of all neurons, and specific subpopulations of
    distance-to-goal tuned neurons and place-direction tuned neurons as identified in
    neGLM analyses
    """
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
    mod_dfs = []
    for mask, label, color in zip(
        [all_neurons_mask, place_direction_mask, distance_mask],
        ["all", "place-direction tuned", "distance-to-goal tuned"],
        ["silver", "darkred", "purple"],
    ):
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
        mod_dfs.append(subj_avg)
    ax.legend(fontsize=6)
    ax.set_ylim(0.95, 1.05)
    if print_stats:
        # test if all curves are significanlty modulated
        all_df, place_df, dist_df = mod_dfs
        print("all neurons")
        tdd._get_decoding_bias_stats(all_df)
        print("place-direction tuned neurons")
        tdd._get_decoding_bias_stats(place_df)
        print("distance-to-goal tuned neurons")
        tdd._get_decoding_bias_stats(dist_df)
        # test offsets between populations
        print("all vs. place-direction")
        tdd.test_theta_offset(all_df, place_df)
        print("all vs. distance")
        tdd.test_theta_offset(all_df, dist_df)
        print("distance vs. place-direction")
        tdd.test_theta_offset(dist_df, place_df)


# %% population theta modulation tuning curves


def get_population_theta_mod_tuning(late_sessions=True, include_multi_units=True):
    """ """
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


# %% More detailed analysis of single cell modulation profiles


def get_theta_mod_split_tuning(
    df_type="feature_tuned", late_sessions=False, corr_thres=0.45, r2_thres=0.7, fr_thres=1, ax=None
):
    """ """
    # load data unique var exp df or feature tuned df
    if df_type == "feature_tuned":
        ft_df = cve.get_cluster_feature_tuned_df(
            late_sessions=late_sessions, full_features=False, add_missing_clusters=False
        )
    elif df_type == "variance_explained":
        ft_df = cve.get_cluster_unique_variance_explained(
            late_sessions=late_sessions, full_features=False, add_missing_clusters=False
        )
        ft_df = ft_df.swaplevel(axis=0)
    else:
        raise ValueError("df_type must be 'feature_tuned' or 'variance_explained'")

    # load theta metrics
    metrics_df = get_pop_theta_mod_metrics(late_sessions=late_sessions)
    theta_mod_df = _filter_for_theta_mod_neurons(
        metrics_df, corr_thres=corr_thres, r2_thres=r2_thres, fr_thres=fr_thres
    )
    theta_mod_clusters = theta_mod_df.cluster_unique_ID.values

    ft_df["theta_modulated"] = ft_df.index.get_level_values(1).isin(theta_mod_clusters)

    df = (
        ft_df.reset_index("subject_ID")
        .groupby(["subject_ID", "theta_modulated"])[["distance_to_goal", "place_direction"]]
        .mean()
    )
    df = df.reset_index()

    df_long = df.melt(
        id_vars=["subject_ID", "theta_modulated"],
        value_vars=["distance_to_goal", "place_direction"],
        var_name="measure",
        value_name="value",
    )
    return df_long


def plot_theta_mod_split_tuning(df):
    fig, axes = plt.subplots(1, 2, figsize=(3, 2), sharey=True)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="k", ls="--", alpha=0.5)

    palette = sns.color_palette("hls", n_colors=len(SUBJECT_IDS))
    for (
        ax,
        measure,
    ) in zip(
        axes,
        ["distance_to_goal", "place_direction"],
    ):
        subdf = df[df["measure"] == measure]
        sns.pointplot(
            data=subdf,
            x="theta_modulated",
            y="value",
            hue="subject_ID",
            dodge=False,
            markers="o",
            linestyle="none",
            palette=palette,
            errorbar=None,
            legend=False,
            ax=ax,
        )
        sns.pointplot(
            data=subdf,
            x="theta_modulated",
            y="value",
            dodge=False,
            markers="_",
            linestyle="none",
            errorbar="se",
            color="k",
            ax=ax,
        )
        ax.set_ylabel("prop. neurons")
        ax.set_xlabel(measure)
    return


def plot_feature_rep_by_theta_phase_pref(late_sessions=False, corr_thres=0.45, r2_thres=0.7, fr_thres=1, axes=None):
    """ """
    # set up fig
    if axes is None:
        f, axes = plt.subplots(1, 2, figsize=(4, 2))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="k", ls="--", alpha=0.5)

    # load theta mod and variance explained data
    metrics_df = get_pop_theta_mod_metrics(late_sessions=late_sessions)
    filt_metrics_df = _filter_for_theta_mod_neurons(
        metrics_df, corr_thres=corr_thres, r2_thres=r2_thres, fr_thres=fr_thres
    )
    unique_ve_df = cve.get_cluster_unique_variance_explained(
        late_sessions=late_sessions, full_features=False, add_missing_clusters=True
    )
    theta_mod_clusters = filt_metrics_df.cluster_unique_ID.values
    filt_metrics_df[("unique_ve", "distance_to_goal")] = unique_ve_df.loc[theta_mod_clusters].distance_to_goal.values
    filt_metrics_df[("unique_ve", "place_direction")] = unique_ve_df.loc[theta_mod_clusters].place_direction.values

    # only include clusters that have unique ve data
    filt_metrics_df = filt_metrics_df[~filt_metrics_df.unique_ve.isna().any(axis=1)].copy()

    # bin phase pref
    bins = np.linspace(-np.pi, np.pi, 6)
    filt_metrics_df[("phase_max_bin", "")] = pd.cut(
        filt_metrics_df.loc[:, ("vonmises", "phase_max_rad")], bins=bins, include_lowest=True
    )

    subject_phase_by_ve = (
        filt_metrics_df.groupby(["subject_ID", "phase_max_bin"], observed=True).unique_ve.mean().unique_ve
    )

    # plot variance explained phase preference
    x = filt_metrics_df.vonmises.phase_max_rad
    for ft, ax, color1, color2 in zip(
        ["distance_to_goal", "place_direction"],
        axes,
        ["cornflowerblue", "palevioletred"],
        ["darkblue", "darkred"],
    ):
        # plot background scatter plot
        y = filt_metrics_df[("unique_ve", ft)]
        sns.scatterplot(x=x, y=y, ax=ax, color=color1, alpha=0.5, s=5)

        ve_df = subject_phase_by_ve[ft].unstack()
        _mean = ve_df.mean()
        _sem = ve_df.sem()
        _phase = [b.mid for b in _mean.index]
        ax.errorbar(
            x=_phase,
            y=_mean,
            yerr=_sem,
            marker="o",
            markersize=3,
            linestyle="none",
            color=color2,
            capsize=0,
        )
        # set log scale
        ax.set_ylim(-10, 60)
        ax.set_ylabel(ft)
        ax.set_xlabel("θ Phase")
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    return


# %% define theta-modulated clusters


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


def _filter_for_theta_mod_neurons(metrics_df, corr_thres=0.45, r2_thres=0.7, fr_thres=1):
    df = metrics_df.copy()
    df = df[df.split_half_corr.gt(corr_thres) & df.vonmises.r2.gt(r2_thres) & df.mean_firing_rate.gt(fr_thres)]
    return df


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
