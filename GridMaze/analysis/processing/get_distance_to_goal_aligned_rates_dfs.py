"""This module characterises the distance to goal tuning at the population level."""
# %% Imports
import regex as re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from ..core import load_data

# %% Global variables

MODIFIED_GAMMA_2P_SCALE = 0.75
# %% Main Function


def get_distance_to_goal_aligned_rates_df(
    processed_data_path,
    analysis_data_path,
    distance_metric="geodesic",
    max_distance=1.8,
    bin_width=0.05,
    smoothed=True,
    smooth_SD=1,
    plot_tuning_fits_comparison=False,
):
    """
    TODO: Add docstring
    """
    # load data from disk (keeping on good clusters)
    try:
        session_info = load_data.load(processed_data_path / "session_info.json")
        navigation_df = load_data.load(analysis_data_path / "frames.navigation.parquet")
        navigation_spike_rates_df = load_data.load(analysis_data_path / "frames.spikeRates.parquet")
    except FileNotFoundError:
        print("Missing requisit processed/analysis data to run get_distance_to_goal_aligned_rates_df. Returning None")
        return None
    session_goals = session_info["goals"]
    navigation_rates_df = pd.concat((navigation_df, navigation_spike_rates_df.reset_index(drop=True)), axis=1)
    # bin distance to goal activity during navigation as per input parameters
    bins = pd.interval_range(start=0, end=max_distance, freq=bin_width, closed="left")
    distance_bin_midpoints = [x.mid for x in bins]
    distance_bins_col = ("distance_to_goal", distance_metric + "_bined")
    navigation_rates_df = navigation_rates_df[navigation_rates_df.trial_phase == "navigation"]
    navigation_rates_df = navigation_rates_df[navigation_rates_df.distance_to_goal[distance_metric] < max_distance]
    navigation_rates_df[distance_bins_col] = pd.cut(navigation_rates_df.distance_to_goal[distance_metric], bins=bins)
    # get average distance to goal aligned rates across goals
    distance_to_goal_aligned_rates_df = navigation_rates_df.groupby(["goal", distance_bins_col], observed=True).firing_rate.mean().firing_rate.T
    distance_to_goal_aligned_rates_df.columns = pd.MultiIndex.from_tuples(
        [
            (distance_metric + "_distance_to_goal",) + (goal,) + (range.mid,)
            for goal, range in distance_to_goal_aligned_rates_df.columns.to_numpy()
        ]
    )
    # get average distance to goal aligned rates across all trials
    av_distance_to_goal_aligned_rates_df = (
        navigation_rates_df.groupby(distance_bins_col, observed=True).firing_rate.mean().firing_rate.T
    )
    av_distance_to_goal_aligned_rates_df.columns = pd.MultiIndex.from_tuples(
        [(distance_metric + "_distance_to_goal", "average", i) for i in distance_bin_midpoints]
    )
    # get mean firing rates for each cluster
    mean_firing_rates = navigation_rates_df.firing_rate.mean().to_numpy()
    # get cluster split halves cross correlations
    cluster_cross_correlations = get_distance_to_goal_aligned_rates_cross_correlation(
        navigation_rates_df,
        distance_metric,
        distance_bins_col,
        bins,
        distance_bin_midpoints,
        smoothed,
        smooth_SD,
        session_goals,
    )
    if cluster_cross_correlations is not None:
        cluster_cross_correlations = cluster_cross_correlations.values()
    else:
        cluster_cross_correlations = np.nan
    # get basic session information
    session_info_df = pd.DataFrame(
        index=navigation_rates_df.firing_rate.columns.to_numpy(),
    )
    session_info_df[("subject_ID", "", "")] = session_info["subject_ID"]
    session_info_df[("maze_name", "", "")] = session_info["maze_name"]
    session_info_df[("day_on_maze", "", "")] = session_info["day_on_maze"]

    distance_aligned_rates_df = pd.concat(
        [
            session_info_df,
            distance_to_goal_aligned_rates_df,
            av_distance_to_goal_aligned_rates_df,
        ],
        axis=1,
    )
    distance_aligned_rates_df[("average_firing_rate", "", "")] = mean_firing_rates
    distance_aligned_rates_df[("split_halves_cross_correlation", "", "")] = cluster_cross_correlations
    # ensure multiindex columns
    distance_aligned_rates_df.columns = pd.MultiIndex.from_tuples(distance_aligned_rates_df.columns)
    # get distance tuning curve fits
    fit_dfs = []
    for fitting_func in [
        modified_gamma_2p,
        modified_gamma_4p,
        polynomial_4p,
        polynomial_5p,
    ]:
        fit_df = get_distance_tuning_fits(fitting_func, distance_aligned_rates_df, distance_metric)
        fit_dfs.append(fit_df)
    distance_aligned_rates_df = pd.concat([distance_aligned_rates_df] + fit_dfs, axis=1)
    if plot_tuning_fits_comparison:
        plot_distance_tuning_fits_comparison(distance_aligned_rates_df, distance_metric)
    return distance_aligned_rates_df


def plot_distance_tuning_fits_comparison(
    distance_aligned_rates_df,
    distance_metric,
    cross_corr_threshold=0.5,
    smoothed=True,
    smooth_SD=2,
):
    """"""
    distance_aligned_rates = distance_aligned_rates_df.copy()
    distance_aligned_rates = distance_aligned_rates[
        distance_aligned_rates.split_halves_cross_correlation > cross_corr_threshold
    ]
    distance_to_goal = distance_metric + "_distance_to_goal"
    distance_bin_midpoints = distance_aligned_rates_df[distance_to_goal]["average"].columns.to_list()
    cluster_uniuqe_IDs = distance_aligned_rates_df.index.to_numpy()
    for cluster_unique_ID in cluster_uniuqe_IDs:
        tuning_cuve = (
            distance_aligned_rates_df.loc[cluster_unique_ID][distance_to_goal]["average"].to_numpy().astype(float)
        )
        if smoothed:
            tuning_cuve = gaussian_filter1d(tuning_cuve, sigma=smooth_SD)
        f, ax = plt.subplots(figsize=(5, 5), clear=True)
        ax.plot(distance_bin_midpoints, tuning_cuve, color="k", linewidth=2, label="Data")
        for fit_func in [
            modified_gamma_2p,
            modified_gamma_4p,
            polynomial_4p,
            polynomial_5p,
        ]:
            fit_info = distance_aligned_rates_df.loc[cluster_unique_ID].curve_fit[fit_func.__name__]
            params = fit_info.to_numpy()[:-1]  # exclude r2
            fit = fit_func(distance_bin_midpoints, *params)
            ax.plot(
                distance_bin_midpoints,
                fit,
                label=fit_func.__name__,
                linewidth=1.5,
                alpha=0.8,
            )
        ax.legend()
        ax.set_xlabel("Distance to goal (m)")
        ax.set_ylabel("Firing rate (Hz)")
        ax.set_title(f"Cluster {cluster_unique_ID}")


def get_cluster_unique_ID2cluster_type(navigation_spike_rates_df, cluster_metrics):
    """
    Generate cluster unique ID based on subject and session information and then maps to cluster type
    from KS labels accounting for instances where clusters numbers have been reordered after KS manual
    sorting.
    """
    cluster_unique_IDs = np.array([c for c in navigation_spike_rates_df.firing_rate.columns])
    cluster_IDs = np.array([eval(re.search(r"cluster(\d+)$", c).group(1)) for c in cluster_unique_IDs])
    # HACK to deal with when cluster analysis data has not been reloaded since changing cluster metrics (KS labels)
    unique_cluster_ID2cluster_ID = dict(zip(cluster_unique_IDs, cluster_IDs))
    cluster_ID22KSLabel = cluster_metrics.set_index("cluster_id").to_dict()["KSLabel"]
    cluster_unique_ID2cluster_type = {
        c: cluster_ID22KSLabel[unique_cluster_ID2cluster_ID[c]] for c in cluster_unique_IDs
    }
    return cluster_unique_ID2cluster_type


# %% Cross correlation between distance aligned rates


def get_distance_to_goal_aligned_rates_cross_correlation(
    navigation_rates_df,
    distance_metric,
    distance_bins_col,
    bins,
    distance_bin_midpoints,
    smoothed,
    smooth_SD,
    session_goals,
    plot_split_halves=False,
    n_itter=10,
):
    """
    TODO: Add docstring
    """
    cluster_unique_IDs = navigation_rates_df.firing_rate.columns.to_numpy()
    all_correlations = []
    navigation_rates_df[distance_bins_col] = pd.cut(navigation_rates_df.distance_to_goal[distance_metric], bins=bins)
    for _ in range(n_itter):
        navigation_rates_split_1, navigation_rates_split_2 = split_navigation_rates_df(
            navigation_rates_df, session_goals
        )
        # get distance to goal aligned rates for each split
        distance_aligned_rates_splits = []
        for navigation_rates_split in [
            navigation_rates_split_1.copy(),
            navigation_rates_split_2.copy(),
        ]:
            distance_grouped_rates = navigation_rates_split.groupby([distance_bins_col], observed=True)
            distance_aligned_rates_split = distance_grouped_rates.firing_rate.mean().firing_rate
            if smoothed:
                data = distance_aligned_rates_split.to_numpy()
                smoothed_data = gaussian_filter1d(data, sigma=smooth_SD, axis=0)
                distance_aligned_rates_split = pd.DataFrame(
                    index=distance_aligned_rates_split.index,
                    columns=distance_aligned_rates_split.columns,
                    data=smoothed_data,
                )
            distance_aligned_rates_splits.append(distance_aligned_rates_split)
        (
            distance_aligned_rates_split_1,
            distance_aligned_rates_split_2,
        ) = distance_aligned_rates_splits
        # calculate cross correlation between the two splits
        if distance_aligned_rates_split_1.shape != distance_aligned_rates_split_2.shape:
            print("Different number of distance bins in each split -> skip itteration")
            continue
        cluster_unique_ID2cross_correlation = {}
        for cluster_unique_ID in cluster_unique_IDs:
            if (  # invalid correlation if data contains NaNs (no valid for distance bin)
                np.isnan(distance_aligned_rates_split_1[cluster_unique_ID]).any()
                or np.isnan(distance_aligned_rates_split_2[cluster_unique_ID]).any()
            ):
                cross_corr = np.nan
            elif (  # invalid correlation if data is constant (no variance)
                np.std(distance_aligned_rates_split_1[cluster_unique_ID]) == 0
                or np.std(distance_aligned_rates_split_2[cluster_unique_ID]) == 0
            ):
                cross_corr = np.nan
            else:
                cross_corr = pearsonr(
                    distance_aligned_rates_split_1[cluster_unique_ID],
                    distance_aligned_rates_split_2[cluster_unique_ID],
                )[0]
            cluster_unique_ID2cross_correlation[cluster_unique_ID] = cross_corr
        all_correlations.append(cluster_unique_ID2cross_correlation)
    if len(all_correlations) == 0:
        print("No valid itterations")
        return None
    cluster_unique_ID2av_cross_correlation = {
        col: np.mean([correlation_dict[col] for correlation_dict in all_correlations])
        for col in distance_aligned_rates_split_1.columns
    }
    # plot distance tuning curves for each last data split if desired
    if plot_split_halves:
        plot_split_halves_distance_aligned_rates(
            distance_aligned_rates_split_1,
            distance_aligned_rates_split_2,
            distance_bin_midpoints,
            cluster_unique_ID2av_cross_correlation,
        )
    return cluster_unique_ID2av_cross_correlation


def split_navigation_rates_df(navigation_rates_df, session_goals, balanced_across_goals=True):
    """
    TODO: Add docstring
    """

    def split_trials(navigation_rates_df, trials):
        split_size = len(trials) // 2
        np.random.shuffle(trials)
        split_1 = navigation_rates_df[navigation_rates_df.trial.isin(trials[:split_size])]
        split_2 = navigation_rates_df[navigation_rates_df.trial.isin(trials[split_size:])]
        return split_1, split_2

    goals = navigation_rates_df.goal.unique()

    if balanced_across_goals and len(session_goals) <= navigation_rates_df.trial.max():
        splits_1, splits_2 = [], []
        for goal in goals:
            goal_df = navigation_rates_df[navigation_rates_df.goal == goal]
            split_1, split_2 = split_trials(goal_df, goal_df.trial.unique())
            splits_1.append(split_1)
            splits_2.append(split_2)
        navigation_rates_split_1 = pd.concat(splits_1, axis=0)
        navigation_rates_split_2 = pd.concat(splits_2, axis=0)
    else:
        navigation_rates_split_1, navigation_rates_split_2 = split_trials(
            navigation_rates_df, navigation_rates_df.trial.unique()
        )

    return navigation_rates_split_1, navigation_rates_split_2


def plot_split_halves_distance_aligned_rates(
    distance_aligned_rates_split_1,
    distance_aligned_rates_split_2,
    distance_bin_midpoints,
    cluster_unique_ID2av_cross_correlation,
):
    """
    TODO: Add docstring
    """
    cluster_unique_IDs = distance_aligned_rates_split_1.columns
    for cluster_unique_ID in cluster_unique_IDs:
        corss_correlation = cluster_unique_ID2av_cross_correlation[cluster_unique_ID]
        f, ax = plt.subplots(figsize=(5, 5), clear=True)
        tuning_curve_1 = distance_aligned_rates_split_1[cluster_unique_ID].to_numpy()
        tuning_curve_2 = distance_aligned_rates_split_2[cluster_unique_ID].to_numpy()
        ax.plot(distance_bin_midpoints, tuning_curve_1, label="split 1")
        ax.plot(distance_bin_midpoints, tuning_curve_2, label="split 2")
        ax.set_xlabel("Distance to goal (m)")
        ax.set_ylabel("Firing rate (Hz)")
        ax.text(
            0.5,
            0.9,
            f"Cross correlation: {corss_correlation:.2f}",
            transform=ax.transAxes,
        )
        ax.text(0.5, 1.0, f"Cluster {cluster_unique_ID}", transform=ax.transAxes)


# %% Ctuning curve fitting functions


def get_distance_tuning_fits(
    fitting_function,
    distance_aligned_rates_df,
    distance_metric,
    maxfev=100_000,
    n_itter=10,
    smoothed=True,
    smooth_SD=2,
    plot=False,
):
    """ """
    distance_to_gaol = distance_metric + "_distance_to_goal"
    lower_bounds, upper_bounds = get_fitting_function_bounds(fitting_function)
    param2init_range = get_fitting_function_init_range(fitting_function)
    param_names = list(param2init_range.keys())
    av_distance_to_goal_aligned_rates_df = distance_aligned_rates_df[distance_to_gaol]["average"]
    distance_bin_midpoints = av_distance_to_goal_aligned_rates_df.columns.to_list()
    cluster_unique_IDs = av_distance_to_goal_aligned_rates_df.index.to_numpy()
    if smoothed:
        smoothed_tuning_curves = gaussian_filter1d(
            av_distance_to_goal_aligned_rates_df.values, sigma=smooth_SD, axis=1
        )
        av_distance_to_goal_aligned_rates_df = pd.DataFrame(
            index=cluster_unique_IDs,
            columns=distance_bin_midpoints,
            data=smoothed_tuning_curves,
        )
    tuning_fits = []
    for cluster in cluster_unique_IDs:
        itter_fits = []
        for _ in range(n_itter):
            p0 = [np.random.uniform(*init_range) for param, init_range in param2init_range.items()]
            tuning_curve = av_distance_to_goal_aligned_rates_df.loc[cluster].to_numpy()
            try:
                p_opt, _ = curve_fit(
                    fitting_function,
                    distance_bin_midpoints,
                    tuning_curve,
                    p0=p0,
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=maxfev,
                )
                r2 = goodness_of_fit(fitting_function, distance_bin_midpoints, tuning_curve, p_opt)
            except RuntimeError:
                p_opt = [np.nan] * len(param_names)
                r2 = np.nan
            itter_fits.append({param: p_opt[i] for i, param in enumerate(param_names)})
            itter_fits[-1]["r2"] = r2
        best_fit = max(itter_fits, key=lambda x: x["r2"])
        tuning_fits.append(best_fit)
        if plot:
            plot_distance_tuning_fits(fitting_function, best_fit, distance_bin_midpoints, tuning_curve)
    tuning_fits_df = pd.DataFrame(index=cluster_unique_IDs, data=tuning_fits)
    columns = pd.MultiIndex.from_tuples(
        [("tuning_curve_fit",) + (fitting_function.__name__,) + (param,) for param in param_names + ["r2"]]
    )
    tuning_fits_df.columns = columns
    return tuning_fits_df


def plot_distance_tuning_fits(fitting_function, best_fit, distance_bin_midpoints, tuning_curve):
    """ """
    params = list(best_fit.values())[:-1]
    r2 = list(best_fit.values())[-1]
    f, ax = plt.subplots(figsize=(5, 5), clear=True)
    ax.plot(distance_bin_midpoints, tuning_curve, "k", label="Tuning curve")
    ax.plot(
        distance_bin_midpoints,
        fitting_function(distance_bin_midpoints, *params),
        "r",
        label="Gamma fit",
    )
    ax.legend()
    ax.text(0.5, 0.9, f"R2 = {r2:.2f}", transform=ax.transAxes)
    return


def modified_gamma_2p(x, size, shape):
    return size * gamma.pdf(list(x), shape, loc=0, scale=MODIFIED_GAMMA_2P_SCALE)


def modified_gamma_4p(x, size, shape, scale, shift):
    return size * gamma.pdf(list(x), shape, loc=0, scale=scale) + shift


def polynomial_3p(x, a, b, c):
    x = np.array(x)
    return a * x**2 + b * x + c


def polynomial_4p(x, a, b, c, d):
    x = np.array(x)
    return a * x**3 + b * x**2 + c * x + d


def polynomial_5p(x, a, b, c, d, e):
    x = np.array(x)
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


def get_fitting_function_bounds(fitting_function):
    """ """
    fit_func2bounds = {
        modified_gamma_2p: ([0] * 2, [np.inf] * 2),
        modified_gamma_4p: ([-np.inf, 0, 0, -50], [np.inf, np.inf, np.inf, 50]),
        polynomial_3p: ([-np.inf] * 3, [np.inf] * 3),
        polynomial_4p: ([-np.inf] * 4, [np.inf] * 4),
        polynomial_5p: ([-np.inf] * 5, [np.inf] * 5),
    }
    try:
        lower_bounds, upper_bounds = fit_func2bounds[fitting_function]
        return lower_bounds, upper_bounds
    except KeyError:
        raise ValueError(f"No bounds available for function: {fitting_function.__name__}")


def get_fitting_function_init_range(fitting_function):
    """Note bounds on shift parameter necessary to prevent fitting run aways"""
    fit_func2innit_range = {
        modified_gamma_2p: {"size": [0, 1], "shape": [0, 1]},
        modified_gamma_4p: {
            "size": [-1, 1],
            "shape": [0, 3],
            "scale": [0, 1],
            "shift": [0, 3],
        },
        polynomial_3p: {"a": [-1, 1], "b": [-1, 1], "c": [-1, 1]},
        polynomial_4p: {"a": [-1, 1], "b": [-1, 1], "c": [-1, 1], "d": [-1, 1]},
        polynomial_5p: {
            "a": [-1, 1],
            "b": [-1, 1],
            "c": [-1, 1],
            "d": [-1, 1],
            "e": [-1, 1],
        },
    }
    try:
        init_range = fit_func2innit_range[fitting_function]
        return init_range
    except KeyError:
        raise ValueError(f"No init range available for function: {fitting_function.__name__}")


def goodness_of_fit(fitting_function, x, data, p_opt):
    predicted = fitting_function(x, *p_opt)
    residuals = data - predicted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data - np.mean(data)) ** 2)
    if ss_tot == 0:
        return np.nan
    r2 = 1 - (ss_res / ss_tot)
    if not 0 <= r2 <= 1:
        return 0
    else:
        return r2
