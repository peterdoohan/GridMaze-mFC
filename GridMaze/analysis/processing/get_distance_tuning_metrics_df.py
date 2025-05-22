"""
Updated lib for calculating distance to goal tuning parameters (define basis fits etc.)
and saving out results to dataframe (clusters.DistanceTuningMetrics.parquet)
"""

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from GridMaze.analysis.core import load_data
from GridMaze.analysis.core import convert
from GridMaze.analysis.cluster_tuning import distance_to_goal as dtg

from scipy.stats import ttest_1samp, zscore
from scipy.stats import gamma, norm
from scipy.optimize import curve_fit

from sklearn.metrics import r2_score


# %% Global Variables

GAMMA_2P_SCALE = 0.75

GAUSSIAN_2P_SCALE = 1

# %% Functions


def get_distance_tuning_metrics_df(
    processed_data_path,
    analysis_data_path,
    distance_metrics=("distance_to_goal", "geodesic"),
    max_steps_to_goal=30,
    moving_only=False,
    bin_spacing=0.05,
    alpha=0.01,
):
    """ """
    # load data
    session_info = load_data.load(processed_data_path / "session_info.json")
    cluster_metrics = load_data.load(processed_data_path / "clusters.metrics.htsv")
    navigation_df = load_data.load(analysis_data_path / "frames.navigation.parquet")
    navigation_spike_rates_df = load_data.load(analysis_data_path / "frames.spikeRates.parquet")
    navigation_spike_rates_df.reset_index(drop=True, inplace=True)
    cluster_unique_IDs = navigation_spike_rates_df.firing_rate.columns.to_numpy()

    # get single units
    single_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID.values
    single_units = convert.cluster_IDs2scluster_unique_IDs(session_info, single_units)
    # isolate relevant navigation columns
    distance_info = navigation_df[
        [("goal", ""), ("trial", ""), ("moving", ""), ("steps_to_goal", "future"), distance_metrics]
    ].droplevel(1, axis=1)
    curve_fit_fns = [gamma_2p, gamma_4p, gaussian_2p, gaussian_4p, polynomial_4p]
    cols = [
        ("single_unit", ""),
        ("distance_tuned", ""),
        ("split_half_corr", "value"),
        ("split_half_corr", "pvalue"),
    ] + [(fn.__name__, x) for fn in curve_fit_fns for x in _get_param_names(fn)]
    metrics_df = pd.DataFrame(index=cluster_unique_IDs, columns=pd.MultiIndex.from_tuples(cols), data=np.nan)
    # fix boolian dtype cols
    for col in [("single_unit", ""), ("distance_tuned", "")]:
        metrics_df[col] = metrics_df[col].astype(bool)
        metrics_df[col] = False  # false by default
    for cluster in single_units:
        metrics_df.loc[cluster, ("single_unit", "")] = True
        cluster_rates = navigation_spike_rates_df.xs(cluster, level=1, axis=1)
        distance_rates_df = pd.concat([distance_info, cluster_rates], axis=1)
        distance_tuning_df = dtg.get_distance_to_goal_tuning_df(
            distance_rates_df,
            metrics=distance_metrics,
            bin_spacing=bin_spacing,
            max_steps_to_goal=max_steps_to_goal,
            moving_only=moving_only,
        )
        mean_corr, p_val, sig = _get_distance_tuning_metrics(distance_tuning_df, n_reps=50, alpha=alpha)
        metrics_df.loc[cluster, ("split_half_corr", "value")] = mean_corr
        metrics_df.loc[cluster, ("split_half_corr", "pvalue")] = p_val
        metrics_df.loc[cluster, ("distance_tuned", "")] = sig
        if sig:
            tc = distance_tuning_df.distance.mean()
            for fit_fn in curve_fit_fns:
                params = tuning_curve_fit(tc, fit_fn)
                for param, value in params.items():
                    fn_name = fit_fn.__name__
                    metrics_df.loc[cluster, (fn_name, param)] = value
    # return
    metrics_df.reset_index(inplace=True)
    metrics_df.rename(columns={"index": "cluster_unique_ID"}, inplace=True)
    return metrics_df.sort_index(axis=1)


def _get_distance_tuning_metrics(distance_tuning_df, n_reps=50, alpha=0.01):
    """ """
    trials = distance_tuning_df.trial.unique()
    mid = len(trials) // 2
    corrs = []
    for _ in range(n_reps):
        trials_shuffled = np.random.permutation(trials)
        split_1 = distance_tuning_df[distance_tuning_df.trial.isin(trials_shuffled[:mid])]
        split_2 = distance_tuning_df[distance_tuning_df.trial.isin(trials_shuffled[mid:])]
        curve_1 = split_1.distance.mean()
        curve_2 = split_2.distance.mean()
        corrs.append(curve_1.corr(curve_2, method="spearman"))
    result = ttest_1samp(corrs, 0, alternative="greater")
    p_val = result.pvalue
    mean_corr = np.mean(corrs)
    sig = True if p_val < alpha else False
    return mean_corr, p_val, sig


# %% Curve fitting functions


def tuning_curve_fit(
    tuning_curve,
    fn,
    n_itter=10,
    plot=False,
    verbose=False,
):
    """ """
    init_range, bounds, param_names = _get_init_range(fn), _get_bounds(fn), _get_param_names(fn)
    x = tuning_curve.index.values.astype(float)
    y = tuning_curve.values.astype(float)
    itter_fits = []
    for i in range(n_itter):
        if verbose:
            print(i)
        p0 = [np.random.uniform(*x) for x in init_range]
        try:
            p_opt, _ = curve_fit(
                fn,
                x,
                y,
                p0=p0,
                bounds=bounds,
                maxfev=10_000,
            )
            r2 = r2_score(y, fn(x, *p_opt))
        except RuntimeError:
            p_opt = [np.nan] * len(param_names)
            r2 = np.nan
        itter_fits.append({param: p_opt[i] for i, param in enumerate(param_names)})
        itter_fits[-1]["r2"] = r2
    best_fit = max(itter_fits, key=lambda x: x["r2"])
    if plot:
        f, ax = plt.subplots(1, 1, figsize=(2, 2))
        plot_params = list(best_fit.values())[:-1]
        plot_r2 = list(best_fit.values())[-1]
        ax.plot(x, y, label="data")
        ax.plot(x, fn(x, *plot_params), label="fit")
        ax.set_title(f"r2: {plot_r2:.2f}")
        ax.legend()
    return best_fit


def _get_init_range(fn):
    """ """
    fn_name = fn.__name__
    if fn_name == "gamma_2p":
        p0 = [[0, 1], [0, 1]]  # size, shape
    elif fn_name == "gamma_4p":
        p0 = [[-1, 1], [0, 3], [0, 1], [0, 3]]  # size, shape, scale, shift
    elif fn_name == "gaussian_2p":
        p0 = [[0, 5], [0, 2]]  # amplitude, mean
    elif fn_name == "gaussian_4p":
        p0 = [[-5, 5], [0, 2], [0, 1], [0, 3]]  # amplitude, mean, stddev, offset
    elif fn_name == "polynomial_4p":
        p0 = [[-1, 1], [-1, 1], [-1, 1], [-1, 1]]  # a, b, c, d
    else:
        raise ValueError(f"Unknown function: {fn_name}")
    return p0


def _get_bounds(fn):
    """ """
    fn_name = fn.__name__
    if fn_name == "gamma_2p":
        bounds = [[0, 0], [np.inf, np.inf]]
    elif fn_name == "gamma_4p":
        bounds = [[-np.inf, 0, 0, -50], [np.inf, np.inf, np.inf, 50]]
    elif fn_name == "gaussian_2p":
        bounds = [(-np.inf, -np.inf), (np.inf, np.inf)]
    elif fn_name == "gaussian_4p":
        bounds = [[-np.inf, 0, 0, -50], [np.inf, np.inf, np.inf, 50]]
    elif fn_name == "polynomial_4p":
        bounds = [[-np.inf] * 4, [np.inf] * 4]
    else:
        raise ValueError(f"Unknown function: {fn_name}")
    return bounds


def _get_param_names(fn):
    """ """
    fn_name = fn.__name__
    if fn_name == "gamma_2p":
        param_names = ["size", "shape"]
    elif fn_name == "gamma_4p":
        param_names = ["size", "shape", "scale", "shift"]
    elif fn_name == "gaussian_2p":
        param_names = ["amplitude", "mean"]
    elif fn_name == "gaussian_4p":
        param_names = ["amplitude", "mean", "stddev", "offset"]
    elif fn_name == "polynomial_4p":
        param_names = ["a", "b", "c", "d"]
    else:
        raise ValueError(f"Unknown function: {fn_name}")
    return param_names


# %% Tuning curve function


def gamma_2p(x, size, shape):
    x_arr = np.asarray(x)
    return size * gamma.pdf(x_arr, shape, loc=0, scale=GAMMA_2P_SCALE)


def gamma_4p(x, size, shape, scale, shift):
    x_arr = np.asarray(x)
    return size * gamma.pdf(x_arr, shape, loc=0, scale=scale) + shift


def gaussian_2p(x, amplitude, mean):
    x_arr = np.asarray(x)
    return amplitude * norm.pdf(x_arr, loc=mean, scale=GAUSSIAN_2P_SCALE)


def gaussian_4p(x, amplitude, mean, stddev, offset):
    x_arr = np.asarray(x)
    return amplitude * norm.pdf(x_arr, loc=mean, scale=stddev) + offset


def polynomial_4p(x, a, b, c, d):
    x = np.array(x)
    return a * x**3 + b * x**2 + c * x + d
