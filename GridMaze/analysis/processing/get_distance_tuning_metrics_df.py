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
from scipy.optimize import curve_fit, minimize

from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


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
    bin_spacing=0.1,
    alpha=0.05,
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
    curve_fit_cv_fns = [gamma_4p]
    cols = (
        [("single_unit", ""), ("distance_tuned", ""), ("split_half_corr", "value"), ("split_half_corr", "pvalue")]
        + [(fn.__name__, x) for fn in curve_fit_fns for x in _get_param_names(fn)]
        + [(fn.__name__ + "_cv", x) for fn in curve_fit_cv_fns for x in _get_param_names(fn) + ["p_value", "sig"]]
    )
    metrics_df = pd.DataFrame(index=cluster_unique_IDs, columns=pd.MultiIndex.from_tuples(cols), data=np.nan)
    # fix boolian dtype cols
    for col in [("single_unit", ""), ("distance_tuned", "")]:
        metrics_df[col] = metrics_df[col].astype(bool)
        metrics_df[col] = False  # false by default
    for fn in curve_fit_cv_fns:
        col = (fn.__name__ + "_cv", "sig")
        metrics_df[col] = metrics_df[col].astype(bool)
    # loop through single unit, calculate distance tuning metrics
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
            tuning_df = distance_tuning_df.distance
            for fit_fn in curve_fit_cv_fns:  # curve_fit_fns:
                params = tuning_curve_fit_cv(tuning_df, fit_fn, plot=False)
                fn_name = fit_fn.__name__
                for param, value in params.items():
                    metrics_df.loc[cluster, (fn_name + "_cv", param)] = value
            tc = tuning_df.mean()
            for fit_fn in curve_fit_fns:
                params = tuning_curve_fit(tc, fit_fn, plot=False)
                fn_name = fit_fn.__name__
                for param, value in params.items():
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


def tuning_curve_fit_cv(tuning_df, fn, n_splits=30, n_inits=5, alpha=0.05, plot=False, verbose=False):
    """
    Calculates distance tuning curves from split halfs of the data (split over trials).
    Find fn fit params from the first half of the data and calculate r2 on the second half, over
    n_splits.

    Note if size param can be both negative and positive across split halve fits.
    If the number of positive and negative fits are equal, the one with the higher r2 is taken (+/- 1).
    If the number of positive and negative fits are not equal, the most common is taken.
    """
    assert fn.__name__ in ["gamma_4p", "gaussian_4p"], "Only gamma_4p and gaussian_4p are supported for CV fitting."
    init_range, bounds, param_names = _get_init_range(fn), _get_bounds(fn), _get_param_names(fn)
    x = tuning_df.columns.values.astype(float)
    idx = tuning_df.index.values
    mid = len(idx) // 2
    split_fits = []
    for i in range(n_splits):
        if verbose:
            print(f"split {i}")
        idx_shuffled = np.random.permutation(idx)
        y_1 = tuning_df.loc[idx_shuffled[:mid]].mean().values
        y_2 = tuning_df.loc[idx_shuffled[mid:]].mean().values
        nan_mask = np.isnan(y_1) | np.isnan(y_2)
        y_1 = y_1[~nan_mask]
        y_2 = y_2[~nan_mask]
        _x = x[~nan_mask]
        itter_fits = []
        for j in range(n_inits):
            if verbose:
                print(f"innit {j}")
            p0 = [np.random.uniform(*x) for x in init_range]
            try:
                p_opt, _ = curve_fit(
                    fn,
                    _x,
                    y_1,
                    p0=p0,
                    bounds=bounds,
                    maxfev=10_000,
                )
                r2 = r2_score(y_2, fn(_x, *p_opt))
            except RuntimeError:
                p_opt = [np.nan] * len(param_names)
                r2 = np.nan
            itter_fits.append({param: p_opt[i] for i, param in enumerate(param_names)})
            itter_fits[-1]["r2"] = r2
        best_fit = max(itter_fits, key=lambda x: x["r2"])
        if verbose:
            print(f"best fit: {best_fit}")
        split_fits.append(best_fit)
    splits_df = pd.DataFrame(split_fits)
    # logic: if number of pos and neg fits are equal +/- 1 take the one with the higher r2 (usually bad fit)
    pos_fits = splits_df[splits_df["size"] > 0]
    neg_fits = splits_df[splits_df["size"] < 0]
    n_pos, n_neg = len(pos_fits), len(neg_fits)
    if n_pos in list(range(n_neg - 1, n_neg + 2)):
        pos_r2 = pos_fits["r2"].mean()
        neg_r2 = neg_fits["r2"].mean()
        if pos_r2 > neg_r2:
            mean_params = pos_fits.median().to_dict()
        else:
            mean_params = neg_fits.median().to_dict()
    # logic: if number of pos and neg fits are not equal (eg outlier), take the most common
    else:
        if n_pos > n_neg:
            mean_params = pos_fits.median().to_dict()
        else:
            mean_params = neg_fits.median().to_dict()
    result = ttest_1samp(splits_df["r2"].values, 0, alternative="greater")  # test from all fits (if size pos or neg)
    mean_params["p_value"] = result.pvalue
    mean_params["sig"] = True if mean_params["p_value"] < alpha else False
    if plot:
        f, ax = plt.subplots(1, 1, figsize=(2, 2))
        plot_params = list(mean_params.values())[:-3]
        y_all = tuning_df.mean().values
        ax.plot(_x, y_all, label="data")
        ax.plot(_x, fn(x, *plot_params), label="fit")
        ax.text(
            0.5,
            -0.2,
            ", ".join([f"{p}:{v:.2g}" for p, v in mean_params.items()]),
            transform=ax.transAxes,
            fontsize=6,
            ha="center",
        )
        ax.legend()
    return mean_params


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
    nan_mask = np.isnan(y)
    x = x[~nan_mask]
    y = y[~nan_mask]
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
        ax.text(
            0.5, -0.2, ", ".join([f"{p:.2g}" for p in plot_params]), transform=ax.transAxes, fontsize=8, ha="center"
        )
        ax.legend()
    return best_fit


def _get_init_range(fn):
    """ """
    fn_name = fn.__name__
    if fn_name == "gamma_2p":
        p0 = [[-5, 5], [0, 1]]  # size, shape
    elif fn_name == "gamma_4p":
        p0 = [[-5, 5], [0.1, 10], [0.1, 1], [0, 3]]  # size, shape, scale, shift
    elif fn_name == "gaussian_2p":
        p0 = [[-5, 5], [0, 2]]  # amplitude, mean
    elif fn_name == "gaussian_4p":
        p0 = [[-5, 5], [0, 2], [0.05, 1], [0, 3]]  # amplitude, mean, stddev, offset
    elif fn_name == "polynomial_4p":
        p0 = [[-1, 1], [-1, 1], [-1, 1], [-1, 1]]  # a, b, c, d
    else:
        raise ValueError(f"Unknown function: {fn_name}")
    return p0


def _get_bounds(fn):
    """ """
    fn_name = fn.__name__
    if fn_name == "gamma_2p":
        bounds = [[-np.inf, 0], [np.inf, np.inf]]  # size, shape
    elif fn_name == "gamma_4p":
        bounds = [[-np.inf, 0.05, 0.05, -100], [np.inf, 20, 2, 100]]  # size, shape, scale, shift
    elif fn_name == "gaussian_2p":
        bounds = [[-np.inf, -np.inf], [np.inf, np.inf]]
    elif fn_name == "gaussian_4p":
        bounds = [[-np.inf, 0, 0.05, -50], [np.inf, np.inf, np.inf, 50]]
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
