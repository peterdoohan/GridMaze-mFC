"""
Library for decoding distance to goal from neural activity as input either spikes or spikes + LFP osscilations like
theta (8-12Hz) or high_delta (2-5Hz) to see if they improve performance.
"""

# %% Imports
from turtle import color
import numpy as np
import pandas as pd
from scipy.stats import gamma, poisson
from scipy.optimize import minimize
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import seaborn as sns

from GridMaze.analysis.core import convert
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import folds
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.distance_to_goal import extract_lfp_phase as elp
from GridMaze.analysis.distance_to_goal import population_tuning as pt
from GridMaze.analysis.distance_to_goal import distributions as dd
from GridMaze.analysis.processing import get_distance_tuning_metrics_df as dtm
from GridMaze.analysis.cluster_tuning import distance_to_goal as dtg

# %% Global Variables

FRAME_RATE = 60
# %% Functions


# %%


def quick_plot2(results_df, osc="theta", metric=("distance_to_goal", "geodesic"), distance_range=None):
    """ """
    if distance_range is None:
        df = results_df.copy()
    else:
        df = results_df[results_df[metric].between(*distance_range)].copy()
    df.loc[:, ("decoding_error", "")] = df[metric] - df.distance_to_goal.decoded
    grouped_df = df.groupby([("lfp_phase_bin", osc)]).decoding_error
    mean_df = grouped_df.mean()
    sem_df = grouped_df.sem()
    f, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.axhline(0, color="k", alpha=0.5, ls="--")
    ax.plot(mean_df.index.values, mean_df.values, color="k", lw=2)
    ax.fill_between(
        mean_df.index.values,
        mean_df.values - sem_df.values,
        mean_df.values + sem_df.values,
        color="k",
        alpha=0.2,
    )
    ax.set_xlabel(f"{osc} phase")
    ax.set_ylabel("Decoding error (m)")
    return


def quick_plot(results_df, metric=("distance_to_goal", "geodesic"), bin_spacing=0.1):
    """ """
    results_df.loc[:, ("decoding_error", "")] = results_df[metric] - results_df.distance_to_goal.decoded
    max_distance = dd.get_distance_percentile(metric, 0.85)
    n_bins = int(max_distance / bin_spacing)
    results_df = results_df[results_df[metric] < max_distance]
    bins = convert._get_distance_bins(
        binning_method="uniform",
        n_distance_bins=n_bins,
        distance_metrics=metric,
        max_distance=max_distance,
    )
    distance_bins = pd.cut(results_df[metric], bins=bins, include_lowest=True)
    results_df.loc[:, ("distance_bin", "")] = distance_bins.apply(lambda x: x.mid).astype(float)
    trial_summary_df = results_df.groupby(["trial", "distance_bin"]).decoding_error.mean()
    grouped_df = trial_summary_df.groupby("distance_bin")
    mean_df = grouped_df.mean()
    sem_df = grouped_df.sem()
    f, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.axhline(0, color="k", alpha=0.5, ls="--")
    ax.plot(mean_df.index.values, mean_df.values, color="k", lw=2)
    ax.fill_between(
        mean_df.index.values,
        mean_df.values - sem_df.values,
        mean_df.values + sem_df.values,
        color="k",
        alpha=0.2,
    )
    ax.set_xlabel("Distance to goal")
    ax.set_ylabel("Decoding error (m)")
    ax.set_ylim(-0.25, 1)
    return


# %%


def plot_true_vs_decoded_distance(results_df, ax=None):
    """ """
    df = results_df.distance_to_goal
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3))
    # sns.kdeplot(data=df, x="geodesic", y="decoded", fill=True, thresh=0, levels=50, cmap="Blues", ax=ax, cbar=True)
    sns.histplot(data=df, x="geodesic", y="decoded", bins=30, cmap="Blues", ax=ax, cbar=True, pmax=0.5)
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, 1.6)
    ax.plot([0, 1.6], [0, 1.6], color="k", lw=1, ls="--")
    ax.set_xlabel("true distance (m)")
    ax.set_ylabel("decoded distance (m)")

    return


def test(subject="m2"):
    """ """
    sessions = gs.get_maze_sessions(
        subject_IDs=[subject],
        maze_names="all",
        days_on_maze="late",
        with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "trials_df"],
    )
    decoding_results = []
    for session in sessions:
        print(session.name)
        decoding_results.append(decode_session_distance_to_goal(session, resolution=0.5, with_LFP=False, plot=False))
    all_decoding_results = pd.concat(decoding_results, axis=0)
    return all_decoding_results


def decode_session_distance_to_goal(
    session,
    resolution=0.5,
    metric=("distance_to_goal", "geodesic"),
    n_folds=10,
    bin_spacing=0.1,
    min_r2=0.5,
    with_LFP=True,
    n_lfp_phase_bins=12,
    test_trial_phases=["navigation"],
    verbose=True,
    plot=True,
):
    """ """
    max_distance = dd.get_distance_percentile(metric, 0.85)
    input_data = get_input_data(session, metric, resolution, with_LFP=with_LFP, n_lfp_phase_bins=n_lfp_phase_bins)
    distance_tuning_df = get_distance_tuning_df(input_data, metric, resolution, bin_spacing)
    empirical_tuning_curves = distance_tuning_df.groupby("distance_bin").mean().spike_count
    distances = empirical_tuning_curves.index.values
    x_bounds = (min(distances), max(distances))
    folds_df = folds.get_folds_df(session, goal_stratified=False, return_unique_IDs=False, n_folds=n_folds)
    _folds = folds_df.columns.get_level_values(0).unique()
    params_df = get_fold_params_df(distance_tuning_df, folds_df, plot=True)
    return
    # remove non distance tuned clusters
    if min_r2 is not None:
        mean_r2 = params_df.groupby("cluster_unique_ID").r2.mean()  # r2 across folds
        reject_clusters = mean_r2[mean_r2 <= min_r2].index
        if verbose:
            print(f"removing {len(reject_clusters)} clusters with mean r2 <= {min_r2}")
        params_df = params_df[~params_df.cluster_unique_ID.isin(reject_clusters)]
        input_data.drop(columns=[("spike_count", c) for c in reject_clusters], inplace=True)
    if verbose:
        print(f"decoding distance to goal across folds in parallel")
    results_df = Parallel(n_jobs=len(_folds))(
        delayed(_decode_fold)(
            input_data,
            folds_df,
            fold,
            test_trial_phases,
            metric,
            max_distance,
            params_df,
            x_bounds,
            resolution,
        )
        for fold in _folds
    )
    results_df = pd.concat(results_df, axis=0).sort_index()
    if plot:
        quick_plot(results_df, metric, bin_spacing)
        quick_plot2(results_df, osc="theta", metric=metric)
        quick_plot2(results_df, osc="4Hz", metric=metric)
    return results_df


def _decode_fold(input_data, folds_df, fold, test_trial_phases, metric, max_distance, params_df, x_bounds, resolution):
    """ """
    fold_df = folds_df[fold]
    test_trials = fold_df["test"].unstack().dropna().values
    test_df = input_data[
        input_data.trial.isin(test_trials)
        & input_data.trial_phase.isin(test_trial_phases)
        & input_data[metric].lt(max_distance)
    ]
    fold_params = params_df[params_df.fold == fold][
        ["size", "shape", "scale", "shift"]
    ].values.T  # n_params, n_clusters
    observed_spikes = test_df.spike_count.values  # n_samples, n_clusters
    d_hats = []
    for i in range(observed_spikes.shape[0]):
        d_hat = decode_distance(
            fold_params,
            observed_spikes[i],
            tuning_curves=None,
            x_bounds=x_bounds,
            resolution=resolution,
            p0_set=[0.1, 0.3, 0.8, 1.4],
            max_distance=max_distance,
        )
        d_hats.append(d_hat)
    results_df = test_df.copy()
    results_df.drop(columns=["spike_count"], inplace=True, level=0)
    results_df.loc[:, ("distance_to_goal", "decoded")] = d_hats
    return results_df


def decode_distance(
    params,
    observed_spikes,
    tuning_curves=None,
    x_bounds=None,
    resolution=0.1,
    p0_set=[0.1, 0.3, 0.8, 1.4],
    max_distance=1.5,
):
    d_opts = []
    neg_log_likelihoods = []
    for p0 in p0_set:
        result = minimize(
            get_spikes_neg_log_likelihood,
            p0,
            bounds=((0, max_distance),),
            args=(observed_spikes, params, tuning_curves, x_bounds, resolution),
            method="L-BFGS-B",
        )
        d_opts.append(result.x[0])
        neg_log_likelihoods.append(result.fun)
    min_index = np.argmin(neg_log_likelihoods)
    d_opt = d_opts[min_index]
    return d_opt


def get_spikes_neg_log_likelihood(
    dist,
    observed_spikes,
    params=None,
    tuning_curves=None,
    x_bounds=None,
    resolution=0.4,
    eps=1e-10,
):
    """ """
    if params is not None:
        # estimate spike rate from params
        assert x_bounds is not None, "x_bounds must be provided if params are given"
        spike_rate = bounded_gamma_4p(dist, *params, x_bounds=x_bounds)
    elif tuning_curves is not None:
        # estimate spike rate from empirical tuning curve
        distances = tuning_curves.index.values
        nearest_dist = np.argmin(np.abs(distances - dist))
        spike_rate = tuning_curves.loc[nearest_dist]
    # get expected spikes (ensuring they are positive)
    expected_spikes = spike_rate * resolution
    expected_spikes[expected_spikes < 0] = 0
    p = poisson.pmf(observed_spikes, expected_spikes)
    p[p < eps] = eps  # avoid log(0) = -inf
    return -np.sum(np.log(p))


def bounded_gamma_4p(x, size, shape, scale, shift, x_bounds=None):
    """ """
    x_arr = np.asarray(x)
    if x_bounds is not None:
        min_x, max_x = x_bounds
        if np.any(x_arr < min_x):
            x_arr = np.where(x_arr > min_x, x_arr, min_x)
        if np.any(x_arr > max_x):
            x_arr = np.where(x_arr < max_x, x_arr, max_x)
    return size * gamma.pdf(x_arr, shape, loc=0, scale=scale) + shift


# %% Distance tuning functions


def get_fold_params_df(distance_tuning_df, folds_df, plot=False):
    """ """
    _folds = folds_df.columns.get_level_values(0).unique()
    _all_trials = distance_tuning_df.index.get_level_values(0).unique().values

    def _get_fold_params(distance_tuning_df, folds_df, fold):
        fold_df = folds_df[fold]
        train_trials = fold_df["train"].unstack().dropna().values
        train_trials = np.intersect1d(train_trials, _all_trials)  # ensure trials are in distance_tuning_df
        tuning_curves = distance_tuning_df.loc[train_trials].groupby("distance_bin").mean().spike_count
        return get_tuning_params(tuning_curves, fold)

    fold_params = Parallel(n_jobs=len(_folds))(
        delayed(_get_fold_params)(distance_tuning_df, folds_df, fold) for fold in _folds
    )
    params_df = pd.concat(fold_params, ignore_index=True)

    if plot:
        tuning_curves = distance_tuning_df.groupby("distance_bin").mean().spike_count
        cluster_unique_IDs = tuning_curves.columns.values
        distances = tuning_curves.index.values
        x = np.linspace(0, max(distances), 100)
        _params_df = params_df.set_index(["cluster_unique_ID", "fold"])
        for cluster in cluster_unique_IDs:
            tc = tuning_curves[cluster]
            f, ax = plt.subplots(1, 1, figsize=(2, 2))
            ax.plot(distances, tc.values, color="k", lw=2)
            for fold in _folds:
                params = _params_df.loc[(cluster, fold)]
                _params = [params.loc[p] for p in ["size", "shape", "scale", "shift"]]
                min_dist, max_dist = min(distances), max(distances)
                ax.plot(x, bounded_gamma_4p(x, *_params, x_bounds=(min_dist, max_dist)), alpha=0.5, lw=1)
            ax.set_title(f"R2 = {_params_df.loc[cluster]["r2"].mean():.2f}")
    return params_df


def get_tuning_params(tuning_curves, fold):
    """ """
    cluster_unique_IDs = tuning_curves.columns.values
    fits = []
    for cluster in cluster_unique_IDs:
        tc = tuning_curves[cluster]
        opt_params = dtm.tuning_curve_fit(tc, dtm.gamma_4p, plot=False)
        fits.append(opt_params)
    params_df = pd.DataFrame(fits)
    params_df["cluster_unique_ID"] = cluster_unique_IDs
    params_df["fold"] = fold
    return params_df


def get_distance_tuning_df(
    input_data,
    metric=("distance_to_goal", "geodesic"),
    resolution=0.4,
    bin_spacing=0.1,
    moving_only=False,
    max_steps_to_goal=20,
):
    """
    Need new version of this funcion that operates with spikes not rates.
    """
    df = input_data[input_data.trial_phase == "navigation"].copy()
    if moving_only:
        df = df[df.moving]
    if max_steps_to_goal is not None:
        df = df[df.steps_to_goal.future < max_steps_to_goal]
    if metric[0] == "distance_to_goal":
        max_distance = dd.get_distance_percentile(metric, 0.85)
        n_bins = int(max_distance / bin_spacing)
        df = df[df[metric] < max_distance]
        bins = convert._get_distance_bins(
            binning_method="uniform",
            n_distance_bins=n_bins,
            distance_metrics=metric,
            max_distance=max_distance,
        )
    else:
        NotImplementedError()
        # bin distances
    distance_bins = pd.cut(df[metric], bins=bins, include_lowest=True)
    df.loc[:, ("distance_bin", "")] = distance_bins.apply(lambda x: x.mid).astype(float)
    trial_dist_grouped_df = df.groupby(["trial", "distance_bin"], observed=True)
    # counts spikes in each trial x distance bin
    trial_spike_counts = trial_dist_grouped_df.spike_count.sum()
    step_counts = trial_dist_grouped_df.time.count()
    # get seconds occupied by each trial x distance bin
    trial_occupancy = step_counts / resolution
    trial_occupancy.replace(0, np.nan, inplace=True)  # unvisited distances in given trial
    trial_spike_rates = trial_spike_counts.div(trial_occupancy, axis=0)
    return trial_spike_rates


# %% Input data functions


def get_input_data(
    session,
    metric=("distance_to_goal", "geodesic"),
    resolution=0.4,
    include_multiunits=False,
    with_LFP=True,
    n_lfp_phase_bins=12,
):
    """ """
    # load data
    navigation_df = session.navigation_df
    spike_counts_df = session.navigation_spike_counts_df.reset_index(drop=True)
    cluster_metrics = session.cluster_metrics
    session_info = session.session_info
    # filter for single units
    if not include_multiunits:
        single_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID
        single_units = convert.cluster_IDs2scluster_unique_IDs(session_info, single_units)
        spike_counts_df = spike_counts_df[[("spike_count", u) for u in single_units]]
    # downsample to specified resolution with sliding window
    ds_nav_df, ds_spike_counts_df = ds.downsample_nav_spikes_data(
        navigation_df,
        spike_counts_df,
        resolution=resolution,
        distance_metrics=[("steps_to_goal", "future"), metric],
    )
    input_df = pd.concat([ds_nav_df, ds_spike_counts_df], axis=1)
    # filter for valid trial times
    input_df = input_df[~input_df.trial_phase.isna()]
    # add LFP phases
    if with_LFP:
        input_df.reset_index(drop=True, inplace=True)
        times = input_df.time.values
        for osc in ["theta", "4Hz"]:
            phase_bins = elp.get_nearest_osc_phase(
                session,
                times,
                signal_type="LFP",
                band=osc,
                return_binned=True,
                n_bins=n_lfp_phase_bins,
            )
            input_df[("lfp_phase_bin", osc)] = phase_bins.apply(lambda x: x.mid).astype(float)
    return input_df
