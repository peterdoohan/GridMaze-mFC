"""
Distance to goal decoding using logistic regression this time.
"""

# %% Imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

from GridMaze.analysis.core import convert
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import folds
from GridMaze.analysis.distance_to_goal import distributions as dd


# %% Global Variables

# %% Functions


def test(session, n_folds=5, sqrt_spikes=False, standardise_spikes=False):
    """ """
    input_data = get_input_data(session)
    distance_bin_mids = np.array(sorted([b.mid for b in input_data.distance_bin.unique()]))
    folds_df = folds.get_folds_df(session, goal_stratified=False, return_unique_IDs=True, n_folds=n_folds)
    _folds = folds_df.columns.get_level_values(0).unique()
    results_df = input_data[["trial"]].droplevel(1, axis=1)
    for res_col in ["true_distance", "decoded_distance", "distance_bin_mids"]:
        results_df[res_col] = np.nan
    for fold in _folds:
        print(fold)
        train_df, test_df = folds._get_test_train_dfs(input_data, folds_df[fold])
        X_train, X_test = train_df.spike_count.values, test_df.spike_count.values
        if sqrt_spikes:
            X_train, X_test = np.sqrt(X_train), np.sqrt(X_test)
        if standardise_spikes:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        y_train, y_test = train_df.distance_bin_id.values, test_df.distance_bin_id.values
        true_dist = test_df.distance_to_goal.geodesic.values  # n_samples
        # fit model
        model = LogisticRegression(C=1, max_iter=10_000)
        model.fit(X_train, y_train)
        # predict
        y_prob = model.predict_proba(X_test)  # n_samples, n_distance_bins
        decoded_dist_weighted = np.dot(y_prob, distance_bin_mids)  # n_samples
        resuls_idx = test_df.index
        results_df.loc[resuls_idx, "true_distance"] = true_dist
        results_df.loc[resuls_idx, "decoded_distance"] = decoded_dist_weighted
        results_df.loc[resuls_idx, "distance_bin_mids"] = test_df.distance_bin.apply(lambda x: x.mid)
    return results_df


def get_opt_reg(input_data, train_df, reg_range=np.logspace(-4, 4, 20), tol=1e-4, patience=5):
    """
    CV search for optimal regulaisation strength (in training data)
    """
    best_alpha = reg_range[0]
    best_score = -np.inf
    history = []
    no_improve_count = 0
    for alpha in reg_range:
        model = LogisticRegression(C=1, max_iter=10_000)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

    return


def get_input_data(
    session,
    resolution=0.5,
    metric=("distance_to_goal", "geodesic"),
    include_multiunits=True,
    moving_only=False,
    max_steps_to_goal=30,
    bin_spacing=0.05,
    bin_method="uniform",
    n_log_bins=25,
):
    """"""
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
    input_df = input_df[input_df.trial_phase == "navigation"]
    # add distance bins
    if moving_only:
        input_df = input_df[input_df.moving]
    if max_steps_to_goal is not None:
        input_df = input_df[input_df.steps_to_goal.future < max_steps_to_goal]
    # remove frames where distance is above max (treat as outliers)
    if metric[0] == "distance_to_goal":
        max_distance = dd.get_distance_percentile(metric, 0.85)
        if bin_method == "uniform":
            n_bins = int(max_distance / bin_spacing)
        elif bin_method == "log":
            n_bins = n_log_bins
        input_df = input_df[input_df[metric] < max_distance]
        bins = convert._get_distance_bins(
            binning_method=bin_method,
            n_distance_bins=n_bins,
            distance_metrics=metric,
            max_distance=max_distance,
        )
    else:
        NotImplementedError()
    # bin distances
    input_df.loc[:, "distance_bin"] = pd.cut(input_df[metric], bins=bins, include_lowest=True).to_numpy()
    bin2bin_id = {b: i for i, b in enumerate(bins)}
    input_df.loc[:, "distance_bin_id"] = input_df.distance_bin.map(bin2bin_id)
    return input_df
