"""
Library for comparing distance to goal tuning metrics
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.preprocessing import StandardScaler

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import folds
from GridMaze.analysis.core import encoding_utils as eu
from GridMaze.analysis.core import convert

from GridMaze.analysis.distance_to_goal import bases as db
from GridMaze.analysis.distance_to_goal import distributions as dd


# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

# %% Functions


def compare_distance_metric_regression_weights(
    session,
    resolution=0.5,
    fixed_alpha=10,
    model="PoissonRegressor",
    n_bases=10,
    basis_type="gamma",
    metric_1="geodesic",
    metric_2="euclidean",
    max_steps_to_goal=25,
):
    """
    Runs a Poission GLM predicting spikes from basis activations of two distance metrics.
    """
    # get input data
    input_data = get_input_data(
        session,
        metric_1=("distance_to_goal", metric_1),
        metric_2=("distance_to_goal", metric_2),
        resolution=resolution,
        max_steps_to_goal=max_steps_to_goal,
    )
    cluster_unique_IDs = input_data.spike_count.columns.values
    # get a set of basis function activates for each distance metric
    basis_activation_dfs = []
    for i, m in enumerate([metric_1, metric_2]):
        _m = ("distance_to_goal", m)
        basis_fn = db.distance_basis_generator(
            n_bases=n_bases,
            basis=basis_type,
            btype="distance",
            max_distance=dd.get_distance_percentile(_m, percentile=85),
        )
        basis_activations = basis_fn(input_data[_m])
        basis_activations = pd.DataFrame(
            basis_activations,
            columns=pd.MultiIndex.from_product([[f"metric_{i+1}"], np.arange(0, n_bases)]),
            index=input_data.index,
        )
        basis_activation_dfs.append(basis_activations)
    # combine basis activations with input data
    input_data = pd.concat([input_data, *basis_activation_dfs], axis=1)
    if not fixed_alpha:
        # get xval opt alpha for each cluster
        folds_df = folds.get_folds_df(session, goal_stratified=False, n_folds=5)
        cluster_alphas = get_test_train_opt_alpha(folds_df, input_data, model=model)
    else:
        cluster_alphas = pd.Series(index=cluster_unique_IDs, data=fixed_alpha)
    # fit each cluster in a Possion GLM with distance metric featrues
    X = np.hstack([input_data.metric_1.values, input_data.metric_2.values])
    # ensure X is scaled when inperpretting betas
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y = input_data.spike_count.values
    results = []
    for i, cluster in enumerate(cluster_unique_IDs):
        y = Y[:, i]
        alpha = cluster_alphas.loc[cluster]
        if model == "PoissonRegressor":
            Model = PoissonRegressor(alpha=alpha, max_iter=10_000)
        elif model == "Ridge":
            Model = Ridge(alpha=alpha, max_iter=10_000, random_state=0)
        else:
            raise ValueError(f"Unknown model: {model}")
        Model.fit(X, y)
        betas = Model.coef_
        beta_metic_1 = betas[:n_bases]
        beta_metic_2 = betas[n_bases:]
        L1_metric_1, L1_metric_2 = np.abs(beta_metic_1).sum(), np.abs(beta_metic_2).sum()
        L1_sum = L1_metric_1 + L1_metric_2
        L2_metric_1, L2_metric_2 = np.linalg.norm(beta_metic_1, ord=2), np.linalg.norm(beta_metic_2, ord=2)
        L2_sum = L2_metric_1 + L2_metric_2
        for metric, L1, L2 in zip([metric_1, metric_2], [L1_metric_1, L1_metric_2], [L2_metric_1, L2_metric_2]):
            results.append(
                {
                    "cluster_unique_ID": cluster,
                    "alpha": alpha,
                    "metric": metric,
                    "L1_ratio": L1 / L1_sum,
                    "L2_ratio": L2 / L2_sum,
                }
            )
    results_df = pd.DataFrame(results)
    results_df["subject_ID"] = session.subject_ID
    results_df["maze_name"] = session.maze_name
    results_df["day_on_maze"] = session.day_on_maze
    return results_df


# %% Get Xvaled regularisation across either test_train splits or folds within training data


def get_test_train_opt_alpha(folds_df, input_data, model="PoissonRegressor", max_jobs=20):
    """ """
    cluster_unique_IDs = input_data.spike_count.columns.values
    _folds = folds_df.columns.get_level_values(0).unique()
    results = []
    for fold in _folds:
        test_trials = folds_df[fold]["test"].unstack().dropna().values
        test_df = input_data[input_data.trial_unique_ID.isin(test_trials)]
        train_trials = folds_df[fold]["train"].unstack().dropna().values
        train_df = input_data[input_data.trial_unique_ID.isin(train_trials)]
        X_train, Y_train, X_test, Y_test = get_test_train_arrays(train_df, test_df, scale_X=True)
        fold_results = Parallel(n_jobs=max_jobs)(
            delayed(_process_cluster)(fold, i, cluster, X_train, Y_train, X_test, Y_test, model=model)
            for i, cluster in enumerate(cluster_unique_IDs)
        )
        results.extend(fold_results)
    reg_df = pd.DataFrame(results)
    # get median best_alpha across folds
    cluster_opt_alphas = reg_df.groupby(["cluster_unique_ID"]).best_alpha.median()
    return cluster_opt_alphas


def _process_cluster(fold, i, cluster, X_train, Y_train, X_test, Y_test, model="PoissonRegressor"):
    print(cluster)
    y_train, y_test = Y_train[:, i], Y_test[:, i]
    best_alpha, best_score = eu.reg_search_regression(
        X_train, y_train, X_test, y_test, model=model, return_as="best", verbose=True, patience=5
    )
    return {
        "fold": fold,
        "cluster_unique_ID": cluster,
        "best_alpha": best_alpha,
        "best_score": best_score,
    }


def get_train_folds_opt_alpha():
    """ """

    return


# %%


def reg_search_PoissionRegression():
    """ """

    return


def get_test_train_arrays(train_df, test_df, scale_X=True):
    """ """
    X_train, X_test = np.hstack([train_df.metric_1.values, train_df.metric_2.values]), np.hstack(
        [test_df.metric_1.values, test_df.metric_2.values]
    )
    Y_train, Y_test = train_df.spike_count.values, test_df.spike_count.values
    # standardise
    if scale_X:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, Y_train, X_test, Y_test


# %%
def get_input_data(session, metric_1, metric_2, resolution=0.2, max_steps_to_goal=25, min_spikes=300):
    """ """
    # load data
    navigation_df = session.navigation_df
    spike_counts_df = session.navigation_spike_counts_df.reset_index(drop=True)
    cluster_metrics_df = session.cluster_metrics
    session_info = session.session_info
    # filter for single units
    single_units = cluster_metrics_df[cluster_metrics_df.single_unit].cluster_ID
    single_units = convert.cluster_IDs2scluster_unique_IDs(session_info, single_units)
    spike_counts_df = spike_counts_df[[("spike_count", c) for c in single_units]]
    # downsample to specified resolution
    nav_info, spike_counts = ds.downsample_nav_spikes_data(
        navigation_df,
        spike_counts_df,
        resolution=resolution,
        distance_metrics=[
            metric_1,
            metric_2,
            ("steps_to_goal", "future"),
            ("distance_to_goal", "future"),
        ],
    )
    # filter for navigation trial phaes and distance / steps to goal
    masks = [
        (nav_info.trial_phase == "navigation"),
        (nav_info.steps_to_goal.future.le(max_steps_to_goal)),
    ]
    mask = np.logical_and.reduce(masks)
    nav_info = nav_info[mask]
    spike_counts = spike_counts[mask]
    # check remaining clusters pass min_spikes
    reject_clusters = spike_counts.columns[spike_counts.spike_count.sum().lt(min_spikes)]
    spike_counts = spike_counts.drop(columns=reject_clusters)
    # remove future distance columns if they are not in metric_1 or metric_2
    if metric_1 != ("distance_to_goal", "future") or metric_2 != ("distance_to_goal", "future"):
        nav_info = nav_info.drop(columns=[("distance_to_goal", "future")])
    if metric_1 != ("steps_to_goal", "future") or metric_2 != ("steps_to_goal", "future"):
        nav_info = nav_info.drop(columns=[("steps_to_goal", "future")])
    # combine and return
    input_data = pd.concat([nav_info, spike_counts], axis=1)
    return input_data
