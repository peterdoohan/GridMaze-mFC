"""
Library for comparing distance to goal tuning metrics
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from itertools import combinations
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_poisson_deviance

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import folds
from GridMaze.analysis.core import encoding_utils as eu
from GridMaze.analysis.core import convert


from GridMaze.analysis.distance_to_goal import bases as db
from GridMaze.analysis.distance_to_goal import distributions as dd


# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "distaance_to_goal" / "distance_metrics"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

# %% L1, L2 ratio comparison function


def get_distance_metric_weight_summaries(
    session,
    resolution=0.5,
    fixed_alpha=False,
    model="PoissonRegressor",
    n_bases=10,
    basis_type="gamma",
    metric_1="geodesic",
    metric_2="euclidean",
    max_steps_to_goal=25,
    max_jobs=20,
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
        if not m == "future":
            _max = dd.get_distance_percentile(_m, percentile=85)
        else:
            # future distance distribution has large tail due to off task trials
            _max = dd.get_distance_percentile("geodesic", percentile=85)
        basis_fn = db.distance_basis_generator(
            n_bases=n_bases,
            basis=basis_type,
            btype="distance",
            max_distance=_max,
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
    # get data to fit
    X = np.hstack([input_data.metric_1.values, input_data.metric_2.values])
    # ensure X is scaled when inperpretting betas
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y = input_data.spike_count.values
    # fit each cluster in a Linear OLS / Possion GLM with distance metric featrues
    cluster_results = Parallel(n_jobs=max_jobs)(
        delayed(_process_cluster_betas)(
            model,
            X,
            Y[:, i],
            cluster_alphas.loc[cluster],
            cluster,
            n_bases,
            metric_1,
            metric_2,
        )
        for i, cluster in enumerate(cluster_unique_IDs)
    )
    results_df = pd.DataFrame([i for j in cluster_results for i in j])
    return results_df


def _process_cluster_betas(model, X, y, alpha, cluster, n_bases, metric_1, metric_2):
    """ """
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
    results = []
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
    return results


# %% CPD function


def get_distance_metric_CPD_summaries(verbose=True):
    """ """
    save_path = RESULTS_DIR / "cpd_summary_df.csv"
    if save_path.exists():
        if verbose:
            print(f"Loading CPD summaries df from {save_path}")
        results_df = pd.read_csv(save_path, index_col=0, header=[0, 1])
    else:
        sessions = gs.get_maze_sessions(
            subject_IDs="all",
            maze_names="all",
            days_on_maze="all",
            with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "trials_df"],
            must_have_data=True,
        )
        dfs = []
        for session in sessions:
            if verbose:
                print(session.name)
            comparisons_df = run_pairwise_CPD_comparisons(session, verbose=verbose)
            dfs.append(comparisons_df)
        results_df = pd.concat(dfs, axis=0)
        # save
        results_df.to_csv(save_path, index=True)
        if verbose:
            print(f"Saved CPD summaries df to {save_path}")
    return results_df


def run_pairwise_CPD_comparisons(session, verbose=True):
    """ """
    distance_metrics = ["geodesic", "euclidean", "manhattan", "future"]
    metric_pairs = list(combinations(distance_metrics, 2))
    cpd_dfs = []
    for metric_1, metric_2 in metric_pairs:
        _name = f"{metric_1}_vs_{metric_2}"
        if verbose:
            print(_name)
        cpd_df = get_distance_metric_CPDs(session, metric_1=metric_1, metric_2=metric_2)
        cpd_df.columns = pd.MultiIndex.from_product([[_name], cpd_df.columns])
        cpd_dfs.append(cpd_df)
    comparisons_df = pd.concat(cpd_dfs, axis=1)
    comparisons_df[("subject_ID", "")] = session.subject_ID
    comparisons_df[("maze_name", "")] = session.maze_name
    comparisons_df[("day_on_maze", "")] = session.day_on_maze
    return comparisons_df


def get_distance_metric_CPDs(
    session,
    metric_1="geodesic",
    metric_2="euclidean",
    resolution=0.5,
    model="PoissonRegressor",
    n_bases=10,
    basis_type="gamma",
    max_steps_to_goal=25,
    max_jobs=20,
):
    """ """
    # get input data
    input_data = get_input_data(
        session,
        metric_1=("distance_to_goal", metric_1),
        metric_2=("distance_to_goal", metric_2),
        resolution=resolution,
        max_steps_to_goal=max_steps_to_goal,
    )
    cluster_unique_IDs = input_data.spike_count.columns.values
    # get a set of basis function activations for each distance metric
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
    folds_df = folds.get_folds_df(session, goal_stratified=False, n_folds=5)
    _folds = folds_df.columns.get_level_values(0).unique()
    model_name2regessor_classes = {
        "full": ["metric_1", "metric_2"],
        f"reduced_{metric_1}": ["metric_2"],
        f"reduced_{metric_2}": ["metric_1"],
    }
    all_results = []
    for fold in _folds:
        fold_df = folds_df[fold]
        fold_results = []
        for model_name, regressor_classes in model_name2regessor_classes.items():
            cluster_alphas = get_train_folds_opt_alpha(
                fold_df, input_data, model=model, regressor_classes=regressor_classes
            )
            train_trials = fold_df["train"].unstack().dropna().values
            test_trials = fold_df["test"].unstack().dropna().values
            train_df = input_data[input_data.trial_unique_ID.isin(train_trials)]
            test_df = input_data[input_data.trial_unique_ID.isin(test_trials)]
            X_train, Y_train, X_test, Y_test = get_test_train_arrays(train_df, test_df, regressor_classes, scale_X=True)
            model_results = Parallel(n_jobs=max_jobs)(
                delayed(_process_cluster_cpd)(
                    X_train,
                    Y_train[:, i],
                    X_test,
                    Y_test[:, i],
                    model,
                    cluster_alphas.loc[cluster],
                    cluster,
                    fold,
                    model_name,
                )
                for i, cluster in enumerate(cluster_unique_IDs)
            )
            fold_results.extend(model_results)
        all_results.extend(fold_results)
    df = pd.DataFrame(all_results)  # every cluster, model, model - socre
    # calculate CPD values for metric_1 and metric_2 by comparing full and reudced models
    metric = "deviance" if model == "PoissonRegressor" else "rss"
    # average metric across folds
    model_metrics = df.groupby(["cluster_unique_ID", "model_name"])[metric].mean().unstack()
    cpd_df = pd.DataFrame(index=model_metrics.index)
    for m in [metric_1, metric_2]:
        reduced = model_metrics[f"reduced_{m}"]
        full = model_metrics["full"]
        cpd_df[m] = (reduced - full) / (reduced)
    return cpd_df


def _process_cluster_cpd(X_train, y_train, X_test, y_test, model, alpha, cluster, fold, model_name):
    if model == "PoissonRegressor":
        Model = PoissonRegressor(alpha=alpha, max_iter=10_000)
        Model.fit(X_train, y_train)
        y_pred = Model.predict(X_test)
        score = Model.score(X_test, y_test)
        deviance = mean_poisson_deviance(y_test, y_pred)
        return {
            "cluster_unique_ID": cluster,
            "fold": fold,
            "score": score,
            "deviance": deviance,
            "alpha": alpha,
            "model_name": model_name,
        }
    elif model == "Ridge":
        Model = Ridge(alpha=alpha, max_iter=10_000, random_state=0)
        Model.fit(X_train, y_train)
        y_pred = Model.predict(X_test)
        score = Model.score(X_test, y_test)
        rss = np.sum((y_test - y_pred) ** 2)
        return {
            "cluster_unique_ID": cluster,
            "fold": fold,
            "score": score,
            "rss": rss,
            "alpha": alpha,
            "model_name": model_name,
        }


# %% Get Xvaled regularisation across either test_train splits or folds within training data


def get_test_train_opt_alpha(folds_df, input_data, model="PoissonRegressor", max_jobs=20):
    """
    Returns best alpha (median across folds) for each cluster in input_data over test_train splits
    """
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
            delayed(_process_cluster_reg_search)(fold, i, cluster, X_train, Y_train, X_test, Y_test, model=model)
            for i, cluster in enumerate(cluster_unique_IDs)
        )
        results.extend(fold_results)
    reg_df = pd.DataFrame(results)
    # get median best_alpha across folds
    cluster_opt_alphas = reg_df.groupby(["cluster_unique_ID"]).best_alpha.median()
    return cluster_opt_alphas


def get_train_folds_opt_alpha(
    fold_df, input_data, model="PoissonRegressor", regressor_classes=["metric_1", "metric_2"], max_jobs=20
):
    """ """
    cluster_unique_IDs = input_data.spike_count.columns.values
    train_df = fold_df["train"]
    train_folds = train_df.columns.values
    train_fold_results = []
    for fold in train_folds:
        vtest_trials = train_df[fold].dropna().values
        vtrain_trials = train_df[[f for f in train_folds if f != fold]].unstack().dropna().values
        vtrain_df = input_data[input_data.trial_unique_ID.isin(vtrain_trials)]
        vtest_df = input_data[input_data.trial_unique_ID.isin(vtest_trials)]
        X_train, Y_train, X_test, Y_test = get_test_train_arrays(vtrain_df, vtest_df, regressor_classes, scale_X=True)
        fold_results = Parallel(n_jobs=max_jobs)(
            delayed(_process_cluster_reg_search)(fold, i, cluster, X_train, Y_train, X_test, Y_test, model=model)
            for i, cluster in enumerate(cluster_unique_IDs)
        )
        train_fold_results.extend(fold_results)
    reg_df = pd.DataFrame(train_fold_results)
    # get median best_alpha across folds
    cluster_opt_alphas = reg_df.groupby(["cluster_unique_ID"]).best_alpha.median()
    return cluster_opt_alphas


def _process_cluster_reg_search(fold, i, cluster, X_train, Y_train, X_test, Y_test, model="PoissonRegressor"):
    y_train, y_test = Y_train[:, i], Y_test[:, i]
    best_alpha, best_score = eu.reg_search_regression(
        X_train, y_train, X_test, y_test, model=model, return_as="best", verbose=False, patience=5
    )
    return {
        "fold": fold,
        "cluster_unique_ID": cluster,
        "best_alpha": best_alpha,
        "best_score": best_score,
    }


# %%


def get_test_train_arrays(train_df, test_df, regressor_classes=["metric_1", "metric_2"], scale_X=True):
    """ """
    X_train, X_test = [], []
    if "metric_1" in regressor_classes:
        X_train.append(train_df.metric_1.values)
        X_test.append(test_df.metric_1.values)
    if "metric_2" in regressor_classes:
        X_train.append(train_df.metric_2.values)
        X_test.append(test_df.metric_2.values)
    if "metric_1" not in regressor_classes and "metric_2" not in regressor_classes:
        raise ValueError("Must include at least one metric for input_features")
    X_train, X_test = np.hstack(X_train), np.hstack(X_test)
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
    distance_metrics = list(
        set(
            [
                metric_1,
                metric_2,
                ("steps_to_goal", "future"),
                ("distance_to_goal", "future"),
            ]
        )
    )
    nav_info, spike_counts = ds.downsample_nav_spikes_data(
        navigation_df,
        spike_counts_df,
        resolution=resolution,
        distance_metrics=distance_metrics,
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
    # combine and return
    input_data = pd.concat([nav_info, spike_counts], axis=1)
    return input_data
