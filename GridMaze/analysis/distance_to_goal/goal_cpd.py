"""
New library for goal-coding encoding analyses. Test if goal-distance explains unique variance over place-direction and distance
in the neural population.
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.metrics import mean_poisson_deviance

from matplotlib import pyplot as plt

from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import convert
from GridMaze.analysis.core import folds

from GridMaze.analysis.place_direction import bases as pdb
from GridMaze.analysis.distance_to_goal import bases as db

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "goal_coding" / "goal_by_distance_CPD"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

# %% Functions


def get_cpd_summary_df():
    save_path = RESULTS_DIR / "cpd_summary_df.csv"
    if save_path.exists():
        results_df = pd.read_csv(save_path, index_col=[0, 1])
    else:
        print(f"Generating CPD summary dataframe")
        subject_dfs = []
        for subject in SUBJECT_IDS:
            sessions = gs.get_maze_sessions(
                subject_IDs=[subject],
                maze_names="all",
                days_on_maze="late",
                goal_subsets=["subset_1", "subset_2"],
                with_data=["navigation_df", "navigation_spike_counts_df", "cluster_metrics", "trials_df"],
                must_have_data=True,
            )
            dfs = []
            for session in sessions:
                dfs.append(get_goal_cpd_df(session))
            subject_cpd_df = pd.concat(dfs)
            subject_cpd_df.index = pd.MultiIndex.from_product([subject_cpd_df.index, [subject]])
            subject_dfs.append(subject_cpd_df)
        results_df = pd.concat(subject_dfs)
        results_df.to_csv(save_path)
    return results_df


def get_goal_cpd_df(
    session,
    resolution=0.5,
    model_type="Ridge",
    spatial_coding="place_direction_onehot",
    distance_metrics=("steps_to_goal", "future"),
    goal_stratified_validation=True,
    trial_phases=["navigation"],
    max_steps_to_goal=30,
    min_spikes=300,
    pd_bases_kwargs={"n_bases": 8, "dim_red": "pca"},
    dtg_bases_kwargs={"n_bases": 5, "basis": "gamma"},
    verbose=True,
    max_jobs=10,
):
    """ """
    simple_maze = session.simple_maze()
    goals = session.goals
    if verbose:
        print(f"Loading basis functions")
    if "bases" in spatial_coding:
        # get place-direction bases
        pd_bases = pdb.get_place_direction_bases(pdb.get_heldout_sessions(session), **pd_bases_kwargs)
    else:
        pd_bases = None
    # get distance to goal bases
    dist_bases = db.distance_basis_generator(
        **dtg_bases_kwargs, btype=distance_metrics[0].split("_")[0], max_steps=max_steps_to_goal
    )
    if verbose:
        print(f"Loading input data")
    # get downsampled input data
    input_data = get_input_data(
        session, resolution, trial_phases=trial_phases, distance_metrics=distance_metrics, min_spikes=min_spikes
    )
    cluster_unique_IDs = input_data.spike_count.columns.values
    # get folds df
    folds_df = folds.get_folds_df(session, goal_stratified=goal_stratified_validation, return_unique_IDs=True)
    _folds = folds_df.columns.get_level_values(0).unique()

    model_name2regressor_classes = {
        "full": [spatial_coding, "distance", "goal_by_distance"],
        "reduced_goal_by_distance": [spatial_coding, "distance"],
        "reduced_distance": [spatial_coding, "goal_by_distance"],
        "reduced_spatial": ["distance", "goal_by_distance"],
    }

    n_jobs = min(len(_folds), max_jobs)

    if verbose:
        print(f"Running across {len(_folds)} folds with n_jobs={n_jobs}")
    cpd_dfs = []
    for fold in _folds:
        if verbose:
            print(f"Processing fold {fold}...")
            results_dfs = []
        for model_name, regressor_classes in model_name2regressor_classes.items():
            results_df = xval_regression(
                fold,
                folds_df,
                input_data,
                regressor_classes,
                model_type,
                cluster_unique_IDs,
                pd_bases,
                dist_bases,
                distance_metrics,
                goals,
                simple_maze,
                verbose,
            )
            results_df["model_name"] = model_name
            results_dfs.append(results_df)
        fold_results = pd.concat(results_dfs)
        if model_type == "Ridge":
            metric = "rss"
        elif model_type == "PossionRegressor":
            metric = "deviance"
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        metric_df = fold_results.set_index(["cluster_unique_ID", "model_name"])[metric].unstack()
        cpd_df = pd.DataFrame(index=metric_df.index)
        for reg_class in ["goal_by_distance", "distance", "spatial"]:
            reduced_model_metric = metric_df[f"reduced_{reg_class}"]
            full_model_metric = metric_df["full"]
            cpd_df[reg_class] = (reduced_model_metric - full_model_metric) / reduced_model_metric
        # tag with fold for later grouping
        cpd_df.index = pd.MultiIndex.from_product([cpd_df.index, [fold]], names=["cluster_unique_ID", "fold"])
        cpd_dfs.append(cpd_df)
    all_cpd = pd.concat(cpd_dfs)
    mean_cpd = all_cpd.groupby("cluster_unique_ID").mean()  # average across folds
    return mean_cpd


# %% L2 OLS or Poisson regression


def xval_regression(
    fold,
    folds_df,
    input_data,
    regressor_classes,
    model,
    cluster_unique_IDs,
    pd_bases,
    dist_bases,
    distance_metrics,
    goals,
    simple_maze,
    verbose,
):
    """ """
    # first find best alpha (regularisation) for every cluster across xval folds of training data
    if verbose:
        print("finding best xval alpha for each cluster...")
    train_df = folds_df[fold]["train"]
    cols = train_df.columns.values
    xval_reg_results = Parallel(n_jobs=len(cols))(
        delayed(_process_reg_validation_fold)(
            input_data,
            train_df,
            regressor_classes,
            cols,
            col,
            pd_bases,
            distance_metrics,
            dist_bases,
            goals,
            simple_maze,
            model=model,
            cluster_unique_IDs=cluster_unique_IDs,
            i=i,
            verbose=verbose,
        )
        for i, col in enumerate(cols)
    )
    xval_reg_results = [result for fold_results in xval_reg_results for result in fold_results]
    reg_df = pd.DataFrame(xval_reg_results)
    # compute average score (R2) for each cluster and alpha, then take the best for each
    cluster_reg_scores = reg_df.groupby(["cluster_unique_ID", "alpha"]).score.mean().unstack()
    cluster2opt_reg = cluster_reg_scores.idxmax(axis=1)
    # using opt alpha to run main test_train cpd
    if verbose:
        print("running main test_train cpd...")
    test_df = folds_df[fold]["test"]
    test_trials = test_df.unstack().dropna().values
    test_df = input_data[input_data.trial_unique_ID.isin(test_trials)]
    train_trials = train_df.unstack().dropna().values
    train_df = input_data[input_data.trial_unique_ID.isin(train_trials)]
    X_train, X_test, Y_train, Y_test = get_test_train_arrays(
        train_df,
        test_df,
        regressor_classes=regressor_classes,
        pd_bases=pd_bases,
        distance_metrics=distance_metrics,
        dist_bases=dist_bases,
        goals=goals,
        simple_maze=simple_maze,
    )
    fold_results = []
    for i, cid in enumerate(cluster_unique_IDs):
        y_train, y_test = Y_train[:, i], Y_test[:, i]
        alpha = cluster2opt_reg[cid]
        if model == "Ridge":
            Model = Ridge(alpha=alpha, max_iter=10_000, random_state=0)
            Model.fit(X_train, y_train)
            score = Model.score(X_test, y_test)
            residuals = y_test - Model.predict(X_test)
            rss = np.sum(residuals**2)
            fold_results.append(
                {
                    "cluster_unique_ID": cid,
                    "alpha": alpha,
                    "score": score,
                    "rss": rss,
                    "fold": fold,
                }
            )
        elif model == "PossionRegressor":
            Model = PoissonRegressor(alpha=alpha, max_iter=10_000)
            Model.fit(X_train, y_train)
            score = Model.score(X_test, y_test)
            y_pred = Model.predict(X_test)
            deviance = mean_poisson_deviance(y_test, y_pred)
            fold_results.append(
                {
                    "cluster_unique_ID": cid,
                    "alpha": alpha,
                    "score": score,
                    "deviance": deviance,
                    "fold": fold,
                }
            )
    return pd.DataFrame(fold_results)


def _process_reg_validation_fold(
    input_data,
    train_df,
    regressor_classes,
    cols,
    col,
    pd_bases,
    distance_metrics,
    dist_bases,
    goals,
    simple_maze,
    model,
    cluster_unique_IDs,
    i,
    verbose,
):
    """
    for parallel processing
    """
    other_cols = cols[cols != col]
    val_trials = train_df[col].dropna().values
    vtrain_trials = train_df[other_cols].unstack().dropna().values
    val_df = input_data[input_data.trial_unique_ID.isin(val_trials)]
    vtrain_df = input_data[input_data.trial_unique_ID.isin(vtrain_trials)]
    X_vtrain, X_val, Y_vtrain, Y_val = get_test_train_arrays(
        vtrain_df,
        val_df,
        regressor_classes=regressor_classes,
        pd_bases=pd_bases,
        distance_metrics=distance_metrics,
        dist_bases=dist_bases,
        goals=goals,
        simple_maze=simple_maze,
    )
    results = []
    for j, cid in enumerate(cluster_unique_IDs):
        y_train, y_val = Y_vtrain[:, j], Y_val[:, j]
        alpha, score = reg_search_regression(X_vtrain, y_train, X_val, y_val, model, return_as="best", verbose=verbose)
        results.append(
            {
                "cluster_unique_ID": cid,
                "alpha": alpha,
                "score": score,
                "vfold": i,
            }
        )
    return results


def reg_search_regression(
    X_train,
    y_train,
    X_test,
    y_test,
    model="Ridge",
    tol=1e-4,
    max_rounds=35,
    patience=15,
    return_as="best",
    verbose=False,
):
    """
    Runs OLS (Ridge) or Poisson regression (PoissonRegressor) with increasing alpha
    until the score stops improving.
    Returns the best alpha and score.
    If return_as="history", returns a array of (alphas, scores).
    Scores are R² for OLS and Pseudo R2 for Poisson regression.
    """

    if model == "Ridge":
        alpha = 1
    else:
        alpha = 1e-2

    best_alpha = alpha
    best_score = -np.inf
    best_round = 0

    history = []
    no_improve_count = 0

    for round_idx in range(1, max_rounds + 1):

        if model == "Ridge":
            Model = Ridge(alpha=alpha, max_iter=10_000, random_state=0)
        elif model == "PossionRegressor":
            Model = PoissonRegressor(alpha=alpha, max_iter=10_000)
        else:
            raise ValueError(f"Unknown model: {model}")

        Model.fit(X_train, y_train)
        score = Model.score(X_test, y_test)
        history.append((alpha, score))
        if verbose:
            print(f"Round {round_idx:2d}: α = {alpha:.3e},  R² = {score:.4f}")

        # update best if we improved by more than tol
        if score > best_score + tol:
            best_score = score
            best_alpha = alpha
            best_round = round_idx
            no_improve_count = 0

        else:
            # only count towards patience if best_score is non-negative
            if best_score >= 0:
                no_improve_count += 1
                if no_improve_count >= patience:
                    break

        alpha *= 5

    if verbose:
        print(f"→ Best α = {best_alpha:.3e} (round {best_round}) with R² = {best_score:.4f}")
        print("")

    if return_as == "history":
        return np.array(history).T
    elif return_as == "best":
        return best_alpha, best_score
    else:
        raise ValueError(f"Unknown return_as: {return_as}")


# %%


def get_test_train_arrays(
    train_df,
    test_df,
    regressor_classes=["place_direction_bases", "distance"],
    pd_bases=None,
    distance_metrics=None,
    dist_bases=None,
    goals=None,
    simple_maze=None,
    scale_features=True,
    block_normalise=False,
    return_feature_names=False,
):
    """
    No whitening makes a big difference to final CPD values
    """
    Xs = []
    for df in train_df, test_df:
        X, n_features = [], []
        for rtype in regressor_classes:
            x = get_regressor_class(
                df,
                rtype,
                pd_bases,
                distance_metrics,
                dist_bases,
                goals,
                simple_maze,
            )
            X.append(x)
            n_features.append(x.shape[1])
        Xs.append(np.hstack(X))
    X_train, X_test = Xs
    Y_train, Y_test = train_df.spike_count.values, test_df.spike_count.values
    if scale_features:
        scaler = StandardScaler()  # mean=0, std=1 per column
        scaler.fit(X_train)  # learn stats on train
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    if block_normalise:
        idx = 0
        for size in n_features:
            scale_factor = np.sqrt(size)
            X_train[:, idx : idx + size] /= scale_factor
            X_test[:, idx : idx + size] /= scale_factor
            idx += size
    if not return_feature_names:
        return X_train, X_test, Y_train, Y_test
    else:
        feature_names = np.repeat(regressor_classes, n_features)
        return (X_train, X_test, Y_train, Y_test), feature_names


def get_regressor_class(df, rtype, pd_bases=None, distance_metrics=None, dist_bases=None, goals=None, simple_maze=None):
    """
    rtype: regressor class name
    """
    if rtype == "place_direction_bases":
        assert pd_bases is not None
        place_directions = (
            df[[("maze_position", "simple"), ("cardinal_movement_direction", "")]].apply(tuple, axis=1).values
        )
        r = pd_bases.loc[place_directions].values  # n_samples x n_place_direction_bases

    elif rtype == "place_direction_onehot":
        assert simple_maze is not None
        place_directions = (
            df[[("maze_position", "simple"), ("cardinal_movement_direction", "")]].apply(tuple, axis=1).values
        )
        place_directions = np.array([f"{x[0]}_{x[1]}" for x in place_directions])
        r = convert.place_direction2onehot(place_directions, simple_maze=simple_maze)

    elif rtype == "place_onehot":
        assert simple_maze is not None
        places = df.maze_position.simple.values
        r = convert.place2onehot(places, simple_maze=simple_maze)

    elif rtype == "distance":
        assert dist_bases is not None
        assert distance_metrics is not None
        r = dist_bases(df[distance_metrics].values)  # n_samples x n_dist_bases

    elif rtype == "goal":
        assert goals is not None
        r = convert.goal2onehot(df.goal.values, goals=goals)  # n_samples x n_goals

    elif rtype == "goal_by_distance":
        assert goals is not None
        assert dist_bases is not None
        assert distance_metrics is not None
        goal_onehot = convert.goal2onehot(df.goal.values, goals=goals)
        dist_bases_activations = dist_bases(df[distance_metrics].values)  # n_samples x n_dist_bases
        r = goal_onehot[:, :, None] * dist_bases_activations[:, None, :]  # n_samples x n_goals x n_dist_bases
        r = r.reshape(goal_onehot.shape[0], -1)  # n_samples x (n_goals * n_dist_bases)

    else:
        raise ValueError(f"Unknown regressor type: {rtype}")
    return r


def get_input_data(
    session,
    resolution=0.5,
    distance_metrics=("steps_to_goal", "future"),
    trial_phases=["navigation"],
    max_steps_to_goal=30,
    min_spikes=300,
):
    """ """
    # load data
    navigation_df = session.navigation_df
    spike_counts_df = session.navigation_spike_counts_df.reset_index(drop=True)
    # filter for single units
    keep_clusters = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=False,
    )
    spike_counts_df = spike_counts_df[spike_counts_df.columns[spike_counts_df.spike_count.columns.isin(keep_clusters)]]
    # downsample data
    navigation_df, spike_counts_df = ds.downsample_nav_spikes_data(
        navigation_df, spike_counts_df, resolution=resolution, distance_metrics=distance_metrics
    )
    # combine
    nav_rates_df = pd.concat([navigation_df, spike_counts_df], axis=1)
    # filter for trial phases
    nav_rates_df = nav_rates_df[nav_rates_df.trial_phase.isin(trial_phases)]
    # filter for max steps to goal
    nav_rates_df = nav_rates_df[nav_rates_df.steps_to_goal.future.le(max_steps_to_goal)]
    # check remaining clusters pass min_spikes
    reject_clusters = nav_rates_df.spike_count.columns[nav_rates_df.spike_count.sum().lt(min_spikes)]
    nav_rates_df = nav_rates_df[nav_rates_df.columns[~nav_rates_df.columns.get_level_values(1).isin(reject_clusters)]]
    # ensure cardinal movement direction is always defined
    nav_rates_df = nav_rates_df[nav_rates_df.cardinal_movement_direction.notna()]
    return nav_rates_df
