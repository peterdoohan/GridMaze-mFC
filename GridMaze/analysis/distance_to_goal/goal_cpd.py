"""
New library for goal-coding encoding analyses. Test if goal-distance explains unique variance over place-direction and distance
in the neural population.
@peterdoohan
"""

# %% Imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_poisson_deviance
from matplotlib import pyplot as plt

from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import convert
from GridMaze.analysis.core import folds

from GridMaze.analysis.place_direction import bases as pdb
from GridMaze.analysis.distance_to_goal import bases as db

# %% Global Variables

# %% Functions


def test(
    session,
    resolution=0.5,
    distance_metrics=("steps_to_goal", "future"),
    goal_stratified_validation=True,
    n_test_trials=None,
    trial_phases=["navigation"],
    max_steps_to_goal=30,
    min_spikes=300,
    pd_bases_kwargs={"n_bases": 6, "dim_red": "pca"},
    dtg_bases_kwargs={"n_bases": 3, "basis": "gamma"},
    verbose=True,
):
    """ """
    goals = session.goals
    if verbose:
        print(f"Loading basis functions")
    # get place-direction bases
    pd_bases = pdb.get_place_direction_bases(pdb.get_heldout_sessions(session), **pd_bases_kwargs)
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
    folds_df = folds.get_folds_df(
        session, goal_stratified=goal_stratified_validation, return_unique_IDs=True, n_test_trials=n_test_trials
    )
    _folds = folds_df.columns.get_level_values(0).unique()

    # find optimal regularisation for each cell on an example fold of the data
    if verbose:
        print(f"Finding optimal regularisation for each cell")
    model2regressor_classes = {
        "full_goal_by_distance": ["place_direction", "distance", "goal_by_distance"],
        "full_goal": ["place_direction", "distance", "goal"],
        "reduced": ["place_direction", "distance"],
    }
    cpd_dfs = []
    for fold in _folds:
        fold_results = []
        if verbose:
            print(fold)
        fold_df = folds_df[fold]
        train_df, test_df = folds._get_test_train_dfs(input_data, fold_df, training_trial_phases=False)
        for _model, regressor_classes in model2regressor_classes.items():
            if verbose:
                print(_model)
            X_train, X_test, Y_train, Y_test = (
                get_test_train_arrays(  # X [n_regressors, n_samples], Y[n_cluster_unique_IDs, n_samples]
                    train_df,
                    test_df,
                    regressor_classes=regressor_classes,
                    pd_bases=pd_bases,
                    distance_metrics=distance_metrics,
                    dist_bases=dist_bases,
                    goals=goals,
                )
            )
            for i, cluster_unique_ID in enumerate(cluster_unique_IDs):
                if verbose:
                    print(cluster_unique_ID)
                y_train, y_test = Y_train[:, i], Y_test[:, i]  # n_train_samples, n_test_samples
                # get the best possible fit searching over many reg values specifically for this fold
                alpha, score, rss = reg_opt_Ridge(X_train, y_train, X_test, y_test, return_as="best", verbose=False)
                fold_results.append(  # save output in df
                    {
                        "score": score,
                        "rss": rss,
                        "alpha": alpha,
                        "fold": fold,
                        "cluster_unique_ID": cluster_unique_ID,
                        "model": _model,
                    }
                )
        # calculate CPD from rss across full and reduced models within fold
        fold_results_df = pd.DataFrame(fold_results)
        rss_df = fold_results_df.set_index(["cluster_unique_ID", "model"]).rss.unstack()
        cpd_df = pd.DataFrame(index=rss_df.index)
        cpd_df["goal"] = (rss_df["reduced"] - rss_df["full_goal"]) / rss_df["reduced"]
        cpd_df["goal_by_distance"] = (rss_df["reduced"] - rss_df["full_goal_by_distance"]) / rss_df["reduced"]
        cpd_df.index = pd.MultiIndex.from_product([cpd_df.index, [fold]])
        cpd_dfs.append(cpd_df)
    cpd_df = pd.concat(cpd_dfs, axis=0)
    # average CPD across folds
    cpd_df = cpd_df.groupby(["cluster_unique_ID"]).mean()
    return cpd_df


def reg_opt_Ridge(
    X_train, y_train, X_test, y_test, tol=1e-4, max_rounds=30, patience=20, return_as="best", verbose=False
):
    """
    Starts alpha=1.0 and doubles it every round.
    - If score >= 0: stops early when you’ve gone `patience` rounds in a row
    without improving R² by > tol.
    - If score  < 0: always keeps going until max_rounds.

    Returns
    -------
    best_alpha : float
        α that gave the highest R²
    best_score : float
        that highest R²
    best_round : int
        the iteration number (1-based) when best_alpha was found
    history : list of (alpha, score) tuples
    """
    alpha = 1.0
    best_alpha = alpha
    best_score = -np.inf
    best_rss = np.inf
    best_round = 0

    history = []
    no_improve_count = 0

    for round_idx in range(1, max_rounds + 1):
        model = Ridge(alpha=alpha, max_iter=10_000, random_state=0)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        residuals = y_test - model.predict(X_test)
        rss = np.sum(residuals**2)
        history.append((alpha, score, rss))
        if verbose:
            print(f"Round {round_idx:2d}: α = {alpha:.3e},  R² = {score:.4f}")

        # update best if we improved by more than tol
        if score > best_score + tol:
            best_score = score
            best_alpha = alpha
            best_rss = rss
            best_round = round_idx
            no_improve_count = 0

        else:
            # only count towards patience if score is non-negative
            if score >= 0:
                no_improve_count += 1
                if no_improve_count >= patience:
                    break

        alpha *= 2

    if verbose:
        print(f"→ Best α = {best_alpha:.3e} (round {best_round}) with R² = {best_score:.4f}")
        print("")

    if return_as == "history":
        return np.array(history).T
    elif return_as == "best":
        return best_alpha, best_score, best_rss
    else:
        raise ValueError(f"Unknown return_as: {return_as}")


# %%


def get_test_train_arrays(
    train_df,
    test_df,
    regressor_classes=["place_direction", "distance"],
    pd_bases=None,
    distance_metrics=None,
    dist_bases=None,
    goals=None,
    whiten_features=True,
    return_feature_names=False,
):
    """ """
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
            )
            X.append(x)
            n_features.append(x.shape[1])
        Xs.append(np.hstack(X))
    X_train, X_test = Xs
    Y_train, Y_test = train_df.spike_count.values, test_df.spike_count.values
    if whiten_features:
        scaler = StandardScaler()  # mean=0, std=1 per column
        scaler.fit(X_train)  # learn stats on train
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    if not return_feature_names:
        return X_train, X_test, Y_train, Y_test
    else:
        feature_names = np.repeat(regressor_classes, n_features)
        return (X_train, X_test, Y_train, Y_test), feature_names


def get_regressor_class(df, rtype, pd_bases=None, distance_metrics=None, dist_bases=None, goals=None):
    """
    rtype: regressor class name
    """
    if rtype == "place_direction":
        assert pd_bases is not None
        place_directions = (
            df[[("maze_position", "simple"), ("cardinal_movement_direction", "")]].apply(tuple, axis=1).values
        )
        r = pd_bases.loc[place_directions].values  # n_samples x n_place_direction_bases

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
    return nav_rates_df
