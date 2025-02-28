"""
Regression analyses to test if mFC codes for where you are going to be in the future
"""

#%% Imports
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt
import seaborn as sns

from ..core import get_sessions as gs 
from ..core import get_clusters as gc
from ..core import filter as cf

#%% Global Variables
from ...paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

FRAME_RATE = 60

OPTIMAL_ALPHA_PATH = Path("../results/routes/now_next_route_cpd/optimal_alpha.json")
if not OPTIMAL_ALPHA_PATH.exists():
    print("Optimal alpha variable for ridge regression regularisation not found. Please run get_optimal_ridge_alpha() to calculate and save optimal alpha \n"
          "and reload the module before running regression analyses.")
else:
    with open(OPTIMAL_ALPHA_PATH, "r") as f:
        OPTIMAL_ALPHA = json.load(f)
#%% Proper Functions

def get_cross_subject_cpds(save=True, 
                           save_path=Path("../results/routes/now_next_route_cpd/cross_subject_cpd.json"),
                           plot=True,
                           ax=None):
    # check is data exists first
    if save_path.exists():
        with open(save_path, "r") as f:
            results = json.load(f)
            subject_cpd_now = results["cpd_now"]
            subject_cpd_next = results["cpd_next"]
    else: # calculate subject cpds
        subject_cpd_now, subject_cpd_next = [], []
        for subject in SUBJECT_IDS:
            cpd_nows, cpd_nexts = get_subject_cpds(subject, plot=False)
            subject_cpd_now.append(np.nanmean(cpd_nows))
            subject_cpd_next.append(np.nanmean(cpd_nexts))
        if save:
            results = {"cpd_now": subject_cpd_now, "cpd_next": subject_cpd_next}
            with open(save_path, "w") as f:
                json.dump(results, f)
    if plot:
        if ax is None:
            f, ax = plt.subplots(1,1, figsize=(2,4))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(0, color='black', linestyle='--', alpha=0.2)
        ax.set_ylabel("CPD")
        ax.set_xticks([0,1])
        ax.set_xticklabels(["Now", "Next"])
        ax.set_ylim(-0.1, 0.1)
        ax.set_xlabel("Route")
        sns.swarmplot(x=[0]*len(subject_cpd_now), y=subject_cpd_now, ax=ax, size=5, alpha=0.5)
        sns.swarmplot(x=[1]*len(subject_cpd_next), y=subject_cpd_next, ax=ax, size=5, alpha=0.5)

    return subject_cpd_now, subject_cpd_next


def get_subject_cpds(subject, 
                        maze_names="all", 
                        smooth_SD=False,
                        cross_validate_via="routes",
                        alpha=OPTIMAL_ALPHA,
                        cpd_nan_range=(-1,1),
                        min_routes=2,
                        max_routes=4,
                        optimal_routes=True,
                        moving_only=True,
                        plot=True):
    """ """
    sessions = gs.get_maze_sessions(subject_IDs=[subject], maze_names=maze_names, days_on_maze="late", 
                                    with_data=["navigation_df", "navigation_spike_rates_df", "navigation_routes_df", "cluster_metrics"],
                                    must_have_data=True)
    cpd_nows, cpd_nexts = [], []
    for session in sessions:
        cpd_now, cpd_next = get_now_and_next_route_cpds(session, 
                                                        smooth_SD, 
                                                        cross_validate_via, 
                                                        alpha, 
                                                        cpd_nan_range, 
                                                        min_routes,
                                                        max_routes,
                                                        optimal_routes,
                                                        moving_only,
                                                        plot=False)
        cpd_nows.extend(cpd_now)
        cpd_nexts.extend(cpd_next)
    if plot:
        print(f"Could not calculate CPD values for {np.isnan(cpd_nows).mean()*100}% of clusters")
        _plot_cpd_2D_hist(cpd_nows, cpd_nexts)
    return cpd_nows, cpd_nexts


def _plot_cpd_2D_hist(cpd_nows, cpd_nexts, ax=None):
    """"""
    if ax is None:
        f, ax = plt.subplots(1,1, figsize=(5,4))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("CPD Current Route")
    ax.set_ylabel("CPD Next Route")
    ax.axhline(0, color='black', linestyle='--', alpha=0.2)
    ax.axvline(0, color='black', linestyle='--', alpha=0.2)
    # plot
    sns.histplot(x=cpd_nows, y=cpd_nexts, bins=50, cbar=True, cbar_kws=dict(shrink=.75), ax=ax)
    return


def get_now_and_next_route_cpds(session, 
                                smooth_SD=False, 
                                cross_validate_via="frames", 
                                alpha=1, 
                                nan_range=(-1,1), 
                                min_routes=2,
                                max_routes=4,
                                optimal_routes=True,
                                moving_only=True,
                                plot=False):
    """ """
    if alpha is None:
        alpha = get_optimal_ridge_alpha(session, smooth_SD=smooth_SD)
    keep_clusters = gc.filter_clusters(session.cluster_metrics, session.session_info, return_unique_IDs=True)
    navigation_df = session.navigation_df
    navigation_routes_df = session.navigation_routes_df
    navigation_rates_df = session.navigation_spike_rates_df
    cpd_nows= []
    cpd_nexts = []
    for cluster in keep_clusters:
        cluster_rates = navigation_rates_df.xs(cluster, level=1, axis=1).firing_rate.values
        if smooth_SD:
            cluster_rates = gaussian_filter1d(cluster_rates, smooth_SD * FRAME_RATE)
        nav_rates_df = pd.concat([navigation_df, navigation_routes_df.reset_index(drop=True)], axis=1)
        nav_rates_df[("firing_rate", "")] = cluster_rates
        nav_rates_df = _filter_data(nav_rates_df, min_routes, max_routes, optimal_routes, moving_only)
        if cross_validate_via == "frames":
            results_df = get_frame_cross_validated_ssres(nav_rates_df, alpha)
        elif cross_validate_via == "trials":
            results_df = get_trial_cross_validated_ssres(nav_rates_df, alpha)
        elif cross_validate_via == "routes":
            results_df = get_route_cross_validated_ssres(nav_rates_df, alpha)
        cpd_now, cpd_next = _get_cpds(results_df)
        if cpd_now < nan_range[0] or cpd_next < nan_range[0] or cpd_now > nan_range[1] or cpd_next > nan_range[1]:
            cpd_now = np.nan
            cpd_next = np.nan
        cpd_nows.append(cpd_now)
        cpd_nexts.append(cpd_next)
    if plot:
        _plot_cpd_2D_hist(cpd_nows, cpd_nexts)
    return cpd_nows, cpd_nexts


def _get_cpds(residuals_df):
    """ """
    cpd_nows, cpd_nexts = [], []
    for fold in residuals_df.itertuples():
        cpd_now = (fold.SSnext - fold.SSfull) / fold.SSnext
        cpd_next = (fold.SSnow - fold.SSfull) / fold.SSnow
        cpd_nows.append(cpd_now)
        cpd_nexts.append(cpd_next)
    return np.mean(cpd_nows), np.mean(cpd_nexts)    

def get_frame_cross_validated_ssres(nav_rates_df, alpha):
    """"""
    X_now = route_id2onehot(nav_rates_df.route['r'].values)
    X_next = route_id2onehot(nav_rates_df.route['r+1'].values)
    y = nav_rates_df.firing_rate.values
    kf = KFold(n_splits=5, shuffle=True)
    results=[]
    for train_index, test_index in kf.split(X_now):
        # Split data into train and test sets
        X_train_now, X_test_now = X_now[train_index], X_now[test_index]
        X_train_next, X_test_next = X_next[train_index], X_next[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ssres_full, ssres_now, ssres_next = _run_regression_models(X_test_now, X_train_now, X_test_next, X_train_next, y_test, y_train, alpha)
        results.append(
            {"SSfull": ssres_full, "SSnow": ssres_now, "SSnext": ssres_next}
        )
    return pd.DataFrame(results)


def get_route_cross_validated_ssres(nav_rates_df, alpha):
    """ """
    validation_folds_df = get_route_stratified_folds_df(nav_rates_df)
    results = []
    for fold in validation_folds_df.columns.get_level_values(0).unique():
        fold_df = validation_folds_df[fold]
        test_trials = fold_df.test.dropna().values.flatten()
        train_trials = fold_df.train.unstack().dropna().values
        test_df = nav_rates_df[nav_rates_df.trial.isin(test_trials)]
        train_df = nav_rates_df[nav_rates_df.trial.isin(train_trials)]
        X_test_now = route_id2onehot(test_df.route['r'].values)
        X_train_now = route_id2onehot(train_df.route['r'].values)
        X_test_next = route_id2onehot(test_df.route['r+1'].values)
        X_train_next = route_id2onehot(train_df.route['r+1'].values)
        y_test = test_df.firing_rate.values
        y_train = train_df.firing_rate.values
        ssres_full, ssres_now, ssres_next = _run_regression_models(X_test_now, X_train_now, X_test_next, X_train_next, y_test, y_train, alpha)
        results.append(
            {"SSfull": ssres_full, "SSnow": ssres_now, "SSnext": ssres_next}
        )
    return pd.DataFrame(results)


def get_trial_cross_validated_ssres(nav_rates_df, alpha):
    """ """
    valid_trials = nav_rates_df.trial.unique()
    validation_folds_df = cf.get_trial_validation_folds_df(valid_trials, splits=5)
    results = []
    for fold in validation_folds_df.columns.get_level_values(0).unique():
        fold_df = validation_folds_df[fold]
        test_trials = fold_df.test.dropna().values
        train_trials = fold_df.train.dropna().values
        test_df = nav_rates_df[nav_rates_df.trial.isin(test_trials)]
        train_df = nav_rates_df[nav_rates_df.trial.isin(train_trials)]
        X_test_now = route_id2onehot(test_df.route['r'].values)
        X_train_now = route_id2onehot(train_df.route['r'].values)
        X_test_next = route_id2onehot(test_df.route['r+1'].values)
        X_train_next = route_id2onehot(train_df.route['r+1'].values)
        y_test = test_df.firing_rate.values
        y_train = train_df.firing_rate.values
        ssres_full, ssres_now, ssres_next = _run_regression_models(X_test_now, X_train_now, X_test_next, X_train_next, y_test, y_train, alpha)
        results.append(
            {"SSfull": ssres_full, "SSnow": ssres_now, "SSnext": ssres_next}
        )
    return pd.DataFrame(results)


def _run_regression_models(X_test_now, X_train_now, X_test_next, X_train_next, y_test, y_train, alpha):
    """Run Ridge regression models and calculate SSres for full and partial models."""
    
    # Full model
    X_train_full = np.hstack((X_train_now, X_train_next))
    X_test_full = np.hstack((X_test_now, X_test_next))
    model_full = Ridge(alpha=alpha)
    model_full.fit(X_train_full, y_train)
    y_pred_full = model_full.predict(X_test_full)
    ssres_full = np.sum((y_test - y_pred_full) ** 2)

    # Now route only model
    model_now = Ridge(alpha=alpha)
    model_now.fit(X_train_now, y_train)
    y_pred_now = model_now.predict(X_test_now)
    ssres_now = np.sum((y_test - y_pred_now) ** 2)

    # Next route only model
    model_next = Ridge(alpha=alpha)
    model_next.fit(X_train_next, y_train)
    y_pred_next = model_next.predict(X_test_next)
    ssres_next = np.sum((y_test - y_pred_next) ** 2)

    return ssres_full, ssres_now, ssres_next




def _filter_data(nav_rates_df, min_routes=2, max_routes=4, optimal_routes=True, moving_only=True):
    """ 
    Take take timepoints where a current route and next route are defined (route['r'] and route['r+1'] != np.nan),
    plus any additional filtering criteria.
    """
    filter_masks = []
    # current route and next route are defined
    filter_masks.append(nav_rates_df.route['r'].notna())
    filter_masks.append(nav_rates_df.route['r+1'].notna())
    if min_routes:
        filter_masks.append(nav_rates_df.n_routes.ge(min_routes))
    if max_routes:
        filter_masks.append(nav_rates_df.n_routes.le(max_routes))
    if optimal_routes:
        filter_masks.append(nav_rates_df.optimal_route)
    if moving_only:
        filter_masks.append(nav_rates_df.moving)
    combined_mask = np.logical_and.reduce(filter_masks)
    return nav_rates_df[combined_mask]


def get_optimal_ridge_alpha(smooth_SD=False, alphas=np.logspace(-4, 4, 10), save=True):
    """
    
    """
    sessions = gs.get_maze_sessions(subject_IDs="all",  # get a session from each subject (may as well be late on maze_1)
                                    maze_names=["maze_1"], 
                                    days_on_maze=[10], 
                                    with_data=["navigation_df", "navigation_spike_rates_df", "navigation_routes_df", "cluster_metrics"])
    subject_alphas = []
    for session in sessions:
        print(f"Calculating optimal alpha for {session.subject_ID}")
        keep_clusters = gc.filter_clusters(session.cluster_metrics, session.session_info, return_unique_IDs=True)
        navigation_df = session.navigation_df
        navigation_routes_df = session.navigation_routes_df
        navigation_rates_df = session.navigation_spike_rates_df
        optimal_alphas = []
        for cluster in keep_clusters:
            cluster_rates = navigation_rates_df.xs(cluster, level=1, axis=1).firing_rate.values
            if smooth_SD:
                cluster_rates = gaussian_filter1d(cluster_rates, smooth_SD * FRAME_RATE)
            nav_rates_df = pd.concat([navigation_df, navigation_routes_df.reset_index(drop=True)], axis=1)
            nav_rates_df[("firing_rate", "")] = cluster_rates
            # filter data going into regression
            nav_rates_df = _filter_data(nav_rates_df)
            optimal_alphas.append(_optimal_alpha_cross_val(nav_rates_df, alphas, cross_validated_via="routes"))
        subject_alphas.append(np.median(optimal_alphas))
    a = np.median(subject_alphas)
    if save:
        with open(OPTIMAL_ALPHA_PATH, "w") as f:
            json.dump({a}, f)
        with open(OPTIMAL_ALPHA_PATH, "w") as outfile:
            outfile.write(json.dumps(a, indent=4))
    return a


def _optimal_alpha_cross_val(nav_rates_df, alphas, cross_validated_via="routes"):
    """
    """
    if not cross_validated_via == "routes":
        raise ValueError("Only cross validation via routes is currently supported")
    validation_folds_df = get_route_stratified_folds_df(nav_rates_df)
    x_val_r2s = []
    for alpha in alphas:
        r2s = []
        for fold in validation_folds_df.columns.get_level_values(0):
            # get train and test data
            fold_df = validation_folds_df[fold]
            test_trials = fold_df.test.dropna().values.flatten()
            train_trials = fold_df.train.unstack().dropna().values
            test_trials = fold_df.test.dropna().values.flatten()
            train_trials = fold_df.train.unstack().dropna().values
            test_df = nav_rates_df[nav_rates_df.trial.isin(test_trials)]
            train_df = nav_rates_df[nav_rates_df.trial.isin(train_trials)]
            X_test_now = route_id2onehot(test_df.route['r'].values)
            X_train_now = route_id2onehot(train_df.route['r'].values)
            X_test_next = route_id2onehot(test_df.route['r+1'].values)
            X_train_next = route_id2onehot(train_df.route['r+1'].values)
            X_full_test = np.hstack((X_test_now, X_test_next))
            X_full_train = np.hstack((X_train_now, X_train_next))
            y_test = test_df.firing_rate.values
            y_train = train_df.firing_rate.values
            # fit model
            model = Ridge(alpha=alpha)
            model.fit(X_full_train, y_train)
            score = model.score(X_full_test, y_test) #R2 score
            r2s.append(score)
        x_val_r2s.append( np.mean(r2s))
    return alphas[np.argmax(x_val_r2s)]



def _optimal_alpha(nav_rates_df, alphas):
    """
    """
    X_now = route_id2onehot(nav_rates_df.route['r'].values)
    X_next = route_id2onehot(nav_rates_df.route['r+1'].values)
    X_full = np.hstack((X_now, X_next))
    y = nav_rates_df.firing_rate.values
    param_grid = {'alpha': alphas}
    ridge = Ridge()
    kf = KFold(n_splits=4, shuffle=True)
    grid_search = GridSearchCV(ridge, param_grid, cv=kf, scoring='r2')
    grid_search.fit(X_full, y)
    return grid_search.best_params_['alpha']


def route_id2onehot(r, n_routes=10):
    """
    """
    route_ids = ["non_route"]+[f"route_{i}"for i in range(n_routes)]
    enc = OneHotEncoder(categories=[route_ids], sparse_output=False, drop="first")
    onehot = enc.fit_transform(np.array(r).reshape(-1,1))
    return onehot


def get_route_stratified_folds_df(nav_rates_df):
    """
    """
    now_route2trial = nav_rates_df.groupby([('route',"r")]).trial.apply(lambda x: np.unique(x))
    next_route2trial = nav_rates_df.groupby([('route',"r+1")]).trial.apply(lambda x: np.unique(x))
    #only keep routes with more than 1 trial this garantees training and test data will include the same set of routes
    route2trial = now_route2trial[(now_route2trial.apply(len) > 1)&(next_route2trial.apply(len) > 1)]     
    route2trial = route2trial.apply(lambda x: np.random.choice(x, size=len(x), replace=False))
    route2trial_df = route2trial.apply(pd.Series)
    fold_dfs = []
    for c in route2trial_df.columns:
        test_df = route2trial_df[c]
        train_df = route2trial_df.drop(columns=c)
        fold_df = pd.concat([train_df, test_df], axis=1)
        fold_df.columns = pd.MultiIndex.from_product([[f"fold_{c}"], ["train"], train_df.columns]).append(pd.MultiIndex.from_product([[f"fold_{c}"], ["test"], [c]]))
        fold_dfs.append(fold_df)
    return pd.concat(fold_dfs, axis=1)