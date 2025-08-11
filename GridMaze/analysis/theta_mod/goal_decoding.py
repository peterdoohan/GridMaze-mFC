"""
Can we decode the goal between if we known about theta phase. Either train and test on different theta-phases
OR do we need to know all theta phases to decode the goal?
@peterdoohan
"""

# %% Imports
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from GridMaze.analysis.goal_coding import decoding_utils as du
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import convert


# %% Globs

FRAME_RATE = 60

# %%


def get_session_theta_phase_goal_decoding():
    """ """

    return


def get_session_theta_mod_goal_decoding(
    session, event="cue", resolution=0.5, window=(-3, 3), include_multi_units=True, zscore=True
):
    """
    Within session compare goal decoding from feature = n_neurons OR
    features = n_neurons x n_theta_phases. Need xval per fold opt regularisation
    bc of different number of features
    """
    input_data = get_input_data(session, event, resolution, window, include_multi_units)
    timepoints = sorted(input_data.event_aligned_time[event].unique())
    folds_df = du.get_folds_df(session, goal_stratified=True, return_unique_IDs=True, n_test_trials=None)
    results = []
    for fold in folds_df.columns.levels[0].unique():
        print(fold)
        fold_df = folds_df[fold]
        test_trials = [t for t in fold_df.test.values.flatten() if isinstance(t, str)]
        train_trials = [t for t in fold_df.train.values.flatten() if isinstance(t, str)]
        train_df = input_data[input_data.trial_unique_ID.isin(train_trials)]
        test_df = input_data[input_data.trial_unique_ID.isin(test_trials)]
        fold_results = Parallel(n_jobs=-1, verbose=10)(
            delayed(_process_timepoint)(train_df, test_df, event, t, zscore, fold_df, fold) for t in timepoints
        )
    return fold_results
    results_df = pd.concat(results, axis=0)
    return results_df


def _process_timepoint(train_df, test_df, event, t, zscore, fold_df, fold):
    """pull out of session level fn for parallelisation"""
    _train_df = train_df[train_df.event_aligned_time[event] == t]
    _test_df = test_df[test_df.event_aligned_time[event] == t]
    if _train_df.empty or _test_df.empty:
        return  # rare cases when no trials for that timepoint (eg, end of session trial)
    y_train, y_test = _train_df.goal.values, _test_df.goal.values
    # all spikes data (collect spikes across theta phases), shape = n_samples, n_neurons
    Xall_train, Xall_test = [df.spike_count.T.groupby(level=0).sum().T.values for df in [_train_df, _test_df]]
    # theta spikes data (shape = n_samples, n_neurons, n_theta_phases)
    Xtheta_train, Xtheta_test = [df.spike_count.values for df in [_train_df, _test_df]]
    # zscore features
    if zscore:
        norm_Xs = []
        for X_train, X_test in zip([Xall_train, Xtheta_train], [Xall_test, Xtheta_test]):
            scaler = StandardScaler()  # mean=0, std=1 per column
            scaler.fit(X_train)  # learn stats on train
            norm_Xs.append((scaler.transform(X_train), scaler.transform(X_test)))
        (Xall_train, Xall_test), (Xtheta_train, Xtheta_test) = norm_Xs
    # get optimal regularisation under each condition
    alpha_all, alpha_theta = _get_opt_regularisation(train_df, fold_df, zscore=zscore)
    # predict goal from feature sets: Xall, Xtheta
    timepoint_results = []
    for (X_train, X_test), alpha, label in zip(
        [(Xall_train, Xall_test), (Xtheta_train, Xtheta_test)], [alpha_all, alpha_theta], ["all", "theta"]
    ):
        model = LogisticRegression(max_iter=10000, random_state=0, class_weight="balanced", C=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = np.mean(y_pred == y_test)
        timepoint_results.append({"fold": fold, "timepoint": t, "feature_set": label, "alpha": alpha, "accuracy": acc})
    return timepoint_results


def _get_opt_regularisation(train_df, fold_df, zscore=True):
    """ """
    _fold_df = fold_df["train"]
    _folds = _fold_df.columns
    reg_dfs = []
    for val_fold in _folds:
        test_trials = _fold_df[val_fold].dropna().values
        val_trials = _fold_df[[f for f in _folds if f != val_fold]].stack().dropna().values
        test_df = train_df[train_df.trial_unique_ID.isin(test_trials)]
        val_df = train_df[train_df.trial_unique_ID.isin(val_trials)]
        y_val, y_test = val_df.goal.values, test_df.goal.values
        Xall_val, Xall_test = [df.spike_count.T.groupby(level=0).sum().T.values for df in [val_df, test_df]]
        Xtheta_val, Xtheta_test = [df.spike_count.values for df in [val_df, test_df]]
        # search for optimal regularisation for Xall and Xtheta
        for (X_train, X_test), label in zip([(Xall_val, Xall_test), (Xtheta_val, Xtheta_test)], ["all", "theta"]):
            # zscore features
            if zscore:
                scaler = StandardScaler()  # mean=0, std=1 per column
                scaler.fit(X_train)  # learn stats on train
                X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
            search_df = _search_regularisations(X_train, X_test, y_val, y_test)
            search_df["fold"] = val_fold
            search_df["feature_set"] = label
            reg_dfs.append(search_df)
    reg_df = pd.concat(reg_dfs, axis=0)
    # get alpha value with best average accuracy across validation splits of training data
    best_regs = reg_df.groupby(["feature_set", "alpha"]).accuracy.mean().unstack(level=0).idxmax()
    return best_regs.values  # opt all, opt theta


def _search_regularisations(X_train, X_test, y_train, y_test, reg_range=np.logspace(-4, 4, 10)):
    """ """
    search_res = []
    for alpha in reg_range:
        model = LogisticRegression(max_iter=10_000, random_state=0, class_weight="balanced", C=alpha)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        acc = np.mean(y_predict == y_test)
        search_res.append({"alpha": alpha, "accuracy": acc})

    return pd.DataFrame(search_res)


# %% Functions


def get_input_data(
    session,
    event="cue",
    resolution=0.5,
    window=(-10, 10),
    include_multi_units=True,
):
    """
    Returns a dataframe with spike counts aligned to event (cue & reward) times.
    """
    # load data
    session_info = session.session_info
    trials_df = session.trials_df
    navigation_df = session.navigation_df
    theta_spike_counts_df = session.navigation_theta_spike_counts_df.reset_index(drop=True)
    # filter clusters
    keep_clusters = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=include_multi_units,
    )
    theta_spike_counts_df = theta_spike_counts_df[
        theta_spike_counts_df.columns[theta_spike_counts_df.columns.get_level_values(1).isin(keep_clusters)]
    ]
    # get rates aligned to event
    frames_before, frames_after = int(window[0] * FRAME_RATE), int(window[1] * FRAME_RATE)
    event_times = trials_df.set_index("trial").time[event]
    trial2goal = trials_df.set_index("trial").goal
    nav_info_dfs, spike_count_dfs = [], []
    for trial, event_time in event_times.items():
        event_frame = (navigation_df.time - event_time).abs().argmin()
        nav_aligned_df = navigation_df.iloc[event_frame + frames_before : event_frame + frames_after].reset_index(
            drop=True
        )
        spikes_aligned_df = theta_spike_counts_df.iloc[
            event_frame + frames_before : event_frame + frames_after
        ].reset_index(drop=True)
        # downsample to speficied resolution
        ds_nav_aligned_df, ds_spikes_aligned_df = ds.downsample_nav_spikes_data(
            nav_aligned_df, spikes_aligned_df, resolution
        )
        # add event aligned time info
        timepoints = np.arange(window[0], window[1], resolution)
        if len(timepoints) > ds_nav_aligned_df.shape[0]:
            # can happen for last trial in session (no more frames)
            timepoints = timepoints[: ds_nav_aligned_df.shape[0]]
        ds_nav_aligned_df[("event_aligned_time", event)] = timepoints
        # update distnace outside navigation where they are not defined (use shortest path
        # upcoming goal (event=cue) or shortest path to just visted goal (event=reward))
        ds_nav_aligned_df[("goal", "")] = trial2goal[trial]
        # update trial info so it is consistent across all aligned times
        ds_nav_aligned_df[("trial", "")] = trial
        ds_nav_aligned_df[("trial_unique_ID", "")] = convert.trial2trial_unique_ID(session_info, trial)
        nav_info_dfs.append(ds_nav_aligned_df)
        spike_count_dfs.append(ds_spikes_aligned_df)
    # combine over trials
    nav_info_df = pd.concat(nav_info_dfs, axis=0).reset_index(drop=True)
    spike_count_df = pd.concat(spike_count_dfs, axis=0).reset_index(drop=True)
    # combine nav_info and spike counts
    nav_info_df.columns = pd.MultiIndex.from_tuples([(*c, "") for c in nav_info_df.columns])
    event_aligned_nav_rates_df = pd.concat([nav_info_df, spike_count_df], axis=1)
    return event_aligned_nav_rates_df


# %% version from analysis/goal_decoding


def get_event_aligned_goal_decoding(
    session,
    event="cue",
    resolution=0.5,
    window=(-10, 10),
    goal_stratified_validation=True,
    n_test_trials=None,
    include_multi_units=True,
    whiten_features=True,
):
    """
    Chance is always 1 / n_goals.
    """
    input_data = du.get_event_aligned_input_data(session, event, resolution, window, include_multi_units)
    timepoints = sorted(input_data.event_aligned_time[event].unique())
    folds_df = du.get_folds_df(session, goal_stratified_validation, return_unique_IDs=True, n_test_trials=n_test_trials)
    results_dfs = []
    for fold in folds_df.columns.levels[0].unique():
        fold_df = folds_df[fold]
        test_trials = [t for t in fold_df.test.values.flatten() if isinstance(t, str)]
        train_trials = [t for t in fold_df.train.values.flatten() if isinstance(t, str)]
        train_df = input_data[input_data.trial_unique_ID.isin(train_trials)]
        test_df = input_data[input_data.trial_unique_ID.isin(test_trials)]
        decoder = LogisticRegression(penalty=None, max_iter=10000, random_state=0, class_weight="balanced")
        for t in timepoints:
            _train_df = train_df[train_df.event_aligned_time[event] == t]
            _test_df = test_df[test_df.event_aligned_time[event] == t]
            if _train_df.empty or _test_df.empty:
                continue  # rare cases when no trials for that timepoint (eg, end of session trial)
            X_train, y_train = _train_df.spike_count.values, _train_df.goal.values
            X_test, y_test = _test_df.spike_count.values, _test_df.goal.values
            if whiten_features:  # zscore features
                scaler = StandardScaler()  # mean=0, std=1 per column
                scaler.fit(X_train)  # learn stats on train
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
            # fit model
            decoder.fit(X_train, y_train)
            # out_df
            Gprobs = decoder.predict_proba(X_test)
            n_samples, n_goals = Gprobs.shape
            goals = list(decoder.classes_)
            df = pd.DataFrame(
                {
                    "timepoint": np.repeat(t, n_samples * n_goals),
                    "true_goal": np.repeat(y_test, n_goals),
                    "trial_unique_ID": np.repeat(_test_df.trial_unique_ID.values, n_goals),
                    "predicted_goal": np.tile(goals, n_samples),
                    "predicted_goal_prob": Gprobs.ravel(),
                }
            )
            df["fold"] = fold
            results_dfs.append(df)
    results_df = pd.concat(results_dfs, axis=0)
    results_df.reset_index(drop=True, inplace=True)
    return results_df
