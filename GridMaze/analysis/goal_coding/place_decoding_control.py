"""
Is goal decoding just place decoding?
Control analysis: neurons -> decoded dist over places --> decode goal (all cv)
@peterdoohan
"""

# %% Imports
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


from GridMaze.maze import representations as mr
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import folds
from GridMaze.analysis.goal_coding import decoding_utils as du


# %% Global Variables

from GridMaze.paths import RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "goal_decoding" / "place_decoding_control"
if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %% plotting results


# %% summary function


def get_spatial_goal_decoding_control_summary(save=False, verbose=True):
    # load cached results if already processed
    save_path = RESULTS_DIR / "event_aligned_decoding_summary.parquet"
    if save_path.exists() and not save:
        if verbose:
            print(f"Loading existing results from {save_path}")
        return pd.read_parquet(save_path)
    if verbose:
        print("Loading sessions ...")
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names="all",
        days_on_maze="late",
        with_data=[
            "navigation_df",
            "navigation_spike_counts_df",
            "cluster_metrics",
            "trials_df",
        ],
        must_have_data=True,
    )
    dfs = []
    failed_sessions = []
    for session in sessions:
        if verbose:
            print(session.name)
        try:
            results_df = get_session_spatial_goal_decoding_control(
                session,
                verbose=verbose,
            )
            dfs.append(results_df)
        except Exception as e:
            if verbose:
                print(f"Failed to process session {session.name}: {e}")
            failed_sessions.append(session.name)
            continue
    summary_df = pd.concat(dfs, axis=0)
    summary_df.reset_index(drop=True, inplace=True)
    if save:
        if verbose:
            print(f"Saving results to {save_path}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_parquet(save_path)
    return summary_df


# %% session level function


def get_session_spatial_goal_decoding_control(
    session,
    resolution=0.5,
    include_multi_units=True,
    window=(-10, 10),
    goal_stratified_validation=True,
    zscore_features=True,
    all_goals_in_test_data=True,
    alpha="opt",
    n_jobs=-1,
    verbose=True,
):
    """ """
    # get downsampled input data containing behavioural info and spike data
    goals = session.goals
    input_data = du.get_place_decoding_input_data(session, resolution, include_multi_units, window, permuted=False)
    simple_maze = session.simple_maze()
    cue_timepoints = np.sort(input_data.event_aligned_bin.cue.dropna().unique())
    reward_timepoints = np.sort(input_data.event_aligned_bin.reward.dropna().unique())
    # organise trials into test-train folds
    folds_df = folds.get_folds_df(session, goal_stratified_validation, return_unique_IDs=True)
    # predict plce/place_direction probabilities from spike counts (for control conditions)
    spatial_probs_dfs = []
    for spatial_output in ["place", "place_direction"]:
        if verbose:
            print(f"Predicting {spatial_output} probabilities from spikes")
        spatial_probs_df = get_predicted_spatial(
            input_data,
            folds_df,
            simple_maze,
            input_type="spikes",
            output_type=spatial_output,
            training_trial_phases=["navigation"],
            n_jobs=n_jobs,
            verbose=False,
        )
        spatial_probs_dfs.append(spatial_probs_df)
    input_data = pd.concat([input_data] + spatial_probs_dfs, axis=1)
    # run xvaled decoding for each condition aross folds
    _folds = folds_df.columns.levels[0].unique()
    if n_jobs:
        fold_results = Parallel(n_jobs=n_jobs)(
            delayed(_process_fold)(
                input_data,
                folds_df,
                fold,
                cue_timepoints,
                reward_timepoints,
                all_goals_in_test_data,
                goals,
                alpha,
                zscore_features,
                verbose,
            )
            for fold in _folds
        )
    else:
        fold_results = [
            _process_fold(
                input_data,
                folds_df,
                fold,
                cue_timepoints,
                reward_timepoints,
                all_goals_in_test_data,
                goals,
                alpha,
                zscore_features,
                verbose,
            )
            for fold in _folds
        ]
    # combine across folds
    all_results = []
    for fold_result in fold_results:
        all_results.extend(fold_result)
    results_df = pd.concat(all_results, axis=0)
    results_df.reset_index(drop=True, inplace=True)
    # add session info
    results_df["subject_ID"] = session.subject_ID
    results_df["maze_name"] = session.maze_name
    results_df["day_on_maze"] = session.day_on_maze
    results_df["goal_subset"] = session.goal_subset
    return results_df


def _process_fold(
    input_data,
    folds_df,
    fold,
    cue_timepoints,
    reward_timepoints,
    all_goals_in_test_data,
    goals,
    alpha,
    zscore_features,
    verbose,
):
    """ """
    if verbose:
        print(fold)
    # decode goal from spikes, place_probs and place_direction_probs
    fold_df = folds_df[fold]
    test_trials = [t for t in fold_df.test.values.flatten() if isinstance(t, str)]
    train_trials = [t for t in fold_df.train.values.flatten() if isinstance(t, str)]
    train_df = input_data[input_data.trial_unique_ID.isin(train_trials)]
    test_df = input_data[input_data.trial_unique_ID.isin(test_trials)]
    fold_results = []
    for event, timepoints in zip(["cue", "reward"], [cue_timepoints, reward_timepoints]):
        for t in timepoints:
            _train_df = train_df[train_df.event_aligned_bin[event] == t]
            _test_df = test_df[test_df.event_aligned_bin[event] == t]
            if _train_df.empty or _test_df.empty:
                continue
            y_train, y_test = _train_df.goal.values, _test_df.goal.values
            unique_train = np.unique(y_train)
            if all_goals_in_test_data:
                if len(np.setdiff1d(goals, unique_train)) > 0:
                    continue  # need all goals in training data
            if len(unique_train) < 2:
                continue  # need at least two goals
            n_test_samp = _test_df.shape[0]
            res = pd.DataFrame(
                {
                    ("aligned_event", ""): np.repeat(event, n_test_samp),
                    ("timepoint", ""): np.repeat(t, n_test_samp),
                    ("trial_unique_ID", ""): _test_df.trial_unique_ID.values,
                    ("steps_to_goal", ""): _test_df.steps_to_goal.future.values,
                    ("trial_phase", ""): _test_df.trial_phase.values,
                    ("true_goal", ""): y_test,
                }
            )
            # predict goal from spikes or spatial probs
            feature_sets = ["spike_count", "place_prob", "place_direction_prob"]
            # get optimal regularisation
            if alpha == "opt":
                opt_alphas = get_feature_opt_alphas(_train_df, fold_df, zscore_features=zscore_features, verbose=False)
            else:
                opt_alphas = {f: alpha for f in feature_sets}
            # get opt regularsation for each feature set?
            for feature_set in feature_sets:
                X_train, X_test = _train_df[feature_set].values, _test_df[feature_set].values
                decoder = LogisticRegression(
                    penalty="l2", C=opt_alphas[feature_set], max_iter=10000, random_state=0, class_weight="balanced"
                )
                if zscore_features:  # zscore features
                    scaler = StandardScaler()  # mean=0, std=1 per column
                    scaler.fit(X_train)  # learn stats on train
                    X_train = scaler.transform(X_train)
                    X_test = scaler.transform(X_test)
                decoder.fit(X_train, y_train)
                y_pred = decoder.predict(X_test)
                res[("predicted_goal", feature_set)] = y_pred
                res[("accuracy", feature_set)] = (y_pred == y_test).astype(int)  # eval
            fold_results.append(res)
    return fold_results


def get_feature_opt_alphas(
    _train_df, fold_df, n_folds=5, zscore_features=True, reg_range=np.logspace(-4, 4, 20), verbose=False
):
    """ """
    v_df = fold_df.train
    v_cols = v_df.columns.values
    feature_sets = ["spike_count", "place_prob", "place_direction_prob"]
    results = np.zeros((len(v_cols), len(feature_sets), len(reg_range)))
    if v_df.shape[1] > 1:
        v_folds = v_cols
        for i, v in enumerate(v_folds):
            if verbose:
                print(f"v_fold: {v}")
            val_trials = v_df[[col for col in v_folds if col != v]].stack().dropna().values
            test_trials = v_df[v].dropna().values
            val_df = _train_df[_train_df.trial_unique_ID.isin(val_trials)]
            test_df = _train_df[_train_df.trial_unique_ID.isin(test_trials)]
            y_val, y_test = val_df.goal.values, test_df.goal.values
            for j, feature_set in enumerate(feature_sets):
                if verbose:
                    print(feature_set)
                X_val, X_test = val_df[feature_set].values, test_df[feature_set].values
                if zscore_features:
                    scaler = StandardScaler()
                    scaler.fit(X_val)
                    X_val = scaler.transform(X_val)
                    X_test = scaler.transform(X_test)
                alphas, accs = search_reg(
                    X_val, X_test, y_val, y_test, verbose=verbose, reg_range=reg_range, return_as="history"
                )
                results[i, j, :] = accs
    opt_regs = reg_range[results.mean(0).argmax(1)]
    return {f: alpha for f, alpha in zip(feature_sets, opt_regs)}


def search_reg(
    X_train,
    X_test,
    y_train,
    y_test,
    reg_range=np.logspace(-4, 4, 20),
    return_as="best",
    verbose=False,
):
    """
    CV search for optimal regulaisation strength (in training data)
    """
    best_alpha = None
    best_acc = 0
    history = []
    for alpha in reg_range:
        model = LogisticRegression(penalty="l2", C=alpha, max_iter=10_000, random_state=0, class_weight="balanced")
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
        if verbose:
            print(f" α = {alpha:.3e},  acc = {acc:.4f}")
        history.append((alpha, acc))
    if return_as == "best":
        return best_alpha, best_acc
    elif return_as == "history":
        return np.array(history).T
    else:
        raise ValueError(f"Unknown return_as: {return_as}. Must be 'best' or 'history'.")


# %%
def get_predicted_spatial(
    input_data,
    folds_df,
    simple_maze,
    input_type="spikes",
    output_type="place",
    training_trial_phases=["navigation"],
    n_jobs=False,
    verbose=True,
):
    """
    From some input_data, and folds_df dataframes, preform cross-validated prediction
    of place_direction from spike counts (w/ Logisitic Rergression classifier).

    Outputs the neural representation of place direction in the data as
    a probability distribution over the place directions or just place.

    W/ automatic regularisation optimisation
    """
    if output_type == "place_direction":
        # precompute all place_directions ("A1_N")
        all_features = mr.get_maze_place_direction_pairs(simple_maze)
        all_features = ["_".join(x) for x in all_features]

        # add place_direction column to input_data
        input_data[("place_direction", "")] = input_data.apply(
            lambda x: f"{x[("maze_position", "simple")]}_{x[("cardinal_movement_direction", "")]}", axis=1
        )
    elif output_type == "place":
        all_features = mr.get_maze_locations(simple_maze)
    else:
        raise ValueError(f"Unknown output type {output_type!r}")
    # get x-valed place-direction prob from spikes on each input_data sample
    _folds = folds_df.columns.levels[0].unique()
    if n_jobs:
        dfs = Parallel(n_jobs=n_jobs, verbose=False)(
            delayed(_process_predict_spatial_fold)(
                fold,
                input_data,
                folds_df,
                input_type,
                output_type,
                training_trial_phases,
                all_features,
                verbose,
            )
            for fold in _folds
        )
    else:
        dfs = [
            _process_predict_spatial_fold(
                fold, input_data, folds_df, input_type, output_type, training_trial_phases, all_features, verbose
            )
            for fold in _folds
        ]
    # combine folds and ensure index lines up with input_data
    probs_df = pd.concat(dfs, axis=0)
    probs_df.sort_index(axis=0, inplace=True)
    assert probs_df.index.equals(input_data.index)
    return probs_df


def _process_predict_spatial_fold(
    fold,
    input_data,
    folds_df,
    input_type,
    output_type,
    training_trial_phases,
    all_features,
    verbose,
):
    """ """
    if verbose:
        print(fold)
    probs_df = du.get_xvaled_decoding_df(
        input_data,
        folds_df,
        fold,
        training_trial_phases,
        input_type,
        output_type=output_type,
        df_engine="polars",
        verbose=verbose,
        return_as="probs_df",
    )
    features = probs_df.columns.levels[1].unique()
    # check for missing place_directions and add columns with value 0
    missing_features = set(all_features) - set(features)
    if len(missing_features) > 0:
        for missing_direction in missing_features:
            probs_df[(f"{output_type}_prob", missing_direction)] = 0
    return probs_df.sort_index(axis=1)
