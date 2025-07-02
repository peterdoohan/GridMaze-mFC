"""
Distance to goal decoding using logistic regression this time.
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from GridMaze.analysis.core import convert
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import folds
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.distance_to_goal import distributions as dd


# %% Global Variables

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "distance_to_goal" / "logreg_decoding"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)


# %% plot basic decoder


def plot_distance_decoding_probs(
    results_df, moving_only=True, maze_names=["maze_1", "maze_2", "rooms_maze"], log_prob=False, vmax=0.1, ax=None
):
    """ """
    if moving_only:
        df = results_df[results_df.moving]
    else:
        df = results_df.copy()
    if maze_names:
        df = df[df.maze_name.isin(maze_names)]
    # average probs over session distance bins
    p_df = df.groupby(["subject_ID", "distance_bin_mid"]).decoded_distance_prob.mean()
    # average over subjects
    p_df = p_df.groupby("distance_bin_mid").decoded_distance_prob.mean()  # [n_dist, n_dist]
    # plotting
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
    # plot heatmap
    data = p_df.values.T
    distances = p_df.index.values
    if log_prob:
        norm = LogNorm(vmin=0.001, vmax=vmax)
        cbar_kwargs = {"ticks": MaxNLocator(5), "format": "%.e", "shrink": 0.8}
    else:
        norm = None
        cbar_kwargs = {"shrink": 0.5}
    sns.heatmap(
        data,
        square=True,
        cmap="Greys",
        norm=norm,
        cbar_kws=cbar_kwargs,
        ax=ax,
        vmax=vmax,
    )
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, len(distances), 5) + 0.5)
    ax.set_xticklabels(np.round(distances[::5], 2), rotation=0)
    ax.set_yticks(np.arange(0, len(distances), 5) + 0.5)
    ax.set_yticklabels(np.round(distances[::5], 2))
    ax.set_xlabel("True distance")
    ax.set_ylabel("Decoded distance")
    # get argmax
    m_df = df.groupby(["subject_ID", "distance_bin_mid"]).decoded_distance_prob.mean()  # av samples in sessions
    # get max prob decoded distance in each session
    m_df = m_df.decoded_distance_prob.idxmax(axis=1).unstack().astype(float)
    # average over subjects
    m_df = m_df.groupby(["subject_ID"]).mean()
    # convert to distance bins
    m = m_df.values
    m = (m / m_df.columns.max()) * m.shape[1]
    # plot mean and sem across subjects
    mean = np.mean(m, axis=0) + 0.5  # add offsets to plot in middle of boxes
    sem = np.std(m, axis=0) / np.sqrt(m.shape[0]) + 0.5
    x = np.arange(len(mean)) + 0.5
    ax.plot(x, mean, color="royalblue", label="mean decoded distance")
    ax.fill_between(
        x,
        mean - sem,
        mean + sem,
        color="royalblue",
        alpha=0.3,
    )
    ax.plot([0, len(mean)], [0, len(mean)], color="royalblue", linestyle="--")


# %% populate basic decoding across all late sesssions


def load_decoding_results(subfolder="all_dist"):
    results_paths = list((RESULTS_DIR / "basic" / subfolder).iterdir())
    dfs = []
    for path in results_paths:
        df = pd.read_parquet(path)
        dfs.append(df)
    results_df = pd.concat(dfs, axis=0)
    return results_df


def populate_decoding_results(max_jobs=20, subfolder=None, max_distance=None, verbose=True):
    """ """

    def _process_session(session):
        if verbose:
            print(f"Processing session {session.name}")
        # set up save path
        if subfolder is None:
            save_path = RESULTS_DIR / "basic" / f"{session.name}.parquet"
        else:
            save_path = RESULTS_DIR / "basic" / subfolder / f"{session.name}.parquet"
        # reun decode with defualt settings
        results_df = decode_session_distance_to_goal(session, max_distance=max_distance, verbose=verbose)
        # add session info
        results_df[("subject_ID", "")] = session.subject_ID
        results_df[("maze_name", "")] = session.maze_name
        results_df[("day_on_maze", "")] = session.day_on_maze
        # save
        results_df.to_parquet(save_path)
        if verbose:
            print(f"Saved results for to {save_path}")

    # load sessions
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
    Parallel(n_jobs=max_jobs)(delayed(_process_session)(session) for session in sessions)

    if verbose:
        print("Finished populating decoding results ...")


# %% basic decoding


def decode_session_distance_to_goal(
    session,
    resolution=0.5,
    metric=("distance_to_goal", "geodesic"),
    include_multiunits=True,
    moving_only=False,
    max_steps_to_goal=30,
    bin_spacing=0.05,
    max_distance=None,
    bin_method="uniform",
    n_log_bins=30,
    balance_distances=False,
    n_folds=5,
    sqrt_spikes=True,
    standardise_spikes=True,
    alpha="opt",
    verbose=False,
):
    """ """
    # get input data
    input_data = get_input_data(
        session,
        resolution,
        metric,
        include_multiunits,
        moving_only,
        max_steps_to_goal,
        bin_spacing,
        bin_method,
        max_distance,
        n_log_bins,
        balance_distances,
    )
    distance_bin_mids = sorted(input_data.distance_bin_mid.unique())
    # set up output df
    results_df = pd.concat(
        [
            input_data[
                [
                    ("trial", ""),
                    ("time", ""),
                    ("moving", ""),
                    ("steps_to_goal", "future"),
                    ("distance_bin_mid", ""),
                    metric,
                ]
            ],
            pd.DataFrame(
                columns=pd.MultiIndex.from_product([["decoded_distance_prob"], distance_bin_mids]),
                index=input_data.index,
            ),
        ],
        axis=1,
    )
    # decode distance CV
    folds_df = folds.get_folds_df(session, goal_stratified=False, return_unique_IDs=True, n_folds=n_folds)
    _folds = folds_df.columns.get_level_values(0).unique()
    for fold in _folds:
        if verbose:
            print(fold)
        fold_df = folds_df[fold]
        # get optimal alpha, CV over training folds
        if alpha == "opt":
            opt_alpha = get_CV_alpha(
                input_data,
                fold_df,
                metric,
                sqrt_spikes=sqrt_spikes,
                standardise_spikes=standardise_spikes,
                return_as="best",
                verbose=verbose,
            )
        else:
            opt_alpha = alpha
        train_df, test_df = folds._get_test_train_dfs(input_data, fold_df)
        X_train, X_test = train_df.spike_count.values, test_df.spike_count.values
        if sqrt_spikes:
            X_train, X_test = np.sqrt(X_train), np.sqrt(X_test)
        if standardise_spikes:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        y_train, y_test = train_df.distance_bin_id.values, test_df.distance_bin_id.values
        # fit model
        model = LogisticRegression(penalty="l2", C=opt_alpha, max_iter=10_000, random_state=0, class_weight="balanced")
        model.fit(X_train, y_train)
        # predict
        y_prob = model.predict_proba(X_test)
        results_df.loc[test_df.index, "decoded_distance_prob"] = y_prob
    return results_df


def get_CV_alpha(
    input_data,
    fold_df,
    metric,
    output="max",
    sqrt_spikes=True,
    standardise_spikes=True,
    return_as="best",
    verbose=False,
):
    """ """
    distance_bin_mids = np.array(sorted([b.mid for b in input_data.distance_bin.unique()]))
    # split training data into folds
    val_df = fold_df["train"]
    _vfolds = val_df.columns.values
    val_results = []
    for i, vfold in enumerate(_vfolds):
        if verbose:
            print(f"    Validation fold {i}")
        # index input data for validation test and train
        test_df = input_data[input_data.trial_unique_ID.isin(val_df[vfold].values)]
        train_df = input_data[input_data.trial_unique_ID.isin(val_df.drop(columns=vfold).unstack().dropna().values)]
        # get X and y
        X_train, X_test = train_df.spike_count.values, test_df.spike_count.values
        if sqrt_spikes:
            X_train, X_test = np.sqrt(X_train), np.sqrt(X_test)
        if standardise_spikes:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        y_train, y_test = train_df.distance_bin_id.values, test_df.distance_bin_id.values
        y_true = test_df[metric].values
        # search over regularisation strengths
        best_alpha, best_MSE = search_reg(
            X_train, X_test, y_train, y_test, y_true, output=output, distances=distance_bin_mids
        )
        val_results.append(
            {
                "vfold": vfold,
                "best_alpha": best_alpha,
                "best_MSE": best_MSE,
            }
        )
    reg_df = pd.DataFrame(val_results)
    if return_as == "df":
        return reg_df
    elif return_as == "best":
        # median opt reg strength across folds
        opt_reg = reg_df.best_alpha.median()
        return opt_reg
    else:
        raise ValueError(f"Return as must be 'df' of 'best'. ")


def search_reg(
    X_train,
    X_test,
    y_train,
    y_test,
    y_true,
    distances,
    output="max",
    reg_range=np.logspace(-4, 4, 20),
    tol=1e-3,
    patience=None,
    return_as="best",
    verbose=False,
):
    """
    CV search for optimal regulaisation strength (in training data)
    """
    best_alpha = None
    best_SE = np.inf
    history = []
    no_improvement_count = 0
    for alpha in reg_range:
        model = LogisticRegression(penalty="l2", C=alpha, max_iter=10_000, random_state=0, class_weight="balanced")
        model.fit(X_train, y_train)
        if output == "weighted":
            y_prob = model.predict_proba(X_test)
            decoded_dist = np.dot(y_prob, distances)  # weighted average of decoded distances
            SE = np.mean(np.abs((decoded_dist - y_true)))
        elif output == "max":
            y_pred = model.predict(X_test)
            decoded_dist = distances[y_pred]
            test_dist = distances[y_test]
            SE = np.mean(np.abs((decoded_dist - test_dist)))
        if SE < best_SE - tol:
            best_SE = SE
            best_alpha = alpha
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if patience is not None and no_improvement_count >= patience:
                if verbose:
                    print(f"Stopping early at α = {alpha:.3e} with SE = {SE:.4f}")
                break
        if verbose:
            print(f" α = {alpha:.3e},  SE = {SE:.4f}")
        history.append((alpha, SE))
    if return_as == "best":
        return best_alpha, best_SE
    elif return_as == "history":
        return np.array(history).T
    else:
        raise ValueError(f"Unknown return_as: {return_as}. Must be 'best' or 'history'.")


def get_input_data(
    session,
    resolution=0.5,
    metric=("distance_to_goal", "geodesic"),
    include_multiunits=True,
    moving_only=False,
    max_steps_to_goal=30,
    bin_spacing=0.05,
    bin_method="uniform",
    max_distance=None,
    n_log_bins=25,
    balance_distances=False,
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
        if max_distance is None:
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
    input_df.loc[:, "distance_bin_mid"] = input_df.distance_bin.apply(lambda x: x.mid)
    input_df.loc[:, "distance_bin_id"] = input_df.distance_bin.map({b: i for i, b in enumerate(bins)})
    if not balance_distances:
        return input_df
    else:  # balance data across distance bins
        max_size = input_df.groupby("distance_bin_id").size().max()
        balanced_data = (
            input_df.groupby("distance_bin_id", group_keys=False)
            .sample(n=max_size, replace=True, random_state=42)
            .reset_index(drop=True)
        )
        return balanced_data
