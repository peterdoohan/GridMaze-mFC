"""
Library for distance to goal rep, theta mod decoding.
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pingouin import multivariate_ttest
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from matplotlib.ticker import ScalarFormatter


from GridMaze.analysis.core import folds
from GridMaze.analysis.core import convert
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.distance_to_goal import distributions as dd
from GridMaze.analysis.distance_to_goal import logreg_decoder as ld
from GridMaze.analysis.place_direction.future_decoding import get_decision_points

# %% Global Variables

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "distance_to_goal" / "logreg_decoding" / "lfp_mod"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

# %% dev: speed split analysis


def plot_speed_split_decoding_bias(results_df, colors=["tan", "indianred"], print_stats=True, ax=None):
    """ """
    df = results_df.copy()
    # subtract mean phase deocding error form each specific lfp phase decoding error
    df.loc[:, "lfp_phase"] = df.lfp_phase.sub(df.mean_phase_decoding, axis=0).values
    median_speed = df.speed.median()
    # median split data into fast and slow decoding samples
    slow_df = df[df.speed < median_speed]
    fast_df = df[df.speed >= median_speed]
    # get mean decoding error for each subject in all conditions
    _df, _slow_df, _fast_df = [df.groupby("subject_ID").lfp_phase.mean() for df in [df, slow_df, fast_df]]
    # normalise phase decoding bias for fast and slow by non split data mean (for each subject)
    slow_norm = _slow_df.sub(_df.mean(axis=1), axis=0).lfp_phase
    fast_norm = _fast_df.sub(_df.mean(axis=1), axis=0).lfp_phase
    # plotting
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.spines[["top", "right"]].set_visible(False)
    for data, color, label in zip([slow_norm, fast_norm], colors, ["slow", "fast"]):
        mean = data.mean()
        sem = data.sem()
        phase = mean.index.astype(float)
        ax.errorbar(
            phase,
            mean.values,
            yerr=sem.values,
            fmt="o-",
            color=color,
            markersize=6,
            linewidth=2,
            capsize=None,
            elinewidth=2,
            label=label,
        )
    _format_ax(ax)
    ax.legend()
    if print_stats:
        stats = get_speed_split_stats(_slow_df, _fast_df)
        print(stats)


def get_speed_split_stats(_slow_df, _fast_df):
    """ """
    slow, fast = _slow_df.lfp_phase, _fast_df.lfp_phase
    delta_df = (fast - slow).T
    # hypoth delta is greater than 0 with multiple t-tests
    p_vals = []
    for i in range(delta_df.shape[0]):
        _, p = ttest_1samp(delta_df.iloc[i], 0, alternative="greater")
        p_vals.append(p)
    p_vals = np.array(p_vals)
    # correct for multiple comparisons
    reject, p_vals_corrected, _, _ = multipletests(p_vals, alpha=0.05, method="fdr_bh")
    # output as pd.Serues
    output = pd.Series(index=delta_df.index.astype(float).round(2), data=p_vals_corrected)
    return output


# %% main theta mod analysis


def plot_decoding_theta_bias(
    results_df,
    maze_names=["maze_1", "maze_2", "rooms_maze"],
    distance_range=None,
    speed_range=None,
    color="indigo",
    print_stats=True,
    ax=None,
):
    """ """
    df = results_df.copy()
    # filter for mazes
    df = df[df.maze_name.isin(maze_names)]
    # filter for distance to goal
    if distance_range is not None:
        df = df[df.distance_to_goal.geodesic.between(*distance_range)]
    # filter for speed
    if speed_range is not None:
        df = df[df.speed.between(*speed_range)]
    _df = df.groupby(["subject_ID"]).lfp_phase.mean()
    x_norm = _df.sub(_df.mean(axis=1), axis=0)
    phase_mean_decoding = x_norm.lfp_phase
    # # average across subjects
    mean = phase_mean_decoding.mean()
    sem = phase_mean_decoding.sem()
    phase = mean.index.astype(float)
    # plotting
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.errorbar(
        phase,
        mean.values,
        yerr=sem.values,
        fmt="o-",
        color=color,
        markersize=6,
        linewidth=2,
        capsize=None,
        elinewidth=2,
    )
    _format_ax(ax)
    if print_stats:
        _get_decoding_bias_stats(phase_mean_decoding)
    return


def _format_ax(ax):
    ax.set_xlabel("theta phase")
    ax.set_ylabel("decoding bias (m)\n (distance-to-goal)")
    ax.set_xticks(np.arange(-np.pi, np.pi + 0.1, np.pi / 2))
    ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    return


def _get_decoding_bias_stats(phase_mean_decoding):
    """ """
    phis = phase_mean_decoding.columns.astype(float)
    data = phase_mean_decoding.values
    beta_cos = data.dot(np.cos(phis))
    beta_sin = data.dot(np.sin(phis))
    betas = np.column_stack([beta_cos, beta_sin])
    zeros = np.zeros_like(betas)
    mv_test = multivariate_ttest(betas, zeros, paired=False)
    return print(mv_test)


# %% populate and load data


def load_decoding_results(lfp_type="theta"):
    """ """
    results_paths = list((RESULTS_DIR / lfp_type).iterdir())
    dfs = []
    for path in results_paths:
        df = pd.read_parquet(path)
        dfs.append(df)
    results_df = pd.concat(dfs, axis=0)
    return results_df


def populate_decoding_results(lfp_type="theta", subfolder=None, max_distance=0.8, max_jobs=False, verbose=True):
    """ """

    def _process_session(session):
        """ """
        if verbose:
            print(f"Processing session {session.name}")
        # set up save path
        if subfolder is None:
            save_path = RESULTS_DIR / lfp_type / f"{session.name}.parquet"
        else:
            save_path = RESULTS_DIR / subfolder / f"{session.name}.parquet"
        # reun decode with defualt settings
        try:
            results_df = get_theta_mod_distance_to_goal_decoding(
                session, lfp_type=lfp_type, max_distance=max_distance, verbose=verbose
            )
            # add session info
            results_df[("subject_ID", "")] = session.subject_ID
            results_df[("maze_name", "")] = session.maze_name
            results_df[("day_on_maze", "")] = session.day_on_maze
            # save
            results_df.to_parquet(save_path)
            if verbose:
                print(f"Saved results for to {save_path}")
        except Exception as e:
            if verbose:  # not enough trials in some early sessions
                print(f"Error processing session {session.name}: {e}")

    with_data = [
        "navigation_df",
        "cluster_metrics",
        "trials_df",
    ]
    if lfp_type == "theta":
        with_data.append("navigation_theta_spike_counts_df")
    elif lfp_type == "4Hz":
        with_data.append("navigation_4Hz_spike_counts_df")
    else:
        raise ValueError(f"lfp_type {lfp_type} not recognised, must be 'theta' or '4Hz'")

    for subject_ID in SUBJECT_IDS:
        if verbose:
            print(f"Loading sessions for for {subject_ID} ...")
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names="all",
            days_on_maze="all",
            with_data=with_data,
            must_have_data=True,
        )
        # process sessions in parallel
        if max_jobs:
            if verbose:
                print(f"Running {len(sessions)} sessions ...")
            Parallel(n_jobs=max_jobs)(delayed(_process_session)(session) for session in sessions)
        else:
            for session in sessions:
                _process_session(session)
        if verbose:
            print(f"Finished populating {subject_ID} decoding results ...")
    return


# %%
def get_theta_mod_distance_to_goal_decoding(
    session,
    resolution=0.4,
    lfp_type="theta",  # 'theta' or '4Hz'
    metric=("distance_to_goal", "geodesic"),
    include_multiunits=True,
    moving_only=True,
    max_steps_to_goal=30,
    bin_spacing=0.04,
    max_distance=0.8,  # best decoding at short distances
    bin_method="uniform",
    n_log_bins=30,
    balance_distances=False,
    n_folds=8,
    sqrt_spikes=True,
    standardise_spikes=True,
    alpha="opt",
    output="max",
    verbose=True,
):
    """ """
    # get input data
    if verbose:
        print("loading input data ...")
    input_data = get_input_data(
        session,
        resolution=resolution,
        lfp_type=lfp_type,
        metric=metric,
        include_multiunits=include_multiunits,
        moving_only=moving_only,
        max_steps_to_goal=max_steps_to_goal,
        bin_spacing=bin_spacing,
        bin_method=bin_method,
        max_distance=max_distance,
        n_log_bins=n_log_bins,
        balance_distances=balance_distances,
    )
    distance_bin_mids = np.array(sorted(input_data.distance_bin_mid.unique()))
    lfp_phases = input_data.spike_count.columns.get_level_values(1).unique().values

    # set up results df
    results_df = pd.concat(
        [
            input_data.drop("spike_count", level=0, axis=1).droplevel(2, axis=1),
            pd.DataFrame(index=input_data.index, columns=pd.MultiIndex.from_product([["lfp_phase"], lfp_phases])),
        ],
        axis=1,
    )  # sample info + err for each sample at each lfp phase
    results_df[("mean_phase_decoding", "")] = np.nan

    # decode distance CV
    folds_df = folds.get_folds_df(session, goal_stratified=False, return_unique_IDs=True, n_folds=n_folds)
    _folds = folds_df.columns.get_level_values(0).unique()
    for fold in _folds:
        if verbose:
            print(fold)
        fold_df = folds_df[fold]
        if alpha == "opt":
            # get CV opt reg strength from just training data
            opt_alpha = get_CV_alpha(
                input_data,
                fold_df,
                metric,
                output,
                sqrt_spikes=sqrt_spikes,
                standardise_spikes=standardise_spikes,
                return_as="best",
                verbose=verbose,
            )
            if verbose:
                print(f"Optimal alpha for {fold} is {opt_alpha}")
        else:
            opt_alpha = alpha

        # split test train
        train_df, test_df = folds._get_test_train_dfs(input_data, fold_df)
        # train on spikes from all theta bins
        X_train = train_df.spike_count.T.groupby(level=0).mean().T.values  # mean spikecount across lfp phases
        if sqrt_spikes:
            X_train = np.sqrt(X_train)
        if standardise_spikes:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
        y_train = train_df.distance_bin_id.values
        # train model on average spike counts across all lfp phases
        if verbose:
            print(f"    Training on mean spike counts across all lfp phases")
        model = LogisticRegression(penalty="l2", C=opt_alpha, max_iter=10_000, random_state=0, class_weight="balanced")
        model.fit(X_train, y_train)

        # test on each lfp_phase separately
        test_spikes_df = test_df.spike_count.swaplevel(axis=1).sort_index(axis=1)
        y_test = test_df.distance_bin_id.values
        true_dist = distance_bin_mids[y_test]
        for i, phase in enumerate(lfp_phases):
            if verbose:
                print(f"    Testing on phase: {phase:.2f}")
            X_test = test_spikes_df[phase].values
            if sqrt_spikes:
                X_test = np.sqrt(X_test)
            if standardise_spikes:
                X_test = scaler.transform(X_test)
            # get output metric
            if output == "weighted":
                y_pred_prob = model.predict_proba(X_test)
                weighted_dist = y_pred_prob.dot(distance_bin_mids)
                err = weighted_dist - true_dist
            elif output == "max":
                y_pred = model.predict(X_test)
                y_pred_dist = distance_bin_mids[y_pred]
                err = y_pred_dist - true_dist
            else:
                NotImplementedError(f"Test metric {output} not implemented.")
            # store results
            results_df.loc[test_df.index, ("lfp_phase", phase)] = err

        # test on mean spike counts across phases (baseline performance)
        if verbose:
            print("         Testing on mean phase spike counts")
        X_test = test_df.spike_count.T.groupby(level=0).mean().T.values
        if sqrt_spikes:
            X_test = np.sqrt(X_test)
        if standardise_spikes:
            X_test = scaler.transform(X_test)
        if output == "weighted":
            y_pred_prob = model.predict_proba(X_test)
            weighted_dist = y_pred_prob.dot(distance_bin_mids)
            err = weighted_dist - true_dist
        elif output == "max":
            y_pred = model.predict(X_test)
            y_pred_dist = distance_bin_mids[y_pred]
            err = y_pred_dist - true_dist
        else:
            pass
        # store results
        results_df.loc[test_df.index, ("mean_phase_decoding", "")] = err

    return results_df


def get_CV_alpha(
    input_data,
    fold_df,
    metric,
    output,
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
        # optimise just mean spikes over lfp phase bins and use for all test decodings later
        X_train, X_test = [df.spike_count.T.groupby(level=0).mean().T for df in [train_df, test_df]]
        if sqrt_spikes:
            X_train, X_test = np.sqrt(X_train), np.sqrt(X_test)
        if standardise_spikes:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        y_train, y_test = train_df.distance_bin_id.values, test_df.distance_bin_id.values
        y_true = test_df[(*metric, "")].values
        # search over regularisation strengths
        best_alpha, best_MSE = ld.search_reg(
            X_train,
            X_test,
            y_train,
            y_test,
            y_true,
            output=output,
            distances=distance_bin_mids,
            verbose=False,
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


# %% get input data


def get_input_data(
    session,
    resolution=0.4,
    lfp_type="theta",
    metric=("distance_to_goal", "geodesic"),
    include_multiunits=True,
    moving_only=True,
    max_steps_to_goal=30,
    bin_spacing=0.04,
    bin_method="uniform",
    max_distance=None,
    n_log_bins=25,
    balance_distances=False,
):
    """ """
    # load data
    navigation_df = session.navigation_df.copy()
    if lfp_type == "theta":
        spike_counts_df = session.navigation_theta_spike_counts_df
    elif lfp_type == "4Hz":
        spike_counts_df = session.navigation_4Hz_spike_counts_df
    else:
        raise ValueError(f"lfp_type {lfp_type} not recognised, must be 'theta' or '4Hz'")
    spike_counts_df.reset_index(inplace=True, drop=True)  # [frames, clusters * 12 lfp phase bins]
    cluster_metrics = session.cluster_metrics
    session_info = session.session_info

    # filter for single units
    if not include_multiunits:
        single_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID
        single_units = convert.cluster_IDs2scluster_unique_IDs(session_info, single_units)
        spike_counts_df = spike_counts_df[
            spike_counts_df.columns[[c in single_units for c in spike_counts_df.columns.get_level_values(1)]]
        ]

    ds_nav_df, ds_spike_counts_df = ds.downsample_nav_spikes_data(
        navigation_df,
        spike_counts_df,
        resolution=resolution,
        distance_metrics=[("steps_to_goal", "future"), metric],
    )
    # add lvl to nav_df to match spike_counts
    ds_nav_df.columns = pd.MultiIndex.from_tuples([(*c, "") for c in ds_nav_df.columns])
    metric = (*metric, "")
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
    input_df.loc[:, ("distance_bin", "", "")] = pd.cut(input_df[metric], bins=bins, include_lowest=True).to_numpy()
    input_df.loc[:, ("distance_bin_mid", "", "")] = input_df.distance_bin.apply(lambda x: x.mid)
    input_df.loc[:, ("distance_bin_id", "", "")] = input_df.distance_bin.map({b: i for i, b in enumerate(bins)})
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


# %%
