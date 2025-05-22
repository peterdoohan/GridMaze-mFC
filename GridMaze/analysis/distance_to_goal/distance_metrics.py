"""
Library for comparing distance to goal tuning metrics
"""

# %% Imports
import json
from tkinter import font
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel
from matplotlib import pyplot as plt
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

with open(EXPERIMENT_INFO_PATH / "maze_day2date.json", "r") as input_file:
    MAZE_DAY2DATE = json.load(input_file)

MAZE2MAX_DAY = {m: int((list(MAZE_DAY2DATE[m].keys())[-1])) for m in MAZE_DAY2DATE.keys()}

DISTANCE_METRICS = ["geodesic", "euclidean", "manhattan", "future"]


# %% plot results from CPD summary


def plot_CPD_timeseries(summary_df, axes=None, comparison="geodesic_vs_euclidean", group_days=3):
    """ """
    # remove CPD outliers
    outlier_mask = summary_df[comparison].lt(-0.5).any(axis=1)
    df = summary_df[~outlier_mask]
    if group_days:
        # update maze day label to group days together
        df.loc[:, ("day_on_maze", "")] = (df.day_on_maze // group_days) * group_days
    # process data
    df = df.groupby(["subject_ID", "maze_name", "day_on_maze"])[comparison].mean()[comparison]
    sub_grouped_df = df.groupby(["maze_name", "day_on_maze"])
    mean_df = sub_grouped_df.mean()
    sem_df = sub_grouped_df.sem()
    # plotting
    metric_1, metric_2 = comparison.split("_vs_")
    mean_df = mean_df.mul(100)  # convert to %
    sem_df = sem_df.mul(100)
    if axes is None:
        f, axes = plt.subplots(1, 3, figsize=(6, 2), sharey=True)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("CPD (%)")

    for maze_name, ax in zip(MAZE_DAY2DATE.keys(), axes):
        for metric in [metric_1, metric_2]:
            mean = mean_df.loc[maze_name][metric]
            sem = sem_df.loc[maze_name][metric]
            ax.plot(mean.index, mean.values, label=metric)
            ax.fill_between(mean.index, mean - sem, mean + sem, alpha=0.2)
            ax.set_xlabel("days on maze")
        ax.set_title(maze_name)
    axes[-1].legend(fontsize=8, loc="lower left")


def plot_cross_subject_CPD_comparison(
    summary_df, comparison="geodesic_vs_euclidean", maze_names=["maze_1", "maze_2"], late_sessions=True, ax=None
):
    """ """
    # filter data
    df = summary_df[summary_df.maze_name.isin(maze_names)]
    if late_sessions:
        df = df[df.apply(_is_late_session, axis=1)]
    df.drop(columns=[("maze_name", ""), ("day_on_maze", "")], inplace=True)
    df.set_index("subject_ID", append=True, inplace=True)
    # process data
    mean_cpd = df[comparison].groupby("subject_ID").mean().unstack().reset_index()
    mean_cpd.columns = ["metric", "subject_ID", "CPD"]
    # plot
    mean_cpd["CPD"] = mean_cpd["CPD"].mul(100)  # convert to %
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2, 2))

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("CPD (%)")
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    sns.pointplot(
        data=mean_cpd,
        x="metric",
        y="CPD",
        hue="subject_ID",
        errorbar=None,
        dodge=False,
        markers="o",
        linestyles="-",
        legend=False,
        markersize=8,
        linewidth=4,
    )
    ax.set_ylim(-1, 2)
    return


def plot_pairwise_CPD_summary(summary_df, late_sessions=True, maze_names=["maze_1", "maze_2"], ax=None):
    """ """
    # process summary data
    df = summary_df[summary_df.maze_name.isin(maze_names)]
    if late_sessions:
        df = df[df.apply(_is_late_session, axis=1)]
    dfs = []
    for subject in SUBJECT_IDS:
        subject_df = df[df.subject_ID == subject].copy()
        subject_df.drop(columns=[("subject_ID", ""), ("maze_name", ""), ("day_on_maze", "")], inplace=True)
        comparisons = subject_df.columns.get_level_values(0).unique()
        cpd_df = pd.DataFrame(index=DISTANCE_METRICS, columns=DISTANCE_METRICS, dtype=float)
        for c in comparisons:
            metric_1, metric_2 = c.split("_vs_")
            mean_cpd = subject_df[c].mean()
            cpd_df.loc[metric_1, metric_2] = mean_cpd.loc[metric_1]
            cpd_df.loc[metric_2, metric_1] = mean_cpd.loc[metric_2]
        cpd_df.fillna(0, inplace=True)
        dfs.append(cpd_df)
    # average CPDs across subjects
    subject_av_cpds = np.mean(np.stack([x.values for x in dfs]), axis=0)
    output_df = pd.DataFrame(index=DISTANCE_METRICS, columns=DISTANCE_METRICS, data=subject_av_cpds)

    # plot
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2, 2), sharey=True)
    output_df = output_df.mul(100)  # convert to %
    for d in DISTANCE_METRICS:
        output_df.loc[d, d] = np.nan
    cmap = sns.color_palette("Reds", as_cmap=True)
    cmap.set_bad(color="lightgrey")
    sns.heatmap(
        output_df,
        vmin=0,
        cmap=cmap,
        ax=ax,
        square=True,
        fmt=".1f",
        cbar_kws={"shrink": 0.75, "label": f"CPD (%)"},
    )
    ax.tick_params(axis="x", which="both", top=True, bottom=False, labeltop=True, labelbottom=False, labelrotation=45)


# %% plot results from weight summary dfs


def plot_weights_comparison_timeseries(
    summary_df,
    comparison="geodesic_vs_euclidean",
    plot_metric="geodesic",
    norm_metric="L1_ratio",
    group_days=3,
    axes=None,
):
    """"""
    # process data
    df = summary_df.copy()
    if group_days:
        # update maze day label to group days together
        df.loc[:, ("day_on_maze", "")] = (df.day_on_maze // group_days) * group_days
    df = (
        df[
            [
                (comparison, norm_metric),
                (comparison, "metric"),
                ("maze_name", ""),
                ("day_on_maze", ""),
                ("subject_ID", ""),
            ]
        ]
        .reset_index()
        .copy()
    )
    timeseries_df = (
        df.groupby([("subject_ID", ""), ("maze_name", ""), ("day_on_maze", ""), (comparison, "metric")])[
            [(comparison, norm_metric)]
        ]
        .mean()[comparison]
        .unstack()
    )
    sub_grouped_df = timeseries_df.groupby([("maze_name", ""), ("day_on_maze", "")])
    mean_df = sub_grouped_df.mean()[norm_metric][plot_metric]
    sem_df = sub_grouped_df.sem()[norm_metric][plot_metric]

    # plotting
    if axes is None:
        f, axes = plt.subplots(1, 3, figsize=(6, 2), sharey=True)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0.5, color="k", linestyle="--", alpha=0.5)
    for maze_name, ax in zip(MAZE_DAY2DATE.keys(), axes):
        mean = mean_df.loc[maze_name]
        sem = sem_df.loc[maze_name]
        ax.plot(mean.index, mean.values, label=plot_metric)
        ax.fill_between(mean.index, mean - sem, mean + sem, alpha=0.2)
        ax.set_xlabel("days on maze")
        ax.set_title(maze_name)
    axes[0].set_ylabel(norm_metric)
    axes[-1].legend(fontsize=8, loc="upper left")


def plot_cross_subject_norm_comparison(
    summary_df,
    comparison="geodesic_vs_euclidean",
    norm_metric="L1_ratio",
    late_sessions=True,
    maze_names=["maze_1", "maze_2"],
    ax=None,
    print_stats=True,
):
    """ """
    # process data
    metric_1, metric_2 = comparison.split("_vs_")
    df = summary_df[summary_df.maze_name.isin(maze_names)]
    if late_sessions:
        df = df[df.apply(_is_late_session, axis=1)].copy()
    # drop info columns
    df.drop(columns=[("maze_name", ""), ("day_on_maze", "")], inplace=True)
    df.set_index("subject_ID", append=True, inplace=True)
    comp_df = df[comparison][["metric", norm_metric]]
    mean_norms_df = comp_df.groupby(["subject_ID", "metric"])[norm_metric].mean().reset_index()
    # plot
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0.45, 0.55)
    ax.set_ylabel(norm_metric)
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.5)
    sns.pointplot(
        data=mean_norms_df,
        order=[metric_1, metric_2],
        x="metric",
        y=norm_metric,
        hue="subject_ID",
        errorbar=None,
        dodge=False,
        markers="o",
        linestyles="-",
        legend=False,
        markersize=8,
        linewidth=4,
    )
    # do stats
    if print_stats:
        wide = mean_norms_df.pivot(index="subject_ID", columns="metric", values=norm_metric)
        t_stat, t_p = ttest_rel(wide["euclidean"], wide["geodesic"])
        print(f"{comparison}: {norm_metric} t-stat: {t_stat:.3f}, p-value: {t_p:.3e}")


def plot_all_pairwise_metric_norm_diffs(
    summary_df, norm_metric="L1_ratio", late_sessions=True, maze_names=["maze_1", "maze_2"], ax=None
):
    """
    Calculate the average L1_ratio and L2_ratio over all (late session) neurons
    for a subject under each distance metric pairwise comparison.
    Compute the difference between each ratio across metric (intuitively, what fraction of extra
    weight are attributed to metric 1 vs metric 2), separately for each subject.
    Averge these matrics across subjects and plot for L1 and L2 separately.
    """
    # process data
    df = summary_df[summary_df.maze_name.isin(maze_names)]
    if late_sessions:
        df = df[df.apply(_is_late_session, axis=1)].copy()
    dfs = []
    for subject in SUBJECT_IDS:
        subject_df = df[df.subject_ID == subject].copy()
        # filter for "late" sessions
        subject_df = subject_df[subject_df.apply(_is_late_session, axis=1)]
        # drop info columns
        subject_df.drop(columns=[("subject_ID", ""), ("maze_name", ""), ("day_on_maze", "")], inplace=True)
        comparisons = subject_df.columns.get_level_values(0).unique()
        norm_diff_df = pd.DataFrame(index=DISTANCE_METRICS, columns=DISTANCE_METRICS)
        for c in comparisons:
            metric_1, metric_2 = c.split("_vs_")
            comp_df = subject_df[c]
            mean_norms = comp_df.groupby("metric")[norm_metric].mean()
            norm_diff = mean_norms.loc[metric_1] - mean_norms.loc[metric_2]
            norm_diff_df.loc[metric_1, metric_2] = norm_diff
        dfs.append(norm_diff_df)
    # average diff norms across subjects
    arr = np.stack([df.values.astype(float) for df in dfs], axis=0)
    masked = np.ma.masked_invalid(arr)  # mask all NaNs
    mean_masked = masked.mean(axis=0)
    df = pd.DataFrame(index=DISTANCE_METRICS, columns=DISTANCE_METRICS, data=mean_masked.filled(np.nan))
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2, 2), sharey=True)
    df = df.mul(100)  # convert to % weight difference
    for d in DISTANCE_METRICS:
        df.loc[d, d] = 0
    sns.heatmap(
        df,
        vmin=-4,
        vmax=4,
        cmap="coolwarm",
        ax=ax,
        square=True,
        fmt=".1f",
        cbar_kws={"shrink": 0.75, "label": f"Δ{norm_metric} (%)"},
    )
    ax.tick_params(axis="x", which="both", top=True, bottom=False, labeltop=True, labelbottom=False, labelrotation=45)


def _is_late_session(row):
    """Def as last seven days on the maze"""
    max_days = MAZE2MAX_DAY[row[("maze_name", "")]]
    if row[("day_on_maze", "")] > max_days - 7:
        return True
    else:
        return False


# %% populate weights pairwise comparisons and big summary df


def get_weight_metrics_summary_df(verbose=False):
    """ """
    save_path = RESULTS_DIR / "weight_metric_summary_df.csv"
    if save_path.exists():
        if verbose:
            print(f"Loading weight metric summaries df from {save_path}")
        results_df = pd.read_csv(save_path, index_col=0, header=[0, 1])
        # fix cols when loading from disk
        results_df.columns = pd.MultiIndex.from_tuples(
            [c if "Unnamed" not in c[1] else (c[0], "") for c in results_df.columns]
        )
    else:
        if verbose:
            print(f"loading sessions ...")
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
            try:
                comparisons_df = run_pairwise_weight_metric_comparisons(session, verbose=verbose)
                dfs.append(comparisons_df)
            except Exception as e:
                if verbose:
                    print(f"Error processing {session.name}: \n {e}")
        results_df = pd.concat(dfs, axis=0)
        # save
        save_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(save_path, index=True)
        if verbose:
            print(f"Saved weight metric summaries df to {save_path}")
    return results_df


def run_pairwise_weight_metric_comparisons(session, verbose=True):
    """ """
    metric_pairs = list(combinations(DISTANCE_METRICS, 2))
    dfs = []
    for metric_1, metric_2 in metric_pairs:
        _name = f"{metric_1}_vs_{metric_2}"
        if verbose:
            print(_name)
        weight_metrics_df = get_distance_metric_weight_summaries(session, metric_1, metric_2)
        weight_metrics_df.columns = pd.MultiIndex.from_product([[_name], weight_metrics_df.columns])
        dfs.append(weight_metrics_df)
    comparisons_df = pd.concat(dfs, axis=1)
    comparisons_df[("subject_ID", "")] = session.subject_ID
    comparisons_df[("maze_name", "")] = session.maze_name
    comparisons_df[("day_on_maze", "")] = session.day_on_maze
    return comparisons_df


def quick_weight_check(comparisons_df):
    comparisons_df.drop(columns=[("subject_ID", ""), ("maze_name", ""), ("day_on_maze", "")], inplace=True)
    parwise_comparisons = comparisons_df.columns.get_level_values(0).unique()
    for pc in parwise_comparisons:
        df = comparisons_df[pc]
        print(df.groupby("metric")[["L1_ratio", "L2_ratio"]].mean())


# %% L1, L2 ratio comparison function


def get_distance_metric_weight_summaries(
    session,
    metric_1="geodesic",
    metric_2="euclidean",
    resolution=0.5,
    fixed_alpha=False,
    model="PoissonRegressor",
    n_bases=10,
    basis_type="gamma",
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
        if m == "future":
            _max = dd.get_distance_percentile(("distance_to_goal", "geodesic"), percentile=85)
        else:
            _max = dd.get_distance_percentile(_m, percentile=85)
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
    results_df.set_index("cluster_unique_ID", inplace=True)
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


def get_distance_metric_CPD_summary_df(verbose=False):
    """ """
    save_path = RESULTS_DIR / "cpd_summary_df.csv"
    if save_path.exists():
        if verbose:
            print(f"Loading CPD summaries df from {save_path}")
        results_df = pd.read_csv(save_path, index_col=0, header=[0, 1])
        # fix cols when loading from disk
        results_df.columns = pd.MultiIndex.from_tuples(
            [c if "Unnamed" not in c[1] else (c[0], "") for c in results_df.columns]
        )
    else:
        if verbose:
            print(f"loading sessions ...")
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
            try:
                comparisons_df = run_pairwise_CPD_comparisons(session, verbose=verbose)
                dfs.append(comparisons_df)
            except Exception as e:
                if verbose:
                    print(f"Error processing {session.name}: \n {e}")
        results_df = pd.concat(dfs, axis=0)
        # save
        save_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(save_path, index=True)
        if verbose:
            print(f"Saved CPD summaries df to {save_path}")
    return results_df


def run_pairwise_CPD_comparisons(session, verbose=True):
    """ """
    metric_pairs = list(combinations(DISTANCE_METRICS, 2))
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
        if m == "future":
            _max = dd.get_distance_percentile(("distance_to_goal", "geodesic"), percentile=85)
        else:
            _max = dd.get_distance_percentile(_m, percentile=85)
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
