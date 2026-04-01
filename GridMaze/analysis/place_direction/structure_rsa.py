"""
Quantification of how maze structures drive neural place-direction representations
using RSA apprach
@peterdoohan
"""

# %% Imports
import json
from os import error
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_1samp, pearsonr
from statsmodels.stats.multitest import multipletests

from GridMaze.maze import metrics as mm
from GridMaze.maze import representations as mr
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.place_direction import rep_similarity as rs

from GridMaze.analysis.neGLM import tuning_summaries as ts
from GridMaze.analysis.unit_match import place_direction as umpd
from GridMaze.analysis.unit_match import get_across_maze_matches as umm


# %% Global variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

RESULTS_DIR = RESULTS_PATH / "place_direction" / "structure_rsa"

# %% Functions


def plot_structure_RSA_summary(maze_names=["maze_1", "maze_2"], plot_null_dist=False, ax=None):
    # load/generate data
    obs_dfs, perm_dfs = [], []
    for maze_name in maze_names:
        obs_df = run_RSA(maze_name=maze_name, plot=False, print_stats=False)
        obs_df["maze_name"] = maze_name
        perm_df = get_RSA_null_df(maze_name=maze_name, verbose=False)
        perm_df["maze_name"] = maze_name
        obs_dfs.append(obs_df)
        perm_dfs.append(perm_df)
    obs_df = pd.concat(obs_dfs, axis=0).reset_index(drop=True)
    perm_df = pd.concat(perm_dfs, axis=0).reset_index(drop=True)

    # set up fig
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("maze feature")
    ax.set_ylabel("RSA beta")

    # avg
    metrics = [col for col in obs_df.columns if col not in ["subject_ID", "intercept", "R2", "maze_name"]]
    obs_mean = obs_df.groupby("subject_ID")[metrics].mean()
    perm_mean = perm_df.groupby(["permutation"])[metrics].mean()

    # plot permuted dist
    if plot_null_dist:
        long_perm_df = perm_mean.melt(var_name="metric", value_name="beta")
        sns.violinplot(
            data=long_perm_df,
            x="metric",
            y="beta",
            color="grey",
            split=True,
            inner=None,
            ax=ax,
        )

    # plot tobs means
    long_obs_df = obs_mean.melt(var_name="metric", value_name="beta")
    sns.pointplot(
        data=long_obs_df,
        x="metric",
        y="beta",
        ax=ax,
        color="darkred",
        markers="o",
        linestyle="none",
        errorbar="se",
        err_kws={"linewidth": 2},
        markeredgewidth=0,
    )
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_ylim(-0.05, 0.29)
    # stats
    stats_df = _get_null_stats(obs_df, perm_df, metrics)
    print(stats_df)


def get_RSA_null_df(maze_name="maze_1", n_permutations=1_000, verbose=True, save=False):
    """ """
    save_path = RESULTS_DIR / f"{maze_name}_null_betas.parquet"
    if save_path.exists() and not save:
        if verbose:
            print(f"Loading existing null betas from {save_path}")
        return pd.read_parquet(save_path)

    if verbose:
        print("loading sessions...")
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names=[maze_name],
        days_on_maze="late",
        with_data=[
            "navigation_df",
            "navigation_spike_rates_df",
            "cluster_metrics",
            "cluster_place_direction_tuning_metrics",
        ],
        must_have_data=True,
    )
    dfs = []
    for i in range(n_permutations):
        if verbose:
            if i % 10 == 0:
                print(f"Permutation {i}/{n_permutations}")
        heatmap_df = get_population_place_tuning_df(sessions=sessions, permute=True)
        res_df = run_RSA(maze_name=maze_name, heatmap_df=heatmap_df, plot=False, print_stats=False)
        res_df["permutation"] = i
        dfs.append(res_df)
    perm_df = pd.concat(dfs, axis=0).reset_index(drop=True)
    if save:
        perm_df.to_parquet(save_path)
    return perm_df


def _get_null_stats(results_df, null_df, metrics):
    """ """
    obs = results_df[metrics].mean()
    perm_avg = null_df.groupby("permutation")[metrics].mean()
    stats = []
    for metric in metrics:
        val = obs[metric]
        null_values = perm_avg[metric].values
        ci95 = (np.percentile(null_values, 0.25).round(3), np.percentile(null_values, 99.75).round(3))
        # one sided p-value: proportion of null values greater than or equal to observed value
        p_val = np.sum(null_values >= val) / len(null_values)
        t_stats = (val - null_values.mean()) / null_values.std(ddof=0)
        stats.append({"metric": metric, "t_stat": t_stats, "p_val": p_val, "CI95": ci95})
    stats_df = pd.DataFrame(stats)
    return stats_df


def run_RSA(
    maze_name="maze_1",
    heatmap_df=None,
    rsa_metrics=[
        "euclidean_distance",
        "geodesic_distance",
        "boundary_distance",
        "betweenness_centrality",
        "subgoal_distance",
        "corner",
    ],
    orthogonalise_pairs=[("euclidean_distance", "geodesic_distance")],
    plot=True,
    print_stats=True,
    verbose=False,
    save=False,
):
    """ """
    save_path = RESULTS_DIR / f"{maze_name}_rsa_betas.parquet"
    if save_path.exists() and not save:
        if verbose:
            print(f"Loading existing RSA results from {save_path}")
        return pd.read_parquet(save_path)
    res = []
    simple_maze = mr.get_simple_maze(maze_name)
    model_RDM_dfs = [mm.get_maze_RDM_df(simple_maze, metric=metric) for metric in rsa_metrics]
    if heatmap_df is None:
        if verbose:
            print("Loading population place tuning heatmaps ...")
        heatmap_df = get_population_place_tuning_df(maze_name=maze_name)
    for subject_ID in SUBJECT_IDS:
        # get RSA inputs
        neural_RDM_df = get_neural_RDM_df(heatmap_df.loc[subject_ID])
        places = list(neural_RDM_df.index)
        D_neural = neural_RDM_df.loc[places, places].values.astype(float)
        y = _vec_upper(D_neural)  # # vectorise neural response (1D array length n_pairs)

        # build design matrix X (columns = vectorised model RDMs)
        X_cols = []
        for Mdf in model_RDM_dfs:
            Mmat = _ensure_df_order(Mdf, places)
            X_cols.append(_vec_upper(Mmat))
        X = np.column_stack(X_cols)

        # optionally orthogonalise predictors
        if orthogonalise_pairs is not None:
            for _pair in orthogonalise_pairs:
                target_name, ref_name = _pair
                if target_name not in rsa_metrics or ref_name not in rsa_metrics:
                    raise ValueError("Orthogonalisation pair names must be in rsa_metrics list.")
                target_idx = rsa_metrics.index(target_name)
                ref_idx = rsa_metrics.index(ref_name)
                # compute residualised column (mean-centered)
                X_resid = orthogonalise_column_against(X, target_idx, ref_idx)
                # replace column in X with residual; keep other columns as-is
                X[:, target_idx] = X_resid

        # standardise predictors and response
        pred_scaler = StandardScaler(with_mean=True, with_std=True)
        Xz = pred_scaler.fit_transform(X)

        y_mu = y.mean()
        y_sd = y.std(ddof=0)
        y_z = (y - y_mu) / y_sd

        # fit OLS regression model (with intercept)
        lr = LinearRegression(fit_intercept=True).fit(Xz, y_z)
        betas = lr.coef_.copy()  # standardized betas
        intercept = float(lr.intercept_)

        # compute R^2 on the fitted (z-scored) data
        yhat = lr.predict(Xz)
        ss_res = np.sum((y_z - yhat) ** 2)
        ss_tot = np.sum((y_z - y_z.mean()) ** 2)
        R2 = 0.0 if ss_tot == 0 else float(1.0 - ss_res / ss_tot)

        # output subject results
        out = {name: float(b) for name, b in zip(rsa_metrics, betas)}
        out["intercept"] = intercept
        out["R2"] = R2
        out["subject_ID"] = subject_ID
        res.append(out)

    results_df = pd.DataFrame(res)
    if print_stats:
        stats_df = _get_beta_stats(results_df)
        print(stats_df)
    if plot:
        _plot_RSA_results(results_df)
    if save:
        results_df.to_parquet(save_path)
    return results_df


def _get_beta_stats(results_df):
    """ """
    degf = len(results_df) - 1
    metrics = [col for col in results_df.columns if col not in ["subject_ID", "intercept", "R2"]]
    stats = []
    for metric in metrics:
        t_stat, p_val = ttest_1samp(results_df[metric], popmean=0.0)
        stats.append({"metric": metric, "t_stat": t_stat, "df": degf, "p_val": p_val})

    stats_df = pd.DataFrame(stats)
    # multiple testing correction
    p_vals = stats_df.p_val.values
    _, corrected_p_vals, _, _ = multipletests(p_vals, alpha=0.05, method="fdr_bh")
    stats_df["corrected_p_val"] = corrected_p_vals

    return stats_df


def _plot_RSA_results(results_df, ax=None):
    """ """
    # set up fig
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("maze feature")
    ax.set_ylabel("RSA beta")

    # process data
    metrics = [col for col in results_df.columns if col not in ["subject_ID", "intercept", "R2"]]
    plot_df = results_df.melt(id_vars=["subject_ID"], value_vars=metrics, var_name="metric", value_name="beta")
    grouped = plot_df.groupby(["metric"]).beta
    _mean = grouped.mean()
    _sem = grouped.sem()
    for i, metric in enumerate(metrics):
        y = _mean[metric]
        yerr = _sem[metric]
        subj_values = plot_df.loc[plot_df.metric == metric, "beta"].values
        ax.errorbar(i, y, yerr=yerr, fmt="o", color="darkred", markersize=7.5, elinewidth=2.5, alpha=1)
        ax.scatter(np.full_like(subj_values, i), subj_values, color="grey", alpha=0.5, s=8)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")


def orthogonalise_column_against(X, col_idx, ref_idx):
    """
    Orthogonalise column X[:, col_idx] w.r.t. X[:, ref_idx].
    Both columns should be 1D arrays (or columns in X). This returns a new column vector.
    Procedure: mean-center both, compute projection, subtract, return residual (mean removed).
    """
    v = X[:, col_idx].astype(float).copy()
    r = X[:, ref_idx].astype(float).copy()
    # mean-center
    v = v - np.mean(v)
    r = r - np.mean(r)
    denom = np.dot(r, r)
    if denom == 0:
        # ref is constant -> residual is just v (already mean-centered)
        return v
    proj_coeff = np.dot(r, v) / denom
    v_res = v - proj_coeff * r
    # ensure mean zero (should be)
    v_res = v_res - np.mean(v_res)
    return v_res


def _vec_upper(mat):
    """Vectorise upper-triangle (k=1) of a square numpy array or DataFrame."""
    if isinstance(mat, pd.DataFrame):
        mat = mat.values
    iu = np.triu_indices_from(mat, k=1)
    return mat[iu]


def _ensure_df_order(df, places):
    """Reorder df (both rows and cols) to match places list; return numpy array."""
    return df.loc[places, places].values.astype(float)


def get_neural_RDM_df(heatmap_df):
    """ """

    R = np.corrcoef(heatmap_df.values.T)
    D = 1 - R
    np.fill_diagonal(D, 0.0)

    places = heatmap_df.columns
    neural_RDM_df = pd.DataFrame(D, index=places, columns=places)

    # sort index and columns
    neural_RDM_df = neural_RDM_df.sort_index().sort_index(axis=1)

    return neural_RDM_df.astype(float)


def get_population_place_tuning_df(
    maze_name="maze_1",
    sessions=None,
    min_split_corr=0.5,
    late_sessions=True,
    include_multi_unit=False,
    permute=False,
):
    """ """
    return rs.get_population_place_tuning(
        subject_IDs="all",
        maze_name=maze_name,
        sessions=sessions,
        late_sessions=late_sessions,
        include_multi_unit=include_multi_unit,
        fill_nans="mean",
        normalisation="length",
        max_steps_to_goal=30,
        min_split_corr=min_split_corr,
        verbose=False,
        expand_index=True,
        permute=permute,
    )


# %% run structure RSA on unit matched data


def plot_cross_maze_beta_correlations(results_df, print_stats=True, min_neurons=10, ax=None):
    """Plot mean +/- SEM of per-subject beta correlations across mazes."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)

    df = results_df.loc[results_df.n_neurons >= min_neurons]
    metrics = df["metric"].unique()
    sns.pointplot(
        data=df,
        x="metric",
        y="pearson_r",
        ax=ax,
        color="darkred",
        markers="o",
        linestyle="none",
        errorbar="se",
        err_kws={"linewidth": 2},
        markeredgewidth=0,
        order=metrics,
    )
    sns.stripplot(
        data=df,
        x="metric",
        y="pearson_r",
        ax=ax,
        color="grey",
        alpha=0.5,
        size=4,
        jitter=0.1,
        order=metrics,
    )
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_xlabel("maze feature")
    ax.set_ylabel("cross-maze beta r")

    if print_stats:
        stats = []
        for metric in metrics:
            vals = df.loc[df.metric == metric, "pearson_r"].values
            t_stat, p_val = ttest_1samp(vals, popmean=0.0)
            stats.append({"metric": metric, "t_stat": t_stat, "p_val": p_val})
        stats_df = pd.DataFrame(stats)
        _, corrected_p_vals, _, _ = multipletests(stats_df.p_val.values, alpha=0.05, method="fdr_bh")
        stats_df["corrected_p_val"] = corrected_p_vals
        print(stats_df.to_string(index=False))

    return ax


def correlate_rsa_features_across_mazes(
    maze_pair=("maze_1", "maze_2"),  # only makes sense for maze_1 and maze_2
    matched_heatmap_df=None,
    rsa_metrics=[
        "euclidean_distance",
        "geodesic_distance",
        "boundary_distance",
        "betweenness_centrality",
        "subgoal_distance",
        "corner",
    ],
    orthogonalise_pairs=[("euclidean_distance", "geodesic_distance")],
    subgoal_degrees=[4],
    verbose=True,
    save=False,
):
    """Fit single-neuron RSA encoding models on matched neurons in each maze,
    then correlate the per-neuron beta vectors across mazes for each feature.
    Returns per-subject correlation results."""
    save_path = RESULTS_DIR / f"{maze_pair[0]}_{maze_pair[1]}_cross_maze_beta_corrs.parquet"
    if not save and save_path.exists():
        if verbose:
            print(f"Loading existing cross-maze beta correlations from {save_path}")
        return pd.read_parquet(save_path)

    if matched_heatmap_df is None:
        matched_heatmap_df = get_matched_place_heatmaps_df(maze_pair=maze_pair)

    # -- precompute design matrices per maze (shared across subjects) --
    Xz_per_maze = {}
    for maze_name in maze_pair:
        simple_maze = mr.get_simple_maze(maze_name)
        model_RDM_dfs = []
        for metric in rsa_metrics:
            kwargs = {"subgoal_degrees": subgoal_degrees} if metric == "subgoal_distance" else {}
            model_RDM_dfs.append(mm.get_maze_RDM_df(simple_maze, metric=metric, kwargs=kwargs))
        places = list(matched_heatmap_df[maze_name].columns)

        X_cols = []
        for Mdf in model_RDM_dfs:
            Mmat = _ensure_df_order(Mdf, places)
            X_cols.append(_vec_upper(Mmat))
        X = np.column_stack(X_cols)

        if orthogonalise_pairs is not None:
            for _pair in orthogonalise_pairs:
                target_name, ref_name = _pair
                if target_name not in rsa_metrics or ref_name not in rsa_metrics:
                    raise ValueError("Orthogonalisation pair names must be in rsa_metrics list.")
                target_idx = rsa_metrics.index(target_name)
                ref_idx = rsa_metrics.index(ref_name)
                X[:, target_idx] = orthogonalise_column_against(X, target_idx, ref_idx)

        pred_scaler = StandardScaler(with_mean=True, with_std=True)
        Xz_per_maze[maze_name] = pred_scaler.fit_transform(X)

    # -- loop over subjects --
    subject_ids = matched_heatmap_df.index.get_level_values(0).unique()
    results = []

    for subject_ID in subject_ids:
        # fit single-neuron betas for each maze
        maze_betas = {}
        for maze_name in maze_pair:
            Xz = Xz_per_maze[maze_name]
            heatmap_df = matched_heatmap_df[maze_name].loc[subject_ID]
            n_neurons = len(heatmap_df)
            neuron_betas = np.full((n_neurons, len(rsa_metrics)), np.nan)

            for i in range(n_neurons):
                tuning = heatmap_df.iloc[i].values.astype(float)
                D_neuron = (tuning[:, None] - tuning[None, :]) ** 2
                y = _vec_upper(D_neuron)
                y_sd = y.std(ddof=0)
                if y_sd == 0:
                    continue
                y_z = (y - y.mean()) / y_sd
                lr = LinearRegression(fit_intercept=True).fit(Xz, y_z)
                neuron_betas[i] = lr.coef_

            maze_betas[maze_name] = neuron_betas

        # correlate beta vectors across mazes
        betas_A = maze_betas[maze_pair[0]]
        betas_B = maze_betas[maze_pair[1]]
        valid = np.isfinite(betas_A[:, 0]) & np.isfinite(betas_B[:, 0])

        for j, metric in enumerate(rsa_metrics):
            r, p = pearsonr(betas_A[valid, j], betas_B[valid, j])
            results.append(
                {
                    "subject_ID": subject_ID,
                    "metric": metric,
                    "pearson_r": r,
                    "p_val": p,
                    "n_neurons": int(valid.sum()),
                }
            )

        if verbose:
            print(f"{subject_ID}: {int(valid.sum())} matched neurons")

    results_df = pd.DataFrame(results)
    if save:
        if verbose:
            print(f"Saving cross-maze beta correlations to {save_path}")
        results_df.to_parquet(save_path)
    return results_df


def cross_maze_rsa_prediction(
    maze_pair=("maze_1", "maze_2"),
    matched_heatmap_df=None,
    rsa_metrics=[
        "euclidean_distance",
        "geodesic_distance",
        "boundary_distance",
        "betweenness_centrality",
        "subgoal_distance",
        "corner",
    ],
    orthogonalise_pairs=[("euclidean_distance", "geodesic_distance")],
    subgoal_degrees=[4],
    verbose=True,
    save=False,
):
    """Cross-maze prediction of neural RDMs using RSA betas.

    For each neuron: fit betas on maze A (all features), then use each feature's
    beta with maze B's design matrix to predict the neural RDM in maze B.
    Correlate predicted vs actual per feature to test which structural features
    generalise across mazes. Both directions (A->B, B->A) are tested.
    """
    save_path = RESULTS_DIR / f"{maze_pair[0]}_{maze_pair[1]}_cross_maze_rsa_prediction.parquet"
    if not save and save_path.exists():
        if verbose:
            print(f"Loading existing cross-maze predictions from {save_path}")
        return pd.read_parquet(save_path)

    if matched_heatmap_df is None:
        matched_heatmap_df = get_matched_place_heatmaps_df(maze_pair=maze_pair)

    # -- precompute design matrices per maze --
    Xz_per_maze = {}
    for maze_name in maze_pair:
        simple_maze = mr.get_simple_maze(maze_name)
        model_RDM_dfs = []
        for metric in rsa_metrics:
            kwargs = {"subgoal_degrees": subgoal_degrees} if metric == "subgoal_distance" else {}
            model_RDM_dfs.append(mm.get_maze_RDM_df(simple_maze, metric=metric, kwargs=kwargs))
        places = list(matched_heatmap_df[maze_name].columns)

        X_cols = []
        for Mdf in model_RDM_dfs:
            Mmat = _ensure_df_order(Mdf, places)
            X_cols.append(_vec_upper(Mmat))
        X = np.column_stack(X_cols)

        if orthogonalise_pairs is not None:
            for _pair in orthogonalise_pairs:
                target_name, ref_name = _pair
                if target_name not in rsa_metrics or ref_name not in rsa_metrics:
                    raise ValueError("Orthogonalisation pair names must be in rsa_metrics list.")
                target_idx = rsa_metrics.index(target_name)
                ref_idx = rsa_metrics.index(ref_name)
                X[:, target_idx] = orthogonalise_column_against(X, target_idx, ref_idx)

        pred_scaler = StandardScaler(with_mean=True, with_std=True)
        Xz_per_maze[maze_name] = pred_scaler.fit_transform(X)

    # -- loop over subjects --
    subject_ids = matched_heatmap_df.index.get_level_values(0).unique()
    results = []

    for subject_ID in subject_ids:
        # fit single-neuron betas for each maze
        maze_betas = {}
        neural_rdms_z = {}
        for maze_name in maze_pair:
            Xz = Xz_per_maze[maze_name]
            heatmap_df = matched_heatmap_df[maze_name].loc[subject_ID]
            n_neurons = len(heatmap_df)
            neuron_betas = np.full((n_neurons, len(rsa_metrics)), np.nan)
            rdms_z = np.full((n_neurons, Xz.shape[0]), np.nan)

            for i in range(n_neurons):
                tuning = heatmap_df.iloc[i].values.astype(float)
                D_neuron = (tuning[:, None] - tuning[None, :]) ** 2
                y = _vec_upper(D_neuron)
                y_sd = y.std(ddof=0)
                if y_sd == 0:
                    continue
                y_z = (y - y.mean()) / y_sd
                lr = LinearRegression(fit_intercept=True).fit(Xz, y_z)
                neuron_betas[i] = lr.coef_
                rdms_z[i] = y_z

            maze_betas[maze_name] = neuron_betas
            neural_rdms_z[maze_name] = rdms_z

        # cross-maze prediction in both directions
        for train_maze, test_maze in [maze_pair, maze_pair[::-1]]:
            betas_train = maze_betas[train_maze]
            rdms_test = neural_rdms_z[test_maze]
            Xz_test = Xz_per_maze[test_maze]

            valid = np.isfinite(betas_train[:, 0]) & np.isfinite(rdms_test[:, 0])

            for j, metric in enumerate(rsa_metrics):
                # per-feature prediction: use beta_j from train maze * column j from test maze
                rs_neurons = []
                for i in np.where(valid)[0]:
                    predicted_rdm = betas_train[i, j] * Xz_test[:, j]
                    r, _ = pearsonr(predicted_rdm, rdms_test[i])
                    rs_neurons.append(r)

                rs_neurons = np.array(rs_neurons)
                mean_r = np.mean(rs_neurons)
                t_stat, p_val = ttest_1samp(rs_neurons, 0)

                results.append(
                    {
                        "subject_ID": subject_ID,
                        "metric": metric,
                        "train_maze": train_maze,
                        "test_maze": test_maze,
                        "mean_r": mean_r,
                        "t_stat": t_stat,
                        "p_val": p_val,
                        "n_neurons": int(valid.sum()),
                    }
                )

        if verbose:
            n_valid = int(
                np.isfinite(maze_betas[maze_pair[0]][:, 0]).sum() & np.isfinite(maze_betas[maze_pair[1]][:, 0]).sum()
            )
            print(f"{subject_ID}: {n_valid} neurons with valid betas")

    results_df = pd.DataFrame(results)
    if save:
        if verbose:
            print(f"Saving cross-maze RSA predictions to {save_path}")
        results_df.to_parquet(save_path)
    return results_df


def plot_cross_maze_rsa_prediction(results_df, ax=None):
    """Plot mean cross-maze prediction r per feature, averaged across directions."""

    # average across both prediction directions per subject x metric
    avg_df = results_df.groupby(["subject_ID", "metric"])["mean_r"].mean().reset_index()

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)

    sns.pointplot(
        data=avg_df,
        x="metric",
        y="mean_r",
        ax=ax,
        color="darkred",
        markers="o",
        linestyle="none",
        errorbar="se",
        err_kws={"linewidth": 2},
        markeredgewidth=0,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("cross-maze prediction r")
    ax.set_xlabel("maze feature")

    # one-sample t-tests per metric
    metrics = avg_df["metric"].unique()
    for metric in metrics:
        vals = avg_df.loc[avg_df["metric"] == metric, "mean_r"].values
        t, p = ttest_1samp(vals, 0)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if sig:
            idx = list(metrics).index(metric)
            ax.text(idx, ax.get_ylim()[1] * 0.95, sig, ha="center", fontsize=10)

    return ax


def get_matched_place_heatmaps_df(
    maze_pair=("maze_1", "maze_2"),
    min_split_half_corr=0.3,
    verbose=True,
):
    """ """
    # load heatmaps
    heatmaps = []
    _maze_pair = f"{maze_pair[0]}.{maze_pair[1]}"
    for maze in maze_pair:
        if verbose:
            print(f"Loading sessions for {maze} maze")
        sessions = gs.get_maze_sessions(
            subject_IDs="all",
            maze_names=[maze],
            days_on_maze=umpd.MAZE_PAIR2VALID_DAYS[_maze_pair][maze],
            with_data=[
                "navigation_df",
                "navigation_spike_rates_df",
                "cluster_metrics",
                "cluster_place_direction_tuning_metrics",
            ],
            must_have_data=True,
        )
        if verbose:
            print(f"generating place heatmaps")
        df = get_population_place_tuning_df(
            sessions=sessions,
            min_split_corr=min_split_half_corr,
            include_multi_unit=False,
        )
        heatmaps.append(df)
    heatmaps_A, heatmaps_B = heatmaps
    # load all clusters matched across maze pair
    all_matches = []
    for subject_ID in SUBJECT_IDS:
        matches = umm.get_cross_maze_matches(
            subject_ID,
            maze_pair,
            single_units=True,
            tuning_metric="place_direction",
            min_split_half_corr=min_split_half_corr,
            return_as="cluster_unique_ID",
            verbose=verbose,
        )
        all_matches.extend(matches)
    all_matches = np.array(all_matches)
    # index matched clusters in maze_A, maze_B heatmaps
    mhm_A = heatmaps_A.droplevel(0, axis=0).loc[all_matches[:, 0]]
    mhm_A.columns = pd.MultiIndex.from_tuples([(maze_pair[0], c) for c in mhm_A.columns])
    mhm_A.index = pd.MultiIndex.from_tuples(
        [(s.split(".")[0], f"matched_cluster_{i}") for i, s in enumerate(mhm_A.index)],
    )
    mhm_B = heatmaps_B.droplevel(0, axis=0).loc[all_matches[:, 1]]
    mhm_B.columns = pd.MultiIndex.from_tuples([(maze_pair[1], c) for c in mhm_B.columns])
    mhm_B.index = pd.MultiIndex.from_tuples(
        [(s.split(".")[0], f"matched_cluster_{i}") for i, s in enumerate(mhm_B.index)]
    )
    # combine
    mhm_df = pd.concat([mhm_A, mhm_B], axis=1)
    return mhm_df
