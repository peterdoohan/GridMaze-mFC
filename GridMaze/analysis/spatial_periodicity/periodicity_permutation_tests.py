"""Updated version of cluster_permutation_tests.py, using a new method for generating heatmap permutations and quantifying
spatial periodicity across the population"""
# %% Imports
import os
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib as mpl
from scipy.stats import zscore, norm
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor
from statsmodels.stats.correlation_tools import cov_nearest


from .. import get_sessions as gs
from ..cluster_tuning import spatial_periodicity as sp
from ...maze import representations as mr
from ...maze import plotting as mp

# %% Gloabl Variables
SMALL_CONSTANT = 1e-3
RESULTS_PATH = "../results/cluster_spatial_periodicity"


# %%
def run_spatial_periodicity_permutation_tests_multiprocesses(maze, n_processes, n_permutations):
    maze_number = [maze] if not maze == "all" else maze
    sessions = gs.get_sessions(
        subject_IDs="all",
        maze_number=maze_number,
        day_on_maze="late",
        with_data=["place_rates_df", "cluster_analysis_metrics_df"],
    )[:4]
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        print("Running spatial periodicity permutation tests")
        executor.map(
            get_spatial_periodicity_permutation_test_summary,
            sessions,
            [n_permutations] * len(sessions),
        )
    return


# %% Main Functions


def get_spatial_periodicity_permutation_test_summary(
    session,
    n_permutations=2000,
    navigation_tuned_only=True,
    max_distance=20,
    save_cluster_summaries=True,
    save_permutation_fits=True,
    save_session_p_values=True,
):
    """This function tests if clusters in a session have a significant amount of spatial periodicity in their firing rate
    heatmaps on a maze. spatial perdiocity is analysed by first calculating the correlation of firing rates n locations
    apart on a maze and then fiting both an exponential decay function and an exponential decay + sinusoid function. The
    frequency of the sinusoid fit + the CPD of the sinusoid fit (additional variance explained in distance correlation curve
    over just the exponential decay fit) of a cluster and compared to a null distribution of values (computed by generating
    permuted heatmaps that match the exponential decay of distance correlation wo/ any periodicity), with a pareto front
    analysis to calculate a p value.

    Note: extensive curve fitting in this analysis necessitates long run times where it is most reasonable to run analyses
    that call this function for many sessions with many permutation on a high performance cluster.


    Args:
        session (session object): with place_rates_df and cluster_analysis_metrics attributes
        n_permutations (int, optional): number of permuted heatmaps include in null distribution
        navigation_tuned_only (bool, optional): Include only navigation tuned neurons. Defaults to True.
        max_distance (int, optional): max distance for distance corrleation curves. Defaults to 20.
        save_cluster_summaries (bool, optional): save out figure (.pdf) of permutation results
        save_permutation_fits (bool, optional): save out df (.parquet) of each clusters permutation fits
        save_session_p_values (bool, optional): save out df (.parquet) of session pvalues and true cluster fit info

    Returns:
        pd.DataFrame: p_value summary df returned if not saved out
    """
    # get initial data structures
    place_rates_df = session.place_rates_df
    if navigation_tuned_only:
        navigation_tuned_clusters = session.get_trial_phase_tuned_clusters(trial_phase="navigation")
        place_rates_df = place_rates_df.loc[navigation_tuned_clusters]
    nan_locations = place_rates_df.columns[place_rates_df.isnull().all(axis=0)]
    simple_maze = session.simple_maze()
    extended_simple_maze = mr.get_extended_simple_maze(simple_maze)
    if max_distance is None:
        max_distance = sp.get_max_maze_distance(extended_simple_maze)
    # get distance correlations curves, then fit with exp decay fn which is used to define location covariance matrices
    logit_fit_params_df = get_logit_fit_params_df(place_rates_df)
    maze_distance_corr_df = get_maze_distance_corr_df(
        place_rates_df,
        extended_simple_maze,
        max_distance=20,
        logit_transformed=True,
        logit_fit_params_df=logit_fit_params_df,
    )
    decay_fit_maze_distance_corr_df = get_decay_fit_maze_distance_corr_df(maze_distance_corr_df, extended_simple_maze)
    cluster2covariance_matrix = get_expected_covariance_matrices(decay_fit_maze_distance_corr_df, extended_simple_maze)
    # initialise permutation data structures
    mean_vector = np.zeros(len(extended_simple_maze.nodes))
    true_fit_params_df = get_spatial_periodicity_fit_params_df(maze_distance_corr_df)
    p_values = []
    for cluster, covariance_matrix in cluster2covariance_matrix.items():
        print(cluster)
        if save_cluster_summaries:
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3), clear=True)
            f.tight_layout()
            f.subplots_adjust(wspace=0.6)
            plot_cluster_heatmap(ax1, cluster, simple_maze, place_rates_df)
            plot_cluster_distance_corrs(ax2, cluster, maze_distance_corr_df, true_fit_params_df, max_distance)
        permuted_normal_heatmaps = np.random.multivariate_normal(
            mean_vector, covariance_matrix.to_numpy(), size=n_permutations
        )
        permutation_heatmaps = pd.DataFrame(data=permuted_normal_heatmaps, columns=covariance_matrix.columns)
        # transform from normal space to firing rate space (see get_logit_fit_parmas fn) to get permuted heatmaps
        logistic_params = logit_fit_params_df.loc[cluster]
        permutation_firing_rate_heatmaps = pd.DataFrame(
            index=permutation_heatmaps.index, columns=permutation_heatmaps.columns
        )
        for permutation, norm_place_rates in permutation_heatmaps.iterrows():
            permutation_firing_rate_heatmaps.loc[permutation] = logistic(norm_place_rates, *logistic_params)
        permutation_firing_rate_heatmaps[nan_locations] = np.nan  # add nans to match og heatmap
        permutation_firing_rate_heatmaps = permutation_firing_rate_heatmaps.clip(lower=0)  # remove neg rates
        # generate distance correlation curves for every permutation
        permutation_maze_distance_corr_df = get_maze_distance_corr_df(
            permutation_firing_rate_heatmaps, extended_simple_maze, max_distance=20, logit_transformed=False
        )
        # fit exp decay and combined models to each permutation (RATE LIMITING STEP)
        permutation_fit_params_df = get_spatial_periodicity_fit_params_df(permutation_maze_distance_corr_df)
        if save_permutation_fits:
            permutation_params_file_path = os.path.join(
                RESULTS_PATH, "cluster_permutation_fit_params", f"{cluster}.parquet"
            )
            permutation_fit_params_df.combined_fit_params.to_parquet(permutation_params_file_path, compression="gzip")
        # use p values to define a null distribution and compute cluster p value
        true_fit_params = true_fit_params_df.loc[cluster].combined_fit_params[["freq", "cpd"]].to_numpy()
        null_distribution_fit_params = permutation_fit_params_df.combined_fit_params[["freq", "cpd"]].to_numpy()
        if save_cluster_summaries:
            pareto_p_value = get_pareto_p_value(
                true_fit_params, null_distribution_fit_params, p_upper_bound=0.5, plot=save_cluster_summaries, ax=ax3
            )
            summary_fig_filepath = os.path.join(RESULTS_PATH, "cluster_permutation_summary_plots", f"{cluster}.pdf")
            f.savefig(summary_fig_filepath)
            plt.close(f)
        else:
            pareto_p_value = get_pareto_p_value(
                true_fit_params, null_distribution_fit_params, p_upper_bound=0.5, plot=False
            )
        p_values.append(pareto_p_value)
    # return/save out p value summary df
    session_p_values_df = true_fit_params_df.combined_fit_params.copy()
    session_p_values_df["p_value"] = p_values
    session_p_values_df["subject_ID"] = session.subject_ID
    session_p_values_df["maze_number"] = session.maze_number
    session_p_values_df["day_on_maze"] = session.day_on_maze
    if save_session_p_values:
        session_p_values_filepath = os.path.join(
            RESULTS_PATH, "session_permutation_p_values", f"{session.name}.parquet"
        )
        session_p_values_df.to_parquet(session_p_values_filepath, compression="gzip")
    else:
        return session_p_values_df


def get_maze_distance_corr_df(
    place_rates_df,
    extended_simple_maze,
    max_distance,
    logit_transformed=True,
    logit_fit_params_df=None,
):
    """Generates a dataframe that contains the correlation between firing rates of locations that a n distance apart
    on a maze (maze distance = shortest path distance between nodes/edges). Works by first generating a dataframe with
    all node pairs that are n distance apart, then mapping the firing rates of those nodes for each cluster.

    Args:
        place_rates_df (pd.DataFrame): clusters, maze locations
        extended_simple_maze (nx object): maze representation where towers and brides are nodes on a graph
        max_distance (int): maximum distance to calculate correlations for, if not speficied calculated emiprically
            from the specified maze
        zscored (bool, optional): applies zscoring to the firing rates. Defaults to True.

    Returns:
        pd.DataFrame: n_clusters, n_distances (multiindex, top level = maze_distance)
    """
    distances = np.arange(1, max_distance + 1)
    distance_pairs_df = []
    for d in distances:
        node_pairs = sp.node_pairs_n_edges_apart(extended_simple_maze, d)
        df = pd.DataFrame(node_pairs, columns=pd.MultiIndex.from_product([[d], ["node1", "node2"]]))
        distance_pairs_df.append(df)
    distance_pairs_df = pd.concat(distance_pairs_df, axis=1)
    distance_corr_df = pd.DataFrame(  # initialise df for distance correlations
        index=place_rates_df.index, columns=pd.MultiIndex.from_product([["maze_distance"], distances])
    )
    if logit_transformed:
        max_limits = logit_fit_params_df["max"].to_numpy(dtype=float).reshape(-1, 1)
        place_rates_array = place_rates_df.to_numpy()
        place_rates_array[place_rates_array < 1e-3] = 1e-2  # avoid small values that cause log(0) errors
        place_rates_array[place_rates_array > max_limits] = (  # avoid values > max fit param that cause log(0) errors
            np.tile(max_limits, (1, place_rates_array.shape[1]))[place_rates_array > max_limits] - 1e-3
        )
        for cluster, place_rates in zip(place_rates_df.index, place_rates_array):  # apply logit transform
            logit_params = logit_fit_params_df.loc[cluster].to_numpy(dtype=float)
            place_rates_df.loc[cluster] = logit(place_rates, *logit_params)
    for cluster, place_rates in place_rates_df.iterrows():
        place2rates = place_rates.to_dict()
        firing_rate_pairs_df = pd.DataFrame(
            {col: distance_pairs_df[col].map(place2rates) for col in distance_pairs_df}
        )  # 10x faster then df.replace(dict)
        distance_correlations = [
            firing_rate_pairs_df[d]["node1"].corr(firing_rate_pairs_df[d]["node2"], method="pearson")
            for d in distances
        ]
        distance_corr_df.loc[cluster] = distance_correlations
    return distance_corr_df.astype(float)  # otherwise values are object from the .map function


def get_decay_fit_maze_distance_corr_df(maze_distance_corr_df, extended_simple_maze, bounds=([0, 0, -1], [5, 5, 1])):
    """Fits an exponential decay function to the maze distance correlations for each cluster and calculates the expected correlation
    at each distance based on just the decay function.

    Args:
        maze_distance_corr_df (pd.DataFrame): generated with get_maze_distance_corr_df
        extended_simple_maze (nx object): maze representation where towers and brides are nodes on a graph
        bounds (tuple, optional): curve fit bounds for the exponential decay function. See multistart_curve_fit
        and exp_decay for details.

    Returns:
        pd.DataFrame: n_clusters, n_distances (multiindex, top level = maze_distance)
    """
    max_distance = sp.get_max_maze_distance(extended_simple_maze)
    distances = maze_distance_corr_df.maze_distance.columns.to_numpy()
    decay_fit_maze_distance_corr_df = pd.DataFrame(
        index=maze_distance_corr_df.index,
        columns=pd.MultiIndex.from_product([["maze_distance"], np.arange(1, max_distance + 1)]),
    )
    for cluster, distance_corrs in maze_distance_corr_df.iterrows():
        lower_bounds, upper_bounds = bounds
        params = multistart_curve_fit(exp_decay, distances, distance_corrs, lower_bounds, upper_bounds)
        decay_fit_maze_distance_corr_df.loc[cluster] = exp_decay(np.arange(1, max_distance + 1), *params)
    return decay_fit_maze_distance_corr_df


def get_expected_covariance_matrices(exp_decay_fit_df, extended_simple_maze):
    """Generates covariance matricies expected by the distance correlations defined by exp decay fit. Starts with a
    geodesic distance matrix [locaions, locations]. Then maps on the corresponding correlation from the exp fits to
    make covariance matrices. Additional step to check matrices are positive semi-definate to later define multivariate
    gaussians, if fails ajusts matrices slightly to meet this condition."""
    distance_corrs_df = exp_decay_fit_df.maze_distance
    node_list = list(extended_simple_maze.nodes)
    node2label = nx.get_node_attributes(extended_simple_maze, "label")
    node_labels = [node2label[node] for node in node_list]
    distance_matrix = nx.floyd_warshall_numpy(extended_simple_maze, nodelist=list(extended_simple_maze.nodes)).astype(
        int
    )
    cluster2covariance_matrices = {}
    for cluster, distance_corrs in distance_corrs_df.iterrows():
        distance_corrs = np.insert(
            distance_corrs.to_numpy(dtype=float), 0, 1
        )  # prepend with 1 so that element correpsond to distance corr
        cov_m = distance_corrs[distance_matrix]
        # check symetric positive semi-definite
        if not (np.all(np.linalg.eigvals(cov_m) >= 0) and np.allclose(cov_m, cov_m.T)):
            cov_m = cov_nearest(cov_m, threshold=1e-12)
        cluster2covariance_matrices[cluster] = pd.DataFrame(index=node_labels, columns=node_labels, data=cov_m)
    return cluster2covariance_matrices


def get_spatial_periodicity_fit_params_df(
    distance_corr_df, bounds=([0, 0, -1, 0, 0, 0], [10, 5, 1, 2, np.pi, 2 * np.pi])
):
    """Fits both exponential dec and combined models (decay + sinusoid) to the distance correlations for each
    cluster/permutation, stores the fit parameters and the fits in the returned dataframe.

    Args:
        distance_corr_df (pd.DataFrame): clusters/permutations, distance correlations
        bounds (tuple, optional): fitting bounds

    Returns:
        pd.DataFrame: cluster/permutation, fit params & fits (multindex)
    """
    lower_bounds, upper_bounds = bounds
    # intialise dataframe
    distances = distance_corr_df.maze_distance.columns.to_numpy()
    fit_params = ["exp_amplitude", "exp_decay_constant", "offset", "sin_amplitude", "freq", "phase", "cpd"]
    distance_corr_cols = pd.MultiIndex.from_product([["decay_distance_corrs", "combined_distance_corrs"], distances])
    fit_params_cols = pd.MultiIndex.from_product([["combined_fit_params"], fit_params]).union(
        pd.MultiIndex.from_product([["decay_fit_params"], fit_params[:3]]), sort=False
    )
    sp_fit_params_df = pd.DataFrame(
        index=distance_corr_df.index, columns=fit_params_cols.union(distance_corr_cols, sort=False)
    )
    for row_label, distance_corrs in distance_corr_df.iterrows():
        distance_corrs = distance_corrs
        exp_decay_params = multistart_curve_fit(
            exp_decay, distances, distance_corrs, lower_bounds[:3], upper_bounds[:3]
        )
        sp_fit_params_df.loc[row_label, ("decay_fit_params")] = exp_decay_params
        fit_exp_decay = exp_decay(distances, *exp_decay_params)
        sp_fit_params_df.loc[row_label, ("decay_distance_corrs")] = fit_exp_decay

        baseline_params = np.append(exp_decay_params, [0, 0, 0])  # use exp decay fit params as baseline
        combined_params = multistart_curve_fit(
            combined_model, distances, distance_corrs, lower_bounds, upper_bounds, baseline_params=baseline_params
        )
        fit_combined = combined_model(distances, *combined_params)
        sp_fit_params_df.loc[row_label, ("combined_distance_corrs")] = fit_combined
        # calculate cpd
        SS_exp = np.sum((distance_corrs - fit_exp_decay) ** 2)
        SS_combined = np.sum((distance_corrs - fit_combined) ** 2)
        cpd = (SS_exp - SS_combined) / SS_exp
        combined_params = np.append(combined_params, cpd)
        sp_fit_params_df.loc[row_label, ("combined_fit_params")] = combined_params
    return sp_fit_params_df.astype(float)


def get_logit_fit_params_df(place_rates_df, bounds=([0, 0, -10], [100, 10, 10])):
    """Fits curves of locations ordered by firing rate curves that have been Normal transformed space with a logit function.
    Parameters allow for a non-linear mapping between firing rate space (Fs) and a normal space (Ns):
        Fs -(logit[params])-> Ns
        Ns -(logistic[params]) -> Fs
    Useful for sampling from multivariate gaussian, see main function.
    """
    sorted_place_rates = np.sort(place_rates_df.to_numpy(), axis=1)  # nans are sorted to the end
    sorted_place_rates = sorted_place_rates[:, : (~np.isnan(sorted_place_rates)).sum(axis=1)[0]]  # remove nans
    percentiles = np.linspace(0.1, 99.9, sorted_place_rates.shape[1])  # avoid infs
    normal_space_percentiles = norm.ppf(percentiles / 100)
    sorted_percentile_place_rates = np.percentile(
        sorted_place_rates, percentiles, axis=1
    ).T  # [n_clusters, n_percentiles]
    # intialise df
    fit_params_df = pd.DataFrame(
        index=place_rates_df.index,
        columns=["max", "scale", "offset"],
    )
    # loop through clusters and do curve fitting
    for cluster, rates in zip(place_rates_df.index, sorted_percentile_place_rates):
        fit_params = multistart_curve_fit(logistic, normal_space_percentiles, rates, bounds[0], bounds[1])
        fit_params_df.loc[cluster] = fit_params
    return fit_params_df.astype(float)


# %% curve fitting functions


def logistic(x, max, scale, offset):
    """Logistic function"""
    return max / (1 + np.exp(-scale * (x - offset)))


def logit(x, max, scale, offset):
    """Logit function, inverse of logistic function"""
    return offset - np.log((max / x) - 1) / scale


def exp_decay(x, amplitude, decay_constant, offset):
    """Exponential decay function"""
    return amplitude * np.exp(-decay_constant * x) + offset


def combined_model(x, exp_amplitude, exp_decay_constant, offset, sin_amplutude, freq, phase):
    """Exponential decay + sinusoid function"""
    return exp_amplitude * np.exp(-exp_decay_constant * x) + sin_amplutude * np.sin(freq * x + phase) + offset


def multistart_curve_fit(
    fitting_function, x, y_true, lower_bounds, upper_bounds, max_itter=150, top_fit_tol=1e-4, baseline_params=None
):
    """Embelished version of scipy.optimize.curve_fit that uses multiple random starting points and returns the average
    fit parameters of the top 3 fits (lowest residuals).

    Args:
        fitting_function (function): function to fit
        lower_bounds/upper_bound (itterable): bounds for fitting function
        max_itter (int, optional): Max number of times to call curve_fit. Defaults to 100.
        top_fit_tol (float, optional): diff between best fits before calling convergance. Defaults to 1e-4.
        baseline_params (itterable, optional): if not None, will compare the fit residuals to a baseline fit and return
            the baseline params if the baseline fit has lower residuals. Defaults to None.

    Returns:
        pd.DataFrame: average fit params
    """
    top_fits = [{"params": None, "residuals": np.inf} for _ in range(3)]
    itter_count = 0
    while itter_count < max_itter:
        p0 = [np.random.uniform(low, high) for low, high in zip(lower_bounds, upper_bounds)]
        try:
            params, _ = curve_fit(fitting_function, x, y_true, p0=p0, bounds=(lower_bounds, upper_bounds))
            y_fit = fitting_function(x, *params)
            residuals = np.sum((y_true - y_fit) ** 2)
            max_residual_in_top = max([f["residuals"] for f in top_fits])
            if residuals < max_residual_in_top:
                worst_top_fit_index = np.argmax([f["residuals"] for f in top_fits])
                top_fits[worst_top_fit_index] = {"params": params, "residuals": residuals}
            top_residuals = [f["residuals"] for f in top_fits]
            if max(top_residuals) - min(top_residuals) < top_fit_tol:
                break

        except Exception as e:
            # Handle any curve_fit exceptions and continue to the next iteration
            pass
        itter_count += 1
    try:
        avg_params = np.mean([f["params"] for f in top_fits if f["params"] is not None], axis=0, dtype=float)
        if baseline_params is not None:
            baseline_fit = fitting_function(x, *baseline_params)
            found_best_fit = fitting_function(x, *avg_params)
            if np.sum((y_true - baseline_fit) ** 2) < np.sum((y_true - found_best_fit) ** 2):
                return baseline_params
    except TypeError:
        avg_params = np.nan
    return avg_params


# %% pareto front functions


def get_pareto_p_value(true_value, null_distribution, p_upper_bound=0.5, plot=False, ax=None):
    """Calculate the p_value of a true value given a null distribution of values. The p_value is calculated by
    finding the last Pareto front that includes the true value, and then calculating the proportion of points in the
    null distribution that are dominated by the true value. If the lower bound of the p_value is greater than the
    p_upper_bound threshold, then the p_value is set to np.nan and the function returns early.
    """
    if pareto_p_lower_bound_check(null_distribution, true_value, p_lower_bound_threshold=p_upper_bound):
        p_value = np.nan
        if plot and ax is not None:
            plot_lower_bound_fail_param_distribution(ax, null_distribution, true_value, p_upper_bound)
    else:
        p_value, (dominated_points, fronts) = get_pareto_fronts(null_distribution, true_value)
        if plot and ax is not None:
            plot_pareto_front_param_distribution(ax, dominated_points, fronts, true_value, p_value)
    return p_value


def get_pareto_fronts(null_distribution, true_value):
    """Calculate the Pareto fronts of a set of points, up until the true value is included in a front."""
    fronts = []
    # append true value to null distribution to get full distribution
    remaining_points = np.append(null_distribution, [true_value], axis=0)
    while len(remaining_points) > 0:
        front = []
        to_remove = []
        for i, point_i in enumerate(remaining_points):
            dominated = False
            for j, point_j in enumerate(remaining_points):
                if i != j and (point_j[0] >= point_i[0] and point_j[1] >= point_i[1]):
                    dominated = True
                    break
            if not dominated:
                front.append(point_i)
                to_remove.append(i)
        remaining_points = np.delete(remaining_points, to_remove, axis=0)
        fronts.append(np.array(front))
        # check if true value is still in remaining points
        if true_value not in remaining_points:
            break
    p_value = (np.vstack(fronts).shape[0] - 1) / null_distribution.shape[0]  # -1 to exclude true value
    return p_value, (remaining_points, fronts)


def pareto_p_lower_bound_check(null_distribution, true_value, p_lower_bound_threshold=0.5):
    """Check if the lower bound of the p_value is greater than the threshold. If so retrun True,
    and not bother with calculating the full p_value. See get_pareto_p_value for details.
    """
    p_lower_bound = np.all(null_distribution > true_value, axis=1).mean()
    if p_lower_bound > p_lower_bound_threshold:
        return True
    else:
        return False


# %% Plotting Functions


def plot_cluster_heatmap(ax, cluster_unique_id, simple_maze, place_rates_df):
    """Plots firing rate heatmap"""
    mp.plot_simple_heatmap(
        simple_maze,
        place_rates_df.loc[cluster_unique_id].to_dict(),
        ax,
        title="",
        value_label="Firing Rate (Hz)",
        node_size=150,
        edge_size=6,
    )


def plot_cluster_distance_corrs(ax, cluster_unique_id, maze_distance_corr_df, true_fit_params_df, max_distance):
    """Plots empirical distance correlation curve and associated exp decay and combined model (exp decay + sinusoid) fits"""
    distances = maze_distance_corr_df.maze_distance.columns.to_numpy()
    true_distance_corr = maze_distance_corr_df.loc[cluster_unique_id].maze_distance.to_numpy(dtype=float)
    decay_fit = true_fit_params_df.loc[cluster_unique_id].decay_distance_corrs.to_numpy(dtype=float)
    combined_fit = true_fit_params_df.loc[cluster_unique_id].combined_distance_corrs.to_numpy(dtype=float)
    ax.plot(distances, true_distance_corr, label="empirical", color="k")
    ax.plot(distances, decay_fit, label="exp decay fit", color="r", alpha=0.5)
    ax.plot(distances, combined_fit, label="combined fit", color="b", alpha=0.5)
    ax.axhline(ls="--", color="k", alpha=0.5)
    ax.set_xlabel("Maze Distance")
    ax.set_ylabel("Correlation")
    ax.legend(fontsize="small")
    ax.set_ylim(-1, 1)
    ax.set_xlim(1, max_distance)


def plot_pareto_front_param_distribution(ax, dominated_points, fronts, real_data, p_value):
    """Plots the null distribution scatter with true value, coloring pareto fronts within the
    distribution and highlighting the last front(red)"""
    colors = mpl.colormaps["cool"](np.linspace(0, 1, len(fronts)))
    last_front = fronts[-1]
    sorted_last_front = last_front[np.lexsort((-last_front[:, 1], last_front[:, 0]))]
    ax.scatter(dominated_points[:, 0], dominated_points[:, 1], s=1, c="k", alpha=0.1)
    for front, color in zip(fronts, colors[::-1]):
        ax.scatter(front[:, 0], front[:, 1], s=1, color=color)
    ax.plot(sorted_last_front[:, 0], sorted_last_front[:, 1], color="red")
    ax.scatter(real_data[0], real_data[1], s=50, c="red", marker="x")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("CPD")
    ax.text(
        0.45,
        0.98,
        f"p = {p_value}",
        transform=ax.transAxes,
        fontsize="small",
    )
    return


def plot_lower_bound_fail_param_distribution(ax, null_distribution, real_data, lower_bound_threshold):
    """Plots null distribution of data and the true value coloring points that dominate the true value
    on either axis."""
    dominated_mask = np.all(null_distribution > real_data, axis=1)
    dominated_points = null_distribution[dominated_mask]
    non_dominated_points = null_distribution[~dominated_mask]
    ax.scatter(dominated_points[:, 0], dominated_points[:, 1], s=1, c="purple", alpha=0.2)
    if len(non_dominated_points) > 0:
        ax.scatter(non_dominated_points[:, 0], non_dominated_points[:, 1], s=1, c="k", alpha=0.1)
    ax.scatter(real_data[0], real_data[1], s=50, c="red", marker="x")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("CPD")
    ax.text(
        0.4,
        0.98,
        f"p > {lower_bound_threshold}",
        transform=ax.transAxes,
        fontsize="small",
    )


# %% Plot results (note results generated on cluster)


def get_permutation_results_df(sig_level=0.05, p_lower_bound_threshold=0.5, high_freq_threshold=2.5):
    fit_results_path = os.path.join(RESULTS_PATH, "session_permutation_p_values")
    session_result_filepaths = [os.path.join(fit_results_path, f) for f in os.listdir(fit_results_path)]
    permutation_results_df = pd.concat([pd.read_parquet(f) for f in session_result_filepaths], axis=0)
    f, axes = plt.subplots(2, 2, figsize=(6, 6), clear=True)
    ax1, ax2, ax3, ax4 = axes.flatten()
    # plot p vs. p curve
    valid_p_values_df = permutation_results_df.dropna()
    sub_freq_thresh_p_values_df = valid_p_values_df[valid_p_values_df.freq < high_freq_threshold]
    for p_value_df, axes, hist_range, sig_color in zip(
        [valid_p_values_df, sub_freq_thresh_p_values_df],
        [(ax1, ax2), (ax3, ax4)],
        [(0, np.pi), (0, high_freq_threshold)],
        ["red", "blue"],
    ):
        sorted_p_values = p_value_df.sort_values(by="p_value").p_value.to_numpy()
        sorted_p_values = sorted_p_values[sorted_p_values < p_lower_bound_threshold]
        sig_p_values = sorted_p_values[sorted_p_values < sig_level]
        non_sig_p_values = sorted_p_values[sorted_p_values >= sig_level]
        axes[0].plot(range(1, len(sig_p_values) + 1), sig_p_values, color=sig_color, lw=2)
        axes[0].plot(
            range(len(sig_p_values), len(sorted_p_values) + 1),
            np.append(sig_p_values[-1], non_sig_p_values),
            color="grey",
            lw=2,
        )
        axes[0].plot([1, len(sorted_p_values)], [0, 0.5], ls="--", color="k", alpha=0.5, lw=0.75)
        axes[0].axhline(0.05, ls="--", color="k", alpha=0.5, lw=0.75)
        axes[0].set_xlabel("Ordered Clusters")
        axes[0].set_ylabel("P Value")
        axes[0].set_ylim(-0.01, 0.5)
        axes[0].set_xlim(1, len(sorted_p_values))
        # plot hist of sig freqs
        sig_freqs = permutation_results_df[permutation_results_df.p_value < sig_level].freq.to_numpy()
        sns.histplot(sig_freqs, bins=50, ax=axes[1], element="step", color=sig_color)
        axes[1].set_xlabel("Frequency")
        axes[1].set_ylabel("Count")
        axes[1].set_xlim(*hist_range)
        axes[1].axvline(high_freq_threshold, ls="--", color="k", alpha=0.5, lw=0.75)
    # get distance to goal tuning of significant clusters
    f2, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), clear=True)
    sig_sin_amps = permutation_results_df[permutation_results_df.p_value < sig_level].sin_amplitude.to_numpy()
    sns.histplot(sig_sin_amps, bins=100, ax=ax1, element="step", color="purple")
    ax1.set_xlabel("Sinusoid Amplitude")
    ax1.set_ylabel("Count")
    amplitude_threshold = 0.25
    ax1.axvline(amplitude_threshold, ls="--", color="k", alpha=0.5, lw=0.75)
    interesting_clusters = permutation_results_df[
        np.logical_and(permutation_results_df.sin_amplitude > 0.25, permutation_results_df.p_value < sig_level)
    ].index.to_numpy()
    distance_tuning_df = get_distance_tuning_of_sig_clusters(interesting_clusters, maze=2, smooth_SD=1)
    distance_tuning_df.columns = pd.Series(distance_tuning_df.columns.astype(float)).round(2).values
    sns.heatmap(distance_tuning_df, ax=ax2, cmap="mako")
    ax2.set_xlabel("Geodesic Distance (m)")
    ax2.set_ylabel("Cluster")
    ax2.set_yticklabels([])
    return permutation_results_df


from scipy.ndimage import gaussian_filter1d


def get_distance_tuning_of_sig_clusters(interesting_clusters, maze=2, smooth_SD=1):
    sessions = gs.get_sessions(maze_number=[maze], with_data=["distance_to_goal_aligned_rates_df"])
    distance_tuning_df = pd.concat(
        [s.distance_to_goal_aligned_rates_df.geodesic_distance_to_goal.average for s in sessions], axis=0
    ).loc[interesting_clusters]
    if smooth_SD:
        distance_tuning_array = distance_tuning_df.to_numpy()
        distance_tuning_array = gaussian_filter1d(distance_tuning_array, axis=1, sigma=smooth_SD)
        distance_tuning_df = pd.DataFrame(
            index=distance_tuning_df.index, columns=distance_tuning_df.columns, data=distance_tuning_array
        )
    distance_tuning_df = distance_tuning_df.apply(lambda x: x / x.max(), axis=1)
    distance_tuning_df = distance_tuning_df.iloc[np.argsort(distance_tuning_df.values.argmax(axis=1))]
    return distance_tuning_df


# %%
