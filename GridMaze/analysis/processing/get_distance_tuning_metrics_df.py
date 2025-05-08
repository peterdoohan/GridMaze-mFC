"""
Updated lib for calculating distance to goal tuning parameters (define basis fits etc.)
and saving out results to dataframe (clusters.DistanceTuningMetrics.parquet)
"""

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from GridMaze.analysis.core import load_data
from GridMaze.analysis.core import convert

from GridMaze.analysis.cluster_tuning import distance_to_goal as dtg
from scipy.stats import ttest_1samp, zscore
from scipy.ndimage import gaussian_filter1d

# %% Global Variables


# %% Functions


def get_distance_tuning_metrics_df(
    processed_data_path,
    analysis_data_path,
    distance_metrics=("distance_to_goal", "geodesic"),
    max_steps_to_goal=30,
    moving_only=False,
    bin_spacing=0.1,
):
    """ """
    # load data
    session_info = load_data.load(processed_data_path / "session_info.json")
    cluster_metrics = load_data.load(processed_data_path / "clusters.metrics.htsv")
    navigation_df = load_data.load(analysis_data_path / "frames.navigation.parquet")
    navigation_spike_rates_df = load_data.load(analysis_data_path / "frames.spikeRates.parquet")
    navigation_spike_rates_df.reset_index(drop=True, inplace=True)
    cluster_unique_IDs = navigation_spike_rates_df.firing_rate.columns.to_numpy()

    # get single units
    single_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID.values
    single_units = convert.cluster_IDs2scluster_unique_IDs(session_info, single_units)
    # isolate relevant navigation columns
    distance_info = navigation_df[
        [("goal", ""), ("trial", ""), ("moving", ""), ("steps_to_goal", "future"), distance_metrics]
    ].droplevel(1, axis=1)

    for cluster in cluster_unique_IDs:
        if cluster not in single_units:
            continue
        cluster = "m2.2022-07-04.maze_cluster99"
        print(cluster)
        cluster_rates = navigation_spike_rates_df.xs(cluster, level=1, axis=1)
        distance_rates_df = pd.concat([distance_info, cluster_rates], axis=1)
        distance_tuning_df = dtg.get_distance_to_goal_tuning_df(
            distance_rates_df,
            metrics=distance_metrics,
            bin_spacing=bin_spacing,
            max_steps_to_goal=max_steps_to_goal,
            moving_only=moving_only,
        )
        return distance_tuning_df
        is_distance_tuned = _is_distance_tuned(distance_tuning_df, n_reps=100)
        f, ax = plt.subplots(1, 1, figsize=(3, 1.5), clear=True)
        dtg.plot_distance_tuning(
            distance_tuning_df,
            distance_metrics,
            ax=ax,
            color="darkcyan",
            smooth_SD=False,
        )
        ax.set_title(f"distance tuned: {is_distance_tuned}", fontsize=8)
        return

    return


def _is_distance_tuned(distance_tuning_df, n_reps=500, alpha=0.005):
    """ """
    trials = distance_tuning_df.trial.unique()
    mid = len(trials) // 2
    corrs = []
    for _ in range(n_reps):
        trials_shuffled = np.random.permutation(trials)
        split_1 = distance_tuning_df[distance_tuning_df.trial.isin(trials_shuffled[:mid])]
        split_2 = distance_tuning_df[distance_tuning_df.trial.isin(trials_shuffled[mid:])]
        curve_1 = split_1.distance.mean()
        curve_2 = split_2.distance.mean()
        corrs.append(curve_1.corr(curve_2))
    result = ttest_1samp(corrs, 0, alternative="greater")
    p_val = result.pvalue
    return p_val
    if p_val < alpha:
        return True
    else:
        return False
