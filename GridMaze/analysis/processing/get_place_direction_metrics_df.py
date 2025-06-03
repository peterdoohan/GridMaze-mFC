"""
Get place-direction distance tuning df, with split halves correlation values used to filter
for distance tuned neurons
"""

# %% Imports
import pandas as pd
import numpy as np

from GridMaze.maze import representations as mr
from GridMaze.analysis.core import load_data
from GridMaze.analysis.core import convert
from GridMaze.analysis.cluster_tuning import spatial

from scipy.stats import ttest_1samp

# %% Global Variables


# %% functions


def get_place_direcion_tuning_metrics_df(
    processed_data_path,
    analysis_data_path,
    n_reps=100,
    alpha=0.01,
    navigation_only=True,
    moving_only=True,
    exclude_time_at_goal=True,
    max_steps_from_goal=30,
    min_occupancy=1,
):
    """
    Implement a random split halves correlation to determine place-direction "tuned" cells
    """
    # load data
    session_info = load_data.load(processed_data_path / "session_info.json")
    cluster_metrics = load_data.load(processed_data_path / "clusters.metrics.htsv")
    navigation_df = load_data.load(analysis_data_path / "frames.navigation.parquet")
    spike_rates_df = load_data.load(analysis_data_path / "frames.spikeRates.parquet")
    spike_rates_df.reset_index(drop=True, inplace=True)
    navigation_rates_df = pd.concat([navigation_df, spike_rates_df], axis=1)
    # get single units
    cluster_unique_IDs = spike_rates_df.firing_rate.columns.to_numpy()
    single_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID.values
    single_units = convert.cluster_IDs2scluster_unique_IDs(session_info, single_units)
    # get maze
    simple_maze = mr.simple_maze(session_info["maze_structure"])
    trials = navigation_rates_df.trial.dropna().unique()
    mid = len(trials) // 2
    # init corr df
    corrs_df = pd.DataFrame(index=cluster_unique_IDs, columns=range(n_reps), data=np.nan)
    for i in range(n_reps):
        trials_shuffled = np.random.permutation(trials)
        split_heatmaps = []
        for _trials in [trials_shuffled[:mid], trials_shuffled[mid:]]:
            _rates_df = navigation_rates_df[navigation_rates_df.trial.isin(_trials)]
            split_heatmaps.append(
                spatial._get_place_direction_df(
                    simple_maze,
                    _rates_df,
                    navigation_only,
                    moving_only,
                    exclude_time_at_goal,
                    min_occupancy,
                    max_steps_from_goal,
                )
            )
        split_1, split_2 = split_heatmaps
        for cluster in cluster_unique_IDs:
            if cluster not in single_units:
                continue
            split_1_cluster = split_1.loc[cluster]
            split_2_cluster = split_2.loc[cluster]
            corr = split_1_cluster.corr(split_2_cluster, method="spearman")
            corrs_df.loc[cluster, i] = corr
    cluster_pvals = corrs_df.apply(lambda x: ttest_1samp(x, 0, alternative="greater")[1], axis=1)
    cluster_mean_corr = corrs_df.mean(axis=1)
    # make output metics df
    metrics_df = pd.DataFrame(index=cluster_unique_IDs)
    metrics_df[("place_direction_tuned", "")] = cluster_pvals.lt(alpha)
    metrics_df[("split_half_corr", "value")] = cluster_mean_corr
    metrics_df[("split_half_corr", "pval")] = cluster_pvals
    metrics_df[("single_unit", "")] = [True if c in single_units else False for c in cluster_unique_IDs]
    metrics_df.columns = pd.MultiIndex.from_tuples(metrics_df.columns)
    return metrics_df
