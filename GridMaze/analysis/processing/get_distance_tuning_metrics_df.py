"""
Updated lib for calculating distance to goal tuning parameters (define basis fits etc.)
and saving out results to dataframe (clusters.DistanceTuningMetrics.parquet)
"""

# %% Imports
import numpy as np

from GridMaze.analysis.cluster_tuning import distance_to_goal as dtg
from scipy.stats import ttest_1samp

# %% Global Variables


# %% Functions


def get_distance_tuning_metrics_df(processed_data_path, analysis_data_path):
    """ """
    navigation_df = None
    return


def is_distance_tuned(distance_tuning_df, n_reps=500, alpha=0.05):
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
    if p_val < alpha:
        return True
    else:
        return False
