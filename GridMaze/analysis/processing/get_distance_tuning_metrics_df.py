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


def is_distance_tuned(distance_tuning_df, n_reps=1_000, alpha=0.05):
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


def is_distance_tuned_vectorized(df, n_reps=1_000, alpha=0.05):
    """
    C
    """
    trial_curves = df.distance.values
    n_trials, n_bins = trial_curves.shape
    mid = n_trials // 2
    # Make one random-matrix and argsort to get n_reps independent shuffles
    randmat = np.random.rand(n_reps, n_trials)
    shuf_idx = np.argsort(randmat, axis=1)  # shape (n_reps, n_trials)
    idx1 = shuf_idx[:, :mid]
    idx2 = shuf_idx[:, mid:]
    # split halve tuning curves
    means1 = np.nanmean(trial_curves[idx1], axis=1)
    means2 = np.nanmean(trial_curves[idx2], axis=1)
    # Pearson r = sum(c1*c2) / sqrt(sum(c1^2)*sum(c2^2)), acounting for NaNs
    m1 = np.nanmean(means1, axis=1, keepdims=True)
    m2 = np.nanmean(means2, axis=1, keepdims=True)
    c1 = means1 - m1
    c2 = means2 - m2
    num = np.nansum(c1 * c2, axis=1)
    denom = np.sqrt(np.nansum(c1**2, axis=1) * np.nansum(c2**2, axis=1))
    corrs = num / denom
    p_val = ttest_1samp(corrs, 0, alternative="greater").pvalue
    return p_val < alpha
