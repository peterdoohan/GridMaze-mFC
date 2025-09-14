""" """

# %% Imports
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from GridMaze.analysis.core import load_data
from GridMaze.analysis.core import convert
from GridMaze.analysis.cluster_tuning import movement as mv

# %% Global Variables


# %% Functions


def get_movement_metrics_df(processed_data_path, analysis_data_path, navigation_only=False):
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

    # get movement data
    speeds, tangential_acc = mv.get_movement_tuning_data(navigation_df)
    navigation_df[("speed", "")] = speeds
    navigation_df[("tangential_acc", "")] = tangential_acc

    # get split half metrics
    metrics = []
    for cluster in cluster_unique_IDs:
        cluster_df = navigation_df.copy()
        cluster_df[("firing_rate", "")] = navigation_spike_rates_df.firing_rate[cluster]
        metrics.append(get_split_half_metrics(cluster_df, navigation_only=navigation_only, n_splits=50))
    metrics_df = pd.DataFrame(metrics)
    metrics_df.columns = pd.MultiIndex.from_tuples(metrics_df.columns)
    metrics_df[("cluster_unique_ID", "")] = cluster_unique_IDs
    metrics_df[("single_unit", "")] = [c in single_units for c in cluster_unique_IDs]
    return metrics_df


def get_split_half_metrics(
    cluster_df,
    navigation_only=False,
    speed_range=(0, 0.3),
    acc_range=(-3, 3),
    speed_bin_size=0.025,
    acc_bin_size=0.25,
    n_splits=50,
):
    """ """
    if navigation_only:
        cluster_df = cluster_df[cluster_df.trial_phase == "navigation"]
    trials = cluster_df.trial.unique()
    mid = len(trials) // 2
    speed_corrs, acc_corrs = [], []
    speed_min, acc_min = [], []
    speed_max, acc_max = [], []
    for _ in range(n_splits):
        trials_shuffled = np.random.permutation(trials)
        split_1 = cluster_df[cluster_df.trial.isin(trials_shuffled[:mid])].copy()
        split_2 = cluster_df[cluster_df.trial.isin(trials_shuffled[mid:])].copy()
        speed_curve_1 = _get_tuning_curve(split_1, metric="speed", range=speed_range, bin_size=speed_bin_size)
        speed_curve_2 = _get_tuning_curve(split_2, metric="speed", range=speed_range, bin_size=speed_bin_size)
        speed_corrs.append(speed_curve_1.corr(speed_curve_2, method="spearman"))
        speed_min.append(speed_curve_1.idxmin())
        speed_max.append(speed_curve_1.idxmax())
        acc_curve_1 = _get_tuning_curve(split_1, metric="tangential_acc", range=acc_range, bin_size=acc_bin_size)
        acc_curve_2 = _get_tuning_curve(split_2, metric="tangential_acc", range=acc_range, bin_size=acc_bin_size)
        acc_corrs.append(acc_curve_1.corr(acc_curve_2, method="spearman"))
        acc_min.append(acc_curve_1.idxmin())
        acc_max.append(acc_curve_1.idxmax())
    metrics = {
        ("speed", "mean_corr"): np.mean(speed_corrs),
        ("speed", "min"): np.mean(speed_min),
        ("speed", "max"): np.mean(speed_max),
        ("tangential_acc", "mean_corr"): np.mean(acc_corrs),
        ("tangential_acc", "min"): np.mean(acc_min),
        ("tangential_acc", "max"): np.mean(acc_max),
    }
    return metrics


def _get_tuning_curve(cluster_df, metric="speed", range=(0, 0.3), bin_size=0.025, occupancy_proportion=0.005):
    bin_edges = np.arange(range[0], range[1] + bin_size, bin_size)
    cluster_df[f"{metric}_bin"] = pd.cut(
        cluster_df[f"{metric}"], bins=bin_edges, labels=(bin_edges[:-1] + bin_size / 2)
    )
    # filter occupancy out
    occupancy = cluster_df.groupby(f"{metric}_bin", observed=True).size()
    occ_threshold = int(len(cluster_df) * occupancy_proportion)
    invalid_bins = occupancy[occupancy < occ_threshold].index
    tuning_curve = cluster_df.groupby(f"{metric}_bin", observed=True).firing_rate.mean()
    tuning_curve[invalid_bins] = np.nan
    return tuning_curve
