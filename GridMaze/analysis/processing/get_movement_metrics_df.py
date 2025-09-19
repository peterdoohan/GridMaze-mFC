""" """

# %% Imports
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from GridMaze.analysis.core import load_data
from GridMaze.analysis.core import convert
from GridMaze.analysis.cluster_tuning import movement as mv

# %% Global Variables
FRAME_RATE = 60

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
    speeds, velocities, tangential_acc = mv.get_movement_tuning_data(
        navigation_df, navigation_only=False
    )  # filter for nav later
    navigation_df[("speed", "")] = speeds
    navigation_df[("velocity", "x")] = velocities[:, 0]
    navigation_df[("velocity", "y")] = velocities[:, 1]

    # get split half metrics
    metrics = []
    for cluster in cluster_unique_IDs:
        if cluster in single_units:
            cluster_df = navigation_df.copy()
            cluster_df[("firing_rate", "")] = navigation_spike_rates_df.firing_rate[cluster]
            metrics.append(get_split_half_metrics(cluster_df, navigation_only=navigation_only, n_splits=50))
        else:
            metrics.append(
                {
                    x: np.nan
                    for x in [
                        ("speed", "mean_corr"),
                        ("speed", "min"),
                        ("speed", "max"),
                        ("velocity", "mean_corr"),
                        ("velocity", "min_x"),
                        ("velocity", "min_y"),
                        ("velocity", "max_x"),
                        ("velocity", "max_y"),
                    ]
                }
            )
    metrics_df = pd.DataFrame(metrics)
    metrics_df.columns = pd.MultiIndex.from_tuples(metrics_df.columns)
    metrics_df[("cluster_unique_ID", "")] = cluster_unique_IDs
    metrics_df[("single_unit", "")] = [c in single_units for c in cluster_unique_IDs]
    return metrics_df


def get_split_half_metrics(
    cluster_df,
    navigation_only=True,
    speed_range=(0, 0.3),
    speed_bin_size=0.025,
    n_splits=20,
):
    """ """
    if navigation_only:
        cluster_df = cluster_df[cluster_df.trial_phase == "navigation"]
    trials = cluster_df.trial.unique()
    mid = len(trials) // 2
    speed_corrs, vel_corrs = [], []
    speed_min, vel_min = [], []
    speed_max, vel_max = [], []
    for _ in range(n_splits):
        trials_shuffled = np.random.permutation(trials)
        split_1 = cluster_df[cluster_df.trial.isin(trials_shuffled[:mid])].copy()
        split_2 = cluster_df[cluster_df.trial.isin(trials_shuffled[mid:])].copy()
        speed_curve_1 = _get_speed_tuning_curve(split_1, range=speed_range, bin_size=speed_bin_size)
        speed_curve_2 = _get_speed_tuning_curve(split_2, range=speed_range, bin_size=speed_bin_size)
        speed_corrs.append(speed_curve_1.corr(speed_curve_2, method="spearman"))
        speed_min.append(speed_curve_1.idxmin())
        speed_max.append(speed_curve_1.idxmax())
        vel_curve_1 = _get_velocity_tuning(split_1)
        vel_curve_2 = _get_velocity_tuning(split_2)
        vel_corrs.append(vel_curve_1.corr(vel_curve_2, method="spearman"))
        vel_min.append(vel_curve_1.idxmin())
        vel_max.append(vel_curve_1.idxmax())
    vel_min = np.array(vel_min)
    vel_max = np.array(vel_max)
    metrics = {
        ("speed", "mean_corr"): np.nanmean(speed_corrs),
        ("speed", "min"): np.nanmean(speed_min),
        ("speed", "max"): np.nanmean(speed_max),
        ("velocity", "mean_corr"): np.nanmean(vel_corrs),
        ("velocity", "min_x"): np.nanmean(vel_min[:, 0]),
        ("velocity", "min_y"): np.nanmean(vel_min[:, 1]),
        ("velocity", "max_x"): np.nanmean(vel_max[:, 0]),
        ("velocity", "max_y"): np.nanmean(vel_max[:, 1]),
    }
    return metrics


def _get_speed_tuning_curve(cluster_df, range=(0, 0.3), bin_size=0.025, min_occupancy=0.5):
    bin_edges = np.arange(range[0], range[1] + bin_size, bin_size)
    cluster_df["speed_bin"] = pd.cut(cluster_df[f"speed"], bins=bin_edges, labels=(bin_edges[:-1] + bin_size / 2))
    # filter occupancy out
    grouped = cluster_df.groupby("speed_bin", observed=True).firing_rate
    occupancy = grouped.count()
    tuning_curve = grouped.mean()
    occupancy_mask = occupancy.lt(min_occupancy * FRAME_RATE)
    tuning_curve[occupancy_mask] = np.nan
    return tuning_curve


def _get_velocity_tuning(cluster_df, range_x=(0, 0.3), range_y=(-0.3, 0.3), bin_size=0.05, min_occupancy=0.5):
    bin_edges_x = np.arange(range_x[0], range_x[1] + bin_size, bin_size)
    bin_edges_y = np.arange(range_y[0], range_y[1] + bin_size, bin_size)
    cluster_df[("velocity_binned", "x")] = pd.cut(
        cluster_df.velocity.x, bins=bin_edges_x, labels=(bin_edges_x[:-1] + bin_size / 2)
    )
    cluster_df[("velocity_binned", "y")] = pd.cut(
        cluster_df.velocity.y, bins=bin_edges_y, labels=(bin_edges_y[:-1] + bin_size / 2)
    )
    # group over bins
    grouped = cluster_df.groupby([("velocity_binned", "x"), ("velocity_binned", "y")], observed=True).firing_rate
    velocity_tuning = grouped.mean()
    occupancy_mask = grouped.count().lt(min_occupancy * FRAME_RATE)
    velocity_tuning[occupancy_mask] = np.nan
    return velocity_tuning
