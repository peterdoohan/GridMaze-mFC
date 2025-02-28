"""
Library for processing data to calculate cluster tuning to various angles (angle to goal, head direction etc.)
"""
#%% Imports
import numpy as np
import pandas as pd
from ..core import load_data
from ..core import filter

#%% Global Variables

def get_head_direction_tuning_df(processed_data_path, analysis_data_path, n_bins=120):
    """See get_angle_tuning_df"""
    return get_angle_tuning_df(processed_data_path, analysis_data_path, metric=("head_direction", "value"), n_bins=n_bins)

def get_allocentric_angle_to_goal_tuning_df(processed_data_path, analysis_data_path, n_bins=120):
    """See get_angle_tuning_df"""
    return get_angle_tuning_df(processed_data_path, analysis_data_path, metric=("angle_to_goal", "allocentric"), n_bins=n_bins)

def get_egocentric_angle_to_goal_tuning_df(processed_data_path, analysis_data_path, n_bins=120):
    """See get_angle_tuning_df"""
    return get_angle_tuning_df(processed_data_path, analysis_data_path, metric=("angle_to_goal", "egocentric"), n_bins=n_bins)


def get_angle_tuning_df(processed_data_path, analysis_data_path, metric=("head_direction", "value"), n_bins=120):
    """
    Generates the angle_to_goal or head_direction tuning on all trials and all clusters from a session
    Args:
        processed_data_path (Path): Path to processed data directory
        analysis_data_path (Path): Path to analysis data directory
        metric (tuple): Tuple of (metric, value) to calculate tuning for
            head_direction tuning, metric=("head_direction", "value")
            allocentric_angle_to_goal tuning, metric=("angle_to_goal", "allocentric")
            egocentric_angle_to_goal tuning, metric=("angle_to_goal", "egocentric")
        n_bins (int): Number of bins to split the metric into
    """
    # load data
    try:
        session_info = load_data.load(processed_data_path / "session_info.json")
        trials_df = load_data.load(processed_data_path / "trials.htsv")
        navigation_df = load_data.load(analysis_data_path / "frames.navigation.parquet")
        navigation_spike_rates_df = load_data.load(analysis_data_path / "frames.spikeRates.parquet")
    except FileNotFoundError:
        print("Missing requisit processed/analysis data to run get_head_direction_tuning_df. Returning None")
        return None
    trial2goal = dict(trials_df[["trial", "goal"]].to_numpy())
    navigation_rates_df = pd.concat((navigation_df, navigation_spike_rates_df.reset_index(drop=True)), axis=1)
    navigation_rates_df = filter.filter_navigation_rates_df(navigation_rates_df, moving_only=True)
    # get head direction tuning
    angle_bins = (metric[0], metric[1] + "_bined")
    bins = pd.IntervalIndex.from_breaks(np.linspace(0, 360, num=n_bins + 1, endpoint=True))
    navigation_rates_df[angle_bins] = pd.cut(navigation_rates_df[metric[0]][metric[1]], bins=bins)
    cluster_unique_IDs = navigation_rates_df.firing_rate.columns.to_numpy()
    trials = navigation_rates_df.trial.unique()
    angle_tuning_dfs = []
    for t in trials:
        # get trial info
        trial_info = pd.DataFrame(
            {
                "subject_ID": session_info["subject_ID"],
                "maze_name": session_info["maze_name"],
                "day_on_maze": session_info["day_on_maze"],
                "trial": t,
                "goal": trial2goal[t],
                "cluster_unique_ID": cluster_unique_IDs,
            }
        )
        trial_info.columns = pd.MultiIndex.from_product([trial_info.columns, [""]])
        trial_df = navigation_rates_df[navigation_rates_df.trial == t]
        angle_grouped_rates = trial_df.groupby([angle_bins], observed=True).firing_rate.mean().firing_rate.T
        missing_bins_df = _get_missing_bins_df(bins, angle_grouped_rates) # df with missing bins filled with NaN
        angle_grouped_rates = pd.concat([angle_grouped_rates, missing_bins_df], axis=1)
        angle_grouped_rates = angle_grouped_rates.sort_index(axis=1)
        if metric[1] == "value":
            tuning = "head_direction_tuning"
        else:
            tuning = f"{metric[1]}_{metric[0]}_tuning"
        angle_grouped_rates.columns = pd.MultiIndex.from_product([[tuning], [b.mid for b in angle_grouped_rates.columns]])
        angle_grouped_rates.reset_index(names="cluster_unique_ID", inplace=True)
        angle_tuning_dfs.append(pd.merge(trial_info, angle_grouped_rates, on=[("cluster_unique_ID","")]))    
    return pd.concat(angle_tuning_dfs, axis=0)




def _get_missing_bins_df(all_bins, hd_grouped_rates):
    current_bins = hd_grouped_rates.columns
    missing_bins = np.setdiff1d(all_bins, current_bins)
    return pd.DataFrame(data=np.nan, index=hd_grouped_rates.index, columns=missing_bins)


