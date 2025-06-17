"""
Library for visualising population tuning aligned to egocentric actions
"""

# %% Imports
import numpy as np
import pandas as pd
from regex import B
from scipy.stats import ttest_1samp, zscore


from GridMaze.analysis.cluster_tuning import actions as act
from GridMaze.analysis.core import load_data
from GridMaze.analysis.core import convert

# %% Global Variables

FRAME_RATE = 60  # Hz

# %% Functions


def test(processed_data_path, analysis_data_path, forced_only=True, window=(-3, 3), step_size=0.25):
    """
    note only loads actions during navigation

    if step_size == False, aligned rates returned at frame rate
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
    # get action aligned rates
    action_tuning_df = act._get_basic_action_tuning(
        navigation_rates_df, actions=["turn_left", "go_forward", "turn_right", "go_back"], window=window
    )
    if step_size:  # average rates within each downsampled step
        aligned_rates = action_tuning_df.action_aligned_rates  # sampled at frame rate
        combine_frames = int(FRAME_RATE * step_size)
        groups = np.arange(aligned_rates.shape[1]) // combine_frames
        ds_rates = aligned_rates.T.groupby(groups).mean().T
        orig_times = aligned_rates.columns.to_numpy()
        new_times = orig_times.reshape(-1, combine_frames).mean(axis=1)
        ds_rates.columns = pd.MultiIndex.from_product([["action_aligned_rates"], new_times])
        action_tuning_df = pd.concat(
            [action_tuning_df.drop(columns=["action_aligned_rates"], level=0), ds_rates], axis=1
        )
    # if forced_only:
    #     action_tuning_df = action_tuning_df[action_tuning_df.choice_degree.gt(2)]
    cluster_unique_IDs = action_tuning_df.cluster_unique_ID.unique()
    # metrics df
    for cluster in cluster_unique_IDs:
        if cluster not in single_units:
            continue
        cluster_df = action_tuning_df[action_tuning_df.cluster_unique_ID == cluster]
        return cluster_df
        split_corrs = get_split_half_corr(cluster_df)
        mean_split_corr = np.mean(split_corrs)


def get_action_split_half_corr(cluster_df, action_type="free", n=100, alpha=0.01):
    """ """
    if action_type == "free":
        df = cluster_df[cluster_df.choice_degree.gt(2)]
    elif action_type == "forced":
        df = cluster_df[cluster_df.choice_degree.le(2)]
    else:
        raise ValueError("action_type must be 'free' or 'forced'")
    actions = ["turn_left", "turn_right", "go_forward"]
    left_df, right_df, forward_df = [cluster_df[cluster_df.basic_action == action] for action in actions]
    left_ids, right_ids, forward_ids = [df.action_number.values for df in [left_df, right_df, forward_df]]
    mid_left, mid_right, mid_forward = [len(ids) // 2 for ids in [left_ids, right_ids, forward_ids]]
    corrs = []
    for i in range(n):
        split_1_tuning, split_2_tuning = [], []
        for df, a_ids, mid in zip(
            [left_df, right_df, forward_df], [left_ids, right_ids, forward_ids], [mid_left, mid_right, mid_forward]
        ):
            shuffle_ids = a_ids.copy()
            np.random.shuffle(shuffle_ids)
            split_1_ids = shuffle_ids[:mid]
            split_2_ids = shuffle_ids[mid:]
            split_1_tuning.append(df[df.action_number.isin(split_1_ids)].action_aligned_rates.mean())
            split_2_tuning.append(df[df.action_number.isin(split_2_ids)].action_aligned_rates.mean())
        left_1, right_1, forward_1 = split_1_tuning
        left_2, right_2, forward_2 = split_2_tuning
        LF_1 = left_1 - forward_1
        LF_2 = left_2 - forward_2
        RF_1 = right_1 - forward_1
        RF_2 = right_2 - forward_2
        LR_1 = left_1 - right_1
        LR_2 = left_2 - right_2
        split_1 = np.hstack([LF_1, RF_1, LR_1])
        split_2 = np.hstack([LF_2, RF_2, LR_2])
        split_corr = np.corrcoef(split_1, split_2)[0, 1]
        corrs.append(split_corr)
    result = ttest_1samp(corrs, 0, alternative="greater")
    p_val = result.pvalue
    mean_corr = np.mean(corrs)
    sig = True if p_val < alpha else False
    return mean_corr, sig, p_val


def get_free_forced_split_half_corr(cluster_df, n=100, alpha=0.01):
    """ """
    free_df = cluster_df[cluster_df.choice_degree.gt(2)]
    forced_df = cluster_df[cluster_df.choice_degree.le(2)]
    actions = ["turn_left", "turn_right", "go_forward"]
    # data for free choice actions
    free_dfs = [free_df[free_df.basic_action == action] for action in actions]
    free_ids = [df.action_number.values for df in free_dfs]
    free_mids = [len(ids) // 2 for ids in free_ids]
    # data for forced choice actions
    forced_dfs = [forced_df[forced_df.basic_action == action] for action in actions]
    forced_ids = [df.action_number.values for df in forced_dfs]
    forced_mids = [len(ids) // 2 for ids in forced_ids]
    corrs = []
    for i in range(n):
        split_1, split_2 = [], []  # (free, forced)
        for dfs, ids, mids in [(free_dfs, free_ids, free_mids), (forced_dfs, forced_ids, forced_mids)]:
            split_1_tuning, split_2_tuning = [], []
            for df, a_ids, mid in zip(dfs, ids, mids):
                shuffle_ids = a_ids.copy()
                np.random.shuffle(shuffle_ids)
                split_1_ids = shuffle_ids[:mid]
                split_2_ids = shuffle_ids[mid:]
                split_1_tuning.append(df[df.action_number.isin(split_1_ids)].action_aligned_rates.mean())
                split_2_tuning.append(df[df.action_number.isin(split_2_ids)].action_aligned_rates.mean())
            split_1.append(split_1_tuning)  # left, right, forward
            split_2.append(split_2_tuning)
        free_1, forced_1 = split_1
        free_2, forced_2 = split_2
        free_m_forced_1 = np.hstack(
            [a - b for a, b in zip(free_1, forced_1)]
        )  # free_L - forced_L, free_R - forced_R, etc.
        free_m_forced_2 = np.hstack([a - b for a, b in zip(free_2, forced_2)])
        split_corr = np.corrcoef(free_m_forced_1, free_m_forced_2)[0, 1]
        corrs.append(split_corr)
    result = ttest_1samp(corrs, 0, alternative="greater")
    p_val = result.pvalue
    mean_corr = np.mean(corrs)
    sig = True if p_val < alpha else False
    return mean_corr, sig, p_val
