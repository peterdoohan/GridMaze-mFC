"""
Library for visualising population tuning aligned to egocentric actions
"""

# %% Imports
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, spearmanr, zscore


from GridMaze.analysis.cluster_tuning import actions as act
from GridMaze.analysis.core import load_data
from GridMaze.analysis.core import convert

# %% Global Variables

FRAME_RATE = 60  # Hz

# %% Functions


def get_egocentric_action_tuning_metrics_df(
    processed_data_path,
    analysis_data_path,
    actions=["turn_left", "go_forward", "turn_right"],
    action_types=["all", "free", "forced"],
    window=(-3, 3),
    step_size=0.25,
):
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
    action_tuning_df = act._get_basic_action_tuning(navigation_rates_df, actions=actions, window=window)
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
    cluster_unique_IDs = action_tuning_df.cluster_unique_ID.unique()
    # metrics df
    _action_types = [f"{a}_action" for a in action_types]
    conds = _action_types + ["free_vs_forced"]
    metrics = ["value", "p_value", "sig"]
    cols = pd.MultiIndex.from_tuples(
        [("single_unit", "", "")]
        + [("split_half_corr", cond, x) for cond in conds for x in metrics]
        + [("pref_action", at, x) for at in _action_types for x in ["name", "frac", "t_max", "factor"]]
    )
    metrics_df = pd.DataFrame(index=cluster_unique_IDs, columns=cols, data=np.nan)
    # fix boolian dtype columns
    metrics_df[("single_unit", "", "")] = False
    for cond in conds:
        metrics_df[("split_half_corr", cond, "sig")] = False
    # fix string type columns
    for at in _action_types:
        metrics_df[("pref_action", at, "name")] = ""
    for cluster in cluster_unique_IDs:
        if cluster not in single_units:
            continue
        cluster_df = action_tuning_df[action_tuning_df.cluster_unique_ID == cluster]
        metrics_df.loc[cluster, ("single_unit", "", "")] = True
        # get cells preferred action
        for action_type in action_types:
            label = f"{action_type}_action"

            pref_action, pre_action_frac, t_max, pref_action_factor = get_pref_action(
                cluster_df, action_type=action_type, normalise="zscore", n=50
            )
            metrics_df.loc[cluster, ("pref_action", label, "name")] = pref_action
            metrics_df.loc[cluster, ("pref_action", label, "frac")] = pre_action_frac
            metrics_df.loc[cluster, ("pref_action", label, "t_max")] = t_max
            metrics_df.loc[cluster, ("pref_action", label, "factor")] = pref_action_factor
        # free/forced action split half correlation
        for action_type in action_types:
            if session_info["maze_name"] == "rooms_maze" and action_type == "forced":
                continue  # not forced actions in rooms maze
            label = f"{action_type}_action"
            mean_corr, sig, p_val = get_action_split_half_corr(cluster_df, action_type=action_type)
            metrics_df.loc[cluster, ("split_half_corr", label, "value")] = mean_corr
            metrics_df.loc[cluster, ("split_half_corr", label, "p_value")] = p_val
            metrics_df.loc[cluster, ("split_half_corr", label, "sig")] = sig
        # free_vs_forced action split half correlation
        if session_info["maze_name"] == "rooms_maze":
            continue
        mean_corr, sig, p_val = get_free_forced_split_half_corr(cluster_df)
        metrics_df.loc[cluster, ("split_half_corr", "free_vs_forced", "value")] = mean_corr
        metrics_df.loc[cluster, ("split_half_corr", "free_vs_forced", "p_value")] = p_val
        metrics_df.loc[cluster, ("split_half_corr", "free_vs_forced", "sig")] = sig
    # reindex
    metrics_df.reset_index(inplace=True)
    metrics_df.rename(columns={"index": "cluster_unique_ID"}, inplace=True)
    # add convience column for filtering egocentric action tuned clusters
    metrics_df[("egocentric_action_tuned")] = metrics_df.split_half_corr.all_action.sig
    return metrics_df.sort_index(axis=1)


# %% action preference metrics


def get_pref_action(
    cluster_df,
    action_type="all",
    normalise="zscore",
    n=50,
):
    """
    Estimates the perfered egocentric action over random trial splits, as well as the t_max of that prefered action and
    the factor by which the cell fires for that action over the average of the other actions.
    """
    # filter for action types
    if action_type == "all":
        pass
    elif action_type == "free":
        cluster_df = cluster_df[cluster_df.choice_degree.gt(2)]
    elif action_type == "forced":
        cluster_df = cluster_df[cluster_df.choice_degree.le(2)]
    trials = cluster_df.trial.unique()
    n_trials = len(trials)
    # get pref action over random trial splits
    pref_action_df = pd.DataFrame(index=range(n), columns=["pref_action", "pref_action_factor", "t_max"])
    for i in range(n):
        # get random split of trials
        shuffle_trials = trials.copy()
        np.random.shuffle(shuffle_trials)
        rs_trials = shuffle_trials[: n_trials // 2]  # random split
        df = cluster_df[cluster_df.trial.isin(rs_trials)]
        action_aligned_rates = df.groupby("basic_action").action_aligned_rates.mean().action_aligned_rates
        if normalise == "zscore":
            tcs = action_aligned_rates.values
            action_aligned_rates = pd.DataFrame(
                zscore(tcs, axis=1), index=action_aligned_rates.index, columns=action_aligned_rates.columns
            )
        try:
            idxmax = action_aligned_rates.stack().idxmax()
            pref_action = idxmax[0]
            # get pref action factor
            _all_actions = action_aligned_rates.index.values
            pref = action_aligned_rates.loc[pref_action]
            _other_actions = [a for a in _all_actions if a != pref_action]
            other = action_aligned_rates.loc[_other_actions].mean()
            diff = pref - other
            pref_action_factor = diff.max()
            t_max = diff.idxmax()
        except ValueError:
            pref_action, pref_action_factor, t_max = np.nan, np.nan, np.nan
        pref_action_df.loc[i] = [pref_action, pref_action_factor, t_max]
    # summarise outcmes over splits
    pref_action_counts = pref_action_df.pref_action.value_counts(normalize=True)
    if pref_action_counts.isna().all():
        return np.nan, np.nan, np.nan, np.nan
    pref_action = pref_action_counts.idxmax()
    pre_action_frac = pref_action_counts.max()
    _pref_action_df = pref_action_df[pref_action_df.pref_action == pref_action]
    t_max = pref_action_df.t_max.median()
    pref_action_factor = _pref_action_df.pref_action_factor.mean()
    return pref_action, pre_action_frac, t_max, pref_action_factor


# %% split half metrics
def get_action_split_half_corr(cluster_df, action_type="free", n=50, alpha=0.01):
    """
    check random splits of trials not actions
    """
    if action_type == "free":
        df = cluster_df[cluster_df.choice_degree.gt(2)]
    elif action_type == "forced":
        df = cluster_df[cluster_df.choice_degree.le(2)]
    elif action_type == "all":
        df = cluster_df.copy()
    else:
        raise ValueError("action_type must be 'free' or 'forced'")
    actions = ["turn_left", "turn_right", "go_forward"]
    left_df, right_df, forward_df = [df[df.basic_action == action] for action in actions]
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
        # demean
        left_1, right_1, forward_1 = [a - np.mean(a) for a in [left_1, right_1, forward_1]]
        left_2, right_2, forward_2 = [a - np.mean(a) for a in [left_2, right_2, forward_2]]
        # calculate differences
        LF_1 = left_1 - forward_1
        LF_2 = left_2 - forward_2
        RF_1 = right_1 - forward_1
        RF_2 = right_2 - forward_2
        LR_1 = left_1 - right_1
        LR_2 = left_2 - right_2
        # calculate split half correlation
        split_1 = np.hstack([LF_1, RF_1, LR_1])
        split_2 = np.hstack([LF_2, RF_2, LR_2])
        split_corr = spearmanr(split_1, split_2)[0]
        corrs.append(split_corr)
    result = ttest_1samp(corrs, 0, alternative="greater")
    p_val = result.pvalue
    mean_corr = np.mean(corrs)
    sig = True if p_val < alpha else False
    return mean_corr, sig, p_val


def get_free_forced_split_half_corr(cluster_df, n=50, alpha=0.01):
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
        # free_L - forced_L, free_R - forced_R, etc.
        free_m_forced_1 = np.hstack([a - b for a, b in zip(free_1, forced_1)])
        free_m_forced_2 = np.hstack([a - b for a, b in zip(free_2, forced_2)])
        split_corr = spearmanr(free_m_forced_1, free_m_forced_2)[0]
        corrs.append(split_corr)
    result = ttest_1samp(corrs, 0, alternative="greater")
    p_val = result.pvalue
    mean_corr = np.mean(corrs)
    sig = True if p_val < alpha else False
    return mean_corr, sig, p_val
