"""This modules calculates cluster rates aligned to left and right turns during navigiation"""
# %% Imports
import numpy as np
import pandas as pd
from ..core import load_data 

# %% Global variables
FRAME_RATE = 60
# %% Get basic action aligned rates df, now included saved in analysis data and loaded in session objects (if specified)


def get_basic_action_aligned_rates_df(
    processed_data_path,
    analysis_data_path,
    actions=["turn_left", "go_forward", "turn_right"],
    window=(-3, 3),
):
    """
    Returns a dataframe with firing rates aligned to basic actions (turn_left, go_forward, turn_right). Where rows are cluster x action -> rates within window
    INPUT:
        - subject_session_path in preprocessing/analysis_data folders: eg, 'm8/2022-07-05-135156'
        - actions: list of basic actions to align to (not including 'go_back' actions from now by default)
        - window: tuple of time window to align to (in seconds)
        - frame_rate: frame rate of video (default 60 Hz)
    """
    # load data
    try:
        navigation_df = load_data.load(analysis_data_path / "frames.navigation.parquet")
        navigation_spike_rates_df = load_data.load(analysis_data_path / "frames.spikeRates.parquet")
    except FileNotFoundError:
        print("Missing requisit processed/analysis data to run get_basic_action_aligned_rates_df. Returning None")
        return None
    navigation_rates_df = pd.concat(
        (navigation_df, navigation_spike_rates_df.reset_index(drop=True)), axis=1
    )
    # process basic action aligned rates
    pre_win, post_win = [w * FRAME_RATE for w in window]
    all_action_aligned_rates_dfs = []
    for action in actions:
        action_rates_df = navigation_rates_df[
            navigation_rates_df.action.basic == action
        ]
        action_inds = action_rates_df.index.to_numpy()
        choice_degrees = action_rates_df.action.choice_degree.to_numpy()
        aligned_timepoints = np.arange(window[0], window[1], 1 / FRAME_RATE)
        cluster_unique_IDs = action_rates_df.firing_rate.columns.to_numpy()
        # initialise action_aligned_rates_df
        columns = [
            ("cluster_unique_ID", ""),
            ("basic_action", ""),
            ("action_number", ""),
            ("choice_degree", ""),
        ]
        columns += [("action_aligned_rates", t) for t in aligned_timepoints]
        init_action_aligned_rates_df = pd.DataFrame(
            columns=pd.MultiIndex.from_tuples(columns)
        )
        init_action_aligned_rates_df[("cluster_unique_ID", "")] = cluster_unique_IDs
        action_aligend_rates_dfs = []
        for i, action_ind in enumerate(action_inds):
            if (
                navigation_rates_df.iloc[action_ind].trial_phase.to_numpy()
                != "navigation"
            ):
                continue  # skip actions that do not occur during navigation
            else:
                action_aligned_rates_df = init_action_aligned_rates_df.copy()
                cluster_action_aligned_rates = navigation_rates_df.iloc[
                    (action_ind + pre_win) : (action_ind + post_win)
                ].firing_rate.T.to_numpy()  # [n_clusters, n_timepoints]
                if (
                    action_aligned_rates_df.action_aligned_rates.shape
                    != cluster_action_aligned_rates.shape
                ):
                    continue  # skip actions where the window ends outside of session (usually last couple of actions in session)
                action_aligned_rates_df[("basic_action", "")] = action
                action_aligned_rates_df[("action_number", "")] = i + 1
                action_aligned_rates_df[("choice_degree", "")] = choice_degrees[i]
                action_aligned_rates_df.action_aligned_rates = (
                    cluster_action_aligned_rates
                )
                action_aligend_rates_dfs.append(action_aligned_rates_df)
        all_action_aligned_rates_dfs.append(pd.concat(action_aligend_rates_dfs, axis=0))
    return pd.concat(all_action_aligned_rates_dfs, axis=0)


# %% 
