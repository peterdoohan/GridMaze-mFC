"""
Library for distance-to-goal alaigned goal decoding.
Eg, build separate decoders for neural activity 1, step from goal, 2 steps from goal, etc.
@peterdoohan
"""

# %% Imports
import numpy as np
import pandas as pd

from ..core import get_sessions as gs

# %% Global Variables

FRAME_RATE = 60
# %% Functions


def test2(session):
    """Currently gets all clusters"""
    place_shifted_df = get_place_shifted_df(session, n=8)
    navigation_rates_df = session.navigation_spike_rates_df.reset_index(drop=True)  # native session times index
    place_shifted_rates_df = pd.concat([place_shifted_df, navigation_rates_df], axis=1)
    place_shifted_rates = place_shifted_rates_df[place_shifted_rates_df.trial_phase == "navigation"]
    return place_shifted_rates


def _downsample_place_shifted_rates(place_shifted_rates_df, window_length):
    """"""
    combine_n_frames = int(FRAME_RATE * window_length)
    nav_info_df = place_shifted_rates_df.drop(columns="firing_rate", level=0, axis=0)
    firing_rates_df = place_shifted_rates_df.xs("firing_rate", level=0, axis=1, drop_level=False)
    ds_firing_rates_df = firing_rates_df.groupby(firing_rates_df.index // combine_n_frames).mean()
    ds_firing_rates_df.reset_index(drop=True, inplace=True)
    mid_window_indicies = (firing_rates_df.index // combine_n_frames).unique() * combine_n_frames + (
        combine_n_frames // 2
    )
    mid_window_indicies = mid_window_indicies[mid_window_indicies < len(nav_info_df)]
    ds_nav_info_df = nav_info_df.iloc[mid_window_indicies]
    ds_nav_info_df.reset_index(drop=True, inplace=True)
    if ds_nav_info_df.shape[1] < ds_firing_rates_df.shape[1]:
        ds_firing_rates_df = ds_firing_rates_df.iloc[:-1]

    return pd.concat([ds_nav_info_df, ds_firing_rates_df], axis=1)


def _downsample_navigation_spike_counts(navigation_spike_counts_df, window_length):
    """
    Reduces resolution of navigation_spike_counts_df, that natively expresses spike counts per frame, to spike counts
    per window (defined in seconds). Note the resultant dataframe counts spike in non-overlapping blocks and for convience
    takes the place, direction, distance_to_goal (and other variables) for the mid window index of the original data structure.
    This could introduce errors where eg, positions change within a window (don't worry about this for now).
    """
    combine_n_frames = int(FRAME_RATE * window_length)
    nav_info_df = navigation_spike_counts_df.drop(columns="spike_count", level=0, axis=0)
    spike_counts_df = navigation_spike_counts_df.xs("spike_count", level=0, axis=1, drop_level=False)
    ds_spike_counts_df = spike_counts_df.groupby(spike_counts_df.index // combine_n_frames).sum()
    ds_spike_counts_df.reset_index(drop=True, inplace=True)
    mid_window_indicies = (spike_counts_df.index // combine_n_frames).unique() * combine_n_frames + (
        combine_n_frames // 2
    )
    mid_window_indicies = mid_window_indicies[mid_window_indicies < len(nav_info_df)]
    ds_nav_info_df = nav_info_df.iloc[mid_window_indicies]
    ds_nav_info_df.reset_index(drop=True, inplace=True)
    if ds_nav_info_df.shape[1] < ds_spike_counts_df.shape[1]:
        ds_spike_counts_df = ds_spike_counts_df.iloc[:-1]
    return pd.concat([ds_nav_info_df, ds_spike_counts_df], axis=1)


def get_place_shifted_df(session, n=8):
    """
    Returns a dataframe with the animal's position and direction at each frame, shifted by n frames into the future
    (within the navigation part of each trial, ITI and reward consumption periods have nans). as well as information
    about trial, goal, and steps from goal. Can easily be concat with navigation spikes and rates dataframes for
    neural analysis.
    Inputs
        session: GridMaze.analysis.core.Session object
        n: int, how many steps (defined in node-edge position transiations) into the future to shift the position
        and direction data
    """
    navigation_df = session.navigation_df  # frame by frame data
    trajectory_decisions_df = session.trajectory_decisions_df  # node by node data (already cleaned up)
    # initiaise output datafame
    decoding_df = _init_decoding_df(navigation_df, n)
    decoding_trial_dfs = []
    for trial in decoding_df.trial.dropna().unique():
        # set up trial df
        decoding_trial_df = decoding_df[(decoding_df.trial == trial) & (decoding_df.trial_phase == "navigation")].copy()
        trial_df = trajectory_decisions_df[
            (trajectory_decisions_df.trial == trial) & (trajectory_decisions_df.trial_phase == "navigation")
        ].copy()
        if trial_df.empty:  # trial starts at goal
            nav_trial_df = navigation_df.iloc[decoding_trial_df.index]
            decoding_trial_df[("steps_from_goal", "")] = 0
            decoding_trial_df[(f"place_shifted", 0)] = nav_trial_df.maze_position.simple
            decoding_trial_df[(f"direction_shifted", 0)] = nav_trial_df.cardinal_movement_direction
            decoding_trial_dfs.append(decoding_trial_df)
        else:
            trial_df = _pad_trial_df(
                trial_df, navigation_df, decoding_trial_df
            )  # add start and end trial times to node by node data (trial_df)
            frame_times = decoding_trial_df.time
            # make shifted position df
            pos = trial_df.maze_position
            pos_shifted_df = pd.DataFrame(index=trial_df.index, columns=range(-n, n + 1))
            for i in range(-n, n + 1):
                pos_shifted_df[i] = pos.shift(i)
            pos_shifted_df = _replace_first_none_value(pos_shifted_df, val="x")
            # make shifted direction df
            dir = trial_df.action
            dir_shifted_df = pd.DataFrame(index=trial_df.index, columns=range(-n, n + 1))
            for i in range(-n, n + 1):
                dir_shifted_df[i] = dir.shift(i)
            dir_shifted_df = _replace_first_none_value(dir_shifted_df, val="x")
            # index into frame-by-frame data (decoding_trial_df)
            for i in trial_df.index:
                frame_index = (frame_times - trial_df.loc[i].time).abs().idxmin()
                for j in range(-n, n + 1):
                    decoding_trial_df.loc[frame_index, (f"place_shifted", j)] = pos_shifted_df.loc[i, j]
                    decoding_trial_df.loc[frame_index, (f"direction_shifted", j)] = dir_shifted_df.loc[i, j]
                # add steps from goal info
                decoding_trial_df.loc[frame_index, ("steps_from_goal", "")] = trial_df.loc[i].steps_to_goal
            # fill in the gaps
            # place direction shifted info
            for c in ["place_shifted", "direction_shifted"]:
                decoding_trial_df[c] = decoding_trial_df[c].ffill()
                decoding_trial_df[c] = decoding_trial_df[c].replace("x", None)  # replace "x" place holders with nan
                # steps from goal info
            filled_steps_from_goal = decoding_trial_df[("steps_from_goal", "")].infer_objects(copy=False).ffill()
            # nan out non-navigation values
            filled_steps_from_goal[decoding_trial_df.trial_phase != "navigation"] = np.nan
            decoding_trial_df[("steps_from_goal", "")] = filled_steps_from_goal
            decoding_trial_dfs.append(decoding_trial_df)
    filled_decoding_df = pd.concat(decoding_trial_dfs, axis=0)
    filled_decoding_df = filled_decoding_df.reindex(
        navigation_df.index
    )  # add back frames before first and after last trial
    # shove things back together
    output_df = pd.concat(
        [
            decoding_df.drop(columns=["place_shifted", "direction_shifted"], level=0),
            filled_decoding_df.xs(key="place_shifted", level=0, axis=1, drop_level=False),
            filled_decoding_df.xs(key="direction_shifted", level=0, axis=1, drop_level=False),
        ],
        axis=1,
    )
    output_df[("steps_from_goal", "")] = filled_decoding_df[("steps_from_goal", "")]
    return output_df


def _pad_trial_df(trial_df, navigation_df, decoding_trial_df):
    """"""
    trial_df = trial_df[["time", "maze_position", "action", "steps_to_goal"]].reset_index(drop=True)
    # add start and end trial times to node by node data (trial_df)
    start_frame = navigation_df.iloc[decoding_trial_df.index[0]]
    prepend_df = pd.DataFrame(
        {
            "time": start_frame.time,
            "maze_position": start_frame.maze_position.simple,
            "action": start_frame.cardinal_movement_direction,
            "steps_to_goal": trial_df.iloc[0].steps_to_goal + 1,
        }
    )
    return pd.concat([prepend_df, trial_df], ignore_index=True)


def _init_decoding_df(navigation_df, n):
    """ """
    place_cols = [("place_shifted", i) for i in range(-n, n + 1)]
    directin_cols = [("direction_shifted", i) for i in range(1, n + 1)]
    all_cols = (
        [
            ("subject_ID", ""),
            ("maze_name", ""),
            ("day_on_maze", ""),
            ("time", ""),
            ("trial", ""),
            ("trial_phase", ""),
            ("goal", ""),
            ("steps_from_goal", ""),
        ]
        + place_cols
        + directin_cols
    )
    decoding_df = pd.DataFrame(index=navigation_df.index, columns=pd.MultiIndex.from_tuples(all_cols), data=None)
    decoding_df[("subject_ID", "")] = navigation_df.subject_ID
    decoding_df[("maze_name", "")] = navigation_df.maze_name
    decoding_df[("day_on_maze", "")] = navigation_df.day_on_maze
    decoding_df[("time", "")] = navigation_df.time
    decoding_df[("trial", "")] = navigation_df.trial
    decoding_df[("trial_phase", "")] = navigation_df.trial_phase
    decoding_df[("goal", "")] = navigation_df.goal
    return decoding_df


def _replace_first_none_value(df, val=np.nan):
    for col in df.columns:
        mask = df[col].isna() & df[col].shift().notna()
        if mask.any():
            first_index = mask.idxmax()
            df.loc[first_index, col] = val
    return df
