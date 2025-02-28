"""This module generates event dataframes and trials dataframes from pycontrol session.txt files"""
# %% Imports
import regex as re
import numpy as np
import pandas as pd
from . import pycontrol_data_import as di


# %% Main functions
def get_events_df(session_directory):
    """Creates a dataframe with all events from a session.
    - Uses the session_dataframe function from data_import"""
    pycontrol_path = session_directory.pycontrol_path 
    events_df = di.session_dataframe(pycontrol_path, pair_end_suffix="_out")
    events_df = events_df[(events_df.name != "rsync") & (events_df.type != "info")].reset_index(
        drop=True
    )  # remove all rsync events
    events_df["time"] = events_df["time"] / 1000  # Convert time to seconds
    events_df["duration"] = events_df["duration"] / 1000  # convert durations to seconds
    return events_df


def get_trials_df(session_directory):
    """Creates a dataframe with trial information from a session."""
    pycontrol_path = session_directory.pycontrol_path
    session = di.Session(pycontrol_path)
    trials_df = pd.DataFrame(
        {
            ("trial"): _get_trial_numbers(session),
            ("goal"): _get_goal_states(session),
            ("errors"): _get_errors(session),
            ("time", "cue"): _get_cue_times(session),
            ("time", "reward"): _get_reward_times(session),
            ("time", "end_reward_consumption"): _get_end_reward_consumption_times(session),
            ("time", "ITI_start"): _get_ITI_start_times(session),
            ("time", "trial_end"): _get_trial_end_times(session),
        }
    )
    columns = pd.MultiIndex.from_tuples(
        [(col, "") if isinstance(col, str) else col for col in trials_df.columns], names=["feature", "event"]
    )
    trials_df.columns = columns
    return trials_df


# %% Subfunctions
def _get_trial_numbers(session):
    """Gets trial numbers from session print lines
    - Note: the last trial is removed if no reward is triggered"""
    trial_numbers = []
    for line in session.print_lines:
        if "Start" in line:
            match = re.search("T#:(\d+)", line)
            if match:
                trial_numbers.append(int(match.group(1)))
    if len(trial_numbers) != len(session.times["ITI"][1:]):
        trial_numbers = trial_numbers[:-1]
    return trial_numbers


def _get_goal_states(session):
    """Retrieves the goal location cued on each trial from the session print lines"""
    goal_states = []
    for line in session.print_lines:
        if "Start" in line:
            match = re.search("S#:(\w+)", line)
            if match:
                goal_states.append(match.group(1))
    if len(goal_states) != len(session.times["ITI"][1:]):
        goal_states = goal_states[:-1]
    return goal_states


def _get_cue_times(session):
    """Gets the start time of each trial from the session print lines
    - Note: the last cue time is removed if it is not followed by a reward"""
    trial_cue_times = []
    for line in session.print_lines:
        if "Start" in line:
            match = re.search("sT#:(\d+)", line)
            if match:
                trial_cue_times.append(int(match.group(1)))
    if len(trial_cue_times) != len(session.times["ITI"][1:]):
        trial_cue_times = trial_cue_times[:-1]
    return np.array(trial_cue_times) / 1000  # seconds


def _get_reward_times(session):
    """Gets the time of reward on each trial.
    - Note this is the same as the end of trial time (eT) mentioned in the End trial print lines
    """
    reward_times = []
    for line in session.print_lines:
        if "End" in line:
            match = re.search("eT#:(\d+)", line)
            if match:
                reward_times.append(int(match.group(1)))
    if len(reward_times) != len(session.times["ITI"][1:]):
        reward_times = reward_times[:-1]
    return np.array(reward_times) / 1000  # seconds


def _get_end_reward_consumption_times(session, next_poke_threshold=3):
    """Gets time animal finishes consuming reward at the goal location.
    - Calculated by looping over trials to get the time when the subject last poked out of the reward port before
      triggering the next reward
    - In cases where there is no poke-out registered from the reward port (leading to an IndexError) end of ITI time
      (=start of the next trial) is taken as the end of reward consumption
    """
    end_reward_consumption_times = []
    for goal, cue_time, ITI_time, next_rew_time in zip(
        _get_goal_states(session),
        session.times["cue"],
        np.append(session.times["ITI"], session.events[-1].time)[1:],
        np.append(session.times["reward"], session.events[-1].time)[1:],
    ):
        goal_out_events = [
            event
            for event in session.events
            if (event.time > cue_time) and (event.time < next_rew_time) and (event.name == goal + "_out")
        ]
        # impose maximum time between reward poke out to avoid long RC periods
        try:
            filtered_goal_out_events = [goal_out_events[0]]
            for i in range(1, len(goal_out_events)):
                if goal_out_events[i].time - filtered_goal_out_events[-1].time > next_poke_threshold * 1000:
                    break
                filtered_goal_out_events.append(goal_out_events[i])
            end_reward_consumption_times.append(filtered_goal_out_events[-1].time)
        except IndexError:  # Poke out event was not registered, use start of ITI instead.
            end_reward_consumption_times.append(ITI_time)
    return np.array(end_reward_consumption_times) / 1000  # seconds


def _get_ITI_start_times(session):
    """Gets start of ITI times on each trial."""
    return session.times["ITI"][1:] / 1000  # seconds


def _get_trial_end_times(session):
    """Gets trial end times (= start time of next trial)"""
    end_trial_times = session.times["cue"][1:]
    if session.times["ITI"][-1] > session.times["cue"][-1]:
        end_trial_times = np.append(end_trial_times, session.events[-1].time)
    return end_trial_times / 1000  # seconds


def _get_errors(session):
    """Returns number of errors made on each trial.
    - This is caluclated in the pycontrol task file and ouput on the End trial print lines
    - errors = number of non-goal ports poked during the session
    - Feeds into get_session_dataframe function
    """
    errors = []
    for line in session.print_lines:
        if "End" in line:
            match = re.search("E#:(\d+)", line)
            if match:
                errors.append(int(match.group(1)))
    if len(errors) != len(session.times["ITI"][1:]):
        errors = errors[:-1]
    return errors


# %%
