""" """

# %% Imports
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from GridMaze.maze import representations as mr
from GridMaze.maze import plotting as mp
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import filter as filt

from GridMaze.analysis.place_direction.dimensionality_reduction import get_nmf_df, get_pca_df

# %% Global Variables

from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)


# %% Functions


def plot_nmf_components(behavioural_sequences_df, simple_maze, n_components=8, cmap="Blues", axes=None):
    """ """
    nmf_df = get_nmf_df(behavioural_sequences_df, n_components)
    if axes is None:
        f, axes = plt.subplots(1, n_components, figsize=(6 * n_components, 6))
    for i in range(n_components):
        c = nmf_df[i]
        ax = axes[i]
        mp.plot_directed_heatmap(
            simple_maze,
            c,
            ax,
            colormap=cmap,
            silhouette_node_size=500,
            silhouette_edge_size=10,
            star_base_length=0.045,
            max_point_length=0.03,
        )
    return


def get_maze_behavioural_sequences_df(
    subject_IDs="all",
    maze_name="maze_1",
    late_sessions=True,
    max_steps_to_goal=30,
    verbose=False,
):
    """
    Behavioural trials from all sessions from the given input, represented as binary vectors
    of place-directions visited on that trial. Similar to population_tuning_df (rows of cluster
    heatmaps) from place_direction/dimensionality_reduction.py
    """
    days_on_maze = "late" if late_sessions else "all"
    if verbose:
        print("Loading sessions ...")
    sessions = gs.get_maze_sessions(
        subject_IDs=subject_IDs,
        maze_names=[maze_name],
        days_on_maze=days_on_maze,
        with_data=["trajectory_decisions_df"],
        must_have_data=True,
    )
    dfs = []
    for session in sessions:
        if verbose:
            print(session.name)
        dfs.append(get_session_behavioural_sequences(session, max_steps_to_goal))
    output_df = pd.concat(dfs, axis=0, ignore_index=True)
    return output_df.sort_index(axis=1)


def get_session_behavioural_sequences(
    session,
    max_steps_to_goal=30,
):
    """
    Behavioural sequences are binary vectors of place-direction pairs visited in each trial
    """
    place_direction2idx = {_pd: i for i, _pd in enumerate(mr.get_maze_place_direction_pairs(session.simple_maze()))}
    trajectories_df = session.trajectory_decisions_df
    trajectories_df = trajectories_df[(trajectories_df.trial_phase == "navigation")]
    trajectories_df = trajectories_df[(trajectories_df.maze_position.notnull())]
    trajectories_df = trajectories_df[(trajectories_df.action.notnull())]
    if max_steps_to_goal is not None:
        trajectories_df = trajectories_df[(trajectories_df.steps_to_goal.lt(max_steps_to_goal))]
    # loop over trials to construct sequence vectors (binary in each place-direction, 1 if visited)
    trials = trajectories_df.trial.unique()
    session_sequences = np.zeros((len(trials), len(place_direction2idx)), dtype=int)
    for i, trial in enumerate(trials):
        trial_df = trajectories_df[trajectories_df.trial == trial]
        place_direction_sequence = list(zip(trial_df.maze_position, trial_df.action))
        for j in place_direction_sequence:
            session_sequences[i, place_direction2idx[j]] += 1
    behaviour_df = pd.DataFrame(data=session_sequences, columns=pd.MultiIndex.from_tuples(place_direction2idx.keys()))
    return behaviour_df.sort_index(axis=1)


def get_session_get_behavioural_sequences_fr(
    sessions,
    navigation_only=True,
    moving_only=True,
    exclude_time_at_goal=True,
    max_steps_from_goal=30,
):
    """
    Behavioural sequences defined over frames not node-egde-node transitions as in get_behavioural_sequences
    Note same nav filter kwargs as used to generlate place-direction heatmaps
    """
    trial_sequences = []
    place_direction2idx = {_pd: i for i, _pd in enumerate(mr.get_maze_place_direction_pairs(sessions[0].simple_maze()))}
    for session in sessions:
        navigation_df = session.navigation_df
        navigation_df = filt.filter_navigation_rates_df(
            navigation_df,
            navigation_only,
            moving_only,
            exclude_time_at_goal,
            max_steps_from_goal,
        )
        # filter edge cases that lack place-direction information
        navigation_df = navigation_df[navigation_df.maze_position.simple.notnull()]
        navigation_df = navigation_df[navigation_df.cardinal_movement_direction.notnull()]
        trials = navigation_df.trial.unique()
        # build binary reps of trails in place-direction space
        session_sequences = np.zeros((len(trials), len(place_direction2idx)), dtype=int)
        for i, trial in enumerate(trials):
            trial_df = navigation_df[navigation_df.trial == trial]
            place_direction_sequence = list(zip(trial_df.maze_position.simple, trial_df.cardinal_movement_direction))
            for j in place_direction_sequence:
                session_sequences[i, place_direction2idx[j]] += 1
        trial_sequences.append(session_sequences)
    behaviour_df = pd.DataFrame(
        data=np.vstack(trial_sequences),
        columns=pd.MultiIndex.from_tuples(place_direction2idx.keys(), names=["maze_position", "direction"]),
    )
    return behaviour_df.sort_index(axis=1)
