"""This module generates vector representations of single trial place direction action sequences and combines 
trial these vectors across sessions to make place-direction behavioural data matrices that feed into other analyses."""

# %% Imports
import numpy as np
import pandas as pd

from GridMaze.analysis.place_direction import plot_components as pc
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.maze import representations as mr

# %% Global variables


# %% Functions


def get_multisession_behavioural_place_direction_df(sessions, normalisation_method="length"):
    behavioural_place_direction_dfs = [
        get_behavioural_place_direction_df(session, normalisation_method=normalisation_method) for session in sessions
    ]
    multisession_neural_place_direction_df = pd.concat(behavioural_place_direction_dfs, axis=0)
    return multisession_neural_place_direction_df.reset_index(drop=True)


def get_behavioural_place_direction_df(
    session, navigation_only=True, trial_length_threshold=2, normalisation_method="length"
):
    trajectory_decisions_df = session.trajectory_decisions_df.dropna()
    trials = trajectory_decisions_df.trial.unique()
    if navigation_only:
        trajectory_decisions_df = trajectory_decisions_df[trajectory_decisions_df.trial_phase == "navigation"]
    place_direction_pairs = mr.get_maze_place_direction_pairs(session.simple_maze())
    pd_pair2idx = {pair: i for i, pair in enumerate(place_direction_pairs)}
    nav_sequence_one_hots = []
    for trial in trials:
        trial_trajectory_df = trajectory_decisions_df[trajectory_decisions_df.trial == trial]
        nav_sequence = list(zip(trial_trajectory_df.maze_position, trial_trajectory_df.action))
        if len(nav_sequence) < trial_length_threshold:
            continue
        else:
            one_hot_vector = np.zeros(len(place_direction_pairs), dtype=int)
            for pair in nav_sequence:
                one_hot_vector[pd_pair2idx[pair]] += 1
            nav_sequence_one_hots.append(one_hot_vector)
    M = np.vstack(nav_sequence_one_hots)
    # normalise methods
    if normalisation_method is not None:
        if normalisation_method == "length":
            M = M / np.linalg.norm(M, axis=1)[:, np.newaxis]
        else:
            assert False, f"normalisation_method {normalisation_method} not recognised"
    behavioural_place_direction_df = pd.DataFrame(
        data=M, columns=pd.MultiIndex.from_tuples(place_direction_pairs, names=["maze_position", "direction"])
    )
    return behavioural_place_direction_df


# %%Supporting functions


def get_analysis_sessions(maze_number, subject, late=True):
    subject = [subject] if not subject == "all" else subject
    days_on_maze = "late" if late == True else "all"
    sessions = gs.get_sessions(
        subject_IDs=subject,
        maze_number=[maze_number],
        day_on_maze=days_on_maze,
        with_data=["trajectory_decisions_df"],
    )
    return sessions


def plot_behavioural_place_direction_trajectories(session):
    """Visualise the behavioural place-direction trajectories for a session on the star heatmap"""
    behavioural_place_direction_df = get_behavioural_place_direction_df(session)
    pc.plot_nmf_components(behavioural_place_direction_df, session.simple_maze(), colormap="Greys")
    return
