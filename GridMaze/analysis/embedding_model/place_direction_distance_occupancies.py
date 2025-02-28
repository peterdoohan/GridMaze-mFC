"""
Library for computing occupancies of the place-direction x distance-to-goal to goal product space.
Used for masking out low occupancy states in the embedding model analysis.
"""

# %% Imports
import json
import numpy as np
import pandas as pd

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import convert
from GridMaze.maze import representations as mr
from GridMaze.maze import representations as mr


# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH
from GridMaze.analysis.embedding_model.run_experiment import DEFAULT_INPUT_KWARGS

OCCUPANCY_RESULTS_PATH = RESULTS_PATH / "embedding_model" / "place_direction_distance_occupancies"

with open(EXPERIMENT_INFO_PATH / "maze_configs.json", "r") as input_file:
    MAZE_CONFIGS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

FRAME_RATE = 60
# %% Save product space counts to disk for convienience


def save_place_direction_distance_occupancies():
    """Save occupancy data for each subject and maze to disk. Run once."""
    # save occupancy counts for each subject and maze (including data combined across all subjects)
    for subject_ID_input in [[s] for s in SUBJECT_IDS] + ["all"]:
        for maze_name in MAZE_CONFIGS.keys():
            print(f"Processing {maze_name} for {subject_ID_input}")
            # get PDD space occupancy counts
            exp_data = get_filtered_navigation_data(maze_name, subject_IDs=subject_ID_input)
            counts = get_place_direction_distance_occupancy_counts(exp_data, maze_name, return_as="series")
            # save to disk
            subject = "all_subjects" if subject_ID_input == "all" else subject_ID_input[0]
            save_path = OCCUPANCY_RESULTS_PATH / f"{subject}.{maze_name}.csv"
            save_path.parent.mkdir(exist_ok=True, parents=True)
            counts.to_csv(save_path)
    # also save input kwargs for reference
    with open(OCCUPANCY_RESULTS_PATH / "input_kwargs.json", "w") as output_file:
        json.dump(DEFAULT_INPUT_KWARGS, output_file, indent=4)


def load_place_direction_distance_occupancies(maze_name, subject_ID):
    """
    Note occcupancies are gernerated from input data filtered by input kwargs
    defined in DEFAULT_INPUT_KWARGS, returns pd.Series with place_direction,
    distance_to_goal MultiIndex.
    """
    subject = "all_subjects" if subject_ID == "all" else subject_ID
    save_path = OCCUPANCY_RESULTS_PATH / f"{subject}.{maze_name}.csv"
    occupancy = pd.read_csv(save_path, index_col=[0, 1])["0"]
    return occupancy


def get_occupancy_mask(maze_name, subject_ID, min_occupancy=0.5, input_kwargs=DEFAULT_INPUT_KWARGS):
    """
    Returns a boolean mask of the place-direction x distance-to-goal product space
    where occupancy is greater than min_occupancy (defined in seconds of input data).
    """
    occupancy = load_place_direction_distance_occupancies(maze_name, subject_ID)
    min_occ_count = min_occupancy / input_kwargs["resolution"]  # convert from s to counts at binned resolution
    return occupancy.ge(min_occ_count)


# %% Get Occupancy counds of each state in the place-direction - distance to goal product space


def get_place_direction_distance_occupancy_counts(
    exp_data, maze_name, input_kwargs=DEFAULT_INPUT_KWARGS, return_as="series"
):
    """
    bins distance data same as embedding_model/get_input_data (w/ paramters specified in DEFAULT_INPUT_KWARGS)
    returns counts of each place-direction x distance-to-goal state in the product space.
    """
    distance_bins = convert._get_distance_bins(
        input_kwargs["distance_bin_method"],
        input_kwargs["n_distance_bins"],
        input_kwargs["distance_metrics"],
        input_kwargs["max_distance"],
    )
    distnace_bin_midpoints = [d.mid for d in distance_bins]
    d2idx = {d: i for i, d in enumerate(distance_bins)}
    simple_maze = mr.get_simple_maze(maze_name)
    place_directions = mr.get_maze_place_direction_pairs(simple_maze)
    pd2idx = {pd: i for i, pd in enumerate(place_directions)}
    pds = [pd2idx[pd] for pd in zip(exp_data.maze_position.simple, exp_data.cardinal_movement_direction)]
    distance_metrics = input_kwargs["distance_metrics"]
    distance_bins_col = (distance_metrics[0], distance_metrics[1] + "_binned")
    ds = exp_data[distance_bins_col].map(d2idx).to_numpy()
    counts = np.zeros((len(pd2idx), len(d2idx)))  # [n_place-directions, n_distances]
    for _pd, d in zip(pds, ds):
        counts[_pd, d] += 1
    if return_as == "matrix":
        return counts
    if return_as == "series":
        return pd.Series(
            counts.flatten(),
            index=pd.MultiIndex.from_product(
                [place_directions, distnace_bin_midpoints], names=["place_direction", "distance_to_goal"]
            ),
        )
    if return_as == "df":
        return pd.DataFrame(
            counts,
            index=pd.MultiIndex.from_tuples(place_directions),
            columns=distnace_bin_midpoints,
        )


# %% Load data functions


def get_filtered_navigation_data(
    maze_name,
    subject_IDs="all",
):
    """
    Combines filtered navigation data with filter parameters specified in DEFAULT_INPUT_KWARGS,
    across session from input subjects for a given maze.
    """
    subject_IDs = subject_IDs if not subject_IDs == "all" else "all"
    sessions = gs.get_maze_sessions(
        subject_IDs=subject_IDs,
        maze_names=[maze_name],
        days_on_maze="late",
        with_data=["navigation_df"],
        must_have_data=True,
    )
    exp_data = pd.concat([get_session_filtered_navigation_data(s) for s in sessions], axis=0)
    return exp_data.reset_index(drop=True)


def get_session_filtered_navigation_data(
    session,
    input_kwargs=DEFAULT_INPUT_KWARGS,
):
    """Mirrors navigation data filtering in embedding_model/get_input_data"""
    navigation_df = session.navigation_df
    navigation_df = _downsample_navigation_df(navigation_df, input_kwargs["resolution"])
    navigation_df = filt.filter_navigation_rates_df(
        navigation_df,
        navigation_only=input_kwargs["navigation_only"],
        moving_only=input_kwargs["moving_only"],
        exclude_time_at_goal=False,
        max_steps_to_goal=input_kwargs["max_steps_to_goal"],
    )
    # bin distance to goal
    distance_bins = convert._get_distance_bins(
        input_kwargs["distance_bin_method"],
        input_kwargs["n_distance_bins"],
        input_kwargs["distance_metrics"],
        input_kwargs["max_distance"],
    )
    # add distance bins to navigation df
    distance_metrics = input_kwargs["distance_metrics"]
    distance_bins_col = (distance_metrics[0], distance_metrics[1] + "_binned")
    navigation_df[distance_bins_col] = pd.cut(
        navigation_df[(distance_metrics[0], distance_metrics[1])],
        bins=distance_bins,
    )
    # remove columns with missing values (happens in edge cases at end of session due to downsampling)
    navigation_df = navigation_df[navigation_df[distance_bins_col].notnull()]
    navigation_df = navigation_df[navigation_df.maze_position.simple.notnull()]
    navigation_df = navigation_df[navigation_df.cardinal_movement_direction.notnull()]
    return navigation_df


def _downsample_navigation_df(navigation_df, window_length):
    """Downsamples frame rate data same as in embedding_model/get_input_data"""
    combine_n_frames = int(FRAME_RATE * window_length)
    mid_window_indicies = (navigation_df.index // combine_n_frames).unique() * combine_n_frames + (
        combine_n_frames // 2
    )
    mid_window_indicies = mid_window_indicies[:-1]  # last index out of range
    return navigation_df.iloc[mid_window_indicies]


# %% save out edges of distribution
