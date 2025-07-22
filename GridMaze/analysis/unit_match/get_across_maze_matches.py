"""
Library for matching cells across mazes & saving out all matches for future analysis by other scripts in tanalysis/unit_match module
"""

# %% Imports
import json
import copy
import random
from collections import Counter

from GridMaze.analysis.core import unit_matching as um
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import convert

# %% Global Variables

from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

RESULTS_DIR = RESULTS_PATH / "unit_match" / "across_maze_matches"

# valid days where cells can be matched between probe advancements
# see experiment_info/probe_depths.htsv
MAZE_PAIR2VALID_DAYS = {
    "maze_1.maze_2": {"maze_1": [10, 11, 12, 13], "maze_2": [1, 2, 3, 4, 5, 6, 7]},
    "maze_2.rooms_maze": {"maze_2": [9, 10, 11], "rooms_maze": [1, 2, 3, 4, 5, 6, 7]},
}

# %% get permuted/speduo matches


def get_permuted_cross_maze_matches(
    subject_ID, maze_pair, n_permutations=1000, single_units=True, tuning_metric=None, min_split_half_corr=None
):
    """
    Generates random matches between paired sessions for true unit matching aross mazes for a given subject.
    Matches the statistics of the true number of matches in each session pair and applies the same filtering
    for single units, tuning to some metric (see get_cross_maze_matches for details) as when finding true matches.
    """
    # get true matches
    true_matches = get_cross_maze_matches(
        subject_ID, maze_pair, single_units, tuning_metric, min_split_half_corr, return_as="cluster_unique_ID"
    )
    # count number of matches per session pair to match sampling stats in permutations
    session_pair2count = _session_pair2n_matches(true_matches)
    # get all availble with same filtering criteria for true matches
    session_name2avail_units = _get_available_units(
        subject_ID, maze_pair, single_units, tuning_metric, min_split_half_corr, return_as="dict"
    )
    # get permuted matches
    permuted_matches = []
    for _ in range(n_permutations):
        pseudo_matches = []
        for session_pair, n_matches in session_pair2count.items():
            A_units, B_units = [copy.copy(session_name2avail_units[s]) for s in session_pair]
            random.shuffle(A_units),
            random.shuffle(B_units)
            random_matches = list(zip(A_units[:n_matches], B_units[:n_matches]))
            pseudo_matches.extend(random_matches)
        permuted_matches.append(pseudo_matches)
    return permuted_matches


def _session_pair2n_matches(all_matches):
    """ """
    match_session_names = [[c.split("_")[0] for c in m] for m in all_matches]
    session_pair_counts = Counter(tuple(pair) for pair in match_session_names)
    return dict(session_pair_counts)


# %% get matched clusters


def get_cross_maze_matches(
    subject_ID,
    maze_pair,
    single_units=True,
    tuning_metric=None,
    min_split_half_corr=None,
    return_as="cluster_unique_ID",
    verbose=False,
):
    """
    Loads cached unit match results for a given subject and maze pair,
    filters those matches based on input critreria, eg only single units with place-direction split-halfs
    correlation above some threshold

    valid tuning metrics: place_direction, distance_to_goal, egocentric_action

    returns list of either paired cluster unique IDs or cluster objects
    """
    # load cached matched units
    subject2cross_maze_matches = load_all_cross_maze_matches()
    maze_A, maze_B = maze_pair
    _maze_pair = f"{maze_A}.{maze_B}"
    all_matches = subject2cross_maze_matches[subject_ID][_maze_pair]
    # get valid units for the given single_units, tuning_metric, split_half_corr inputs
    valid_units = _get_available_units(
        subject_ID, maze_pair, single_units, tuning_metric, min_split_half_corr, return_as="list"
    )
    # check that each pair of matches are valid
    matches = []
    for unit_1, unit_2 in all_matches:
        # check if both units are valid
        if unit_1 in valid_units and unit_2 in valid_units:
            matches.append((unit_1, unit_2))

    if len(matches) == 0:
        if verbose:
            print(f"No matches found for {subject_ID}, {maze_A}.{maze_B} with the given criteria.")
        return None
    # return
    if verbose:
        print(f"Found {len(matches)} matches for {subject_ID}, {maze_A}.{maze_B}, with the given criteria.")
    if return_as == "cluster_unique_ID":
        return matches
    elif return_as == "cluster_objects":
        matched_clusters = [[gc.get_cluster(cluster_unique_ID) for cluster_unique_ID in match] for match in matches]
        return matched_clusters
    else:
        raise ValueError(f"Unknown return_as value: {return_as}. Use 'cluster_unique_ID' or 'cluster_objects'.")


def _get_available_units(
    subject_ID,
    maze_pair,
    single_units=True,
    tuning_metric=None,
    min_split_half_corr=None,
    return_as="dict",
):
    """
    for given subject and maze pair which defines a set of valid sessions,
    filter through the avialble units in each of those valid sessions under the
    specified criteria in the unit, same as get_cross_maze_matches
    """
    # define data needed
    with_data = ["cluster_metrics"]
    if tuning_metric == "distance_to_goal":
        with_data.append("cluster_distance_tuning_metrics")
    elif tuning_metric == "place_direction":
        with_data.append("cluster_place_direction_tuning_metrics")
    elif tuning_metric == "egocentric_action":
        with_data.append("cluster_egocentric_action_tuning_metrics")
    else:
        ValueError(f"Unknown tuning metric: {tuning_metric}")

    _maze_pair = f"{maze_pair[0]}.{maze_pair[1]}"
    session_name2single_units = {}
    for maze in maze_pair:
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names=[maze],
            days_on_maze=MAZE_PAIR2VALID_DAYS[_maze_pair][maze],
            with_data=with_data,
            must_have_data=True,
        )
        for session in sessions:
            if tuning_metric is None:
                df = session.cluster_metrics
                session_info = session.session_info
                if single_units:
                    avail_units = df[df.single_unit].cluster_ID
                else:
                    avail_units = df[df.single_unit | df.multi_unit].cluster_ID
                avail_units = convert.cluster_IDs2scluster_unique_IDs(session_info, avail_units)
            else:
                # only have metrics calculated for single units
                assert single_units
                if tuning_metric == "distance_to_goal":
                    df = session.cluster_distance_tuning_metrics
                    avail_units = df[df.split_half_corr.value.gt(min_split_half_corr)].cluster_unique_ID

                elif tuning_metric == "place_direction":
                    df = session.cluster_place_direction_tuning_metrics
                    avail_units = df[df.split_half_corr.value.gt(min_split_half_corr)].index
                elif tuning_metric == "egocentric_action":
                    df = session.cluster_egocentric_action_tuning_metrics
                    avail_units = df[df.split_half_corr.all_action.value.gt(min_split_half_corr)].cluster_unique_ID
                # single unit by default when filtering for tuning
            session_name2single_units[session.name] = list(avail_units)
    if return_as == "dict":
        return session_name2single_units
    elif return_as == "list":
        # return a list of all available units across all sessions
        all_units = []
        for session_name, units in session_name2single_units.items():
            all_units.extend(units)
        return all_units
    else:
        raise ValueError(f"Unknown return_as value: {return_as}. Use 'dict' or 'list'.")


# %% populate and load functions


def load_all_cross_maze_matches():
    """Load all cross maze matches from the saved JSON file."""
    save_path = RESULTS_DIR / "all_cross_maze_matches.json"
    if not save_path.exists():
        raise FileNotFoundError(
            f"No cross maze matches found at {save_path}.  \n Please run populate_all_cross_maze_matches() first."
        )
    with open(save_path, "r") as f:
        data = json.load(f)
    return data


def populate_all_cross_maze_matches(save=True, verbose=True):
    """ """
    save_path = RESULTS_DIR / "all_cross_maze_matches.json"
    if not save and save_path.exists():
        print(f"matches already populated, to repopulate set save=True")
        return
    subject2cross_maze_matches = {}
    for subject in SUBJECT_IDS:
        maze_pair2matches = {}
        for maze_pair, maze_days in MAZE_PAIR2VALID_DAYS.items():
            maze_A, maze_B = maze_days.keys()
            A_days, B_days = maze_days.values()
            all_matches = []
            for day_A in A_days:
                for day_B in B_days:
                    if verbose:
                        print(f"Matching {subject} {maze_A} day {day_A} with {maze_B} day {day_B}")
                    matches = _match_across_mazes(subject, maze_A, day_A, maze_B, day_B, verbose)
                    if matches is not None:
                        all_matches.extend(matches)
            all_matches = [m for m in all_matches if m is not None]
            if len(all_matches) == 0:
                raise ValueError(f"No matches for {subject}, {maze_A}-{maze_B}")
            if verbose:
                print(f"Found {len(all_matches)} matches for {subject}, {maze_A}.{maze_B}")
            maze_pair2matches[maze_pair] = all_matches
        subject2cross_maze_matches[subject] = maze_pair2matches
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(subject2cross_maze_matches, f, indent=4)
        if verbose:
            print(f"Saved cross maze matches to {save_path}")
    return subject2cross_maze_matches


def _match_across_mazes(subject, maze_A, day_A, maze_B, day_B, verbose=True):
    """ """
    try:
        matched_clusters = um.get_matched_clusters(
            subject_ID=subject,
            list_of_dicts=[
                {
                    "session_types": ["maze"],
                    "maze_names": [maze_A],
                    "days_on_maze": [day_A],
                    "goal_subsets": "all",
                },
                {
                    "session_types": ["maze"],
                    "maze_names": [maze_B],
                    "days_on_maze": [day_B],
                    "goal_subsets": "all",
                },
            ],
            return_cluster_objects=False,
        )
    except Exception as e:
        # index error during drift correction? not enough cells?
        if verbose:
            print(f"Error matching across {maze_A} and {maze_B} for subject {subject}: {e}")
        return None
    if len(matched_clusters) == 0:
        return None
    else:
        return matched_clusters
