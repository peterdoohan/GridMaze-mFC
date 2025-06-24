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

# %% Global Variables

from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

RESULTS_DIR = RESULTS_PATH / "unit_match" / "across_maze_matches"

# valid days where cells can be matched between probe advancements
# see experiment_info/probe_depths.htsv
TISSUE_SAMPLE2MAZE_DAYS = {
    "B": {"maze_1": [10, 11, 12, 13], "maze_2": [1, 2, 3, 4, 5, 6, 7]},
    "C": {"maze_2": [9, 10, 11], "rooms_maze": [1, 2, 3, 4, 5, 6, 7]},
}

MAZE_PAIR2VALID_DAYS = {
    "maze_1.maze_2": {"maze_1": [10, 11, 12, 13], "maze_2": [1, 2, 3, 4, 5, 6, 7]},
    "maze_2.rooms_maze": {"maze_2": [9, 10, 11], "rooms_maze": [1, 2, 3, 4, 5, 6, 7]},
}

# %% get permuted/speduo matches


def get_permuted_cluster_matches(subject_ID="m2", maze_pair=("maze_1", "maze_2"), n_permutations=1000):
    """ """
    # get number of matches for each session pair in true data (match for each permutation)
    session_pair2count = _session_pair2n_matches(subject_ID, maze_pair)
    # get all availble in a session for matching
    session_name2single_units = _session_name2single_units(subject_ID, maze_pair)
    # get permuted matches
    permuted_matches = []
    for _ in range(n_permutations):
        pseudo_matches = []
        for session_pair, n_matches in session_pair2count.items():
            A_units, B_units = [copy(session_name2single_units[s]) for s in session_pair]
            random.shuffle(A_units),
            random.shuffle(B_units)
            random_matches = list(zip(A_units[:n_matches], B_units[:n_matches]))
            pseudo_matches.extend(random_matches)
        permuted_matches.append(pseudo_matches)
    return permuted_matches


def _session_pair2n_matches(subject_ID, maze_pair=("maze_1", "maze_2")):
    """ """
    true_matches = get_cross_maze_matches(subject_ID, maze_pair[0], maze_pair[1])
    match_session_names = [[c.split("_")[0] for c in m] for m in true_matches]
    session_pair_counts = Counter(tuple(pair) for pair in match_session_names)
    return dict(session_pair_counts)


def _session_name2single_units(subject_ID="m2", maze_pair=("maze_1", "maze_2")):
    """
    Note unit-match only considers single units when doing matching
    """
    _maze_pair = f"{maze_pair[0]}.{maze_pair[1]}"
    session_name2single_units = {}
    for maze in maze_pair:
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names=[maze],
            days_on_maze=MAZE_PAIR2VALID_DAYS[_maze_pair][maze],
            with_data=["cluster_metrics"],
            must_have_data=True,
        )
        for session in sessions:
            df = session.cluster_metrics
            session_info = session.session_info
            single_units = df[df.single_unit].cluster_ID
            session_name2single_units[session.name] = list(
                convert.cluster_IDs2scluster_unique_IDs(session_info, single_units)
            )
    return session_name2single_units


# %% get matched clusters


def get_cross_maze_matches(
    subject,
    maze_A,
    maze_B,
    single_unit=True,
    tuning_metric=None,
    split_half_corr=None,
    return_as="cluster_unique_ID",
    verbose=False,
):
    """ """
    # load cached matched units
    subject2cross_maze_matches = load_all_cross_maze_matches()
    maze_pair = f"{maze_A}.{maze_B}"
    all_matches = subject2cross_maze_matches[subject][maze_pair]
    # apply filters
    if single_unit:
        # check both matched clusters are single units
        NotImplementedError
    if tuning_metric is not None:
        # check both clustes are tunned to a particular metric
        assert split_half_corr is not None
        if tuning_metric == "distance_to_goal":
            NotImplementedError
        elif tuning_metric == "place_direction":
            NotImplementedError
        elif tuning_metric == "egocentric_action":
            NotImplementedError
        else:
            raise ValueError(f"Unknown tuning metric: {tuning_metric}")
    else:
        matches = all_matches

    # return
    if verbose:
        print(f"Found {len(matches)} matches for {subject}, {maze_A}.{maze_B}")
    if return_as == "cluster_unique_ID":
        return matches
    elif return_as == "cluster_objects":
        matched_clusters = [[gc.get_cluster(cluster_unique_ID) for cluster_unique_ID in match] for match in matches]
        return matched_clusters
    else:
        raise ValueError(f"Unknown return_as value: {return_as}. Use 'cluster_unique_ID' or 'cluster_objects'.")


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
        for tissue_sample, maze_days in TISSUE_SAMPLE2MAZE_DAYS.items():
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
            maze_pair2matches[f"{maze_A}.{maze_B}"] = all_matches
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
