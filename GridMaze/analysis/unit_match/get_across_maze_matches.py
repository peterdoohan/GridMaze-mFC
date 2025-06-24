"""
Library for matching cells across mazes & saving out all matches for future analysis by other scripts in tanalysis/unit_match module
"""

# %% Imports
import json
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

# %% convienience functions


def get_cross_maze_matches(subject, maze_A, maze_B, return_as="cluster_unique_ID", verbose=False):
    """ """
    # load cached matched units
    subject2cross_maze_matches = load_all_cross_maze_matches()
    maze_pair = f"{maze_A}.{maze_B}"
    all_matches = subject2cross_maze_matches[subject][maze_pair]
    if verbose:
        print(f"Found {len(all_matches)} matches for {subject}, {maze_A}.{maze_B}")
    if return_as == "cluster_unique_ID":
        return all_matches
    elif return_as == "cluster_objects":
        matched_clusters = [[gc.get_cluster(cluster_unique_ID) for cluster_unique_ID in match] for match in all_matches]
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
