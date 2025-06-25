"""A script to unitmatch for tracking cells across sessions (within and across days).
Primarily based off of UMPy_spike_interface_demo.ipynb.
Integrated with other scripts to spikesort from raw data collected using open_ephys.
Notably, inherits paths from spikesort_session.

The following document long with many different functions so here's an overview

1. Going from processed_data paths to running unit match
    -The results from here will be a set of unique ID's of matched clusters.
    -from here we can run unitmatch on selected sessions or all sessions

2. Running unit match on all sessions for a subject
    -we save out unique ID assignments
    -we also save out subscores and a large probability matrix

3. Extracting matches from probability matrices
    -rather than using UM's unique ID assignment we make our own matches
    -we do this as a partition of all clusters to form an equivalence relation.

4. Custom probability matrices
    -rather than relying on UM's naive bayes posterior probability approach
    -we can define probabilities using SVM classifiers or other methods.

5. Quality control
    -Here we have functions to get pairwise unitmatch reports
    -We also try characterise match quality across entire dataset
    -This is used to compare different unit match methods.

6. Miscellaneous Utility functions
    -some UnitMatch subfunctions

7. Clean-up functions
@charlesdgburns"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date
from datetime import datetime
from collections import defaultdict
from IPython.display import Image, display

# For getting data
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import convert
from GridMaze.analysis.core import get_clusters as gc

# from GridMaze.preprocessing import get_data_directory as gdd
from GridMaze.analysis.core import load_data

# import UnitMatchPy
import UnitMatchPy.bayes_functions as bf
import UnitMatchPy.utils as util
import UnitMatchPy.metric_functions as mf
import UnitMatchPy.overlord as ov
import UnitMatchPy.save_utils as su
import UnitMatchPy.GUI as gui
import UnitMatchPy.assign_unique_id as aid
import UnitMatchPy.default_params as default_params

# For pairwise report plotting
import spikeinterface.full as si
import spikeinterface.widgets as sw
import matplotlib.pyplot as plt
from SpikeSorting import spikesort_session as sps  # some useful utils for handling probe location info

# For custom probability matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# For custom uid assignment
import igraph as ig
import leidenalg as la
from sklearn.cluster import AgglomerativeClustering


# %% GLOBAL VARIABLES

## example dictionary we use to select sessions
example_dict = {
    "session_types": ["maze"],
    "maze_names": ["maze_1"],
    "days_on_maze": [3, 4],
    "goal_subsets": ["all"],
}

all_sessions_dict = {
    "session_types": ["maze", "rest"],
    "maze_names": "all",
    "days_on_maze": "all",
    "goal_subsets": "all",
}

## SET UP UNIT_MATCH PARAMETERS

PARAM = default_params.get_default_param()  # default is ready for Neuropixel 2.0 probes

# These parameters should be retrievable from elsewhere to be honest. Maybe considered experiment info?
PARAM["no_shanks"] = 6  # Camb Neurotech
PARAM["shank_dist"] = 155  ## SHOULD BE THE DISTANCE WITHIN WHICH YOU CONSIDER A CENTROID TO BE WITHIN A CHANNEL
# Changing distances and radii might be useful for across-day matching.
PARAM["max_dist"] = 100  # default for Neuropixel 2.0 is 100
PARAM["channel_radius"] = 150  # default for Neuropixel 2.0 is 150
PARAM["max_n_channels"] = (
    64  # Set to Neuropixel 1.0; this is used for padding when channels are removed before spikesorting.
)


from ...paths import EXPERIMENT_INFO_PATH, PROCESSED_DATA_PATH, RESULTS_PATH

with open(EXPERIMENT_INFO_PATH / "maze_configs.json", "r") as input_file:
    MAZE_CONFIGS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "maze_day2date.json", "r") as input_file:
    MAZE_DAY2DATE = json.load(input_file)


# %% FUNCTIONS

# %% 1. Going from processed_data paths to running unit match


def get_matched_clusters(
    subject_ID: str,
    list_of_dicts: list,
    ignore_missing_paths=False,
    best_within_session_matches=True,
    matches_all_sessions=True,
    return_cluster_objects=True,
    method="um_selected_agg",
):
    """INPUT: Specify sessions by a list containing a dictionary for each session type.
    list_of_dicts = [dict1, dict2,...]
    where dict = {'session_types': ['open_field','maze'],
                'maze_names': ['fully_connected'],
                'days_on_maze': [8],
                'goal_subsets': ['all']}
    OUTPUT: list of lists of matched clusters:
        [[CUID1,CUID2],[CUID3,CUID5],...]
        where CUID is of the form: subject.date.sessiontype_clusterN
    """
    # the list of sessions we want matches for.
    all_paths = paths_from_list_of_dicts(list_of_dicts, subject_ID, ignore_missing_paths)

    # 1. get the appropriate probability matrix
    if "um_all" in method:
        # make sure you've run unitmatch first on all subjects
        prob_matrix_df = load_prob_matrix_df(subject_ID)
        if "uid" in method:  # this takes a while so we have a special if statement.
            unitmatch_df = pd.read_csv(f"../data/processed_data/{subject_ID}/UnitMatch/unitmatch_df.htsv", sep="\t")
    if "um_selected" in method:
        unitmatch_df, _, _ = run_unitmatch(all_paths)
        unitmatch_df = add_CUID_to_unitmatch_df(unitmatch_df, all_paths)
        prob_matrix_df = pd.pivot(unitmatch_df, values="UM Probabilities", index="CUID1", columns="CUID2")
        prob_matrix_df = prob_matrix_df.astype(dtype=np.float32)
    if "svm_all" in method:
        prob_matrix_df = get_SVM_prob_matrix(subject_ID)
    ## lastly, we make sure that the probability matrix is symmetric:
    prob_matrix_df = (prob_matrix_df + prob_matrix_df.values.T) / 2

    # 2. Match clusters according to probabilities
    if "uid" in method:
        print("Matching with unitmatch unique ID algorithm")
        matched_clusters = match_with_uid(unitmatch_df)  # using the UID algorithm.
    if "leiden" in method:
        matched_clusters = match_with_leiden(prob_matrix_df)
    if "agg" in method:
        print("Matching with agglomerative clustering")
        matched_clusters = match_with_agglomerative(prob_matrix_df)
    if "partition" in method:
        matched_clusters = match_with_partition(prob_matrix_df)

    # 3. Clean the cluster lists
    matched_clusters = filter_to_sessions(matched_clusters, all_paths)

    if best_within_session_matches:
        matched_clusters = get_best_within_session_matches(matched_clusters, prob_matrix_df)

    if matches_all_sessions:
        matched_clusters = [x for x in matched_clusters if len(x) == len(all_paths)]

    matched_clusters = [
        x for x in matched_clusters if len(x) > 1
    ]  # remove singletons, since these are technically not matched.

    if return_cluster_objects:
        matched_clusters = [[gc.get_cluster(s) for s in mc] for mc in matched_clusters]

    return matched_clusters


def paths_from_list_of_dicts(list_of_dicts, subject_ID, ignore_missing_paths):
    """Short utility function to simplify function calls.
    INPUT: list of dictionaries of sessions we want.
            we then call get_paths_to_UM_session() for each subject_ID given:
            session_type, maze_name, day_on_maze, and goal_subset,"""
    all_paths = []
    for dict in list_of_dicts:
        paths = get_paths_to_UM_sessions(
            subject_ID,
            dict["session_types"],
            dict["maze_names"],
            dict["days_on_maze"],
            dict["goal_subsets"],
            ignore_missing_paths,
        )
        all_paths += paths
    return all_paths


def get_paths_to_UM_sessions(
    subject_ID, session_types, maze_names, days_on_maze, goal_subsets, ignore_missing_paths=False
):
    """INPUT: specified sessions to match by a list of dictionaries for each session type
    e.g. {session_type: 'open_field',}
    OUTPUT: list of analysis_data_paths
    """

    maze_names = list(MAZE_CONFIGS.keys()) if maze_names == "all" else maze_names
    if days_on_maze == "all":
        days_on_maze = list(range(1, 13))
    elif days_on_maze == "late":
        days_on_maze = list(range(4, 13))
    goal_subsets = ["all", "subset_1", "subset_2"] if goal_subsets == "all" else goal_subsets

    requested_paths = []  # initialise
    tissue_samples = []
    for maze in maze_names:
        for day_on_maze in days_on_maze:
            # check day_on_maze is valid
            if str(day_on_maze) not in MAZE_DAY2DATE[maze].keys():
                continue
            for session_type in session_types:
                # check session type valid for date
                session_date = MAZE_DAY2DATE[maze][str(day_on_maze)]
                tissue_samples.append(load_data._get_tissue_sample(subject_ID, date.fromisoformat(session_date)))
                session_name = f"{session_date}.{session_type}"
                processed_data_path = PROCESSED_DATA_PATH / subject_ID / session_name
                # check goal subset is
                session_info = gs.load_data.load(processed_data_path / "session_info.json")
                if session_type == "maze":
                    if not session_info["goal_subset"] in goal_subsets:
                        print(
                            f'session_info["goal_subset"] not in {goal_subsets} for {session_type} {maze} day {day_on_maze}'
                        )
                        continue
                requested_paths.append(processed_data_path)

    # check probe not moved between requested sessions
    if len(set(tissue_samples)) != 1:
        print(f"Multiple tissue samples found: {set(tissue_samples)}")
        raise ValueError(
            f"Consider experiment_info/probe_depths.htsv to ensure requested sessions \n for unit match were recording in the same tissue sample."
        )

    if len(requested_paths) == 0:  # i.e. if list is still empty
        raise FileNotFoundError(f"No files were found. Please change request from: \n {dict}")

    # check that UnitMatch is possible for all paths
    missing_sessions = []
    no_good_unit_sessions = []
    for each_path in requested_paths:
        if not (each_path / "UnitMatch").exists():
            missing_sessions.append(each_path)
        else:  # check that there are 'good' units
            labels = pd.read_csv(each_path / "UnitMatch" / "cluster_group.tsv", sep="\t")
            n_good = len(labels.query('KSLabel=="good"'))
            if n_good == 0:
                no_good_unit_sessions.append(each_path)

    # give useful error messages:
    if len(missing_sessions) > 0:
        if ignore_missing_paths == True:
            print(f"Missing UnitMatch inputs for {len(missing_sessions)} sessions: \n {missing_sessions}")
            requested_paths = [x for x in requested_paths if x not in missing_sessions]
        else:
            raise FileNotFoundError(
                f"Missing UnitMatch inputs for {len(missing_sessions)} sessions: \n {missing_sessions}"
            )
    if len(no_good_unit_sessions) > 0:
        if ignore_missing_paths == True:
            print(f"No good units for UnitMatch for {len(no_good_unit_sessions)} sessions: \n {no_good_unit_sessions}")
            requested_paths = [x for x in requested_paths if x not in no_good_unit_sessions]
        else:
            raise ValueError(
                f"No good units for UnitMatch for {len(no_good_unit_sessions)} sessions: \n {no_good_unit_sessions}"
            )

    return requested_paths


def run_unitmatch(processed_data_paths, param=PARAM):
    """Code to match units across sessions, taking inputs in the form of:
    INPUT: list of session path objects: 'processed_data/subject/datetime/.
            parameters for unit match (n_shanks, max_dist, channel_radius et.c.)

    OUTPUT: unitmatch_df #unitmatch outputs as a large dataframe.
            clus_info #information about which session each cluster belongs to
            wave_dict #information about waveforms and locations
    """
    UM_input_paths = [x / "UnitMatch" for x in processed_data_paths]

    print("Running unitmatch")
    try:
        wave_paths, unit_label_paths, channel_pos = util.paths_from_KS(UM_input_paths)
        waveform, session_id, session_switch, within_session, good_units, param = util.load_good_waveforms(
            wave_paths, unit_label_paths, param, good_units_only=True
        )  # this can break if = False; unit match doesn't detect bad units itself.
    except:
        print("Likely padding issue. Trying padding and running again.")
        for each_path in UM_input_paths:
            pad_unitmatch_inputs(each_path)
        wave_paths, unit_label_paths, channel_pos = util.paths_from_KS(UM_input_paths)
        waveform, session_id, session_switch, within_session, good_units, param = util.load_good_waveforms(
            wave_paths, unit_label_paths, param, good_units_only=True
        )  # this can break if = False; unit match doesn't detect bad units itself.

    # param['peak_loc'] = #may need to set as a value if the peak location is NOT ~ half the spike width

    # create clus_info, contains all unit id/session related info
    clus_info = {"session_switch": session_switch, "session_id": session_id, "original_ids": np.concatenate(good_units)}

    # Extract parameters from waveform into a wave properties dictionary
    wave_dict = ov.extract_parameters(waveform, channel_pos, clus_info, param)

    # Extract metric scores
    total_score, candidate_pairs, scores_to_include, predictors = ov.extract_metric_scores(
        wave_dict, session_switch, within_session, param, niter=2
    )

    # Probability analysis
    output_prob_matrix = get_output_prob_matrix(param, total_score, candidate_pairs, scores_to_include, predictors)

    util.evaluate_output(output_prob_matrix, param, within_session, session_switch, match_threshold=0.75)
    output_threshold = np.zeros_like(output_prob_matrix)
    output_threshold[output_prob_matrix > 0.75] = 1  # might want to set match threshold to something other than 0.75.
    matches = np.argwhere(((output_threshold * within_session)) == True)  # exclude within session matches
    UIDs = aid.assign_unique_id(output_prob_matrix, param, clus_info)
    unitmatch_df = su.make_match_table(
        scores_to_include,
        matches,
        output_prob_matrix,
        total_score,
        output_threshold,
        clus_info,
        param,
        UIDs=UIDs,
        matches_curated=None,
    )  # options

    return unitmatch_df, clus_info, wave_dict


def match_with_uid(unitmatch_df):
    """INPUTS: unitmatch dataframe with CUID's appended to it.
    OUTPUTS: list of lists of matched clusters:
    [[CUID1,CUID2],[CUID3,CUID5],...]
    where CUID is of the form: subject.date.sessiontype_clusterN"""
    matched_df = unitmatch_df.query("`RecSes 1` != `RecSes 2` and `UID int 1` == `UM UID int 2`")
    matched_clusters_list = [x for x in matched_df.groupby("UID int 1")["CUID1"].apply(list)]
    matched_clusters_list = remove_duplicates_list_of_lists(matched_clusters_list)
    return matched_clusters_list


# %% 2. Running unit match on all sessions for a subject


def run_unitmatch_all_sessions():
    """Top level function to run unit match (for each subject) for all sessions for each mouse.
    Data will be saved under ../data/processed_data/subject/
    Submits jobs to clusters for each subject."""

    for each_subject in sps.get_ephys_paths_df()["subject_ID"].unique():
        # check jobs folder exits
        for jobs_folder in ["slurm", "out", "err"]:
            if not Path(f"SpikeSorting/jobs/{jobs_folder}").exists():
                os.mkdir(f"SpikeSorting/jobs/{jobs_folder}")

        print(f"Submitting {each_subject} pairs to HPC")
        script_path = get_unitmatch_SLURM_script(subject=each_subject)
        os.system(f"chmod +x {script_path}")
        os.system(f"sbatch {script_path}")
    return print("All ephys preprocessing jobs submitted to HPC. Check progress with 'squeue -u <username>'")


def get_unitmatch_SLURM_script(subject, RAM="256GB", time_limit="12:00:00"):
    """Writes out script to perform pairwise unit matching for all pairs of sessions
    for a given subject."""
    subject_ID = f"{subject}"
    script = f"""#!/bin/bash
#SBATCH --job-name=ephys_preprocessing_{subject_ID}
#SBATCH --output=SpikeSorting/jobs/out/processed_unit_matching_{subject_ID}.out
#SBATCH --error=SpikeSorting/jobs/err/processed_unit_matching_{subject_ID}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem={RAM}
#SBATCH --time={time_limit}

source $(conda info --base)/etc/profile.d/conda.sh
module load miniconda
module load cuda/11.8
conda deactivate
conda activate maze_ephys

python -c \"
import GridMaze2.analysis.core.unit_matching as um
print('Starting UnitMatch for {subject_ID}')
um.save_subject_unitmatch_data(subject_ID='{subject_ID}')
\"
"""
    script_path = f"SpikeSorting/jobs/slurm/unit_matching_{subject_ID}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    return script_path


def save_subject_unitmatch_data(subject_ID):
    """Saves out a unitmatch DF generated from all sessions of a subject,
    but also a json dictionary with all putative matched clusters across all sessions."""
    # get the huge unitmatch df
    # somehow get mapping onto unique cluster ID's and session info

    all_paths = paths_from_list_of_dicts([all_sessions_dict], subject_ID, ignore_missing_paths=True)

    unitmatch_path = PROCESSED_DATA_PATH / subject_ID / "UnitMatch"
    if not unitmatch_path.exists():
        unitmatch_path.mkdir(parents=True)
    if not (unitmatch_path / "unitmatch_df.htsv").exists():
        unitmatch_df, _, _ = run_unitmatch(all_paths)
        print("adding CUID to the dataframe")
        unitmatch_df = add_CUID_to_unitmatch_df(unitmatch_df, all_paths)

        print(f"Saving unitmatch_df for {subject_ID}")
        unitmatch_df.to_csv((unitmatch_path / "unitmatch_df.htsv"), index=False, sep="\t")
    else:
        print("loading unitmatch_df")
        unitmatch_df = pd.read_csv((unitmatch_path / "unitmatch_df.htsv"), sep="\t")

    if not (unitmatch_path / "matched_clusters.json").exists():
        print(f"Getting dictionary of matched clusters for {subject_ID}")
        matched_clusters_list = match_with_uid(unitmatch_df)
        keys = [f"{subject_ID}_matched_cluster_{x}" for x in np.arange(len(matched_clusters_list))]
        matched_dict = dict(zip(keys, matched_clusters_list))
        with open((PROCESSED_DATA_PATH / subject_ID / "matched_clusters.json"), "w") as file:
            json.dump(matched_dict, file)

    if not (unitmatch_path / "probabilities.htsv").exists():
        print(f"Saving out probabilities and score matrices for {subject_ID}")
        unitmatch_df2matrices(unitmatch_df, unitmatch_path)
    return print("Saved out unitmatch data")


def add_CUID_to_unitmatch_df(unitmatch_df, processed_data_paths):
    """Adds cluster_unique_ID (CUID) to the unitmatch dataframe.
    Note that some rows will have CUID1 == CUID2 when it's a within-session comparison"""

    CUID1_list = []
    CUID2_list = []
    for each_row in range(len(unitmatch_df)):
        # index out information of a cluster from the matched_cluster_df
        for i in [1, 2]:  # for each split half / session
            unit_session_idx = int(unitmatch_df.iloc[each_row][f"RecSes {i}"] - 1)  # python indexing starting at 0
            unit_cluster_idx = unitmatch_df.iloc[each_row][f"ID{i}"]  # kilosort cluster index

            # make it processed_data session level unique ID
            session_info = gs.load_data.load(processed_data_paths[unit_session_idx] / "session_info.json")
            session_cluster_UID = convert.cluster_IDs2scluster_unique_IDs(session_info, unit_cluster_idx)
            if i == 1:
                CUID1_list.append(session_cluster_UID)
            elif i == 2:
                CUID2_list.append(session_cluster_UID)

    unitmatch_df["CUID1"] = CUID1_list
    unitmatch_df["CUID2"] = CUID2_list

    return unitmatch_df


def unitmatch_df2matrices(unitmatch_df, unitmatch_path):
    """Saves out matrices with CUID as columns and indices,
    values are the named unit match measure.
    NB: must have appended CUID to unitmatch using add_CUID_to_unitmatch_df()"""

    for each_column in [
        "UM Probabilities",
        "TotalScore",
        "amp_score",
        "spatial_decay_score",
        "centroid_overlord_score",
        "centroid_dist",
        "waveform_score",
        "trajectory_score",
    ]:
        matrix = pd.pivot(unitmatch_df, values=each_column, index="CUID1", columns="CUID2")
        if each_column == "UM Probabilities":
            each_column = "um_probabilities"  # renaming the matrix here.
        matrix.to_csv((unitmatch_path / f"{each_column}.htsv"), index=False, sep="\t")


def load_matched_clusters(subject_ID: str, list_of_dicts: list, ignore_missing_paths=False, matches_all_sessions=True):
    # 1. initialise list of matched_clusters
    cluster_matches = []
    for session_dict in list_of_dicts:

        # 2. get a filter
        paths = get_paths_to_UM_sessions(
            subject_ID,
            session_dict["session_types"],
            session_dict["maze_names"],
            session_dict["days_on_maze"],
            session_dict["goal_subsets"],
            ignore_missing_paths=ignore_missing_paths,
        )
        filter_list = [x.parts[-1] for x in paths]

        # 3. load the saved out matched_clusters dictionary from all sessions
        json_path = Path(f"../data/processed_data/{subject_ID}/UnitMatch/matched_clusters.json")
        with open(json_path, "r") as file:
            all_matched_clusters = json.load(file)

        # 4. select only clusters of interest
        for each_match in all_matched_clusters:
            filtered_clusters = []
            for each_cluster in all_matched_clusters[each_match]:
                if np.any([x in each_cluster for x in filter_list]):
                    filtered_clusters.append(each_cluster)
            if matches_all_sessions:
                if len(filtered_clusters) == len(filter_list):
                    cluster_matches.append(filtered_clusters)
            else:
                cluster_matches.append(filtered_clusters)
    return cluster_matches


# %% 3. Extracting matches from probability matrices
def load_prob_matrix_df(subject_ID):
    """Function to load probability matrix from saved file.

    Parameters:
    -----------
    subject_ID: str
        name of a subject used for filepaths.

    Returns:
    --------
    prob_matrix_df: pandas dataframe object
        Matrix containing unitmatch probabilities with cluster unique ID as columns and index."""

    try:
        prob_matrix_path = Path(f"../data/processed_data/{subject_ID}/UnitMatch/um_probabilities.htsv")
        prob_matrix_df = pd.read_csv(prob_matrix_path, sep="\t")
        prob_matrix_df.index = prob_matrix_df.columns
    except:
        FileNotFoundError("No saved out probability matrix. See save_subject_unitmatch_data()")
    return prob_matrix_df


def get_long_prob_matrix_df(prob_matrix_df):
    """Function to turn symmetric probability matrix into longform dataframe.

    Parameters:
    -----------
    prob_matrix_df: pandas Dataframe
        with CUID as columns and index, UM probabilities as values

    Returns:
    --------
    long_prob_df: pandas DataFrame
        with cuid_1, cuid_2, match_prob"""

    # Turn the original prob_matrix_df and turn it into a longform_df
    upper_tri_indices = np.triu_indices_from(
        prob_matrix_df, k=0
    )  # k=0 here includes the diagonal (prob of match to itself).

    # Create cuid_1, cuid_2, and match_prob columns
    cuid_1 = prob_matrix_df.index.values[upper_tri_indices[0]]
    cuid_2 = prob_matrix_df.columns.values[upper_tri_indices[1]]
    match_prob = prob_matrix_df.values[upper_tri_indices]

    # Create the long-form DataFrame
    long_prob_df = pd.DataFrame({"cuid_1": cuid_1, "cuid_2": cuid_2, "match_prob": match_prob})


def exclude_within_session_df(um_matrix_df):
    """INPUT: a dataframe containing a matrix of unitmatch outputs (probabilities, total score, or subscore)
    OUT: a mask (len(um_matrix_df),len(um_matrix_df))"""
    mask = get_within_session_mask(um_matrix_df)
    not_within_session_mask = [mask != 1][0]
    return um_matrix_df * not_within_session_mask


def get_within_session_mask(um_matrix_df):
    """INPUT: a pandas dataframe with cluster unique id's as columns,
         matrix values can be probabilities, total score or subscore
    OUT: a mask (len(um_matrix_df),len(um_matrix_df)) with diagonal blocks as 1s and evertyghing else 0s"""
    dates = np.array([col.split("_cluster") for col in um_matrix_df.columns])[:, 0]
    _, counts = np.unique(dates, return_counts=True)
    total_size = sum(counts)  # Total size of the square matrix
    mask = np.zeros((total_size, total_size))  # Initialize with zeros

    start = 0
    for size in counts:
        # Fill in the 1s for the current block
        mask[start : start + size, start : start + size] = 1
        start += size  # Update the starting position for the next block
    return mask


def get_best_within_session_matches(matched_clusters_list, prob_matrix_df):
    """Takes a list of matched clusters and returns another where each match contains only one cluster per session.

    Parameters
    ----------
    matched_clusters_list: list of lists
        e.g. [[CUID1,CUID2],[CUID3,CUID4]] where CUID is cluster unique ID of the form
        subject.session_type.date.cluster_ID
    prob_matrix_df: pandas dataframe
        CUID as index and columns, with UM probabilities as values

    Returns
    -------
    edited_clusters_list: list of lists
        new list of matches, with only one cluster per session in each match."""

    # We flatten the probability matrix, so first we ensure it's symmetric:
    prob_matrix_df = (prob_matrix_df + prob_matrix_df.T) / 2

    edited_clusters_list = []

    for each_match in matched_clusters_list:
        if len(each_match) > 1:
            # we build a dataframe with match_prob to clusters in other sessions for each cluster:
            # initialising lists here:
            cluster_session_list = []
            mean_match_probs = []
            for each_cluster in each_match:  # run over each cluster
                cluster_session_list.append(each_cluster.split("_cluster")[0])
                other_clusters = [x for x in each_match if cluster_session_list[-1] not in x]
                mean_match_probs.append(prob_matrix_df.loc[each_cluster, other_clusters].mean())
            avg_df = pd.DataFrame({"cuid": each_match, "match_prob": mean_match_probs, "session": cluster_session_list})
            avg_df.index = avg_df["cuid"]
            edited_clusters_list.append(avg_df.groupby("session").idxmax()["match_prob"].to_list())
        else:
            edited_clusters_list.append(each_match)
    return edited_clusters_list


def filter_to_sessions(matched_clusters, session_paths):
    """Using the session info from paths, filters the list of clusters to only include those with paths"""
    filter_list = [x.parts[-1] for x in session_paths]
    filtered_matches = []
    for each_match in matched_clusters:
        filtered_units = []
        for each_cluster in each_match:
            for each_session in filter_list:
                if each_session in each_cluster:
                    filtered_units.append(each_cluster)
        if len(filtered_units) > 0:
            filtered_matches.append(filtered_units)

    return filtered_matches


def merge_partition(old_partition, prob_matrix_df, merge_threshold=0.5, across_session_matching=True):
    """INPUT:
    old_partition: list of lists, e.g. [[CUID1,CUID2],[CUID3,CUID4],...]
    prob_matrix_df: pandas with CUID as columns and index
    merge_threshold: the average match probability to be considered for merging
    OUTPUT:
    new_partition: list of lists, e.g. [[CUID1,CUID2,CUID3,CUID4],[CIUD5,CUID6],...]

    NOTES: if across_session_matching is true, the mean match probability across subsets
        will be computed only across sessions (CUID1 match prob to CUID3 only if CUID1 and CUID3 are in different sessions).
        This may lead to within_session matches whenever both units within a session match well to the same set of units across sessions.
        if this is false, we are including the within session probabilities.
            These can be set to 0 to bias against within session matches.

        We recommend later cleaning the partition to resolve_within_session matches and excluding failed matches.
    """

    if across_session_matching:
        # we want to compute match prob only based on match probability to units across sessions
        prob_matrix_df = exclude_within_session_df(prob_matrix_df)
        prob_matrix_df = prob_matrix_df.replace(0, np.nan)

    # Turn the original prob_matrix_df and turn it into a longform_df
    upper_tri_indices = np.triu_indices_from(
        prob_matrix_df, k=0
    )  # k=0 here includes the diagonal (prob of match to itself).

    # Create cuid_1, cuid_2, and match_prob columns
    cuid_1 = prob_matrix_df.index.values[upper_tri_indices[0]]
    cuid_2 = prob_matrix_df.columns.values[upper_tri_indices[1]]
    match_prob = prob_matrix_df.values[upper_tri_indices]

    # Create the long-form DataFrame
    longform_df = pd.DataFrame({"cuid_1": cuid_1, "cuid_2": cuid_2, "match_prob": match_prob})

    # we want to map each 'cuid' column to a match_ID and then map these to a match_ID_pair
    # we do this by reverse indexing;
    from collections import defaultdict

    cluster_to_lists = defaultdict(list)
    for i, sublist in enumerate(old_partition):
        for cluster in sublist:
            cluster_to_lists[cluster].append(i)

    longform_df["match_ID_1"] = longform_df.cuid_1.apply(
        lambda x: cluster_to_lists[str(x)][0] if len(cluster_to_lists[str(x)]) > 0 else np.nan
    )
    longform_df["match_ID_2"] = longform_df.cuid_2.apply(
        lambda x: cluster_to_lists[str(x)][0] if len(cluster_to_lists[str(x)]) > 0 else np.nan
    )
    longform_df["match_ID_pair"] = list(zip(longform_df["match_ID_1"], longform_df["match_ID_2"]))
    # we sort and threshold the long dataframe here:
    sorted_df = (
        longform_df.groupby("match_ID_pair")
        .mean("match_prob")
        .sort_values(by="match_prob", ascending=False)
        .query(f"match_ID_1!=match_ID_2 and match_prob>{merge_threshold}")
    )
    print(f"Matching from: \n {sorted_df}")
    # Now we're ready to merge:
    new_partition = []  # initialise partition
    visited_matches = []
    for each_row in sorted_df.itertuples():
        if (each_row.match_ID_1 in visited_matches) or (each_row.match_ID_2 in visited_matches):
            continue
        else:
            matched_cluster_list_1 = old_partition[int(each_row.match_ID_1)]
            matched_cluster_list_2 = old_partition[int(each_row.match_ID_2)]
            new_partition.append(matched_cluster_list_1 + matched_cluster_list_2)
            visited_matches += [each_row.match_ID_1] + [each_row.match_ID_2]
    # add unmatched units to the partition (singletons)
    unvisited_match_IDs = [x for x in np.arange(len(old_partition)) if x not in visited_matches]
    new_partition += [old_partition[x] for x in unvisited_match_IDs]
    return new_partition


def match_with_partition(prob_matrix_df, threshold=0.5, across_session_matching=False, exclude_within_session=True):
    """INPUT: pandas dataframe probability matrix with unit labels as columns and index,
            threshold for merging two subsets of a partition.
            across_session_matching option to consider only the match probability across sessions (ignoring within-session probabilities)
    NOTES: if across_session_matching is true, the mean match probability across subsets
        will be computed only across sessions (CUID1 match prob to CUID3 only if CUID1 and CUID3 are in different sessions).
        This may lead to within_session matches whenever both units within a session match well to the same set of units across sessions.
        if this is false, we are including the within session probabilities.
            These can be set to 0 to bias against within session matches.
        We recommend later cleaning the partition to resolve_within_session matches and excluding failed matches.
    """

    if exclude_within_session:  # sets within_session match probabilities to 0
        prob_matrix_df = exclude_within_session_df(prob_matrix_df)

    old_partition = [[x] for x in prob_matrix_df.columns]  # trivial partition, where each unit belongs to a sublist
    old_length = len(old_partition)
    new_length = 0
    while new_length != old_length:
        old_length = len(old_partition)
        new_partition = merge_partition(
            old_partition, prob_matrix_df, merge_threshold=threshold, across_session_matching=False
        )
        old_partition = new_partition
        new_length = len(new_partition)
        print(f"{old_length}=>{new_length}")

    return new_partition


def match_with_leiden(prob_matrix, threshold=0.5, resolution_parameter=0.1):
    """
    Convert a probability matrix to a graph and cluster it into communities using the Leiden algorithm.

    Parameters:
    -----------
    prob_matrix : pd.DataFrame
        A symmetric DataFrame representing probabilities between nodes (e.g., clusters).
    threshold : float, optional
        Minimum probability required to include an edge in the graph. Default is 0.5.

    Returns:
    --------
    clusters : list of lists
        A list where each sublist contains the indices of nodes belonging to the same community.
    """
    # Validate input
    if not isinstance(prob_matrix, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not (prob_matrix.values == prob_matrix.values.T).all():
        raise ValueError("The input DataFrame must be symmetric.")

    # Create an igraph Graph object
    g = ig.Graph()

    # Add nodes to the graph (using the index/columns of the DataFrame as labels)
    node_labels = list(prob_matrix.columns)
    g.add_vertices(node_labels)

    # Add edges with weights above the threshold
    edges = []
    weights = []

    for i in range(prob_matrix.shape[0]):
        for j in range(i + 1, prob_matrix.shape[1]):  # Only upper triangular part
            if prob_matrix.iloc[i, j] >= threshold:
                edges.append((node_labels[i], node_labels[j]))
                weights.append(prob_matrix.iloc[i, j])

    g.add_edges(edges)
    g.es["weight"] = weights

    # Perform Leiden clustering
    partition = la.find_partition(g, la.CPMVertexPartition, weights="weight", resolution_parameter=resolution_parameter)

    # Extract clusters
    matched_clusters_list = [[node_labels[node] for node in community] for community in partition]

    return matched_clusters_list


def match_with_agglomerative(prob_matrix, distance_threshold=0.5, linkage="complete"):
    """
    Convert a probability matrix to a distance matrix and cluster using Agglomerative Clustering.

    Parameters:
    -----------
    prob_matrix : pd.DataFrame
        A symmetric DataFrame representing probabilities between nodes (e.g., clusters).
    distance_threshold : float, optional
        Maximum distance for two points to be in the same cluster. Default is 0.5.

    Returns:
    --------
    clusters : list of lists
        A list where each sublist contains the indices of nodes belonging to the same community.
    """
    # Validate input
    if not isinstance(prob_matrix, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not (prob_matrix.values == prob_matrix.values.T).all():
        raise ValueError("The input DataFrame must be symmetric.")

    # Convert probabilities to distances (1 - probability)
    distance_matrix = 1 - prob_matrix.values

    # Ensure the distance matrix is symmetric
    assert np.allclose(distance_matrix, distance_matrix.T), "Distance matrix must be symmetric."

    # Fit Agglomerative Clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,  # Automatically determine the number of clusters
        metric="precomputed",
        linkage=linkage,
        distance_threshold=distance_threshold,
    )
    clustering.fit(distance_matrix)

    # Extract clusters
    labels = clustering.labels_
    matched_clusters_list = []
    for cluster_id in np.unique(labels):
        matched_clusters_list.append(list(prob_matrix.index[labels == cluster_id]))

    return matched_clusters_list


def resolve_one_to_many(matched_clusters_list, prob_matrix):
    """
    Function to make sure that each cluster only belongs to one list of matched clusters.
    """

    # 1. Identify which 'matched_clusters' a cell is assigned to.
    # We use something called a 'reverse index'
    cluster_to_lists = defaultdict(list)
    for i, sublist in enumerate(matched_clusters_list):
        for cluster in sublist:
            cluster_to_lists[cluster].append(i)
    # NB: due to the way we match, if a cell appears in exactly 2 clusters, these two clusters must be identical.
    # if a cell appears in 1 cluster, there was no match and it is latter filtered away.

    # 2. We want to remove a cell from clusters that are suboptimally matched
    edited_clusters_list = matched_clusters_list.copy()
    for cluster, list_indices in cluster_to_lists.items():
        if len(list_indices) > 2:  # we only care if there are conflicts (a cell in more than 2 clusters)
            # Multiple lists: find the best cluster family based on probabilities
            mean_match_probs = []
            for each_match in list_indices:
                mean_match_probs.append(
                    np.mean(
                        [
                            prob_matrix.loc[cluster, other]
                            for other in matched_clusters_list[each_match]
                            if other != cluster
                        ]
                    )
                )
            best_list_idx = list_indices[np.argmax(mean_match_probs)]
            for idx in list_indices:
                if idx != best_list_idx:
                    edited_clusters_list[idx].remove(cluster)
    return edited_clusters_list


def remove_duplicates_list_of_lists(input_list):
    """In cases such as [[CUID1,CUID2],[CUID2,CUID1],[CUID4,CUID5,CUID6]]
    where a match is counted twice (with opposite orderings) but matches have different list lengths
    -removes duplicate lists.
    """
    seen = set()  # Keep track of seen items
    unique_lists = []  # Store unique lists
    for sublist in input_list:
        sublist_tuple = tuple(sorted(sublist))  # Convert to sorted tuple for consistent order
        if sublist_tuple not in seen:
            seen.add(sublist_tuple)  # Mark as seen
            unique_lists.append(list(np.unique(np.array(sublist))))  # Add original list to results
    return unique_lists


# %% 4. Custom probability matrices


def get_SVM_prob_matrix(subject_ID, unitmatch_df=None):
    """
    Addressing match probabilities using SVM and all subscores.
    Parameters:
    -----------
    subject_ID : str()
    unitmatch_df : pandas dataframe, optional
        Option to exctract predictors from unitmatch output dataframe if provided

    Returns:
    --------
    prob_matrix_df : pandas dataframe
        Large n_units x n_units matrix with match probabilities. CUID as columns and index
    """

    um_path = Path(f"../data/processed_data/{subject_ID}/UnitMatch/")
    svm_probs_path = um_path / "svm_probabilities.htsv"
    # load if saved
    if svm_probs_path.exists():
        prob_matrix_df = pd.read_csv(svm_probs_path, sep="\t")
    else:
        # generate otherwise

        # 1. get 'predictors', namely the subscores for each split-half of the same unit
        predictor_list = []  #
        for each_score in [
            "amp_score",
            "centroid_dist",
            "centroid_overlord_score",
            "spatial_decay_score",
            "trajectory_score",
            "waveform_score",
        ]:
            if unitmatch_df is None:  # loading from saved data
                try:
                    predictor_df = pd.read_csv(
                        f"../data/processed_data/{subject_ID}/UnitMatch/{each_score}.htsv", sep="\t"
                    )
                except:
                    FileNotFoundError("Failed to load saved out similarity score. Check save_subject_unitmatch_data() ")
            if unitmatch_df is not None:  #
                if each_score == "um_probabilities":
                    each_score = "UM Probabilities"  # rename the matrix in this case.
                predictor_df = pd.pivot(unitmatch_df, values=each_score, index="CUID1", columns="CUID2")
            predictor_list.append(predictor_df)
        predictors = np.stack(predictor_list, axis=2)  # shape (n_units, n_units, n_predictors)

        # 2. split into test and training data to check accuracy
        """NB: we train on data within a session, where we assume units only match to themselves.
        This gives us match (1) and no match (0) labels for training the classifiers.
        """
        within_session = get_within_session_mask(
            predictor_df
        )  # we can use the last predictor_df from the above loop to get a mask
        match_to_self = (
            np.eye(within_session.shape[0], within_session.shape[1]) == 1
        )  # serves as ground truth for training a classifier
        X = predictors[within_session == True]  # training on
        y = match_to_self[within_session == True]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )  # imported function from sklearn

        # 3. train a classifier and estimate match probabilities
        classifier = SVC(
            kernel="poly",
            degree=2,
            gamma="auto",
            C=1,
            decision_function_shape="ovr",  #'one-versus-rest'
            probability=True,
        )
        classifier.fit(X_train, y_train)
        accuracy = classifier.score(X_test, y_test)
        print(f"Within-session matching cross-validated accuracy: {accuracy}")
        all_data = predictors.reshape(predictors.shape[0] * predictors.shape[1], predictors.shape[2])
        all_match_probs = [x[1] for x in classifier.predict_proba(all_data)]
        match_prob_matrix = np.array(all_match_probs).reshape(
            int(np.sqrt(len(all_match_probs))), int(np.sqrt(len(all_match_probs)))
        )
        prob_matrix_df = pd.DataFrame(match_prob_matrix, columns=predictor_df.columns)
        # save out the prob matrix for later fast reading
        prob_matrix_df.to_csv(svm_probs_path, index=False, sep="\t")

    prob_matrix_df.index = prob_matrix_df.columns

    return prob_matrix_df


# %% 5. Functions for QC of UnitMatch matches.


def get_pairwise_report(cluster_a, cluster_b, save=True):
    """Generates a plot to quality control a pair of matched clusters.
    INPUT: two cluster objects and an optional save toggle
    OUTPUT: a plot overlaying average waveforms, waveform trajectories, and distribution of spikes.
    if save=True, it saves to experiment/results/unit_match_reports/subject/session (both session folders)
    Notes: The order of the clustering will change the UM probabilities score since these are split-half dependent;
           this is computed with reference to cluster_a as the first split-half."""

    # We want to save each plot as we make it so we can retrieve faster later:

    save_path_a = (
        RESULTS_PATH / "unit_match_reports" / f"{cluster_a.subject_ID}" / f"{cluster_a.processed_data_path.parts[-1]}"
    )
    save_path_a.mkdir(exist_ok=True, parents=True)
    save_path_b = (
        RESULTS_PATH / "unit_match_reports" / f"{cluster_b.subject_ID}" / f"{cluster_b.processed_data_path.parts[-1]}"
    )
    save_path_b.mkdir(exist_ok=True, parents=True)
    filename = f"{cluster_a.cluster_unique_ID}X{cluster_b.cluster_unique_ID}.jpg"

    # Quickly check if already saved:
    if (save_path_a / filename).exists() or (save_path_b / filename).exists():
        display(Image(filename=(save_path_a / filename)))
    else:  # otherwise generate!
        # Run unitmatch pairwise
        processed_paths = [x.processed_data_path for x in [cluster_a, cluster_b]]
        unitmatch_df, clus_info, wave_dict = run_unitmatch(processed_paths)

        # get match information with respect to cluster_a as the first split half
        across_session_df = unitmatch_df.query("`RecSes 1`!= `RecSes 2`")
        match_info = across_session_df.query(
            f"`RecSes 1`==1 and ID1=={cluster_a.cluster_ID} and `RecSes 2`==2 and ID2=={cluster_b.cluster_ID}"
        ).iloc[0]

        # extract info for plotting
        unit_a_session = int(match_info["RecSes 1"] - 1)
        unit_b_session = int(match_info["RecSes 2"] - 1)
        unit_a_mask = (clus_info["original_ids"] == match_info["ID1"]).T * (clus_info["session_id"] == unit_a_session)
        unit_b_mask = (clus_info["original_ids"] == match_info["ID2"]).T * (clus_info["session_id"] == unit_b_session)
        wave_dict_idx_a = np.argwhere(unit_a_mask[0] == True)[0][0]
        wave_dict_idx_b = np.argwhere(unit_b_mask[0] == True)[0][0]
        wave_dict_idxs = [wave_dict_idx_a, wave_dict_idx_b]
        avg_wave = np.mean(
            wave_dict["avg_waveform"], axis=2
        )  # (n_timesteps, n_clusters), averaging over split halves or 'cv'
        avg_pos = np.mean(wave_dict["avg_waveform_per_tp"], axis=3)  # (n_coords,n_clusters,n_timesteps)
        channel_pos = get_cluster_channel_pos(cluster_a)
        # load analyzers for autocorrelograms and spike distributions
        # this is using data computed and stored via spikesort_session.py
        analyzer_paths = [cluster_obj2preprocessed_data_path(x) / "sorting_analyzer" for x in [cluster_a, cluster_b]]
        analyzers = [si.load_sorting_analyzer(x) for x in analyzer_paths]
        analyzer_idxs = [int(match_info["ID1"]), int(match_info["ID2"])]

        # Add colour coded title text
        colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # blue is 0, orange is 1
        subject = cluster_a.subject_ID  # a bit finicky, but inherited from data structure

        match_text = (
            f"{round(match_info['UM Probabilities']*100,2)}% match \n {round(match_info['TotalScore'],3)} total score"
        )
        text_a = cluster_a.cluster_unique_ID
        text_b = cluster_b.cluster_unique_ID

        # Set up the figure and axes.
        fig = plt.figure(figsize=(10, 4))
        subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[1, 3])  # large column to left,
        axsLeft = subfigs[0].subplots(1, 1)
        axs = subfigs[1].subplots(2, 2)

        fig.text(0.1, 1.05, match_text, ha="center", va="bottom", size="large")
        fig.text(0.4, 1.05, text_a, ha="center", va="bottom", size="large", color=colours[0])
        fig.text(0.8, 1.05, text_b, ha="center", va="bottom", size="large", color=colours[1])

        for each_unit in range(2):
            axsLeft.set(title="Waveform templates")
            sw.plot_unit_waveforms(
                analyzers[each_unit],
                plot_waveforms=True,
                plot_templates=True,
                alpha_waveforms=0.001,
                alpha_templates=0.5,
                unit_ids=[analyzer_idxs[each_unit]],
                unit_colors={analyzer_idxs[each_unit]: colours[each_unit]},
                set_title=False,
                plot_legend=False,
                backend="matplotlib",
                same_axis=True,
                **{"ax": axsLeft},
            )

            axs[0, 0].set(title="Average waveforms")
            axs[0, 0].plot(avg_wave[:, wave_dict_idxs[each_unit]])
            axs[1, 0].set(title="Average centroid")
            # first we want to plot on a scaffold of channel locations around the centroid
            max_channel_idx = wave_dict["max_site"][wave_dict_idxs[each_unit], 0]
            axs[1, 0].scatter(
                x=channel_pos[(max_channel_idx - 4) : (max_channel_idx + 4), 0],
                y=channel_pos[(max_channel_idx - 4) : (max_channel_idx + 4), 1],
                marker="s",
                color="gray",
            )
            axs[1, 0].scatter(
                x=avg_pos[1, wave_dict_idxs[each_unit], :],
                y=avg_pos[2, wave_dict_idxs[each_unit], :],
                alpha=0.3,
                color=colours[each_unit],
            )

            axs[0, 1].set(title="Spike amplitude distributions")
            sw.plot_amplitudes(
                analyzers[each_unit],
                plot_histograms=False,
                plot_legend=False,
                unit_ids=[analyzer_idxs[each_unit]],
                unit_colors={analyzer_idxs[each_unit]: colours[each_unit]},
                backend="matplotlib",
                **{"ax": axs[0, 1]},
            )
            for artist in axs[0, 1].collections:
                artist.set_alpha(0.3)  # Adjust alpha for all scatter points
            # lineplot autocorrelograms
            axs[1, 1].set(title="Normalised AutoCorrelogram")
            corr_data = analyzers[each_unit].get_extension("correlograms").get_data()
            bins = corr_data[1]  # Time bins for correlograms
            corr_values = corr_data[0][
                analyzer_idxs[each_unit], analyzer_idxs[each_unit], :
            ]  # CCG values for the specific unit pair
            corr_normalised = corr_values / max(corr_values)
            axs[1, 1].plot(bins[:-1], corr_normalised, linestyle="-", alpha=0.5)
        fig.tight_layout()
        if save:
            fig.savefig(save_path_a / filename, bbox_inches="tight")
            fig.savefig(save_path_b / filename, bbox_inches="tight")
    return


# %% Utility functions
def get_cluster_channel_pos(cluster):
    """INPUT: cluster object
    OUTPUT: channel positions for plotting behind centroid trajectories.

    Very niche subfunction"""

    # get's a teeny bit convoluted for handling multiple probes
    subject_ID, date, session_type = cluster.name.split(".")
    n_probes = sps.get_n_probes(subject_ID)
    if n_probes == 1:
        probe_params_path = sps.SPIKESORTING_PATH / "probe_params" / subject_ID
    elif n_probes == 2:
        data_paths_df = gdd.get_sessions_data_directory()
        filtered_df = data_paths_df.query(
            f'subject_ID=="{subject_ID}" and date=="{date}" and session_type=="{session_type}"'
        )
        probe_suffix = Path(filtered_df.iloc[0].spikesorting_path).parts[-1]
        probe_params_path = sps.SPIKESORTING_PATH / "probe_params" / cluster.subject_ID / probe_suffix
    saved_probe = pd.read_csv(probe_params_path / "probe_layout.tsv")
    channel_pos = [[x, y] for x, y in zip(saved_probe["x"], saved_probe["y"])]
    return np.array(channel_pos)


def cluster_obj2processed_data_path(cluster_obj):
    subject_ID, date, session_type = cluster_obj.name.split(".")
    return Path(f"../data/processed_data/{subject_ID}/{date}.{session_type}")


def cluster_obj2preprocessed_data_path(cluster_obj):
    """Returns the preprocessed_data/subject/session path for a given cluster object"""
    preprocessed_data_path = Path("../data/preprocessed_data")
    if not preprocessed_data_path.exists():
        raise FileNotFoundError("preprocessed_data directory not found. Must be mounted to display reports.")
    else:
        # we need to map cluster ID's to unitmatch reports. This is as follows:
        data_paths_df = gdd.get_sessions_data_directory()  # gridmaze utility function
        datetimes = []
        unit_id = []

        subject_ID, date, session_type = cluster_obj.name.split(".")

        filtered_df = data_paths_df.query(
            f'subject_ID=="{subject_ID}" and date=="{date}" and session_type=="{session_type}"'
        )
        for each_part in Path(filtered_df["ephys_path"].values[0]).parts:
            # going over a loop of all file path parts, since multi-probe data may have slight variation in the order in the filepath that datetime appears in
            try:
                dt = datetime.strptime(each_part, "%Y-%m-%d_%H-%M-%S").isoformat()
            except:
                continue
            datetimes.append(dt)

    return preprocessed_data_path / "spikesorting" / subject_ID / dt


# %% Unit match subfunctions


def extract_metric_scores(extracted_wave_properties, session_switch, within_session, param, niter=2):
    """
    This function runs all of the metric calculations and drift correction to calculate the probability
    distribution needed for UnitMatch.

    Parameters
    ----------
    extracted_wave_properties : dict
        The extracted properties from extract_parameters()
    session_switch : ndarray
        An array which indicates when anew recording session starts
    within_session : ndarray
        The array which gives each unit a label depending on their session
    param : dict
        The param dictionary
    niter : int, optional
        The number of pass through the function, 1 mean no drift correction
            2 is one pass of drift correction, by default 2

    Returns
    -------
    ndarrays
        The total scores and candidate pairs needed for probability analysis
    """

    # unpack needed arrays from the ExtractedWaveProperties dictionary
    amplitude = extracted_wave_properties["amplitude"]
    spatial_decay = extracted_wave_properties["spatial_decay"]
    spatial_decay_fit = extracted_wave_properties["spatial_decay_fit"]
    avg_waveform = extracted_wave_properties["avg_waveform"]
    avg_waveform_per_tp = extracted_wave_properties["avg_waveform_per_tp"]
    avg_centroid = extracted_wave_properties["avg_centroid"]

    # These scores are NOT effected by the drift correction
    amp_score = mf.get_simple_metric(amplitude)
    spatial_decay_score = mf.get_simple_metric(spatial_decay)
    spatial_decay_fit_score = mf.get_simple_metric(spatial_decay_fit, outlier=True)
    wave_corr_score = mf.get_wave_corr(avg_waveform, param)
    wave_mse_score = mf.get_waveforms_mse(avg_waveform, param)

    # affected by drift
    for i in range(niter):
        avg_waveform_per_tp_flip = mf.flip_dim(avg_waveform_per_tp, param)
        euclid_dist = mf.get_Euclidean_dist(avg_waveform_per_tp_flip, param)

        centroid_dist, centroid_var = mf.centroid_metrics(euclid_dist, param)

        euclid_dist_rc = mf.get_recentered_euclidean_dist(avg_waveform_per_tp_flip, avg_centroid, param)

        centroid_dist_recentered = mf.recentered_metrics(euclid_dist_rc)
        traj_angle_score, traj_dist_score = mf.dist_angle(avg_waveform_per_tp_flip, param)

        # Average Euc Dist
        euclid_dist = np.nanmin(euclid_dist[:, param["peak_loc"] - param["waveidx"] == 0, :, :].squeeze(), axis=1)

        # TotalScore
        include_these_pairs = np.argwhere(euclid_dist < param["max_dist"])  # array indices of pairs to include
        include_these_pairs_idx = np.zeros_like(euclid_dist)
        include_these_pairs_idx[euclid_dist < param["max_dist"]] = 1

        # Make a dictionary of score to include
        centroid_overlord_score = (centroid_dist_recentered + centroid_var) / 2
        waveform_score = (wave_corr_score + wave_mse_score) / 2
        trajectory_score = (traj_angle_score + traj_dist_score) / 2

        scores_to_include = {
            "amp_score": amp_score,
            "spatial_decay_score": spatial_decay_score,
            "centroid_overlord_score": centroid_overlord_score,
            "centroid_dist": centroid_dist,
            "waveform_score": waveform_score,
            "trajectory_score": trajectory_score,
        }

        total_score, predictors = mf.get_total_score(scores_to_include, param)

        # Initial thresholding
        if i < niter - 1:
            # get the thershold for a match
            thrs_opt = mf.get_threshold(total_score, within_session, euclid_dist, param, is_first_pass=True)

            param["n_expected_matches"] = np.sum((total_score > thrs_opt).astype(int))
            prior_match = 1 - (param["n_expected_matches"] / len(include_these_pairs))
            candidate_pairs = total_score > thrs_opt
            drifts, avg_centroid, avg_waveform_per_tp = mf.drift_n_sessions(
                candidate_pairs, session_switch, avg_centroid, avg_waveform_per_tp, total_score, param, best_drift=False
            )  # NB: set to false to apply only basic drift correction

    thrs_opt = mf.get_threshold(total_score, within_session, euclid_dist, param, is_first_pass=False)
    param["n_expected_matches"] = np.sum((total_score > thrs_opt).astype(int))
    prior_match = 1 - (param["n_expected_matches"] / len(include_these_pairs))
    print(prior_match)
    if abs(prior_match) < 1:  # if there's a weird error after drift correction, we ignore it, otherwise
        thrs_opt = np.quantile(total_score[include_these_pairs_idx.astype(bool)], prior_match)
        candidate_pairs = total_score > thrs_opt
    else:
        print(f"Odd value of Prior Match = {prior_match}, so ignoring drift correction")

    return total_score, candidate_pairs, scores_to_include, predictors


def get_output_prob_matrix(param, total_score, candidate_pairs, scores_to_include, predictors):
    prior_match = 1 - (param["n_expected_matches"] / param["n_units"] ** 2)  # freedom of choose in prior prob
    priors = np.array((prior_match, 1 - prior_match))

    labels = candidate_pairs.astype(int)
    cond = np.unique(labels)
    score_vector = param["score_vector"]
    parameter_kernels = np.full((len(score_vector), len(scores_to_include), len(cond)), np.nan)

    parameter_kernels = bf.get_parameter_kernels(scores_to_include, labels, cond, param, add_one=1)

    probability = bf.apply_naive_bayes(parameter_kernels, priors, predictors, param, cond)

    output_prob_matrix = probability[:, 1].reshape(param["n_units"], param["n_units"])
    return output_prob_matrix


def zero_center_waveform(waveform):
    """
    Centers waveform about zero, by subtracting the mean of the first 15 time points.
    This function is useful for Spike Interface where the waveforms are not centered about 0.

    Arguments:
        waveform - ndarray (nUnits, Time Points, Channels, CV)

    Returns:
        Zero centered waveform
    """
    waveform = waveform - np.broadcast_to(waveform[:, :15, :, :].mean(axis=1)[:, np.newaxis, :, :], waveform.shape)
    return waveform


# %% OBS: This function shouldn't exist, but is here due to IBL preprocessing and previously failed padding:
# Here simply implemented at processed data level


def pad_unitmatch_inputs(UM_input_path, max_n_channels=PARAM["max_n_channels"]):
    """Function to account for different numbers of channels between sessions,
    due to outside brain channels being removed in preprocessing."""

    # First we pad positions, shaped [n_channels,n_coords]:
    positions = np.load(UM_input_path / "channel_positions.npy")  # (n_channels,2) is the shape here
    if positions.shape[0] < max_n_channels:
        n_pad = max_n_channels - positions.shape[0]
        positions_padded = np.concatenate([positions, np.zeros((n_pad, 2))], axis=0)
        print(f"Padding data to {max_n_channels} channels")
        np.save((UM_input_path / "channel_positions.npy"), positions_padded)

    # Next, we pad units
    raw_waveforms_dir = UM_input_path / "RawWaveforms"
    unit_files = list(raw_waveforms_dir.glob("Unit*_RawSpikes.npy"))
    for unit_file in unit_files:
        # Load raw spikes and positions for each unit
        raw_spikes = np.load(unit_file)
        # double-check padding is needed
        if raw_spikes.shape[1] < max_n_channels:
            # Pad raw spikes along the second axis (channels)
            n_pad = max_n_channels - raw_spikes.shape[1]
            raw_spikes_padded = np.concatenate(
                [raw_spikes, np.zeros((raw_spikes.shape[0], n_pad, raw_spikes.shape[2]))], axis=1
            )
            # Save the padded data, overwriting the original files
            np.save(unit_file, raw_spikes_padded)  # Overwrite the raw spikes file

    return print(f"checked padding for unitmatch inputs at {UM_input_path}")


# the following function also more clearly rejects bad units, by adjusting single unit criteria:
def save_unitmatch_labels(subject_ID):
    """INPUT: path object to processed data
    OUTPUT: cluster_group.tsv file with 'mua' and 'good' labels assigned by quality metrics"""
    for each_session in os.listdir(PROCESSED_DATA_PATH / subject_ID):
        # some dataframe wrangling
        try:
            metrics_df = pd.read_csv(
                PROCESSED_DATA_PATH / subject_ID / each_session / "clusters.metrics.htsv", sep="\t"
            )
        except:
            print(f"Missing {Path(each_session)/'clusters.metrics.htsv'}. Skipping.")
            continue
        metrics_df.columns = pd.MultiIndex.from_tuples(
            [col.split(".") if "." in col else [col, ""] for col in metrics_df.columns]
        )
        quality_metrics_df = metrics_df.quality_metrics
        quality_metrics_df["unit_id"] = metrics_df["cluster_ID"]
        # assign single units with standard thresholding
        single_units = sps.get_single_units(
            quality_metrics_df,
            isi_violations_ratio_thres=0.1,
            amplitude_cutoff_thres=0.1,
            firing_rate_thres=0.1,
            presence_ratio_thres=0.9,
            amplitude_median_thres=50,
        )
        cluster_group_path = PROCESSED_DATA_PATH / subject_ID / each_session / "UnitMatch" / "cluster_group.tsv"
        cluster_group_df = pd.read_csv(cluster_group_path, sep="\t")
        cluster_group_df["KSLabel"] = "mua"
        cluster_group_df.loc[single_units, "KSLabel"] = "good"
        cluster_group_df.to_csv(cluster_group_path, sep="\t", index=False)  # index false is important here.
    return cluster_group_df


def get_within_day_matches_df():
    """Generates within-day-matches maze-open_field matches using UM_selected."""

    save_path = Path("../analyses/gridcells/within_day_match_df.htsv")

    if save_path.exists():
        within_day_matches_df = pd.read_csv(save_path, sep="\t")
    else:
        # generate a big dataframe with the following columns:
        of_CUID = []
        maze_CUID = []
        subject_ID = []
        maze_name = []
        day_on_maze = []
        goal_subset = []
        of_gridscore_60 = []

        for each_subject in ["mEC_2", "mEC_5", "mEC_6", "mEC_7", "mEC_8"]:
            for each_maze in ["fully_connected", "maze_1", "maze_2", "rooms_maze"]:
                for each_day in np.arange(12) + 1:
                    sessions_dict = {
                        "session_types": ["open_field", "maze"],
                        "maze_names": [each_maze],
                        "days_on_maze": [each_day],
                        "goal_subsets": "all",
                    }
                    try:
                        matches = get_matched_clusters(
                            each_subject, [sessions_dict], method="um_selected_uid", return_cluster_objects=False
                        )
                    except:
                        print(f"Failed to unitmatch for {each_subject} {each_maze} on day {each_day}")
                        continue

                    for each_match in matches:
                        of_CUID.append(each_match[0])
                        maze_CUID.append(each_match[1])
                        subject_ID.append(each_subject)
                        maze_name.append(each_maze)
                        day_on_maze.append(each_day)
                        goal_subset.append(gc.get_cluster(each_match[0]).goal_subset)
                        try:
                            metrics = gc.get_cluster(each_match[0]).load_tuning_data("metrics")
                            of_gridscore_60.append(metrics.gridscore["60"])
                        except:
                            of_gridscore_60.append(np.nan)
                            continue
        within_day_matches_df = pd.DataFrame(
            {
                "of_CUID": of_CUID,
                "maze_CUID": maze_CUID,
                "subject_ID": subject_ID,
                "maze_name": maze_name,
                "day_on_maze": day_on_maze,
                "goal_subset": goal_subset,
                "of_gridscore_60": of_gridscore_60,
            }
        )
        within_day_matches_df.to_csv(save_path, sep="\t", index=False)

    return within_day_matches_df
