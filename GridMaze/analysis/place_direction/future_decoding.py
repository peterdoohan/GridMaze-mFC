"""
Can we decode the future position/place-direction of the animal from neural avtivity
(above that predicted by the current place-direction)?
@krisjensen @peterdoohan
"""

# %% Imports
import json
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import folds
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import convert

from GridMaze.maze import representations as mr

# %% Global Variables

from GridMaze.paths import RESULTS_PATH, EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

RESULTS_DIR = RESULTS_PATH / "place_direction" / "future_decoding"

# %% Functions


def kris_version(
    session,
    include_multi_units=True,
    max_steps_to_goal=30,
    resolution=0.1,
    max_offset=6,
    state_type="place",
    min_spikes=300,
    sqrt_spikes=True,
    n_folds=5,
    normalise_X=True,
    spikes_reg_weight=0.1,
    verbose=True,
):
    # input data (see get_input_df for details)
    navigation_spikes_df = get_input_df(
        session, include_multi_units, max_steps_to_goal, resolution, max_offset, state_type, min_spikes
    )
    simple_maze = session.simple_maze()
    # prep data for decoding
    spike_counts = navigation_spikes_df.spike_count.values
    if sqrt_spikes:
        spike_counts = np.sqrt(spike_counts)
    # add current place-direction and goal nuissance regressor array
    PD_1hot = convert.place_direction2onehot(navigation_spikes_df.place_direction.values, simple_maze=simple_maze)
    # target values are the location we are at now, and that we will be at at different points in the past/future
    Ys = np.array([navigation_spikes_df.future.values, navigation_spikes_df.past.values])
    #
    #  convert to one-hot for regression
    if state_type == "place":
        Ys_1hot = np.array([[convert.place2onehot(Y[:, i], simple_maze) for i in range(Y.shape[-1])] for Y in Ys])
    elif state_type == "place_direction":
        Ys_1hot = np.array(
            [[convert.place_direction2onehot(Y[:, i], simple_maze) for i in range(Y.shape[-1])] for Y in Ys]
        )
    else:
        raise ValueError(f"Unknown state type: {state_type}. Must be 'place' or 'place_direction'.")
    if verbose:
        print(
            "amount of data for the future/past at different delays:\nfuture:",
            Ys_1hot[0].sum((-1, -2)),
            "\npast:",
            Ys_1hot[1].sum((-1, -2)),
        )

    # CRITICALLY: filter data for decision points where past and future up to max_offset are available
    decision_points = get_decision_points(simple_maze)
    at_decision_point = np.array([sa in decision_points for sa in navigation_spikes_df.place_direction.values])
    future_and_past_avail = Ys_1hot.sum(-1).mean((0, 1)) == 1
    keep_inds = np.where(future_and_past_avail & at_decision_point)[0]
    if verbose:
        print("keeping", len(keep_inds), "data points")

    X_spikes = spike_counts[keep_inds, :]  # spike counts for relevant data
    Ys_final = Ys_1hot[..., keep_inds, :].argmax(-1)  # future/past location for relevant data
    X_SA = PD_1hot[keep_inds, :]  # state-action regressors for relevant data (can use X_SA or X_SAG)
    trials = navigation_spikes_df.trial.values[keep_inds]  # trial numbers

    # run cv decoding
    # split trial into cv folds
    unique_trials = np.unique(trials)
    trial_splits = [[] for _ in range(n_folds)]
    for trial in unique_trials:
        trial_splits[int(trial) % n_folds].append(trial)

    # data indices corresponding to each fold
    trial_split_inds = [
        np.concatenate([np.where(trials == trial_id)[0] for trial_id in trial_split]) for trial_split in trial_splits
    ]
    if normalise_X:
        X_spikes, X_SA = [(X - X.mean(0)[None, :]) / (1e-10 + X.std(0)[None, :]) for X in [X_spikes, X_SA]]  # normalize

    # try to decode from either just spikes, just state-actions, or both
    possible_Xs = [
        X_spikes,
        X_SA,
        np.concatenate(
            [spikes_reg_weight * X_spikes, X_SA], axis=-1
        ),  # weight spikes to increase effective reg strength w/ more regressors
    ]  # for first one (this worked)
    num_regressors = len(possible_Xs)
    # run regression for each of these models. Total result shape is (regressions, future vs past, offset, fold)
    scores = np.zeros((num_regressors, 2, max_offset + 1, n_folds))  # array to store result
    for itype in range(2):  # decode future or past
        if verbose:
            print(itype)
        for iX, X in enumerate(possible_Xs):
            for ishift in range(max_offset + 1):
                if verbose:
                    print(ishift)
                y = Ys_final[
                    itype, ishift, :
                ]  # target is the location ishift into the future or past (depending on itype)
                for fold in range(n_folds):  # for each fold
                    if verbose:
                        print(f"folds {fold}")
                    # training and test indices
                    test, train = trial_split_inds[fold], np.concatenate(
                        [trial_split_inds[f] for f in range(n_folds) if f != fold]
                    )

                    # could do nested crossvalidation to set the regularization strength, but just doing something simple to start
                    clf = LogisticRegression(C=1e-0, max_iter=10_000)
                    clf.fit(X[train, :], y[train])  # fit the model

                    # test the model
                    scores[iX, itype, ishift, fold] = clf.score(X[test, :], y[test])
    return scores


# %%


def test(
    session,
    include_multi_units=True,
    max_steps_to_goal=30,
    resolution=0.1,
    past_offset=6,
    future_offset=6,
    state_type="place",
    min_spikes=300,
    sqrt_spikes=True,
    n_folds=5,
    normalise_X=True,
    spikes_reg_weight=0.1,
    verbose=True,
):
    """ """
    # input data (see get_input_df for details)
    input_df = get_input_df(
        session, include_multi_units, max_steps_to_goal, resolution, past_offset, future_offset, state_type, min_spikes
    )
    # filter data for times where animal is at decision point
    simple_maze = session.simple_maze()
    decision_points = get_decision_points(simple_maze)
    input_df = input_df[input_df.place_direction.isin(decision_points)]
    # get cv folds df
    folds_df = folds.get_folds_df(session, goal_stratified=True, n_folds=n_folds, return_unique_IDs=False)
    _folds = folds_df.columns.get_level_values(0).unique().values  # folds names
    # run decoding
    for _type, offsets in zip(["future", "past"], [np.arange(0, future_offset + 1), np.arange(1, past_offset + 1)]):
        for offset in offsets:
            # keep data only where future/past states are defined
            _df = input_df[input_df[_type][offset].notna()]
            for fold in _folds:
                # get training and test data
                fold_df = folds_df[fold]
                train_trials, test_trials = (
                    fold_df.train.unstack().dropna().values,
                    fold_df.test.unstack().dropna().values,
                )
                train_df, test_df = (
                    _df[_df.trial.isin(train_trials)],
                    _df[_df.trial.isin(test_trials)],
                )
                # get sets of regressors
                Xtrain_spikes, Xtest_spikes = [df.spike_count.values for df in [train_df, test_df]]
                if sqrt_spikes:
                    Xtrain_spikes, Xtest_spikes = np.sqrt(Xtrain_spikes), np.sqrt(Xtest_spikes)
                Xtrain_pd, Xtest_pd = [
                    convert.place_direction2onehot(df.place_direction.values, simple_maze=simple_maze)
                    for df in [train_df, test_df]
                ]
                Xtrain_spikes_pd, Xtest_spikes_pd = [
                    np.concatenate([spikes_reg_weight * X_spikes, X_pd], axis=-1)
                    for X_spikes, X_pd in zip([Xtrain_spikes, Xtest_spikes], [Xtrain_pd, Xtest_pd])
                ]
                for label, (X_test, X_train) in zip(
                    ["spikes", "place_direction", "spikes_place_direction"],
                    [
                        (Xtest_spikes, Xtrain_spikes),
                        (Xtest_pd, Xtrain_pd),
                        (Xtest_spikes_pd, Xtrain_spikes_pd),
                    ],
                ):
                    if normalise_X:
                        X_train, X_test = [
                            (X - X.mean(0)[None, :]) / (1e-10 + X.std(0)[None, :]) for X in [X_train, X_test]
                        ]
                    ylabel_train, ylabel_test = [df[_type][offset].values for df in [train_df, test_df]]
                    if state_type == "place":
                        y_train, y_test = [
                            convert.place2onehot(y, simple_maze=simple_maze).argmax(-1)
                            for y in [ylabel_train, ylabel_test]
                        ]
                    elif state_type == "place_direction":
                        y_train, y_test = [
                            convert.place_direction2onehot(y, simple_maze=simple_maze).argmax(-1)
                            for y in [ylabel_train, ylabel_test]
                        ]
                    else:
                        raise ValueError(f"Unknown state type: {state_type}. Must be 'place' or 'place_direction'.")
                    # fit model
                    clf = LogisticRegression(C=1e-0, max_iter=10_000)
                    clf.fit(X_train, y_train)
                    score = clf.score(X_test, y_test)
                    return score

    return


# %%
def get_input_df(
    session,
    include_multi_units=True,
    max_steps_to_goal=30,
    resolution=0.1,
    past_offset=6,
    future_offset=6,
    state_type="place",
    min_spikes=300,
):
    # load data
    navigation_df = session.navigation_df
    spike_counts_df = session.navigation_spike_counts_df.reset_index(drop=True)

    # filter clusters
    keep_clusters = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=include_multi_units,
    )
    spike_counts_df = spike_counts_df[spike_counts_df.columns[spike_counts_df.spike_count.columns.isin(keep_clusters)]]

    # downsample data
    ds_nav_df, ds_spikes_df = ds.downsample_nav_spikes_data(
        navigation_df, spike_counts_df, resolution=resolution, distance_metrics=[("steps_to_goal", "future")]
    )
    navigation_spikes_df = pd.concat([ds_nav_df, ds_spikes_df], axis=1)

    # add place_direction column
    navigation_spikes_df[("place_direction", "")] = (
        navigation_spikes_df.maze_position.simple + "_" + navigation_spikes_df.cardinal_movement_direction
    )

    # add future, past state information
    future_past_df = get_past_and_future_states(
        navigation_spikes_df, state_type=state_type, past_offset=past_offset, future_offset=future_offset
    )
    navigation_spikes_df = pd.concat([navigation_spikes_df, future_past_df], axis=1)

    # filter data
    navigation_spikes_df = filt.filter_navigation_rates_df(
        navigation_spikes_df,
        navigation_only=True,
        moving_only=False,
        exclude_time_at_goal=True,
        max_steps_to_goal=max_steps_to_goal,
    )
    # filter out clustes with min activity during navigation
    if min_spikes is not None:
        _spikes = navigation_spikes_df.spike_count
        reject_clusters = _spikes.columns[_spikes.sum(axis=0) < min_spikes].values
        navigation_spikes_df = navigation_spikes_df.drop(columns=reject_clusters, level=1, axis=1)

    return navigation_spikes_df


def get_past_and_future_states(
    navigation_spikes_df,
    state_type="place",
    past_offset=6,
    future_offset=6,
):
    """ """
    future_offsets = np.arange(1, future_offset + 1)
    past_offsets = np.arange(1, past_offset + 1)

    if state_type == "place":
        all_states = navigation_spikes_df.maze_position.simple.values
    elif state_type == "place_direction":
        all_states = navigation_spikes_df.place_direction.values
    else:
        raise ValueError(f"Unknown state type: {state_type}. Must be 'place' or 'place_direction'.")

    # only keep data from navigation
    phases = navigation_spikes_df.trial_phase.values
    all_states[phases != "navigation"] = None  # remove data not during navigation

    # also remove any data where the mouse is at the goal location
    all_trial_nums = navigation_spikes_df.trial.unique()
    for trial in all_trial_nums[~np.isnan(all_trial_nums)]:
        trial_inds = np.where(navigation_spikes_df.trial == trial)[0]
        goal_inds = trial_inds[
            (
                navigation_spikes_df.maze_position.simple.values[trial_inds]
                == navigation_spikes_df.goal.values[trial_inds[0]]
            )
        ]
        all_states[goal_inds] = None

    # find the df indices corresponding to every time the mouse moves to a different state
    boundaries = np.concatenate(
        [np.zeros(1).astype(int), np.where(all_states[1:] != all_states[:-1])[0] + 1]
    )  # first index in each state

    # also instantiate an array in which we will store current, future, and past states
    # this will have shape (2, max_offset+1, num_datapoints)
    # it will store the state (tower or bridge) that the mouse will be in after (0,1,...,num_offset) actions, and (0,1, ..., num_offset) actions before
    offset_array = np.array(
        [
            np.vstack(
                [copy.deepcopy(all_states)]
                + [np.array([None for _ in range(len(all_states))]) for _ in range((past_offset + future_offset))]
            )
            for _ in range(2)
        ]
    )  # future and past

    # populate this array
    for itype, (type_, offsets) in enumerate(
        zip(["future", "past"], [future_offsets, past_offsets])
    ):  # first consider future, then past states
        sign = +1 if type_ == "future" else -1  # the sign of our offset
        # boundaries between states going either forwards (to compute future states) or backwards (to compute past)
        dir_boundaries = (
            boundaries if (type_ == "future") else np.flip(boundaries) - 1
        )  # go either forwards or backwards
        for i_off, offset in enumerate(offsets):  # for every offset we are interested in
            ref = offset_array[
                itype, offset - 1, :
            ]  # what is the state at the previous offset (e.g. what is the previous state if we now want the state two before)
            next_ = offset_array[
                itype, offset, :
            ]  # the array we're trying to populate corresponding to the current offset
            for i_b, b in enumerate(dir_boundaries[:-1]):  # for every state transition
                inds = np.arange(b, dir_boundaries[i_b + 1], sign)  # indices of the df  where I was in that state
                cur_state = ref[inds[0]]  # where will I be in 'offset-1' steps
                next_state = ref[inds[-1] + sign]  # where will I be in 'offset' steps

                if None in [cur_state, next_state]:  # if we're at a trial boundary
                    next_[inds] = None  # don't have a next state
                else:
                    assert cur_state != next_state  # make sure that we have actually moved
                    next_[inds] = next_state  # store the next state
    output_df = pd.DataFrame(index=navigation_spikes_df.index)  # create a new dataframe to store the results
    for offset in [0] + list(future_offsets):  # make this data part of the big dataframe
        output_df[("future", offset)] = offset_array[0, offset, :]
    for offset in [0] + list(past_offsets):  # make this data part of the big dataframe
        output_df[("past", offset)] = offset_array[1, offset, :]
    return output_df


def get_decision_points(simple_maze):
    """
    Computes and returns the set of decision point identifiers in a given maze.

    A decision point is defined as a position (or an intermediate bridge) in the maze from which
    a move in a specific cardinal direction (North, South, East, West) leads to a node having three
    or more neighbors. The identifiers are generated by concatenating a label (from the maze's coordinate-label
    mapping) with an underscore and the corresponding direction.

    Returns:
        A set of strings, each representing a decision point in the maze. Each string is formed by combining
        the label (either of a node or an intermediate position along an edge) with the direction taken from that node.
    """
    coord2label = coord2label = mr.get_maze_coord2label(simple_maze)
    deltas = {(1, 0): "E", (-1, 0): "W", (0, 1): "N", (0, -1): "S"}  # mapping from action vector to direction
    decision_points = set()
    for node1 in simple_maze.nodes:
        for node2 in simple_maze.neighbors(node1):  #
            if len(list(simple_maze.neighbors(node2))) >= 3:

                dir_ = deltas[tuple(np.array(node2) - np.array(node1))]
                decision_points.add(
                    coord2label[node1] + "_" + dir_
                )  # going in this direction from node1 yields a decision point
                try:
                    decision_points.add(coord2label[(node1, node2)] + "_" + dir_)
                except:
                    decision_points.add(coord2label[(node2, node1)] + "_" + dir_)
    return decision_points
