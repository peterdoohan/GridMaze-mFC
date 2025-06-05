""" """

# %% Imports


# %% Global Variables


# %% Functions


def _get_behavioural_sequences(
    sessions,
    max_steps_from_goal=DATA_FILTER_KWARGS["max_steps_from_goal"],
    synthetic=False,
):
    """
    Behavioural sequences are binary vectors of place-direction pairs visited in each trial
    if synthetic in ["optimal", "random_diffusion", "forward_diffusion"] then the behaviour is generated
    synthetically (for control analyses)
    """
    trials_sequences = []
    place_direction2idx = {_pd: i for i, _pd in enumerate(mr.get_maze_place_direction_pairs(sessions[0].simple_maze()))}
    for session in sessions:
        # load
        if not synthetic:
            trajectories_df = session.trajectory_decisions_df
        else:
            trajectories_df = cb.get_synthetic_behaviour(session, policy=synthetic)
        # filter
        trajectories_df = trajectories_df[(trajectories_df.trial_phase == "navigation")]
        trajectories_df = trajectories_df[(trajectories_df.maze_position.notnull())]
        trajectories_df = trajectories_df[(trajectories_df.action.notnull())]
        if max_steps_from_goal is not None:
            trajectories_df = trajectories_df[(trajectories_df.steps_to_goal.lt(max_steps_from_goal))]
        # loop over trials to construct sequence vectors (binary in each place-direction, 1 if visited)
        trials = trajectories_df.trial.unique()
        session_sequences = np.zeros((len(trials), len(place_direction2idx)), dtype=int)
        for i, trial in enumerate(trials):
            trial_df = trajectories_df[trajectories_df.trial == trial]
            place_direction_sequence = list(zip(trial_df.maze_position, trial_df.action))
            for j in place_direction_sequence:
                session_sequences[i, place_direction2idx[j]] += 1
        trials_sequences.append(session_sequences)
    behaviour_df = pd.DataFrame(
        data=np.vstack(trials_sequences),
        columns=pd.MultiIndex.from_tuples(place_direction2idx.keys(), names=["maze_position", "direction"]),
    )
    return behaviour_df.sort_index(axis=1)


def _get_behavioural_sequences_ALT(sessions, filter_kwargs=DATA_FILTER_KWARGS):
    """
    Behavioural sequences defined over frames not node-egde-node transitions as in _get_behavioural_sequences

    Slighly slower var exp, although very slighly
    """
    trial_sequences = []
    place_direction2idx = {_pd: i for i, _pd in enumerate(mr.get_maze_place_direction_pairs(sessions[0].simple_maze()))}
    for session in sessions:
        navigation_df = session.navigation_df
        navigation_df = filt.filter_navigation_rates_df(
            navigation_df,
            navigation_only=filter_kwargs["navigation_only"],
            moving_only=filter_kwargs["moving_only"],
            exclude_time_at_goal=filter_kwargs["exclude_time_at_goal"],
            max_steps_to_goal=filter_kwargs["max_steps_from_goal"],
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
