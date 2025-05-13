"""
New library for goal-coding encoding analyses. Test if goal-distance explains unique variance over place-direction and distance
in the neural population.
@peterdoohan
"""

# %% Imports
import pandas as pd

from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import folds

from GridMaze.analysis.place_direction import bases as pdb
from GridMaze.analysis.distance_to_goal import bases as db

# %% Global Variables

# %% Functions


def test(
    session,
    resolution=0.5,
    distance_metrics=("steps_to_goal", "future"),
    goal_stratified_validation=True,
    n_test_trials=None,
    trial_phases=["navigation"],
    max_steps_to_goal=30,
    pd_bases_kwargs={"n_bases": 8, "dim_red": "nmf"},
    dtg_bases_kwargs={"n_bases": 4, "basis": "gamma"},
    verbose=True,
):
    """ """
    if verbose:
        print(f"Loading basis functions")
    # get place-direction bases
    pd_bases = pdb.get_place_direction_bases(pdb.get_heldout_sessions(session), **pd_bases_kwargs)
    # get distance to goal bases
    dist_bases = db.distance_basis_generator(
        **dtg_bases_kwargs, btype=distance_metrics[0].split("_")[0], max_steps=max_steps_to_goal
    )
    if verbose:
        print(f"Loading input data")
    # get downsampled input data
    input_data = get_input_data(session, resolution, trial_phases=trial_phases, distance_metrics=distance_metrics)
    # get folds df
    folds_df = folds.get_folds_df(
        session, goal_stratified=goal_stratified_validation, return_unique_IDs=True, n_test_trials=n_test_trials
    )
    _folds = folds_df.columns.get_level_values(0).unique()
    for fold in _folds:
        fold_df = folds_df[fold]

    return pd_bases, dist_bases, input_data, folds_df


def get_input_data(
    session,
    resolution=0.5,
    distance_metrics=("steps_to_goal", "future"),
    trial_phases=["navigation"],
    max_steps_to_goal=30,
):
    """ """
    # load data
    navigation_df = session.navigation_df
    spike_counts_df = session.navigation_spike_counts_df.reset_index(drop=True)
    # filter for single units
    keep_clusters = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=False,
    )
    spike_counts_df = spike_counts_df[spike_counts_df.columns[spike_counts_df.spike_count.columns.isin(keep_clusters)]]
    # downsample data
    navigation_df, spike_counts_df = ds.downsample_nav_spikes_data(
        navigation_df, spike_counts_df, resolution=resolution, distance_metrics=distance_metrics
    )
    # combine
    nav_rates_df = pd.concat([navigation_df, spike_counts_df], axis=1)
    # filter for trial phases
    nav_rates_df = nav_rates_df[nav_rates_df.trial_phase.isin(trial_phases)]
    # filter for max steps to goal
    nav_rates_df = nav_rates_df[nav_rates_df.steps_to_goal.future.le(max_steps_to_goal)]
    return nav_rates_df
