"""
New library for goal-coding encoding analyses. Test if goal-distance explains unique variance over place-direction and distance
in the neural population.
@peterdoohan
"""

# %% Imports
from GridMaze.analysis.place_direction import bases as pdb
from GridMaze.analysis.distance_to_goal import bases as db
from GridMaze.analysis.distance_to_goal import decoding_utils as du

# %% Global Variables

# %% Functions


def test(session, pd_bases_kwargs={"n_bases": 8, "dim_red": "nmf"}, dtg_bases_kwargs={"n_bases": 4, "basis": "gamma"}):
    """ """
    # get place-direction bases
    pd_bases = pdb.get_place_direction_bases(pdb.get_heldout_sessions(session), **pd_bases_kwargs)
    # get distance to goal bases
    dist_bases = db.distance_basis_generator(**dtg_bases_kwargs)
    # get downsampled input data
    input_data = du.get_place_decoding_input_data(session, resolution=0.5, include_multi_units=False)

    return


def get_input_data():

    return
