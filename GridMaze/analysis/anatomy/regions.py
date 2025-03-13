"""
Library for looking at the distribution of brain regions sampled in this experiment
@peterdoohan
"""

# %% Imports

from GridMaze.analysis.core import get_sessions as gs

# %% Global Variables


# %% Functions


def test():
    """ """
    session = gs.get_maze_sessions(
        subject_IDs=["m2"], maze_names=["maze_1"], days_on_maze=[12], with_data=["cluster_metrics"], must_have_data=True
    )

    return
