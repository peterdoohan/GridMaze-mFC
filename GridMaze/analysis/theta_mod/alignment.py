"""
Neural activity unfolds over a trial as a trajectory though a high-dimensional space (with some smoothing).
Neighboring timepoints on this trajectory define the current direction of movement (vector) through this trajectory.
Points between different phases of a theta oscillation can also define a vector in this high-d space. If representations
move from the present to the future (including up to a goal a long way-away) over theta cycles, vectors between theta phases and
vectors between neural timepoints should be aligned (non-orthogoal as predicted by chance).
@peterdoohan
"""

# %% Imports
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.theta_mod import utils as tmu

# %% Global Variables

# %% Functions


def test():
    return
