"""
Library for distance to goal tuning analyses: curve fits, headmaps etc.
"""

# %% Imports

from GridMaze.analysis.core import get_sessions as gs

# %% Global Variables


# %% Functions


def test(late_sessions=True):
    """
    Compare how well different distribution shapes (gamma, gaussian, polynomial) fit empirical
    distance to goal tuning curves.
    """
    days_on_maze = "late" if late_sessions else "all"
    sessions = gs.get_maze_sessions(
        subject_IDs="all", maze_names="all", days_on_maze=days_on_maze, with_data=["cluster_distna"]
    )
    return
