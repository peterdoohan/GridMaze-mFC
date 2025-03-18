"""
library for plotting lfp aligned to trial events
"""

# %% Imports
from GridMaze.analysis.core import get_sessions as gs

# %% Global Variables


# %% Functions


def get_test_lfp(session):
    session = gs.get_maze_sessions(
        subject_IDs=["m2"],
        maze_names=["maze_2"],
        days_on_maze=[11],
        with_data=[
            "trials_df",
            "lfp_signal",
            "lfp_times",
            "lfp_metrics",
            "cluster_metrics",
        ],
    )
    # load_data
    lfp_metrics = session.lfp_metrics
    lfp_signal = session.lfp_signal
    lfp_times = session.lfp_times
    # remove bad channels
    good_channel_mask = lfp_metrics.contact.qc == "good"
    lfp_metrics = lfp_metrics[good_channel_mask].reset_index(drop=True)
    lfp_signal = lfp_signal[:, good_channel_mask]
    # common average reference
    lfp_signal = lfp_signal - lfp_signal.mean(axis=1)[:, None]
    # choose a channel with lots of single units as an example channel
    cluster_metrics = session.cluster_metrics
    best_channels = cluster_metrics[cluster_metrics.single_unit].contact.id.mode().values
    for c in best_channels:
        contact_info = lfp_metrics[lfp_metrics.contact.id == c]
        if not contact_info.empty:
            break
    channel_ind = contact_info.index[0]
    lfp_signal = session.lfp_signal[:, channel_ind]
    # return with times
    lfp_times = session.lfp_times
    return lfp_signal, lfp_times
