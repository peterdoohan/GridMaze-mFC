"""
Library for distance to goal rep, theta mod decoding.
"""

# %% Imports
import numpy as np
import pandas as pd

from GridMaze.analysis.core import convert
from GridMaze.analysis.distance_to_goal import logreg_decoder as ld
from GridMaze.analysis.distance_to_goal import extract_lfp_pahse as elp

# %% Global Variables


# %% Functions


def get_input_data(session, include_multiunits=True, n_lfp_phase_bins=12):
    """ """
    # load data
    navigation_df = session.navigation_df
    spike_counts_df = session.navigation_spike_counts_df.reset_index(drop=True)
    cluster_metrics = session.cluster_metrics
    session_info = session.session_info
    # filter for single units
    if not include_multiunits:
        single_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID
        single_units = convert.cluster_IDs2scluster_unique_IDs(session_info, single_units)
        spike_counts_df = spike_counts_df[[("spike_count", u) for u in single_units]]

    times = navigation_df.time.values
    for osc in ["theta", "4Hz"]:
        phase_bins = elp.get_nearest_osc_phase(
            session,
            times,
            signal_type="LFP",
            band=osc,
            return_binned=True,
            n_bins=n_lfp_phase_bins,
        )

    return
