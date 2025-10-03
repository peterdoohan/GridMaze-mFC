"""
Test if there is a difference in theta-modulation as across population of cells tuned to
distance-to-goal and place-direction.
@peterdoohan
"""

# %% Imports
import numpy as np
import pandas as pd

from GridMaze.analysis.core import convert
from GridMaze.analysis.core import filter as filt

# %% Global Variables
from GridMaze.paths import ANALYSIS_INFO_PATH

VARIANCE_EXPLAINED_DF = pd.read_parquet(ANALYSIS_INFO_PATH / "cluster_unique_variance_explained.parquet")

# %% Functions


def test(session):
    """ """

    return


def get_session_theta_mod(
    session, navigation_only=True, include_multi_unit=False, moving_only=True, max_steps_to_goal=30, min_spikes=300
):
    """ """
    # load data
    session_info = session.session_info
    cluster_metrics = session.cluster_metrics
    navigation_df = session.navigation_df.copy()
    theta_spike_counts_df = session.navigation_theta_spike_counts_df.reset_index(drop=True)
    # filter for single units
    if not include_multi_unit:
        keep_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID
    else:
        keep_units = cluster_metrics[cluster_metrics.single_unit | cluster_metrics.multi_unit].cluster_ID
    keep_units = convert.cluster_IDs2scluster_unique_IDs(session_info, keep_units)
    theta_spike_counts_df = theta_spike_counts_df[
        theta_spike_counts_df.columns[[c in keep_units for c in theta_spike_counts_df.columns.get_level_values(1)]]
    ]
    # combine nav and spikes
    navigation_df.columns = pd.MultiIndex.from_tuples([(*c, "") for c in navigation_df.columns])
    nav_spike_counts_df = pd.concat([navigation_df, theta_spike_counts_df], axis=1)
    # filter for moving, navigation, on task etc.
    nav_spike_counts_df = filt.filter_navigation_rates_df(
        nav_spike_counts_df, navigation_only, moving_only, max_steps_to_goal=max_steps_to_goal
    )
    cluster_phase_spike_counts = nav_spike_counts_df.spike_count.sum().unstack()
    # filter for cluster with few spikes in filtered data (eg, non-navigation tuned)
    cluster_phase_spike_counts = cluster_phase_spike_counts[cluster_phase_spike_counts.sum(axis=1) > min_spikes]
    # normalise to % average spikes
    df = cluster_phase_spike_counts.div(cluster_phase_spike_counts.mean(axis=1), axis=0)
    # fit a sine wave to each cluster
    df.columns = pd.MultiIndex.from_product([["prop_spikes"], df.columns])
    # add other info
    df[("subject_ID", "")] = session.subject_ID
    df[("maze_name", "")] = session.maze_name
    df[("day_on_maze", "")] = session.day_on_maze
    df[("tissue_sample", "")] = session.tissue_sample
    df[("probe_depth", "")] = session.probe_depth
    return df


def _get_theta_mod_metrics():
    """
    Collects theta modulation metrics:
    - depth of modulation (max - min)
    - pref phase (phase bin with max spikes)
    - rayleigh test for non-uniformity of circular data (p-value)
    - sine fit
        - amplitude
        - phase offset
        - r2
        - phase max
    """
    return


def fit_phase_sine(theta_mod_tuning):
    # Convert to numpy arrays
    x = np.asarray(theta_mod_tuning.index, dtype=float)
    y = np.asarray(theta_mod_tuning.values, dtype=float)

    # Linear model: y ≈ B*sin(x) + C*cos(x)
    X = np.column_stack([np.sin(x), np.cos(x)])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    B, C = beta

    # Recover amplitude and phase offset
    amplitude = np.hypot(B, C)
    phase_offset = np.arctan2(C, B)

    # Predicted values and R²
    y_hat = amplitude * np.sin(x + phase_offset)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return amplitude, phase_offset, r2
