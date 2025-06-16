"""
Library for visualising population tuning aligned to egocentric actions
"""

# %% Imports
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt

from GridMaze.analysis.cluster_tuning import actions as act
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc

# %% Global Variables

FRAME_RATE = 60  # Hz

# %% Functions


def test(session, forced_only=True, window=(-3, 3), step_size=0.2):
    """
    note only loads actions during navigation

    if step_size == False, aligned rates returned at frame rate
    """
    navigation_rates_df = session.get_navigation_activity_df(type="rates", cluster_kwargs={"single_units": True})
    action_tuning_df = act._get_basic_action_tuning(
        navigation_rates_df, actions=["turn_left", "go_forward", "turn_right", "go_back"], window=window
    )
    if step_size:  # average rates within each downsampled step
        aligned_rates = action_tuning_df.action_aligned_rates  # sampled at frame rate
        combine_frames = int(FRAME_RATE * step_size)
        old_times = aligned_rates.columns.values.astype(float)
        new_times = np.arange(old_times[0], old_times[-1] + step_size, step_size)
        rates = aligned_rates.values
        new_rates = np.zeros((rates.shape[0], len(new_times)))
        for i, time in enumerate(new_times):
            start = int((time - old_times[0]) * FRAME_RATE)
            end = start + combine_frames
            new_rates[:, i] = np.mean(rates[:, start:end], axis=1)
        new_aligned_rates = pd.DataFrame(
            new_rates,
            index=action_tuning_df.index,
            columns=pd.MultiIndex.from_product([["action_aligned_rates"], new_times]),
        )
        action_tuning_df = pd.concat(
            [action_tuning_df.drop(columns=["action_aligned_rates"], level=0), new_aligned_rates], axis=1
        )
    if forced_only:
        action_tuning_df = action_tuning_df[action_tuning_df.choice_degree.gt(2)]
    cluster_unique_IDs = action_tuning_df.cluster_unique_ID.unique()
    for cluster in cluster_unique_IDs:
        cluster_df = action_tuning_df[action_tuning_df.cluster_unique_ID == cluster]
        df = _format_cluster_df(cluster_df)
        model = smf.ols("firing_rate ~ C(action) * C(timepoint)", data=df).fit()
        anova_results = sm.stats.anova_lm(model, typ=2)
        action_F = anova_results.loc["C(action)", "F"]
        action_p = anova_results.loc["C(action)", "PR(>F)"]
        interaction_F = anova_results.loc["C(action):C(timepoint)", "F"]
        interaction_p = anova_results.loc["C(action):C(timepoint)", "PR(>F)"]
        f, ax = plt.subplots(1, 1, figsize=(6, 4), clear=True)
        Clust = gc.get_cluster(cluster)
        Clust.plot_tuning("actions", feature_kwargs={"concise": True}, ax=ax)
        ax.set_title(f"F={interaction_F:.2f}, p={interaction_p:.3f}\nF={action_F:.2f}, p={action_p:.3f}")
    return


def _format_cluster_df(cluster_df):
    """
    reformat cluster df ready for statsmodels
    """
    df = cluster_df.set_index(["basic_action", "action_number"]).action_aligned_rates.stack().reset_index()
    df.columns = ["action", "action_number", "timepoint", "firing_rate"]
    return df
