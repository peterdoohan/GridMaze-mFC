"""
Same variance explained framework now with neural distance tuning
"""

# %% Imports
import numpy as np
import pandas as pd
from GridMaze.analysis.core import convert
from GridMaze.analysis.core import filter as filt

from GridMaze.analysis.efficient_coding import place_direction as ec

from matplotlib import pyplot as plt

# %% Global Variables
from GridMaze.analysis.embedding_model.run_experiment import DEFAULT_INPUT_KWARGS

# %%


def run_main_analysis(X, ve_method="svd", demean=True, norm_length=True, plot=True):
    """"""
    if ve_method == "pca":
        ve_fn = ec.get_pca_variance_explained
    elif ve_method == "svd":
        ve_fn = ec.get_svd_variance_explained
    else:
        raise NotImplementedError(f"ve_method {ve_method} not recognised")
    n_components = X[0]["neurons"]["train"].shape[-1]
    results = np.zeros((len(X), 4, n_components + 1))  # [n_splits, 4, n_components]
    for i, data in enumerate(X):
        # neural data
        train_neurons, test_neurons = data["neurons"]["train"], data["neurons"]["test"]
        # fill nans in neural data with mean (unvistied place-directions)
        train_neurons = train_neurons.apply(lambda row: row.fillna(row.mean()), axis=1).values
        test_neurons = test_neurons.apply(lambda row: row.fillna(row.mean()), axis=1).values
        # behaviour data
        train_behaviour, test_behaviour = data["behaviour"]["train"].values, data["behaviour"]["test"].values
        if demean:
            train_neurons, test_neurons, train_behaviour, test_behaviour = [
                arr - arr.mean(-1, keepdims=True)
                for arr in [train_neurons, test_neurons, train_behaviour, test_behaviour]
            ]
        if norm_length:
            train_neurons, test_neurons, train_behaviour, test_behaviour = [
                arr / np.linalg.norm(arr, axis=1, keepdims=True)
                for arr in [train_neurons, test_neurons, train_behaviour, test_behaviour]
            ]

        beb = ve_fn(train_behaviour, test_behaviour)
        nen = ve_fn(train_neurons, test_neurons)
        ben = ve_fn(train_behaviour, test_neurons)
        neb = ve_fn(train_neurons, test_behaviour)
        results[i] = np.array([beb, nen, ben, neb])
    # plotting (make pretty later)
    if plot:
        f, axes = plt.subplots(1, 2, figsize=(5, 3), clear=True, sharex=True, sharey=True)
        for ax in axes.flatten():
            ax.spines[["top", "right"]].set_visible(False)
            ax.plot([0, n_components], [0, 1], color="black", ls="--")
        # behaviour explains plot
        beb_mean = results[:, 0].mean(axis=0)
        beb_sem = results[:, 0].std(axis=0) / np.sqrt(results.shape[0])
        axes[0].plot(beb_mean, label="Behaviour", color="blue")
        axes[0].fill_between(range(len(beb_mean)), beb_mean - beb_sem, beb_mean + beb_sem, color="blue", alpha=0.3)
        neb_mean = results[:, 3].mean(axis=0)
        neb_sem = results[:, 3].std(axis=0) / np.sqrt(results.shape[0])
        axes[0].plot(neb_mean, label="Neurons", color="red")
        axes[0].fill_between(range(len(neb_mean)), neb_mean - neb_sem, neb_mean + neb_sem, color="red", alpha=0.3)
        axes[0].set_xlabel("Number of components")
        axes[0].set_ylabel("Cum. var exp")
        axes[0].set_title("Behaviour explained by")
        axes[0].legend(fontsize="xx-small")
        # neurons explains plot
        nen_mean = results[:, 1].mean(axis=0)
        nen_sem = results[:, 1].std(axis=0) / np.sqrt(results.shape[0])
        axes[1].plot(nen_mean, label="Neurons", color="red")
        axes[1].fill_between(range(len(nen_mean)), nen_mean - nen_sem, nen_mean + nen_sem, color="red", alpha=0.3)
        ben_mean = results[:, 2].mean(axis=0)
        ben_sem = results[:, 2].std(axis=0) / np.sqrt(results.shape[0])
        axes[1].plot(ben_mean, label="Behaviour", color="blue")
        axes[1].fill_between(range(len(ben_mean)), ben_mean - ben_sem, ben_mean + ben_sem, color="blue", alpha=0.3)
        axes[1].legend(fontsize="xx-small")
        axes[1].set_xlabel("Number of components")
        axes[1].set_title("Neurons explained by")
        f.tight_layout()
        f.subplots_adjust(wspace=0.8)
        # axes[0].set_xlim([0, 20])
        # axes[1].set_xlim([0, 20])
    return results


# %% Input data functions


def get_joint_neural_behaviour_distance_to_goal_dfs(sessions, n_splits=5, test_size=0.2):
    """ """
    split_sessions = ec._get_session_splits(sessions, n_splits, test_size)
    X = []
    for train_sessions, test_session in split_sessions:
        X.append(
            {
                "neurons": {
                    "train": _get_neural_tuning(train_sessions),  # df [n_neurons, n_place_directions]
                    "test": _get_neural_tuning(test_session),
                },
                "behaviour": {
                    "train": _get_behavioural_sequences(train_sessions),  # df [n_trials, n_place_directions]
                    "test": _get_behavioural_sequences(test_session),
                },
            }
        )
    return X


def _get_behavioural_sequences(sessions, input_kwargs=DEFAULT_INPUT_KWARGS, clipped=True):
    """ """
    distance_metric = input_kwargs["distance_metrics"]
    bin_col = (distance_metric[0], distance_metric[1] + "_binned")
    distance_bins = convert._get_distance_bins(
        input_kwargs["distance_bin_method"],
        input_kwargs["n_distance_bins"],
        input_kwargs["distance_metrics"],
        max_distance=1.8,
    )
    bin2indx = {bin: i for i, bin in enumerate(distance_bins)}
    trial_sequences = []
    for session in sessions:
        # load
        navigation_df = session.navigation_df
        # filter
        navigation_df = filt.filter_navigation_rates_df(
            navigation_df,
            navigation_only=input_kwargs["navigation_only"],
            moving_only=input_kwargs["moving_only"],
            exclude_time_at_goal=False,
            max_steps_to_goal=input_kwargs["max_steps_to_goal"],
        )
        # add distance bins
        navigation_df[bin_col] = pd.cut(navigation_df[distance_metric], bins=distance_bins, include_lowest=True)
        navigation_df = navigation_df[navigation_df[bin_col].notna()]
        # get sequences
        trials = navigation_df.trial.unique()
        session_sequences = np.zeros((len(trials), len(bin2indx)))
        for i, trial in enumerate(trials):
            trial_df = navigation_df[navigation_df.trial == trial]
            distance_sequence = trial_df[bin_col].map(bin2indx).values.astype(int)
            for j in distance_sequence:
                session_sequences[i, j] += 1
        trial_sequences.append(session_sequences)
    trial_sequences = np.vstack(trial_sequences)
    if clipped:
        trial_sequences = np.clip(trial_sequences, 0, 1)
    return pd.DataFrame(data=trial_sequences, columns=distance_bins)


def _get_neural_tuning(sessions, input_kwargs=DEFAULT_INPUT_KWARGS, min_firing_rate=1):
    """ """
    # get distance bins
    distance_metric = input_kwargs["distance_metrics"]
    bin_col = (distance_metric[0], distance_metric[1] + "_binned")
    distance_bins = convert._get_distance_bins(
        input_kwargs["distance_bin_method"],
        input_kwargs["n_distance_bins"],
        input_kwargs["distance_metrics"],
        max_distance=1.8,
    )
    all_distance_tuning = []
    for session in sessions:
        # load
        navigation_rates_df = session.get_navigation_activity_df(
            type="rates", cluster_kwargs={"single_units": True, "multi_units": False}
        )
        # filter
        navigation_rates_df = filt.filter_navigation_rates_df(
            navigation_rates_df,
            navigation_only=input_kwargs["navigation_only"],
            moving_only=input_kwargs["moving_only"],
            exclude_time_at_goal=False,
            max_steps_to_goal=input_kwargs["max_steps_to_goal"],
        )
        if min_firing_rate:  # remove non-navigation tuned neurons
            rates = navigation_rates_df.xs("firing_rate", level=0, axis=1, drop_level=False)
            keep_clusters = rates.mean(axis=0).gt(min_firing_rate).index
            navigation_rates_df = pd.concat(
                [
                    navigation_rates_df.drop("firing_rate", level=0, axis=1),
                    rates[keep_clusters],
                ],
                axis=1,
            )
        # add distance bins
        navigation_rates_df[bin_col] = pd.cut(
            navigation_rates_df[distance_metric], bins=distance_bins, include_lowest=True
        )
        # average over distance bins to get tuning
        distance_tuning = (
            navigation_rates_df.groupby([bin_col], observed=True).firing_rate.mean().firing_rate.T
        )  # [n_clusters, n_bins]
        all_distance_tuning.append(distance_tuning)
    return pd.concat(all_distance_tuning, axis=0)  # [n_clusters (total), n_bins]
