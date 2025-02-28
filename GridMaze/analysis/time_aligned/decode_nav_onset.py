"""
Can we decode the true (latent) onset of navigation from neural data?
"""

# %% Imports
import numpy as np
import pandas as pd
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import filter as filt

from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

# %% Globs
FRAME_RATE = 60


# %% Functions
def test(session):
    """ """
    input_data = get_input_data(session, n_folds=20, include_speed=True)
    decode_trial_phase(input_data)
    return


def decode_trial_phase(
    input_data,
):
    """"""
    for fold in input_data[:3]:
        X_train, y_train, X_test, y_test, test_meta_data = fold.values()
        # fit model
        model = LogisticRegression(penalty="l2", class_weight="balanced", C=0.01, max_iter=1_000)
        model.fit(X_train, y_train)
        classes = model.classes_
        # predict
        y_pred_prob = model.predict_proba(X_test)
        ITI_prob = y_pred_prob[:, classes == "ITI"]
        navigation_prob = y_pred_prob[:, classes == "navigation"]
        plot_decoding_probs(test_meta_data, y_test, ITI_prob, navigation_prob)
    return


def test_optimal_alpha(session, alphas=[0.1, 1, 10]):
    """ """
    input_data = get_input_data(session, n_folds=10, include_speed=True)
    test_perf, train_perf = np.zeros((len(alphas), len(input_data))), np.zeros((len(alphas), len(input_data)))
    for i, alpha in enumerate(alphas):
        print(alpha)
        for j, fold in enumerate(input_data):
            X_train, y_train, X_test, y_test, test_meta_data = fold.values()
            # fit model
            model = LogisticRegression(penalty="l2", class_weight="balanced", C=alpha, max_iter=1_000)
            model.fit(X_train, y_train)
            # test acc
            y_pred = model.predict(X_test)
            ITI_acc = (y_pred[y_test == "ITI"] == "ITI").mean()
            nav_acc = (y_pred[y_test == "navigation"] == "navigation").mean()
            test_perf[i, j] = np.mean([ITI_acc, nav_acc])
            # train acc
            y_pred = model.predict(X_train)
            ITI_acc = (y_pred[y_train == "ITI"] == "ITI").mean()
            nav_acc = (y_pred[y_train == "navigation"] == "navigation").mean()
            train_perf[i, j] = np.mean([ITI_acc, nav_acc])
    # plot
    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.errorbar(alphas, test_perf.mean(axis=1), yerr=test_perf.std(axis=1), label="test")
    ax.errorbar(alphas, train_perf.mean(axis=1), yerr=train_perf.std(axis=1), label="train")
    ax.set_xscale("log")
    ax.set_xticklabels(alphas)
    ax.set_xlabel("alpha")
    ax.legend()
    return test_perf, train_perf


def plot_decoding_probs(test_meta_data, y_test, ITI_prob, navigation_prob, window=(-6, 40)):
    """"""
    df = pd.DataFrame(index=test_meta_data.index)
    df["time"] = test_meta_data.time
    df["trial"] = test_meta_data.trial
    df["true_trial_phase"] = y_test
    df["prob_ITI"] = ITI_prob
    df["prob_navigation"] = navigation_prob
    test_trials = df.trial.dropna().unique()
    f, axes = plt.subplots(len(test_trials), 1, figsize=(10, 1.5 * len(test_trials)), sharex=True, sharey=True)
    if len(test_trials) == 1:
        axes = [axes]
    for trial, ax in zip(test_trials, axes):
        ax.spines[["top", "right"]].set_visible(False)
        trial_df = df[df.trial == trial]
        cue_time = trial_df[trial_df.true_trial_phase == "navigation"].time.values[0]
        mask = (trial_df.time >= cue_time + window[0]) & (trial_df.time <= cue_time + window[1])
        true = trial_df.true_trial_phase.map({"ITI": 0, "navigation": 1}).values[mask]
        ITI_prob = trial_df.prob_ITI.values[mask]
        navigation_prob = trial_df.prob_navigation.values[mask]
        ax.plot(true, label="true_state", color="black")
        ax.plot(ITI_prob, label="ITI_prob", color="blue")
        ax.plot(navigation_prob, label="navigation_prob", color="red")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["ITI", "navigation"])
        ax.set_title(f"Trial {int(trial)}", fontsize=12, loc="left")
    # ax.legend()
    f.tight_layout()
    return


def get_input_data(
    session,
    resolution=0.2,
    n_folds=10,
    ITI_padding=0.5,
    navigation_pad_start=1,
    navigation_pad_end=1,
    navigation_moving_only=False,
    include_speed=False,
):
    """ """
    navigation_spikes_df = session.get_navigation_activity_df(
        type="spikes", cluster_kwargs={"single_units": True, "multi_units": True}
    )
    # update trial and trial phase definitions
    navigation_spikes_df = update_trial_definitions(navigation_spikes_df)
    # could later downsample navigation df to lower resolution
    navigation_spikes_df = add_trial_phase_timers(
        navigation_spikes_df
    )  # get  times within trial phases for later filtering
    if resolution:
        navigation_spikes_df = filt.downsample_navigation_activity_df(navigation_spikes_df, window_length=resolution)
    navigation_spikes_df = navigation_spikes_df[navigation_spikes_df.trial_phase.isin(["navigation", "ITI"])]
    trials = navigation_spikes_df.trial.dropna().unique()
    validation_folds_df = filt.get_trial_validation_folds_df(trials, splits=n_folds)
    folds = validation_folds_df.columns.get_level_values(0).unique()
    decoding_input = []
    # for fold in folds:
    # fold_df = validation_folds_df[fold]
    # test_trials = fold_df.test.dropna().values.astype(int)
    # train_trials = fold_df.train.dropna().values.astype(int)
    for trial in trials:
        test_trials = [trial]
        train_trials = trials[trials != trial]
        # remove time around erc and cue from the training data (see _padding inputs)
        training_data_df = navigation_spikes_df[navigation_spikes_df.trial.isin(train_trials)]
        navigation_mask = (
            (training_data_df.trial_phase == "navigation")
            & (training_data_df.trial_phase_time.from_start.gt(navigation_pad_start))
            & (training_data_df.trial_phase_time.to_end.gt(navigation_pad_end))
        )
        ITI_mask = (
            (training_data_df.trial_phase == "ITI")
            & (training_data_df.trial_phase_time.from_start.gt(ITI_padding))
            & (training_data_df.trial_phase_time.to_end.gt(ITI_padding))
        )
        training_data_df = training_data_df.loc[ITI_mask | navigation_mask]
        if navigation_moving_only:
            nav_not_moving = training_data_df[
                (training_data_df.trial_phase == "navigation") & (~training_data_df.moving)
            ].index
            training_data_df = training_data_df.drop(nav_not_moving)
        X_train = training_data_df.spike_count.values  # n_samples, n_neurons
        if include_speed:
            speed = training_data_df.speed.values  # (n_samples, )
            X_train = np.concatenate([X_train, speed[:, None]], axis=1)
        y_train = training_data_df.trial_phase.values
        # get training data with details about trial (include all times)
        test_data_df = navigation_spikes_df[navigation_spikes_df.trial.isin(test_trials)]
        meta_data = test_data_df[["time", "trial"]].reset_index(drop=True)
        X_test = test_data_df.spike_count.values
        if include_speed:
            speed = test_data_df.speed.values
            X_test = np.concatenate([X_test, speed[:, None]], axis=1)
        y_test = test_data_df.trial_phase.values
        decoding_input.append(
            {
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "test_meta_data": meta_data,
            }
        )
    return decoding_input


# %% Supporting functions


def update_trial_definitions(navigation_df):
    """
    Currently trials are defined to start in navigation then move through reward_consumption and ITI.
    Bc/ we want to look at ITI -> navigation transitions. Redefine trials to start in ITI and end in
    reward_consumption.
    """
    new_trials, new_trial_phases = (
        pd.Series(index=navigation_df.index, data=np.nan, dtype=object),
        pd.Series(index=navigation_df.index, data=np.nan, dtype=object),
    )
    trials = navigation_df.trial.dropna().unique()
    for i in range(len(trials)):
        current_trial = trials[i]
        current_trial_df = navigation_df[navigation_df.trial == current_trial]
        if i == 0:
            # pretend ITI started 4 seconds before first cue
            start_ind = current_trial_df.index[0]
            pseudo_ITI_start = start_ind - 4 * FRAME_RATE
            pseudo_ITI_inds = pd.Index(np.arange(pseudo_ITI_start, start_ind))
            nav_inds = current_trial_df[(current_trial_df.trial_phase == "navigation")].index
            rc_inds = current_trial_df[(current_trial_df.trial_phase == "reward_consumption")].index
            combined_inds = np.concatenate([pseudo_ITI_inds, nav_inds, rc_inds])
            new_trial_phases.loc[pseudo_ITI_inds] = "ITI"
            new_trial_phases.loc[nav_inds] = "navigation"
            new_trial_phases.loc[rc_inds] = "reward_consumption"
            new_trials.loc[combined_inds] = current_trial
        else:
            previous_trial = trials[i - 1]
            previous_trial_df = navigation_df[navigation_df.trial == previous_trial]
            ITI_inds = previous_trial_df[(previous_trial_df.trial_phase == "ITI")].index
            nav_inds = current_trial_df[(current_trial_df.trial_phase == "navigation")].index
            reward_inds = current_trial_df[(current_trial_df.trial_phase == "reward_consumption")].index
            combined_inds = np.concatenate([ITI_inds, nav_inds, reward_inds])
            new_trial_phases.loc[ITI_inds] = "ITI"
            new_trial_phases.loc[nav_inds] = "navigation"
            new_trial_phases.loc[reward_inds] = "reward_consumption"
            new_trials.loc[combined_inds] = current_trial
    # overwrite navigation_df
    navigation_df[("trial", "")] = new_trials
    navigation_df[("trial_phase", "")] = new_trial_phases
    return navigation_df


def add_trial_phase_timers(navigation_df):
    """
    Add columns to navigation df that detial the time since you entered, or the time until you leave a trial phase
    (navigation, reward_consumption, ITI) in each trial. Will later be used to filter out data from around events
    like cue for the main decoding in this library.
    """
    time_from_trial_phase_start, time_to_trial_phase_end = pd.Series(index=navigation_df.index, data=np.nan), pd.Series(
        index=navigation_df.index, data=np.nan
    )
    pd.Series(index=navigation_df.index, data=np.nan)
    trials = navigation_df.trial.dropna().unique()
    times_from_start, times_to_end = [], []
    for t in trials:
        trial_df = navigation_df[navigation_df.trial == t]
        for phase in ["ITI", "navigation", "reward_consumption"]:
            phase_df = trial_df[trial_df.trial_phase == phase]
            if phase_df.empty:
                continue
            start_time, end_time = phase_df.time.iloc[[0, -1]]
            times_from_start.append(phase_df.time.sub(start_time))
            times_to_end.append(phase_df.time.sub(end_time).abs())
    time_from_trial_phase_start.update(pd.concat(times_from_start, axis=0))
    time_to_trial_phase_end.update(pd.concat(times_to_end, axis=0))
    # add to navigation_df
    navigation_df[("trial_phase_time", "from_start")] = time_from_trial_phase_start
    navigation_df[("trial_phase_time", "to_end")] = time_to_trial_phase_end
    return navigation_df
