"""
Library of calculating and ploting mFC population activity aligned to cue, reward and other trial events
@ peterdoohan
"""

# %% Imports

# %% Global Variables


# %% Functions


def get_population_average_aligned_activity(
    plot=True, aligned_to="event", normalise_clusters="max", normalise_sessions="max"
):
    """"""
    av_rates = []
    data_structure = aligned_to + "_aligned_rates_df"
    for subject in EXP_INFO["subject_IDs"]:
        sessions = gs.get_sessions(
            subject_IDs=[subject], maze_number="all", day_on_maze="late", with_data=[data_structure]
        )
        session_av_rates = []
        for session in sessions:
            aligned_rates_df = getattr(session, data_structure)
            aligned_rates_df = aligned_rates_df[aligned_rates_df.cluster_type == "good"]
            # average neurons over trials
            trial_average_rates = (
                aligned_rates_df.set_index("cluster_unique_ID").groupby("cluster_unique_ID").mean().firing_rate
            )
            if normalise_clusters == "max":
                if aligned_to == "event":
                    for event in ["cue_aligned", "reward_aligned"]:
                        trial_average_rates[event] = trial_average_rates[event].apply(lambda x: x / x.max(), axis=1)
                else:
                    trial_average_rates = trial_average_rates.apply(lambda x: x / x.max(), axis=1)
            population_average_rates = aligned_rates_df.firing_rate.mean(axis=0)
            session_av_rates.append(population_average_rates)
        subject_av_rates = pd.concat(session_av_rates, axis=1).T
        if normalise_sessions == "max":
            subject_av_rates = subject_av_rates.apply(lambda x: x / x.max(), axis=1)
        elif normalise_sessions == "zscore":
            subject_av_rates = subject_av_rates.apply(zscore, axis=1)
        subject_av_rates = subject_av_rates.mean()
        av_rates.append(subject_av_rates)
    population_average_rates = pd.concat(av_rates, axis=1).T
    population_average_rates.index = EXP_INFO["subject_IDs"]
    if plot:
        if aligned_to == "event":
            _plot_population_event_aligned_activity(population_average_rates)
        elif aligned_to == "trial":
            _plot_population_trial_aligned_activity(population_average_rates)
    return population_average_rates


# %% Plotting
def _plot_population_event_aligned_activity(population_average_rates):
    f, axes = plt.subplots(1, 2, figsize=(6, 3), clear=True, sharey=True)
    for i, event in enumerate(["cue_aligned", "reward_aligned"]):
        event_aligned_activity = population_average_rates[event]
        time = event_aligned_activity.columns.to_numpy(dtype=float)
        y = event_aligned_activity.mean(axis=0).to_numpy()
        sem = event_aligned_activity.sem(axis=0).to_numpy()
        axes[i].plot(time, y, color="orange")
        axes[i].fill_between(time, y - sem, y + sem, color="orange", alpha=0.5)
        axes[i].axvline(0, color="k", linewidth=1, alpha=0.5, zorder=0)
        axes[i].set_xlabel(f"{event} time (s)")
        axes[i].spines["right"].set_visible(False)
        axes[i].spines["top"].set_visible(False)
        if i == 0:
            axes[i].set_ylabel("Population average firing rate")
    return


def _plot_population_trial_aligned_activity(population_average_rates):
    f, ax = plt.subplots(1, 1, figsize=(6, 3), clear=True)
    time = population_average_rates.columns.to_numpy(dtype=float)
    y = population_average_rates.mean(axis=0).to_numpy()
    sem = population_average_rates.sem(axis=0).to_numpy()
    ax.plot(time, y, color="orange")
    ax.fill_between(time, y - sem, y + sem, color="orange", alpha=0.5)
    for x in EXP_INFO["intra_trial_interval_times"]:
        ax.axvline(x, color="k", linewidth=1, ls="--", alpha=0.5, zorder=0)
    ax.set_xlabel("Time (s)")
    ax.set_xticks([float(x) for x in EXP_INFO["intra_trial_interval_times"]])
    ax.set_xticklabels(["Cue", "Reward", "ITI", "end"])
    ax.set_ylabel("Population average firing rate")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return
