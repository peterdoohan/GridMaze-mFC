"""
Library for event aligned LFP analysis.
@peterdoohan
"""

# %% Imports
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.multitest import multipletests
from itertools import combinations

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import load_data
from GridMaze.analysis.event_aligned import delta_distance_to_goal as ddtg
from GridMaze.analysis.event_aligned import lfp_utils as lu

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

LFP_RESULTS = RESULTS_PATH / "event_aligned" / "lfp"
if not LFP_RESULTS.exists():
    LFP_RESULTS.mkdir(parents=True)

with open(EXPERIMENT_INFO_PATH / "maze_configs.json", "r") as input_file:
    MAZE_CONFIGS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

with open(EXPERIMENT_INFO_PATH / "maze_day2date.json", "r") as input_file:
    MAZE_DAY2DATE = json.load(input_file)

FS = 1500  # lfp sampling frequency

# %% Spectrogram plots


def plot_cue_aligned_spectrogram_residuals(
    spectrogram_df,
    events=["cue_goal_directed", "cue_non_goal_directed"],
    ax=None,
    window=(-1, 1),
):
    """ """
    # prepare axes
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3))

    # process data for spectrogram plot
    df = spectrogram_df[spectrogram_df.late_session]
    event_dfs = [df[df.event == e] for e in events]
    av_spec_dfs = [_df.groupby("frequency").time.mean().time for _df in event_dfs]
    residual_df = av_spec_dfs[0] - av_spec_dfs[1]
    t = residual_df.columns.values.astype(np.float64)
    freqs = residual_df.index.values
    spec = residual_df.values
    t_mask = (t >= window[0]) & (t <= window[1])
    _min = spec[:, t_mask].min()
    _max = spec[:, t_mask].max()
    # plot
    _plot_spectrogram(spec, t, freqs, ax=ax, _min=_min, _max=_max)
    ax.set_xlim(*window)


def plot_average_spectrogram(
    spectrogram_df,
    axes=None,
    windows={"cue": (-1, 2), "reward": (-3, 3), "end_reward_consumption": (-1, 1)},
    vmax=None,
    vmin=None,
):
    """ """
    # prepare axes
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(7, 3), width_ratios=[0.5, 1, 0.5], sharey=True)
    # process data for plot
    events = ["cue", "reward", "end_reward_consumption"]
    df = spectrogram_df[spectrogram_df.late_session]
    event_dfs = [df[df.event == e] for e in events]
    av_spec_dfs = [_df.groupby("frequency").time.mean().time for _df in event_dfs]
    _max = max([av_spec_df.max().max() for av_spec_df in av_spec_dfs]) if vmax is None else vmax
    _min = min([av_spec_df.min().min() for av_spec_df in av_spec_dfs]) if vmin is None else vmin
    for i, (ax, av_spec_df) in enumerate(zip(axes, av_spec_dfs)):
        t = av_spec_df.columns.values.astype(np.float64)
        spec = av_spec_df.values
        freqs = av_spec_df.index.values
        cbar = True if i == 2 else False
        y_label = True if i == 0 else False
        _plot_spectrogram(
            spec,
            t,
            freqs,
            ax=ax,
            _min=_min,
            _max=_max,
            colorbar=cbar,
            y_label=y_label,
        )
        event = events[i]
        ax.set_title(event)
        ax.set_xlim(*windows[event])
    # fig.tight_layout()
    return


def _plot_spectrogram(x, times, freqs, ax=None, _min=None, _max=None, colorbar=True, y_label=True):
    """ """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    _min = x.min() if _min is None else _min
    _max = x.max() if _max is None else _max
    pcmesh = ax.pcolormesh(
        times,
        freqs,
        x,
        shading="auto",
        cmap="coolwarm",
        vmin=_min,
        vmax=_max,
        rasterized=True,
    )
    ax.axvline(0, color="white", linestyle="--")
    ax.grid(False)
    ax.set_yscale("log")
    if y_label:
        ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    if colorbar:
        cbar = plt.colorbar(pcmesh, ax=ax, orientation="vertical")
        cbar.set_label("Power (z-scored)")
        for spine in cbar.ax.spines.values():
            spine.set_visible(False)


# %% PSD plots


def plot_PSD(
    PSD_df,
    ax=None,
    normalise=False,
    fmax=150,
):
    """
    Looks much better with LFP, maybe too low s/n in CSD
    change to only plot trial phases
    """
    # prepare axes
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), sharex=True, sharey=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_xscale("log")
    # process data for plot
    df = PSD_df[PSD_df.late_session]
    trial_phases = ["navigation", "RC", "ITI"]
    av_subject_psd = df.groupby(["subject_ID", "frequency"]).power.mean()
    if normalise:
        subject_max_power = av_subject_psd.groupby(level="subject_ID")["power"].max().max(1)
        av_subject_psd = av_subject_psd.div(
            av_subject_psd.index.get_level_values("subject_ID").map(subject_max_power), axis=0
        )
    subject_grouped_psd = av_subject_psd.reset_index("subject_ID").groupby("frequency").power
    psd_mean = subject_grouped_psd.mean().power
    psd_sem = subject_grouped_psd.sem().power
    freqs = psd_mean.index.values
    for s, color in zip(trial_phases, ["darkred", "grey", "dodgerblue"]):
        mean = psd_mean[s].values
        sem = psd_sem[s].values
        ax.plot(freqs, mean, label=s, color=color)
        ax.fill_between(freqs, mean - sem, mean + sem, alpha=0.2, color=color)
        ax.legend()
    if fmax:
        ax.set_xlim(1, fmax)


def _get_PSD_stats(
    PSD_df,
    freq_ranges={"1-3Hz": (2, 3), "4-5Hz": (3, 5), "theta": (7, 10)},
):
    """
    compare fequency ranges between trial phases.
    """
    for name, fr in freq_ranges.items():
        _df = PSD_df[PSD_df.frequency.between(*fr)]
        trial_phase_mean_psd = _df.groupby("subject_ID").power.mean().power[["navigation", "RC", "ITI"]]
        # run anova
        df_long = trial_phase_mean_psd.reset_index().melt(
            id_vars="subject_ID", var_name="condition", value_name="value"
        )
        # Repeated-measures ANOVA
        aov = AnovaRM(data=df_long, depvar="value", subject="subject_ID", within=["condition"]).fit()
        print(aov.summary())
        # Parametric post-hoc (paired t-tests) with Holm correction
        pairs = list(combinations(trial_phase_mean_psd.columns, 2))
        t_results = []
        pvals = []
        for c1, c2 in pairs:
            # statsmodels' ttest_ind returns (tstat, pvalue, df)
            tstat, pval, dof = ttest_ind(
                trial_phase_mean_psd[c1].values, trial_phase_mean_psd[c2].values, usevar="pooled"
            )
            t_results.append({"contrast": f"{c1} vs {c2}", "t": tstat, "df": dof, "p_raw": pval})
            pvals.append(pval)
        p_df = pd.DataFrame(t_results)
        p_df["p_corrected"] = multipletests(pvals, method="holm")[1]
        print(p_df)


# %% Event aligned signal plots


def plot_cue_aligned_signal_residuals(
    signal_df,
    conditions=["goal_directed", "non_goal_directed"],
    ddtg_thresholds=(lu.DDTG_LOWER_THRES, lu.DDTG_UPPER_THRES),
    window=(-1, 1),
    ax=None,
):
    """ """
    # prepare axes
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("residual uV")
    ax.axhline(0, color="black", linestyle="--", alpha=0.2)
    ax.axvline(0, color="black", linestyle="--", alpha=0.2)

    # process data for plot
    def _get_trial_unique_ID(row, multi_index=False):
        if multi_index:
            row = row.droplevel(1)
        return f"{row.subject_ID}_{row.maze_name}_{int(row.day_on_maze)}_{int(row.trial)}"

    ddtg_df = ddtg.get_all_sessions_ddtg_at_cue()
    ddtg_df["trial_unique_ID"] = ddtg_df.apply(_get_trial_unique_ID, axis=1)
    condition2tu1Ds = {
        "goal_directed": ddtg_df[ddtg_df.ddtg.le(ddtg_thresholds[0])].trial_unique_ID.values,
        "non_goal_directed": ddtg_df[ddtg_df.ddtg.ge(ddtg_thresholds[1])].trial_unique_ID.values,
        "not_moving": ddtg_df[ddtg_df.ddtg == 0].trial_unique_ID.values,
    }
    df = signal_df[(signal_df.late_session) & (signal_df.event == "cue")].copy()
    df[("trial_unique_ID", "")] = df.apply(lambda row: _get_trial_unique_ID(row, True), axis=1)
    times = df.time.columns.values.astype(np.float64)
    results = np.zeros((len(SUBJECT_IDS), len(times)))
    for i, subject in enumerate(SUBJECT_IDS):
        subject_df = df[df.subject_ID == subject]
        cond_1_df = subject_df[subject_df.trial_unique_ID.isin(condition2tu1Ds[conditions[0]])]
        cond_2_df = subject_df[subject_df.trial_unique_ID.isin(condition2tu1Ds[conditions[1]])]
        cond_1_mean = cond_1_df.time.mean(axis=0).values
        cond_2_mean = cond_2_df.time.mean(axis=0).values
        residual = cond_1_mean - cond_2_mean
        results[i, :] = residual
    mean = results.mean(axis=0)
    sem = results.std(axis=0) / np.sqrt(len(SUBJECT_IDS))
    # plot
    ax.plot(times, mean, label="residual", color="k")
    ax.fill_between(times, mean - sem, mean + sem, alpha=0.2)
    ax.set_xlim(*window)


def plot_av_event_aligned_signal(
    signal_df, axes=None, windows={"cue": (-0.5, 0.5), "reward": (-1.5, 1.5), "end_reward_consumption": (-0.5, 0.5)}
):
    """mean, sem across subjects plotted"""
    # prepare axes
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(6, 3), width_ratios=[1, 3, 1], sharey=True)
    for ax in axes.flatten():
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("uV")
        ax.axhline(0, color="black", linestyle="--", alpha=0.2)
        ax.axvline(0, color="black", linestyle="--", alpha=0.2)
    # process data for plot
    df = signal_df[signal_df.late_session]
    event_dfs = [df[df.event == e] for e in ["cue", "reward", "end_reward_consumption"]]
    av_subject_dfs = [df.groupby("subject_ID").time.mean().time for df in event_dfs]
    t = av_subject_dfs[0].columns.values.astype(np.float64)
    for event, ax, av_subject_df in zip(["cue", "reward", "end_reward_consumption"], axes, av_subject_dfs):
        mean = av_subject_df.mean(axis=0).values
        sem = av_subject_df.sem(axis=0).values
        ax.plot(t, mean, label=event, color="k")
        ax.fill_between(t, mean - sem, mean + sem, alpha=0.3, color="k")
        ax.set_title(event)
        ax.set_xlim(*windows[event])
        ax.set_ylim(-0.5, 0.5)


def plot_av_subject_signal(
    signal_df,
    event="cue",
    window=(-1, 1),
    axes=None,
):
    """subjects plotted in separate pannels"""
    # prepare axes
    if axes is None:
        fig, axes = plt.subplots(2, 3, figsize=(6, 4), sharey=True)
    for ax in axes.flatten():
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.axhline(0, color="black", linestyle="--", alpha=0.2)
        ax.axvline(0, color="black", linestyle="--", alpha=0.2)
    # process data for plot
    df = signal_df[signal_df.late_session]
    df = df[df.event == event]
    times = df.time.columns.values.astype(np.float64)
    for subject, ax in zip(SUBJECT_IDS, axes.flatten()):
        subject_df = df[df.subject_ID == subject]
        mean = subject_df.time.mean(0).values
        ax.plot(times, mean, c="k")
        ax.set_title(subject)
    for ax in axes.flatten():
        ax.set_xlim(*window)
    fig.tight_layout()


# %% Load functions


def load_spectrogram_df(signal_type):
    """ """
    save_path = LFP_RESULTS / f"aligned_spectrograms_{signal_type}.parquet"
    if not save_path.exists():
        print(f"event aligned spectrogram dfs not found, save with fn: save_all_spectrogram_dfs")
    df = load_data._load_multiindex_parquet(save_path)
    return df


def load_PSD_df(signal_type):
    """ """
    save_path = LFP_RESULTS / f"trial_PSD_{signal_type}.parquet"
    if not save_path.exists():
        print(f"event aligned PSD dfs not found, save with fn: save_all_PSD_dfs")
    df = load_data._load_multiindex_parquet(save_path)
    return df


def load_signal_df(signal_type):
    """ """
    save_path = LFP_RESULTS / f"aligned_{signal_type}.parquet"
    if not save_path.exists():
        print(f"event aligned signal dfs not found, save with fn: save_all_signal_dfs")
    df = load_data._load_multiindex_parquet(save_path)
    return df


# %% Save functions


def _save_all():
    save_all_spectrogram_dfs(overwrite=False, signal_type="CSD", verbose=True, save_session_dfs=False)
    save_all_spectrogram_dfs(overwrite=False, signal_type="LFP", verbose=True, save_session_dfs=False)
    save_all_signal_dfs(overwrite=False, signal_type="CSD", verbose=True)
    save_all_signal_dfs(overwrite=False, signal_type="LFP", verbose=True)
    save_all_PSD_dfs(signal_type="CSD", overwrite=False, verbose=True)
    save_all_PSD_dfs(signal_type="LFP", overwrite=False, verbose=True)


def save_all_spectrogram_dfs(overwrite=False, signal_type="CSD", verbose=False, save_session_dfs=True):
    save_path = LFP_RESULTS / f"aligned_spectrograms_{signal_type}.parquet"
    if save_path.exists() and not overwrite:
        print(f"event aligned spectrogram dfs already populated: {save_path.name}")
        return
    spectrogram_dfs = []
    for subject in SUBJECT_IDS:
        for maze in MAZE_CONFIGS.keys():
            for day in [int(d) for d in MAZE_DAY2DATE[maze].keys()]:
                try:
                    session = gs.get_maze_sessions(
                        subject_IDs=[subject],
                        maze_names=[maze],
                        days_on_maze=[day],
                        with_data=[
                            "trials_df",
                            "lfp_times",
                            "lfp_signal",
                            "lfp_metrics",
                            "cluster_metrics",
                            "navigation_df",
                        ],
                        must_have_data=True,
                    )
                except FileNotFoundError as e:
                    if verbose:
                        print(f"skipping session: {subject} {maze} {day} - MISSING DATA")
                        continue
                try:
                    spectrogram_df = lu._get_spectrogram_df(
                        session,
                        events=[
                            "cue",
                            "reward",
                            "end_reward_consumption",
                            "cue_goal_directed",
                            "cue_non_goal_directed",
                            "cue_not_moving",
                        ],
                        signal_type=signal_type,
                        single_channel=False,
                        window=(-3, 3),
                    )
                    if save_session_dfs:
                        save_path = LFP_RESULTS / "aligned_spectrograms" / signal_type / f"{session.name}.parquet"
                        if not overwrite and save_path.exists():
                            continue
                        else:
                            print(f"Saving {save_path.name}")
                            spectrogram_df.columns = spectrogram_df.columns.map(lambda x: str(x))
                            spectrogram_df.to_parquet(save_path, compression="gzip")
                    spectrogram_dfs.append(spectrogram_df)
                except ValueError as e:
                    if verbose:
                        print(e)
                        print(f"Failed to generate spectrogram for session: {session.name}")
    save_path = LFP_RESULTS / f"aligned_spectrograms_{signal_type}.parquet"
    combined_dfs = pd.concat(spectrogram_dfs, axis=0).reset_index(drop=True)
    print(f"Saving {save_path.name}")
    combined_dfs.columns = combined_dfs.columns.map(lambda x: str(x))
    combined_dfs.to_parquet(save_path, compression="gzip")
    return combined_dfs


def save_all_signal_dfs(overwrite=False, signal_type="CSD", verbose=False):
    """Generates a dataframe of event-aligned LFP/CSD data for all sessions"""
    save_path = LFP_RESULTS / f"aligned_{signal_type}.parquet"
    if save_path.exists() and not overwrite:
        print(f"event aligned signal dfs already populated: {save_path.name}")
        return
    signal_dfs = []
    for subject in SUBJECT_IDS:
        for maze in MAZE_CONFIGS.keys():
            for day in [int(d) for d in MAZE_DAY2DATE[maze].keys()]:
                try:
                    session = gs.get_maze_sessions(
                        subject_IDs=[subject],
                        maze_names=[maze],
                        days_on_maze=[day],
                        with_data=["trials_df", "lfp_times", "lfp_signal", "lfp_metrics", "cluster_metrics"],
                        must_have_data=True,
                    )
                except FileNotFoundError as e:
                    if verbose:
                        print(f"skipping session: {subject} {maze} {day} - MISSING DATA")
                        continue
                if verbose:
                    print(session.name)
                try:
                    signal_dfs.append(
                        lu._get_signal_df(
                            session,
                            events=["cue", "reward", "end_reward_consumption"],
                            signal_type=signal_type,
                            single_channel=False,
                            window=(-3, 3),
                        )
                    )
                except ValueError as e:
                    if verbose:
                        print(e)
                        print(f"Failed generate event aligned signal for session: {session.name}")
    df = pd.concat(signal_dfs, axis=0)
    # save
    df.columns = df.columns.map(lambda x: str(x))
    df.to_parquet(save_path, compression="gzip")
    return df


def save_all_PSD_dfs(signal_type="CSD", overwrite=False, verbose=False):
    """ """
    save_path = LFP_RESULTS / f"trial_PSD_{signal_type}.parquet"
    if save_path.exists() and not overwrite:
        print(f"event aligned signal dfs already populated: {save_path.name}")
        return
    PSD_dfs = []
    for subject in SUBJECT_IDS:
        for maze in MAZE_CONFIGS.keys():
            for day in [int(d) for d in MAZE_DAY2DATE[maze].keys()]:
                try:
                    session = gs.get_maze_sessions(
                        subject_IDs=[subject],
                        maze_names=[maze],
                        days_on_maze=[day],
                        with_data=["trials_df", "lfp_times", "lfp_signal", "lfp_metrics", "cluster_metrics"],
                        must_have_data=True,
                    )
                except FileNotFoundError as e:
                    if verbose:
                        print(f"skipping session: {subject} {maze} {day} - MISSING DATA")
                        continue
                if verbose:
                    print(session.name)
                try:
                    PSD_dfs.append(lu._get_trial_phase_PSD_df(session, signal_type=signal_type))
                except ValueError as e:
                    if verbose:
                        print(e)
                        print(f"Failed generate event aligned signal for session: {session.name}")
    df = pd.concat(PSD_dfs, axis=0)
    # save
    df.columns = df.columns.map(lambda x: str(x))
    df.to_parquet(save_path, compression="gzip")
    return df
