""" """

# %% Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


# %% Globs


# %% Functions


def plot_performance_validation(
    results_df,
    input_features=[
        "place",
        "place_direction",
        "place_direction_distance_to_goal",
        "place_direction_distance_to_goal_egocentric_action",
    ],
    outlier_threshold=-0.3,
    plot_single_subjects=True,
    print_stats=True,
    ax=None,
):
    """ """
    # set up figure
    if ax is None:
        f, ax = plt.subplots(figsize=(5, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("input features")
    ax.set_ylabel("CV Poisson deviance")
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)

    # prcoess data
    df = results_df.copy()
    # average over folds
    df = df.groupby(["cluster_unique_ID", "model_name"]).cv_score.mean().unstack(level=1)  # n_neurons, n_models
    # add back subject_ID
    df["subject_ID"] = [i.split(".")[0] for i in df.index]
    df = df.set_index("subject_ID", append=True)
    # remove outlier score values (useually due to few spikes in Poisson GLM)
    df = df[df.gt(outlier_threshold).all(axis=1)]  # (n_neurons, n_models)
    # set multiindex column for baseline vs emebedding comparison
    df.columns = pd.MultiIndex.from_tuples([tuple(c.split("_", 1)) for c in df.columns])

    df = df[df.columns[df.columns.get_level_values(1).isin(input_features)]]
    subj_mean = df.groupby("subject_ID").mean()
    if print_stats:
        print("baseline vs embedding:")
        print(_performance_validation_stats(subj_mean))

    subj_sem = df.groupby("subject_ID").sem()
    grand_mean = subj_mean.mean()
    grand_sem = subj_mean.sem()

    models = df.columns.get_level_values(1).unique()
    versions = df.columns.get_level_values(0).unique()
    subjects = subj_mean.index
    n_models = len(models)
    x = np.arange(n_models)
    off = 0.15
    jitter = 0.025

    if plot_single_subjects:
        palette = sns.color_palette("hls", len(subjects))
        subject_colors = dict(zip(subjects, palette))
        for subj in subjects:
            for i, model in enumerate(models):
                for version in versions:
                    y = subj_mean.loc[subj, (version, model)]
                    yerr = subj_sem.loc[subj, (version, model)]
                    base = i + (-off if version == "baseline" else off)
                    xpos = base + np.random.uniform(-jitter, jitter)
                    plt.errorbar(
                        xpos,
                        y,
                        yerr=yerr,
                        fmt="o",
                        ecolor=subject_colors[subj],
                        markeredgecolor=subject_colors[subj],
                        markerfacecolor=subject_colors[subj],
                        alpha=0.5,
                        markersize=5,
                        capsize=0,
                    )

    version_colors = {"baseline": "grey", "embedding": "purple"}
    # Plot grand means ± SEM with flat '-' markers
    for version in versions:
        means = [grand_mean.loc[(version, m)] for m in models]
        sems = [grand_sem.loc[(version, m)] for m in models]
        xpos = x - off if version == "baseline" else x + off
        ax.errorbar(
            xpos,
            means,
            yerr=sems,
            linestyle="",
            marker="_",
            markersize=14,
            markeredgewidth=3,
            color=version_colors[version],
            ecolor=version_colors[version],
            elinewidth=2,
            capsize=0,
            label=version.capitalize(),
        )

    # Final formatting
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    return


def _performance_validation_stats(subject_mean_df):
    """cross subejct t-test for baseline vs embedding performance"""
    models = subject_mean_df.columns.get_level_values(1).unique()
    stats = []
    for model in models:
        _df = subject_mean_df.xs(model, level=1, axis=1)
        t_stat, p_val = ttest_rel(_df["baseline"], _df["embedding"])
        stats.append({"model": model, "t_stat": t_stat, "p_val": p_val})
    stats_df = pd.DataFrame(stats)
    stats_df["p_val_corr"] = multipletests(stats_df.p_val, method="fdr_bh")[1]
    return stats_df
