""" 
Library for plotting results from exps that compare the input features of the embedding model.
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from . import plot_latents
from .load_experiment import load_exp_results
from matplotlib import pyplot as plt
import seaborn as sns

# %% Globall Variabesl
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

EMBEDDING_MODEL_RESULTS = RESULTS_PATH / "embedding_model" / "exps"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

MAZE_NAMES = ["maze_1", "maze_2", "rooms_maze"]

# %% All vs one subjects exps


def plot_all_vs_one_training_results():
    """ """
    f, axes = plt.subplots(len(SUBJECT_IDS), len(MAZE_NAMES), sharex=True, sharey=True, figsize=(10, 10), clear=True)
    for i, subject_ID in enumerate(SUBJECT_IDS):
        for j, maze_name in enumerate(MAZE_NAMES):
            legend = True if i == 0 and j == 0 else False
            exp_name = f"{subject_ID}_{maze_name}_state_action_distance"
            plot_latents.plot_crossval_training_perf(exp_name, ax=axes[i, j], legend=legend)
            axes[i, j].set_title(f"{subject_ID} {maze_name}")
    return


def plot_all_vs_one_cluster_perf(perf_range=(-0.5, 1), with_embedding=True):
    """ """
    # load data from single subject exps
    single_subj_dfs = []
    for maze_name in MAZE_NAMES:
        for subject_ID in SUBJECT_IDS:
            exp_name = f"{subject_ID}_{maze_name}_state_action_distance"
            single_subj_dfs.append(load_exp_results(exp_name, "cluster_crossval_perf"))
    single_subj_df = pd.concat(single_subj_dfs, axis=0).reset_index(drop=True)
    single_subj_df["all_subjects"] = False
    # load data from all subjects exp
    all_subj_dfs = []
    for maze_name in MAZE_NAMES:
        exp_name = f"all_subjects_{maze_name}_state_action_distance"
        all_subj_dfs.append(load_exp_results(exp_name, "cluster_crossval_perf"))
    all_subj_df = pd.concat(all_subj_dfs, axis=0).reset_index(drop=True)
    all_subj_df["all_subjects"] = True
    results_df = pd.concat([single_subj_df, all_subj_df], axis=0).reset_index(drop=True)
    # HACK: if maze_1 filter from days 7-13
    results_df = results_df[~((results_df.maze_name == "maze_1") & (~results_df.day_on_maze.between(7, 13)))]
    # average over folds
    cluster_mean_perf = (
        results_df.groupby(
            ["subject_ID", "maze_name", "day_on_maze", "with_embedding", "cluster_unique_ID", "all_subjects"]
        )
        .cv_performance.mean()
        .reset_index()
    )
    # optionally exclude custers with performance outside specific range (large neg values for poor fits)
    if perf_range:
        cluster_mean_perf = cluster_mean_perf[cluster_mean_perf.cv_performance.between(*perf_range)]

    sns.catplot(
        data=cluster_mean_perf[cluster_mean_perf.with_embedding == with_embedding],
        x="subject_ID",
        y="cv_performance",
        col="maze_name",
        kind="bar",
        hue="all_subjects",
        palette="tab10",
    )
    plt.title(maze_name)
    plt.show()
    return cluster_mean_perf


def plot_performance_across_days(maze_name, clip_perf=(-0.2, 1)):
    """ """
    cluster_xval_perf_dfs = []
    for subject_ID in SUBJECT_IDS:
        exp_name = f"{subject_ID}_{maze_name}_state_action_distance"
        cluster_xval_perf_dfs.append(load_exp_results(exp_name, "cluster_crossval_perf"))
    cluster_perf_df = pd.concat(cluster_xval_perf_dfs, axis=0).reset_index(drop=True)
    # average performance scores across folds and clusters to get a mean performance for each subject-day
    session_mean_perf = (
        cluster_perf_df.groupby(["subject_ID", "maze_name", "day_on_maze", "with_embedding"])
        .cv_performance.mean()
        .reset_index()
    )
    # filter performance outside range usually large neg values
    if clip_perf:
        session_mean_perf = session_mean_perf[session_mean_perf.cv_performance.between(*clip_perf)]
    for with_embedding in [True, False]:
        sns.lineplot(
            data=session_mean_perf[session_mean_perf.with_embedding == with_embedding],
            x="day_on_maze",
            y="cv_performance",
            hue="subject_ID",
        )
        plt.title(f"{maze_name} with_embedding={with_embedding}")
        plt.show()
    return


# %% Resolution (window length) exps


def plot_resolution_exp_results():
    """Note this was run for just one subject as a test"""
    results_dfs = []
    for resolution in [0.1, 0.2, 0.5]:
        exp_name = f"resolution_{resolution}"
        df = load_exp_results(exp_name, "cluster_crossval_perf")
        df["resolution"] = resolution
        results_dfs.append(df)
    results_df = pd.concat(results_dfs, axis=0).reset_index(drop=True)
    # average over folds
    cluster_mean_perf = (
        results_df.groupby(["with_embedding", "cluster_unique_ID", "resolution"]).cv_performance.mean().reset_index()
    )
    # plot
    sns.barplot(data=cluster_mean_perf, x="resolution", y="cv_performance", hue="with_embedding")


# %% 10 vs 20 latent experiments


def plot_n_latent_experiment_results():
    """ """
    results_dfs = []
    for maze_name in MAZE_NAMES:
        for subject in SUBJECT_IDS:
            for n_latent in [10, 20]:
                exp_name = f"{subject}_{maze_name}_default_latents_{n_latent}"
                try:
                    df = load_exp_results(exp_name, "cluster_crossval_perf")
                    df["n_latents"] = n_latent
                    results_dfs.append(df)
                except FileNotFoundError:
                    print(f"{exp_name} not found, probably still running. Skip for now.")
                    continue
    results_df = pd.concat(results_dfs, axis=0).reset_index(drop=True)
    # HACK: if maze_1 filter from days 7-13
    results_df = results_df[~((results_df.maze_name == "maze_1") & (~results_df.day_on_maze.between(7, 13)))]
    # average over folds
    cluster_mean_perf = (
        results_df.groupby(["subject_ID", "maze_name", "day_on_maze", "n_latents", "with_embedding"])
        .cv_performance.mean()
        .reset_index()
    )
    # plot
    sns.catplot(
        data=cluster_mean_perf,
        x="subject_ID",
        y="cv_performance",
        col="maze_name",
        kind="bar",
        hue="n_latents",
        palette="tab10",
    )
    return cluster_mean_perf


# %% state action experiments


def plot_state_action_results(maze_name):
    """
    X
    """
    # load and combine all experiments results saved to disk
    results_dfs = []
    for subject in SUBJECT_IDS:
        for inv_link in ["softplus", "exp"]:
            for l in ["linear", "nonlinear"]:
                exp_name = f"{subject}_{maze_name}_{inv_link}_state_action_{l}"
                try:
                    df = load_exp_results(exp_name, "state_action_interactions", "cluster_crossval_perf")
                    df["inv_link"] = inv_link
                    df["interaction"] = l
                    df["features"] = "state_action"
                    results_dfs.append(df)
                except FileNotFoundError:
                    print(f"{exp_name} not found, probably still running. Skip for now.")
                    continue
            for f in ["state", "action"]:
                exp_name = f"{subject}_{maze_name}_{inv_link}_{f}"
                try:
                    df = load_exp_results(exp_name, "state_action_interactions", "cluster_crossval_perf")
                    df["inv_link"] = inv_link
                    df["interaction"] = None
                    df["features"] = f
                    results_dfs.append(df)
                except FileNotFoundError:
                    print(f"{exp_name} not found, probably still running. Skip for now.")
                    continue
    results_df = pd.concat(results_dfs, axis=0).reset_index(drop=True)
    # average over folds
    cluster_mean_perf = (
        results_df.groupby(
            [
                "subject_ID",
                "maze_name",
                "day_on_maze",
                "inv_link",
                "interaction",
                "features",
                "cluster_unique_ID",
            ]
        )
        .cv_performance.mean()
        .reset_index()
    )
    # plot data
    interaction_df = results_df[(results_df.features == "state_action")]
    interaction_df = interaction_df[interaction_df.day_on_maze.between(7, 13)]
    cluster_mean_perf = (
        interaction_df.groupby(
            [
                "subject_ID",
                "maze_name",
                "day_on_maze",
                "cluster_unique_ID",
                "inv_link",
                "interaction",
            ]
        )
        .cv_performance.mean()
        .reset_index()
    )
    sns.catplot(
        data=cluster_mean_perf,
        x="subject_ID",
        y="cv_performance",
        col="inv_link",
        kind="bar",
        hue="interaction",
        palette="tab10",
    )
    sns.catplot(
        data=cluster_mean_perf,
        x="subject_ID",
        y="cv_performance",
        col="interaction",
        kind="bar",
        hue="inv_link",
        palette="tab10",
    )
    #
    single_fetures_df = results_df[results_df.with_embedding & (results_df.features != "state_action")]
    single_fetures_df = single_fetures_df[single_fetures_df.day_on_maze.between(7, 13)]
    cluster_mean_perf = (
        single_fetures_df.groupby(
            [
                "subject_ID",
                "maze_name",
                "day_on_maze",
                "cluster_unique_ID",
                "inv_link",
                "features",
            ]
        )
        .cv_performance.mean()
        .reset_index()
    )
    sns.catplot(
        data=cluster_mean_perf,
        x="subject_ID",
        y="cv_performance",
        col="inv_link",
        kind="bar",
        hue="features",
        palette="tab10",
    )
    #
    without_embedding_df = results_df[~results_df.with_embedding & (results_df.interaction != "non-linear")]
    without_embedding_df = without_embedding_df[without_embedding_df.day_on_maze.between(7, 13)]
    cluster_mean_perf = (
        without_embedding_df.groupby(
            [
                "subject_ID",
                "maze_name",
                "day_on_maze",
                "cluster_unique_ID",
                "inv_link",
                "features",
            ]
        )
        .cv_performance.mean()
        .reset_index()
    )
    sns.catplot(
        data=cluster_mean_perf,
        x="subject_ID",
        y="cv_performance",
        col="inv_link",
        kind="bar",
        hue="features",
        palette="tab10",
    )


# %% new state-action results


def _load_state_action_interaction_results(exp_set="state_action_interactions", maze_name="maze_1", day_range=(7, 13)):
    """ """
    results_dfs = []
    for subject_ID in SUBJECT_IDS:
        for inv_link in ["exp", "softplus"]:
            for feature in ["state", "action"]:
                exp_name = f"{subject_ID}_{maze_name}_{inv_link}_{feature}"
                results_dfs.append(_load_results_df(exp_name, exp_set, abbrev=f"{inv_link}_{feature}"))
            for interaction in ["linear", "nonlinear", "conjunctive"]:
                exp_name = f"{subject_ID}_{maze_name}_{inv_link}_state_action_{interaction}"
                results_dfs.append(_load_results_df(exp_name, exp_set, abbrev=f"{inv_link}_state_action_{interaction}"))
        for feature in ["state", "action", "state_action"]:
            exp_name = f"{subject_ID}_{maze_name}_{feature}_no-embedding"
            results_dfs.append(_load_results_df(exp_name, exp_set, abbrev=f"{feature}_no-embedding"))
    results_df = pd.concat(results_dfs, axis=0).reset_index(drop=True)
    if day_range:
        results_df = results_df[results_df.day_on_maze.between(*day_range)]
    return results_df


def _load_state_action_interaction_results2(exp_set="state_action_interactions", maze_name="maze_1", day_range=(7, 13)):
    """ """
    exp_names, abbrevs = [], []
    results_dir = EMBEDDING_MODEL_RESULTS / exp_set
    exp_names = [d.name for d in results_dir.iterdir()]
    abbrevs = [n.split("_", 3)[-1] for n in exp_names]
    results_df = _load_results(exp_names, exp_set, abbrevs)
    if day_range:
        results_df = results_df[results_df.day_on_maze.between(*day_range)]
    return results_df


def _load_results_df(exp_name, exp_set, abbrev):
    """"""
    try:
        df = load_exp_results(exp_name, exp_set, data_structure="cluster_crossval_perf", average_over_folds=True)
        df["abbrev"] = abbrev
        return df
    except:
        print(f"{exp_name} not found, probably still running. Returning None.")
        return None


def _load_results(exp_set, average_over_folds=True, day_range=(7, 13)):
    """"""

    def _load(exp_name, exp_set, abbrev):
        try:
            df = load_exp_results(
                exp_name, exp_set, data_structure="cluster_crossval_perf", average_over_folds=average_over_folds
            )
            df["abbrev"] = abbrev
            return df
        except:
            print(f"{exp_name} not found, probably still running. Returning None.")
            return None

    # from exp folder names, get exp names and abbrevs
    exp_names, abbrevs = [], []
    results_dir = EMBEDDING_MODEL_RESULTS / exp_set
    exp_names = [d.name for d in results_dir.iterdir()]
    abbrevs = [n.split("_", 3)[-1] for n in exp_names]
    # load and combine into one df
    results_dfs = []
    for exp_name, abbrev in zip(exp_names, abbrevs):
        df = _load(exp_name, exp_set, abbrev)
        if df is not None:
            results_dfs.append(df)
    results_df = pd.concat(results_dfs, axis=0).reset_index(drop=True)
    if day_range:
        results_df = results_df[results_df.day_on_maze.between(*day_range)]
    return results_df


def _plot_results(results_df, x_label_order=None, ax=None):
    """ """
    if x_label_order is None:
        x_label_order = list(results_df.abbrev.unique())
    results_df["abbrev"] = pd.Categorical(results_df["abbrev"], x_label_order)
    subject_cond_mean_perf = (
        results_df.groupby(["subject_ID", "abbrev"], observed=True).cv_performance.mean().reset_index()
    )
    # plotting
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(len(x_label_order), 5), clear=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="black", lw=1, ls="--")
    if len(results_df.subject_ID.unique()) > 1:
        hue = "subject_ID"
        dodge = 0.4
    else:
        hue = None
        dodge = 0
    sns.pointplot(
        results_df,
        x="abbrev",
        y="cv_performance",
        hue=hue,
        dodge=dodge,
        linestyle="none",
        alpha=0.3,
        markeredgewidth=0,
        markersize=5,
        err_kws={"linewidth": 2},
        ax=ax,
    )
    if len(results_df.subject_ID.unique()) > 1:
        sns.pointplot(
            subject_cond_mean_perf,
            x="abbrev",
            y="cv_performance",
            marker="_",
            markersize=20,
            markeredgewidth=3,
            color="black",
            linestyle="none",
            errorbar=None,
            ax=ax,
        )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("")


def plot_state_action_interaction_results():
    """ """
    results_df = _load_state_action_interaction_results()
    # process data
    x_label_order = [
        "action_no-embedding",
        "exp_action",
        "softplus_action",
        "state_no-embedding",
        "exp_state",
        "softplus_state",
        "state_action_no-embedding",
        "exp_state_action_linear",
        "softplus_state_action_linear",
        "exp_state_action_nonlinear",
        "softplus_state_action_nonlinear",
        "exp_state_action_conjunctive",
        "softplus_state_action_conjunctive",
    ]
    _plot_results(results_df, x_label_order)


# %% state-action distance interaction experiments


def _load_state_action_distance_interaction_results(
    exp_set="state-action_distance_interactions", maze_name="maze_1", day_range=(7, 13)
):
    """ """
    results_dfs = []
    for subject_ID in SUBJECT_IDS:
        for inv_link in ["exp", "softplus"]:
            for feature in ["state-action", "distance"]:
                exp_name = f"{subject_ID}_{maze_name}_{inv_link}_{feature}"
                results_dfs.append(_load_results_df(exp_name, exp_set, abbrev=f"{inv_link}_{feature}"))
                exp_name = f"{subject_ID}_{maze_name}_{feature}_no-embedding"
                results_dfs.append(_load_results_df(exp_name, exp_set, abbrev=f"{feature}_no-embedding"))
            for interaction in ["linear", "nonlinear"]:
                exp_name = f"{subject_ID}_{maze_name}_{inv_link}_state-action_distance_{interaction}"
                results_dfs.append(
                    _load_results_df(exp_name, exp_set, abbrev=f"{inv_link}_state-action_distance_{interaction}")
                )
        exp_name = f"{subject_ID}_{maze_name}_state-action-distance_no-embedding"
        results_dfs.append(_load_results_df(exp_name, exp_set, abbrev="state-action-distance_no-embedding"))
        # linear guassian stuff
        for interaction in ["linear", "nonlinear"]:
            exp_name = f"{subject_ID}_{maze_name}_state-action_distance_{interaction}-gaussian"
            results_dfs.append(
                _load_results_df(exp_name, exp_set, abbrev=f"state-action_distance_{interaction}-gaussian")
            )
        for feature in ["state-action", "distance"]:
            exp_name = f"{subject_ID}_{maze_name}_{feature}_gaussian"
            results_dfs.append(_load_results_df(exp_name, exp_set, abbrev=f"{feature}_gaussian"))

    results_df = pd.concat(results_dfs, axis=0).reset_index(drop=True)
    if day_range:
        results_df = results_df[results_df.day_on_maze.between(*day_range)]
    return results_df


def plot_state_action_distance_interaction_results():
    """ """
    results_df = _load_state_action_distance_interaction_results()
    x_label_order = [
        "state-action-distance_no-embedding",
        "distance_no-embedding",
        "exp_distance",
        "softplus_distance",
        "state-action_no-embedding",
        "exp_state-action",
        "softplus_state-action",
        "exp_state-action_distance_linear",
        "softplus_state-action_distance_linear",
        "exp_state-action_distance_nonlinear",
        "softplus_state-action_distance_nonlinear",
        "distance_gaussian",
        "state-action_gaussian",
        "state-action_distance_linear-gaussian",
        "state-action_distance_nonlinear-gaussian",
    ]
    _plot_results(results_df, x_label_order)


# %% distance metrics comparison


def _load_distance_metric_comparison_results(maze_name="maze_1", exp_set="distance_metric_comparison"):
    """ """
    results_df = []
    for subject in SUBJECT_IDS:
        for distance_metric in ["euclidean", "manhattan", "geodesic", "future"]:
            exp_name = f"{subject}_{maze_name}_distance_to_goal_{distance_metric}"
            results_df.append(_load_results_df(exp_name, exp_set, abbrev=distance_metric))
        for progress_metric in ["time", "path_length"]:
            exp_name = f"{subject}_{maze_name}_progress_to_goal_{progress_metric}"
            results_df.append(_load_results_df(exp_name, exp_set, abbrev=progress_metric))
    results_df = pd.concat(results_df, axis=0).reset_index(drop=True)
    return results_df


def plot_distance_metric_comparison_results():
    """ """
    results_df = _load_distance_metric_comparison_results()
    x_order = ["euclidean", "manhattan", "geodesic", "future", "time", "path_length"]
    _plot_results(results_df, x_order)


# %% var explained analysis


def _load_var_explained_results(maze_name="maze_1", exp_set="var_explained"):
    """"""
    results_df = []
    for subject in SUBJECT_IDS:
        for i in ["full_model", "big_full_model", "full_model_partitioned"]:
            exp_name = f"{subject}_{maze_name}_{i}"
            results_df.append(_load_results_df(exp_name, exp_set, abbrev=i))
    results_df = pd.concat(results_df, axis=0).reset_index(drop=True)
    return results_df


def plot_var_explained_results():
    results_df = _load_var_explained_results()
    x_order = ["full_model", "big_full_model", "full_model_partitioned"]
    _plot_results(results_df, x_order)


# %% egocentric angle exp


def _load_egocentric_action_results(exp_set="egocentric_action2"):
    """ """
    results_df = []
    for subject in ["m2"]:
        for maze_name in ["maze_1"]:
            for i in [
                "full_model",
                "reduced_all",
                "reduced_LR",
                "reduced_choice",
                "SAD_full_model",
                "SAD_reduced_choice",
                "SAD_reduced_D",
                "SAD_reduced_LR",
                "SAD_reduced_LR_choice",
                "SAD_reduced_SA",
            ]:
                exp_name = f"{subject}_{maze_name}_{i}"
                try:
                    results_df.append(_load_results_df(exp_name, exp_set, abbrev=i))
                except FileNotFoundError:
                    print(f"{exp_name} not found, probably still running. Skip for now.")
                    continue
    results_df = pd.concat(results_df, axis=0).reset_index(drop=True)
    return results_df


# %% Linear-Linear comparison


def _load_linear_linear_results(exp_set="state-action_distance_full_linear2"):
    """ """
    results_df = []
    for subject in ["m2"]:
        for maze_name in ["maze_1"]:
            for i in ["product-space_full_linear", "onehots_full_linear", "nonlin"]:
                exp_name = f"{subject}_{maze_name}_{i}"
                try:
                    results_df.append(_load_results_df(exp_name, exp_set, abbrev=i))
                except FileNotFoundError:
                    print(f"{exp_name} not found, probably still running. Skip for now.")
                    continue
    return pd.concat(results_df, axis=0).reset_index(drop=True)


# %% Other bits
def _load_embedding_troubleshoot_results(exp_set="embedding_trouble_shoot"):
    results_df = []
    for test in [
        "last_3",
        "last_5",
        "last_7",
        "late_24_goals",
        "Nlat_10",
        "Nlat_20",
        "Nlat_50",
    ]:
        exp_name = f"state-action_embedding_{test}"
        try:
            df = load_exp_results(exp_name, exp_set, "cluster_crossval_perf")
            df["abbrev"] = test
            results_df.append(df)
        except FileNotFoundError:
            print(f"{exp_name} not found, probably still running. Skip for now.")
            continue
    return pd.concat(results_df, axis=0).reset_index(drop=True)


def plot_embedding_troubleshoot(ax1=None, ax2=None):
    results_df = _load_embedding_troubleshoot_results()
    embed_sessions_df = results_df[results_df.abbrev.isin(["last_3", "last_5", "last_7", "late_24_goals"])]
    embed_sessions_df = embed_sessions_df[embed_sessions_df.day_on_maze.isin([9, 10, 11, 12, 13])]
    if ax1 is None:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5), clear=True)
    ax1.spines[["top", "right"]].set_visible(False)
    sns.pointplot(
        embed_sessions_df,
        x="abbrev",
        y="cv_performance",
        ax=ax1,
    )
    ax1.set_ylim(0, 0.08)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    ax1.set_xlabel("")

    if ax2 is None:
        fig, ax2 = plt.subplots(1, 1, figsize=(10, 5), clear=True)
    ax2.spines[["top", "right"]].set_visible(False)
    sns.pointplot(
        results_df[results_df.abbrev.isin(["Nlat_10", "Nlat_20", "Nlat_50"])],
        x="abbrev",
        y="cv_performance",
        ax=ax2,
    )
    ax2.set_ylim(0, 0.08)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    ax2.set_xlabel("")
    return


def _load_all_input_results(exp_set="all_inputs", day_range=(7, 13)):
    """ """
    results_dfs = []
    for subject_ID in SUBJECT_IDS:
        exp_name = f"{subject_ID}_all_inputs"
        results_dfs.append(_load_results_df(exp_name, exp_set, abbrev="all_inputs"))
    results_df = pd.concat(results_dfs, axis=0).reset_index(drop=True)
    if day_range:
        results_df = results_df[results_df.day_on_maze.between(*day_range)]
    return results_df


def plot_all_input_results(
    ax=None,
    compare_to="exp_state-action_distance_linear",
):
    all_input_results_df = _load_all_input_results()
    compare_to_df = _load_state_action_distance_interaction_results()
    compare_to_df = compare_to_df[compare_to_df.abbrev == compare_to]
    combined_df = pd.concat([all_input_results_df, compare_to_df], axis=0).reset_index(drop=True)
    x_order = [compare_to, "all_inputs"]
    combined_df["abbrev"] = pd.Categorical(combined_df["abbrev"], x_order)
    # plotting
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), clear=True)
    ax.spines[["top", "right"]].set_visible(False)
    sns.pointplot(
        combined_df,
        x="abbrev",
        y="cv_performance",
        hue="subject_ID",
        dodge=0.1,
        linestyle="-",
        markeredgewidth=0,
        markersize=5,
        err_kws={"linewidth": 2},
        ax=ax,
        legend=False,
    )
    ax.set_ylim(0, 0.16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("")


# %%
