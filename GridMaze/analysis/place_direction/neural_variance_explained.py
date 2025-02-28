"""This module if for analysing the dimensionality of neural place-direction tuning"""
# %% Imports
import json
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import sem
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from . import get_neural_place_direction_df as npd

# %% Global variables
with open("../data/experiment_info.json") as input_file:
    EXP_INFO = json.load(input_file)


# %% Functions
def plot_summary_variance_explained_analysis(maze_number, n_permuted):
    subject2auc = variance_explained_random_effects_analysis(maze_number, n_permuted=n_permuted, plot=False)
    condition2var_exp = variance_explained_fixed_effects_analysis(maze_number, n_permuted=n_permuted, plot=False)
    # plotting
    fig = plt.figure(figsize=(8, 5), clear=True)
    fig.tight_layout()
    gs = GridSpec(3, 5, figure=fig, wspace=0.5, hspace=0.5)
    ax1 = fig.add_subplot(gs[:3, :3])  # fixed effects analysis
    ax1_inset = inset_axes(ax1, width="25%", height="45%", loc="lower right")  #  w/ random effects inset
    ax2 = fig.add_subplot(gs[0, 3])
    ax3 = fig.add_subplot(gs[0, 4])
    ax4 = fig.add_subplot(gs[1, 3])
    ax5 = fig.add_subplot(gs[1, 4])
    ax6 = fig.add_subplot(gs[2, 3])
    ax7 = fig.add_subplot(gs[2, 4])
    # fixed effects
    x_components = np.arange(0, len(condition2var_exp["true"]) + 1)
    ax1.plot(
        x_components,
        np.concatenate(([0], condition2var_exp["true"])),
        label="Neural Place-Direction",
        color="red",
        lw=2,
    )
    ax1.plot(
        x_components,
        np.concatenate(([0], condition2var_exp["permuted"])),
        label="Permuted Place-Direction",
        color="black",
        lw=2,
    )
    ax1.fill_between(
        x_components,
        np.concatenate(([0], condition2var_exp["permuted_95"][0])),
        np.concatenate(([0], condition2var_exp["permuted_95"][1])),
        color="black",
        alpha=0.5,
    )
    ax1.plot([1, len(x_components) + 1], [0, 1], color="silver", ls="--")
    ax1.set_xlabel("Number of Components")
    ax1.set_ylabel("Variance Explained")
    ax1.legend(loc="upper left", fontsize="small")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    # random effects
    delta_aucs = [subject2auc[s]["delta"] for s in subject2auc.keys()]
    ax1_inset.axhline(0, color="silver", ls="--")
    sns.stripplot(y=delta_aucs, size=8, color="thistle", edgecolor="white", linewidth=0.5, ax=ax1_inset)
    ax1_inset.set_ylim(-np.std(delta_aucs), max(delta_aucs) + 2 * np.std(delta_aucs))
    ax1_inset.set_ylabel("\u0394 AUC")
    add_ttest_results_to_plot(delta_aucs, 0, ax1_inset)
    # subject null distributions
    for ax, subject in zip([ax2, ax3, ax4, ax5, ax6, ax7], subject2auc.keys()):
        ax.hist(subject2auc[subject]["null_distribution"], bins=100, color="black")
        ax.axvline(subject2auc[subject]["true"], color="red")
        ax.set_title(subject, pad=-100)
        ax.set_ylabel("Count")
        if subject in ["m7", "m8"]:  # only plot booton ax labels
            ax.set_xlabel("AUC")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return


def add_ttest_results_to_plot(values, different_from, axis):
    t_stat, p_value = stats.ttest_1samp(values, different_from)
    if p_value > 0.05:
        symbol = "n.s."
    elif p_value > 0.01:
        symbol = "*"
    elif p_value > 0.001:
        symbol = "**"
    else:
        symbol = "***"
    axis.text(
        0.5,
        0.9,
        f"{symbol}",
        size="large",
        transform=axis.transAxes,
        ha="center",
        va="center",
    )


def variance_explained_random_effects_analysis(maze_number, n_permuted=100, plot=True):
    sessions = npd.get_analysis_sessions(maze_number, subject="all", late=True)
    neural_place_direction_df = npd.get_neural_place_direction_df(
        sessions, normalisation_method="length", n_permuted=False
    )
    permuted_neural_place_direction_dfs = npd.load_permuted_neural_place_direction_dfs(
        maze_number, n_permuted=n_permuted
    )
    fig, axes = plt.subplots(3, 2, figsize=(6, 9), clear=True)  # fig for subject null distributions
    all_neural_data = neural_place_direction_df.to_numpy()
    delta_aucs = []
    subject2auc = {}
    for ax, subject in zip(axes.flatten(), EXP_INFO["subject_IDs"]):
        # get true AUC value
        subject_neural_df = neural_place_direction_df.loc[neural_place_direction_df.index.str.contains(subject)]
        subject_data = subject_neural_df.to_numpy()
        var_exp = get_variance_explained(all_neural_data, subject_data)
        auc = np.trapz(var_exp)
        # get null distribution of AUCs
        subject_permuted_auc = []
        for permuted_df in permuted_neural_place_direction_dfs:  # each df has data from all subjects
            subject_permuted_df = permuted_df.loc[permuted_df.index.str.contains(subject)]
            subject_data = subject_permuted_df.to_numpy()
            all_permuted_data = permuted_df.to_numpy()
            p_var_exp = get_variance_explained(all_permuted_data, subject_data)
            p_auc = np.trapz(p_var_exp)
            subject_permuted_auc.append(p_auc)
        median_p_auc = np.mean(subject_permuted_auc)
        delta_auc = auc - median_p_auc
        delta_aucs.append(delta_auc)
        subject2auc[subject] = {"true": auc, "null_distribution": subject_permuted_auc, "delta": delta_auc}
        # plotting
        if plot:
            ax.hist(subject_permuted_auc, bins=100, color="black")
            ax.axvline(auc, color="red")
            ax.set_title(subject)
            ax.set_xlabel("AUC")
            ax.set_ylabel("Count")
    if not plot:
        plt.close(fig)
    return subject2auc


def variance_explained_fixed_effects_analysis(maze_number, n_permuted, plot=True):
    sessions = npd.get_analysis_sessions(maze_number, subject="all", late=True)
    neural_place_direction_df = npd.get_neural_place_direction_df(
        sessions, normalisation_method="length", n_permuted=False
    )
    permuted_neural_place_direction_dfs = npd.load_permuted_neural_place_direction_dfs(
        maze_number, n_permuted=n_permuted
    )
    N = neural_place_direction_df.to_numpy()
    var_exp = get_variance_explained(N, N)
    permuted_var_exps = []
    for permuted_df in permuted_neural_place_direction_dfs:
        P = permuted_df.to_numpy()
        p_var_exp = get_variance_explained(P, P)
        permuted_var_exps.append(p_var_exp)
    permuted_var_exps = np.vstack(permuted_var_exps)
    av_permuted_var_exp = permuted_var_exps.mean(axis=0)
    lower_95 = np.percentile(permuted_var_exps, 2.5, axis=0)
    upper_95 = np.percentile(permuted_var_exps, 97.5, axis=0)
    # plotting
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), clear=True)
        ax.plot(var_exp, label="Neural Place-Direction", color="red", lw=2)
        ax.plot(av_permuted_var_exp, label="Permuted Neural Place-Direction", color="black", lw=2)
        ax.fill_between(
            np.arange(len(av_permuted_var_exp)),
            lower_95,
            upper_95,
            color="black",
            alpha=0.5,
        )
    return {"true": var_exp, "permuted": av_permuted_var_exp, "permuted_95": (lower_95, upper_95)}


# %% Supporting functions


def get_variance_explained(A, B):  # A & B: [n_samples, n_features]
    """Calculates the cumulative variance of matrix B that's explained by the first i components of matrix A."""
    model = PCA(random_state=0)
    model.fit(A)
    M = model.transform(B)  # [n_samples, n_components]
    pc_exp_var = np.square(M).sum(axis=0)
    cumsum_exp_var = np.cumsum(pc_exp_var)
    return cumsum_exp_var
