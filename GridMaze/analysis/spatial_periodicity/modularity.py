"""This script is for testing if the cells with spaital periodicity fit into modules with
multimodal frequency distributions"""
# %% Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .. import get_sessions as gs

# %% Global varaibles

# %% Functions


def filter_maze_periodicity_df(
    maze_periodicity_df,
    fit_r2_cutoff=0.9,
    spatial_correlation_cutoff=0.3,
    freq_range=(0.05, 2),
    amp_range=(0.1, 0.5),
):
    filter_conditions = np.logical_and.reduce(
        [
            maze_periodicity_df.fit_params.r2 > fit_r2_cutoff,
            maze_periodicity_df.spatial_correlation > spatial_correlation_cutoff,
            maze_periodicity_df.fit_params.freq > freq_range[0],
            maze_periodicity_df.fit_params.freq < freq_range[1],
            maze_periodicity_df.fit_params.sin_scale > amp_range[0],
            maze_periodicity_df.fit_params.sin_scale < amp_range[1],
        ]
    )
    return maze_periodicity_df[filter_conditions]


def plot_amplitude_frequency_2d_hist(params_df, ax, freq_range=(0.05, 2)):
    sns.histplot(data=params_df, x="freq", y="sin_scale", bins=50, ax=ax, cbar=True)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(freq_range)
    ax.set_ylim(0, 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return


def get_multisession_maze_periodicity_df(maze_number):
    maze_number = [maze_number] if not maze_number == "all" else "all"
    sessions = gs.get_sessions(
        subject_IDs="all",
        maze_number=maze_number,
        day_on_maze="late",
        with_data=["spatial_periodicity_df"],
    )
    multisession_maze_periodicity_df = pd.concat([s.spatial_periodicity_df for s in sessions], axis=0)
    return multisession_maze_periodicity_df


def plot_population_gridyness_summary(multisession_maze_periodicity_df):
    f, axes = plt.subplots(3, 3, figsize=(20, 20), clear=True)
    f.tight_layout()
    for i, fit_r2_cutoff in enumerate(np.arange(0.3, 0.99, 0.3)):
        for j, spatial_corr_cutoff in enumerate(np.arange(0.2, 0.61, 0.2)):
            filtered_df = filter_maze_periodicity_df(
                multisession_maze_periodicity_df,
                fit_r2_cutoff=fit_r2_cutoff,
                spatial_correlation_cutoff=spatial_corr_cutoff,
            )
            ax = axes[i, j]
            plot_amplitude_frequency_2d_hist(filtered_df.fit_params, ax)
            ax.text(
                0.95,
                0.95,
                f"fit r2 > {fit_r2_cutoff:.2f}\n spatial r > {spatial_corr_cutoff:.2f}",
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=12,
                color="black",
            )

    return


def plot_most_periodic_cells(maze_periodicity_df):
    filt_df = filter_maze_periodicity_df(
        maze_periodicity_df,
        fit_r2_cutoff=0.95,
        spatial_correlation_cutoff=0.7,
        freq_range=(0.1, 1.5),
        amp_range=(0.07, 0.5),
    )
    for cluster in filt_df.index:
        f, ax = plt.subplots(1, 1, figsize=(5, 5), clear=True)
        distance_corrs = filt_df.loc[cluster].distance_correlations
        distances = np.arange(1, len(distance_corrs) + 1)
        ax.plot(distances, distance_corrs, color="k", lw=1, alpha=1)
        ax.set_xlabel("Maze Distance")
        ax.set_ylabel("Correlation")
        ax.set_ylim(-1, 1)
        ax.axhline(0, color="silver", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return
