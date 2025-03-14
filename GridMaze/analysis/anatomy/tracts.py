"""
Library for visualising the silicon probe shank tracts in Allen CCF Space
@peterdoohan
"""

# %% Imports
import json
import numpy as np
from matplotlib import pyplot as plt
import brainglobe_heatmap as bgh
import matplotlib as mpl

from GridMaze.analysis.core import load_data

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

ALLEN_ATLAS_RESOLUTION = 25  # um (10, 25, 50)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as infile:
    SUBJECT_IDS = json.load(infile)


# %% Functions


def plot_probe_tracks_2D(axes=None, show_contacts=False):
    values = dict(PL=0, ACAd=1, ACAv=1)  # scalar values for each region
    # build custom CMAP:
    custom_cmap = mpl.colors.LinearSegmentedColormap.from_list("custom_cmap", ["violet", "lightcyan"], N=2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), width_ratios=[1, 1.3])
    scene1 = bgh.Heatmap(
        values,
        position=(3000, 0, 0),
        orientation="frontal",  # or 'sagittal', or 'horizontal' or a tuple (x,y,z)
        thickness=10,
        title="",
        hemisphere="left",
        cmap=custom_cmap,
        format="2D",
    ).plot_subplot(fig, axes[0])

    scene2 = bgh.Heatmap(
        values,
        position=(0, 0, 6000),
        orientation="sagittal",  # or 'sagittal', or 'horizontal' or a tuple (x,y,z)
        thickness=10,
        title="",
        hemisphere="left",
        cmap=custom_cmap,
        format="2D",
    ).plot_subplot(fig, axes[1])

    for ax in axes:
        ax.axis("off")

    colorbar1 = fig.axes[-1]
    colorbar1.remove()
    colorbar2 = fig.axes[-1]
    colorbar2.remove()

    axes[1].legend(loc="lower left", bbox_to_anchor=(1.05, 0.5), ncol=1, fontsize=1)

    # plot shanks
    for subject in SUBJECT_IDS:
        probe_anatomy_df = load_data.load_probe(subject)
        shanks = probe_anatomy_df.contact.shank.unique()
        for shank in shanks:
            shank_df = probe_anatomy_df[probe_anatomy_df.contact.shank == shank]
            contact_voxels = np.array(shank_df.voxel.values)
            # plot idealised shanks
            shank_um = extend_line(contact_voxels, 00, num_points=100) * ALLEN_ATLAS_RESOLUTION
            axes[0].plot(shank_um[:, 2], shank_um[:, 1], lw=0.5, color="black", alpha=0.5)
            axes[1].plot(shank_um[:, 0], shank_um[:, 1], lw=0.5, color="black", alpha=0.5)
            # plot contacts
            if show_contacts:
                contacts_um = contact_voxels * ALLEN_ATLAS_RESOLUTION
                axes[0].scatter(
                    contacts_um[:, 2],
                    contacts_um[:, 1],
                    s=5,
                    color="red",
                )
                axes[1].scatter(
                    contacts_um[:, 0],
                    contacts_um[:, 1],
                    s=5,
                    color="red",
                )
    fig.tight_layout()


# %%


def extend_line(points, new_y_min, num_points=100):
    """
    Given a numpy array `points` of shape (n, 3) representing (x, y, z) coordinates,
    this function fits a line to these points and returns a new array of points along
    the best-fit line, with y-values spanning from `new_y_min` to the original maximum y.

    Parameters:
      - points: np.array, shape (n, 3)
      - new_y_min: float, the new minimum y value to extend the line to
      - num_points: int, the number of points to generate along the extended line

    Returns:
      - extended_points: np.array of shape (num_points, 3)
    """
    # Compute the centroid of the points
    centroid = np.mean(points, axis=0)

    # Center the points and perform SVD to extract the principal component
    pts_centered = points - centroid
    _, _, Vt = np.linalg.svd(pts_centered)
    direction = Vt[0]  # First principal component gives the line direction

    # Ensure that the y component of the direction is nonzero to avoid division by zero
    if np.isclose(direction[1], 0):
        raise ValueError("The y component of the line direction is zero; cannot extend along y.")

    # For the line, the y coordinate is: y = centroid[1] + t * direction[1]
    # Solve for t for a given y: t = (y - centroid[1]) / direction[1]
    t_y_min = (new_y_min - centroid[1]) / direction[1]
    t_y_max = (points[:, 1].max() - centroid[1]) / direction[1]

    # Sort the t-values so that t_min corresponds to the new minimum y and t_max to the original max y
    t_min, t_max = sorted([t_y_min, t_y_max])

    # Generate a series of t values between t_min and t_max
    t_values = np.linspace(t_min, t_max, num_points)

    # Compute the corresponding points along the extended line
    extended_points = centroid + np.outer(t_values, direction)

    return extended_points
