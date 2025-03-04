"""
This library contains preprocessing functions that load preprocessed histology data that has been aligned
to the Allen mouse brain atlas (CCF coordinates) with HERBS (https://github.com/JingyiGF/HERBS), and uses the AllenSDK
API to convert the voxel coordinates of each site on the Cabridge Neurotech probe to anatomical information 
(structure ID, acronym, name, rgb_triplet).

Note: this code must be run in an environment with allenSDK installed and an internet connection to download the Allen
mouse brain atlas data.

@peterdoohan
"""

# %% Import
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import shutil
import ast
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from probeinterface import Probe, get_probe
from brainrender import Scene
from brainrender.actors import Points
from datetime import date
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
import matplotlib.colors as mcolors

import vedo

vedo.settings.default_backend = "vtk"

# %% Globs
from GridMaze.paths import PREPROCESSED_DATA_PATH, EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

HERBS_DATA_FOLDER = PREPROCESSED_DATA_PATH / "HERBS"  # folder containing subject_ID/proble.pkl files

ALLEN_ATLAS_RESOLUTION = 25  # um (10, 25 or 50)

HERBS_ATLAS_RESOLUTION = 25  # um
# HERBS_ATLAS_SHAPE = (1140, 1320, 800)  # (z,x,y) voxels for 10 um atals
HERBS_ATLAS_SHAPE = (456, 528, 320)  # for 25 um atlas

# Load Allen anatomy objects from web
# this will create a new folder coe/mouse_connectivity
# remove this folder at the end of preprocessing with fn x
MCC = MouseConnectivityCache(
    resolution=ALLEN_ATLAS_RESOLUTION, manifest_file=str(EXPERIMENT_INFO_PATH / "allensdk/manifest.json")
)
ANNOTATION_VOLUME, _ = MCC.get_annotation_volume()
STRUCTURE_TREE = MCC.get_structure_tree()

SHANK_NO2SHANK_ID = {
    1: "A",  # most posterior shank
    2: "B",
    3: "C",
    4: "D",
    5: "E",
    6: "F",  # most anterior shank
}

SHANK_SPACING = 200  # um

SHANK_ID2SHANK_NO = {v: k for k, v in SHANK_NO2SHANK_ID.items()}

# load probe interface probe object corresponding to the probe used in this exp
PROBE = get_probe(manufacturer="cambridgeneurotech", probe_name="ASSY-236-F")
PROBE.wiring_to_device("cambridgeneurotech_mini-amp-64")

# load probe depth information
PROBE_DEPTHS_DF = pd.read_csv(EXPERIMENT_INFO_PATH / "probe_depths.htsv", sep="\t")

REGION2COLOR = {
    "ACAd5": "deepskyblue",
    "ACAd6a": "deepskyblue",
    "ACAv6a": "deepskyblue",
    "ACAv5": "deepskyblue",
    "PL5": "magenta",
    "PL6a": "magenta",
    "PL2/3": "magenta",
    "MOs5": "lime",
    "ORBm2/3": "yellow",
    "ILA2/3": "red",
    "ILA5": "red",
}

# %% ProbeFit Class


class ProbeFit:
    def __init__(self, subject, n_shanks=6, verbose=False):
        self.subject = subject
        self.n_shanks = n_shanks
        self.probe_df = PROBE.to_dataframe()
        self.contact_ids = self.probe_df.contact_ids.values.astype(int)
        # load tract data available for subject
        self.shank_id2tract = load_subject_shanks(subject, n_shanks, verbose)
        self.tracts = list(self.shank_id2tract.values())
        # get best fit of full probe location given tract data, see called fn for details
        (
            self.fit_tracts,
            (self.plane_centroid, self.plane_normal, self.plane_u, self.plane_v),
            self.new_centers,
            self.common_direction,
            self.spacing_direction,
        ) = fit_tracts_evenly_spaced_with_missing(self.tracts)
        self.brain_surface_depths = self.get_brain_surface_depths()
        self.brain_surface_depth = np.max(self.brain_surface_depths)

    def get_contact_anatomy_info(self, _date, contact_id):
        """
        Returns dict of anatomical info for a given contact on a given shank on a given day
        during the expderiment (main function of the class)
        """
        # get probe depth for subject on date
        contact_info = self.probe_df.query(f"contact_ids=='{contact_id}'").iloc[0].to_dict()
        shank_no = int(contact_info["shank_ids"])
        probe_depth = self.get_probe_depth(_date)  # um
        probe_depth_v = probe_depth / HERBS_ATLAS_RESOLUTION  # vox
        # get voxel y coordinate of top of the brain (where probe depth is defined from)
        # brain_surface_y = self.brain_surface_depths[shank_no - 1]
        brain_surface_y = self.brain_surface_depth
        y = probe_depth_v + brain_surface_y
        probe_tip_vox = self.query_voxel(shank_no, y)
        # get contact offset relative to probe tip (in voxels), correct towards origin (-)
        x_offset = (contact_info["x"] - shank_no * SHANK_SPACING) / HERBS_ATLAS_RESOLUTION
        y_offset = contact_info["y"] / HERBS_ATLAS_RESOLUTION
        contact_vox = probe_tip_vox - np.array([x_offset, y_offset, 0])  # -6
        contact_vox = contact_vox.astype(int)
        vol_id = ANNOTATION_VOLUME[contact_vox[0], contact_vox[1], contact_vox[2]]
        structure_info = STRUCTURE_TREE.get_structures_by_id([vol_id])[0]
        return contact_vox, structure_info

    def plot_anatomical_regions(self, _date, ax=None):
        """ """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax.axis("off")
        shank_labled = np.zeros(self.n_shanks).astype(bool)
        for contact_id in self.contact_ids:
            contact_info = self.probe_df.query(f"contact_ids=='{contact_id}'").iloc[0].to_dict()
            shank_no = int(contact_info["shank_ids"])
            if not shank_labled[shank_no - 1]:
                shank_id = SHANK_NO2SHANK_ID[shank_no + 1]
                shank_x = shank_no * SHANK_SPACING
                ax.text(shank_x, -15, shank_id, fontsize=12, ha="center", va="center")
                shank_labled[shank_no - 1] = True
            x = contact_info["x"]
            y = contact_info["y"]
            _, strucutre_info = self.get_contact_anatomy_info(_date, contact_id)
            acroynm = strucutre_info["acronym"]
            color = REGION2COLOR[acroynm]
            ax.scatter(x, y, color=color, marker="s", s=50)
        return ax

    def plot_reconstructed_probe_tracts(
        self, full_brain=True, plot_original_tracts=True, plot_contacts_on_day=False, visualise_regions=["PL"]
    ):
        # initialise brainrender scene
        scene = Scene(title="", root=full_brain)
        scene.plotter.axes = False
        for region in visualise_regions:
            scene.add_brain_region(region, alpha=0.15, hemisphere="left", color="magenta", silhouette=False)
        # add probe tracts
        # load (n_sites in the brain, 3) np.array of voxel coordinates
        if plot_original_tracts:
            for t in self.tracts:
                if t is None:
                    continue
                probe_track_um = t * HERBS_ATLAS_RESOLUTION  # convert to um
                scene.add(Points(probe_track_um, colors="grey"))  # color=color
        for ft in self.fit_tracts:
            if ft is None:
                continue
            probe_track_um = ft * HERBS_ATLAS_RESOLUTION
            scene.add(Points(probe_track_um, colors="green"))
        if plot_contacts_on_day:
            contact_voxs = []
            for contact_id in self.contact_ids:
                contact_vox, _ = self.get_contact_anatomy_info(plot_contacts_on_day, contact_id)
                contact_voxs.append(contact_vox)
            contact_voxs = np.array(contact_voxs) * HERBS_ATLAS_RESOLUTION
            scene.add(Points(contact_voxs, colors="red", radius=40))
        scene.render()
        return scene

    def get_brain_surface_depths(self):
        surface_ts = []
        for s in range(self.n_shanks):
            t = 0
            found_brain = False
            while not found_brain:
                voxel = self.query_voxel(s, t)
                vol = ANNOTATION_VOLUME[voxel[0], voxel[1], voxel[2]]
                if vol != 0:  # 0 = outside of brain accoring to AllenSDK
                    found_brain = True
                    surface_ts.append(t)
                else:
                    t += 1
        return surface_ts

    def get_probe_depth(self, _date):
        """
        Returns the most recent probe depth measurement for a given subject on or before a given date.
        date should be datetime object
        """
        # Make a copy and ensure the date column is datetime
        df = PROBE_DEPTHS_DF.copy()
        df["date"] = df.date.apply(date.fromisoformat)

        # Filter for the subject and for measurements on or before the provided date
        subject_df = df[(df["subject"] == self.subject) & (df["date"] <= _date)]

        if subject_df.empty:
            raise ValueError(
                f"No probe depth data available for subject {self.subject} on or before {_date.toisoformat()}"
            )

        # Get the row with the latest date (i.e. the most recent measurement)
        latest_row = subject_df.sort_values("date", ascending=False).iloc[0]

        return latest_row["probe_depth"]

    def query_voxel(self, shank_no, global_depth):
        """
        Returns the voxel coordinate along the probe track for a given shank such that
        the returned voxel's y-coordinate is equal to the provided global_depth. The
        computation is done by first finding the point on the track that lies at y=0,
        and then moving along the track until the y-coordinate reaches global_depth.

        Parameters:
        shank_no (int): The 1-indexed shank number whose track is queried.
        global_depth (float): The desired global y-coordinate (in voxel units) in the atlas space,
                                where y=0 is the reference (e.g., top of the brain).

        Returns:
        np.ndarray: A 3-element integer array representing the voxel coordinates (x, y, z)
                    at the specified global depth along the shank's track.
        """
        # Get the computed center for the specified shank.
        center = self.new_centers[shank_no - 1]

        # Ensure that the common direction has a significant y-component.
        if abs(self.common_direction[1]) < 1e-8:
            raise ValueError("common_direction has negligible y component; cannot determine depth mapping.")

        # Step 1: Find the point on the track that has a y-coordinate of 0.
        # Solve: center[1] + t0 * common_direction[1] = 0  =>  t0 = -center[1] / common_direction[1]
        t0 = -center[1] / self.common_direction[1]
        point_y0 = center + t0 * self.common_direction  # This point has y ~ 0.

        # Step 2: From this y=0 point, move along the track until the y-coordinate equals global_depth.
        # Since point_y0[1] is 0, we need t_depth such that:
        #     0 + t_depth * common_direction[1] = global_depth  =>  t_depth = global_depth / common_direction[1]
        t_depth = global_depth / self.common_direction[1]
        point = point_y0 + t_depth * self.common_direction

        # Convert to discrete voxel coordinates using rounding.
        voxel = np.rint(point).astype(int)
        return voxel


# %% maths


def best_fit_plane(points):
    """
    Computes the best-fit plane for a set of 3D points using SVD.
    Returns the plane's centroid, its normal, and two orthonormal basis vectors spanning the plane.
    """
    centroid = np.mean(points, axis=0)
    pts_centered = points - centroid
    # SVD: the last singular vector is the plane normal.
    _, _, Vt = np.linalg.svd(pts_centered)
    normal = Vt[-1]
    # The first two singular vectors span the plane.
    u = Vt[0]
    v = Vt[1]
    return centroid, normal, u, v


def project_point_onto_plane(point, plane_point, plane_normal):
    """
    Projects a point onto a plane defined by plane_point and plane_normal.
    """
    return point - np.dot(point - plane_point, plane_normal) * plane_normal


def best_fit_line(points):
    """
    Computes the best-fit line for a set of 3D points using SVD.
    Returns the line's centroid and its primary direction.
    """
    pts = np.array(points)
    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid
    _, _, Vt = np.linalg.svd(pts_centered)
    direction = Vt[0]
    return centroid, direction


def compute_common_direction(tracts, plane_normal):
    """
    For each known tract, compute its best-fit line direction,
    project it onto the plane, and then average (after aligning signs)
    to get a common direction.
    """
    directions = []
    for tract in tracts:
        _, d = best_fit_line(tract)
        # Project d onto the plane.
        d_proj = d - np.dot(d, plane_normal) * plane_normal
        norm = np.linalg.norm(d_proj)
        if norm < 1e-8:
            continue
        d_proj /= norm
        directions.append(d_proj)
    if not directions:
        raise ValueError("No valid tract directions found.")
    # Align directions so that they all point roughly the same way.
    ref = directions[0]
    for i in range(len(directions)):
        if np.dot(directions[i], ref) < 0:
            directions[i] = -directions[i]
    common_direction = np.mean(directions, axis=0)
    common_direction /= np.linalg.norm(common_direction)
    return common_direction


def fit_tracts_evenly_spaced_with_missing(tracts, expected_count=6, num_points=200, t_range=(30, 130)):
    """
    Given an ordered list of voxel tracts (length == expected_count) where missing tracts are None,
    this function:
      1. Uses only the known (non-None) tracts to compute the overall best-fit plane.
      2. Computes a common in-plane (tract) direction from the known data.
      3. Computes the perpendicular in-plane (spacing) direction.
      4. For each known tract, computes its projected center in the plane and expresses it in
         2D coordinates (a, b) relative to (common_direction, spacing_direction).
      5. Uses the known shank indices and their b-values to fit a line so that b for all expected
         shanks (indices 0 to expected_count-1) can be determined.
      6. Computes the new center for every expected shank as:
             new_center = plane_centroid + (new_a)*common_direction + (new_b)*spacing_direction,
         where we fix new_a to the mean of the known a-values.
      7. Computes the global extent along common_direction (using all known points).
      8. Generates a new tract for each expected shank as a straight line along common_direction,
         spanning the global t-range (centered about zero).

    Returns:
      new_tracts: List (length expected_count) of new tracts (as numpy arrays) for each shank.
      plane_info: Tuple (plane_centroid, plane_normal, plane_u, plane_v)
      new_centers: The computed new centers (3D points) for each shank.
      common_direction: The common in-plane direction.
      spacing_direction: The in-plane spacing direction.
    """
    # --- Step 1: Separate known tracts.
    known_indices = [i for i, tract in enumerate(tracts) if tract is not None]
    if len(known_indices) < 2:
        raise ValueError("At least two tracts are required to determine spacing.")
    known_tracts = [tract for tract in tracts if tract is not None]

    # Combine all points from known tracts to compute the best-fit plane.
    all_points = np.concatenate(known_tracts, axis=0)
    plane_centroid, plane_normal, plane_u, plane_v = best_fit_plane(all_points)

    # --- Step 2: Compute the common in-plane direction.
    common_direction = compute_common_direction(known_tracts, plane_normal)
    # --- Step 3: The spacing direction is perpendicular to common_direction in the plane.
    spacing_direction = np.cross(plane_normal, common_direction)
    spacing_direction /= np.linalg.norm(spacing_direction)

    # --- Step 4: For each known tract, compute its projected center in the plane
    # and then its 2D coordinates (a, b).
    known_a = []
    known_b = []
    for idx in known_indices:
        tract = tracts[idx]
        cent, _ = best_fit_line(tract)
        proj_center = project_point_onto_plane(cent, plane_centroid, plane_normal)
        diff = proj_center - plane_centroid
        a_val = np.dot(diff, common_direction)
        b_val = np.dot(diff, spacing_direction)
        known_a.append(a_val)
        known_b.append(b_val)
    known_a = np.array(known_a)
    known_b = np.array(known_b)

    # --- Step 5: Fit a line for b vs. shank index.
    # We assume that the expected indices are 0,...,expected_count-1.
    # Fit using the known indices.
    poly_coeffs = np.polyfit(known_indices, known_b, 1)  # b = m*i + c
    m, c = poly_coeffs
    # Compute new b for each expected shank.
    new_bs = np.array([m * i + c for i in range(expected_count)])

    # --- Step 6: Use the mean of known a-values for all new centers.
    new_a = np.mean(known_a)
    # Compute new centers (in 3D) for each expected shank.
    new_centers = []
    for i in range(expected_count):
        center_i = plane_centroid + new_a * common_direction + new_bs[i] * spacing_direction
        new_centers.append(center_i)
    new_centers = np.array(new_centers)

    # --- Step 7: Generate new tracts using global voxel depths.
    # For each shank, we want to define the tract so that it spans a specified range of global y-values.
    # Using the same logic as query_voxel:
    #   For a given shank, let center = new_center.
    #   Find t0 such that: center[1] + t0 * common_direction[1] = 0.
    #   Then the point at global depth D is: point_y0 + (D / common_direction[1]) * common_direction.
    if t_range is not None:
        if not isinstance(t_range, (list, tuple)) or len(t_range) != 2:
            raise ValueError("t_range must be a tuple (start, end) representing global voxel depths.")
        global_depths = np.linspace(t_range[0], t_range[1], num_points)
    else:
        # Fallback: compute a range based on the known data (not necessarily global)
        T_vals = np.dot(all_points - plane_centroid, common_direction)
        global_depths = np.linspace(np.min(T_vals), np.max(T_vals), num_points)

    new_tracts = []
    for center in new_centers:
        # Compute offset t0 so that the point on the line with parameter t0 has y==0.
        t0 = -center[1] / common_direction[1]
        point_y0 = center + t0 * common_direction  # This point has y approximately 0.
        # For each desired global voxel depth D, compute:
        #   t = D / common_direction[1]    (so that y offset = D)
        # and then the corresponding point is point_y0 + t * common_direction.
        tract_points = point_y0 + np.outer(global_depths / common_direction[1], common_direction)
        new_tracts.append(tract_points)

    return (
        new_tracts,
        (plane_centroid, plane_normal, plane_u, plane_v),
        new_centers,
        common_direction,
        spacing_direction,
    )


# %%


def load_subject_shanks(subject="m2", n_shanks=6, verbose=False):
    """ """
    subject_folder = HERBS_DATA_FOLDER / subject
    # find all  HERBS output probe files
    shank_id2probe_coords = {}
    for i in range(n_shanks):
        shank_id = SHANK_NO2SHANK_ID[i + 1]
        probe_path = subject_folder / f"shank_{i+1}.probe.pkl"
        if probe_path.exists():
            shank_id2probe_coords[shank_id] = get_probe_tract(probe_path)
        else:
            if verbose:
                print(f"{subject}, missing shank {i+1}")
            shank_id2probe_coords[shank_id] = None
    return shank_id2probe_coords


def get_probe_tract(HERBS_probe_path):
    """
    Load probe tract from HERBS output probe.pkl and return as numpy array
    of shape (n_probe_sites, 3[x,y,z])
    Note in this case probe_sites are arbitrary and they just define a line in 3D CCF space
    """
    with open(HERBS_probe_path, "rb") as file:
        HERBS_data = pickle.load(file)
    probe_coords = HERBS_data["data"]["sites_vox"][0]  # voxel coordinates (z,y,x) origin bottom left of left cerebellum
    z, x, y = probe_coords.T
    # translate to origin top right of right olfactory bulb (Allen standard)
    z = HERBS_ATLAS_SHAPE[0] - z
    y = HERBS_ATLAS_SHAPE[2] - y
    x = HERBS_ATLAS_SHAPE[1] - x
    # remap origin
    probe_coords = np.array([x, y, z]).T
    # for some reson HERBS output only lists a voxel for every second site, assume adjacent sites lie in the same voxel
    probe_coords = np.repeat(probe_coords, repeats=2, axis=0)
    return probe_coords.astype(int)


def _get_probe_move_dates():
    """Get dates when all subjects probes were moved.
    Ignore speciall instances where a single subjects probe was moved"""
    probe_move_dates = []
    for _date in PROBE_DEPTHS_DF.date.unique():
        if len(np.setdiff1d(SUBJECT_IDS, PROBE_DEPTHS_DF[PROBE_DEPTHS_DF.date == _date].subject.values)) == 0:
            probe_move_dates.append(_date)
    return [date.fromisoformat(str(_date)) for _date in probe_move_dates]


# %% QC


def plot_all_probes():
    scene = Scene(title="", root=True)
    scene.plotter.axes = False
    scene.add_brain_region("PL", alpha=0.15, hemisphere="left", color="magenta", silhouette=False)
    scene.add_brain_region("ACAd", alpha=0.15, hemisphere="left", color="deepskyblue", silhouette=False)
    scene.add_brain_region("ACAv", alpha=0.15, hemisphere="left", color="deepskyblue", silhouette=False)
    # add probe tracts
    # load (n_sites in the brain, 3) np.array of voxel coordinates
    # colors = ["red", "blue", "green", "yellow", "purple", "orange"]
    cmap = plt.cm.nipy_spectral
    colors = cmap(np.linspace(0, 1, len(SUBJECT_IDS)))
    colors = [to_hex(color) for color in colors]
    for i, subject in enumerate(SUBJECT_IDS):
        pf = ProbeFit(subject)
        for ft in pf.fit_tracts:
            probe_track_um = ft * HERBS_ATLAS_RESOLUTION
            scene.add(Points(probe_track_um, colors="k", radius=15, alpha=0.5))
    scene.render(
        camera={
            "pos": (-20169, -7298, -28832),
            "viewup": (0, -1, 0),
            "clipping_range": (16955, 58963),
        }
    )
    return scene


def plot_all_anatomical_regions():
    """ """
    PF = ProbeFit("m8")
    probe_move_dates = _get_probe_move_dates()
    for _date in probe_move_dates:
        f, axes = plt.subplots(1, 6, figsize=(2, 4))
        for ax in axes:
            ax.axis("off")
        probe_df = PROBE.to_dataframe()
        for shank in range(0, 6):
            shank_id = SHANK_NO2SHANK_ID[shank + 1]
            shank_x = shank - 5.5
            ax.text(shank_x, -2, shank_id, fontsize=12, ha="center", va="center")
            shank_df = probe_df[probe_df.shank_ids == f"{shank}"]
            regions = []
            for contact_id in shank_df.sort_values("y").contact_ids:
                _, anat_info = PF.get_contact_anatomy_info(_date, contact_id)
                regions.append(anat_info["acronym"])
            colors = [REGION2COLOR[acro] for acro in regions]
            colors_rgba = np.array([mcolors.to_rgba(color) for color in colors])
            colors_rgba = colors_rgba.reshape(-1, 1, 4)
            axes[shank].imshow(colors_rgba, origin="lower")

    return
