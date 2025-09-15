"""Library for plotting speed and acceleration tuning curves."""

# %% Imports
import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from GridMaze.analysis.core import get_clusters as gc

from GridMaze.maze import plotting as mp
from scipy.ndimage import gaussian_filter1d


# %% Global Variables
FRAME_RATE = 60  # Hz
# %% Functions


def plot_session_movement_tuning(session):
    """ """
    navigation_activity_df = session.get_navigation_activity_df(
        type="rates", cluster_kwargs={"single_units": True, "multi_units": False}
    )
    # get movement data
    speeds, tangential_acc = get_movement_tuning_data(navigation_activity_df)
    cluster_unique_IDs = navigation_activity_df.firing_rate.columns.values
    for cluster in cluster_unique_IDs:
        firing_rate = navigation_activity_df.firing_rate[cluster].values
        plot_movement_tuning(speeds, tangential_acc, firing_rate)
    return


## plotting


def plot_acceleration_aligned_activity(
    speed,
    acceleration,
    firing_rate,
    speed_threshold=0.05,
    acc_threshold=2,
    min_frames=5,
    max_frames=1,
    pre_window=60,
    post_window=60,
    axes=None,
):
    """
    Create plots of firing rate and speed aligned to movement bout start/stop times.

    Parameters:
    -----------
    navigation_activity_df : pandas DataFrame
        DataFrame containing speed, acceleration and firing rate data
    test_cluster : object
        Cluster object containing unique ID
    speed_threshold : float
        Speed threshold for movement detection
    acc_threshold : float
        Acceleration threshold for bout detection
    min_frames : int
        Minimum consecutive frames below speed threshold for valid stop
    pre_window : int
        Number of frames before alignment point
    post_window : int
        Number of frames after alignment point
    """

    # Identify movement bouts
    start_indices, stop_indices = identify_movement_bouts(
        speed, acceleration, speed_threshold, acc_threshold, min_frames, max_frames
    )
    indices = {"start": start_indices, "stop": stop_indices}
    if len(start_indices) == 0 and len(stop_indices) == 0:
        print("No valid movement bouts found matching criteria")
        return None

    # Create figure
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(5, 5))

    # Process movement starts
    for i, movement in enumerate(["start", "stop"]):
        if len(start_indices) > 0:
            speed_windows, valid_events = extract_aligned_windows(speed, indices[movement], pre_window, post_window)
            rate_windows, _ = extract_aligned_windows(firing_rate, indices[movement], pre_window, post_window)
            valid_starts = indices[movement][valid_events]

            if len(valid_starts) > 0:
                time = np.arange(-pre_window, post_window) / 60  # Convert to seconds

                # Plot speed on second y-axis
                ax_speed = axes[i].twinx()

                # Plot firing rate
                mean_rate = np.mean(rate_windows, axis=0)
                sem_rate = np.std(rate_windows, axis=0) / np.sqrt(len(valid_starts))
                axes[i].plot(time, mean_rate, "r-", label="Firing rate")
                axes[i].fill_between(time, mean_rate - sem_rate, mean_rate + sem_rate, color="r", alpha=0.2)

                # Plot speed
                mean_speed = np.mean(speed_windows, axis=0)
                sem_speed = np.std(speed_windows, axis=0) / np.sqrt(len(valid_starts))
                ax_speed.plot(time, mean_speed, "gray", label="Speed")
                ax_speed.fill_between(time, mean_speed - sem_speed, mean_speed + sem_speed, color="gray", alpha=0.2)

                axes[i].set_title(f"{movement} (n={len(valid_starts)} bouts)")
                axes[i].set_ylabel("Firing rate (Hz)", color="r")
                ax_speed.set_ylabel("Speed (m/s)", color="gray")
                axes[i].tick_params(axis="y", labelcolor="r")
                ax_speed.tick_params(axis="y", labelcolor="gray")

                # Add legends
                lines1, labels1 = axes[i].get_legend_handles_labels()
                lines2, labels2 = ax_speed.get_legend_handles_labels()
                axes[i].legend(lines1 + lines2, labels1 + labels2, loc="upper right")
                axes[i].axvline(x=0, color="k", linestyle="--", alpha=0.5)
                axes[i].set_xlabel("Time (s)")


def plot_movement_tuning(
    speed,
    tangential_acc,
    firing_rate,
    speed_range=(0, 0.3),
    acc_range=(-3, 3),
    speed_bin_size=0.025,
    acc_bin_size=0.25,
    occupancy_proportion=0.005,
    ax1=None,
):
    """
    Get the acceleration and speed data from the navigation activity dataframe.

    Parameters:
    ----------
    speed: float
        estimate of instantaneous speed in m/s
    tangential_acc: float
        estimated tangential acceleration in m/s^2
    speed_bin_size: float
        bin size for the speed data in m/s.
    acc_bin_size: float
        bin size for the acceleration data in m/s^2.
    occupancy_proportion: float
        Minimum proportion of data for binned averaging (default 0.5%)
    """

    # Get the speed and acceleration data
    movement_df = pd.DataFrame({"firing_rate": firing_rate, "speed": speed, "tangential_acc": tangential_acc})

    # bin the data
    bin_edges = {}
    data = {}
    bin_sizes = [speed_bin_size, acc_bin_size]
    for i, (stat, _range) in enumerate(zip(["speed", "tangential_acc"], [speed_range, acc_range])):

        lower_bound, upper_bound = _range
        stat_bin_edges = np.arange(lower_bound, upper_bound + bin_sizes[i], bin_sizes[i])
        bin_edges.update({stat: stat_bin_edges})
        movement_df[f"{stat}_bin"] = pd.cut(
            movement_df[f"{stat}"], bins=stat_bin_edges, labels=(stat_bin_edges[:-1] + bin_sizes[i] / 2)
        )
        # filter occupancy out
        occupancy = movement_df.groupby(f"{stat}_bin", observed=True).size()
        occ_threshold = int(len(movement_df) * occupancy_proportion)
        valid_bins = occupancy[occupancy >= occ_threshold].index
        valid_data = movement_df[movement_df[f"{stat}_bin"].isin(valid_bins)]
        data.update({stat: valid_data})

    # Plot! but both on the same plot (two axes)

    # Create figure with two x-axes
    if ax1 is None:
        fig, ax1 = plt.subplots(figsize=(2, 2))

    ax2 = ax1.twiny()  # Create second x-axis sharing the same y-axis

    # Plot firing rate vs speed on bottom x-axis
    sns.lineplot(data=data["speed"], x="speed_bin", y="firing_rate", color="royalblue", ax=ax1)

    # Plot firing rate vs acceleration on top x-axis
    sns.lineplot(data=data["tangential_acc"], x="tangential_acc_bin", y="firing_rate", color="gray", ax=ax2)

    # Set labels and title
    ax1.set_xlabel("Speed (m/s)", color="royalblue")
    ax2.set_xlabel("Tang. acc. (m/s²)", color="gray")
    ax1.set_ylabel("Firing rate (Hz)")

    # Color the tick labels to match the lines
    ax1.tick_params(axis="x", colors="royalblue")
    ax2.tick_params(axis="x", colors="gray")


## computing functions


def get_movement_tuning_data(navigation_df, position_smoothing_ms=1000 * 1 / FRAME_RATE, frame_rate=FRAME_RATE):
    """
    Returns speed data and tangential acceleration data,
    computed from gaussian-smoothed position data.
    """
    # Get the speed data
    positions = navigation_df.centroid_position.values
    smoothed_positions = gaussian_filter1d(positions, position_smoothing_ms / 1000 * frame_rate, axis=0)
    velocities = np.gradient(smoothed_positions, axis=0) * frame_rate
    accelerations = np.gradient(velocities, axis=0) * frame_rate
    speeds = np.linalg.norm(velocities, axis=1)
    ## Calculate tangential acceleration via finite differences on smoothed speed.
    # Compute tangential acceleration
    vel_minus_acc = velocities - accelerations
    angles = np.arctan2(vel_minus_acc[:, 1], vel_minus_acc[:, 0])
    tangential_acc = np.sin(np.pi / 2 - angles) * np.linalg.norm(accelerations, axis=1)

    return speeds, tangential_acc


## for the 'acceleration-aligned' tuning curves


def identify_movement_bouts(speed, acceleration, speed_threshold=0.05, acc_threshold=2.0, min_frames=5, max_frames=1):
    """
    Identify movement bouts by first finding clear stops, then checking for
    acceleration/deceleration events within a limited window around these stops.

    Parameters:
    -----------
    speed : array-like
        Array of speed measurements
    acceleration : array-like
        Array of acceleration measurements
    speed_threshold : float
        Speed threshold for movement detection (m/s)
    acc_threshold : float
        Acceleration threshold for bout detection (m/s²)
    min_frames : int
        Minimum consecutive frames below speed threshold for valid stop
    max_frames : int
        Maximum frames to look before/after stop period for acceleration events

    Returns:
    --------
    start_indices : array-like
        Frame indices where acceleration bouts begin
    stop_indices : array-like
        Frame indices where deceleration bouts begin
    """
    # First identify periods where speed is below threshold
    below_threshold = speed < speed_threshold

    # Find all transitions
    state_changes = np.diff(below_threshold.astype(int))
    still_starts = np.where(state_changes == 1)[0] + 1  # +1 because diff reduces length by 1
    still_ends = np.where(state_changes == -1)[0] + 1

    # Handle edge cases
    if below_threshold[0]:  # Started in stillness
        still_starts = np.insert(still_starts, 0, 0)
    if below_threshold[-1]:  # Ended in stillness
        still_ends = np.append(still_ends, len(speed))

    # Group stillness periods that meet minimum duration
    stops = []  # Will store (start, end) tuples of valid stop periods

    for i in range(len(still_starts)):
        duration = still_ends[i] - still_starts[i]
        if duration >= min_frames:
            stops.append((still_starts[i], still_ends[i]))

    # Initialize bout lists
    start_indices = []
    stop_indices = []

    # For each valid stop period, look for acceleration/deceleration events
    for stop_start, stop_end in stops:
        # Look for deceleration just before stop began
        search_start = max(0, stop_start - max_frames)
        if search_start < stop_start:  # handle edge case for the first stop.
            dec_window = acceleration[search_start:stop_start]
            if np.min(dec_window) < -acc_threshold:
                # Find first deceleration threshold crossing
                dec_idx = search_start + np.where(dec_window < -acc_threshold)[0][0]
                # Only include if it's within max_frames of stop
                if stop_start - dec_idx <= max_frames:
                    stop_indices.append(dec_idx)

        # Look for acceleration just after stop ended
        search_end = min(len(acceleration), stop_end + max_frames)
        if stop_end < search_end:
            acc_window = acceleration[stop_end:search_end]
            if np.max(acc_window) > acc_threshold:
                # Find first acceleration threshold crossing
                acc_idx = stop_end + np.where(acc_window > acc_threshold)[0][0]
                # Only include if it's within max_frames of stop
                if acc_idx - stop_end <= max_frames:
                    start_indices.append(acc_idx)

    return np.array(start_indices), np.array(stop_indices)


def extract_aligned_windows(data, event_indices, pre_window=120, post_window=120):
    """
    Extract windows of data aligned to events, handling edge cases near session boundaries.

    Parameters:
    -----------
    data : array-like
        Time series data to align
    event_indices : array-like
        Indices of alignment events
    pre_window : int
        Number of frames before event
    post_window : int
        Number of frames after event

    Returns:
    --------
    windows : array-like
        Array of aligned windows (n_events x window_length)
    valid_events : array-like
        Boolean mask indicating which events had complete windows
    """
    window_length = pre_window + post_window
    windows = np.zeros((len(event_indices), window_length))
    valid_events = np.ones(len(event_indices), dtype=bool)

    for i, event_idx in enumerate(event_indices):
        # Calculate window boundaries
        start_idx = event_idx - pre_window
        end_idx = event_idx + post_window

        # Check if window extends beyond data boundaries
        if start_idx < 0 or end_idx > len(data):
            valid_events[i] = False
            continue

        # Extract window
        windows[i, :] = data[start_idx:end_idx]

    # Return only windows from valid events
    return windows[valid_events], valid_events


# %%
