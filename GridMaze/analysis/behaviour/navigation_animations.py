"""Module for making colourful animations of subjects navigating the maze (twitter eye candy)"""
# %% Imports
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter1d

from .. import get_sessions as gs
from ...maze import plotting as mp

# %%
session = gs.get_sessions(subject_IDs=["m2"], maze_number=[1], day_on_maze=[12], with_data=["navigation_df"])[0]


def plot_session_trajectory_animation2(session, trials, downsample_factor=2, smooth_SD=5):
    simple_maze = session.simple_maze()
    goal2_standard_color = mp.get_goal2standard_color()
    navigation_df = session.navigation_df
    navigation_df = navigation_df[navigation_df.trial.isin(trials)]
    navigation_df = navigation_df[::downsample_factor].reset_index(drop=True)

    navigation_df.loc[:, ("centroid_position", "x")] = gaussian_filter1d(
        navigation_df["centroid_position"]["x"], sigma=5
    )
    navigation_df.loc[:, ("centroid_position", "y")] = gaussian_filter1d(
        navigation_df["centroid_position"]["y"], sigma=5
    )
    f, ax = plt.subplots(figsize=(5, 5), clear=True)
    f.tight_layout()

    def update_frame(frame):
        ax.clear()
        row = navigation_df.iloc[frame]
        goal = row[("goal", "")]
        x = row[("centroid_position", "x")]
        y = row[("centroid_position", "y")]
        maze_pos = row[("maze_position", "simple")]
        if row[("trial_phase", "")] == "navigation":
            location2color = {goal: goal2_standard_color[goal]}
        else:
            location2color = {}
        mp.plot_simple_maze_silhouette(simple_maze, ax, color="silver", special_location2color=location2color)
        ax.plot(x, y, "o", color="black", markersize=10)

    anim = FuncAnimation(f, update_frame, frames=len(navigation_df), repeat=False)
    anim.save("../results/animated_trajectories/test_anim.mp4", writer="ffmpeg", fps=60, dpi=300)
    return


# %%
