"""This module visualises navigational trajectories on the maze"""

# %% imports
import numpy as np
from ...maze import plotting as mp
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %% Global variables
plt.rcParams["pdf.fonttype"] = 42

# %%


def plot_session_trajectory(session, smoothed=True):
    simple_maze = session.simple_maze()
    navigation_df = session.navigation_df
    x_traj = navigation_df.centroid_position.x
    y_traj = navigation_df.centroid_position.y
    if smoothed:
        x_traj = gaussian_filter1d(x_traj, sigma=10)
        y_traj = gaussian_filter1d(y_traj, sigma=10)
    goals = session.goals
    f, ax = plt.subplots(figsize=(6, 6), clear=True)
    ax.axis("off")
    mp.plot_simple_maze_silhouette(simple_maze, ax, color="lightgrey", highlight_nodes=goals, highlight_color="red")
    ax.scatter(x_traj, y_traj, color="blue", s=5, alpha=0.01, zorder=3)


# %%


def plot_trial_trajectory(
    session,
    trial,
    smooth_SD=5,
    traj_color="red",
    start_color="black",
    goal_color="blue",
    ax=None,
):
    simple_maze = session.simple_maze()
    navigation_df = session.navigation_df
    trajectory_df = navigation_df[(navigation_df.trial == trial) & (navigation_df.trial_phase == "navigation")]
    if trajectory_df.empty:
        return print(f"No navigation data for trial {trial}")
    goal = trajectory_df.goal.values[0]
    start = trajectory_df.maze_position.simple.values[0]
    x_traj = trajectory_df.centroid_position.x
    y_traj = trajectory_df.centroid_position.y
    if smooth_SD:
        x_traj = gaussian_filter1d(x_traj, sigma=smooth_SD)
        y_traj = gaussian_filter1d(y_traj, sigma=smooth_SD)
    if ax is None:
        f, ax = plt.subplots(figsize=(3, 3), clear=True)
    mp.plot_simple_maze_silhouette(
        simple_maze,
        ax,
        color="lightgrey",
        special_location2color={start: start_color, goal: goal_color},
        node_size=150,
        edge_size=6,
    )
    ax.plot(x_traj, y_traj, color=traj_color, linewidth=5, alpha=0.7, zorder=3)
    return


# %%


def plot_goal_trajectories(session, smoothed=True, smooth_SD=5, colormap="viridis"):
    simple_maze = session.simple_maze()
    navigation_df = session.navigation_df
    navigation_df = navigation_df[navigation_df.trial_phase == "navigation"]
    goals = navigation_df.goal.unique()
    max_goal_trials = np.max([len(navigation_df[navigation_df.goal == goal].trial.unique()) for goal in goals])
    cm = plt.cm.get_cmap(colormap, max_goal_trials)
    trial_colors = [mcolors.rgb2hex(cm(i)[:3]) for i in range(cm.N)]

    fig_rows, fig_cols = 3, 4
    num_plots_per_fig = fig_rows * fig_cols
    num_figs = np.ceil(len(goals) / num_plots_per_fig).astype(int)

    fig_index = 0
    axs = None

    for i, goal in enumerate(goals):
        if i % num_plots_per_fig == 0:
            if i > 0:
                plt.tight_layout()
                plt.show()
            fig, axs = plt.subplots(fig_rows, fig_cols, figsize=(24, 6 * fig_rows), clear=True)
            axs = axs.flatten()
            fig_index = 0

        goal_navigation_df = navigation_df[navigation_df.goal == goal]
        trials = goal_navigation_df.trial.unique()
        start_location2color = {}

        for j, trial in enumerate(trials):
            trial_color = trial_colors[j]
            trial_df = goal_navigation_df[goal_navigation_df.trial == trial]
            start_location = trial_df.maze_position.simple.to_numpy()[0]
            start_location2color[start_location] = trial_color
            x_traj = trial_df.centroid_position.x
            y_traj = trial_df.centroid_position.y
            if smoothed:
                x_traj = gaussian_filter1d(x_traj, sigma=smooth_SD)
                y_traj = gaussian_filter1d(y_traj, sigma=smooth_SD)
            ax = axs[fig_index]
            ax.plot(x_traj, y_traj, color=trial_color, linewidth=5, alpha=0.7, zorder=3)
        mp.plot_simple_maze_silhouette(
            simple_maze,
            ax,
            color="silver",
            highlight_nodes=[goal],
            highlight_color="deepskyblue",
            special_location2color=start_location2color,
            node_size=250,
            edge_size=6,
        )
        ax.set_title(f"Goal {goal}")
        fig_index += 1

    plt.tight_layout()
    plt.show()


# %% Functions for plotting single trial rate/spike maps


def plot_cluster_rate_trajectories(
    session, cluster_unique_ID, smooth_traj=True, smooth_rates=True, smooth_SD=10, colormap="plasma"
):
    """"""
    simple_maze = session.simple_maze()
    navigation_rates_df = session.get_navigation_activity_df(type="rates")
    navigation_rates_df = navigation_rates_df[navigation_rates_df.trial_phase == "navigation"]
    trials = navigation_rates_df.trial.unique()
    no_figs = (len(trials) + 8) // 9
    all_cluster_rates = navigation_rates_df.firing_rate[cluster_unique_ID].to_numpy()
    if smooth_rates:
        all_cluster_rates = gaussian_filter1d(all_cluster_rates, sigma=smooth_SD)
    max_rate = np.max(all_cluster_rates)
    min_rate = np.min(all_cluster_rates)
    for fig_no in range(no_figs):
        fig, axs = plt.subplots(3, 3, figsize=(10, 10), clear=True)
        axs = axs.flatten()
        fig.suptitle(f"{cluster_unique_ID}", fontsize=14, x=0.5, y=0.98)
        for ax_idx in range(9):
            trial_index = fig_no * 9 + ax_idx
            if trial_index >= len(trials):
                break
            trial = trials[trial_index]
            ax = axs[ax_idx]
            ax.set_title(f"Trial {int(trial)}")
            trial_navigation_rates_df = navigation_rates_df[navigation_rates_df.trial == trial]
            start_location = trial_navigation_rates_df.maze_position.simple.to_numpy()[0]
            goal_location = trial_navigation_rates_df.goal.to_numpy()[0]
            x_traj = trial_navigation_rates_df.centroid_position.x
            y_traj = trial_navigation_rates_df.centroid_position.y
            if smooth_traj:
                x_traj = gaussian_filter1d(x_traj, sigma=smooth_SD)
                y_traj = gaussian_filter1d(y_traj, sigma=smooth_SD)
            firing_rate = trial_navigation_rates_df.firing_rate[cluster_unique_ID].to_numpy()
            if smooth_rates:
                firing_rate = gaussian_filter1d(firing_rate, sigma=smooth_SD)
            # plotting
            # make traj line colored by rates
            points = np.array([x_traj, y_traj]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(
                segments,
                cmap=colormap,
                norm=Normalize(firing_rate.min(), firing_rate.max()),
                zorder=3,
                antialiased=True,
            )
            lc.set_array(firing_rate)
            lc.set_linewidth(5)
            ax.add_collection(lc)
            # make colorbar for last subplot of each figure
            if ax_idx == 8 or trial_index == len(trials) - 1:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(mp.get_colorbar(min_rate, max_rate, colormap), cax=cax)
                cbar.outline.set_visible(False)
                cbar.set_label("Firing Rate (Hz)", labelpad=10, fontsize=14)
            mp.plot_simple_maze_silhouette(
                simple_maze,
                ax,
                color="silver",
                special_location2color={start_location: "yellowgreen", goal_location: "deepskyblue"},
                node_size=150,
                edge_size=6,
            )
        plt.tight_layout()


def plot_cluster_spike_trajectories(
    session, cluster_unique_ID, smooth=True, smooth_SD=5, spike_color="red", traj_color="navy"
):
    """"""
    simple_maze = session.simple_maze()
    navigation_rates_df = session.get_navigation_activity_df(type="spikes")
    navigation_rates_df = navigation_rates_df[navigation_rates_df.trial_phase == "navigation"]
    trials = navigation_rates_df.trial.unique()
    no_figs = (len(trials) + 8) // 9
    for fig_no in range(no_figs):
        fig, axs = plt.subplots(3, 3, figsize=(10, 10), clear=True)
        axs = axs.flatten()
        fig.suptitle(f"{cluster_unique_ID}", fontsize=14, x=0.5, y=0.98)
        for ax_idx in range(9):
            trial_index = fig_no * 9 + ax_idx
            if trial_index >= len(trials):
                break
            trial = trials[trial_index]
            ax = axs[ax_idx]
            ax.set_title(f"Trial {int(trial)}")
            trial_navigation_rates_df = navigation_rates_df[navigation_rates_df.trial == trial]
            start_location = trial_navigation_rates_df.maze_position.simple.to_numpy()[0]
            goal_location = trial_navigation_rates_df.goal.to_numpy()[0]
            x_traj = trial_navigation_rates_df.centroid_position.x
            y_traj = trial_navigation_rates_df.centroid_position.y
            if smooth:
                x_traj = gaussian_filter1d(x_traj, sigma=smooth_SD)
                y_traj = gaussian_filter1d(y_traj, sigma=smooth_SD)
            spike_counts = trial_navigation_rates_df.spike_count[cluster_unique_ID].to_numpy()
            # Duplicate each
            x_traj_spikes = np.repeat(x_traj, spike_counts)
            y_traj_spikes = np.repeat(y_traj, spike_counts)
            # plotting
            ax.plot(x_traj, y_traj, color=traj_color, linewidth=5, alpha=0.7, zorder=3)
            ax.scatter(x_traj_spikes, y_traj_spikes, color=spike_color, s=10, alpha=0.3, zorder=4)
            mp.plot_simple_maze_silhouette(
                simple_maze,
                ax,
                color="silver",
                special_location2color={start_location: "yellowgreen", goal_location: "deepskyblue"},
                node_size=150,
                edge_size=6,
            )
        plt.tight_layout()


# %% plot trajectories with colored left right turns for quality control

from matplotlib.colors import ListedColormap


def plot_trial_with_eogcentric_action(
    session,
    trial,
    smooth_SD=5,
    action_colors={"forward": "grey", "back": "black", "left": "blue", "right": "red"},
    ax=None,
):
    if ax is None:
        f, ax = plt.subplots(figsize=(4, 4), clear=True)
    action_label2int = {"go_forward": 0, "go_back": 1, "turn_left": 2, "turn_right": 3}
    custom_cmap = ListedColormap(action_colors.values())
    navigation_df = session.navigation_df.copy()
    # navigation_df[[("action", "basic"), ("action", "choice_degree")]] = navigation_df[
    #     [("action", "basic"), ("action", "choice_degree")]
    # ].ffill()
    # select requested trial during navigation
    trial_df = navigation_df[(navigation_df.trial == trial) & (navigation_df.trial_phase == "navigation")]
    start_location = trial_df.maze_position.simple.to_numpy()[0]
    goal_location = trial_df.goal.to_numpy()[0]
    x_traj = trial_df.centroid_position.x
    y_traj = trial_df.centroid_position.y
    if smooth_SD:
        x_traj = gaussian_filter1d(x_traj, sigma=smooth_SD)
        y_traj = gaussian_filter1d(y_traj, sigma=smooth_SD)
    ego_actions_int = trial_df.action.basic.ffill().map(action_label2int).values
    # plotting
    points = np.array([x_traj, y_traj]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments,
        cmap=custom_cmap,
        zorder=3,
        antialiased=True,
    )
    lc.set_array(ego_actions_int)
    lc.set_linewidth(5)
    ax.add_collection(lc)
    mp.plot_simple_maze_silhouette(
        session.simple_maze(),
        ax,
        color="silver",
        special_location2color={start_location: "yellowgreen", goal_location: "deepskyblue"},
        node_size=150,
        edge_size=6,
    )

    return


from matplotlib.colors import to_rgba
from sklearn.preprocessing import MinMaxScaler


def plot_trial_with_eogcentric_action2(
    session,
    trial,
    smooth_SD=5,
    action_colors={"go_forward": "green", "go_back": "black", "turn_left": "blue", "turn_right": "red"},
    ax=None,
):
    """
    This one just plot the trajectory with scatter point for each punctate action
    """
    if ax is None:
        f, ax = plt.subplots(figsize=(6, 6), clear=True)
    navigation_df = session.navigation_df.copy()
    at_goal_mask = navigation_df.maze_position.simple == navigation_df.goal
    ego_actions = navigation_df.action.basic
    convolved_actions = convolve_egocentric_action(ego_actions.values)
    convolved_actions[at_goal_mask] = 0
    trial_mask = (navigation_df.trial == trial) & (navigation_df.trial_phase == "navigation")
    trial_df = navigation_df[trial_mask]
    start_location = trial_df.maze_position.simple.to_numpy()[0]
    goal_location = trial_df.goal.to_numpy()[0]
    x_traj = trial_df.centroid_position.x
    y_traj = trial_df.centroid_position.y
    if smooth_SD:
        x_traj = gaussian_filter1d(x_traj, sigma=smooth_SD)
        y_traj = gaussian_filter1d(y_traj, sigma=smooth_SD)
    # plotting
    mp.plot_simple_maze_silhouette(
        session.simple_maze(),
        ax,
        color="silver",
        special_location2color={start_location: "grey", goal_location: "deepskyblue"},
        node_size=150,
        edge_size=6,
    )
    ax.plot(x_traj, y_traj, color="grey", linewidth=5, alpha=0.7, zorder=3)
    convolved_actions = convolved_actions[trial_mask]
    for i, ego_action in enumerate(["go_forward", "go_back", "turn_left", "turn_right"]):
        alphas = convolved_actions[:, i]
        scaler = MinMaxScaler(feature_range=(0, 1))
        alphas = scaler.fit_transform(alphas.reshape(-1, 1)).flatten()
        mask = alphas > 0
        base_color = to_rgba(action_colors[ego_action])[:3]
        colors = np.array([(*base_color, alpha) for alpha in alphas])
        ax.scatter(
            x_traj[mask],
            y_traj[mask],
            color=colors[mask],
            s=20,
            zorder=i + 2,
        )
    return


def plot_trial_with_eogcentric_action3(
    navigation_df,
    simple_maze,
    trial,
    smooth_SD=5,
    action_colors={"go_forward": "green", "go_back": "black", "turn_left": "blue", "turn_right": "red"},
    ax=None,
):
    """
    This one just plot the trajectory with scatter point for each punctate action
    """
    if ax is None:
        f, ax = plt.subplots(figsize=(6, 6), clear=True)
    ego_actions = navigation_df.action.basic
    trial_mask = (navigation_df.trial == trial) & (navigation_df.trial_phase == "navigation")
    trial_df = navigation_df[trial_mask]
    start_location = trial_df.maze_position.simple.to_numpy()[0]
    goal_location = trial_df.goal.to_numpy()[0]
    x_traj = trial_df.centroid_position.x
    y_traj = trial_df.centroid_position.y
    if smooth_SD:
        x_traj = gaussian_filter1d(x_traj, sigma=smooth_SD)
        y_traj = gaussian_filter1d(y_traj, sigma=smooth_SD)
    # plotting
    mp.plot_simple_maze_silhouette(
        simple_maze,
        ax,
        color="silver",
        special_location2color={start_location: "grey", goal_location: "deepskyblue"},
        node_size=150,
        edge_size=6,
    )
    ax.plot(x_traj, y_traj, color="grey", linewidth=5, alpha=0.7, zorder=3)
    ego_actions = ego_actions[trial_mask]
    for i, ego_action in enumerate(["go_forward", "go_back", "turn_left", "turn_right"]):
        mask = ego_actions == ego_action
        ax.scatter(
            x_traj[mask],
            y_traj[mask],
            color=action_colors[ego_action],
            s=20,
            zorder=i + 2,
        )
    return


# %%

from scipy.ndimage import convolve1d
from scipy.signal.windows import gaussian


def convolve_egocentric_action(a, egocentric_actions=["go_forward", "go_back", "turn_left", "turn_right"], sigma=10):
    """ """
    actions_array = np.array([(a == e).astype(float) for e in egocentric_actions]).T  # [n_actions, n_frames]
    convolved_actions = gaussian_filter1d(actions_array, sigma=sigma, axis=0)
    # convolved_actions = convolved_actions / convolved_actions.max()
    return np.abs(convolved_actions)
