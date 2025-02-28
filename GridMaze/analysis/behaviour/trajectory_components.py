"""
This module will port some of Xiao's behavioural analysis code for application to my data, 
with new plotting functions
"""
# %% Imports
import json
import string
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA, NMF


from .. import get_sessions as gs
from ..processing import get_trajectory_decisions_dfs as tdf
from ...maze import plotting as mp
from ...maze import representations as mr

# %% Global variables
with open("../data/experiment_info.json") as input_file:
    EXP_INFO = json.load(input_file)


# %%
def get_combined_trajectory_decisions_df(subject, maze_number, late_only=True):
    subject = EXP_INFO["subject_IDs"] if subject == "all" else [subject]
    days = list(np.arange(5, 14)) if late_only else "all"
    sessions = gs.get_sessions(
        subject_IDs=subject, maze_number=[maze_number], day_on_maze=days, with_data=["trajectory_decisions_df"]
    )
    subject_maze_trajectory_decisions_df = pd.concat([s.trajectory_decisions_df for s in sessions])
    return subject_maze_trajectory_decisions_df.sort_values(by=["day_on_maze", "time"])


# %%


def plot_behavioural_components_cross_subject_comparison(maze_number, n_components=12, method="NMF_nav"):
    simple_maze = mr.simple_maze(EXP_INFO["maze_config2info"][f"maze_{maze_number}"]["structure"])
    for subject in EXP_INFO["subject_IDs"]:
        traj_df = get_combined_trajectory_decisions_df(subject=subject, maze_number=maze_number, late_only=True)
        if method == "NMF":
            NMF_df = get_behavioural_components_df(traj_df, simple_maze, model="NMF", n_components=n_components)
            behaviour_plotting_dicts = process_NMF_components_for_plotting(NMF_df)
            plot_simple_maze_NMF_components(behaviour_plotting_dicts, simple_maze, n_components, subject)
        elif method == "NMF_nav":
            NMF_df = get_non_exp_NMF_components_df(traj_df, simple_maze, n_components, normalise=False)
            behaviour_plotting_dicts = process_NMF_components_for_plotting(NMF_df)
            plot_simple_maze_NMF_components(behaviour_plotting_dicts, simple_maze, n_components, subject)
        elif method == "PCA":
            PCA_df = get_behavioural_components_df(traj_df, simple_maze, model="PCA", n_components=n_components)
            behaviour_plotting_dicts = process_PCA_components_for_plotting(PCA_df)
            plot_simple_maze_PCA_components(behaviour_plotting_dicts, simple_maze, n_components, subject)
    return


def plot_behavioural_components_summary_fig(maze_number, n_components=12, method="NMF_nav"):
    simple_maze = mr.simple_maze(EXP_INFO["maze_config2info"][f"maze_{maze_number}"]["structure"])
    subject_trajectory_decisions_df = get_combined_trajectory_decisions_df(
        subject="all", maze_number=maze_number, late_only=True
    )
    total_trials = (
        subject_trajectory_decisions_df.groupby(["subject_ID", "maze_number", "day_on_maze", "trial"]).count().shape[0]
    )
    total_steps = subject_trajectory_decisions_df.shape[0]
    opt_trajectory_decisions_df = tdf.get_optimal_trajectory_decisions_df(
        maze_number=maze_number, n_trials=total_trials, method="random_optimal_path", max_steps=total_steps
    )
    random_trajectory_decisions_df = tdf.get_optimal_trajectory_decisions_df(
        maze_number=maze_number, n_trials=total_trials, method="modified_random_walk", max_steps=total_steps
    )
    for traj_df, title in zip(
        [subject_trajectory_decisions_df, opt_trajectory_decisions_df, random_trajectory_decisions_df],
        ["subject", "optimal", "random"],
    ):
        if method == "NMF":
            NMF_df = get_behavioural_components_df(traj_df, simple_maze, model="NMF", n_components=n_components)
            behaviour_plotting_dicts = process_NMF_components_for_plotting(NMF_df)
            plot_simple_maze_NMF_components(behaviour_plotting_dicts, simple_maze, n_components, title)
        elif method == "NMF_nav":
            NMF_df = get_non_exp_NMF_components_df(traj_df, simple_maze, n_components, normalise=False)
            behaviour_plotting_dicts = process_NMF_components_for_plotting(NMF_df)
            plot_simple_maze_NMF_components(behaviour_plotting_dicts, simple_maze, n_components, title)
        elif method == "PCA":
            PCA_df = get_behavioural_components_df(traj_df, simple_maze, model="PCA", n_components=n_components)
            behaviour_plotting_dicts = process_PCA_components_for_plotting(PCA_df)
            plot_simple_maze_PCA_components(behaviour_plotting_dicts, simple_maze, n_components, title)

    return


# %% Plotting functions


def plot_simple_maze_PCA_components(component_plotting_dics, simple_maze, n_components, title):
    # Number of figures needed
    num_figures = (n_components + 3) // 4  # 4 PCs per figure

    for fig_num in range(num_figures):
        # Create a new figure with a 2x4 grid
        f, axes = plt.subplots(2, 4, figsize=(12, 6))
        f.suptitle(f"{title} - Figure {fig_num + 1}")

        # Calculate the number of PCs to be plotted in this figure
        pcs_in_this_figure = min(n_components - fig_num * 4, 4)  # Maximum of 4 PCs per figure

        # Plot the PCs
        for j in range(pcs_in_this_figure):
            PC_idx = fig_num * 4 + j
            PC_plotting_dict = component_plotting_dics[PC_idx]
            pos_info, neg_info = PC_plotting_dict
            pos_PCA_location2value, pos_PCA_location2NSEW = pos_info
            neg_PCA_location2value, neg_PCA_location2NSEW = neg_info

            # Correct the axis indexing for side-by-side plotting
            col_start_idx = (j * 2) % 4
            current_axes = [axes[j // 2, col_start_idx], axes[j // 2, col_start_idx + 1]]

            # Plot positive and negative components side-by-side
            for loc2value, loc2NSEW, ax, c_map, value_label in zip(
                [pos_PCA_location2value, neg_PCA_location2value],
                [pos_PCA_location2NSEW, neg_PCA_location2NSEW],
                current_axes,
                ["Reds", "Blues"],
                [f"PC {PC_idx + 1} +", f"PC {PC_idx + 1} -"],
            ):
                mp.plot_simple_star_heatmap(
                    simple_maze,
                    loc2value,
                    loc2NSEW,
                    ax=ax,
                    colormap=c_map,
                    title=None,
                    value_label=None,
                    silhouette_node_size=100,
                    silhouette_edge_size=5,
                )
                ax.set_title(value_label)

        plt.tight_layout()
        plt.show()

    return


def plot_simple_maze_NMF_components(component_plotting_dicts, simple_maze, n_components, title):
    # Calculate the number of figures required
    num_figures = (n_components + 15) // 16

    for fig_num in range(num_figures):
        components_to_plot = min(n_components - fig_num * 16, 16)

        # Create subplots
        rows = (components_to_plot + 3) // 4
        f, axes = plt.subplots(rows, 4, figsize=(9, 3 * rows))

        # Ensure axes is always 2-dimensional
        if components_to_plot == 1:
            axes = np.array([[axes]])
        elif components_to_plot <= 4:
            axes = np.expand_dims(axes, axis=0)

        f.tight_layout()
        f.suptitle(f"{title} - Figure {fig_num + 1}")

        for i in range(components_to_plot):
            PC_plotting_dict = component_plotting_dicts[fig_num * 16 + i]
            location2value, location2NSEW = PC_plotting_dict
            mp.plot_simple_star_heatmap(
                simple_maze,
                location2value,
                location2NSEW,
                ax=axes[i // 4, i % 4],
                colormap="Purples",
                title=None,
                value_label=None,
                silhouette_color="silver",
                silhouette_node_size=250,
                silhouette_edge_size=8,
            )
            axes[i // 4, i % 4].set_title(f"Component {fig_num * 16 + i}")

        # Set any leftover axes to blank
        for i in range(components_to_plot, 16):
            row_idx = i // 4
            col_idx = i % 4

        # Check if the index is valid for the current shape of axes
        if row_idx < axes.shape[0] and col_idx < axes.shape[1]:
            axes[row_idx, col_idx].axis("off")
    return


# %%


def process_NMF_components_for_plotting(NMF_df):
    locations = list(NMF_df.index.get_level_values(0).unique())
    NMF_cols = list(NMF_df.columns)
    NMF_plotting_dicts = []
    for NMF in NMF_cols:
        comp_NMF_df = NMF_df[NMF].reset_index()
        location2value = {}
        location2NSEW = {}
        for loc in locations:
            loc_NMF_df = comp_NMF_df[comp_NMF_df.maze_position == loc]
            values = loc_NMF_df[NMF].values
            values_sum = values.sum()
            location2value[loc] = values.max()
            directions = list(loc_NMF_df.direction)
            NSEW = {}
            for dir in directions:
                if values_sum == 0:
                    norm_value = 0
                else:
                    value = loc_NMF_df[loc_NMF_df.direction == dir][NMF].values[0]
                    norm_value = value / values_sum
                NSEW[dir] = {"value": norm_value, "valid": True}
            location2NSEW[loc] = NSEW
        NMF_plotting_dicts.append((location2value, location2NSEW))
    return NMF_plotting_dicts


def process_PCA_components_for_plotting(PCA_df):
    locations = list(PCA_df.index.get_level_values(0).unique())
    pos_PCA_df = PCA_df.copy()
    pos_PCA_df[pos_PCA_df < 0] = 0
    neg_PCA_df = PCA_df.copy()
    neg_PCA_df[neg_PCA_df > 0] = 0
    neg_PCA_df = neg_PCA_df.abs()
    PC_cols = list(PCA_df.columns)
    PC_plotting_dicts = []
    for PC in PC_cols:
        plotting_dicts = []
        for signed_PCA_df in [pos_PCA_df, neg_PCA_df]:
            PC_df = signed_PCA_df[PC].reset_index()
            location2value = {}
            location2NSEW = {}
            for loc in locations:
                loc_PC_df = PC_df[PC_df.maze_position == loc]
                values = loc_PC_df[PC].values
                values_sum = values.sum()
                location2value[loc] = values.max()
                directions = list(loc_PC_df.direction)
                NSEW = {}
                for dir in directions:
                    if values_sum == 0:
                        norm_value = 0
                    else:
                        value = loc_PC_df[loc_PC_df.direction == dir][PC].values[0]
                        norm_value = value / values_sum
                    NSEW[dir] = {"value": norm_value, "valid": True}
                location2NSEW[loc] = NSEW
            plotting_dicts.append((location2value, location2NSEW))
        PC_plotting_dicts.append(plotting_dicts)
    return PC_plotting_dicts


# %% Modified version's of Xiao's exponential decay model fitting


def get_behavioural_components_df(trajectory_decisions_df, simple_maze, model="PCA", n_components=None):
    # convert data to Xiao's format
    trajectory_decisions_df = refactor_trajectory_decisions_df(trajectory_decisions_df)
    pos = torch.tensor(trajectory_decisions_df.pos_idx.to_numpy())
    act = torch.tensor(trajectory_decisions_df.action_class.to_numpy())
    obs = act * 225 + pos
    start_idx = torch.zeros(1)
    if model == "PCA":
        v = exponential_decay_PCA(obs.long(), start_idx=start_idx.long(), tot_obs=900, n_components=n_components)
    elif model == "NMF":
        v = exponential_decay_NMF(obs.long(), start_idx=start_idx.long(), tot_obs=900, n_components=n_components)
    # Convert labels back to Peters format
    df, inds = reset_maze_position_direction_format()
    vr = torch.real(v)
    n = vr.shape[-1]
    components_df = {}
    for i in range(n):
        components_df[f"component_{i}"] = vr[inds, i].tolist()
    components_df = pd.DataFrame(data=components_df)
    components_df = pd.concat([df, components_df], axis=1)
    # This df contains edges that are not in the maze (but values are 0)
    # Remove these df rows
    components_df = remove_invalid_edge_features(simple_maze, components_df)
    return components_df


def exponential_decay_PCA(idx, start_idx, n_components=None, tot_obs=196, alpha=0.1):
    """ """
    exp_state_one_hot = get_exp_state_one_hot(idx, start_idx, tot_obs, alpha)
    model = PCA(n_components=n_components)
    model.fit(exp_state_one_hot)
    v = torch.tensor(model.components_).t()
    return v


def exponential_decay_NMF(idx, start_idx, n_components=10, tot_obs=196, alpha=0.1):
    exp_state_one_hot = get_exp_state_one_hot(idx, start_idx, tot_obs, alpha)
    model = NMF(n_components=n_components, max_iter=1000)
    model.fit(exp_state_one_hot)
    v = torch.tensor(model.components_).t()
    return v


def get_exp_state_one_hot(idx, start_idx, tot_obs, alpha):
    """
    :param idx: tensor of observations/state action apirs
    :type idx: torch.Tensor
    :param start_idx: tensor of starting position of each session
    :type start_idx: torch.Tensor
    :param tot_obs: total possible number of unique observations
    :type tot_obs: int
    :param alpha: 1-alpha is the decay between each time step
    :type alpha: float
    :return:
    :rtype:
    """
    n = idx.shape[0]
    exp_state_one_hot = []
    start_idx = torch.cat([start_idx, torch.tensor([n])])
    state_one_hot = F.one_hot(idx, tot_obs)
    for i in range(start_idx.shape[0] - 1):
        i_start = start_idx[i]
        i_end = start_idx[i + 1]
        exp_state_one_hot.append(exponential_average(state_one_hot[i_start:i_end], alpha=alpha))
        exp_state_one_hot.append(exponential_average(torch.flip(state_one_hot[i_start:i_end], dims=(0,)), alpha=alpha))
    exp_state_one_hot = torch.cat(exp_state_one_hot)
    exp_state_one_hot = exp_state_one_hot / (exp_state_one_hot.pow(2).sum(dim=-1, keepdim=True)).pow(
        0.5
    )  # normalize so each vector has a magnitude of 1
    return exp_state_one_hot


def exponential_average(x, alpha=0.1):
    """
    :param x: shape : n x d where d are the dimensions that need to be averaged
    :type x: torch tensor
    :param alpha: higher alpha, smaller decay
    :type alpha:
    :return: shape n x d
    :rtype: torch tensor
    """
    exp_x = []
    for i, p in enumerate(x):
        if i == 0:
            exp_avg = alpha * p
        else:
            exp_avg = (1 - alpha) * exp_avg + alpha * p
        exp_x.append(exp_avg.unsqueeze(0))
    return torch.cat(exp_x)


# %% Refactoring functions


def refactor_trajectory_decisions_df(trajectory_decisions_df):
    """
    Refactor's Peters trajectory_decision_df format to Xiao's analysis format to be compatible wither her functions
    """
    df = trajectory_decisions_df.copy()
    df = df.dropna()
    al = string.ascii_uppercase
    df = df.assign(
        x_pos=[
            al.index(a[0]) * 2 + 1 if len(a) == 2 else (al.index(a[0]) + al.index(a[3])) + 1 for a in df.maze_position
        ]
    )
    df = df.assign(y_pos=[int(a[1]) * 2 - 1 if len(a) == 2 else (int(a[1]) + int(a[4])) - 1 for a in df.maze_position])
    df = df.assign(pos_idx=[a.x_pos * 15 + a.y_pos for row, a in df.iterrows()])
    action_class = {"E": 0, "N": 1, "W": 2, "S": 3}
    df = df.assign(action_class=[action_class[x.action] for _, x in df.iterrows()])
    return df


def reset_maze_position_direction_format():
    alphabet = string.ascii_uppercase
    state, a = [], []
    dir = "ENWS"
    inds = list(range(900))
    for i in range(900):
        ai = i // 225
        s = i % 225
        x, y = s // 15, s % 15
        if x == 0 or y == 0 or x == 14 or y == 14:
            inds.remove(i)
            continue
        if x % 2 == 0 and y % 2 == 0:
            inds.remove(i)
            continue
        xi = alphabet[x // 2 - 1] * (x % 2 == 0) + alphabet[x // 2]
        yi = str((y // 2)) * (y % 2 == 0) + str(y // 2 + 1)
        state_list = [q + p for q in xi for p in yi]
        state_collapsed = state_list[0]
        if len(state_list) > 1:
            state_collapsed = f"{state_collapsed}-{state_list[1]}"
        state.append(state_collapsed)
        a.append(dir[ai])
    d = {"maze_position": state, "direction": a}
    df = pd.DataFrame(data=d)
    return df, inds


def remove_invalid_edge_features(simple_maze, df):
    maze_position2valid_NSEW = mr.get_maze_location2NSEW(simple_maze)
    valid_locationNSEW_tuples = [
        (k, v_single) for k, v_list in maze_position2valid_NSEW.items() for v_single in v_list
    ]
    df_new = df.copy()
    df_new = df_new.set_index(["maze_position", "direction"])
    df_new = df_new.loc[valid_locationNSEW_tuples]
    return df_new


# %% My Own version of the behaviour NMF decomposition


def get_non_exp_NMF_components_df(
    trajectory_decisions_df,
    simple_maze,
    method="NMF",
    n_components=10,
    navigation_only=True,
    nav_length_threshold=2,
    normalise=False,
):
    if navigation_only:
        trajectory_decisions_df = trajectory_decisions_df[trajectory_decisions_df.trial_phase == "navigation"]
    trial_unique_IDs = trajectory_decisions_df.trial_unique_ID.unique()
    nav_sequences = []
    nav_action_pairs = get_maze_position_action_pairs(simple_maze)
    pair2idx = {pair: i for i, pair in enumerate(nav_action_pairs)}
    one_hots = []
    for trial in trial_unique_IDs:
        trail_traj_df = trajectory_decisions_df[trajectory_decisions_df.trial_unique_ID == trial]
        nav_sequence = list(zip(trail_traj_df.maze_position, trail_traj_df.action))
        if len(nav_sequence) < nav_length_threshold:
            continue
        else:
            nav_sequences.append(nav_sequence)
            one_hot_vector = np.zeros(len(nav_action_pairs), dtype=int)
            for pair in nav_sequence:
                one_hot_vector[pair2idx[pair]] += 1
            one_hots.append(one_hot_vector)
    M = np.vstack(one_hots)
    if normalise:  # Normalise each one-hot to have a magnitude of 1
        M = M / np.linalg.norm(M, axis=1)[:, np.newaxis]
    if method == "NMF":
        # nmf_kwargs = {
        #     "init": "random",
        #     "random_state": 0,
        #     "solver": "mu",
        #     "beta_loss": "kullback-leibler",
        #     "max_iter": 1000,
        # }
        model = NMF(n_components=n_components, max_iter=1000)
    elif method == "PCA":
        model = PCA(n_components=n_components)
    model.fit(M)
    v = model.components_
    col_titles = (
        [f"component_{i}" for i in range(n_components)]
        if n_components is not None
        else [f"component_{i}" for i in range(M.shape[1])]
    )
    components_df = pd.DataFrame(data=v.T, columns=col_titles)
    maze_position, action = zip(*nav_action_pairs)
    components_df["maze_position"] = maze_position
    components_df["direction"] = action
    components_df = components_df.set_index(["maze_position", "direction"])
    return components_df


def get_maze_position_action_pairs(simple_maze):
    node_coord2label = nx.get_node_attributes(simple_maze, "label")
    edge_coord2label = nx.get_edge_attributes(simple_maze, "label")
    location_action_pairs = []
    for node in simple_maze.nodes:
        neighbors = list(simple_maze.neighbors(node))
        for neighbor in neighbors:
            if neighbor[0] == node[0] + 1:
                direction = "E"
            elif neighbor[0] == node[0] - 1:
                direction = "W"
            elif neighbor[1] == node[1] + 1:
                direction = "N"
            elif neighbor[1] == node[1] - 1:
                direction = "S"
            location_action_pairs.append((node_coord2label[node], direction))
    for edge in simple_maze.edges:
        node1, node2 = edge
        if node1[0] == (node2[0] + 1) or node1[0] == (node2[0] - 1):
            directions = ["E", "W"]
        elif node1[1] == (node2[1] + 1) or node1[1] == (node2[1] - 1):
            directions = ["N", "S"]
        location_action_pairs.append((edge_coord2label[edge], directions[0]))
        location_action_pairs.append((edge_coord2label[edge], directions[1]))
    return location_action_pairs
