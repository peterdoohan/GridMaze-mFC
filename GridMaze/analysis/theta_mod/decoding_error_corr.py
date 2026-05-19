"""
Single-sample correlation of place and distance-to-goal decoding errors.

Tests whether, on individual samples, the place decoder's bias toward a neighbouring
trajectory point (prev vs. next) is coordinated with the distance decoder's bias
toward the corresponding distance (dist_prev vs. dist_next). If so, the two
representations are linked rather than independent — a motivating result for the
downstream theta-modulation analyses in this folder.
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr, ttest_1samp, linregress

from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.place_direction import future_decoding as fd
from GridMaze.analysis.neGLM import load_model_sets as lms
from GridMaze.analysis.neGLM import variance_explained as ve

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS2_PATH

RESULTS_DIR = RESULTS2_PATH / "theta_mod"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)


# %% plotting


def plot_decoding_error_corr(
    results_df,
    maze_names=("maze_1", "maze_2"),
    good_sessions_only=True,
    regress_out_true_dist=False,
    regress_out_dirs=False,
    color="indigo",
    ax=None,
    print_stats=True,
):
    """ """
    df = results_df.copy()
    if maze_names is not None:
        df = df[df.maze_name.isin(maze_names)]
    if good_sessions_only:
        df = df[df.good_session]

    # deltas: (next - prev) / (next + prev) for each decoder
    df["delta_loc"] = (df.p_loc_next - df.p_loc_prev) / (df.p_loc_next + df.p_loc_prev)
    df["delta_dist"] = (df.p_dist_next - df.p_dist_prev) / (df.p_dist_next + df.p_dist_prev)
    df = df.dropna(subset=["delta_loc", "delta_dist"])

    subject_corrs = []
    for subject_ID, sdf in df.groupby("subject_ID"):
        d_loc = sdf.delta_loc.values
        d_dist = sdf.delta_dist.values
        if regress_out_true_dist:
            d_loc = _residualise(d_loc, sdf.true_dist_residual.values)
            d_dist = _residualise(d_dist, sdf.true_dist_residual.values)
        if regress_out_dirs:
            d_loc = _residualise(d_loc, sdf.dirs.values)
            d_dist = _residualise(d_dist, sdf.dirs.values)
        corr = pearsonr(d_loc, d_dist)[0]
        subject_corrs.append((subject_ID, corr))

    subjects, corrs = zip(*subject_corrs)
    corrs = np.array(corrs)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(0.5, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", ls="--", alpha=0.5)
    # individual subject points in faint grey
    ax.scatter(
        np.zeros_like(corrs) + np.linspace(-0.1, 0.1, len(corrs)),
        corrs,
        color="grey",
        alpha=0.4,
        s=25,
        edgecolors="none",
        zorder=2,
    )
    # mean ± SEM across subjects
    plot_df = pd.DataFrame({"x": np.zeros(len(corrs)), "corr": corrs})
    sns.pointplot(
        data=plot_df,
        x="x",
        y="corr",
        errorbar="se",
        color=color,
        marker="o",
        markersize=8,
        linestyle="none",
        ax=ax,
        zorder=3,
    )
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("correlation\n(place δ ↔ distance δ)")
    ax.set_ylim(-0.005, 0.06)

    if print_stats:
        t, p = ttest_1samp(corrs, 0, alternative="greater")
        print(f"mean={corrs.mean():.3f}, sem={corrs.std()/np.sqrt(len(corrs)):.3f}, t={t:.3f}, p={p:.4f}")

    return ax


def _residualise(y, x):
    """Regress x out of y and return residuals."""
    reg = linregress(x, y)
    return y - (reg.intercept + reg.slope * x)


# %% experiment-level populate + cache-load


def get_decoding_error_corr_df(
    maze_names=["maze_1", "maze_2", "rooms_maze"],
    days_on_maze="late",
    n_jobs=-1,
    save=False,
    verbose=True,
):
    """
    Populate (or load cached) per-sample results across all subjects.
    If save=False and the parquet exists, it is loaded and returned.
    If save=True, recomputes from scratch and writes the parquet.
    """
    save_path = RESULTS_DIR / "decoding_error_corr_df.parquet"
    if save_path.exists() and not save:
        if verbose:
            print(f"Loading cached results from {save_path} ...")
        return pd.read_parquet(save_path)

    # pre-compute tuned neuron sets once (shared across sessions)
    distance_tuned, place_tuned = get_tuned_neurons()

    results_dfs = []
    for subject_ID in SUBJECT_IDS:
        for maze_name in maze_names:
            if verbose:
                print(f"Loading sessions for {subject_ID} - {maze_name} ...")
            sessions = gs.get_maze_sessions(
                subject_IDs=[subject_ID],
                maze_names=[maze_name],
                days_on_maze=days_on_maze,
                with_data=[
                    "navigation_df",
                    "cluster_metrics",
                    "trials_df",
                    "navigation_spike_counts_df",
                ],
                must_have_data=True,
            )
            if n_jobs and n_jobs != 1:
                dfs = Parallel(n_jobs=n_jobs)(
                    delayed(get_session_decoding_error_corr_df)(
                        s, distance_tuned=distance_tuned, place_tuned=place_tuned, verbose=False
                    )
                    for s in sessions
                )
            else:
                dfs = [
                    get_session_decoding_error_corr_df(
                        s, distance_tuned=distance_tuned, place_tuned=place_tuned, verbose=verbose
                    )
                    for s in sessions
                ]
            results_dfs.extend([d for d in dfs if d is not None and len(d) > 0])

    results_df = pd.concat(results_dfs, axis=0, ignore_index=True)
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(save_path)
        if verbose:
            print(f"Saved results to {save_path}.")
    return results_df


# %% session-level


def get_session_decoding_error_corr_df(
    session,
    distance_tuned=None,
    place_tuned=None,
    tower_or_bridge_modes=("tower", "bridge"),
    resolution=0.2,
    place_offset=2,
    max_steps_to_goal=16,
    min_rate=0.5,
    mindist=0.5,
    maxdist=7.5,
    mincount=15.5,
    min_trial_samples=2.5,
    reg_C=1e-1,
    good_session_chance_mult=2.0,
    verbose=True,
):
    """
    Returns a long-form per-sample DataFrame for one session, with decoder-triplet
    probabilities for both place and distance-to-goal. Runs the tower and bridge
    subsets separately (different discrete distance definitions) and concatenates.
    """
    if distance_tuned is None or place_tuned is None:
        distance_tuned, place_tuned = get_tuned_neurons()

    if verbose:
        print(f"{session.name}: loading input data ...")
    input_df = get_input_data(
        session,
        resolution=resolution,
        place_offset=place_offset,
        max_steps_to_goal=max_steps_to_goal,
    )

    rows = []
    for mode in tower_or_bridge_modes:
        if verbose:
            print(f"    {mode}")
        mode_df = _run_decoder_pass(
            session=session,
            input_df=input_df,
            tower_or_bridge=mode,
            distance_tuned=distance_tuned,
            place_tuned=place_tuned,
            resolution=resolution,
            min_rate=min_rate,
            mindist=mindist,
            maxdist=maxdist,
            mincount=mincount,
            min_trial_samples=min_trial_samples,
            reg_C=reg_C,
            good_session_chance_mult=good_session_chance_mult,
            verbose=verbose,
        )
        if mode_df is not None and len(mode_df) > 0:
            rows.append(mode_df)

    if not rows:
        return None
    return pd.concat(rows, axis=0, ignore_index=True)


def _run_decoder_pass(
    session,
    input_df,
    tower_or_bridge,
    distance_tuned,
    place_tuned,
    resolution,
    min_rate,
    mindist,
    maxdist,
    mincount,
    min_trial_samples,
    reg_C,
    good_session_chance_mult,
    verbose,
):
    """One tower-or-bridge pass of the session-level analysis."""
    dist_func, bias, subset_mask_fn = _get_dist_func(session, tower_or_bridge)

    locs_all = np.array(input_df["future"][0])
    keep = subset_mask_fn(locs_all)
    if keep.sum() == 0:
        return None
    train_df = input_df.loc[keep, :]

    # core arrays (mirrors kris 170-185)
    locs = np.array(train_df["future"][0])
    next_locs = np.array(train_df["future"][2])
    prev_locs = np.array(train_df["past"][2])
    trial = np.array(train_df["trial"]).astype(int)
    goals = np.array(train_df["goal"])
    new_true_dists = np.array(train_df["distance_to_goal"]["geodesic"])

    spikes = np.array(train_df["spike_count"])
    units = np.array(train_df["spike_count"].columns)
    keep_units = np.where(spikes.mean(0) / resolution > min_rate)[0]
    spikes = np.sqrt(spikes[:, keep_units])
    units = units[keep_units]

    dist_tuned_mask = np.array([u in distance_tuned for u in units])
    loc_tuned_mask = np.array([u in place_tuned for u in units])
    if loc_tuned_mask.sum() == 0 or dist_tuned_mask.sum() == 0:
        if verbose:
            print(
                f"    skipping {tower_or_bridge}: "
                f"{loc_tuned_mask.sum()} place-tuned, {dist_tuned_mask.sum()} distance-tuned units"
            )
        return None

    # discrete step distances
    dists = np.array([dist_func(locs[i], goals[i]) for i in range(len(goals))]).astype(float)
    dists_next = np.array([dist_func(next_locs[i], goals[i]) for i in range(len(goals))]).astype(float)
    dists_prev = np.array([dist_func(prev_locs[i], goals[i]) for i in range(len(goals))]).astype(float)
    dirs = np.sign(dists - dists_next)
    true_dist_residual = (dists + bias) * 0.18 - new_true_dists

    # per-sample filters
    cond_dist = (dists < maxdist) & (dists > mindist)
    cond_dist_next = (dists_next < maxdist) & (dists_next > mindist)
    cond_dist_prev = (dists_prev < maxdist) & (dists_prev > mindist)
    unique_locs, counts = np.unique(locs, return_counts=True)
    keep_locs = unique_locs[counts > mincount]
    cond_loc = np.array([loc in keep_locs for loc in locs])
    cond_next = np.array([n in keep_locs for n in next_locs])
    cond_prev = np.array([p in keep_locs for p in prev_locs])
    cond_all = cond_dist & cond_dist_prev & cond_dist_next & cond_loc & cond_next & cond_prev

    # decoder training/test sets per variable (kris 211-216)
    Xloc, yloc, trials_loc = spikes[cond_loc, :][:, loc_tuned_mask], locs[cond_loc], trial[cond_loc]
    Xdist, ydist, trials_dist = spikes[cond_dist, :][:, dist_tuned_mask], dists[cond_dist], trial[cond_dist]

    if len(yloc) == 0 or len(ydist) == 0:
        return None

    ulocs = np.unique(yloc)
    udists = np.unique(ydist)
    yloc_id = (yloc[:, None] == ulocs[None, :]).astype(float).argmax(-1)
    ydist_id = (ydist[:, None] == udists[None, :]).astype(float).argmax(-1)
    n_loc_classes = len(ulocs)
    n_dist_classes = len(udists)
    place_chance = 1.0 / n_loc_classes
    dist_chance = 1.0 / n_dist_classes

    # LOO per-trial CV (kris 222-259)
    trial_ids, tcounts = np.unique(trial[cond_loc & cond_dist], return_counts=True)
    test_trials = trial_ids[tcounts >= min_trial_samples]

    accs_loc, accs_dist = [], []
    sample_records = []
    for test_trial in test_trials:
        train_loc_mask = trials_loc != test_trial
        train_dist_mask = trials_dist != test_trial
        if train_loc_mask.sum() == 0 or (~train_loc_mask).sum() == 0:
            continue
        if train_dist_mask.sum() == 0 or (~train_dist_mask).sum() == 0:
            continue

        clf_loc = LogisticRegression(random_state=0, class_weight="balanced", C=reg_C, max_iter=500).fit(
            Xloc[train_loc_mask], yloc_id[train_loc_mask]
        )
        clf_dist = LogisticRegression(random_state=0, class_weight="balanced", C=reg_C, max_iter=500).fit(
            Xdist[train_dist_mask], ydist_id[train_dist_mask]
        )

        accs_loc.append(clf_loc.score(Xloc[~train_loc_mask], yloc_id[~train_loc_mask]))
        accs_dist.append(clf_dist.score(Xdist[~train_dist_mask], ydist_id[~train_dist_mask]))

        # per-sample triplet probs (only on full cond_all samples in this trial)
        sample_idx = np.where((trial == test_trial) & cond_all)[0]
        if len(sample_idx) == 0:
            continue

        loc_probs = clf_loc.predict_proba(spikes[sample_idx][:, loc_tuned_mask])
        dist_probs = clf_dist.predict_proba(spikes[sample_idx][:, dist_tuned_mask])

        # map triplet labels to decoder class columns
        loc_triplets = np.stack([prev_locs[sample_idx], locs[sample_idx], next_locs[sample_idx]], axis=1)
        dist_triplets = np.stack([dists_prev[sample_idx], dists[sample_idx], dists_next[sample_idx]], axis=1)
        loc_cols = (loc_triplets[..., None] == ulocs[None, None, :]).astype(float).argmax(-1)
        dist_cols = (dist_triplets[..., None] == udists[None, None, :]).astype(float).argmax(-1)

        for k, i in enumerate(sample_idx):
            p_loc = loc_probs[k][loc_cols[k]]
            p_dist = dist_probs[k][dist_cols[k]]
            sample_records.append(
                {
                    "trial": int(trial[i]),
                    "p_loc_prev": p_loc[0],
                    "p_loc_cur": p_loc[1],
                    "p_loc_next": p_loc[2],
                    "p_dist_prev": p_dist[0],
                    "p_dist_cur": p_dist[1],
                    "p_dist_next": p_dist[2],
                    "dist_cur": dists[i],
                    "dist_next": dists_next[i],
                    "dist_prev": dists_prev[i],
                    "dirs": dirs[i],
                    "true_dist_residual": true_dist_residual[i],
                }
            )

    if not sample_records:
        return None

    out = pd.DataFrame.from_records(sample_records)
    place_acc_mean = float(np.mean(accs_loc)) if accs_loc else np.nan
    dist_acc_mean = float(np.mean(accs_dist)) if accs_dist else np.nan
    out["place_acc_mean"] = place_acc_mean
    out["place_chance"] = place_chance
    out["dist_acc_mean"] = dist_acc_mean
    out["dist_chance"] = dist_chance
    out["good_session"] = place_acc_mean > good_session_chance_mult * place_chance
    out["tower_or_bridge"] = tower_or_bridge
    out["subject_ID"] = session.subject_ID
    out["maze_name"] = session.maze_name
    out["day_on_maze"] = session.day_on_maze
    out["session_name"] = session.name
    return out


# %% input data


def get_input_data(
    session,
    resolution=0.2,
    place_offset=2,
    max_steps_to_goal=16,
):
    """Downsampled, filtered nav + spike_count DataFrame with past/future place offsets."""
    navigation_df = session.navigation_df.copy()
    spike_counts_df = session.navigation_spike_counts_df.copy()
    spike_counts_df.reset_index(inplace=True, drop=True)
    spike_counts_df.columns = pd.MultiIndex.from_tuples([(*c, "") for c in spike_counts_df.columns])

    keep_clusters = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=True,
    )
    spike_counts_df = spike_counts_df[
        spike_counts_df.columns[spike_counts_df.columns.get_level_values(1).isin(keep_clusters)]
    ]

    ds_nav_df, ds_spike_counts_df = ds.downsample_nav_spikes_data(
        navigation_df,
        spike_counts_df,
        resolution=resolution,
        distance_metrics=[("steps_to_goal", "future"), ("distance_to_goal", "geodesic")],
    )
    ds_nav_df.columns = pd.MultiIndex.from_tuples([(*c, "") for c in ds_nav_df.columns])
    input_df = pd.concat([ds_nav_df, ds_spike_counts_df], axis=1).reset_index(drop=True)

    input_df[("place_direction", "", "")] = input_df.maze_position.simple + "_" + input_df.cardinal_movement_direction
    offset_df = fd.get_past_and_future_states(
        input_df, state_type="place", past_offset=place_offset, future_offset=place_offset
    )
    offset_df.columns = pd.MultiIndex.from_tuples([(*col, "") for col in offset_df.columns])
    input_df = pd.concat([input_df, offset_df], axis=1).sort_index(axis=1)

    input_df = filt.filter_navigation_rates_df(
        input_df,
        navigation_only=True,
        moving_only=True,
        exclude_time_at_goal=True,
        max_steps_to_goal=max_steps_to_goal,
    )
    input_df = input_df.droplevel(2, axis=1)
    return input_df


# %% tuned neuron selection (duplicated from double_decoding_simple for independence)


def get_tuned_neurons():
    """
    Cluster IDs of neurons selectively tuned (via neGLM variance-explained) to distance-to-goal
    or place-direction but not both.
    """
    feature_tuned_df = ve.get_feature_tuned_df(
        lms.load_model_set_cv_scores("variance_explained_multiunit"),
        reduced_models=["remove_distance_to_goal", "remove_place_direction"],
    )
    distance_tuned = (
        feature_tuned_df[feature_tuned_df.distance_to_goal & ~feature_tuned_df.place_direction]
        .index.get_level_values(1)
        .values
    )
    place_tuned = (
        feature_tuned_df[~feature_tuned_df.distance_to_goal & feature_tuned_df.place_direction]
        .index.get_level_values(1)
        .values
    )
    return distance_tuned, place_tuned


# %% maze / distance helpers


def _get_dist_func(session, tower_or_bridge):
    """
    Returns (dist_func, bias, subset_mask_fn) for the given tower/bridge mode.
    - tower: dist = shortest path in integer steps between tower nodes.
    - bridge: dist = min over the two adjacent towers; +0.5 step bias for bridge midpoints.
    """
    maze = session.simple_maze()
    nodes = {node[1]["label"]: node[0] for node in maze.nodes.items()}
    base_dist = lambda loc, goal: nx.shortest_path_length(maze, nodes[loc], nodes[goal], weight=None)

    if tower_or_bridge == "tower":

        def dist_func(loc, goal):
            return base_dist(loc, goal) if len(str(loc)) == 2 else np.nan

        bias = 0.0
        subset_mask_fn = lambda arr: np.array([len(loc) == 2 for loc in arr])
    elif tower_or_bridge == "bridge":

        def dist_func(loc, goal):
            if len(str(loc)) != 5:
                return np.nan
            return np.amin([base_dist(adj, goal) for adj in loc.split("-")])

        bias = 0.5
        subset_mask_fn = lambda arr: np.array([len(loc) == 5 for loc in arr])
    else:
        raise ValueError(f"tower_or_bridge must be 'tower' or 'bridge', got {tower_or_bridge!r}")

    return dist_func, bias, subset_mask_fn
