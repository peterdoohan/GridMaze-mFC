"""
Independent test for theta-modulation of place-direction tuning using tuning curves
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

from GridMaze.maze import representations as mr
from GridMaze.maze import plotting as mp

from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc
from GridMaze.analysis.processing.get_navigation_df import get_cardinal_movement_direction

from GridMaze.analysis.theta_mod import theta_utils as tu


# %% Global variables
FRAME_RATE = 60

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "theta_mod" / "place_direction_tuning"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

# %% Functionss


def test(
    results_df,
    slope_thres=0.0075,
    fr_thres=0.005,
    r2_thres=0.6,
):
    """ """
    id_cols = ["cluster_unique_ID", "triplet_name"]
    df = results_df.set_index(id_cols)
    avg_mask = df.theta_phase.isna()
    avg_df, theta_df = df[avg_mask], df[~avg_mask]
    dfs = []
    for _sign in ["pos", "neg"]:
        fit_mask = avg_df.r2.gt(r2_thres)
        if _sign == "pos":
            fr_mask = avg_df.loc_3.gt(fr_thres)
            slope_mask = avg_df.slope.gt(slope_thres)
        else:
            fr_mask = avg_df.loc_1.gt(fr_thres)
            slope_mask = avg_df.slope.lt(-slope_thres)
        keep_triplets = avg_df[fit_mask & fr_mask & slope_mask].index
        _avg = avg_df.loc[keep_triplets]
        _theta = theta_df.loc[keep_triplets]
        theta_metric = _theta.pivot_table(
            index=["cluster_unique_ID", "triplet_name"], columns="theta_phase", values="loc_2"
        ).sort_index(axis=0)
        avg_metric = _avg["loc_2"].sort_index(axis=0)
        theta_mod = theta_metric.sub(avg_metric, axis=0).div(avg_metric, axis=0)
        dfs.append(theta_mod)
    df = pd.concat([dfs[0].mul(-1), dfs[1]], axis=0)
    df.mean().plot()
    plt.show()
    dfs[0].mean().plot()
    plt.show()
    dfs[1].mean().plot()


def test2(
    results_df,
    slope_thres=0.0075,
    fr_thres=0.0075,
    r2_thres=0.8,
):
    id_cols = ["cluster_unique_ID", "triplet_name"]
    fr_cols = ["loc_1", "loc_2", "loc_3"]
    df = results_df.set_index(id_cols)
    avg_mask = df.theta_phase.isna()
    avg_df, theta_df = df[avg_mask], df[~avg_mask]
    dfs = []
    for _sign in ["pos", "neg"]:
        fit_mask = avg_df.r2.gt(r2_thres)
        fr_mask = avg_df.loc_2.gt(fr_thres)
        if _sign == "pos":
            slope_mask = avg_df.slope.gt(slope_thres)
        else:
            slope_mask = avg_df.slope.lt(-slope_thres)
        keep_triplets = avg_df[fit_mask & fr_mask & slope_mask].index
        _avg = avg_df.loc[keep_triplets][["subject_ID"] + fr_cols]
        _theta = theta_df.loc[keep_triplets][["subject_ID", "theta_phase"] + fr_cols]
        theta_norm = _theta.copy()
        _avg_broadcast = _avg.reindex(_theta.index)
        theta_norm[fr_cols] = _theta[fr_cols] - _avg_broadcast[fr_cols]  # / _avg_broadcast
        avg_mod = theta_norm.groupby(["subject_ID", "theta_phase"]).mean()["loc_2"].unstack(level=1)
        dfs.append(avg_mod)
        # tu.plot_decoding_bias(avg_mod, norm=False, print_stats=True, ylabel="tuning offset (au)")

        # _theta.groupby(["subject_ID", "theta_phase"]).mean().groupby(level=1).mean().T.plot(cmap="bwr", legend=False)
        # _avg.groupby("subject_ID").mean().mean(axis=0).plot(color="k")
        # plt.show()
        plot_triplet_phase_offsets(_theta, _avg)

    combined_mod = (dfs[1].mul(-1) + dfs[0]) / 2
    tu.plot_decoding_bias(
        combined_mod,
        color="darkred",
        norm=False,
        print_stats=True,
        ylabel="tuning offset (au)",
    )
    return


def plot_triplet_phase_offsets(_theta, _avg, cmap="coolwarm", with_error=False, ax=None):
    # set up fig
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("position")
    ax.set_ylabel("Norm Firing Rate")

    # process data
    theta = _theta.groupby(["subject_ID", "theta_phase"]).mean()
    theta_grouped = theta.groupby(level=1)
    theta_mean = theta_grouped.mean().T
    theta_sem = theta_grouped.sem().T
    avg_grouped = _avg.groupby("subject_ID").mean()
    avg_mean = avg_grouped.mean().values
    avg_sem = avg_grouped.sem().values

    # plot
    phases = theta_mean.columns
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(phases)))
    for phase, color in zip(phases, colors):
        _mean = theta_mean[phase].values
        ax.plot([1, 2, 3], _mean, color=color, lw=1, alpha=0.8)
        if with_error:
            _sem = theta_sem[phase].values
            ax.fill_between([1, 2, 3], _mean - _sem, _mean + _sem, color=color, alpha=0.3)
    ax.plot([1, 2, 3], avg_mean, color="k", linewidth=1)
    if with_error:
        ax.fill_between([1, 2, 3], avg_mean - avg_sem, avg_mean + avg_sem, color="k", alpha=0.3)


# %%


def get_place_direction_triplet_summary_df(
    split_half_corr_thres=0.6,
    min_occ=1,
    norm="sum",
    save=False,
    verbose=True,
    n_jobs=-1,
):
    """ """
    save_path = RESULTS_DIR / f"place_direction_triplet_summary_df2.parquet"
    if not save and save_path.exists():
        if verbose:
            print(f"loading existing summary df from disk...")
        return pd.read_parquet(save_path)
    dfs = []
    for subject in SUBJECT_IDS:
        for maze_name in ["maze_1", "maze_2", "rooms_maze"]:
            if verbose:
                print(f"Processing subject: {subject}")
            sessions = gs.get_maze_sessions(
                subject_IDs=[subject],
                maze_names=[maze_name],
                days_on_maze="all",
                with_data=[
                    "cluster_place_direction_tuning_metrics",
                    "navigation_df",
                    "navigation_theta_spike_counts_df",
                ],
                must_have_data=True,
            )
            if n_jobs:
                _dfs = Parallel(n_jobs=n_jobs)(
                    delayed(get_session_triplet_df)(
                        session,
                        split_half_corr_thres=split_half_corr_thres,
                        min_occ=min_occ,
                        norm=norm,
                        plot_summary=False,
                        verbose=verbose,
                    )
                    for session in sessions
                )
            else:
                _dfs = []
                for session in sessions:
                    if verbose:
                        print(session.name)
                    df = get_session_triplet_df(
                        session,
                        split_half_corr_thres=split_half_corr_thres,
                        min_occ=min_occ,
                        norm=norm,
                        plot_summary=False,
                        verbose=verbose,
                    )
                    _dfs.append(df)
            dfs.extend(_dfs)
    summary_df = pd.concat(dfs, axis=0).reset_index(drop=True)
    if save:
        if verbose:
            print(f"saving summary df to {save_path}")
        summary_df.to_parquet(save_path)
    return summary_df


def get_session_triplet_df(
    session,
    split_half_corr_thres=0.6,
    min_occ=1,
    norm="sum",
    plot_summary=False,
    verbose=True,
):
    # load data
    simple_maze = session.simple_maze()
    place_dir_metrics = session.cluster_place_direction_tuning_metrics.copy()
    navigation_df = session.navigation_df.copy()
    spikes_df = session.navigation_theta_spike_counts_df.reset_index(drop=True)
    theta_phases = spikes_df.columns.get_level_values(2).unique()
    # filter for clusters with strong pd tuning
    consider_clusters = place_dir_metrics[
        place_dir_metrics.split_half_corr.value.gt(split_half_corr_thres)
    ].index.values
    if len(consider_clusters) == 0:
        if verbose:
            print(f"No place-dir. tuned cluster for session: {session.name}")
        return
    spikes_df = spikes_df[spikes_df.columns[spikes_df.columns.get_level_values(1).isin(consider_clusters)]]
    # combine spikes and nav data
    navigation_df.columns = pd.MultiIndex.from_tuples([(*col, "") for col in navigation_df.columns])
    nav_spikes_df = pd.concat([navigation_df, spikes_df], axis=1).copy()
    # filter data same as normal place-direction heatmaps
    nav_spikes_df = filt.filter_navigation_rates_df(
        nav_spikes_df, navigation_only=True, moving_only=True, exclude_time_at_goal=True, max_steps_to_goal=30
    )
    nav_spikes_df = nav_spikes_df.reset_index(drop=True).sort_index(axis=0)
    theta_nav_spikes_df = nav_spikes_df.copy()  # df with spikes counts per theta phase
    # get df with avg spike counts across theta phases
    avg_theta_spikes = nav_spikes_df.spike_count.T.groupby(level=0).mean().T
    avg_theta_spikes.columns = pd.MultiIndex.from_tuples([("spike_count", col) for col in avg_theta_spikes.columns])
    avg_nav_spikes_df = pd.concat(
        [
            nav_spikes_df.drop("spike_count", level=0, axis=1).droplevel(2, axis=1),
            avg_theta_spikes,
        ],
        axis=1,
    )
    avg_nav_spikes_df = avg_nav_spikes_df.sort_index(axis=1)
    theta_nav_spikes_df = theta_nav_spikes_df.sort_index(axis=1)
    # set some occupancy threshold for including place-dirs. (same as tuning curves)
    group_cols = [("maze_position", "simple"), ("cardinal_movement_direction", "")]
    pd_occ = avg_nav_spikes_df.groupby(group_cols).size() * (1 / FRAME_RATE)
    occ_mask = pd_occ.gt(min_occ)
    # get theta phase and theta avg pd tuning
    theta_pd_tuning = theta_nav_spikes_df.groupby(group_cols).spike_count.sum().spike_count
    avg_pd_tuning = avg_nav_spikes_df.groupby(group_cols).spike_count.sum().spike_count
    # filter for min occupancy
    pd_occ = pd_occ[occ_mask]
    theta_pd_tuning = theta_pd_tuning[occ_mask]
    avg_pd_tuning = avg_pd_tuning[occ_mask]
    # convert to rates (spikes/s)
    theta_pd_tuning = theta_pd_tuning.div(pd_occ, axis=0)
    avg_pd_tuning = avg_pd_tuning.div(pd_occ, axis=0)
    # get all pd triplets
    pd_triplets = get_place_direction_triplets(session)
    # filter triplerts based on if they include all valid place-dirs
    valid_pds = avg_pd_tuning.index.to_list()
    valid_triplets = [t for t in pd_triplets if all(pd in valid_pds for pd in t)]
    xs = np.array([1, 2, 3])  # x values for linear fit of tuning across triplet
    results = []
    for cluster in consider_clusters:
        if verbose:
            print(cluster)
        theta_tuning = theta_pd_tuning[cluster]
        avg_tuning = avg_pd_tuning[cluster]
        # normalise avg tuning and apply same norm to each theta phase
        if norm == "sum":
            _sum = avg_tuning.sum()
            avg_tuning = avg_tuning / _sum
            theta_tuning = theta_tuning / _sum
        elif norm == "max":
            _max = avg_tuning.max()
            avg_tuning = avg_tuning / _max
            theta_tuning = theta_tuning / _max
        # loop over tripplets to calc slopes etc.
        theta_results = []
        avg_results = []
        for triplet in valid_triplets:
            triplet_name = ".".join(["_".join(t) for t in triplet])
            # process theta avg
            _avg = avg_tuning.loc[triplet]
            slope, intr, r_squared = fit_triplet(xs, _avg.values)
            avg_results.append(
                {
                    "cluster_unique_ID": cluster,
                    "triplet_name": triplet_name,
                    "theta_phase": np.nan,  # no assigned phase == average
                    "slope": slope,
                    "intercept": intr,
                    "r2": r_squared,
                    "loc_1": _avg.iloc[0],
                    "loc_2": _avg.iloc[1],
                    "loc_3": _avg.iloc[2],
                }
            )
            for theta_phase in theta_phases:
                _theta = theta_tuning.loc[triplet, theta_phase]
                slope, intr, r_squared = fit_triplet(xs, _theta.values)
                theta_results.append(
                    {
                        "cluster_unique_ID": cluster,
                        "triplet_name": triplet_name,
                        "theta_phase": theta_phase,
                        "slope": slope,
                        "intercept": intr,
                        "r2": r_squared,
                        "loc_1": _theta.iloc[0],
                        "loc_2": _theta.iloc[1],
                        "loc_3": _theta.iloc[2],
                    }
                )
        avg_df = pd.DataFrame(avg_results)
        theta_df = pd.DataFrame(theta_results)
        results.append(avg_df)
        results.append(theta_df)
        if plot_summary:
            _plot_cluster_summary(
                simple_maze,
                avg_tuning,
                avg_df,
            )
    results_df = pd.concat(results, axis=0).reset_index(drop=True)
    results_df["subject_ID"] = session.subject_ID
    results_df["maze_name"] = session.maze_name
    results_df["day_on_maze"] = session.day_on_maze
    return results_df


def fit_triplet(x, y):
    slope, intr = np.polyfit(x, y, 1)
    y_pred = np.polyval([slope, intr], x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    # Avoid division by zero
    if ss_tot == 0:
        r_squared = np.nan
    else:
        r_squared = 1 - (ss_res / ss_tot)

    return slope, intr, r_squared


def _plot_cluster_summary(
    simple_maze,
    avg_tuning,
    avg_df,
    slope_thres=0.0075,
    fr_thres=0.005,
    r2_thres=0.6,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    mp.plot_directed_heatmap(simple_maze, avg_tuning, colorbar=False, ax=axes[0])
    # plot pos slopes
    df = avg_df[avg_df.loc_2.gt(fr_thres)].copy()
    pos_df = df[df.slope.gt(slope_thres) & df.r2.gt(r2_thres)].copy()
    neg_df = df[df.slope.lt(-slope_thres) & df.r2.gt(r2_thres)].copy()
    xs = [1, 2, 3]
    for df, ax in zip([pos_df, neg_df], axes[1:]):
        if df.empty:
            ax.set_title("No valid triplets")
            continue
        for i, row in df.iterrows():
            triplet = [tuple(t.split("_")) for t in row.triplet_name.split(".")]
            ys = avg_tuning.loc[triplet].values
            ax.plot(xs, ys, alpha=0.5, label=row.triplet_name)
        ax.legend(fontsize="small")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("tower/edge")
        ax.set_ylabel("normalized firing rate")
    fig.tight_layout()


def get_place_direction_triplets(session):
    """ """
    # load maze and defs
    simple_maze = session.simple_maze()
    extended_maze = mr.get_extended_simple_maze(simple_maze)
    coord2label = mr.get_maze_coord2label(simple_maze)

    # get all connected quadruples of nodes
    node_seqs = []
    nodes = list(extended_maze.nodes)
    for source in nodes:
        for target in nodes:
            if source != target:
                for path in nx.all_simple_paths(extended_maze, source, target, cutoff=3):
                    if len(path) == 4:
                        node_seqs.append(path)

    # build as lists of pd pairs
    pd_triplets = []
    for node_seq in node_seqs:
        trip = []
        for i in range(3):
            n1 = node_seq[i]
            n2 = node_seq[i + 1]
            dir = get_cardinal_movement_direction(n2, n1)
            trip.append((coord2label[n1], dir))
        pd_triplets.append(trip)

    return pd_triplets
