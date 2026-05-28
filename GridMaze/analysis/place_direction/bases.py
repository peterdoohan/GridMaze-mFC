"""
Library for generating place-direction bases used in encoding analyses.

Per-cluster place-direction heatmaps (one row per cluster, one column per
"{position}_{direction}" pair, row-normalised to sum to 1) are precomputed once
across all sessions and cached per-maze on disk. To get bases for a given
session, load the cached heatmaps for that maze, drop rows belonging to the
session under analysis (optionally restrict to same/other subject and/or
late-only sessions), then fit PCA / NMF on the remaining rows.

@peterdoohan
"""

# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.decomposition import NMF, PCA

from GridMaze.maze import plotting as mp
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.cluster_tuning.spatial import _get_place_direction_df

# %% Global Variables

from GridMaze.paths import RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "place_direction" / "bases"

METADATA_COLS = ["session_name", "subject_ID", "day_on_maze", "late_session"]

# Generation parameters locked at heatmap-cache time. Regenerate with save=True
# if these ever need to change.
HEATMAP_KWARGS = dict(
    navigation_only=True,
    moving_only=True,
    exclude_time_at_goal=True,
    minimum_occupancy=0.5,
    max_steps_from_goal=30,
)


# %% Big df: one parquet per maze


def get_pd_heatmaps_df(maze_name, save=False, verbose=False, n_jobs=-1):
    """
    Load or compute the big per-cluster place-direction heatmaps df for one maze.

    Rows: clusters (across all subjects, all days, all sessions on this maze).
    Index: cluster_unique_ID.
    Columns: 2-level MultiIndex — metadata at ("session_name", ""), ("subject_ID", ""),
             ("day_on_maze", ""), ("late_session", "") and PD pairs at (pos, dir).
             PD values are RAW firing rates (NaNs intact, not normalised).
             NaN-fill and normalisation happen later in `fit_pd_bases`.
    """
    save_path = RESULTS_DIR / f"{maze_name}_pd_heatmaps.parquet"
    if not save and save_path.exists():
        if verbose:
            print(f"Loading cached heatmaps from {save_path}")
        return pd.read_parquet(save_path)
    if verbose:
        print(f"Computing PD heatmaps for {maze_name} ...")
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        maze_names=[maze_name],
        days_on_maze="all",
        with_data=["navigation_df", "cluster_metrics", "navigation_spike_rates_df"],
        must_have_data=True,
    )
    if n_jobs in (None, 1):
        dfs = [_compute_session_pd_heatmaps(s) for s in sessions]
    else:
        dfs = Parallel(n_jobs=n_jobs, verbose=int(verbose))(delayed(_compute_session_pd_heatmaps)(s) for s in sessions)
    dfs = [d for d in dfs if d is not None and not d.empty]
    if not dfs:
        raise RuntimeError(f"No heatmaps computed for {maze_name}")
    big_df = pd.concat(dfs, axis=0)
    if save:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Saving to {save_path}")
        big_df.to_parquet(save_path)
    return big_df


def _compute_session_pd_heatmaps(session):
    """
    Raw per-cluster PD heatmaps + metadata block for one session.

    Stores firing rates as returned by `_get_place_direction_df` (NaNs intact, no
    normalisation). NaN-fill and per-cluster normalisation are deferred to
    `fit_pd_bases` so different downstream analyses can choose their own processing
    without regenerating the cache.

    Returns a df with cluster_unique_ID index and 2-level MultiIndex columns:
        - PD pairs from _get_place_direction_df, e.g. ("A1", "N")
        - metadata at ("session_name", ""), ("subject_ID", ""), ("day_on_maze", ""),
          ("late_session", "")
    Returns None if no clusters survive `HEATMAP_KWARGS` filtering.
    """
    simple_maze = session.simple_maze()
    navigation_rates_df = session.get_navigation_activity_df(type="rates", cluster_kwargs={"single_units": True})
    heatmaps_df = _get_place_direction_df(simple_maze, navigation_rates_df, **HEATMAP_KWARGS)
    if heatmaps_df.empty:
        return None
    # prepend metadata as ("name", "") columns to keep a 2-level MultiIndex throughout
    metadata_df = pd.DataFrame(
        {
            ("session_name", ""): session.name,
            ("subject_ID", ""): session.subject_ID,
            ("day_on_maze", ""): session.day_on_maze,
            ("late_session", ""): session.late_session,
        },
        index=heatmaps_df.index,
    )
    return pd.concat([metadata_df, heatmaps_df], axis=1)


# %% Fit bases on a (filtered) heatmaps df


def fit_pd_bases(
    heatmaps_df,
    n_bases=8,
    dim_red="nmf",
    fill_nans="mean",
    normalisation="length",
    plot=False,
    simple_maze=None,
):
    """
    Fit NMF or PCA on the PD-heatmap rows of `heatmaps_df` (metadata cols ignored).

    `fill_nans` and `normalisation` follow
    dimensionality_reduction.get_session_place_direction_tuning (defaults match its
    defaults). Clusters with entirely-NaN heatmaps are dropped before fill so that
    `fill_nans="mean"` (which uses per-cluster mean) doesn't leave NaNs behind.

    Returns bases_df with index = MultiIndex of (pos, dir) tuples,
    columns = 0..n_bases-1.
    """
    pd_only = heatmaps_df.drop(columns=METADATA_COLS, level=0)
    # drop clusters with no PD coverage (row-mean would be NaN — fillna can't rescue)
    pd_only = pd_only.dropna(how="all")
    # fill NaN bins per cluster
    if fill_nans == "mean":
        pd_only.T.fillna(pd_only.mean(axis=1), inplace=True)
    elif fill_nans == "zero":
        pd_only = pd_only.fillna(0)
    elif fill_nans:
        raise ValueError(f"fill_nans must be 'mean', 'zero', or None/False; got {fill_nans!r}")
    # normalise per cluster
    if normalisation == "length":
        pd_only = pd_only.div(pd_only.pow(2).sum(axis=1).pow(0.5), axis=0)
    elif normalisation == "mean":
        pd_only = pd_only.div(pd_only.mean(axis=1), axis=0)
    elif normalisation == "max":
        pd_only = pd_only.div(pd_only.max(axis=1), axis=0)
    elif normalisation:
        raise ValueError(f"normalisation must be 'length', 'mean', 'max', or None/False; got {normalisation!r}")
    pd_cols = pd_only.columns
    data_matrix = pd_only.to_numpy()
    if dim_red == "pca":
        model = PCA(n_components=n_bases, random_state=0)
    elif dim_red == "nmf":
        model = NMF(
            n_components=n_bases,
            init="random",
            random_state=0,
            solver="mu",
            beta_loss="kullback-leibler",
            max_iter=10_000,
        )
    else:
        raise ValueError("dim_red must be 'pca' or 'nmf'")
    decomp_components = model.fit(data_matrix).components_
    bases_df = pd.DataFrame(data=decomp_components, index=range(n_bases), columns=pd_cols).T
    if plot:
        if simple_maze is None:
            raise ValueError("simple_maze required when plot=True")
        plot_bases(bases_df, simple_maze, dim_red=dim_red)
    return bases_df


# %% Convenience for the dominant call pattern


def get_session_pd_bases(
    session,
    n_bases=30,
    dim_red="pca",
    fill_nans="mean",
    normalisation="length",
    subject_filter=None,  # None | "same" | "other"
    late_only=False,
    save=False,
    verbose=False,
):
    """
    Bases for `session`: load cached heatmaps for session.maze_name, drop rows
    from this session, optionally filter by subject relationship / late-only,
    then fit PCA/NMF.
    """
    heatmaps_df = get_pd_heatmaps_df(session.maze_name, save=save, verbose=verbose)
    heatmaps_df = heatmaps_df[heatmaps_df[("session_name", "")] != session.name]
    if subject_filter == "same":
        heatmaps_df = heatmaps_df[heatmaps_df[("subject_ID", "")] == session.subject_ID]
    elif subject_filter == "other":
        heatmaps_df = heatmaps_df[heatmaps_df[("subject_ID", "")] != session.subject_ID]
    elif subject_filter is not None:
        raise ValueError(f"subject_filter must be None, 'same', or 'other'; got {subject_filter!r}")
    if late_only:
        heatmaps_df = heatmaps_df[heatmaps_df[("late_session", "")]]
    if heatmaps_df.empty:
        raise RuntimeError(
            f"No heatmaps remain after filtering for session {session.name} "
            f"(subject_filter={subject_filter}, late_only={late_only})"
        )
    return fit_pd_bases(
        heatmaps_df,
        n_bases=n_bases,
        dim_red=dim_red,
        fill_nans=fill_nans,
        normalisation=normalisation,
    )


# %% Plotting


def plot_bases(bases_df, simple_maze, dim_red="pca", axes=None):
    """Plot each basis as a directed heatmap over the maze."""
    if dim_red == "pca":
        cmap = "coolwarm"
        neg = True
    elif dim_red == "nmf":
        cmap = "silver2red"
        neg = False
    else:
        raise ValueError("dim_red must be 'pca' or 'nmf'")
    n_bases = bases_df.shape[1]
    if axes is None:
        fig, axes = plt.subplots(2, n_bases // 2, figsize=(80, 10), sharex=True)
        axes = axes.flatten()
    for i in range(n_bases):
        ax = axes[i]
        basis = bases_df[i]  # Series indexed by MultiIndex (pos, dir)
        mp.plot_directed_heatmap(
            simple_maze,
            basis,
            ax=ax,
            colormap=cmap,
            allow_negative=neg,
            silhouette_node_size=400,
            silhouette_edge_size=8,
        )
    fig.tight_layout()
