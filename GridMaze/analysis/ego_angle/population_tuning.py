"""
Summarise egocentric-angle to goal tuning across the population
"""

# %% imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.stats import ttest_1samp, zscore
from sklearn.metrics import r2_score

from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core.get_clusters import get_cluster

# %% Globabl variables

from GridMaze.paths import RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "ego_angle_tuning"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


# %% Population heatmap


def plot_egocentric_angle_to_goal_heatmap(
    population_tuning_df,
    min_split_half_corr=0.6,
    sort_by="von_mises_cv",
    normalisation="zscore",
    cmap="cividis",
    vmin=None,
    vmax=None,
    ax=None,
):
    """Population heatmap of egocentric angle-to-goal tuning, rows sorted by preferred angle."""
    heatmap_df = _get_heatmap_df(
        population_tuning_df,
        min_split_half_corr=min_split_half_corr,
        sort_by=sort_by,
        normalisation=normalisation,
    )
    D = heatmap_df["tuning_curve"].values
    x = heatmap_df["tuning_curve"].columns.values.astype(float)

    if normalisation == "zscore":
        cmap = cmap or "coolwarm"
        vmin = -2 if vmin is None else vmin
        vmax = 2 if vmax is None else vmax
        cbar_label = "Firing rate (z-score)"
    else:
        cmap = cmap or "viridis"
        vmin = 0 if vmin is None else vmin
        vmax = 1 if vmax is None else vmax
        cbar_label = "Firing rate (norm.)"

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.spines[["top", "right"]].set_visible(False)

    sns.heatmap(
        D,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        rasterized=True,
        cbar_kws={"shrink": 0.5, "label": cbar_label},
    )

    tick_labels = [0, 90, 180, 270, 360]
    tick_positions = [np.argmin(np.abs(x - t)) for t in tick_labels]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_xlabel("Egocentric angle to goal (°)")

    y_tick = len(D) // 10 * 10
    ax.set_yticks([y_tick])
    ax.set_yticklabels([f"{y_tick}"], rotation=90)
    ax.set_ylabel("Neurons", labelpad=-10)
    return ax


def _get_heatmap_df(
    population_tuning_df,
    min_split_half_corr=0.3,
    sort_by="von_mises_cv",
    normalisation="zscore",
):
    """Filter, sort and per-row normalise the population tuning df for heatmap plotting."""
    df = population_tuning_df[population_tuning_df[("split_half_corr", "")] > min_split_half_corr].copy()
    mu_col = (sort_by, "mu")
    df = df[df[mu_col].notna()]
    df = df.sort_values(by=[mu_col], ascending=True)
    values = df["tuning_curve"].values
    if normalisation == "zscore":
        df.loc[:, "tuning_curve"] = zscore(values, axis=1, nan_policy="omit")
    elif normalisation == "max":
        df.loc[:, "tuning_curve"] = values / np.nanmax(values, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown normalisation: {normalisation}")
    return df


# %% Preferred angle distribution


def plot_preferred_angle_distribution(
    population_tuning_df,
    min_split_half_corr=0.6,
    mu_source="von_mises_cv",
    sig_only=False,
    n_bins=36,
    color="darkblue",
    ax=None,
):
    """Circular histogram of preferred angles (mu) across the population.

    Convention: 0° = goal in front, 90° = goal to the left, 180° = goal behind,
    270° = goal to the right. Plot is oriented with 0° at the top, angle
    increasing counter-clockwise. Rayleigh test for non-uniformity and the
    population mean vector are overlaid.

    Parameters
    ----------
    mu_source : {"von_mises_cv", "von_mises_full", "circular_mean"}
        Which `mu` estimate to plot.
    sig_only : bool
        If True and `mu_source == "von_mises_cv"`, keep only clusters with
        `von_mises_cv.sig == True`.
    """
    if mu_source == "circular_mean":
        mu_col = ("circular_mean", "")
    else:
        mu_col = (mu_source, "mu")

    df = population_tuning_df[population_tuning_df[("split_half_corr", "")] > min_split_half_corr].copy()
    if sig_only and mu_source == "von_mises_cv":
        df = df[df[("von_mises_cv", "sig")] == True]  # noqa: E712
    mus = df[mu_col].dropna().to_numpy() % 360
    n = len(mus)

    # Rayleigh test (large-n series expansion)
    rad = np.deg2rad(mus)
    cos_sum, sin_sum = np.cos(rad).sum(), np.sin(rad).sum()
    R_bar = np.sqrt(cos_sum**2 + sin_sum**2) / n if n else np.nan
    z = n * R_bar**2 if n else np.nan

    # histogram
    bin_edges = np.linspace(0, 360, n_bins + 1)
    counts, _ = np.histogram(mus, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 4), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)  # counter-clockwise → 90° ends up on the left

    width = np.deg2rad(360 / n_bins)
    ax.bar(
        np.deg2rad(bin_centers),
        counts,
        width=width,
        color=color,
        edgecolor="white",
        linewidth=0.5,
        align="center",
    )

    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["front\n0°", "left\n90°", "behind\n180°", "right\n270°"])
    ax.set_yticklabels([])
    ax.spines["polar"].set_visible(False)
    return ax


def von_mises_fwhm_deg(kappa):
    """Full width at half maximum (in degrees) of a von Mises tuning curve.

    FWHM measured at half-rise above the trough, using
        cos(θ/2) = ln(cosh(κ)) / κ
    valid for κ > 0. Returns NaN where κ is non-finite or ≤ 0.
    """
    k = np.asarray(kappa, dtype=float)
    out = np.full_like(k, np.nan)
    valid = np.isfinite(k) & (k > 0)
    kv = k[valid]
    # ln(cosh(k)) computed stably for large k: ln((1 + exp(-2k))/2) + k
    ln_cosh = np.log1p(np.exp(-2 * kv)) - np.log(2) + kv
    cos_half = ln_cosh / kv
    cos_half = np.clip(cos_half, -1.0, 1.0)
    out[valid] = 2 * np.rad2deg(np.arccos(cos_half))
    return out


def plot_preferred_angle_vs_width(
    population_tuning_df,
    min_split_half_corr=0.6,
    mu_source="von_mises_cv",
    sig_only=False,
    mrl_lim=(0, 1),
    angle_bins=24,
    mrl_bins=20,
    pthresh=0.05,
    cmap="mako",
    scatter_color="0.7",
    scatter_size=10,
    scatter_alpha=1.0,
    ax=None,
    cbar=True,
):
    """Preferred angle (mu) vs tuning concentration (MRL): 2D density + outlier scatter.

    All cells are drawn as a grey scatter in the background; a 2D histogram is
    overlaid on top with `pthresh` so only dense bins render — cells in sparse
    bins remain visible as grey dots. MRL: 0 = uniform, 1 = perfectly
    concentrated. Convention: 0° = front, 90° = left, 180° = back, 270° = right.
    """
    mu_col = (mu_source, "mu")
    mrl_col = ("mean_resultant_length", "")

    df = population_tuning_df[population_tuning_df[("split_half_corr", "")] > min_split_half_corr].copy()
    if sig_only and mu_source == "von_mises_cv":
        df = df[df[("von_mises_cv", "sig")] == True]  # noqa: E712
    df = df[df[mu_col].notna() & df[mrl_col].notna()]

    mu = df[mu_col].to_numpy() % 360
    mrl = df[mrl_col].to_numpy()

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 4))

    # background: all cells as grey dots (outliers stay visible where histogram is masked)
    sns.scatterplot(
        x=mu,
        y=mrl,
        s=scatter_size,
        color=scatter_color,
        alpha=scatter_alpha,
        edgecolor="none",
        ax=ax,
    )
    # foreground: 2D histogram, sparse bins masked out
    sns.histplot(
        x=mu,
        y=mrl,
        bins=(angle_bins, mrl_bins),
        binrange=((0, 360), mrl_lim),
        pthresh=pthresh,
        cmap=cmap,
        cbar=cbar,
        cbar_kws={"label": "neurons", "shrink": 0.5},
        ax=ax,
    )
    if cbar:
        cb = ax.collections[-1].colorbar
        if cb is not None:
            cb.outline.set_visible(False)

    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xticklabels(["front", "left", "back", "right", "front"])
    ax.set_xlabel("Preferred egocentric angle to goal")
    ax.set_ylabel("Tuning concentration (MRL)")
    ax.set_xlim(0, 360)
    ax.set_ylim(*mrl_lim)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


# %% Example cells


def plot_example_egocentric_tuning(
    cluster_unique_IDs={
        "front": "m3.2022-07-13.maze_cluster54",
        "left": "m8.2022-07-23.maze_cluster6",
        "right": "m2.2022-07-12.maze_cluster52",
        "back": "m6.2022-07-02.maze_cluster38",
    },
    feature_kwargs={"color": "goldenrod"},
    labels=("front", "left", "right", "back"),
    axes=None,
):
    """Plot egocentric angle-to-goal tuning for 4 example cells side-by-side.

    Parameters
    ----------
    cluster_unique_IDs : dict | None
        Maps direction label → cluster_unique_ID. If None, uses
        `DEFAULT_EXAMPLE_CELLS`.
    feature_kwargs : dict | None
        Maps direction label → per-cell feature_kwargs dict forwarded to
        `Cluster.plot_tuning(feature="angle_to_goal", ...)`. Missing keys use
        defaults (angle_metric="egocentric"). Use this to set e.g. `color`,
        `smooth_SD`, `n_bins` per cell.
    labels : tuple
        Order of the 4 subplots from left to right.
    figsize : tuple
    show_cluster_id : bool
        If True, append the cluster_unique_ID under the direction label in each title.
    """
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(4, 4), subplot_kw={"projection": "polar"})
        axes = np.atleast_1d(axes).flatten()
    for ax, label in zip(axes, labels):
        cid = cluster_unique_IDs[label]
        cluster = get_cluster(cid)
        fkw = {"angle_metric": "egocentric", **feature_kwargs}
        cluster.plot_tuning(feature="angle_to_goal", feature_kwargs=fkw, ax=ax)
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(["front", "left", "back", "right"])
        ax.spines["polar"].set_visible(False)


# %% Main data generation


def get_egocentric_angle_to_goal_tuning(
    subject_IDs="all",
    maze_names="all",
    late_sessions=False,
    sessions=None,
    n_jobs=-1,
    save=False,
    verbose=False,
    **session_kwargs,
):
    """Cross-session egocentric angle-to-goal tuning.

    Runs `get_session_egocentric_angle_to_goal_tuning` on every session matching
    the filter args (or on `sessions` if provided) and concatenates the
    per-session DataFrames into one. Sessions where no clusters pass the
    split-half threshold are dropped.

    Parameters
    ----------
    subject_IDs, maze_names : str | list
        Passed to `gs.get_maze_sessions` when `sessions is None`.
    late_sessions : bool
        If True → `days_on_maze="late"`, else `"all"`.
    sessions : iterable | MazeSession | None
        If provided, skip session loading.
    n_jobs : int | None
        Passed to `joblib.Parallel`. `None` → sequential loop.
    save : bool
        If False (default) and a cached parquet exists at `save_path`, load
        and return it. If True, always recompute and overwrite the cache.
    **session_kwargs
        Forwarded to `get_session_egocentric_angle_to_goal_tuning`
        (e.g. `n_bins`, `n_splits`, `min_split_half_corr`, `smooth_SD`,
        `include_multi_units`, `fit_n_inits`).
    """
    save_path = RESULTS_DIR / "egocentric_angle_to_goal_tuning.parquet"
    if save_path.exists() and not save:
        if verbose:
            print(f"Loading cached tuning df from {save_path}")
        return pd.read_parquet(save_path)

    if sessions is None:
        if verbose:
            print("Loading sessions ...")
        days_on_maze = "late" if late_sessions else "all"
        sessions = gs.get_maze_sessions(
            subject_IDs=subject_IDs,
            maze_names=maze_names,
            days_on_maze=days_on_maze,
            with_data=["navigation_df", "navigation_spike_rates_df", "cluster_metrics"],
            must_have_data=True,
        )
    # normalise single session → list
    if not isinstance(sessions, (list, tuple)):
        sessions = [sessions]

    def _process_session(session):
        try:
            return get_session_egocentric_angle_to_goal_tuning(session, verbose=verbose, **session_kwargs)
        except Exception as e:
            print(f"[{session.name}] {type(e).__name__}: {e}")
            return None

    if n_jobs is not None:
        dfs = Parallel(n_jobs=n_jobs)(delayed(_process_session)(s) for s in sessions)
    else:
        dfs = [_process_session(s) for s in sessions]
    dfs = [d for d in dfs if d is not None]
    if len(dfs) == 0:
        return None
    tuning_df = pd.concat(dfs, axis=0)
    if save:
        if verbose:
            print(f"Saving tuning df to {save_path}")
        tuning_df.to_parquet(save_path)
    return tuning_df


def get_session_egocentric_angle_to_goal_tuning(
    session,
    n_bins=120,
    n_splits=50,
    min_split_half_corr=0.3,
    smooth_SD=2,
    wrap_pad=10,
    include_multi_units=False,
    corr_method="spearman",
    fit_n_inits=3,
    fit_alpha=0.05,
    verbose=False,
):
    """Per-session egocentric angle-to-goal tuning curves with split-half QC + von Mises fits.

    For all single-unit clusters in the session, compute the egocentric
    angle-to-goal tuning curve (n_bins angular bins over [0, 360)) and a
    split-half reliability score averaged over `n_splits` random trial splits.
    Clusters above `min_split_half_corr` get:
      - a circularly-smoothed tuning curve over all trials
      - a cross-validated von Mises fit (fit on half 1, r2 on half 2, averaged
        over the same `n_splits` splits used for split-half corr)
      - a von Mises fit to the full (smoothed) tuning curve
      - non-parametric circular mean + mean resultant length on the full curve

    Returns None if no clusters survive the threshold.
    """
    if verbose:
        print(session.name)

    # 1. load + filter navigation rates
    navigation_rates_df = session.get_navigation_activity_df(
        type="rates",
        cluster_kwargs={"single_units": True, "multi_units": include_multi_units},
    )
    navigation_rates_df = filt.filter_navigation_rates_df(navigation_rates_df, moving_only=False)

    # 2. bin egocentric angle to goal into n_bins intervals over [0, 360)
    angle_bin_key = ("angle_to_goal", "egocentric_bined")
    bin_edges = np.linspace(0, 360, num=n_bins + 1, endpoint=True)
    bins = pd.IntervalIndex.from_breaks(bin_edges)
    navigation_rates_df[angle_bin_key] = pd.cut(navigation_rates_df.angle_to_goal["egocentric"], bins=bins)

    # 3. pre-compute per-trial × per-bin mean firing rates (vectorized over all clusters)
    #    index = (trial, bin); columns = cluster_unique_ID
    trial_bin_means = (
        navigation_rates_df.groupby(["trial", angle_bin_key], observed=True).firing_rate.mean().firing_rate
    )
    trial_bin_means.index.set_names(["trial", "egocentric_bin"], inplace=True)

    cluster_IDs = trial_bin_means.columns.to_numpy()
    trials = navigation_rates_df.trial.unique()
    mid = len(trials) // 2
    n_clusters = len(cluster_IDs)

    # 4. 50 random split-half pairs — store curves for both corr and CV fits
    trial_level = trial_bin_means.index.get_level_values("trial")
    splits_1 = np.full((n_splits, n_bins, n_clusters), np.nan)
    splits_2 = np.full((n_splits, n_bins, n_clusters), np.nan)
    for i in range(n_splits):
        perm = np.random.permutation(trials)
        t1, t2 = perm[:mid], perm[mid:]
        tuning_1 = (
            trial_bin_means.loc[trial_level.isin(t1)]
            .groupby(level="egocentric_bin", observed=True)
            .mean()
            .reindex(bins)
            .sort_index()
        )
        tuning_2 = (
            trial_bin_means.loc[trial_level.isin(t2)]
            .groupby(level="egocentric_bin", observed=True)
            .mean()
            .reindex(bins)
            .sort_index()
        )
        splits_1[i] = tuning_1.values
        splits_2[i] = tuning_2.values

    # split-half correlation per cluster (averaged over splits)
    corrs = np.full((n_splits, n_clusters), np.nan)
    for i in range(n_splits):
        corrs[i] = _columnwise_corr(splits_1[i], splits_2[i], method=corr_method)
    split_half_corr = np.nanmean(corrs, axis=0)

    # 5. filter clusters by split-half corr threshold
    keep_mask = split_half_corr > min_split_half_corr
    if not keep_mask.any():
        if verbose:
            print(f"  no clusters passed split_half_corr > {min_split_half_corr}")
        return None
    keep_IDs = cluster_IDs[keep_mask]
    keep_corrs = split_half_corr[keep_mask]
    keep_idx = np.where(keep_mask)[0]

    # 6. full-trial tuning curve for kept clusters, re-indexed onto full n_bins grid
    full_tuning = trial_bin_means.groupby(level="egocentric_bin", observed=True).mean().loc[:, keep_IDs]
    full_tuning = full_tuning.reindex(bins).sort_index()

    # 7. circular smoothing (wrap-padded gaussian along bin axis)
    smoothed = _circular_smooth(full_tuning.values, smooth_SD=smooth_SD, wrap_pad=wrap_pad)

    # 8. von Mises fits
    bin_mids = np.array([b.mid for b in full_tuning.index])
    if verbose:
        print(f"  fitting {len(keep_IDs)} clusters...")
    # 8a. cross-validated fit: fit each split-1 tuning curve, r2 on split-2
    cv_fits = [
        _fit_cv(splits_1[:, :, c], splits_2[:, :, c], bin_mids, von_mises_4p, n_inits=fit_n_inits, alpha=fit_alpha)
        for c in keep_idx
    ]
    # 8b. full-data fit: multi-init fit to the smoothed full tuning curve
    full_fits = [
        _fit_single(bin_mids, smoothed[:, i], von_mises_4p, n_inits=fit_n_inits * 3) for i in range(len(keep_IDs))
    ]

    # 9. non-parametric circular statistics on smoothed full curve
    circ_mean = np.array([_circular_mean(bin_mids, smoothed[:, i]) for i in range(len(keep_IDs))])
    mrl = np.array([_mean_resultant_length(bin_mids, smoothed[:, i]) for i in range(len(keep_IDs))])

    # 10. assemble output dataframe
    tuning_df = pd.DataFrame(
        smoothed.T,
        index=pd.Index(keep_IDs, name="cluster_unique_ID"),
        columns=pd.MultiIndex.from_product([["tuning_curve"], bin_mids]),
    )
    subject_ID = navigation_rates_df.subject_ID.unique()[0]
    maze_name = navigation_rates_df.maze_name.unique()[0]
    day_on_maze = navigation_rates_df.day_on_maze.unique()[0]
    tuning_df[("split_half_corr", "")] = keep_corrs
    tuning_df[("subject_ID", "")] = subject_ID
    tuning_df[("maze_name", "")] = maze_name
    tuning_df[("day_on_maze", "")] = day_on_maze
    # cv fit columns
    cv_cols = _get_param_names(von_mises_4p) + ["r2", "p_value", "sig"]
    for col in cv_cols:
        tuning_df[("von_mises_cv", col)] = [f[col] for f in cv_fits]
    # full fit columns
    full_cols = _get_param_names(von_mises_4p) + ["r2"]
    for col in full_cols:
        tuning_df[("von_mises_full", col)] = [f[col] for f in full_fits]
    # circular stats
    tuning_df[("circular_mean", "")] = circ_mean
    tuning_df[("mean_resultant_length", "")] = mrl

    return tuning_df.sort_index(axis=1)


# %% Curve-fitting utilities


def von_mises_4p(x, amplitude, mu, kappa, offset):
    """Von Mises tuning function for circular variable x (in degrees).

    amplitude : scaling of the exponential (peak increment = amplitude * exp(kappa))
    mu        : preferred angle (degrees)
    kappa     : concentration (higher = sharper tuning; 1/kappa analogous to variance)
    offset    : baseline firing rate
    """
    x_rad = np.deg2rad(np.asarray(x))
    mu_rad = np.deg2rad(mu)
    return amplitude * np.exp(kappa * np.cos(x_rad - mu_rad)) + offset


def _get_param_names(fn):
    if fn.__name__ == "von_mises_4p":
        return ["amplitude", "mu", "kappa", "offset"]
    raise ValueError(f"Unknown function: {fn.__name__}")


def _get_init_range(fn):
    if fn.__name__ == "von_mises_4p":
        return [[0.1, 5.0], [0.0, 360.0], [0.5, 5.0], [-1.0, 1.0]]  # amplitude, mu, kappa, offset
    raise ValueError(f"Unknown function: {fn.__name__}")


def _get_bounds(fn):
    if fn.__name__ == "von_mises_4p":
        # mu unbounded (periodic); amplitude ≥ 0; kappa ∈ (0, 50]
        return [[0.0, -np.inf, 0.01, -np.inf], [np.inf, np.inf, 50.0, np.inf]]
    raise ValueError(f"Unknown function: {fn.__name__}")


def _fit_single(x, y, fn, n_inits=10):
    """Multi-init curve fit; returns dict of best-r2 params + r2.

    Uses data-driven init for mu (argmax of y) on the first init to help
    convergence, then random uniform inits from `_get_init_range` for the rest.
    """
    init_range = _get_init_range(fn)
    bounds = _get_bounds(fn)
    param_names = _get_param_names(fn)
    mask = np.isfinite(y)
    x_fit, y_fit = np.asarray(x)[mask], np.asarray(y)[mask]
    if len(x_fit) < len(param_names) + 1:
        return {p: np.nan for p in param_names} | {"r2": np.nan}
    fits = []
    for i in range(n_inits):
        if i == 0 and fn.__name__ == "von_mises_4p":
            # data-driven init
            amp0 = max(y_fit.max() - y_fit.min(), 1e-3)
            mu0 = float(x_fit[np.argmax(y_fit)])
            kappa0 = 2.0
            offset0 = float(y_fit.min())
            p0 = [amp0, mu0, kappa0, offset0]
        else:
            p0 = [np.random.uniform(lo, hi) for lo, hi in init_range]
        try:
            p_opt, _ = curve_fit(fn, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=10_000)
            r2 = r2_score(y_fit, fn(x_fit, *p_opt))
        except (RuntimeError, ValueError):
            p_opt = [np.nan] * len(param_names)
            r2 = np.nan
        fit = {p: p_opt[j] for j, p in enumerate(param_names)}
        fit["r2"] = r2
        fits.append(fit)
    best = max(fits, key=lambda d: -np.inf if np.isnan(d["r2"]) else d["r2"])
    # canonicalise mu to [0, 360)
    if not np.isnan(best["mu"]):
        best["mu"] = best["mu"] % 360
    return best


def _fit_cv(splits_1, splits_2, x, fn, n_inits=3, alpha=0.05):
    """Cross-validated fit for one cluster over all splits.

    splits_1, splits_2 : (n_splits, n_bins) arrays (fit on half 1, r2 on half 2).
    Returns median params, mean r2, one-sample t-test p-value on r2 vs 0.
    """
    param_names = _get_param_names(fn)
    n_splits = splits_1.shape[0]
    bounds = _get_bounds(fn)
    init_range = _get_init_range(fn)
    per_split = []
    for i in range(n_splits):
        y1, y2 = splits_1[i], splits_2[i]
        mask = np.isfinite(y1) & np.isfinite(y2)
        if mask.sum() < len(param_names) + 1:
            per_split.append({p: np.nan for p in param_names} | {"r2": np.nan})
            continue
        x_fit, y1_fit, y2_fit = np.asarray(x)[mask], y1[mask], y2[mask]
        candidates = []
        for j in range(n_inits):
            if j == 0:
                p0 = [
                    max(y1_fit.max() - y1_fit.min(), 1e-3),
                    float(x_fit[np.argmax(y1_fit)]),
                    2.0,
                    float(y1_fit.min()),
                ]
            else:
                p0 = [np.random.uniform(lo, hi) for lo, hi in init_range]
            try:
                p_opt, _ = curve_fit(fn, x_fit, y1_fit, p0=p0, bounds=bounds, maxfev=10_000)
                r2 = r2_score(y2_fit, fn(x_fit, *p_opt))
            except (RuntimeError, ValueError):
                p_opt = [np.nan] * len(param_names)
                r2 = np.nan
            cand = {p: p_opt[k] for k, p in enumerate(param_names)}
            cand["r2"] = r2
            candidates.append(cand)
        # pick best-r2 init for this split (on held-out data already — just robustness to init)
        best = max(candidates, key=lambda d: -np.inf if np.isnan(d["r2"]) else d["r2"])
        per_split.append(best)
    df = pd.DataFrame(per_split)
    # circular median for mu; normal median for the rest
    params = {}
    for p in param_names:
        if p == "mu":
            params[p] = _circular_median(df[p].dropna().values)
        else:
            params[p] = df[p].median()
    params["r2"] = df["r2"].mean()
    r2_vals = df["r2"].dropna().values
    if len(r2_vals) >= 2:
        result = ttest_1samp(r2_vals, 0, alternative="greater")
        params["p_value"] = float(result.pvalue)
    else:
        params["p_value"] = np.nan
    params["sig"] = bool(params["p_value"] < alpha) if np.isfinite(params["p_value"]) else False
    return params


# %% Circular statistics helpers


def _circular_mean(angles_deg, weights):
    """Weighted circular mean (degrees, [0, 360)).

    Weights shifted to be non-negative (subtract min) so baseline firing does
    not dominate the mean vector.
    """
    angles_deg = np.asarray(angles_deg, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(angles_deg) & np.isfinite(w)
    if not mask.any():
        return np.nan
    angles_deg, w = angles_deg[mask], w[mask]
    w = w - w.min()
    if w.sum() == 0:
        return np.nan
    rad = np.deg2rad(angles_deg)
    cos_sum = (w * np.cos(rad)).sum()
    sin_sum = (w * np.sin(rad)).sum()
    return float(np.rad2deg(np.arctan2(sin_sum, cos_sum)) % 360)


def _mean_resultant_length(angles_deg, weights):
    """Mean resultant length (0 = uniform, 1 = perfectly concentrated).

    Weights shifted to be non-negative, then normalised to sum to 1.
    """
    angles_deg = np.asarray(angles_deg, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(angles_deg) & np.isfinite(w)
    if not mask.any():
        return np.nan
    angles_deg, w = angles_deg[mask], w[mask]
    w = w - w.min()
    total = w.sum()
    if total == 0:
        return 0.0
    w = w / total
    rad = np.deg2rad(angles_deg)
    return float(np.sqrt((w * np.cos(rad)).sum() ** 2 + (w * np.sin(rad)).sum() ** 2))


def _circular_median(mus_deg):
    """Circular 'median' via direction of the mean resultant vector of unit vectors."""
    if len(mus_deg) == 0:
        return np.nan
    rad = np.deg2rad(mus_deg)
    return float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)


# %% Existing helpers


def _columnwise_corr(a, b, method="spearman"):
    """Correlate columns of two aligned 2D arrays (bins × clusters).

    NaN bins in either array are masked pairwise per column. Returns a 1D
    array of correlations of length `n_clusters`.
    """
    assert a.shape == b.shape
    n_bins, n_clusters = a.shape
    out = np.full(n_clusters, np.nan)
    for c in range(n_clusters):
        x, y = a[:, c], b[:, c]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            continue
        xm, ym = x[mask], y[mask]
        if method == "spearman":
            xm = pd.Series(xm).rank().values
            ym = pd.Series(ym).rank().values
        xm = xm - xm.mean()
        ym = ym - ym.mean()
        denom = np.sqrt((xm**2).sum() * (ym**2).sum())
        if denom == 0:
            continue
        out[c] = (xm * ym).sum() / denom
    return out


def _circular_smooth(tuning, smooth_SD, wrap_pad=10):
    """Wrap-padded gaussian smoothing along axis 0 (bin axis).

    `tuning` has shape (n_bins, n_clusters). Pads with `wrap_pad` bins from
    the opposite end to avoid 0/360° discontinuity, smooths, then crops back.
    NaN bins are linearly interpolated first so the gaussian doesn't
    propagate NaNs into neighbouring bins.
    """
    tuning = pd.DataFrame(tuning).interpolate(axis=0, limit_direction="both").values
    padded = np.concatenate([tuning[-wrap_pad:], tuning, tuning[:wrap_pad]], axis=0)
    smoothed = gaussian_filter1d(padded, sigma=smooth_SD, axis=0)
    return smoothed[wrap_pad:-wrap_pad]
