"""
Cross-pipeline phase-offset comparison between place and distance theta-mod decoding.
@peterdoohan
"""

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import circmean, wilcoxon
from pingouin import circ_rayleigh

from GridMaze.analysis.theta_mod import theta_utils as tmu
from GridMaze.analysis.theta_mod import place_direction_decoding as pdd
from GridMaze.analysis.theta_mod import distance_to_goal_decoder2 as ddv2

# %% Data loaders


def get_place_bias(
    distance_to_goal=None,
    decision_points=False,
    all_envelope_defined=True,
    min_chance_ratio=2.0,
):
    """Per-subject place-decoding bias along theta phase (m), per-subject mean-subtracted.

    Returns subjects × phases df. Filtering mirrors place_direction_decoding.plot_theta_mod_trajectory_error.
    """
    summary_df = pdd.get_theta_mod_trajectory_error_df()
    df = pdd._filter_summary_df(
        summary_df,
        distance_to_goal=distance_to_goal,
        decision_points=decision_points,
        all_envelope_defined=all_envelope_defined,
        min_chance_ratio=min_chance_ratio,
    )
    bias = df.groupby(["subject_ID", "theta_phase"])["signed_error"].mean().unstack(0).T
    return bias.sub(bias.mean(axis=1), axis=0)


def get_distance_bias(
    distance_to_goal=None,
    speed_range=None,
    maze_names=None,
    max_baseline_mae=None,
):
    """Per-subject distance-decoding bias along theta phase (m), per-subject mean-subtracted.

    Returns subjects × phases df.
    """
    summary_df = ddv2.get_theta_mod_distance_error_df()
    df = ddv2._filter_summary_df(
        summary_df,
        distance_to_goal=distance_to_goal,
        speed_range=speed_range,
        maze_names=maze_names,
        max_baseline_mae=max_baseline_mae,
    )
    bias = df.groupby(["subject_ID", "theta_phase"])["signed_error"].mean().unstack(0).T
    return bias.sub(bias.mean(axis=1), axis=0)


# %% Comparison plots


def plot_phase_offset_polar(
    place_bias,
    dist_bias,
    n_bootstraps=1_000,
    seed=0,
    theta_freq_hz=8.5,
    color="purple",
    subject_color="grey",
    ref_color="grey",
    show_mean_arrow=True,
    show_ci_arc=True,
    annotate_stats=True,
    ax=None,
):
    """Per-subject Δφ = (φ_place − φ_dist) plotted on a polar axis.

    Sign convention matches `tmu.test_theta_offset(dist_bias, place_bias)`:
      Δφ < 0  → distance leads place
      Δφ = 0  → in-phase
      Δφ = ±π → anti-phase
    Mean arrow length = mean resultant length R (across-subject consistency).
    CI arc = bootstrap 95% CI on the circular mean.
    """
    # 1. per-subject Δφ = φ_place − φ_dist
    subjects = sorted(set(place_bias.index) & set(dist_bias.index))
    phases_place = place_bias.columns.values.astype(float)
    phases_dist = dist_bias.columns.values.astype(float)
    deltas = []
    for s in subjects:
        phi_p = tmu.fit_sinusoid(phases_place, place_bias.loc[s].values, fit_constant=True, return_as="params")["phi"]
        phi_d = tmu.fit_sinusoid(phases_dist, dist_bias.loc[s].values, fit_constant=True, return_as="params")["phi"]
        deltas.append((phi_p - phi_d + np.pi) % (2 * np.pi) - np.pi)
    deltas = np.asarray(deltas)
    n = len(deltas)

    # 2. circular stats (mirrors test_theta_offset)
    mean_theta = circmean(deltas, high=np.pi, low=-np.pi)
    R = np.abs(np.mean(np.exp(1j * deltas)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstraps, n))
    boot_means = circmean(deltas[idx], high=np.pi, low=-np.pi, axis=1)
    boot_unwrapped = (boot_means - mean_theta + np.pi) % (2 * np.pi) - np.pi + mean_theta
    ci_low, ci_high = np.percentile(boot_unwrapped, [2.5, 97.5])
    sign = np.sign(mean_theta) if mean_theta != 0 else 1.0
    p_boot = 2 * min((np.sign(boot_means) != sign).mean(), (np.sign(boot_means) == sign).mean())
    _, p_ray = circ_rayleigh(deltas, d=np.pi / 6)

    # 3. render on polar axes
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(1.5, 1.5), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)
    ax.set_thetalim(-np.pi, np.pi)
    ax.set_ylim(0, 1.15)
    ax.set_yticklabels([])
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels(["", "", "0", "", "±π"])
    ax.grid(True, alpha=0.3)
    ax.spines["polar"].set_alpha(0.4)

    for ref_theta, lbl in [(0, "in-phase"), (np.pi, "anti-phase")]:
        ax.plot([ref_theta, ref_theta], [0, 1.0], color=ref_color, ls="--", lw=1, alpha=0.6, zorder=1)
        ax.text(ref_theta, 1.20, lbl, ha="center", va="center", fontsize=7, color=ref_color)

    ax.text(-np.pi / 2, 1.30, "place\nlags\ndistance", ha="center", va="center", fontsize=7, color=ref_color)
    ax.text(np.pi / 2, 1.30, "distance\nlags\nplace", ha="center", va="center", fontsize=7, color=ref_color)

    ax.scatter(
        deltas,
        np.full(n, 0.9),
        s=40,
        color=subject_color,
        alpha=0.5,
        linewidths=0,
        zorder=3,
    )

    if show_mean_arrow:
        ax.annotate(
            "",
            xy=(mean_theta, R),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=color, lw=2),
            zorder=4,
        )

    if show_ci_arc:
        arc = np.linspace(ci_low, ci_high, 200)
        ax.fill_between(arc, 1.05, 1.10, color=color, alpha=0.4, zorder=2, linewidth=0)

    if annotate_stats:
        T_ms = 1000.0 / theta_freq_hz
        dt_ms = mean_theta * T_ms / (2 * np.pi)
        ci_low_ms = ci_low * T_ms / (2 * np.pi)
        ci_high_ms = ci_high * T_ms / (2 * np.pi)
        print("place vs distance phase offset (Δφ = φ_place − φ_dist):")
        print(f"  mean Δφ:       {mean_theta:+.3f} rad   95% CI [{ci_low:+.3f}, {ci_high:+.3f}]")
        print(f"  mean Δt:       {dt_ms:+.1f} ms     95% CI [{ci_low_ms:+.1f}, {ci_high_ms:+.1f}]   ")
        print(f"  Rayleigh p:    {p_ray:.4g}")
        print(f"  bootstrap p:   {p_boot:.4g}")


# %% Lagged paired regression (sample-level alternative to the polar Δφ)


def get_lagged_regression_betas(
    double_df,
    lags=None,
    center_per_sample=True,
):
    """Per-subject β(δ) from regressing within-sample signed_error_place on
    signed_error_distance shifted by δ phase bins.

    Sign convention: δ > 0 rolls distance into the future relative to place, so
    β(+δ) > 0 means distance at phase k−δ predicts place at phase k (i.e.
    distance leads place by δ phase bins).

    Returns
    -------
    betas_df : pd.DataFrame
        index = subject_ID, columns = lag δ (int), values = β(δ).
    """
    if lags is None:
        lags = list(range(-5, 6))

    df = double_df.dropna(subset=["signed_error_place", "signed_error_distance"])

    out = {}
    for subject, sdf in df.groupby("subject_ID"):
        # Pivot each sample's 12 theta_phase rows into 12 columns per pipeline.
        # (trial_unique_ID, time) uniquely identifies a sample within a subject.
        wide = sdf.pivot_table(
            index=["trial_unique_ID", "time"],
            columns="theta_phase",
            values=["signed_error_place", "signed_error_distance"],
        )
        wide = wide.sort_index(axis=1, level=1)  # sort phase ascending → consistent np.roll
        P = wide["signed_error_place"].to_numpy()
        D = wide["signed_error_distance"].to_numpy()
        # Drop samples with any NaN cell (rare, but pivot may introduce them)
        mask = ~np.isnan(P).any(axis=1) & ~np.isnan(D).any(axis=1)
        P, D = P[mask], D[mask]
        if center_per_sample:
            P = P - P.mean(axis=1, keepdims=True)
            D = D - D.mean(axis=1, keepdims=True)

        betas = []
        for delta in lags:
            D_rolled = np.roll(D, -delta, axis=1)
            x = D_rolled.ravel()
            y = P.ravel()
            var_x = np.var(x, ddof=1)
            beta = np.cov(x, y, ddof=1)[0, 1] / var_x if var_x > 0 else np.nan
            betas.append(beta)
        out[subject] = betas

    betas_df = pd.DataFrame(out, index=lags).T
    betas_df.index.name = "subject_ID"
    betas_df.columns.name = "lag"
    return betas_df


def plot_lagged_regression_betas(
    betas_df,
    subject_color="grey",
    mean_color="purple",
    print_stats=True,
    ax=None,
):
    """Per-subject β(δ) curves + cross-subject mean ± SEM.

    Prints (if `print_stats=True`):
      - per-|δ| asymmetry `mean(β(+δ) − β(−δ))` across subjects + Wilcoxon p.
      - per-subject peak-lag distribution + Wilcoxon vs δ=0.
    """
    lags = betas_df.columns.to_numpy()
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.axvline(0, color="k", lw=0.5, alpha=0.5)
    for _, row in betas_df.iterrows():
        ax.plot(lags, row.values, color=subject_color, alpha=0.4, lw=1)
    mean = betas_df.mean(axis=0).values
    sem = betas_df.sem(axis=0).values
    ax.fill_between(lags, mean - sem, mean + sem, color=mean_color, alpha=0.25, lw=0)
    ax.plot(lags, mean, color=mean_color, lw=2)
    ax.set_xlabel("phase lag δ (bins)")
    ax.set_ylabel("β (place ~ distance shifted)")

    if print_stats:
        print("asymmetry  (mean β(+δ) − β(−δ),  Wilcoxon p across subjects):")
        for d in sorted({abs(int(L)) for L in lags if L > 0}):
            if (d in betas_df.columns) and (-d in betas_df.columns):
                asym = betas_df[d] - betas_df[-d]
                try:
                    _, p = wilcoxon(asym)
                except ValueError:
                    p = np.nan
                print(f"  |δ|={d}:  mean={asym.mean():+.3f}   p={p:.4g}")
        peak_lags = betas_df.idxmax(axis=1).astype(int)
        print(f"\nper-subject peak lags: {peak_lags.to_dict()}")
        try:
            _, p = wilcoxon(peak_lags.values)
            print(f"peak-lag distribution: mean={peak_lags.mean():+.2f} bins   " f"Wilcoxon vs 0 p={p:.4g}")
        except ValueError:
            pass


# %%
