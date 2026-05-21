"""
Cross-pipeline phase-offset comparison between place and distance theta-mod decoding.
@peterdoohan
"""

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import circmean
from pingouin import circ_rayleigh

from GridMaze.analysis.theta_mod import theta_utils as tmu
from GridMaze.analysis.theta_mod import place_direction_decoding as pdd
from GridMaze.analysis.theta_mod import distance_to_goal_decoder2 as ddv2
from GridMaze.analysis.theta_mod import distance_to_goal_tuning as dgt

# %% Data loaders


def get_place_bias(
    late_sessions=False,
    maze_names=None,
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
        late_sessions=late_sessions,
        maze_names=maze_names,
    )
    bias = df.groupby(["subject_ID", "theta_phase"])["signed_error"].mean().unstack(0).T
    return bias.sub(bias.mean(axis=1), axis=0)


def get_distance_bias(
    late_sessions=False,
    maze_names=None,
    distance_to_goal=None,
    speed_range=None,
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
        late_sessions=late_sessions,
    )
    bias = df.groupby(["subject_ID", "theta_phase"])["signed_error"].mean().unstack(0).T
    return bias.sub(bias.mean(axis=1), axis=0)


def get_distance_tuning_bias():
    """Per-subject distance-tuning bias along theta phase (m), per-subject mean-subtracted.

    Loads from `distance_to_goal_tuning.get_population_distance_tuning_theta_x_shifts_all_phases`
    and reshapes to the subjects × phases format used by the other bias loaders, so it can
    be passed straight into `plot_phase_offset_polar`.

    Returns subjects × phases df.
    """
    results_df = dgt.get_population_distance_tuning_theta_x_shifts_all_phases(
        save=False, verbose=False
    )  # late sessions
    bias = results_df.T  # subjects × phases, in metres
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
