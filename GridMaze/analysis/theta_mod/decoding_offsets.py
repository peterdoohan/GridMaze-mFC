"""
Cross-pipeline phase-offset comparison between place and distance theta-mod decoding.
@peterdoohan
"""

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import circmean
from pingouin import circ_rayleigh

from GridMaze.analysis.lfp import theta_modulation as tm
from GridMaze.analysis.theta_mod import theta_utils as tmu
from GridMaze.analysis.theta_mod import place_direction_decoding as pdd
from GridMaze.analysis.theta_mod import distance_to_goal_decoder as tdd
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
    summary_df = tdd.get_theta_mod_distance_error_df()
    df = tdd._filter_summary_df(
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


def get_lfp_theta_mod():
    """ """
    tm_df = tm.get_theta_aligned_lfp_df(n_bins=12, save=False, verbose=False)  # 12 bins late session match decoders
    # average across sessions per subject
    mod = tm_df.T.groupby(level=0).mean()
    return mod


def get_double_decoding_bias(late_sessions=False, maze_names=None):
    """Per-subject place and distance decoding biases along theta phase (m), per-subject
    mean-subtracted, from the matched-sample double-decoding pipeline.

    Returns (place_bias, distance_bias), each a subjects × phases df.
    """
    from GridMaze.analysis.theta_mod import double_decoding as ddec  # lazy: avoid circular import

    summary_df = ddec.get_theta_mod_double_decoding_df()
    df = ddec._filter_summary_df(summary_df, late_sessions=late_sessions, maze_names=maze_names)
    biases = {}
    for kind, col in [("place", "signed_error_place"), ("distance", "signed_error_distance")]:
        b = df.groupby(["subject_ID", "theta_phase"])[col].mean().unstack(0).T
        biases[kind] = b.sub(b.mean(axis=1), axis=0)
    return biases["place"], biases["distance"]


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


# %% Avg mod ref lfp plots


def plot_decoder_vs_lfp(
    place_color="darkred",
    distance_color="darkblue",
    distance_peak_color="dodgerblue",
    distance_trough_color="darkblue",
    late_sessions=True,
    maze_names=None,
    double_decoder=False,
    markers=False,
    place_k=3,
    distance_k=3,
    print_ranges=True,
    ax=None,
):
    """Place and distance decoder modulation curves over the black, amplitude-normalised
    LFP theta cycle, tiled across 2 theta cycles.

    Each curve is normalised by its own fitted sinusoid amplitude so all three sinusoids
    have amplitude 1; the y-axis is modulation (a.u.). LFP is per-subject mean-subtracted
    before fitting.

    `double_decoder`: if True, load place + distance from the matched-sample
    double-decoding pipeline (`get_double_decoding_bias`); else load from the
    independent pipelines (`get_place_bias` / `get_distance_bias`, default).
    `place_k` / `distance_k`: total number of phase bins in the highlighted band
    (any positive int, odd or even). Odd → symmetric about the centre bin; even →
    asymmetric with the extra bin on the higher-phase (right) side. Band edges
    are drawn at the outer bin edges so the band literally covers k bins. Place
    band marks the place trough (both cycles); distance bands mark the distance
    peak (`distance_peak_color`) and trough (`distance_trough_color`) on cycle i only.
    """

    def _band_bin_offsets(k):
        """(low, high) bin offsets s.t. band covers k bins; extra goes right for even k."""
        low = (k - 1) // 2
        high = k - 1 - low
        return low, high

    n_cycles = 2
    lfp_mod = get_lfp_theta_mod()
    if double_decoder:
        place_bias, distance_bias = get_double_decoding_bias(late_sessions=late_sessions, maze_names=maze_names)
    else:
        place_bias = get_place_bias(late_sessions=late_sessions, maze_names=maze_names)
        distance_bias = get_distance_bias(late_sessions=late_sessions, maze_names=maze_names)
    decoders = {
        "place": (place_bias, place_color),
        "distance": (distance_bias, distance_color),
    }

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(3.5, 2))
    ax.axhline(0, color="k", ls="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("theta phase")
    ax.set_ylabel("modulation (a.u.)")

    # LFP: per-subject mean-subtract, fit sinusoid, normalise by its amplitude
    lfp_phases = lfp_mod.columns.values.astype(float)
    lfp_centered = lfp_mod.sub(lfp_mod.mean(axis=1), axis=0)
    lfp_mean = lfp_centered.mean().values
    lfp_sem = lfp_centered.sem().values
    lfp_fit = tmu.fit_sinusoid(lfp_phases, lfp_mean, fit_constant=True, return_as="params")
    lfp_scale = 1.0 / lfp_fit["A"] if lfp_fit["A"] > 0 else 1.0
    lfp_mean_n = lfp_mean * lfp_scale
    lfp_sem_n = lfp_sem * lfp_scale

    # decoders: fit sinusoid to subject-mean, normalise by its amplitude
    dec_fits = {}
    for name, (dec_bias, _color) in decoders.items():
        phases = dec_bias.columns.values.astype(float)
        dec_mean = dec_bias.mean().values
        dec_sem = dec_bias.sem().values
        dec_fit = tmu.fit_sinusoid(phases, dec_mean, fit_constant=True, return_as="params")
        dec_scale = 1.0 / dec_fit["A"] if dec_fit["A"] > 0 else 1.0
        dec_fits[name] = {
            "phases": phases,
            "mean_n": dec_mean * dec_scale,
            "sem_n": dec_sem * dec_scale,
            "fit": dec_fit,
            "scale": dec_scale,
            "color": _color,
        }

    # place-trough band: k-bin-wide band centred on the binned place sinusoid trough, on every cycle
    place_phases = dec_fits["place"]["phases"]
    place_fit = dec_fits["place"]["fit"]
    place_fitted = np.sin(place_phases + place_fit["phi"])  # amplitude-normalised fit at bins
    place_trough_idx = int(np.argmin(place_fitted))
    place_trough_phase = float(place_phases[place_trough_idx])
    dphi = 2 * np.pi / len(place_phases)
    p_low, p_high = _band_bin_offsets(place_k)
    place_left = (p_low + 0.5) * dphi
    place_right = (p_high + 0.5) * dphi
    for k in range(n_cycles):
        off = 2 * np.pi * k
        ax.axvspan(
            place_trough_phase + off - place_left,
            place_trough_phase + off + place_right,
            color=place_color,
            alpha=0.15,
            zorder=0,
            linewidth=0,
        )

    # distance peak + trough bands: cycle i (2nd cycle) only
    dist_phases = dec_fits["distance"]["phases"]
    dist_fit = dec_fits["distance"]["fit"]
    dist_fitted = np.sin(dist_phases + dist_fit["phi"])
    dist_peak_idx = int(np.argmax(dist_fitted))
    dist_trough_idx = int(np.argmin(dist_fitted))
    dist_peak_phase = float(dist_phases[dist_peak_idx])
    dist_trough_phase = float(dist_phases[dist_trough_idx])
    dphi_d = 2 * np.pi / len(dist_phases)
    d_low, d_high = _band_bin_offsets(distance_k)
    dist_left = (d_low + 0.5) * dphi_d
    dist_right = (d_high + 0.5) * dphi_d
    cycle_i_off = 2 * np.pi * (n_cycles - 1)
    for phase, _color in [(dist_peak_phase, distance_peak_color), (dist_trough_phase, distance_trough_color)]:
        ax.axvspan(
            phase + cycle_i_off - dist_left,
            phase + cycle_i_off + dist_right,
            color=_color,
            alpha=0.2,
            zorder=0,
            linewidth=0,
        )

    if print_ranges:
        n_place_bins = len(place_phases)
        n_dist_bins = len(dist_phases)
        place_bins = [(place_trough_idx + i) % n_place_bins for i in range(-p_low, p_high + 1)]
        peak_bins = [(dist_peak_idx + i) % n_dist_bins for i in range(-d_low, d_high + 1)]
        trough_bins = [(dist_trough_idx + i) % n_dist_bins for i in range(-d_low, d_high + 1)]
        print("cycle i phase ranges (rad)  |  bin ids (-π=0, +π=n-1):")
        print(
            f"  place trough:    [{place_trough_phase - place_left:+.3f}, "
            f"{place_trough_phase + place_right:+.3f}]  |  {place_bins}"
        )
        print(
            f"  distance peak:   [{dist_peak_phase - dist_left:+.3f}, "
            f"{dist_peak_phase + dist_right:+.3f}]  |  {peak_bins}"
        )
        print(
            f"  distance trough: [{dist_trough_phase - dist_left:+.3f}, "
            f"{dist_trough_phase + dist_right:+.3f}]  |  {trough_bins}"
        )

    # tile across cycles
    x_curve = np.linspace(-np.pi, np.pi, 200)
    for k in range(n_cycles):
        off = 2 * np.pi * k
        y_lfp = np.sin(x_curve + lfp_fit["phi"])
        ax.plot(x_curve + off, y_lfp, color="black", lw=1.5, label="LFP (ref.)" if k == 0 else None)
        if markers:
            ax.errorbar(
                lfp_phases + off,
                lfp_mean_n,
                yerr=lfp_sem_n,
                fmt="o",
                color="black",
                markersize=4,
                elinewidth=1.5,
            )
        for name, d in dec_fits.items():
            if markers:
                ax.errorbar(
                    d["phases"] + off,
                    d["mean_n"],
                    yerr=d["sem_n"],
                    fmt="o",
                    color=d["color"],
                    markersize=4,
                    elinewidth=1.5,
                    label=name if k == 0 else None,
                )
            y_dec = np.sin(x_curve + d["fit"]["phi"]) + d["fit"]["C"] * d["scale"]
            ax.plot(
                x_curve + off,
                y_dec,
                color=d["color"],
                lw=1.5,
                label=name if (not markers and k == 0) else None,
            )

    # dashed verticals at inner cycle boundaries
    for k in range(1, n_cycles):
        ax.axvline(-np.pi + 2 * np.pi * k, color="k", ls="--", alpha=0.5)

    # per-cycle labels ("cycle i-(n-1)", ..., "cycle i-1", "cycle i") at top of axes
    for k in range(n_cycles):
        offset_from_last = n_cycles - 1 - k
        label = "cycle i" if offset_from_last == 0 else f"cycle i-{offset_from_last}"
        ax.text(2 * np.pi * k, 1.02, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)

    # π-spaced ticks, labelled per-cycle (-π → π each cycle; inner boundaries = "π/-π")
    tick_locs = np.arange(-np.pi, -np.pi + 2 * np.pi * n_cycles + 0.01, np.pi)
    tick_labels = []
    for i in range(len(tick_locs)):
        if i == 0:
            tick_labels.append("-π")
        elif i == len(tick_locs) - 1:
            tick_labels.append("π")
        elif i % 2 == 1:
            tick_labels.append("0")
        else:
            tick_labels.append("π/-π")
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(-0.15, 1.02))
