"""
collect functions used across theta mod analyses
@peterdoohan
"""

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import circmean, circstd
from pingouin import multivariate_ttest, circ_rayleigh

# %% Global variables

# %% Functions


def plot_decoding_bias(
    decoding_bias,
    color="grey",
    label=None,
    ylabel="decoding bias",
    norm=False,
    print_stats=True,
    ax=None,
):
    # set up figure
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.axhline(0, color="k", ls="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("theta phase")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(-np.pi, np.pi + 0.1, np.pi / 2))
    ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])

    df = decoding_bias.copy()
    phases = decoding_bias.columns.values.astype(float)

    if norm:
        # normalise df to max
        df = df / np.max(np.abs(df.values))
    mean = df.mean().values
    sem = df.sem().values
    # plot datapoints
    ax.errorbar(
        phases,
        mean,
        yerr=sem,
        fmt="o",
        color=color,
        markersize=5,
        linewidth=None,
        capsize=None,
        elinewidth=1.5,
    )
    # plot curvefit
    _x, _y = fit_sinusoid(phases, mean, fit_constant=True, return_as="curve")
    ax.plot(_x, _y, color=color, linewidth=1.5, label=label)
    if label is not None:
        ax.legend(frameon=False, fontsize=8)

    # test sinusoidal random effects across subjects
    if print_stats:
        print(label)
        test_theta_modulation(df)


def test_theta_offset(mod_df1, mod_df2, subject_IDs=None, phases=None):
    """ """
    if subject_IDs is None:
        subject_IDs = mod_df1.index.values
    if phases is None:
        phases = mod_df1.columns.values.astype(float)

    offsets = []
    for subject in subject_IDs:
        curve_1 = mod_df1.loc[subject].values
        fit_1 = fit_sinusoid(phases, curve_1, fit_constant=True, return_as="params")
        curve_2 = mod_df2.loc[subject].values
        fit_2 = fit_sinusoid(phases, curve_2, fit_constant=True, return_as="params")
        # get phase offset
        off = fit_2["phi"] - fit_1["phi"]
        # wrap to [-pi, pi]
        w_off = (off + np.pi) % (2 * np.pi) - np.pi
        offsets.append(w_off)
    z, p = circ_rayleigh(offsets, d=np.pi / 6)
    mean_offset = circmean(offsets, high=np.pi, low=-np.pi)
    sem_offset = circstd(offsets, high=np.pi, low=-np.pi) / np.sqrt(len(offsets))
    print(f"offset: {mean_offset:.3f} ± {sem_offset:.3f}. Rayleigh test: z={z:.3f}, p={p:.3f}")


def fit_sinusoid(x, y, fit_constant=True, return_as="params"):
    """
    Fit y(x) ≈ alpha*sin(x) + beta*cos(x) + C  (period = 2π -> ω = 1)
    Returns dict with alpha, beta, C, A, phi (radians), residuals.
    Notes:
      - A = sqrt(alpha^2 + beta^2)
      - phi = atan2(beta, alpha)  (so model = A * sin(x + phi) + C)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    # design matrix: columns [sin(x), cos(x), (1)]
    X = np.column_stack([np.sin(x), np.cos(x)])
    if fit_constant:
        X = np.column_stack([X, np.ones_like(x)])
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    alpha = coeffs[0]
    beta = coeffs[1]
    C = coeffs[2] if fit_constant else 0.0
    A = np.hypot(alpha, beta)
    phi = np.atan2(beta, alpha)  # returns phase in radians
    if return_as == "params":
        return {"alpha": alpha, "beta": beta, "C": C, "A": A, "phi": phi}
    elif return_as == "curve":
        _x = np.linspace(-np.pi, np.pi, 100)
        _y = A * np.sin(_x + phi) + C
        return _x, _y
    else:
        raise ValueError(f"return_as must be 'params' or 'curve'.")


def plot_phase_colormap_key(n=12, cmap="coolwarm", ax=None):
    """
    Plot a sinusoid from -pi to pi (peak at 0, troughs at ±pi) coloured by
    phase using a colormap. Useful as a legend for phase-encoded plots.

    Parameters
    ----------
    n : int
        Number of colour segments to divide the sinusoid into.
    cmap : str
        Matplotlib colormap name.
    ax : matplotlib Axes, optional
        Axes to plot into. If None, a new figure is created.

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2.5, 1.5))

    phase = np.linspace(-np.pi, np.pi, 500)
    y = np.cos(phase)
    cm = plt.get_cmap(cmap)

    # break into n segments, colour each by its midpoint phase mapped to [0,1]
    indices = np.array_split(np.arange(len(phase)), n)
    for idx in indices:
        seg_phase = phase[idx]
        seg_y = y[idx]
        mid_phase = seg_phase[len(seg_phase) // 2]
        color = cm((mid_phase + np.pi) / (2 * np.pi))
        ax.plot(seg_phase, seg_y, color=color, linewidth=2.5)

    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])
    ax.set_yticks([])
    ax.set_xlabel("theta phase")
    ax.spines[["top", "right", "left"]].set_visible(False)
    return ax


def test_theta_modulation(theta_bias_df):
    """
    input = df [rows=subjects, columns=theta phases, values=decoding bias]
    """
    phis = theta_bias_df.columns.astype(float)
    data = theta_bias_df.values
    beta_cos = data.dot(np.cos(phis))
    beta_sin = data.dot(np.sin(phis))
    betas = np.column_stack([beta_cos, beta_sin])
    zeros = np.zeros_like(betas)
    mv_test = multivariate_ttest(betas, zeros, paired=False)
    return print(mv_test)
