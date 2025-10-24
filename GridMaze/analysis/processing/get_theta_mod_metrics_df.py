"""
Make analysis data structure that contains theta mod metrics for every cluster
"""

# %% Imports
import pandas as pd
import numpy as np
from requests import session
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from pingouin import circ_rayleigh


from GridMaze.analysis.core import load_data
from GridMaze.analysis.core import convert

# %% Global Variables
FRAME_RATE = 60

# %% Functions


def get_theta_mod_metrics_df(
    processed_data_path,
    analysis_data_path,
    navigation_only=True,
    moving_only=True,
    max_steps_to_goal=30,
):
    """
    Collects theta modulation metrics:
    - rayleigh test for non-uniformity of circular data (p-value)
    - sine fit
        - amplitude
        - phase offset
        - r2
        - phase max
        - phase min
        - mod depth
    - split-half correlation
    + anatomical and session info
    """
    # load data
    session_info = load_data.load(processed_data_path / "session_info.json")
    cluster_metrics = load_data.load(processed_data_path / "clusters.metrics.htsv")
    navigation_df = load_data.load(analysis_data_path / "frames.navigation.parquet")
    theta_spike_counts_df = load_data.load(analysis_data_path / "frames.thetaSpikeCounts.parquet")
    theta_spike_counts_df.reset_index(drop=True, inplace=True)
    # filter for moving, navigation, on task etc.
    mask = []
    if navigation_only:
        mask.append((navigation_df.trial_phase == "navigation").values)
    if moving_only:
        mask.append(navigation_df.moving.values)
    if max_steps_to_goal is not None:
        mask.append(navigation_df.steps_to_goal.future.le(max_steps_to_goal).values)
    if len(mask) > 0:
        combined_mask = np.logical_and.reduce(mask)
        navigation_df = navigation_df[combined_mask]
        theta_spike_counts_df = theta_spike_counts_df[combined_mask]
    # for each cluster, get cal avg. theta mod related metrics
    cluster_unique_IDs = theta_spike_counts_df.spike_count.columns.get_level_values(0).unique().values
    all_metrics = []
    for cluster in cluster_unique_IDs:
        print(cluster)
        theta_spikes = theta_spike_counts_df.xs(cluster, level=1, axis=1)
        nav_spike_counts_df = pd.concat([navigation_df, theta_spikes], axis=1)
        # get_basic metrics
        cluster_ID = convert._reverse_cluster_unique(cluster)
        _metrics = cluster_metrics.set_index("cluster_ID").loc[cluster_ID]
        metrics = {
            ("subject_ID", ""): session_info["subject_ID"],
            ("maze_name", ""): session_info["maze_name"],
            ("day_on_maze", ""): session_info["day_on_maze"],
            ("cluster_unique_ID", ""): cluster,
            ("single_unit", ""): _metrics[("single_unit", "")],
            ("multi_unit", ""): _metrics[("multi_unit", "")],
            ("tissue_sample", ""): _metrics[("tissue_sample", "")],
            ("region", "acronym"): _metrics[("region", "acronym")],
            ("voxel", "x"): _metrics[("voxel", "x")],
            ("voxel", "y"): _metrics[("voxel", "y")],
            ("voxel", "z"): _metrics[("voxel", "z")],
        }
        # get theta mod metrics
        thet_mod_metrics = get_theta_metrics(nav_spike_counts_df)
        metrics.update(thet_mod_metrics)
        all_metrics.append(metrics)
    df = pd.DataFrame(all_metrics)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def get_theta_metrics(nav_spike_counts_df, n=50):
    """ """
    # compute split-half correlation
    trials = nav_spike_counts_df.trial.unique()
    n_trials = len(trials)
    split_half_corrs = np.zeros(n)
    for i in range(n):
        shuffled_trials = np.random.permutation(trials)
        trials_split_1 = shuffled_trials[: n_trials // 2]
        trials_split_2 = shuffled_trials[n_trials // 2 :]
        data_split_1 = nav_spike_counts_df[nav_spike_counts_df.trial.isin(trials_split_1)]
        data_split_2 = nav_spike_counts_df[nav_spike_counts_df.trial.isin(trials_split_2)]
        # get theta mod for each split
        theta_mod_1 = data_split_1.spike_count.sum()
        theta_mod_2 = data_split_2.spike_count.sum()
        # compute correlation between splits
        correlation = np.corrcoef(theta_mod_1.values, theta_mod_2.values)[0, 1]
        split_half_corrs[i] = correlation
    split_half_corr = np.mean(split_half_corrs)

    # fit vonmises-shaped theta-mod tuning curve
    theta_mod = nav_spike_counts_df.spike_count.sum()
    theta_mod = theta_mod.div(theta_mod.mean())  # norm to mean 1
    vm_params = fit_vonmises(theta_mod)

    # do rayleigh test for non-uniformity
    z, p = circ_rayleigh(theta_mod.values, d=theta_mod.index.values)

    # get mean firing rate (some cells have low rates that are tuned outside of navigation)
    fr = nav_spike_counts_df.spike_count.sum().sum() / (nav_spike_counts_df.shape[0] / FRAME_RATE)
    return {
        ("split_half_corr", ""): split_half_corr,
        **{(f"vonmises", k): v for k, v in vm_params.items()},
        ("rayleigh", "z"): z,
        ("rayleigh", "p"): p,
        ("mean_firing_rate", ""): fr,
    }


# %%


def fit_vonmises(series, plot=False):
    """
    Fit a von Mises-shaped modulation:
        rate(θ) = baseline + amp * exp(kappa * cos(θ - mu))
    to circular phase–rate data.

    Parameters
    ----------
    series : pandas.Series
        index = phases (radians in [-π, π])
        values = firing rates (mean ~1 is fine)

    Returns
    -------
    dict
        {
          'baseline': float,
          'amp': float,
          'kappa': float,
          'mu': float,              # radians (in [-π, π])
          'phase_max_rad': float,   # wrapped to [0, 2π)
          'phase_min_rad': float,
          'phase_max_deg': float,
          'phase_min_deg': float,
          'modulation_depth': float,
          'r2': float
        }
    """

    def vm_model(theta, baseline, amp, kappa, mu):
        return baseline + amp * np.exp(kappa * np.cos(theta - mu))

    # Extract data
    theta = np.asarray(series.index, dtype=float)
    y = np.asarray(series.values, dtype=float)

    # Initial parameter guesses
    baseline0 = 1
    amp0 = max(np.max(y) - baseline0, 1e-6)
    mu0 = float(theta[np.nanargmax(y)])  # phase of max rate
    kappa0 = 1.0
    p0 = [baseline0, amp0, kappa0, mu0]

    # Reasonable bounds
    bounds_lower = [0.8, 0.0, 0.0, -np.pi]
    bounds_upper = [1.2, np.inf, 1e3, np.pi]

    # Fit model
    popt, _ = curve_fit(
        vm_model,
        theta,
        y,
        p0=p0,
        bounds=(bounds_lower, bounds_upper),
        maxfev=20000,
    )
    baseline, amp, kappa, mu = popt

    # Compute fitted curve and metrics
    y_fit = vm_model(theta, baseline, amp, kappa, mu)
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Modulation depth (max-min)/(max+min)
    theta_dense = np.linspace(-np.pi, np.pi, 360)
    y_fit_dense = vm_model(theta_dense, baseline, amp, kappa, mu)
    y_max, y_min = np.max(y_fit_dense), np.min(y_fit_dense)
    phase_min = theta_dense[y_fit_dense.argmin()]
    phase_max = theta_dense[y_fit_dense.argmax()]
    modulation_depth = (y_max - y_min) / (y_max + y_min) if (y_max + y_min) != 0 else np.nan

    if plot:
        f, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.spines[["top", "right"]].set_visible(False)
        ax.plot(theta, y, label="data", color="k")
        ax.plot(theta_dense, y_fit_dense, label="Von Mises fit")
        ax.set_xlabel("Phase (rad.)")
        ax.set_ylabel("Norm Rate")
        ax.set_title(f"Von Mises Fit\n$R^2$={r2:.3f}, depth={modulation_depth:.3f}")
        ax.legend()

    return {
        "baseline": float(baseline),
        "amp": float(amp),
        "kappa": float(kappa),
        "mu": float(mu),
        "phase_max_rad": float(phase_max),
        "phase_min_rad": float(phase_min),
        "modulation_depth": float(modulation_depth),
        "r2": float(r2),
    }


# %% test


def pass_metrics(
    metrics,
    min_firing_rate=1,
    min_split_half_corr=0.4,
    min_r2=0.75,
):
    if metrics[("single_unit", "")]:
        if metrics[("mean_firing_rate", "")] >= min_firing_rate:
            if metrics[("split_half_corr", "")] >= min_split_half_corr:
                if metrics[("vonmises", "r2")] >= min_r2:
                    if metrics[("rayleigh", "p")] < 0.05:
                        return True
                    else:
                        return False
