"""
Library for generating an analysis data strucutre the represents neural firing rates aligned to changes in
route during navigational trials.
@peterdoohan (using the wonderful infered routes by @xiaoqin)
"""

# %% Imports
import numpy as np
import pandas as pd
from scipy.stats import norm

from ..core import convert
from ..core import load_data

from .align_activity import align_spikes


# %% Global Variables


# %% Functions


def get_route_change_aligned_rates(processed_data_path, analysis_data_path, window=5, n_previous_routes=4):
    """
    Generates a dataframe with neural firing rates aligned to changes in latent 'route' state during navigation.
    Route data is inferred from the animal's trajectory using a hidden markov model, in this repo is it is loaded from preprocessed
    data and integrated with existing analysis data structures as: frames.navigation.parquet and frames.routes.parquet.

    DataFrames contains basic session info, subject_ID, maze_name, day_on_maze, trial, goal, & cluster_unique_ID.
    Firing rates aligned to each route change during the session are stored under top level columns route_change_i,
    with lower levels firing_rate --> timepoints, which contains the firing rates for each cluster at each aligned
    timepoint, and latent --> pre/post, which contains the latent state before and after the route change. These are refered to
    as 'latent' states because the can be any route_i or cue and reward. Note that the latent_post (i.e the latent state the route
    change moves into) is always 'reward' for route_change_0 and simiply represents the firing rates aligned to reward which are kept
    in the dataframe for completeness. Also note the in other route_change_i columns latent_pre can be the cue state, if there was just
    one or two routes used in a trial, these trials can be removed for later analysis but are kept for completion.

    This data structure can be used to plot route change aligned firing rates stratified by goal or route :)
    """
    # load data
    try:
        trials_df = load_data.load(processed_data_path / "trials.htsv")
        spike_clusters = load_data.load(processed_data_path / "spikes.clusters.npy")
        spike_times = load_data.load(processed_data_path / "spikes.times.npy")
        session_info = load_data.load(processed_data_path / "session_info.json")
        navigation_df = load_data.load(analysis_data_path / "frames.navigation.parquet")
        navigation_routes_df = load_data.load(analysis_data_path / "frames.routes.parquet")
    except FileNotFoundError:
        print(
            f"Missing prerequisit processed and/or analysis data, cannot generate route change aligned rates for {processed_data_path}."
        )
        return None
    navigation_routes_df = pd.concat([navigation_df, navigation_routes_df.reset_index(drop=True)], axis=1)
    trial2goal = trials_df.set_index(("trial", "")).goal.to_dict()
    trial2n_routes = _get_trial2n_routes(navigation_routes_df)
    route_changes_df = _get_route_changes_df(trials_df, navigation_routes_df, n_previous_routes=n_previous_routes)
    dfs = []
    for i, route_change in enumerate(route_changes_df.columns.get_level_values(0).unique()):
        times = route_changes_df[route_change].time.to_list()
        aligned_rates, t_out, cluster_IDs = get_time_aligned_activity(
            spike_clusters, spike_times, times, window_size=window
        ).values()
        trial_dfs = []
        for j, trial in enumerate(route_changes_df.index):
            time_aligned_activity = aligned_rates[j, :, :]
            trial_activity_df = pd.DataFrame(
                data=time_aligned_activity, columns=pd.MultiIndex.from_product([[route_change], ["firing_rate"], t_out])
            )
            trial_change_info = route_changes_df[route_change].loc[trial]
            trial_activity_df.loc[:, (route_change, "latent", "pre")] = trial_change_info.latent_pre
            trial_activity_df.loc[:, (route_change, "latent", "post")] = trial_change_info.latent_post
            if i == 0:
                trial_info_df = _get_trial_info_df(session_info, trial, cluster_IDs, trial2goal, trial2n_routes)
                trial_activity_df = pd.concat([trial_info_df, trial_activity_df], axis=1)
            trial_dfs.append(trial_activity_df)
        route_change_activity_df = pd.concat(trial_dfs, axis=0)
        dfs.append(route_change_activity_df)
    route_change_activity_df = pd.concat(dfs, axis=1)
    return route_change_activity_df


def _get_trial_info_df(session_info, trial, cluster_IDs, trial2goal, trial2n_routes):
    """
    Generate a DataFrame containing trial information for a given session.
    Parameters:
    session_info (dict): Dictionary containing session information such as subject ID, maze name, and day on maze.
    trial (int): The trial number.
    cluster_IDs (list): List of cluster IDs.
    trial2goal (dict): Dictionary mapping trials to their respective goals.
    Returns:
    pd.DataFrame: A DataFrame with multi-index columns containing the trial information.
    """
    trial_info_df = pd.DataFrame(
        {
            ("subject_ID", "", ""): session_info["subject_ID"],
            ("maze_name", "", ""): session_info["maze_name"],
            ("day_on_maze", "", ""): session_info["day_on_maze"],
            ("trial", "", ""): trial,
            ("goal", "", ""): trial2goal[trial],
            ("n_routes", "", ""): trial2n_routes[trial],
            ("cluster_unique_ID", "", ""): convert.cluster_IDs2scluster_unique_IDs(session_info, cluster_IDs),
        }
    )
    trial_info_df.columns = pd.MultiIndex.from_tuples(trial_info_df.columns)
    return trial_info_df


def _get_trial2n_routes(navigation_routes_df):
    """ """
    trial2n_routes = {}
    for trial in navigation_routes_df.trial.dropna().unique():
        trial_df = navigation_routes_df[
            (navigation_routes_df.trial == trial) & (navigation_routes_df.trial_phase == "navigation")
        ]
        trial2n_routes[trial] = int(trial_df.route_change.sum() + 1)
    return trial2n_routes


def get_time_aligned_activity(spike_clusters, spike_times, times, window_size=5, fs_out=25, smooth_SD="default"):
    """
    Calcualtes firing rates aligned to speicfied list of times for all clusters in a session.
    """
    if smooth_SD == "default":
        smooth_SD = 1 / fs_out
    spikes = np.vstack((spike_clusters, spike_times))
    cluster_IDs = np.sort(np.unique(spike_clusters)).astype(np.float64)
    n_clusters = len(cluster_IDs)
    t_out = np.arange(-window_size, window_size, 1 / fs_out)
    pad_len = smooth_SD * 4
    event_aligned_rates = np.full([len(times), n_clusters, len(t_out)], np.nan)
    for t, time in enumerate(times):  # Loop over trials.
        if np.isnan(time):
            continue
        event_aligned_spikes = spikes[
            :,
            (time - window_size - pad_len < spikes[1, :]) & (spikes[1, :] < time + window_size + pad_len),
        ]
        event_aligned_spike_IDs = event_aligned_spikes[0, :]
        event_aligned_spike_times = event_aligned_spikes[1, :] - time
        for j, n in enumerate(cluster_IDs):  # Loop over clusters.
            if n in event_aligned_spike_IDs:
                neuron_mask = event_aligned_spike_IDs == n
                n_spike_times = event_aligned_spike_times[neuron_mask]
                event_aligned_rates[t, j, :] = np.sum(
                    norm.pdf(n_spike_times[None, :] - t_out[:, None], scale=smooth_SD),
                    axis=1,
                )
            else:
                event_aligned_rates[t, j, :] = 0
    return {
        "event_aligned_rates": event_aligned_rates,  # [n_times, n_clusters, n_times]
        "t_out": t_out,
        "cluster_IDs": cluster_IDs,
    }


def _nan_pad_list(input_list, target_length):
    """ """
    nans_to_add = target_length - len(input_list)
    element = input_list[0]
    if isinstance(element, tuple):
        padded_list = [(np.nan,) * len(element)] * nans_to_add + input_list
    else:
        padded_list = [np.nan] * nans_to_add + input_list
    return padded_list


# %% Do warping instead of just aligning on time


def get_route_aligned_rates_df(processed_data_path, analysis_data_path, n_previous_routes=3):
    """ """
    # load_data
    try:
        trials_df = load_data.load(processed_data_path / "trials.htsv")
        spike_clusters = load_data.load(processed_data_path / "spikes.clusters.npy")
        spike_times = load_data.load(processed_data_path / "spikes.times.npy")
        session_info = load_data.load(processed_data_path / "session_info.json")
        navigation_df = load_data.load(analysis_data_path / "frames.navigation.parquet")
        navigation_routes_df = load_data.load(analysis_data_path / "frames.routes.parquet")
    except FileNotFoundError:
        print(
            f"Missing prerequisit processed and/or analysis data, cannot generate route change aligned rates for {processed_data_path}."
        )
        return None
    navigation_routes_df = pd.concat([navigation_df, navigation_routes_df.reset_index(drop=True)], axis=1)
    trial2goal = (
        navigation_df[[("trial", ""), ("goal", "")]]
        .dropna()
        .drop_duplicates()
        .set_index("trial")[("goal", "")]
        .to_dict()
    )
    trial2n_routes = _get_trial2n_routes(navigation_routes_df)
    route_changes_df = _get_route_changes_df(
        trials_df, navigation_routes_df, n_previous_routes=n_previous_routes + 1
    )  # need to add 1 because input here is expecting n_previous_route chages to get the timepoints at each end
    route_latents_df = get_route_latents_df(navigation_routes_df, n_previous_routes=n_previous_routes)
    # get route aligned activity
    aligned_rates, t_out, min_max_stretch, cluster_IDs = get_route_warped_activity(
        route_changes_df, spike_clusters, spike_times
    ).values()
    trp = int(len(t_out) / n_previous_routes)  # timepoints per route
    route_order_columns = [
        f"route_order_{i}" for i in reversed(range(n_previous_routes))
    ]  # route order defined from reward back
    dfs = []
    t = 0
    for trial in trial2goal.keys():
        trial_df = navigation_df[(navigation_df.trial == trial) & (navigation_df.trial_phase == "navigation")]
        if trial_df.empty:  # check if trial df is empty and skip
            continue
        trial_info_df = _get_trial_info_df(session_info, trial, cluster_IDs, trial2goal, trial2n_routes)
        # add min max stretch info
        trial_info_df.loc[:, ("trial_stretch", "min", "")] = min_max_stretch[t, 0]
        trial_info_df.loc[:, ("trial_stretch", "max", "")] = min_max_stretch[t, 1]
        trial_latent_info = route_latents_df.loc[trial]
        route_activity_dfs = []
        for i, route_order in enumerate(route_order_columns):
            route_activity_df = pd.DataFrame(
                data=aligned_rates[t, :, i * trp : (i + 1) * trp],
                columns=pd.MultiIndex.from_product([[route_order], ["firing_rate"], np.linspace(0, 1, trp)]),
            )
            # add information about latent states occcurred during, prefore and after the route (latent, l == current route)
            route_order_latents = trial_latent_info.loc[route_order]
            for shift, latent in route_order_latents.items():
                route_activity_df.loc[:, (route_order, "latent", shift)] = latent
            route_activity_dfs.append(route_activity_df)
        trial_activity_df = pd.concat([trial_info_df] + route_activity_dfs, axis=1)
        dfs.append(trial_activity_df)
        t += 1
    return pd.concat(dfs, axis=0)


def get_route_warped_activity(route_changes_df, spike_clusters, spike_times, fs_out=50):
    """"""
    route_change_times = route_changes_df.xs("time", level=1, axis=1)
    nan_mask = route_change_times.isna().to_numpy()  # remember trials that didn't have enough route changes
    # where timepoints are NaN replace with the value in the next column minus 1 (applied recursively)
    for col in range(route_change_times.shape[1] - 2, -1, -1):
        route_change_times.iloc[:, col] = route_change_times.iloc[:, col].fillna(
            route_change_times.iloc[:, col + 1] - 1
        )
    times = route_change_times.to_numpy() * 1000  # needs to be in ms for align_spikes fn
    # warp spike times to route change times
    spike_times_ms = spike_times * 1000  # convert to ms
    spikes = np.vstack((spike_clusters, spike_times_ms))
    target_times = (
        np.arange(route_change_times.shape[1]) * 1000
    )  # split into even bins represented in ms for align_spikes fn
    aligned_activity = align_spikes(times, target_times, spikes, plot=False, fs_out=fs_out)  # require time units  in ms
    aligned_activity["t_out"] = aligned_activity["t_out"] / 1000  # seconds
    # NaN out trials without specific route changes
    expanded_nan_mask = _expand_nan_mask(nan_mask, n_timepoints=fs_out, n_neurons=len(aligned_activity["cluster_IDs"]))
    aligned_rates = aligned_activity["aligned_rates"]
    aligned_rates[expanded_nan_mask] = np.nan
    aligned_activity["aligned_rates"] = aligned_rates
    return aligned_activity


def get_route_latents_df(navigation_routes_df, n_previous_routes=3, n_latent_shifts=3):
    """ """
    trials = []
    route_orders = []
    for trial in navigation_routes_df.trial.dropna().unique():
        trial_df = navigation_routes_df[
            (navigation_routes_df.trial == trial) & (navigation_routes_df.trial_phase == "navigation")
        ]
        if trial_df.empty:
            continue
        routes = trial_df.route.r
        routes_sequence = routes[routes != routes.shift(1)]
        latent_sequence = pd.Series(["cue"] + routes_sequence.to_list() + ["reward"])
        latent_shift_df = pd.DataFrame(index=latent_sequence.index)
        latent_shift_df["l"] = latent_sequence
        for i in range(1, n_latent_shifts + 1):
            latent_shift_df[f"l+{i}"] = latent_sequence.shift(-i)
        for i in range(1, n_latent_shifts + 1):
            latent_shift_df[f"l-{i}"] = latent_sequence.shift(+i)
        latent_shift_df = latent_shift_df.iloc[1:-1]  # remove cue and reward l states
        # add optimal route column
        latent_shift_df["optimal_route"] = trial_df[routes != routes.shift(1)].optimal_route.to_numpy()
        if latent_shift_df.shape[0] < n_previous_routes:  # add empty rows when fewer than n_previous_routes
            n_pad = n_previous_routes - latent_shift_df.shape[0]
            pad_df = pd.DataFrame(columns=latent_shift_df.columns, index=range(n_pad), data=np.nan)
            latent_shift_df = pd.concat([pad_df, latent_shift_df], axis=0)
        latent_shift_df = latent_shift_df.iloc[-n_previous_routes:]  # only keep the last n_previous_routes
        latent_shift_df.index = [
            f"route_order_{i}" for i in reversed(range(n_previous_routes))
        ]  # label routes back from reward
        route_orders.append(latent_shift_df.T.unstack())
        trials.append(trial)
    route_latents_df = pd.concat(route_orders, axis=1).T
    route_latents_df = route_latents_df.infer_objects(copy=False).replace({None: np.nan})
    route_latents_df.index = trials
    return route_latents_df


def _get_route_changes_df(trials_df, navigation_routes_df, n_previous_routes=3):
    """
    Generate a DataFrame containing route change times and latent state labels for each trial.
    Parameters:
    -----------
    trials_df : pd.DataFrame
        DataFrame containing trial information with columns for trial, time.reward, and time.cue.
    navigation_df : pd.DataFrame
        DataFrame containing navigation data with columns for trial, trial_phase, route, and route_change.
    navigation_routes_df : pd.DataFrame
        DataFrame containing navigation route information.
    n_previous_routes : int, optional
        Number of previous routes to consider for each trial (default is 4).
    Returns:
    --------
    pd.DataFrame
        DataFrame indexed by trial, with columns for route change times and latent state labels
        before and after each route change. The columns are multi-indexed with levels
        (route_change_i, time/latent_pre/latent_post) where i ranges from 0 to n_previous_routes-1.
    """
    trial2reward_time = trials_df.set_index(("trial", "")).time.reward.to_dict()
    trial2start_time = trials_df.set_index(("trial", "")).time.cue.to_dict()
    nav_df = navigation_routes_df
    trial_times = []
    for trial in nav_df.trial.dropna().unique():
        trial_df = nav_df[(nav_df.trial == trial) & (nav_df.trial_phase == "navigation")]
        if trial_df.empty:
            continue
        # get the latent state label before and after each route/state change
        route_start_label = trial_df.route.r.iloc[0]
        route_changes = trial_df[trial_df.route_change.astype(bool)]
        latent_labels = ["cue"] + [route_start_label] + route_changes.route.r.to_list() + ["reward"]
        latent_pre_post_change = [(latent_labels[i], latent_labels[i + 1]) for i in range(len(latent_labels) - 1)]
        latent_pre_post_change = latent_pre_post_change[-n_previous_routes:]
        labels = _nan_pad_list(latent_pre_post_change, n_previous_routes)
        route_change_times = route_changes.time.to_list()  # counting back from goal
        latent_state_change_times = (
            [trial2start_time[trial]] + route_change_times + [trial2reward_time[trial]]
        )  # times [cue/start, route_change_n, ....route_change_0, reward]
        times = latent_state_change_times[-n_previous_routes:]
        # add nans up to n_previous_routes if few routes used in trial
        times = _nan_pad_list(times, n_previous_routes)
        t = {("trial", ""): trial}
        for i, time in enumerate(times[::-1]):
            latent_pre, latent_post = labels[::-1][i]
            t[(f"route_change_{i}", "time")] = time  # route change 0 = reward
            t[(f"route_change_{i}", "latent_pre")] = latent_pre
            t[(f"route_change_{i}", "latent_post")] = latent_post
        trial_times.append(t)
    route_changes_df = pd.DataFrame(trial_times)
    route_changes_df.columns = pd.MultiIndex.from_tuples(route_changes_df.columns)
    route_changes_df.set_index(("trial", ""), inplace=True)
    # change output order to L-->R forward in time during trial
    route_changes_df = route_changes_df[route_changes_df.columns[::-1]]
    return route_changes_df


def _expand_nan_mask(nan_mask, n_timepoints=100, n_neurons=177):
    """
    Exapnds a nan_mask of shape [n_trials, n_alignment_times] to a mask of shape
    [n_trials, n_neurons, n_alignment_times*n_timespoints] to match the shape of the aligned activity array
    output by the aligned activity function.
    """
    n_trials = nan_mask.shape[0]
    trim_mask = nan_mask[:, :-1]  # last timepoint is reward and is always present
    expanded_timepoints = np.repeat(trim_mask[:, :, np.newaxis], n_timepoints, axis=2)
    expanded_timepoints = expanded_timepoints.reshape(n_trials, -1)
    expanded_neurons = np.repeat(expanded_timepoints[:, np.newaxis, :], n_neurons, axis=1)
    return expanded_neurons
