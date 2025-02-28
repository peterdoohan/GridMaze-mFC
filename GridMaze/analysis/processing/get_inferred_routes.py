"""
This special preprocessing library takes Xiao's inferred_route preproessed data (maybe a special case of 
externally processed analysis data) and converts it to a format that can be integrated with other analysis data
structures in this project.
"""

# %% Imports
import networkx as nx
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from ...maze import representations as mr
from ..core import load_data

# %% Global Variables
from ...paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "maze_measurements.json", "r") as f:
    MAZE_MEASUREMENTS = json.load(f)
GRIDMAZE_SIZE = MAZE_MEASUREMENTS["maze_node_dimensions"]

with open(EXPERIMENT_INFO_PATH / "maze_day2date.json", "r") as f:
    MAZE_DAY2DATE = json.load(f)

PREPROCESSED_ROUTES_PATH = Path("../data/preprocessed_data/inferred_routes/best_routes_07_08_2024")
PREPROCESSED_ROUTE_PROBABILITIES_PATH = Path(
    "../data/preprocessed_data/inferred_routes/inferred_viterbi_routes_24_10_15"
)

MAZE_NAME_MAP = {  # mapping from Xiao's maze names to the maze names used in this project
    "maze_1": "maze1",
    "maze_2": "maze2",
    "rooms_maze": "maze3",
}

# %% Main Function


def get_routes_prior(processed_data_path, analysis_data_path):
    """Route priors are scalar values for a given route, functions as the variance explained in the behaviour by the route."""
    routes_data_path = get_inferred_route_preprocessed_data_path(processed_data_path, "routes")
    # translate from Xiao's representtion to GridMaze representation
    loaded_data = torch.load(routes_data_path, map_location=torch.device("cpu"), weights_only=True)
    pi = loaded_data["pi"]
    routes_prior = pi.sigmoid() / (pi.sigmoid().sum(dim=-1, keepdims=True) + 1e-12)
    routes_prior = routes_prior.numpy()[:-1]
    route2prior = {f"route_{i}": float(prior) for i, prior in enumerate(routes_prior)}
    return route2prior


def get_routes_df(processed_data_path, analysis_data_path):
    """
    Finds subject_ID and maze_name from processed_data_path, the loads and translates Xiao's inferred routes
    data (probability of occupying each place direction given you are on route r) to a pandas DataFrame.
    respembling a cluster place direction heatmap.
    To be saved out as routes.parquet in each session folder.
    Note: in the current determination of Xiao's routes, we have one set of routes defined for each subject over
    a whole maze. In future it might be desirable to define routes specifically per session using only the behavioural
    history up to that point, not the entire data for a given maze.
    """
    # get preprocessed data path
    routes_data_path = get_inferred_route_preprocessed_data_path(processed_data_path, "routes")
    # translate from Xiao's representtion to GridMaze representation
    loaded_data = torch.load(routes_data_path, map_location=torch.device("cpu"), weights_only=True)
    R = loaded_data["R"]
    routes = R.sigmoid() / (R.sigmoid().sum(dim=-1, keepdims=True) + 1e-12)
    routes = routes.numpy()[:-1]  # last 'route' is for fitting off-task/ non-route behaviour, ignore.
    place_direction_index = vector_index_to_place_direction_index()
    routes_df = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(place_direction_index, names=("maze_position", "direction")), data=routes.T
    )
    routes_df.sort_index(inplace=True)
    # remove invalid place directions
    maze_name = _date2maze_name(processed_data_path.name.split(".")[0])
    simple_maze = mr.get_simple_maze(maze_name)
    valid_place_directions = mr.get_maze_place_direction_pairs(simple_maze, edges=False)
    routes_df = routes_df.loc[valid_place_directions]
    routes_df = routes_df[routes_df.index.isin(valid_place_directions)]
    # add edge direction values interpolated from values at adjacent nodes
    interpolated_routes_df = interpolate_edge_direction_values(routes_df, simple_maze)
    # change column names
    interpolated_routes_df.columns = [f"route_{i}" for i in range(interpolated_routes_df.shape[1])]
    return interpolated_routes_df.T


def get_navigation_routes_df(processed_data_path, analysis_data_path, n=4):
    """
    Return a MultiIndex dataframe with, the prbability you are on a given route at any time (frame of video), generated from
    get_route_probabilities_df. And information about the current route you are on (route with max probability now), the next
    route you will be on the route you were previously on etc, generated from get_future_routes_df.
    """
    route_probabilities_df = get_route_probabilities_df(processed_data_path)
    future_routes_df = get_future_routes_df(processed_data_path, n=n)
    route_progress_df = get_routes_progress_df(processed_data_path, analysis_data_path)
    navigation_routes_df = pd.concat([route_probabilities_df, future_routes_df, route_progress_df], axis=1)
    navigation_routes_df[("optimal_route", "")] = get_route_optimality_df(processed_data_path, analysis_data_path)
    return navigation_routes_df


def get_route_optimality_df(processed_data_path, analysis_data_path):
    """
    Records if a route was optimal or not, with a soft optimality constraint of that the distance to goal,
    and the end of the route is less than than the distance to goal at the start of the route.
    """
    # load data
    try:
        navigation_df = load_data.load(analysis_data_path / "frames.navigation.parquet")
        future_routes_df = get_future_routes_df(processed_data_path)
    except FileNotFoundError:
        print(f"Could not load data needed to make get_route_optimality df from {processed_data_path}. Returning None.")
        return None
    # comnbine data
    nav_routes_df = pd.concat([navigation_df, future_routes_df.reset_index(drop=True)], axis=1)
    route_optimalities = pd.Series(index=nav_routes_df.index, data=np.nan)
    for trial in nav_routes_df.trial.dropna().unique():
        trial_df = nav_routes_df[nav_routes_df.trial == trial]
        route_order2optimal = {}
        for r in trial_df.route_order.to_goal.dropna().unique():
            route_df = trial_df[trial_df.route_order.to_goal == r]
            dtg = route_df.distance_to_goal.geodesic  # distance to goal
            optimal = 1 if dtg.iloc[0] > dtg.iloc[-1] else 0
            route_order2optimal[r] = optimal
        route_optimalities.update(trial_df.route_order.to_goal.map(route_order2optimal))
    return route_optimalities.to_numpy()


def get_routes_progress_df(processed_data_path, analysis_data_path):
    """ """
    # load data
    try:
        session_info = load_data.load(processed_data_path / "session_info.json")
        skeleton_maze = mr.skeleton_maze(session_info["maze_structure"])
        navigation_df = load_data.load(analysis_data_path / "frames.navigation.parquet")
        future_routes_df = get_future_routes_df(processed_data_path)
    except FileNotFoundError:
        print(f"Could not load data needed to make routes_progress_df from {processed_data_path}. Returning None.")
        return None
    # filter and combine data
    navigation_df.loc[:, ("route", "")] = future_routes_df.route.r.reset_index(drop=True)
    # precompute path distances data
    sk_label2sk_coord = {v: k for k, v in nx.get_node_attributes(skeleton_maze, "label").items()}
    shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(skeleton_maze, weight="weight"))
    # loop over trials
    route_progress_df = pd.DataFrame(
        index=navigation_df.index,
        columns=pd.MultiIndex.from_product([["route_progress"], ["time", "path_length"]]),
        data=np.nan,
    )
    for trial in navigation_df.trial.dropna().unique():
        trial_mask = (navigation_df.trial == trial) & (navigation_df.trial_phase == "navigation")
        trial_df = navigation_df[trial_mask]
        if trial_df.empty:
            continue
        route_sequence = trial_df[trial_df.route != trial_df.route.shift(1)].route
        # get start index, route change indicies and end index to segment out routes
        route_segment_indecies = np.hstack([route_sequence.index.values, trial_df.index[-1]])
        for i in range(len(route_segment_indecies) - 1):
            start, end = route_segment_indecies[i], route_segment_indecies[i + 1]
            route_segment_df = trial_df.loc[start:end]
            # progress defined over future time (1 = start, 0=end, low to high same as distance metrics)
            time_progress = pd.Series(np.linspace(0, 1, len(route_segment_df))[::-1], index=route_segment_df.index)
            route_progress_df.loc[route_segment_df.index, ("route_progress", "time")] = time_progress
            # progress defined over future path length on route (low to high same as above)
            path_distance_progress = _get_route_path_distance_progress(
                route_segment_df, sk_label2sk_coord, shortest_path_lengths
            )
            route_progress_df.loc[route_segment_df.index, ("route_progress", "path_length")] = path_distance_progress
    route_progress_df.index = future_routes_df.index  # set time as index
    return route_progress_df


def _get_route_path_distance_progress(
    route_segment_df, sk_label2sk_coord, shortest_path_lengths, remove_backtracking=True
):
    """ """
    sk_pos = route_segment_df.maze_position.skeleton
    sk_traj = sk_pos[sk_pos.ne(sk_pos.shift(-1))]
    # removes instances where sk positions flutter between two locations (this could easily rack up path length)
    if remove_backtracking:
        mask = pd.Series(True, index=sk_traj.index)
        for i in range(len(sk_traj) - 2):
            if sk_traj.iloc[i] == sk_traj.iloc[i + 2]:
                mask.iloc[i + 1] = mask.iloc[i + 2] = False
            elif sk_traj.iloc[i] == sk_traj.iloc[i + 1]:
                mask.iloc[i + 1] = False
        sk_traj = sk_traj[mask]
    # calculate path length
    traj = sk_traj.map(sk_label2sk_coord)
    np_traj = traj.to_numpy()
    step_distances = [shortest_path_lengths[np_traj[i]][np_traj[i + 1]] for i in range(len(np_traj) - 1)]
    step_distances.append(0)
    future_distances = pd.Series(np.array(step_distances)[::-1].cumsum()[::-1], index=sk_traj.index)
    route_progress = future_distances / future_distances.max()
    filled_route_progress = pd.Series(np.nan, index=sk_pos.index)
    filled_route_progress.update(route_progress)
    filled_route_progress = filled_route_progress.bfill()
    filled_route_progress.fillna(0, inplace=True)  # replace nans at the end of trajectories to 0
    return filled_route_progress


def get_future_routes_df(processed_data_path, n=3):
    """
    Generates a DataFrame containing inferred future and past routes for each frame in the trajectories data.

    Parameters:
    processed_data_path (Path): Path to the directory containing the processed data files.
    n (int): Number of future and past routes to infer. Default is 4.

    Returns:
    DataFrame: A DataFrame indexed by time, with columns representing the current route, future routes (r+1, r+2, ..., r+n),
               past routes (r-1, r-2, ..., r-n), and a boolean column indicating route changes. Frames have np.nan value if not
               during navigaiton Returns None if data files. Returns None if data cannot be loaded.

    Note:
    - The last decision point at each trial (as subject enter goal location is removed from Xiao's preprocessed routes data
        because it often redundantly switches routes).
    """
    # load_data
    try:
        trajectories_df = load_data.load(processed_data_path / "frames.trajectories.htsv")
        trial_info_df = load_data.load(processed_data_path / "frames.trialInfo.htsv")
        routes_analysis_df = _get_routes_analysis_df(processed_data_path)
    except FileNotFoundError:
        print(f"Could not load data needed to make future_routes_df from {processed_data_path}. Returning None.")
        return None
    # take current route as the viterbi path output from Xiao's analysis (mapped from into to route_name str)
    route_probability_columns = [c for c in routes_analysis_df.columns if "route" in c.split("_")]
    n_routes = len(route_probability_columns)
    route_id2name = {i: f"route_{i}" for i in range(n_routes - 1)}
    route_id2name[n_routes - 1] = "non_route"
    routes_analysis_df["current_route"] = routes_analysis_df.viterbi_path.map(route_id2name)
    # initialise dataframe with index node transitions
    route_cols = (
        ["r"] + [f"r+{i}" for i in range(1, n + 1)] + [f"r-{i}" for i in range(1, n + 1)]
    )  # r = route, r+1 = next route, r-1 = previous route
    nodes_df = pd.DataFrame(index=routes_analysis_df.index, columns=route_cols, data=None)
    nodes_df["time"] = routes_analysis_df.time
    # initialise dataframe with index frames
    frames_df = pd.DataFrame(index=trajectories_df.index, columns=route_cols, data=None)
    frames_df["time"] = trajectories_df.time
    # iterate over trials
    frame_trials_dfs = []
    for trial in routes_analysis_df.trial.unique():
        trial_mask = trial_info_df.trial.eq(trial)
        frames_trial_df = frames_df[trial_mask].copy()
        frame_times = frames_trial_df.time
        node_trial_df = nodes_df[routes_analysis_df.trial == trial].copy()
        node_trial_df.loc[:, "r"] = routes_analysis_df.loc[node_trial_df.index, "current_route"]
        route_change = node_trial_df.r != node_trial_df.r.shift(1)
        # dataframe that just includes successive routes
        distilled_routes = node_trial_df[route_change].r
        distilled_routes_df = pd.DataFrame(index=distilled_routes.index, columns=route_cols, data=None)
        distilled_routes_df["r"] = distilled_routes
        for i in range(1, n + 1):  # next routes
            distilled_routes_df[f"r+{i}"] = distilled_routes.shift(-i)
        for i in range(1, n + 1):  # previous routes
            distilled_routes_df[f"r-{i}"] = distilled_routes.shift(+i)
        distilled_routes_df.index = node_trial_df[route_change].time
        # find time corresponding to each route change and insert into frames dataframe
        for time, row in distilled_routes_df.iterrows():
            frames_index = (frame_times - time).abs().idxmin()
            frames_trial_df.loc[frames_index, route_cols] = row.values
        frames_trial_df = frames_trial_df.infer_objects(copy=False)
        # fill frames between route changes appropriately
        frames_trial_df = frames_trial_df.ffill(limit_area="inside")
        for col in route_cols:
            frames_trial_df[col] = _forward_fill_untill_none(frames_trial_df[col])
        route_change = frames_trial_df.r != frames_trial_df.r.shift(1)
        route_change.iloc[0] = False  # first row is always a change
        frames_trial_df["route_change"] = route_change
        # add n_routes (number of routes in trial)
        frames_trial_df["n_routes"] = len(distilled_routes)
        # get route order (counting to goal and from goal)
        route_order_to_goal = route_change.cumsum() + 1
        route_order_from_goal = (route_order_to_goal - route_order_to_goal.max()).abs()
        frames_trial_df["to_goal"] = route_order_to_goal
        frames_trial_df["from_goal"] = route_order_from_goal
        frame_trials_dfs.append(frames_trial_df)
    future_routes_df = pd.concat(frame_trials_dfs, axis=0)
    future_routes_df = future_routes_df.reindex(trajectories_df.index)  # add back frames before and after last trial
    future_routes_df[trial_info_df.trial_phase != "navigation"] = np.nan  # map non-navigation frames back to NaN
    future_routes_df.time = trajectories_df.time
    # configure to multiindex
    future_routes_df = future_routes_df.set_index("time")
    future_routes_df.columns = (
        pd.MultiIndex.from_product([["route"], route_cols]).append(
            pd.MultiIndex.from_tuples([("route_change", ""), ("n_routes", "")])
        )
    ).append(pd.MultiIndex.from_product([["route_order"], ["to_goal", "from_goal"]]))
    return future_routes_df


def get_route_probabilities_df(processed_data_path):
    """
    Generates a DataFrame containing route probabilities indexed by frame times.

    This function loads trajectory, trial information, and route analysis data from the specified
    processed data path. It then processes this data to create a DataFrame where each row corresponds
    to a frame time and contains the probabilities of different routes at that time. The probabilities
    are forward-filled and backfilled to ensure no missing values within navigation periods.

    Parameters:
    processed_data_path (Path): The path to the directory containing the processed data files.

    Returns:
    pd.DataFrame: A DataFrame indexed by frame times with multi-index columns containing route probabilities.
                  Frames that are not part of a navigation trial are set to NaN.
                  Returns None if the necessary data files are not found.
    """
    """ """
    # load_data
    try:
        trajectories_df = load_data.load(processed_data_path / "frames.trajectories.htsv")
        trial_info_df = load_data.load(processed_data_path / "frames.trialInfo.htsv")
        routes_analysis_df = _get_routes_analysis_df(processed_data_path)
    except FileNotFoundError:
        print(f"Could not load data needed to make future_routes_df from {processed_data_path}. Returning None.")
        return None
    # remove non route probability columns and index with times
    route_probability_columns = [c for c in routes_analysis_df.columns if "route" in c.split("_")]
    probabilities_df = routes_analysis_df[route_probability_columns]
    probabilities_df.index = routes_analysis_df.time
    frame_times = trajectories_df.time
    # initialise dataframe with index frames
    probabilities_frames_df = pd.DataFrame(index=trajectories_df.index, columns=probabilities_df.columns, data=None)
    # find time corresponding to each route change and insert into frames dataframe
    for time, row in probabilities_df.iterrows():
        frames_index = (frame_times - time).abs().idxmin()
        probabilities_frames_df.loc[frames_index] = row.values
    # forward fill frames between route changes (and to end of session), then backfill NaNs at start of session
    probabilities_frames_df = probabilities_frames_df.infer_objects(copy=False)
    probabilities_frames_df = probabilities_frames_df.ffill().bfill()
    # map out of trial non-navigation frames (that we over filled above) back to NaN
    probabilities_frames_df[trial_info_df.trial_phase != "navigation"] = np.nan
    # map frames where subjects are at the goal as NaN, where routes often switch redundantly
    probabilities_frames_df[trajectories_df.maze_position.simple == trial_info_df.goal] = np.nan
    # configre multiindex
    probabilities_frames_df.set_index(trajectories_df.time, inplace=True)
    probabilities_frames_df.columns = pd.MultiIndex.from_product(
        [["route_probability"], probabilities_frames_df.columns]
    )
    return probabilities_frames_df


# %% Supporting Functions


def get_inferred_route_preprocessed_data_path(processed_data_path, data_type):
    """
    data_type: str, one of ["routes", "route_probabilities"]
    Returns the path to the preprocessed data file containing the inferred routes or route probabilities
    for the given processed_data_path.
    """
    subject_ID = processed_data_path.parts[-2]
    session_date = processed_data_path.name.split(".")[0]
    maze_name = _date2maze_name(session_date)
    if data_type == "routes":
        return PREPROCESSED_ROUTES_PATH / f"{MAZE_NAME_MAP[maze_name]}_subject{subject_ID}.pt"
    elif data_type == "route_probabilities":
        return PREPROCESSED_ROUTE_PROBABILITIES_PATH / f"{MAZE_NAME_MAP[maze_name]}_subject{subject_ID}.csv"
    else:
        raise ValueError(f"Invalid data_type: {data_type}")


def _date2maze_name(date: str):
    for maze_name, day in MAZE_DAY2DATE.items():
        if date in day.values():
            return maze_name
    return None


def _date2day_on_maze(date: str):
    """ """
    for _, day2date in MAZE_DAY2DATE.items():
        for day, _date in day2date.items():
            if _date == date:
                return int(day)
    return None


def vector_index_to_place_direction_index():
    """Returns place-direction (tuples) index that maps to the corresponding elements in each routes vectors."""
    action_map = {0: "E", 1: "N", 2: "W", 3: "S"}  # Mapping for actions
    place_labels = [
        chr(65 + i) + str(j + 1) for i in range(GRIDMAZE_SIZE[0]) for j in range(GRIDMAZE_SIZE[1])
    ]  # Ordered alpha neurmeric states
    place_direction_index = [
        (place_label, action_map[action]) for action in range(len(action_map)) for place_label in place_labels
    ]
    return place_direction_index


def interpolate_edge_direction_values(node_routes_df, simple_maze):
    """
    Adds edge information to routes_df by interpolating each edge, direction the same value
    as the node it connects in that direction.
    """
    edge_directions_pairs = mr.get_maze_place_direction_pairs(simple_maze, nodes=False)
    edge_direction_values_df = pd.DataFrame(
        columns=node_routes_df.columns,
        index=pd.MultiIndex.from_tuples(edge_directions_pairs, names=("maze_position", "direction")),
    )
    for edge, dir in edge_directions_pairs:
        origin_node = _get_origin_node(edge, dir)
        origin_node_direction_values = node_routes_df.loc[(origin_node, dir)]
        edge_direction_values_df.loc[(edge, dir)] = origin_node_direction_values
    return pd.concat([node_routes_df, edge_direction_values_df], axis=0).sort_index()


def _get_origin_node(edge, dir):
    """
    Returns the origin node of the current edge in an action/direction.
    E.g. if I'm on 'B2-C2' moving 'E' I must have previously been on 'B2'.
    """
    node1, node2 = edge.split("-")
    if dir == "N" or dir == "E":
        return node1
    if dir == "S" or dir == "W":
        return node2


def _get_routes_analysis_df(processed_data_path):
    """
    Loads Xiao's rouutes analsis data structures from disk (including the probability of being on a given route given the current state action pair)
    Route probabilites are in columns route_i and non_route (which captures behaviour that is not using route planning).

    !Note! at current Xiao's preprocessed data structures have mislabeld route_9 probabilities as non_route and non_route values
    are missing. No worries we can fill in the gaps
    """
    routes_analysis_data_path = get_inferred_route_preprocessed_data_path(processed_data_path, "route_probabilities")
    route_analysis_df = pd.read_csv(routes_analysis_data_path)
    # extract just the trajectoies for the given day specified in the processed_data_path
    day_on_maze = _date2day_on_maze(processed_data_path.name.split(".")[0])
    route_analysis_df = route_analysis_df[route_analysis_df.day_on_maze == day_on_maze]
    # fix bug in this df
    route_analysis_df.rename(columns={"non_route": "route_9"}, inplace=True)
    route_columns = [c for c in route_analysis_df.columns if "route" in c.split("_")]
    route_analysis_df["non_route"] = route_analysis_df[route_columns].sum(axis=1).subtract(1).abs()
    return route_analysis_df


def _forward_fill_untill_none(series):
    """ """
    if series.isna().all():
        return series
    s = series.copy()
    last_route = s[~s.isna()].iloc[-1]
    first_route = s[~s.isna()].iloc[0]
    last_route_index = s[~s.isna()].index[-1]
    none_indices = s[s.apply(lambda x: x is None)].index
    if len(none_indices) == 0:  # current_route
        s = s.ffill()
        s = s.bfill()
        return s
    else:
        if (none_indices < last_route_index).all():  # backward route shift
            s = s.ffill()
            last_none_index = none_indices[-1]
            s.loc[last_none_index + 1] = first_route
            s = s.bfill(limit_area="inside")
            return s
        else:  # forward route shift
            first_none_index = none_indices[0]
            s.loc[first_none_index - 1] = last_route
            s = s.ffill(limit_area="inside")
            s = s.bfill()
            return s
