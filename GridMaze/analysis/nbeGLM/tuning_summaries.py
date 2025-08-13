"""
Use nbeGLM to find cells with only place-direction tuning OR only distance-to-goal tuning and revisit population
tuning summary methods. i.e distance-to-goal -> population heatmap, place-direction -> NMF/PCA components
@peterdoohan
"""

# %% Imports

from matplotlib import pyplot as plt

from GridMaze.analysis.distance_to_goal import population_tuning as dpt
from GridMaze.analysis.place_direction import dimensionality_reduction as ppt

from GridMaze.analysis.nbeGLM import load_model_sets as lms
from GridMaze.analysis.nbeGLM import variance_explained as ve

# %% Global Variables

# %% Functions


def get_single_tuned_clusters(feature="distance_to_goal", maze_names=["maze_1", "maze_2", "rooms_maze"]):
    """ """
    # load model set with full model (dtg, pd and ea feature groups) and reduced models for each feature
    results_df = lms.load_model_set_cv_scores("variance_explained", maze_names=maze_names, all_completed=True)
    # for every cell-feature check if cpd is sig > 0 across folds
    feature_tuned_df = ve.get_feature_tuned_df(
        results_df,
        reduced_models=[
            "remove_distance_to_goal",
            "remove_place_direction",
            "remove_egocentric_action_action",
        ],
        multiple_comparisons_corrected=False,
        alpha=0.05,
    )
    # filter for clusters tuned to just one feature
    if feature == "distance_to_goal":
        mask = (
            feature_tuned_df.distance_to_goal
            & ~feature_tuned_df.place_direction
            & ~feature_tuned_df.egocentric_action_action
        )
    elif feature == "place_direction":
        mask = (
            ~feature_tuned_df.distance_to_goal
            & feature_tuned_df.place_direction
            & ~feature_tuned_df.egocentric_action_action
        )
    else:
        raise ValueError("Feature must be 'distance_to_goal' or 'place_direction'")
    # output cluster unique IDs
    select_cluster_df = feature_tuned_df[mask]
    single_tuned_clusters = select_cluster_df.index.get_level_values(1).to_list()
    return single_tuned_clusters


# %% distance to goal tuning


def plot_unique_distance_to_goal_tuning_heatmap(pop_tuning_df, axes=None, v_range=(-1, 2.5)):
    """ """
    if axes is None:
        f, axes = plt.subplots(2, 1, figsize=(3, 6), height_ratios=[1.5, 1])
    for ax, sign in zip(axes, ["pos", "neg"]):
        dpt.plot_distance_tunned_heatmap(
            pop_tuning_df,
            ax=ax,
            cv_fit=False,
            sign=sign,
            v_range=v_range,
        )


def get_population_unique_distance_to_goal_tuning_df():
    """ """
    # get all distance to goal tuning curves
    population_distance_tuning = dpt.get_population_tuning_df(
        late_sessions=True,  # late session only in nbeGLM analyses
        metrics=("distance_to_goal", "geodesic"),
        min_split_half_corr=None,
    )
    # get clusters only tuned to distance-to-goal
    single_tuned_clusters = get_single_tuned_clusters("distance_to_goal", maze_names=["maze_1", "maze_2", "rooms_maze"])
    # filter population tuning df
    pop_dist_tuning = population_distance_tuning.set_index("cluster_unique_ID").loc[single_tuned_clusters]
    return pop_dist_tuning


# %% place-direction tuning


def plot_unique_place_direction_components(
    place_direction_tuning,
    simple_maze,
    dim_red="nmf",
    n_components=8,
    axes=None,
):
    """ """
    if axes is None:
        f, axes = plt.subplots(1, n_components, figsize=(6 * n_components, 6))
    if dim_red == "nmf":
        ppt.plot_nmf_components(place_direction_tuning, simple_maze, n_components, axes=axes)
    elif dim_red == "pca":
        ppt.plot_pca_components(place_direction_tuning, simple_maze, n_components, axes=axes)
    else:
        raise ValueError("dim_red must be 'nmf' or 'pca'")


def get_population_unique_place_direction_tuning_df(maze_name="maze_1"):
    # get all place-direction tuning curves
    population_place_direction_tuning = ppt.get_population_place_direction_tuning(
        subject_IDs="all",
        maze_name=maze_name,
        late_sessions=True,  # late session only in nbeGLM analysis
        include_multi_unit=False,
        fill_nans="mean",
        normalisation="length",
        min_split_corr=None,
        max_steps_to_goal=30,
        place_direction_tuned=False,
    )
    # get clusters only tuned to place-direction
    single_tuned_clusters = get_single_tuned_clusters(feature="place_direction", maze_names=[maze_name])
    # filter
    place_direction_tuning = population_place_direction_tuning.loc[single_tuned_clusters]
    return place_direction_tuning
