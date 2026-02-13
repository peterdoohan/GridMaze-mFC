"""
Decode both place-direction and distance to goal (over all spikes) and see if errors are dynamically correlated
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from polars import corr
import seaborn as sns
from matplotlib import lines, pyplot as plt
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_1samp

from GridMaze.analysis.core import folds
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.theta_mod import double_decoding as tdd

# %% global variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "theta_mod" / "double_decoding_simple"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

# %% load summary of what feature each neurons is tuned to from neGLM model comparisons


def get_tuned_neurons():
    from GridMaze.analysis.neGLM import load_model_sets as lms
    from GridMaze.analysis.neGLM import variance_explained as ve

    feature_tuned_df = ve.get_feature_tuned_df(
        lms.load_model_set_cv_scores("variance_explained_multiunit"),
        reduced_models=["remove_distance_to_goal", "remove_place_direction"],
    )

    distance_tuned = (
        feature_tuned_df[(feature_tuned_df.distance_to_goal & ~feature_tuned_df.place_direction)]
        .index.get_level_values(1)
        .values
    )
    place_tuned = (
        feature_tuned_df[(~feature_tuned_df.distance_to_goal & feature_tuned_df.place_direction)]
        .index.get_level_values(1)
        .values
    )
    return distance_tuned, place_tuned


# %% Functions


def plot_double_decoding_errors(
    results_df,
    dist_error_range=(-0.6, 0.6),
    place_error_range=(-0.015, 0.015),
    n_bins=10,
    pthresh=1e-3,
    print_stats=True,
    ax=None,
):
    """ """
    # set up figure
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2.5, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("decoded distance-to-goal (m) \n from distance-to-goal neurons")
    ax.set_ylabel("decoded distance-to-goal (m) \n from place-direction neurons")
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)

    df = results_df.copy()
    # include samples where place decoder on taj is balanced for past and future
    df = df[df.decoding_info.all_traj_defined]
    # remove any nans (from no-training data for test in place decoding insts)
    nan_mask = df.decoded_distance.isna().any(axis=1)
    df = df[~nan_mask].copy()
    # calc distance errors
    df[("decoding_error", "from_distance")] = df.decoded_distance.from_distance.sub(df.distance_bin_mid)
    df[("decoding_error", "from_place")] = df.decoded_distance.from_place  # .sub(df.distance_bin_mid)

    # bin distance errors
    dist_bins = np.linspace(dist_error_range[0], dist_error_range[1], n_bins + 1)
    df[("decoding_error_bin", "from_distance")] = pd.cut(df[("decoding_error", "from_distance")], bins=dist_bins)
    df[("decoding_error_bin_mid", "from_distance")] = df.decoding_error_bin.from_distance.apply(lambda x: x.mid)
    place_bins = np.linspace(place_error_range[0], place_error_range[1], n_bins + 1)
    df[("decoding_error_bin", "from_place")] = pd.cut(df[("decoding_error", "from_place")], bins=place_bins)
    df[("decoding_error_bin_mid", "from_place")] = df.decoding_error_bin.from_place.apply(lambda x: x.mid)

    # plot heatmap
    group_cols = [
        ("subject_ID", ""),
        ("decoding_error_bin_mid", "from_distance"),
        ("decoding_error_bin_mid", "from_place"),
    ]
    counts = df.groupby(group_cols, observed=True).size()
    # normalise each subject by the total numer of samples
    norm_counts = counts.groupby(level=0).transform(lambda x: x / x.sum())
    # average over subjects to get
    hm = norm_counts.groupby(level=[1, 2], observed=True).mean().unstack(level=1)
    if pthresh is not None:
        hm[hm < pthresh] = np.nan
    im = ax.imshow(
        hm.values.T,
        origin="lower",
        extent=[dist_error_range[0], dist_error_range[1], place_error_range[0], place_error_range[1]],
        aspect="auto",
        cmap="Purples",
    )
    cbar = f.colorbar(im, ax=ax, shrink=0.8)  # shrink factor < 1
    cbar.set_label("density")
    cbar.outline.set_visible(False)

    # plot linear fit for each subject
    subjects = df.subject_ID.unique()
    slopes, intercepts, corrs = [], [], []
    for subject in subjects:
        subj_df = df[df.subject_ID == subject]
        _x = subj_df[("decoding_error", "from_distance")].values
        _y = subj_df[("decoding_error", "from_place")].values
        corrs.append(np.corrcoef(_x, _y)[0, 1])
        # get linear fit with intercept
        slope, intercept = np.polyfit(_x, _y, 1)
        slopes.append(slope)
        intercepts.append(intercept)
    mean_slope, mean_int = np.mean(slopes), np.mean(intercepts)
    sem_slope, sem_int = np.std(slopes) / np.sqrt(len(slopes)), np.std(intercepts) / np.sqrt(len(intercepts))
    _x_plot = np.linspace(dist_error_range[0] + 0.1, dist_error_range[1] - 0.1, 100)
    _y_plot = mean_slope * _x_plot + mean_int
    ax.plot(_x_plot, _y_plot, color="red")
    ax.fill_between(
        _x_plot,
        _y_plot - sem_slope * _x_plot - sem_int,
        _y_plot + sem_slope * _x_plot + sem_int,
        color="red",
        alpha=0.2,
    )
    if print_stats:
        # t-test slopes from 0
        t_stat, p_val = ttest_1samp(slopes, 0)
        print(f"t-test: t({(len(slopes)-1)})={t_stat:.3f}, p={p_val:.3f}")
        # mean corr
        mean_corr = np.mean(corrs)
        sem_corr = np.std(corrs) / np.sqrt(len(corrs))
        print(f"mean corr.: r={mean_corr:.3f} +/- {sem_corr:.3f}")


def get_double_decoding_df(verbose=True, n_jobs=-1, save=False):
    """
    version notes:
    - _df: first attempt strictly dist only and pd only neurons into decoders
    - _df2: use only dist neurons for dist decoder, and all other for pd decoder (still orth, gives pd best we can do)
    - _df3: linearised +/- 3 decoding from place, fixed bug in neuron tuned defs?
    - _df4: remebred bias in place error if, now balanced, also fixed bug in "all other" for place tuned neurons
    """
    save_path = RESULTS_DIR / "double_decoding_simple_df4.parquet"
    if save_path.exists() and not save:
        if verbose:
            print(f"Loading existing double results from {save_path} ...")
        return pd.read_parquet(save_path)

    if verbose:
        print("Loading tuned neurons from neGLM results...")
    place_tuned_neurons, distance_tuned_neurons = get_tuned_neurons()

    results_dfs = []
    for subject_ID in SUBJECT_IDS:
        if verbose:
            print(f"Loading sessions for {subject_ID} ...")
        sessions = gs.get_maze_sessions(
            subject_IDs=[subject_ID],
            maze_names="all",
            days_on_maze="late",
            with_data=[
                "navigation_df",
                "cluster_metrics",
                "trials_df",
                "navigation_spike_counts_df",
            ],
            must_have_data=True,
        )
        if n_jobs:
            dfs = Parallel(n_jobs=n_jobs)(
                delayed(get_session_double_decoding_df)(
                    session,
                    place_tuned_neurons="all_other",  # use non-dist neurons
                    distance_tuned_neurons=distance_tuned_neurons,
                    verbose=verbose,
                )
                for session in sessions
            )
        else:
            dfs = [
                get_session_double_decoding_df(
                    session,
                    place_tuned_neurons="all_other",
                    distance_tuned_neurons=distance_tuned_neurons,
                    verbose=verbose,
                )
                for session in sessions
            ]
        if verbose:
            valid_outputs = [df is not None for df in dfs]
            print(f"Decoded {np.sum(valid_outputs)}/{len(dfs)} sessions for {subject_ID}:")
        results_dfs.extend(dfs)
    results_df = pd.concat(results_dfs, axis=0, ignore_index=True)
    if save:
        results_df.to_parquet(save_path)
        if verbose:
            print(f"Saved double decoding results to {save_path}.")
    return results_df


def quick_plot(res):

    df = res[res.decoded_distance.notna().all(axis=1)]
    dist_error = res.decoded_distance.from_distance - res.distance_to_goal.geodesic
    place_error = res.decoded_distance.from_place  # already error from place on traj
    plt.scatter(dist_error, place_error, s=1)
    plt.show()
    # corr = np.corrcoef(dist_error.values.astype(float), place_error.values.astype(float))[0, 1]
    # print(f"corr: {corr:.3f}")
    plt.hist(dist_error.dropna(), bins=100)
    plt.show()
    print(f"dist error: {dist_error.mean():.3f} +/- {dist_error.sem():.3f}")
    plt.hist(place_error.dropna(), bins=100)
    plt.show()
    print(f"place error: {place_error.mean():.3f} +/- {place_error.sem():.3f}")
    return


def get_session_double_decoding_df(
    session,
    resolution=0.1,
    sum_spike_window=0.4,
    moving_only=True,
    bin_spacing=0.08,
    max_distance=None,
    max_steps_from_goal=30,
    min_neurons_for_decoding=5,
    n_folds=8,
    sqrt_spikes=True,
    normalise_X=True,
    alphas=(1e-2, 1),  # opt
    output="weighted",
    verbose=True,
    place_tuned_neurons=None,
    distance_tuned_neurons=None,
    place_offset=4,
    permute=False,
):
    """ """
    # load data
    if verbose:
        print(f"{session.name}: loading input data...")
    input_data = tdd.get_input_data(
        session,
        theta_split=False,
        resolution=resolution,
        sum_spike_window=sum_spike_window,
        moving_only=moving_only,
        bin_spacing=bin_spacing,
        max_distance=max_distance,
        max_steps_to_goal=max_steps_from_goal,
        place_offset=place_offset,
        all_offset_defined=True,
        permute=permute,
    )
    input_data = input_data.droplevel(2, axis=1)
    # split neurons by place and distanced tunned
    cluster_unique_IDs = input_data.spike_count.columns.values
    if distance_tuned_neurons is None:
        place_tuned_neurons, distance_tuned_neurons = get_tuned_neurons()
    dist_neurons = [n for n in cluster_unique_IDs if n in distance_tuned_neurons]
    if place_tuned_neurons == "all_other":
        place_neurons = [n for n in cluster_unique_IDs if n not in dist_neurons]
    else:
        place_neurons = [n for n in cluster_unique_IDs if n in place_tuned_neurons]

    # only proceed with decoding if we have enough neurons of each type
    if len(place_neurons) < min_neurons_for_decoding or len(dist_neurons) < min_neurons_for_decoding:
        if verbose:
            print(
                f"Not enough neurons for decoding in {session.name} \n (place: {len(place_neurons)}, distance: {len(dist_neurons)})"
            )
            return None
    # generate variables to be used across folds, reg validation etc.
    distances = np.sort(input_data.distance_bin_mid.unique())  # in order corresponding to bin_id [0, 1, ...]
    distance_bin_ids = np.sort(input_data.distance_bin_id.unique())
    all_pairs_path_length = tdd._get_all_pairs_path_length(session)
    folds_df = folds.get_folds_df(
        session,
        goal_stratified=False,
        n_folds=n_folds,
        return_unique_IDs=False,
    )

    # init results df
    results_df = pd.concat(
        [
            input_data.drop(["spike_count", "past", "future"], axis=1, level=0).copy(),
            pd.DataFrame(
                index=input_data.index,
                columns=pd.MultiIndex.from_product((["decoded_distance"], ["from_distance", "from_place"])),
            ),
        ],
        axis=1,
    )
    results_df[("decoding_info", "place_in_train")] = False
    results_df[("decoding_info", "all_traj_defined")] = False
    results_df[("decoding_info", "n_dist_neurons")] = len(dist_neurons)
    results_df[("decoding_info", "n_place_neurons")] = len(place_neurons)

    _folds = folds_df.columns.get_level_values(0).unique()
    for fold in _folds:
        if verbose:
            print(fold)
        fold_df = folds_df[fold]
        train_trials, test_trials = [fold_df[t].unstack().dropna().values for t in ["train", "test"]]
        train_df, test_df = [input_data[input_data.trial.isin(trials)] for trials in [train_trials, test_trials]]
        # train decoder on mean spikes across theta phases
        Xd_train, Xd_test = [df.spike_count[dist_neurons].values for df in [train_df, test_df]]
        Xp_train, Xp_test = [df.spike_count[place_neurons].values for df in [train_df, test_df]]
        if sqrt_spikes:
            Xd_train, Xd_test = np.sqrt(Xd_train), np.sqrt(Xd_test)
            Xp_train, Xp_test = np.sqrt(Xp_train), np.sqrt(Xp_test)
        if normalise_X:
            scaler_d = StandardScaler().fit(Xd_train)
            Xd_train, Xd_test = scaler_d.transform(Xd_train), scaler_d.transform(Xd_test)
            scaler_p = StandardScaler().fit(Xp_train)
            Xp_train, Xp_test = scaler_p.transform(Xp_train), scaler_p.transform(Xp_test)
        # decoder either distace-to-goal or place (we will set up different decoders for each)
        Yd_train, Yd_test = [df.distance_bin_id.values for df in [train_df, test_df]]
        Yp_train, Yp_test = [df.maze_position.simple.values for df in [train_df, test_df]]
        # optionaly find optimal xval regularisation
        if alphas == "opt":
            if verbose:
                print("    Finding optimal alpha for distance decoder...")
            d_alpha = get_opt_alpha(
                fold_df,
                train_df,
                var="distance_to_goal",
                include_neurons=dist_neurons,
                normalise_X=normalise_X,
                sqrt_spikes=sqrt_spikes,
                distances=distances,
                distance_bin_ids=distance_bin_ids,
                output=output,
                verbose=verbose,
            )
            if verbose:
                print("    Finding optimal alpha for place decoder...")
            p_alpha = get_opt_alpha(
                fold_df,
                train_df,
                var="place",
                include_neurons=place_neurons,
                normalise_X=normalise_X,
                sqrt_spikes=sqrt_spikes,
                output=output,
                all_pairs_path_length=all_pairs_path_length,
                verbose=verbose,
            )
            if verbose:
                print(f"Optimal alpha: distance decoder: {d_alpha:.4f}, place decoder: {p_alpha:.4f}")
        else:
            d_alpha, p_alpha = alphas
        # train decoders
        d_decoder = LogisticRegression(C=d_alpha, random_state=0, max_iter=10_000, class_weight="balanced")
        d_decoder.fit(Xd_train, Yd_train)
        train_distances_bin_ids = d_decoder.classes_
        p_decoder = LogisticRegression(C=p_alpha, random_state=0, max_iter=10_000, class_weight="balanced")
        p_decoder.fit(Xp_train, Yp_train)
        train_locations = p_decoder.classes_
        # test decoders
        Yd_prob = d_decoder.predict_proba(Xd_test)
        d_pred = tdd._get_distance_pred_distance(
            Yd_prob,
            distances=distances,
            distance_bin_ids=distance_bin_ids,
            decoder_classes=train_distances_bin_ids,
            output=output,
        )
        results_df.loc[test_df.index, ("decoded_distance", "from_distance")] = d_pred
        Yp_prob = p_decoder.predict_proba(Xp_test)
        p_pred, p_traj_def, p_in_train = tdd._get_place_pred_distance(
            Yp_prob,
            Yp_test,
            test_df,
            decoder_classes=train_locations,
            all_pairs_path_length=all_pairs_path_length,
            restrict_to_traj=True,
            output=output,
            distance_ref="pos",
            return_as="all",
        )
        results_df.loc[test_df.index, ("decoded_distance", "from_place")] = p_pred
        results_df.loc[test_df.index, ("decoding_info", "place_in_train")] = p_in_train
        results_df.loc[test_df.index, ("decoding_info", "all_traj_defined")] = p_traj_def

    return results_df.reset_index(drop=True)


def get_opt_alpha(
    fold_df,
    train_df,
    var="distance_to_goal",
    include_neurons=None,
    normalise_X=True,
    sqrt_spikes=True,
    reg_range=np.logspace(-4, 4, 10),
    output="weighted",
    all_pairs_path_length=None,
    distances=None,
    distance_bin_ids=None,
    verbose=False,
):
    """ """
    # check inputs
    if var not in ["distance_to_goal", "place"]:
        raise ValueError(f"var must be 'distance_to_goal' or 'place'.")
    if var == "place" and all_pairs_path_length is None:
        raise ValueError(f"Must provide all_pairs_path_length for place decoding.")
    if var == "distance_to_goal" and distances is None:
        raise ValueError(f"Must provide distances for distance_to_goal decoding.")

    vfolds_df = fold_df.train
    vfolds = vfolds_df.columns
    results = np.zeros((len(vfolds), len(reg_range)))
    for i, vfold in enumerate(vfolds):
        if verbose:
            print(f"        vfold: {i}")
        val_trials = vfolds_df[vfold].dropna().values
        train_trials = vfolds_df[[t for t in vfolds if t != vfold]].unstack().dropna().values
        _train_df = train_df[train_df.trial.isin(train_trials)]
        _val_df = train_df[train_df.trial.isin(val_trials)]
        # train and test on average spikes over theta phases (reg search is theta independent)
        if include_neurons is not None:
            X_train, X_val = [df.spike_count[include_neurons].values for df in [_train_df, _val_df]]
        else:
            X_train, X_val = [df.spike_count.values for df in [_train_df, _val_df]]
        if X_train.shape[0] == 0 or X_val.shape[0] == 0:
            continue
        if var == "place":
            Y_train, Y_val = [df.maze_position.simple.values for df in [_train_df, _val_df]]
        if var == "distance_to_goal":
            Y_train, Y_val = [df.distance_bin_id.values for df in [_train_df, _val_df]]
        val_distance_to_goal = _val_df.distance_bin_mid.values
        # standardise
        if sqrt_spikes:
            X_train, X_val = np.sqrt(X_train), np.sqrt(X_val)
        if normalise_X:
            scaler = StandardScaler().fit(X_train)
            X_train, X_val = scaler.transform(X_train), scaler.transform(X_val)
        # fit model
        for j, alpha in enumerate(reg_range):
            decoder = LogisticRegression(C=alpha, random_state=0, max_iter=10_000, class_weight="balanced")
            decoder.fit(X_train, Y_train)
            decoder_classes = decoder.classes_
            Yprob = decoder.predict_proba(X_val)
            if var == "distance_to_goal":
                pred_dist = tdd._get_distance_pred_distance(
                    Yprob=Yprob,
                    distances=distances,
                    distance_bin_ids=distance_bin_ids,
                    decoder_classes=decoder_classes,
                    output=output,
                )
            if var == "place":
                pred_dist = tdd._get_place_pred_distance(
                    Yprob,
                    Y_val,
                    _val_df,
                    decoder_classes,
                    all_pairs_path_length,
                    output=output,
                    distance_ref="pos",
                    restrict_to_traj=True,
                    return_as="dist",
                )
            scores = val_distance_to_goal - pred_dist
            if not np.isfinite(scores).any():
                results[i, j] = np.nan
                continue
            results[i, j] = np.nanmean(scores**2)
    opt_alpha = reg_range[np.nanmean(results, axis=0).argmin()]
    return opt_alpha
