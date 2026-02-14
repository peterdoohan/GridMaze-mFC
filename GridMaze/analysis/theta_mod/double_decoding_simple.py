"""
Decode both place-direction and distance to goal (over all spikes) and see if errors are dynamically correlated
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_1samp

from GridMaze.maze import representations as mr

from GridMaze.analysis.core import folds
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.core import convert
from GridMaze.analysis.core import filter as filt
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import get_clusters as gc


from GridMaze.analysis.theta_mod import double_decoding as tdd
from GridMaze.analysis.place_direction import future_decoding as fd

# %% global variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "theta_mod" / "double_decoding_simple"

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as input_file:
    SUBJECT_IDS = json.load(input_file)

FRAME_RATE = 60
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
    dist_error_range=(-0.05, 0.05),
    place_error_range=(-0.05, 0.05),
    n_bins=15,
    pthresh=None,
    print_stats=True,
    ax=None,
):
    """ """
    # set up figure
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2.5, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("error (m) \n distance-to-goal")
    ax.set_ylabel("error (m) \n place-direction")
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)

    df = results_df.copy()
    dist_decode_valid = df.distance_decoding.drop(columns=["error"]).all(axis=1)
    place_decode_valid = df.place_decoding.drop(columns=["error"]).all(axis=1)
    valid_mask = dist_decode_valid & place_decode_valid
    df = df[valid_mask]
    # bin distance errors
    dist_bins = np.linspace(dist_error_range[0], dist_error_range[1], n_bins + 1)
    df[("distance_decoding", "error_bin")] = pd.cut(df[("distance_decoding", "error")], bins=dist_bins)
    df[("distance_decoding", "error_bin_mid")] = df.distance_decoding.error_bin.apply(lambda x: x.mid)
    place_bins = np.linspace(place_error_range[0], place_error_range[1], n_bins + 1)
    df[("place_decoding", "error_bin")] = pd.cut(df[("place_decoding", "error")], bins=place_bins)
    df[("place_decoding", "error_bin_mid")] = df.place_decoding.error_bin.apply(lambda x: x.mid)

    # plot heatmap
    group_cols = [
        ("subject_ID", ""),
        ("distance_decoding", "error_bin_mid"),
        ("place_decoding", "error_bin_mid"),
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
        _x = subj_df[("distance_decoding", "error")].values.astype(float)
        _y = subj_df[("place_decoding", "error")].values.astype(float)
        corrs.append(np.corrcoef(_x, _y)[0, 1])
        # get linear fit with intercept
        slope, intercept = np.polyfit(_x, _y, 1)
        slopes.append(slope)
        intercepts.append(intercept)
    mean_slope, mean_int = np.mean(slopes), np.mean(intercepts)
    sem_slope, sem_int = np.std(slopes) / np.sqrt(len(slopes)), np.std(intercepts) / np.sqrt(len(intercepts))
    _x_plot = np.linspace(dist_error_range[0] * 0.8, dist_error_range[1] * 0.8, 100)
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
    save_path = RESULTS_DIR / "double_decoding_simple_df5.parquet"
    if save_path.exists() and not save:
        if verbose:
            print(f"Loading existing double results from {save_path} ...")
        return pd.read_parquet(save_path)

    if verbose:
        print("Loading tuned neurons from neGLM results...")
    distance_tuned, place_tuned = get_tuned_neurons()

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
                    distance_tuned_neurons=distance_tuned,
                    verbose=verbose,
                )
                for session in sessions
            )
        else:
            dfs = [
                get_session_double_decoding_df(
                    session,
                    place_tuned_neurons="all_other",  # use non-dist neurons
                    distance_tuned_neurons=distance_tuned,
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


# %%
def quick_plot(res):
    f, axes = plt.subplots(1, 3, figsize=(9, 3))
    dist_decode_valid = res.distance_decoding.drop(columns=["error"]).all(axis=1)
    place_decode_valid = res.place_decoding.drop(columns=["error"]).all(axis=1)
    valid_mask = dist_decode_valid & place_decode_valid
    dist_error = res.distance_decoding.error.values[valid_mask]
    place_error = res.place_decoding.error.values[valid_mask]
    axes[0].scatter(dist_error, place_error, s=1)
    axes[1].hist(dist_error, bins=100)
    axes[2].hist(place_error, bins=100)
    corr = np.corrcoef(dist_error.astype(float), place_error.astype(float))[0, 1]
    print(f"corr: {corr:.3f}")
    print(f"dist error: {dist_error.mean():.3f} +/- {dist_error.std():.3f}")
    print(f"place error: {place_error.mean():.3f} +/- {place_error.std():.3f}")
    return


def get_session_double_decoding_df(
    session,
    resolution=0.2,
    sum_spike_window=None,
    moving_only=True,
    distance_bins=15,
    max_steps_from_goal=20,
    min_neurons_for_decoding=5,
    n_folds=10,
    sqrt_spikes=True,
    normalise_X=True,
    alphas=(1, 1),  # opt
    place_tuned_neurons=None,
    distance_tuned_neurons=None,
    place_offset=2,  # tower/edge
    distance_offset=1,  # bins
    permute=False,
    verbose=True,
):
    """ """
    # load data
    if verbose:
        print(f"{session.name}: loading input data...")
    input_data = get_input_data(
        session,
        theta_split=False,
        resolution=resolution,
        sum_spike_window=sum_spike_window,
        moving_only=moving_only,
        distance_bins=distance_bins,
        max_steps_to_goal=max_steps_from_goal,
        place_offset=place_offset,
        all_offset_defined=True,
        permute=permute,
    )

    # split neurons by place and distanced tunned
    cluster_unique_IDs = input_data.spike_count.columns.values
    if distance_tuned_neurons is None and place_tuned_neurons is None:
        dist_neurons = place_neurons = cluster_unique_IDs
    else:
        dist_neurons = [n for n in cluster_unique_IDs if n in distance_tuned_neurons]
        if place_tuned_neurons == "all_other":
            place_neurons = [n for n in cluster_unique_IDs if n not in dist_neurons]
        else:
            place_neurons = [n for n in cluster_unique_IDs if n in place_tuned_neurons]

    # check decoders have enough neurons
    if len(place_neurons) < min_neurons_for_decoding or len(dist_neurons) < min_neurons_for_decoding:
        if verbose:
            print(
                f"Not enough neurons for decoding in {session.name} \n (place: {len(place_neurons)}, distance: {len(dist_neurons)})"
            )
            return None

    # generate variables to be used across folds, reg validation etc.
    distances = np.sort(input_data.distance_bin_mid.unique())  # in order corresponding to bin_id [0, 1, ...]
    distance_bin_ids = np.sort(input_data.distance_bin_id.unique())
    all_pairs_path_length = _get_all_pairs_path_length(session)
    folds_df = folds.get_folds_df(
        session,
        goal_stratified=False,
        n_folds=n_folds,
        return_unique_IDs=False,
    )

    # init results df
    _decoding_cols = ["error", "test_in_train", "offset_in_train", "full_offset_defined"]
    results_df = pd.concat(
        [
            input_data.drop(["spike_count", "past", "future"], axis=1, level=0).copy(),
            pd.DataFrame(
                index=input_data.index,
                columns=pd.MultiIndex.from_product((["distance_decoding", "place_decoding"], _decoding_cols)),
            ),
        ],
        axis=1,
    )
    results_df[("decoding_info", "n_dist_neurons")] = len(dist_neurons)
    results_df[("decoding_info", "n_place_neurons")] = len(place_neurons)

    # perform cv double decodings
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
                distance_offset=distance_offset,
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
        d_res = distance_pred_distance_error(  # weighted_errs, test_in_train, offsets_in_train, full_offset_defined
            Yd_prob,
            Yd_test,
            distances=distances,
            distance_bin_ids=distance_bin_ids,
            decoder_classes=train_distances_bin_ids,
            distance_offset=distance_offset,
            return_as="all",
        )
        for c, res in zip(_decoding_cols, d_res):  # assign distance results to df
            results_df.loc[test_df.index, ("distance_decoding", c)] = res

        Yp_prob = p_decoder.predict_proba(Xp_test)
        p_res = place_pred_distance_error(
            Yp_prob,
            Yp_test,
            test_df,
            decoder_classes=train_locations,
            all_pairs_path_length=all_pairs_path_length,
            return_as="all",
        )
        for c, res in zip(_decoding_cols, p_res):  # assign place results to df
            results_df.loc[test_df.index, ("place_decoding", c)] = res

    return results_df.reset_index(drop=True)


def get_opt_alpha(
    fold_df,
    train_df,
    var="distance_to_goal",
    include_neurons=None,
    normalise_X=True,
    sqrt_spikes=True,
    reg_range=np.logspace(-4, 4, 10),
    all_pairs_path_length=None,
    distances=None,
    distance_bin_ids=None,
    distance_offset=1,
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
                errors = distance_pred_distance_error(
                    Yd_prob=Yprob,
                    Yd_test=Y_val,
                    distances=distances,
                    distance_bin_ids=distance_bin_ids,
                    decoder_classes=decoder_classes,
                    distance_offset=distance_offset,
                    return_as="error",
                )
            if var == "place":
                errors = place_pred_distance_error(
                    Yprob,
                    Y_val,
                    _val_df,
                    decoder_classes,
                    all_pairs_path_length,
                    return_as="error",
                )
            if not np.isfinite(errors).any():
                results[i, j] = np.nan
                continue
            results[i, j] = np.nanmean(errors**2)
    opt_alpha = reg_range[np.nanmean(results, axis=0).argmin()]
    return opt_alpha


# %% test evaluation functions


def distance_pred_distance_error(
    Yd_prob,
    Yd_test,
    distances,  # all distances bins (m)
    distance_bin_ids,  # all distance bin ids (corresponding to distances)
    decoder_classes,  # distance bins in training
    distance_offset,
    return_as="error",
    norm_offset_probs=True,
):
    """ """
    bin_width = np.diff(distances)[0]
    offsets = np.arange(1, distance_offset + 1)
    offset_distances = bin_width * offsets

    samples = Yd_prob.shape[0]
    full_offset_defined = np.ones(samples, dtype=bool)
    offsets_in_train = np.ones(samples, dtype=bool)
    test_in_train = np.ones(samples, dtype=bool)
    weighted_errs = np.zeros(samples)

    for i in range(samples):
        probs = Yd_prob[i]
        true_bin = Yd_test[i]
        if true_bin not in decoder_classes:
            full_offset_defined[i] = False
            test_in_train[i] = False
            weighted_errs[i] = np.nan
            offsets_in_train[i] = False
            continue
        future_candidates = true_bin - offsets  # future is by def closer to goal (lower bin id)
        past_candidates = true_bin + offsets
        all_candidates = np.concatenate([future_candidates, [true_bin], past_candidates])
        # check if all offsets are defined in (start and end of trajs)
        if (
            np.isin(future_candidates, distance_bin_ids).sum() != distance_offset
            or np.isin(past_candidates, distance_bin_ids).sum() != distance_offset
        ):
            full_offset_defined[i] = False
            continue
        # check if offsets appear in training data
        future_mask = np.isin(decoder_classes, future_candidates)
        past_mask = np.isin(decoder_classes, past_candidates)  # if bin not in train will be False
        all_mask = np.isin(decoder_classes, all_candidates)
        if future_mask.sum() != distance_offset or past_mask.sum() != distance_offset:
            offsets_in_train[i] = False
            continue
        # calculate decoding error distance over the trajectory (+ve = more past, -ve = more future)
        future_probs = probs[future_mask]
        past_probs = probs[past_mask]
        all_probs = probs[all_mask]
        if norm_offset_probs:
            future_probs = future_probs / (all_probs.sum() + 1e-10)
            past_probs = past_probs / (all_probs.sum() + +1e-10)
        weighted_err = np.nansum(past_probs * offset_distances) - np.nansum(future_probs * offset_distances)
        weighted_errs[i] = weighted_err
    if return_as == "error":
        return weighted_errs
    else:
        return weighted_errs, test_in_train, offsets_in_train, full_offset_defined


def place_pred_distance_error(
    Yprob,
    Y_test,
    test_df,
    decoder_classes,
    all_pairs_path_length,
    return_as="error",
    norm_offset_probs=True,
):
    """
    This is complicated, should write doc-string
    """

    # extract rel info
    traj_envelope = test_df[["past", "future"]]  # past & future parts of trajectory (offset)

    # init outputs
    samples = Yprob.shape[0]
    weighted_err = np.zeros(samples)
    full_offset_defined = np.ones(samples, dtype=bool)
    test_in_train = np.ones(samples, dtype=bool)
    offsets_in_train = np.ones(samples, dtype=bool)
    for i in range(samples):
        y = Y_test[i]  # place
        if y not in decoder_classes:
            full_offset_defined[i] = False
            weighted_err[i] = np.nan
            test_in_train[i] = False
            offsets_in_train[i] = False
            continue
        probs = Yprob[i]
        loc2prob = dict(zip(decoder_classes, probs))
        traj = traj_envelope.iloc[i]
        if traj.isnull().any():
            full_offset_defined[i] = False
        past_locs = traj.loc["past"].iloc[1:].dropna().unique()
        future_locs = traj.loc["future"].iloc[1:].dropna().unique()
        all_locs = traj.dropna().unique()
        # catch instances where past or future offsets are not in training data for place decoder
        if np.isin(past_locs, decoder_classes).sum() != len(past_locs) or np.isin(
            future_locs, decoder_classes
        ).sum() != len(future_locs):
            offsets_in_train[i] = False
        # but still calc weighted error over what we have (filter later)
        past_probs = np.array([loc2prob[loc] if loc in loc2prob.keys() else np.nan for loc in past_locs])
        future_probs = np.array([loc2prob[loc] if loc in loc2prob.keys() else np.nan for loc in future_locs])
        all_probs = np.array([loc2prob[loc] if loc in loc2prob.keys() else np.nan for loc in all_locs])
        if norm_offset_probs:
            past_probs = past_probs / np.nansum(all_probs + 1e-10)
            future_probs = future_probs / np.nansum(all_probs + 1e-10)
        past_dists = np.array([all_pairs_path_length[y][loc] for loc in past_locs])
        future_dists = np.array([all_pairs_path_length[y][loc] for loc in future_locs])
        # calculate decoding error distance over the trajectory (+ve = more past, -ve = more future)
        weighted_err[i] = np.nansum(past_probs * past_dists) - np.nansum(future_probs * future_dists)
    if return_as == "error":
        return weighted_err
    else:
        return (weighted_err, test_in_train, offsets_in_train, full_offset_defined)


# %% get input data :)


def get_input_data(
    session,
    theta_split=False,
    resolution=0.1,
    sum_spike_window=0.4,
    metric=("distance_to_goal", "future"),
    include_multiunits=True,
    moving_only=True,
    navigation_only=True,
    remove_time_at_goal=True,
    max_steps_to_goal=20,
    distance_bins=15,
    bin_method="uniform",
    place_offset=2,
    all_offset_defined=True,
    permute=False,
):
    """ """
    # load data
    navigation_df = session.navigation_df.copy()
    if theta_split:
        spike_counts_df = session.navigation_theta_spike_counts_df  # [frames, clusters * 12 lfp phase bins]
        spike_counts_df.reset_index(inplace=True, drop=True)
    else:
        spike_counts_df = session.navigation_spike_counts_df  # [frames, clusters]
        spike_counts_df.reset_index(inplace=True, drop=True)
        spike_counts_df.columns = pd.MultiIndex.from_tuples([(*c, "") for c in spike_counts_df.columns])
    if permute:
        # circular shift of spikes in time to break rel between neurons and behaviour
        _n = len(spike_counts_df)
        spike_counts_df = spike_counts_df.iloc[np.roll(np.arange(_n), np.random.randint(_n))]

    # filter clusters
    keep_clusters = gc.filter_clusters(
        session.cluster_metrics,
        session.session_info,
        return_unique_IDs=True,
        single_units=True,
        multi_units=include_multiunits,
    )
    spike_counts_df = spike_counts_df[
        spike_counts_df.columns[spike_counts_df.columns.get_level_values(1).isin(keep_clusters)]
    ]

    if sum_spike_window is None or sum_spike_window == resolution:
        # sum spikes and downsample behaviour to same resolution
        ds_nav_df, ds_spike_counts_df = ds.downsample_nav_spikes_data(
            navigation_df,
            spike_counts_df,
            resolution=resolution,
            distance_metrics=[("steps_to_goal", "future"), metric],
        )
    else:
        # sum spikes over spike_window (smooth)
        sum_frames = int(sum_spike_window * FRAME_RATE)
        spike_counts_df = spike_counts_df.rolling(window=sum_frames, center=True).sum().fillna(0).astype(int)
        # downsample (usually higher rate than sum_spikes)
        every_n_frames = int(resolution * FRAME_RATE)
        ds_spike_counts_df = spike_counts_df.iloc[::every_n_frames].reset_index(drop=True)
        ds_nav_df = navigation_df.iloc[::every_n_frames].reset_index(drop=True)
    ds_nav_df.columns = pd.MultiIndex.from_tuples([(*c, "") for c in ds_nav_df.columns])
    input_df = pd.concat([ds_nav_df, ds_spike_counts_df], axis=1).reset_index(drop=True)

    # add future, past state (place) information
    input_df[("place_direction", "", "")] = input_df.maze_position.simple + "_" + input_df.cardinal_movement_direction
    offset_df = fd.get_past_and_future_states(
        input_df, state_type="place", past_offset=place_offset, future_offset=place_offset
    )
    offset_df.columns = pd.MultiIndex.from_tuples([(*col, "") for col in offset_df.columns])
    input_df = pd.concat([input_df, offset_df], axis=1)
    input_df = input_df.sort_index(axis=1)  # sort columns for easier indexing later

    # ensure data is balanced for past and future on traj is defined
    if all_offset_defined:
        input_df = input_df[input_df[["past", "future"]].notnull().all(axis=1)]

    # filter data
    input_df = filt.filter_navigation_rates_df(
        input_df,
        navigation_only=navigation_only,
        moving_only=moving_only,
        exclude_time_at_goal=remove_time_at_goal,
        max_steps_to_goal=max_steps_to_goal,
    )

    # get binned distance to goal
    bins = convert._get_distance_bins(
        binning_method=bin_method,
        n_distance_bins=distance_bins,
        distance_metrics=metric,
        max_distance=input_df[metric].max() + 0.001,
        min_distance=input_df[metric].min() - 0.001,
    )
    input_df.loc[:, ("distance_bin", "", "")] = pd.cut(input_df[metric], bins=bins, include_lowest=True).to_numpy()
    input_df.loc[:, ("distance_bin_mid", "", "")] = input_df.distance_bin.apply(lambda x: x.mid)
    input_df.loc[:, ("distance_bin_id", "", "")] = input_df.distance_bin.map({b: i for i, b in enumerate(bins)})

    # add other info
    input_df[("subject_ID", "", "")] = session.subject_ID
    input_df[("maze_name", "", "")] = session.maze_name
    input_df[("day_on_maze", "", "")] = session.day_on_maze
    # done!
    if theta_split:
        return input_df
    else:
        return input_df.droplevel(2, axis=1)


# %% misc


def _get_all_pairs_path_length(session):
    skeleton_maze = session.skeleton_maze()
    dists = dict(nx.all_pairs_dijkstra_path_length(skeleton_maze, weight="weight"))
    coord2label = mr.get_maze_coord2label(skeleton_maze)
    _dists = {}
    for src in dists.keys():
        if src[-1] == 0:  # center of each tower/bridge
            src_dists = dists[src]
            __dists = {}
            for targ in src_dists.keys():
                if targ[-1] == 0:
                    __dists[coord2label[targ].split("_")[0]] = src_dists[targ]
                _dists[coord2label[src].split("_")[0]] = __dists
    return _dists
