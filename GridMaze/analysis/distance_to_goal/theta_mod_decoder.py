"""
Library for distance to goal rep, theta mod decoding.
"""

# %% Imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


from GridMaze.analysis.core import folds
from GridMaze.analysis.core import convert
from GridMaze.analysis.core import downsample as ds
from GridMaze.analysis.distance_to_goal import distributions as dd
from GridMaze.analysis.distance_to_goal import logreg_decoder as ld

# %% Global Variables


# %%


def populate_decoding_results():
    """ """
    return


def get_theta_mod_distance_to_goal_decoding(
    session,
    resolution=0.5,
    metric=("distance_to_goal", "geodesic"),
    include_multiunits=True,
    moving_only=True,
    max_steps_to_goal=30,
    bin_spacing=0.04,
    max_distance=0.8,  # best decoding at short distances
    bin_method="uniform",
    n_log_bins=30,
    balance_distances=False,
    n_folds=8,
    sqrt_spikes=True,
    standardise_spikes=True,
    alpha="opt",
    output="max",
    verbose=True,
):
    """ """
    # get input data
    if verbose:
        print("loading input data ...")
    input_data = get_input_data(
        session,
        resolution=resolution,
        metric=metric,
        include_multiunits=include_multiunits,
        moving_only=moving_only,
        max_steps_to_goal=max_steps_to_goal,
        bin_spacing=bin_spacing,
        bin_method=bin_method,
        max_distance=max_distance,
        n_log_bins=n_log_bins,
        balance_distances=balance_distances,
    )
    distance_bin_mids = np.array(sorted(input_data.distance_bin_mid.unique()))
    lfp_phases = input_data.spike_count.columns.get_level_values(1).unique().values

    # set up results df
    results_df = pd.concat(
        [
            input_data[[("trial", "", ""), ("time", "", ""), (*metric, ""), ("distance_bin_mid", "", "")]].droplevel(
                2, axis=1
            ),
            pd.DataFrame(index=input_data.index, columns=pd.MultiIndex.from_product([["lfp_phase"], lfp_phases])),
        ],
        axis=1,
    )  # sample info + err for each sample at each lfp phase

    # decode distance CV
    folds_df = folds.get_folds_df(session, goal_stratified=False, return_unique_IDs=True, n_folds=n_folds)
    _folds = folds_df.columns.get_level_values(0).unique()
    for fold in _folds:
        if verbose:
            print(fold)
        fold_df = folds_df[fold]
        if alpha == "opt":
            # get CV opt reg strength from just training data
            opt_alpha = get_CV_alpha(
                input_data,
                fold_df,
                metric,
                output,
                sqrt_spikes=sqrt_spikes,
                standardise_spikes=standardise_spikes,
                return_as="best",
                verbose=verbose,
            )
            if verbose:
                print(f"Optimal alpha for {fold} is {opt_alpha}")
        else:
            opt_alpha = alpha

        # split test train
        train_df, test_df = folds._get_test_train_dfs(input_data, fold_df)
        # train on spikes from all theta bins
        X_train = train_df.spike_count.T.groupby(level=0).mean().T.values  # mean spikecount across lfp phases
        if sqrt_spikes:
            X_train = np.sqrt(X_train)
        if standardise_spikes:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
        y_train = train_df.distance_bin_id.values
        # train model on average spike counts across all lfp phases
        if verbose:
            print(f"    Training on mean spike counts across all lfp phases")
        model = LogisticRegression(penalty="l2", C=opt_alpha, max_iter=10_000, random_state=0, class_weight="balanced")
        model.fit(X_train, y_train)

        # test on each lfp_phase separately
        test_spikes_df = test_df.spike_count.swaplevel(axis=1).sort_index(axis=1)
        for i, phase in enumerate(lfp_phases):
            if verbose:
                print(f"    Testing on phase: {phase:.2f}")
            X_test = test_spikes_df[phase].values
            if sqrt_spikes:
                X_test = np.sqrt(X_test)
            if standardise_spikes:
                X_test = scaler.transform(X_test)
            y_test = test_df.distance_bin_id.values
            true_dist = distance_bin_mids[y_test]
            # get output metric
            if output == "weighted":
                y_pred_prob = model.predict_proba(X_test)
                weighted_dist = y_pred_prob.dot(distance_bin_mids)
                err = weighted_dist - true_dist
            elif output == "max":
                y_pred = model.predict(X_test)
                y_pred_dist = distance_bin_mids[y_pred]
                err = y_pred_dist - true_dist
            else:
                NotImplementedError(f"Test metric {output} not implemented.")
            # store results
            results_df.loc[test_df.index, ("lfp_phase", phase)] = err
    return results_df


def get_CV_alpha(
    input_data,
    fold_df,
    metric,
    output,
    sqrt_spikes=True,
    standardise_spikes=True,
    return_as="best",
    verbose=False,
):
    """ """
    distance_bin_mids = np.array(sorted([b.mid for b in input_data.distance_bin.unique()]))
    # split training data into folds
    val_df = fold_df["train"]
    _vfolds = val_df.columns.values
    val_results = []
    for i, vfold in enumerate(_vfolds):
        if verbose:
            print(f"    Validation fold {i}")
        # index input data for validation test and train
        test_df = input_data[input_data.trial_unique_ID.isin(val_df[vfold].values)]
        train_df = input_data[input_data.trial_unique_ID.isin(val_df.drop(columns=vfold).unstack().dropna().values)]
        # get X and y
        # optimise just mean spikes over lfp phase bins and use for all test decodings later
        X_train, X_test = [df.spike_count.T.groupby(level=0).mean().T for df in [train_df, test_df]]
        if sqrt_spikes:
            X_train, X_test = np.sqrt(X_train), np.sqrt(X_test)
        if standardise_spikes:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        y_train, y_test = train_df.distance_bin_id.values, test_df.distance_bin_id.values
        y_true = test_df[(*metric, "")].values
        # search over regularisation strengths
        best_alpha, best_MSE = ld.search_reg(
            X_train,
            X_test,
            y_train,
            y_test,
            y_true,
            output=output,
            distances=distance_bin_mids,
            verbose=False,
        )
        val_results.append(
            {
                "vfold": vfold,
                "best_alpha": best_alpha,
                "best_MSE": best_MSE,
            }
        )
    reg_df = pd.DataFrame(val_results)
    if return_as == "df":
        return reg_df
    elif return_as == "best":
        # median opt reg strength across folds
        opt_reg = reg_df.best_alpha.median()
        return opt_reg
    else:
        raise ValueError(f"Return as must be 'df' of 'best'. ")


# %% get input data


def get_input_data(
    session,
    resolution=0.4,
    metric=("distance_to_goal", "geodesic"),
    include_multiunits=True,
    moving_only=True,
    max_steps_to_goal=30,
    bin_spacing=0.05,
    bin_method="uniform",
    max_distance=None,
    n_log_bins=25,
    balance_distances=False,
):
    """ """
    # load data
    navigation_df = session.navigation_df.copy()
    spike_counts_df = session.navigation_theta_spike_counts_df.reset_index(
        drop=True
    )  # [frames, clusters * 12 lfp phase bins]
    cluster_metrics = session.cluster_metrics
    session_info = session.session_info

    # filter for single units
    if not include_multiunits:
        single_units = cluster_metrics[cluster_metrics.single_unit].cluster_ID
        single_units = convert.cluster_IDs2scluster_unique_IDs(session_info, single_units)
        spike_counts_df = spike_counts_df[
            spike_counts_df.columns[[c in single_units for c in spike_counts_df.columns.get_level_values(1)]]
        ]

    ds_nav_df, ds_spike_counts_df = ds.downsample_nav_spikes_data(
        navigation_df,
        spike_counts_df,
        resolution=resolution,
        distance_metrics=[("steps_to_goal", "future"), metric],
    )
    # add lvl to nav_df to match spike_counts
    ds_nav_df.columns = pd.MultiIndex.from_tuples([(*c, "") for c in ds_nav_df.columns])
    metric = (*metric, "")
    input_df = pd.concat([ds_nav_df, ds_spike_counts_df], axis=1)
    # filter for valid trial times
    input_df = input_df[input_df.trial_phase == "navigation"]
    # add distance bins
    if moving_only:
        input_df = input_df[input_df.moving]
    if max_steps_to_goal is not None:
        input_df = input_df[input_df.steps_to_goal.future < max_steps_to_goal]
    # remove frames where distance is above max (treat as outliers)
    if metric[0] == "distance_to_goal":
        if max_distance is None:
            max_distance = dd.get_distance_percentile(metric, 0.85)
        if bin_method == "uniform":
            n_bins = int(max_distance / bin_spacing)
        elif bin_method == "log":
            n_bins = n_log_bins
        input_df = input_df[input_df[metric] < max_distance]
        bins = convert._get_distance_bins(
            binning_method=bin_method,
            n_distance_bins=n_bins,
            distance_metrics=metric,
            max_distance=max_distance,
        )
    else:
        NotImplementedError()
    # bin distances
    input_df.loc[:, ("distance_bin", "", "")] = pd.cut(input_df[metric], bins=bins, include_lowest=True).to_numpy()
    input_df.loc[:, ("distance_bin_mid", "", "")] = input_df.distance_bin.apply(lambda x: x.mid)
    input_df.loc[:, ("distance_bin_id", "", "")] = input_df.distance_bin.map({b: i for i, b in enumerate(bins)})
    if not balance_distances:
        return input_df
    else:  # balance data across distance bins
        max_size = input_df.groupby("distance_bin_id").size().max()
        balanced_data = (
            input_df.groupby("distance_bin_id", group_keys=False)
            .sample(n=max_size, replace=True, random_state=42)
            .reset_index(drop=True)
        )
        return balanced_data


# %%
