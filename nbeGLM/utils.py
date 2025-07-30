import numpy as np
from sklearn import linear_model
from joblib import delayed, Parallel


def eval_function(x_train, y_train, x_test, y_test, alpha=1e-3):
    clf = linear_model.PoissonRegressor(alpha=alpha)
    clf.fit(x_train, y_train)
    return clf.score(x_test, y_test)  # evaluate the goodness of the fit


def find_optimal_regularization_strength(x, y, trials, alphas=10.0 ** np.arange(2, -5, -1), n_folds=5):

    Tf = len(y)
    inds = split_trials(trials, n_folds)

    perfs = np.zeros((n_folds, len(alphas)))
    for fold in range(n_folds):
        test, train = inds[fold], np.concatenate([inds[f] for f in range(n_folds) if f != fold])
        # split the data into train and test for this fold
        x_train, y_train, x_test, y_test = x[train], y[train], x[test], y[test]
        for ialpha, alpha in enumerate(alphas):

            if ialpha == 0:  # instantiate to begin with
                clf = linear_model.PoissonRegressor(alpha=alpha, warm_start=True)  # then continue from warm start
            else:
                clf.alpha = alpha  # just overwrite alpha

            clf.fit(x_train, y_train)  # train on part of the data
            perfs[fold, ialpha] = clf.score(x_test, y_test)  # test on the rest

    mean_perfs = perfs.mean(0)  # mean across folds for each alpha
    best_alpha = alphas[np.argmax(mean_perfs)]  # best regularization strength

    return best_alpha


def split_trials(trials, n_folds):
    """randomly assigns trials into cv different folds"""
    unique_trials = np.unique(trials)
    trial_splits = [[] for _ in range(n_folds)]
    for trial in unique_trials:
        trial_splits[int(trial) % n_folds].append(trial)
    inds = [
        np.concatenate([np.where(trials == trial_id)[0] for trial_id in trial_split]) for trial_split in trial_splits
    ]
    return inds


def eval_representation(
    x, y, trials=None, n_folds=None, optimal_alpha=False, optimal_alpha_range=10.0 ** np.arange(2, -5, -1), alpha=1e-3
):
    """
    function for evaluating the utility of the learned embedding on some dataset
    x: input data to be embedded. Shape:
    y: output data to regress embedding onto. Shape: (number of neurons, number of time points)
    trials: trial index for each time point
    """

    N, T = y.shape
    scores = np.zeros(N) + np.nan if n_folds is None else np.zeros((N, n_folds)) + np.nan

    # for each neuron in the test data
    # y ~ Poisson( lambda = exp(W z) )

    if n_folds is None:
        enough_spikes = np.ones(N, dtype=bool)  # no cross-validation, so no need to check for spikes in each fold
    if n_folds is not None:
        assert trials is not None
        inds = split_trials(trials, n_folds)
        # require spikes in all splits
        enough_spikes = np.array([(np.amin([y[n, :][ind].sum() for ind in inds]) > 0) for n in range(N)])

    for n in np.where(enough_spikes)[0]:
        y_n = y[n, :]  # target spike counts
        # fit a Poisson regression model from the embeddings
        if n_folds is None:  # no crossvalidation; just test representation on the whole thing
            scores[n] = eval_function(x, y_n, x, y_n, alpha=alpha)
        else:
            for fold in range(n_folds):
                test, train = inds[fold], np.concatenate([inds[f] for f in range(n_folds) if f != fold])
                x_train, y_n_train, trials_train = x[..., train, :], y_n[train], trials[train]
                x_test, y_n_test = x[..., test, :], y_n[test]

                if optimal_alpha:
                    # first find the optimal regularization strength through crossvalidation on the training data
                    alpha = find_optimal_regularization_strength(
                        x_train, y_n_train, trials_train, alphas=optimal_alpha_range, n_folds=n_folds
                    )

                # then fit a model to the full training data with that regularization strength
                scores[n, fold] = eval_function(x_train, y_n_train, x_test, y_n_test, alpha=alpha)

                assert not np.isnan(scores[n, fold])

    return scores  # (neurons by folds)


# %%


def eval_representation2(
    x,
    y,
    trials=None,
    n_folds=None,
    optimal_alpha=False,
    optimal_alpha_range=10.0 ** np.arange(2, -5, -1),
    alpha=1e-3,
    n_jobs=16,
):
    """
    for each neuron in the test data
    y ~ Poisson( lambda = exp(W z) )

    function for evaluating the utility of the learned embedding on some dataset
    x: input data to be embedded. Shape:
    y: output data to regress embedding onto. Shape: (number of neurons, number of time points)
    trials: trial index for each time point
    """

    N, T = y.shape
    if n_folds is not None:
        assert trials is not None
        inds = split_trials(trials, n_folds)
        # require spikes in all splits
        enough_spikes = np.array([(np.amin([y[n, :][ind].sum() for ind in inds]) > 0) for n in range(N)])
    else:
        inds = None  # no cv
        enough_spikes = np.ones(N, dtype=bool)  # no cross-validation, so no need to check for spikes in each fold

    # optionally run eval in parallel over neurons
    if n_jobs is not None:
        scores = Parallel(n_jobs=n_jobs)(
            delayed(_eval_neuron)(
                n, x, y, enough_spikes, trials, n_folds, inds, alpha, optimal_alpha, optimal_alpha_range
            )
            for n in range(N)
        )
    # otherwise run eval sequentially
    else:
        scores = [
            _eval_neuron(n, x, y, enough_spikes, trials, n_folds, inds, alpha, optimal_alpha, optimal_alpha_range)
            for n in range(N)
        ]
    scores = np.array(scores)
    return scores


def _eval_neuron(n, x, y, enough_spikes, trials, n_folds, inds, alpha, optimal_alpha, optimal_alpha_range):
    """ """
    y_n = y[n, :]  # target spike counts
    if not enough_spikes[n]:
        return np.nan if n_folds is None else np.zeros(n_folds) + np.nan

    # fit a Poisson regression model from the embeddings
    if n_folds is None:  # no crossvalidation; just test representation on the whole thing
        return eval_function(x, y_n, x, y_n, alpha=alpha)
    else:
        neuron_scores = np.zeros(n_folds)
        for fold in range(n_folds):
            test, train = inds[fold], np.concatenate([inds[f] for f in range(n_folds) if f != fold])
            x_train, y_n_train, trials_train = x[..., train, :], y_n[train], trials[train]
            x_test, y_n_test = x[..., test, :], y_n[test]

            if optimal_alpha:
                # first find the optimal regularization strength through crossvalidation on the training data
                alpha = find_optimal_regularization_strength(
                    x_train, y_n_train, trials_train, alphas=optimal_alpha_range, n_folds=n_folds
                )

            # then fit a model to the full training data with that regularization strength
            neuron_scores[fold] = eval_function(x_train, y_n_train, x_test, y_n_test, alpha=alpha)

            assert not np.isnan(neuron_scores[fold])

        return neuron_scores
