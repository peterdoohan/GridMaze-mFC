"""
Utility functions for models.py
"""

# %% Imports
import numpy as np
from sklearn import linear_model
from joblib import delayed, Parallel

# %% Functions


def eval_function(x_train, y_train, x_test, y_test, alpha=1e-3, loss="poisson"):
    """
    loss="poisson"  → PoissonRegressor (exp link, requires y >= 0); .score returns D².
    loss="gaussian" → Ridge (identity link, accepts real-valued y); .score returns R².
    Use "gaussian" when y is not non-negative integer counts (e.g. rotationally permuted
    spikes) — Poisson rejects negative y.
    """
    if loss == "poisson":
        clf = linear_model.PoissonRegressor(alpha=alpha, max_iter=10_000)
    elif loss == "gaussian":
        clf = linear_model.Ridge(alpha=alpha)
    else:
        raise ValueError(f"Unknown loss '{loss}'; expected 'poisson' or 'gaussian'.")
    clf.fit(x_train, y_train)
    return clf.score(x_test, y_test)  # evaluate the goodness of the fit


def find_optimal_regularization_strength(
    x, y, trials, alphas=10.0 ** np.arange(2, -5, -1), n_folds=5, loss="poisson", split_seed=None
):

    inds = split_trials(trials, n_folds, seed=split_seed)

    perfs = np.zeros((n_folds, len(alphas)))
    for fold in range(n_folds):
        test, train = inds[fold], np.concatenate([inds[f] for f in range(n_folds) if f != fold])
        # split the data into train and test for this fold
        x_train, y_train, x_test, y_test = x[train], y[train], x[test], y[test]
        for ialpha, alpha in enumerate(alphas):

            if loss == "poisson":
                if ialpha == 0:  # instantiate to begin with
                    clf = linear_model.PoissonRegressor(
                        alpha=alpha, warm_start=True, max_iter=10_000
                    )  # then continue from warm start
                else:
                    clf.alpha = alpha  # just overwrite alpha
            elif loss == "gaussian":
                clf = linear_model.Ridge(alpha=alpha)  # Ridge has no warm_start; instantiate fresh per alpha
            else:
                raise ValueError(f"Unknown loss '{loss}'; expected 'poisson' or 'gaussian'.")

            clf.fit(x_train, y_train)  # train on part of the data
            perfs[fold, ialpha] = clf.score(x_test, y_test)  # test on the rest

    mean_perfs = perfs.mean(0)  # mean across folds for each alpha
    best_alpha = alphas[np.argmax(mean_perfs)]  # best regularization strength

    return best_alpha


def split_trials(trials, n_folds, seed=None):
    unique_trials = np.unique(trials)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_trials)

    trial_splits = np.array_split(unique_trials, n_folds)

    inds = [np.concatenate([np.where(trials == trial_id)[0] for trial_id in split]) for split in trial_splits]
    return inds


def eval_representation(
    x,
    y,
    trials=None,
    n_folds=None,
    optimal_alpha=False,
    optimal_alpha_range=10.0 ** np.arange(2, -5, -1),
    optimal_alpha_n_folds=5,
    alpha=1e-3,
    loss="poisson",
    split_seed=None,
    n_jobs=None,
    verbose=False,
):
    """
    for each neuron in the test data
        loss="poisson" : y ~ Poisson( lambda = exp(W z) )
        loss="gaussian": y ~ Normal( mean = W z, ... ); use when y is not non-negative
                         counts (e.g. rotationally permuted spikes).

    function for evaluating the utility of the learned embedding on some dataset
    x: input data to be embedded. Shape:
    y: output data to regress embedding onto. Shape: (number of neurons, number of time points)
    trials: trial index for each time point
    """
    N, T = y.shape
    if n_folds is not None:
        assert trials is not None
        trials = np.asarray(trials)
        inds = split_trials(trials, n_folds, seed=split_seed)
        if loss == "poisson":
            # require spikes in all splits — only meaningful for non-negative counts
            enough_spikes = np.array([(np.amin([y[n, :][ind].sum() for ind in inds]) > 0) for n in range(N)])
        else:
            enough_spikes = np.ones(N, dtype=bool)  # gaussian: no spike-positivity gate
    else:
        inds = None  # no cv
        enough_spikes = np.ones(N, dtype=bool)  # no cross-validation, so no need to check for spikes in each fold

    # optionally run eval in parallel over neurons
    if n_jobs is not None:
        if verbose:
            print(f"Evaluating {N} neurons in parallel with {n_jobs} jobs")
        scores = Parallel(n_jobs=n_jobs)(
            delayed(_eval_neuron)(
                n, x, y, enough_spikes, trials, n_folds, inds, alpha, optimal_alpha, optimal_alpha_range, optimal_alpha_n_folds, loss, split_seed
            )
            for n in range(N)
        )
    # otherwise run eval sequentially
    else:
        if verbose:
            print(f"Evaluating {N} neurons")
        scores = [
            _eval_neuron(
                n, x, y, enough_spikes, trials, n_folds, inds, alpha, optimal_alpha, optimal_alpha_range, optimal_alpha_n_folds, loss, split_seed
            )
            for n in range(N)
        ]
    scores = np.array(scores)
    return scores


def _eval_neuron(
    n, x, y, enough_spikes, trials, n_folds, inds, alpha, optimal_alpha, optimal_alpha_range, optimal_alpha_n_folds=5, loss="poisson", split_seed=None
):
    """ """
    y_n = y[n, :]  # target spike counts
    if not enough_spikes[n]:
        return np.nan if n_folds is None else np.zeros(n_folds) + np.nan

    # fit a regression model from the embeddings (Poisson or Gaussian, see `loss`)
    if n_folds is None:  # no crossvalidation; just test representation on the whole thing
        return eval_function(x, y_n, x, y_n, alpha=alpha, loss=loss)
    else:
        neuron_scores = np.zeros(n_folds)
        for fold in range(n_folds):
            test, train = inds[fold], np.concatenate([inds[f] for f in range(n_folds) if f != fold])
            x_train, y_n_train, trials_train = x[..., train, :], y_n[train], trials[train]
            x_test, y_n_test = x[..., test, :], y_n[test]

            if optimal_alpha:
                # first find the optimal regularization strength through crossvalidation on the training data
                alpha = find_optimal_regularization_strength(
                    x_train,
                    y_n_train,
                    trials_train,
                    alphas=optimal_alpha_range,
                    n_folds=optimal_alpha_n_folds,
                    loss=loss,
                    split_seed=split_seed,
                )

            # then fit a model to the full training data with that regularization strength
            neuron_scores[fold] = eval_function(x_train, y_n_train, x_test, y_n_test, alpha=alpha, loss=loss)

            assert not np.isnan(neuron_scores[fold])

        return neuron_scores
