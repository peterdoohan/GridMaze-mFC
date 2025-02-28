# play around with regressions

import numpy as np
from sklearn._loss.link import BaseLink, Interval
from sklearn._loss.loss import BaseLoss, HalfPoissonLoss
from sklearn.linear_model._glm.glm import _GeneralizedLinearRegressor
from sklearn._loss._loss import CyHalfPoissonLoss
from scipy.special import xlogy
from sklearn.metrics import mean_poisson_deviance
from pyglmnet import GLM, simulate_glm

def calc_poisson_deviance(yhat, y):
    test_dev = mean_poisson_deviance(y, yhat)
    ref_dev = mean_poisson_deviance(y, np.ones(len(y)) * np.mean(y))
    return 1 - test_dev / ref_dev

class SoftplusLink(BaseLink):
    """The log link function g(x)=log(x)."""

    interval_y_pred = Interval(0, np.inf, False, False)

    def link(self, y_pred, out=None):
        assert np.amax(y_pred) < 700
        return np.log(np.exp(y_pred) - 1, out = out)

    def inverse(self, raw_prediction, out=None):
        return np.logaddexp(0, raw_prediction, out = out)
        #return np.log(1 + np.exp(raw_prediction, out=out), out = out)


class HalfSoftPoissonLoss(BaseLoss):
    """Half Poisson deviance loss with log-link, for regression.

    Domain:
    y_true in non-negative real numbers
    y_pred in positive real numbers

    Link:
    y_pred = exp(raw_prediction)

    For a given sample x_i, half the Poisson deviance is defined as::

        loss(x_i) = y_true_i * log(y_true_i/exp(raw_prediction_i))
                    - y_true_i + exp(raw_prediction_i)

    Half the Poisson deviance is actually the negative log-likelihood up to
    constant terms (not involving raw_prediction) and simplifies the
    computation of the gradients.
    We also skip the constant term `y_true_i * log(y_true_i) - y_true_i`.
    """

    def __init__(self, sample_weight=None):
        super().__init__(closs=CyHalfPoissonLoss(), link=SoftplusLink())
        self.interval_y_true = Interval(0, np.inf, True, False)

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        term = xlogy(y_true, y_true) - y_true
        if sample_weight is not None:
            term *= sample_weight
        return term
    
    
class SoftPoissonRegressor(_GeneralizedLinearRegressor):
    """Generalized Linear Model with a Poisson distribution.

    This regressor uses the 'softplus' inverse link function.

    """

    _parameter_constraints: dict = {
        **_GeneralizedLinearRegressor._parameter_constraints
    }

    def __init__(
        self,
        *,
        alpha=1.0,
        fit_intercept=True,
        solver="lbfgs",
        max_iter=100,
        tol=1e-4,
        warm_start=False,
        verbose=0,
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            verbose=verbose,
        )

    def _get_loss(self):
        return HalfSoftPoissonLoss()
        #return HalfPoissonLoss()
    
    
from sklearn import linear_model
perfs = []
perfs2 = []

for _ in range(200):
    X = np.random.normal(0, 1, (1000,2))
    W = np.random.normal(0, 1, (2,1))

    clf = linear_model.PoissonRegressor()
    #clf = SoftPoissonRegressor()
    
    clf = GLM(distr='poisson', score_metric='pseudo_R2', reg_lambda=0.0)
    clf = GLM(distr='softplus', score_metric='pseudo_R2', reg_lambda=0.0)
    
    yhat = np.exp(X@W)
    yhat = np.logaddexp(0, X@W+1.2)
    yhat = yhat.flatten()
    
    y = yhat
    y = np.random.poisson(yhat)
    
    clf.fit(X[::2], y[::2])
    #clf.coef_ = -clf.coef_
    score = clf.score(X[1::2], y[1::2])
    perfs.append(score)
    
    pred = clf.predict(X[1::2])
    score2 = calc_poisson_deviance(pred, y[1::2])
    perfs2.append(score2)
    
    #print(score)
    #print(clf.coef_, W.flatten())
    #print(clf.intercept_)
    
    
print(np.mean(perfs), np.std(perfs)/np.sqrt(len(perfs)))
print(np.mean(perfs2), np.std(perfs2)/np.sqrt(len(perfs)))



