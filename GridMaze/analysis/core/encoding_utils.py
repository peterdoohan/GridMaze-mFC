"""
Core code for encoding analysis, eg LinearRegression or Poission GLMs.
"""

# %% Imports
import numpy as np
from sklearn.linear_model import Ridge, PoissonRegressor

# %% Global Variables


# %% Abstracted functions


# %% existing functions from goal_coding/cpd.py


def reg_search_regression(
    X_train,
    y_train,
    X_test,
    y_test,
    model="Ridge",
    reg_range=np.logspace(-4, 4, 20),
    tol=1e-4,
    patience=5,
    return_as="best",
    verbose=False,
):
    """
    Runs OLS (Ridge) or Poisson regression (PoissonRegressor) with increasing alpha
    until the score stops improving.
    Returns the best alpha and score.
    If return_as="history", returns a array of (alphas, scores).
    Scores are R² for OLS and Pseudo R2 for Poisson regression.
    """
    best_alpha = reg_range[0]
    best_score = -np.inf
    history = []
    no_improve_count = 0
    for alpha in reg_range:
        if model == "Ridge":
            Model = Ridge(alpha=alpha, max_iter=10_000, random_state=0)
        elif model == "PoissonRegressor":
            Model = PoissonRegressor(alpha=alpha, max_iter=10_000)
        else:
            raise ValueError(f"Unknown model: {model}")
        Model.fit(X_train, y_train)
        score = Model.score(X_test, y_test)
        history.append((alpha, score))
        if verbose:
            print(f" α = {alpha:.3e},  R² = {score:.4f}")
        # update best if we improved by more than tol
        if score > best_score + tol:
            best_score = score
            best_alpha = alpha
            no_improve_count = 0
        else:
            # only count towards patience if best_score is non-negative
            if best_score >= 0:
                no_improve_count += 1
                if no_improve_count >= patience:
                    break
    if verbose:
        print(f"→ Best α = {best_alpha:.3e} with R² = {best_score:.4f}")
        print("")

    if return_as == "history":
        return np.array(history).T
    elif return_as == "best":
        return best_alpha, best_score
    else:
        raise ValueError(f"Unknown return_as: {return_as}")
