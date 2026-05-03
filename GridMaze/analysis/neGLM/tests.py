"""
Smoke tests + diagnostic comparisons for the neGLM pipeline.

These tests train a single embedding (one fold of LOO) and surface comparisons that
matter when reasoning about the Ridge-baseline-vs-rotation-null analysis.
@peterdoohan @krisjensen
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from scipy.stats import spearmanr

from GridMaze.analysis.neGLM.get_input_data import get_input_data
from GridMaze.analysis.neGLM.run_neGLM import (
    RESULTS_DIR,
    _run_single_fold,
    _save_outputs,
)
from jobs.neGLM.utils import (
    DEFAULT_INPUT_DATA_KWARGS,
    DEFAULT_MODEL_INIT_KWARGS,
    DEFAULT_MODEL_TRAIN_KWARGS,
    DEFAULT_SCORE_KWARGS,
)

# %% functions


def test_neGLM(
    input_data_kwargs=DEFAULT_INPUT_DATA_KWARGS,
    model_init_kwargs=DEFAULT_MODEL_INIT_KWARGS,
    model_train_kwargs=DEFAULT_MODEL_TRAIN_KWARGS,
    score_kwargs=DEFAULT_SCORE_KWARGS,
    test_session_idx=0,
    n_permutations=50,
    seed=0,
    verbose=True,
    save=False,
):
    """End-to-end pipeline sanity check: train one neGLM embedding on all-but-one
    session and score on session `test_session_idx`, with `n_permutations` rotational
    permutations. Returns
    (learning_curve_df, cv_score_df, ridge_cv_score_df, perm_cv_score_df).

    save=True persists outputs to RESULTS_DIR/tests/<YYYYMMDD_HHMMSS>/."""
    model_params = {**locals(), "fn": "test_neGLM"}

    save_path = None
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = RESULTS_DIR / "tests" / timestamp

    np.random.seed(seed)
    torch.manual_seed(seed)

    if verbose:
        print("Loading input data ...")
    input_data = get_input_data(**input_data_kwargs)

    test_session = input_data[test_session_idx]
    train_sessions = input_data[:test_session_idx] + input_data[test_session_idx + 1 :]

    if verbose:
        print(
            f"Smoke-test neGLM: train on {len(train_sessions)} sessions, "
            f"test on session {test_session_idx} with {n_permutations} permutations ..."
        )

    learning_curve_df, cv_score_df, ridge_cv_score_df, perm_cv_score_df = _run_single_fold(
        train_sessions=train_sessions,
        test_session=test_session,
        model_init_kwargs=model_init_kwargs,
        model_train_kwargs=model_train_kwargs,
        score_kwargs=score_kwargs,
        n_permutations=n_permutations,
        fold_idx=test_session_idx,  # also seeds the rotation null per (test_session_idx, perm_idx)
        verbose=verbose,
    )

    if save_path is not None:
        _save_outputs(
            save_path,
            model_params=model_params,
            training_df=learning_curve_df,
            cv_scores_df=cv_score_df,
            ridge_cv_scores_df=ridge_cv_score_df,
            perm_cv_scores_df=perm_cv_score_df,
            verbose=verbose,
        )

    return learning_curve_df, cv_score_df, ridge_cv_score_df, perm_cv_score_df


def test_higher_regularisation(
    input_data_kwargs=DEFAULT_INPUT_DATA_KWARGS,
    model_init_kwargs=DEFAULT_MODEL_INIT_KWARGS,
    model_train_kwargs=DEFAULT_MODEL_TRAIN_KWARGS,
    score_kwargs=DEFAULT_SCORE_KWARGS,
    test_session_idx=0,
    n_permutations=50,
    seed=0,
    verbose=True,
    save=False,
):
    """Thin wrapper around test_neGLM that forces near-zero regularisation in the
    per-neuron CV scoring (alpha=1e-12, optimal_alpha=False), to isolate whether the
    rotated-Ridge > real-Ridge gap is driven by the L2 penalty or by per-neuron
    variance equalisation under rotation.

    Note: only removes test-time scoring regularisation. The embedding-training L2
    penalty (model_init_kwargs["beta_weight"]) is left untouched — `z` is still shaped
    the same way as in test_neGLM.

    Returns (learning_curve_df, cv_score_df, ridge_cv_score_df, perm_cv_score_df)."""
    no_reg_score_kwargs = deepcopy(score_kwargs)
    no_reg_score_kwargs["alpha"] = 10
    no_reg_score_kwargs["optimal_alpha"] = False
    return test_neGLM(
        input_data_kwargs=input_data_kwargs,
        model_init_kwargs=model_init_kwargs,
        model_train_kwargs=model_train_kwargs,
        score_kwargs=no_reg_score_kwargs,
        test_session_idx=test_session_idx,
        n_permutations=n_permutations,
        seed=seed,
        verbose=verbose,
        save=save,
    )


def plot_test_neGLM_results(results_path):
    """Load cv_scores.csv, ridge_cv_scores.csv, perm_cv_scores.csv from `results_path`
    and plot per-neuron mean ± SEM across neurons as a single seaborn pointplot with
    three categories: Poisson-real, Ridge-real, Ridge-permuted.

    Per-neuron aggregation: mean across `fold` for the real-y CSVs; mean across `fold`
    AND `permutation` for the perm CSV. Mean ± SEM are then computed across neurons.
    Returns the matplotlib figure."""
    results_path = Path(results_path)

    cv_df = pd.read_csv(results_path / "cv_scores.csv")
    ridge_df = pd.read_csv(results_path / "ridge_cv_scores.csv")
    perm_df = pd.read_csv(results_path / "perm_cv_scores.csv")

    # collapse to one score per neuron
    poisson_real = cv_df.groupby("cluster_unique_ID", as_index=False)["cv_score"].mean()
    poisson_real["method"] = "Poisson-real"

    ridge_real = ridge_df.groupby("cluster_unique_ID", as_index=False)["cv_score"].mean()
    ridge_real["method"] = "Ridge-real"

    ridge_perm = perm_df.groupby("cluster_unique_ID", as_index=False)["cv_score"].mean()
    ridge_perm["method"] = "Ridge-permuted"

    long_df = pd.concat([poisson_real, ridge_real, ridge_perm], axis=0, ignore_index=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    sns.pointplot(
        data=long_df,
        x="method",
        y="cv_score",
        order=["Poisson-real", "Ridge-real", "Ridge-permuted"],
        errorbar="se",
        linestyle="none",
        ax=ax,
    )
    ax.set_ylabel("score (D² for Poisson, R² for Ridge)")
    ax.set_xlabel("")
    ax.set_title(f"cv performance")


def test_unique_variance(
    input_data_kwargs=DEFAULT_INPUT_DATA_KWARGS,
    model_init_kwargs=DEFAULT_MODEL_INIT_KWARGS,
    model_train_kwargs=DEFAULT_MODEL_TRAIN_KWARGS,
    score_kwargs=DEFAULT_SCORE_KWARGS,
    test_session_idx=0,
    n_permutations=50,
    seed=0,
    verbose=True,
    save=False,
):
    """Train one full neGLM and two reduced neGLMs (one variable each) on the same
    train/test split, score each with Poisson, Ridge, and Ridge-on-permuted, then
    compute unique variance explained per neuron per variable per method.

    Returns a long-form DataFrame with columns
    [subject_ID, maze_name, day_on_maze, cluster_unique_ID, variable, method,
    unique_score], where method ∈ {Poisson-real, Ridge-real, Ridge-permuted} and
    variable iterates over `input_data_kwargs["input_groups"]`. Unique variance for
    variable A is computed as `score(full_model) - score(remove_A)` per neuron — the
    same convention used in jobs/neGLM/variance_explained/submit.py.

    Permutation seeding uses (test_session_idx, perm_idx), so the K rotations are
    byte-identical across the three model variants — the unique-R²-on-permuted
    subtraction is matched.

    save=True persists outputs to RESULTS_DIR/tests/<YYYYMMDD_HHMMSS>/.

    Note: this is meant to run quickly in an interactive session — the production
    pipeline (jobs/neGLM/variance_explained/submit.py) submits SLURM jobs for the full
    dataset with bigger n_folds and lower min_spike_count."""
    model_params = {**locals(), "fn": "test_unique_variance"}

    all_input_groups = list(input_data_kwargs["input_groups"])
    assert n_permutations > 0, "unique variance test requires n_permutations > 0 to compute Ridge-permuted scores"

    save_path = None
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = RESULTS_DIR / "tests" / timestamp

    np.random.seed(seed)
    torch.manual_seed(seed)

    # variant set: full model + one reduced model per input_group, mirroring
    # jobs/neGLM/variance_explained/submit.py's (remove_groups, model_name) pattern
    variants = [([], "full_model")] + [([group], f"remove_{group}") for group in all_input_groups]

    fold_outputs = {}
    for remove_groups, model_name in variants:
        # update default input data kwargs
        variant_input_kwargs = deepcopy(input_data_kwargs)
        variant_input_kwargs["input_groups"] = [g for g in all_input_groups if g not in remove_groups]

        if verbose:
            print(f"\n=== {model_name} (input_groups={variant_input_kwargs['input_groups']}) ===")
            print("Loading input data ...")
        input_data = get_input_data(**variant_input_kwargs)

        test_session = input_data[test_session_idx]
        train_sessions = input_data[:test_session_idx] + input_data[test_session_idx + 1 :]

        _, cv_df, ridge_df, perm_df = _run_single_fold(
            train_sessions=train_sessions,
            test_session=test_session,
            model_init_kwargs=model_init_kwargs,
            model_train_kwargs=model_train_kwargs,
            score_kwargs=score_kwargs,
            n_permutations=n_permutations,
            fold_idx=test_session_idx,  # matched permutation seeds across variants
            verbose=verbose,
        )
        fold_outputs[model_name] = (cv_df, ridge_df, perm_df)

        # optionally persist underlying per-variant scores for debugging
        if save_path is not None:
            _save_outputs(
                save_path / model_name,
                cv_scores_df=cv_df,
                ridge_cv_scores_df=ridge_df,
                perm_cv_scores_df=perm_df,
                write_done=False,
            )

    unique_var_df = _compute_unique_variance(fold_outputs, all_input_groups)

    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        params_to_save = {**model_params, "save_path": str(save_path)}
        with open(save_path / "model_params.json", "w") as f:
            json.dump(params_to_save, f, indent=4)
        unique_var_df.to_csv(save_path / "unique_variance.csv", index=False)
        with open(save_path / "DONE.txt", "w") as f:
            f.write("DONE")

    return unique_var_df


def _compute_unique_variance(fold_outputs, all_input_groups):
    """Subtract reduced-model per-neuron scores from full-model per-neuron scores to
    get unique variance explained per (variable, method, neuron). For the perm case,
    per-neuron scores are averaged across both `fold` and `permutation`.

    Expects fold_outputs keys to follow submit.py's convention:
        "full_model", "remove_<input_group>" for each input_group."""
    NEURON_KEYS = ["subject_ID", "maze_name", "day_on_maze", "cluster_unique_ID"]

    cv_full, ridge_full, perm_full = fold_outputs["full_model"]

    def _per_neuron_mean(df):
        return df.groupby(NEURON_KEYS, as_index=False)["cv_score"].mean()

    def _unique(full_df, reduced_df, variable, method):
        merged = _per_neuron_mean(full_df).merge(
            _per_neuron_mean(reduced_df), on=NEURON_KEYS, suffixes=("_full", "_reduced")
        )
        merged["unique_score"] = merged["cv_score_full"] - merged["cv_score_reduced"]
        merged["variable"] = variable
        merged["method"] = method
        return merged[NEURON_KEYS + ["variable", "method", "unique_score"]]

    rows = []
    for variable in all_input_groups:
        cv_red, ridge_red, perm_red = fold_outputs[f"remove_{variable}"]
        rows.append(_unique(cv_full, cv_red, variable, "Poisson-real"))
        rows.append(_unique(ridge_full, ridge_red, variable, "Ridge-real"))
        rows.append(_unique(perm_full, perm_red, variable, "Ridge-permuted"))
    return pd.concat(rows, axis=0, ignore_index=True)


def plot_test_unique_variance_results(results_path):
    """Visualise the output of `test_unique_variance`. Produces a 1×4 figure:

    Panels 1–3: scatter of per-cell unique variance (%) for variable_a vs variable_b,
    one panel each for Poisson-real, Ridge-real, and the FIRST Ridge-permuted draw.
    Each panel is annotated with the across-cells Spearman ρ.

    Panel 4: histogram of Spearman ρ across the K Ridge-permuted draws (the null
    distribution of the negative-correlation factorisation signature). The Poisson-real
    and Ridge-real ρ from panels 1–2 are overlaid as vertical lines for reference.

    Reads `unique_variance.csv` (for Poisson-real / Ridge-real) and the per-variant
    `{full_model, remove_<v>}/perm_cv_scores.csv` files (for per-permutation Ridge)."""
    results_path = Path(results_path)
    NEURON_KEYS = ["subject_ID", "maze_name", "day_on_maze", "cluster_unique_ID"]

    unique_var = pd.read_csv(results_path / "unique_variance.csv")
    variables = list(unique_var["variable"].unique())
    assert len(variables) == 2, f"plot expects exactly 2 variables, got {variables}"
    var_a, var_b = variables

    # per-permutation unique variance, recomputed from the per-variant perm_cv_scores csvs
    perm_full = pd.read_csv(results_path / "full_model" / "perm_cv_scores.csv")
    perm_red = {v: pd.read_csv(results_path / f"remove_{v}" / "perm_cv_scores.csv") for v in variables}

    def _per_neuron_per_perm_mean(df):
        return df.groupby(NEURON_KEYS + ["permutation"], as_index=False)["cv_score"].mean()

    pf = _per_neuron_per_perm_mean(perm_full)

    def _perm_unique(reduced_df, variable):
        merged = pf.merge(
            _per_neuron_per_perm_mean(reduced_df),
            on=NEURON_KEYS + ["permutation"],
            suffixes=("_full", "_reduced"),
        )
        merged["unique_score"] = merged["cv_score_full"] - merged["cv_score_reduced"]
        merged["variable"] = variable
        return merged[NEURON_KEYS + ["permutation", "variable", "unique_score"]]

    perm_unique = {v: _perm_unique(perm_red[v], v) for v in variables}

    # Spearman ρ per permutation across cells
    perm_corrs = []
    for k in sorted(perm_unique[var_a]["permutation"].unique()):
        a_k = perm_unique[var_a].query("permutation == @k").rename(columns={"unique_score": "ua"})
        b_k = perm_unique[var_b].query("permutation == @k").rename(columns={"unique_score": "ub"})
        merged_k = a_k.merge(b_k, on=NEURON_KEYS)
        rho_k, _ = spearmanr(merged_k["ua"], merged_k["ub"], nan_policy="omit")
        perm_corrs.append(rho_k)
    perm_corrs = np.array(perm_corrs)

    # wide-form per-cell tables for the scatter panels
    def _to_wide(df):
        return df.pivot_table(index=NEURON_KEYS, columns="variable", values="unique_score").reset_index()

    poisson_wide = _to_wide(unique_var[unique_var["method"] == "Poisson-real"])
    ridge_wide = _to_wide(unique_var[unique_var["method"] == "Ridge-real"])
    perm_first_long = pd.concat(
        [perm_unique[v].query("permutation == 0") for v in variables], axis=0, ignore_index=True
    )
    perm_first_wide = _to_wide(perm_first_long[NEURON_KEYS + ["variable", "unique_score"]])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    panels = [
        (poisson_wide, "Poisson (real)"),
        (ridge_wide, "Ridge (real)"),
        (perm_first_wide, "Ridge (permutation 0)"),
    ]
    for ax, (data_wide, title) in zip(axes[:3], panels):
        x = data_wide[var_a].to_numpy() * 100  # → percent
        y = data_wide[var_b].to_numpy() * 100
        rho, _ = spearmanr(x, y, nan_policy="omit")
        ax.scatter(x, y, alpha=0.5, s=12)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_xlabel(f"{var_a} unique var (%)")
        ax.set_ylabel(f"{var_b} unique var (%)")
        ax.set_title(f"{title}\nSpearman ρ = {rho:.3f}")

    # Panel 4: histogram of perm Spearman ρ + true lines
    ax = axes[3]
    ax.hist(perm_corrs, bins=20, color="gray", alpha=0.7)
    poisson_rho, _ = spearmanr(poisson_wide[var_a], poisson_wide[var_b], nan_policy="omit")
    ridge_rho, _ = spearmanr(ridge_wide[var_a], ridge_wide[var_b], nan_policy="omit")
    ax.axvline(poisson_rho, color="C0", linewidth=2, label=f"Poisson ρ = {poisson_rho:.3f}")
    ax.axvline(ridge_rho, color="C1", linewidth=2, label=f"Ridge ρ = {ridge_rho:.3f}")
    ax.set_xlabel(f"Spearman ρ\n(unique {var_a} vs unique {var_b})")
    ax.set_ylabel("count")
    ax.set_title(f"Null dist (n={len(perm_corrs)} perms)")
    ax.legend(fontsize="small")

    fig.tight_layout()
