# `jobs/`

HPC (SLURM) submission scripts for the analyses in this project. Each subfolder corresponds to one analysis pipeline and contains a `submit.py` that:

1. builds a list of model/job parameter dicts (`get_model_set_params(...)`),
2. writes a per-job `.sh` script into `slurm/`,
3. submits each job with `sbatch`.

Stdout / stderr from cluster jobs land in `out/` and `err/` next to `submit.py`. Final result artefacts (model weights, CV scores, decoding outputs, …) are written under `GridMaze.paths.RESULTS_PATH`, not into the `jobs/` tree.


---

# `neGLM/` — neural-embedding GLM analyses

The neGLM (neural embedding GLM) fits a latent-embedding model to spike counts using behavioural feature groups (`place`, `direction`, `place_direction`, `distance_to_goal`, `velocity`, `head_direction`, `egocentric_action`, etc.) as inputs. Each subfolder defines a different *model set* — i.e. a comparison across feature subsets, factorisations, or hyperparameters — and submits one SLURM job per (maze, model) pair.

All subfolders share the same scaffolding:

- **`utils.py`** — central definition of `DEFAULT_INPUT_DATA_KWARGS`, `DEFAULT_MODEL_INIT_KWARGS`, `DEFAULT_MODEL_TRAIN_KWARGS`, `DEFAULT_SCORE_KWARGS`, plus:
  - `get_SLURM_script(...)` — writes a `.sh` that runs `GridMaze.analysis.neGLM.run_neGLM.<run_fn>(**model_params)` (default `run_cv_neGLM`; `run_cv_baselineGLM` for non-embedding baselines).
  - `find_missing(model_set_params)` — filters to jobs whose `save_path` is missing a `DONE.txt`, so resubmission only re-runs incomplete jobs.
  - `submit_all_jobs()` — sequentially submits the eight **core** subfolders below.

- **`submit.py`** in each subfolder — defines `get_model_set_params(seed, subfolder)` which returns a list of jobs (one per maze × model_name), saves the spec to `model_set_params.json` in the results dir, and `sbatch`s each.

Each model writes results to `<RESULTS_PATH>/neGLM/<subfolder>/<maze_name>/<model_name>/` and signals completion with a `DONE.txt`.

## Core comparisons

These eight model sets form the main validation + variance-explained pipeline of the paper. Unless noted, each runs across `maze_1`, `maze_2`, and `rooms_maze`, on `days_on_maze="late"`, with default model/training hyperparameters.

| Subfolder | Question it answers | Models compared |
|---|---|---|
| `performance_validation/` | Does the latent embedding improve decoding over a standard (non-embedding) GLM with the same inputs? | neGLM and baseline GLM each fit with `direction`, `place`, `place_direction`, and `place_direction + distance_to_goal` input sets. Baselines run on CPU (`run_cv_baselineGLM`). |
| `interaction_validation/` | Are place and direction coded conjunctively, factorised, or as a learned non-linear interaction? Same question for `place_direction × distance_to_goal`. | Single-feature, factorised (`partition=[…]`), non-linear, and conjunctive (`place_direction`) variants. |
| `variance_explained/` | How much variance does each of `place_direction` and `distance_to_goal` contribute? | `full_model`, `remove_place_direction`, `remove_distance_to_goal`. `n_folds=10` (double default) and `min_spike_count=150` to retain power across folds. |
| `other_features/` | What additional behavioural variables (head direction, ego/allocentric angle-to-goal, speed, velocity, goal identity) add explanatory variance on top of `place_direction + distance_to_goal`? | Cumulative additions through a feature list, plus targeted goal-encoding and speed/velocity comparisons. |
| `variance_explained_all_sessions/` | Same as `variance_explained` but using **all** session days, not just late sessions. | Identical model set to `variance_explained`, with `days_on_maze="all"`. |
| `variance_explained_null/` | Permutation null distribution for the variance-explained comparison: same models as `variance_explained` but with `n_permutations=1000`, `save_fold_models=True`, and a 10-day SLURM wall-time. Uses all session days. | Compare to non-permuted versions|
