"""
Load nbeGLM mode set results (defined and run in jobs/nbeGLM/{model_set_name}/submit.py) from the results folder.
@peterdoohan
"""

# %% Imports
import json
import pickle
import pandas as pd

# %% Global Variables

from GridMaze.paths import RESULTS2_PATH

RESULTS_DIR = RESULTS2_PATH / "neGLM"

# %% Functions


def load_model_set_cv_scores(model_set, maze_names=["maze_1", "maze_2", "rooms_maze"], all_completed=True):
    """
    See results/neGLM/{model_set}/model_set_params.json to see find model params
    Note these results folders have folder structure
       - model_set/maze_name/model_name/cv_scores.csv (containing cv fit scores for each cluster across folds)
    """
    model_set_dir = RESULTS_DIR / model_set
    dfs = []
    for maze_name in maze_names:
        _dir = model_set_dir / maze_name
        if not _dir.exists():
            raise FileNotFoundError(f"Model set directory does not exist: {_dir}")
        _dfs = _get_result_dfs(_dir, all_completed=all_completed)
        dfs.extend(_dfs)
    return pd.concat(dfs, ignore_index=True)


def load_model_set_rotation_null(
    model_set,
    maze_names=["maze_1", "maze_2", "rooms_maze"],
    all_completed=True,
):
    """
    Load the rotation-null triplet for a model set whose runs were generated with
    n_permutations > 0 (see run_neGLM.run_cv_neGLM).

    Returns (cv_scores_df, ridge_cv_scores_df, perm_cv_scores_df):
      - cv_scores_df:       Poisson D² on true held-out spikes (headline embedding score).
      - ridge_cv_scores_df: Ridge R² on true held-out spikes — matched baseline for the
                            rotation null (this, not cv_scores_df, is the apples-to-apples
                            reference for perm_cv_scores_df).
      - perm_cv_scores_df:  Ridge R² on Haar-rotated held-out spikes; one row per
                            (neuron, fold, permutation).

    Folder structure: model_set/maze_name/model_name/{cv_scores,ridge_cv_scores,perm_cv_scores}.csv
    """
    model_set_dir = RESULTS_DIR / model_set
    cv_dfs, ridge_dfs, perm_dfs = [], [], []
    for maze_name in maze_names:
        _dir = model_set_dir / maze_name
        if not _dir.exists():
            raise FileNotFoundError(f"Model set directory does not exist: {_dir}")
        ridge_dfs.extend(_get_result_dfs(_dir, all_completed=all_completed, filename="ridge_cv_scores.csv"))
        perm_dfs.extend(_get_result_dfs(_dir, all_completed=all_completed, filename="perm_cv_scores.csv"))
    return (
        pd.concat(ridge_dfs, ignore_index=True),
        pd.concat(perm_dfs, ignore_index=True),
    )


def _get_result_dfs(_dir, all_completed=True, permutation=None, filename="cv_scores.csv"):
    _dfs = []
    results_dirs = [f for f in _dir.iterdir() if f.is_dir()]
    for results_dir in results_dirs:
        # check if results have been processed
        if (results_dir / "DONE.txt").exists():
            df = pd.read_csv(results_dir / filename)
            model_name = results_dir.name
            df["model_name"] = model_name
            if permutation is not None:
                df["permutation"] = permutation
            _dfs.append(df)
        else:
            if all_completed:
                raise FileNotFoundError(f"Results directory not completed: {results_dir}")
            else:
                continue
    return _dfs


def load_model_set_training(model_set, maze_names=["maze_1", "maze_2", "rooms_maze"], all_completed=True):
    """
    Load training logs for every model in a model set.
    Folder structure: model_set/maze_name/model_name/training.csv
    (columns: epoch, train_loss, train_embedding_perf, test_embedding_perf, subject_ID, maze_name, day_on_maze)
    """
    model_set_dir = RESULTS_DIR / model_set
    dfs = []
    for maze_name in maze_names:
        _dir = model_set_dir / maze_name
        if not _dir.exists():
            raise FileNotFoundError(f"Model set directory does not exist: {_dir}")
        dfs.extend(_get_result_dfs(_dir, all_completed=all_completed, filename="training.csv"))
    return pd.concat(dfs, ignore_index=True)


def load_model(
    model_set="full_models",
    model_name="full_model",
    maze_name="maze_1",
    with_model_params=False,
):
    """ """
    model_path = RESULTS_DIR / model_set / maze_name / model_name / "model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    if not with_model_params:
        return model
    else:
        model_params_path = RESULTS_DIR / model_set / maze_name / model_name / "model_params.json"
        with open(model_params_path, "r") as f:
            model_params = json.load(f)
    return model, model_params
