"""
Load nbeGLM mode set results (defined and run in jobs/nbeGLM/{model_set_name}/submit.py) from the results folder.
@peterdoohan
"""

# %% Imports
import json
import pickle
import pandas as pd

# %% Global Variables

from GridMaze.paths import RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "neGLM"

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


def load_permuted_model_set_cv_scores(
    model_set, maze_names=["maze_1", "maze_2", "rooms_maze"], ps=[0], all_completed=True
):
    """
    See results/neGLM/{model_set}/permuation/model_set_params.json to see find model params
    Note these results folders have folder structure
       - model_set/permutation/maze_name/model_name/cv_scores.csv (containing cv fit scores for each cluster
         where spikes and behaviour data have been randomly circularly permuted)
    """
    dfs = []
    for p in ps:
        model_set_dir = RESULTS_DIR / model_set / str(p)
        if not model_set_dir.exists():
            raise FileNotFoundError(f"Model set directory does not exist: {model_set_dir}")
        for maze_name in maze_names:
            _dir = model_set_dir / maze_name
            if not _dir.exists():
                raise FileNotFoundError(f"Model set directory does not exist: {_dir}")
            _dfs = _get_result_dfs(_dir, all_completed=all_completed, permutation=p)
            dfs.extend(_dfs)
    return pd.concat(dfs, ignore_index=True)


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
