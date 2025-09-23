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


def load_model_set_cv_scores(model_set, maze_names=["maze_1"], all_completed=True):
    """ """
    model_set_dir = RESULTS_DIR / model_set
    dfs = []
    for maze_name in maze_names:
        _dir = model_set_dir / maze_name
        if not _dir.exists():
            raise FileNotFoundError(f"Model set directory does not exist: {_dir}")
        results_dirs = [f for f in _dir.iterdir() if f.is_dir()]
        for results_dir in results_dirs:
            # check if results have been processed
            if (results_dir / "DONE.txt").exists():
                # load cv scores
                cv_scores_path = results_dir / "cv_scores.csv"
                df = pd.read_csv(cv_scores_path)
                model_name = results_dir.name
                df["model_name"] = model_name
                dfs.append(df)
            else:
                if all_completed:
                    raise FileNotFoundError(f"Results directory not completed: {results_dir}")
                else:
                    continue
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
