"""
Load nbeGLM mode set results (defined and run in jobs/nbeGLM/{model_set_name}/submit.py) from the results folder.
@peterdoohan
"""

# %% Imports
import pandas as pd

# %% Global Variables

from GridMaze.paths import RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "nbeGLM"

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
