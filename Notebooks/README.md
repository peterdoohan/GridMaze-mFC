<div align="center">

# 📓 Notebooks

</div>

The main entry point to the analyses in the companion paper. Each notebook reproduces the figures from one section of the paper and is numbered by paper figure (e.g. `4.structured_reps.ipynb` → Figure 4).

---

## 🔁 General workflow

Each notebook follows the same recipe:

1. **Set the working directory** to the `code/` folder so the `GridMaze` package is importable:
    ```python
    import os
    os.chdir("/path/to/parent_folder/code")
    ```
2. **Import** the relevant `GridMaze` analysis modules.
3. **Call** functions that either:
    - load `processed_data/` and `analysis_data/` using `GridMaze/analysis/core/get_sessions.py` 
    - or load already processed analysis results data from `../results`, 
4. **Run** the analysis
5. **Plot** the summary figures.

Figures are saved to `../results/figures/<notebook_name>/`.

> 💡 If you are interested in a specific analysis from the paper, open the notebook for that figure and scroll to the relevant cell. The `GridMaze` functions being called are the analysis implementation — **Ctrl/Cmd-click the import in any IDE (or GitHub)** to jump straight to the source.

---

## 📚 Directory

| Notebook | Paper figure | Contents |
|---|---|---|
| [`1.maze_behaviour.ipynb`](1.maze_behaviour.ipynb) | Fig. 1 | Maze layouts, example trajectories, learning curves, mixture-of-strategies model fits + comparisons |
| [`3.task_structure.ipynb`](3.task_structure.ipynb) | Fig. 3 | trial-event aligned population activity & LFP/CSD spectrograms, example cells, trial-aligned cluster heatmaps |
| [`4.structured_reps.ipynb`](4.structured_reps.ipynb) | Fig. 4 | Example place-direction cells, NMF/PCA dim-red on neural & behavioural populations, efficient-coding, past/future decoding, RSA, cross-maze remapping (UnitMatch) |
| [`5.flexible_reps.ipynb`](5.flexible_reps.ipynb) | Fig. 5 | Allocentric goal coding, distance-to-goal tuning (single units + population), goal & distance decoding |
| [`6.neGLM.ipynb`](6.neGLM.ipynb) | Fig. 6 | neGLM model comparisons, variance explained, mixed vs. factorised comparisons|
| [`7.theta_mod.ipynb`](7.theta_mod.ipynb) | Fig. 7 | Theta-band LFP, theta modulation of distance-to-goal rep, theta-modulation of place-direction rep |

> Figure 2 relates to the opto experiment, which lives in a separate repo — *(link TBD)*.

---
## ⚙️ Running notebooks yourself

Make sure you have:
1. Set up this repo next to `data` and `results` folders (see [main README](../README.md)).
2. Downloaded `data` (and optionally `results` to save rerunning intensive analyses) — from Zenodo or via the [companion data repo](https://github.com/peterdoohan/GridMaze-mFC-ephys-DATA).
3. Populated `analysis_data`:
   ```python
   from GridMaze.analysis.processing import populate_analysis_data as pad
   pad.populate_analysis_data()
   ```
   See [`GridMaze/analysis/processing/README.md`](../GridMaze/analysis/processing/README.md) for the full recipe.
4. By default `GridMaze/paths.py` resolves `data/` and `results/` relative to `code/` — no edits needed. Only edit it if you placed data somewhere else.
5. Try running the notebooks — [raise a GitHub issue](https://github.com/peterdoohan/GridMaze-mFC/issues) if things go wrong!

> ⚠️ `analysis_data/` is ~50 GB and slow to generate without multiprocessing.
---

## 💻 Related code

- [`GridMaze/`](../GridMaze/README.md) — `get_maze_sessions` API, subpackage map, processed-data format spec
- [`GridMaze/analysis/processing/`](../GridMaze/analysis/processing/README.md) — `analysis_data/` generation recipes
- [`jobs/`](../jobs/README.md) — SLURM submission scripts for compute-heavy analyses

---

## 🔗 Related repos

- ⚡ **Opto experiment code and results** — *(link TBD)*
- 🌈 **Neural Embedding GLM** — *(link TBD)*
