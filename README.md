<div align="center">

### Code repository for  *Doohan et al., 2026* [electrophysiology experiment]

![Python](https://img.shields.io/badge/python-3.12-blue)
![Platform](https://img.shields.io/badge/platform-linux-lightgrey)
![Status](https://img.shields.io/badge/status-research-orange)
![License](https://img.shields.io/badge/license-BSD--style-green)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7863716.svg)](https://doi.org/10.5281/zenodo.7863716)

<pre>
●     ●─────●     ●     ●─────●─────●
│     │     │     │     │     │      
●─────●     ●     ●─────●     ●     ●
│           │     │           │     │
●─────●─────●─────●     ●─────●─────●
│           │     │     │     │     │
●     ●─────●     ●     ●     ●     ●
│           │     │           │      
●─────●     ●     ●─────●─────●─────●
│           │           │     │     │
●     ●─────●─────●     ●     ●     ●
│     │                 │     │     │
●─────●─────●─────●─────●     ●     ●
</pre>

# Structured and flexible representations in medial-frontal cortex supports goal-directed navigation

</div>

---

> 🚧 **Work in progress** 🛠️ — code in this repository is actively maintained and subject to change ahead of final publication of the accompanying manuscript. If you encounter any issues, please [open a GitHub issue](https://github.com/peterdoohan/GridMaze-mFC/issues) and I'll get back to you as soon as I can.

> 📊 **Just want the data, not these analyses?** A lightweight companion repo at [github.com/peterdoohan/GridMaze-mFC-ephys-DATA](https://github.com/peterdoohan/GridMaze-mFC-ephys-DATA) hosts the same Zenodo data and results without this codebase — recommended starting point if you're interested in the dataset but not the specific analyses in our paper.

---

**This repo contains code for:**
- **Preprocessing** raw data into the GridMaze standard format
- **Analysis** codebase implementing all ephys analyses presented in the parent paper
- **Notebook summaries** of those analyses
- **Ephys spike sorting pipeline** (built with [SpikeInterface](https://spikeinterface.readthedocs.io/))
- **pyControl task files** for running experiments on GridMaze hardware

---

## 📁 Project organisation

This code repo is designed to live inside a parent folder alongside its `data/` and `results/` directories:

```
parent_folder/
├── 💻 code/      <- this repo
├── 📦 data/      <- raw, preprocessed, processed, and analysis data
└── 📈 results/   <- figures and saved analysis outputs
```

The sections below walk through downloading the data, setting up the environment, and running the code in that order.

---

## 📥 Downloading data and results

Both `data/` and `results/` are archived on Zenodo:

- 🆔 [10.5281/zenodo.7863716](https://doi.org/10.5281/zenodo.7863716)

The record contains two zips:

| File          | Contents                                              | Size    |
|---------------|-------------------------------------------------------|---------|
| `data.zip`    | `processed_data/` + `experiment_info/`                | 73.1 GB |
| `results.zip` | saved analysis outputs (permutation tests, etc.)      | 30.9 GB |

> 📈 **About `results/`:** all contents of `results/` can be regenerated locally from the code in this repo and `processed_data/` from `data.zip`. `results.zip` is provided for convenience: many saved structures (permutation tests, compute-heavy analyses) take a long time to rebuild, and downloading is faster than recomputing.

Pick whichever of the three options below suits you.

#### Option 1 — Manual download

1. Open [zenodo.org/records/20267467](https://zenodo.org/records/20267467) in a browser.
2. Download `data.zip` (required) and `results.zip` (optional).
3. Unzip them so the layout next to the cloned repo is:
   ```
   parent_folder/
   ├── code/      <- this repo
   ├── data/      <- from data.zip
   └── results/   <- from results.zip
   ```
4. If you placed `data/` and `results/` somewhere other than next to `code/`, edit `code/GridMaze/paths.py` to point at your local copies (see [Configuring paths](#configuring-paths-only-if-needed) below). 

#### Option 2 — curl

From inside `parent_folder/`:

```bash
# required
curl -L -o data.zip    https://zenodo.org/records/20267467/files/data.zip
unzip data.zip && rm data.zip

# optional
curl -L -o results.zip https://zenodo.org/records/20267467/files/results.zip
unzip results.zip && rm results.zip
```

By default `code/GridMaze/paths.py` resolves `data/` and `results/` relative to the `code/` directory, so this layout works out of the box. If you placed the data elsewhere, see [Configuring paths](#configuring-paths-only-if-needed).

#### Option 3 — `download_data.sh` helper script

A helper script in the repo root handles the curl + unzip dance and lands the data and results in the correct sibling folders, so the default `paths.py` resolution works without further configuration. Run from inside `code/`:

```bash
# defaults: download both data.zip + results.zip, verify, unzip into ../data and ../results
bash download_data.sh

# data only, skip the (large) saved results
bash download_data.sh --no-results

# skip extracting LFP files — saves ~270 MB per session on disk
bash download_data.sh --no-lfp

# custom destinations
bash download_data.sh --data-dir /scratch/gridmaze/data --results-dir /scratch/gridmaze/results
```

---

## ⚙️ Environment installation

The Python environment is managed with [miniconda](https://docs.conda.io/projects/miniconda/). Once miniconda is installed, recreate the environment from the provided spec:

```bash
git clone <repo-url> code
cd code
conda env create -f environment.yml
conda activate GridMaze_mFC
```

This installs Python 3.12 and pins all dependencies used across preprocessing, analysis, and notebooks. Tested on Linux; should also resolve on macOS / Windows (CPU-only PyTorch by default — install a CUDA build separately if you need GPU).

> 🧠 **Optional:** `GridMaze/preprocessing/probe_fit.py` requires [`allensdk`](https://allensdk.readthedocs.io/), which is excluded from `environment.yml` due to its heavy and version-sensitive dependencies. Install it into a separate env (`pip install allensdk`) if you want to play around with this code.

---

## 📦 Data organisation

The Zenodo bundle ships `processed_data/` (standardised per-session data ready for analysis) and `experiment_info/` (subject IDs, dates, maze configs). Just enough to give you the gist:

```
data/
├── raw_data/                <- data as it comes off the rig (not shipped)
│   ├── pycontrol/           <- behavioural task readout
│   ├── video/               <- top-down video of animals on the maze
│   └── ephys/               <- raw electrophysiology recordings
├── preprocessed_data/       <- outputs from raw-data preprocessing (not shipped)
│   ├── DeepLabCut/          <- video tracking
│   ├── spikesorting/        <- SpikeInterface + Kilosort 4 output
│   └── HERBS/               <- histology registered to the Allen Atlas
├── processed_data/          <- standardised, human-readable data (subject_ID/session_ID/)
├── analysis_data/           <- compressed, analysis-optimised cache (generated locally)
└── experiment_info/         <- subject IDs, dates, maze configurations, etc.
```

For the full processed-data format spec — file types, naming conventions, units — see [`GridMaze/README.md`](GridMaze/README.md). If you only want the dataset without this codebase, the [companion data repo](https://github.com/peterdoohan/GridMaze-mFC-ephys-DATA) is a friendlier starting point.

---

## 💻 Code organisation

```
code/
├── GridMaze/                <- core package: preprocessing + analysis for GridMaze
│   ├── preprocessing/       <- maps raw data → standardised processed_data
│   ├── analysis/            <- generates all analyses presented in the paper
│   ├── maze/                <- networkx-based maze representations + plotting
│   └── paths.py             <- central registry of data + results paths
├── Notebooks/               <- main entry point — analyses by paper section
├── jobs/                    <- SLURM job submission for compute-heavy analyses
├── neGLM/                   <- Neural Embedding GLM tool (standalone)
├── SpikeSorting/            <- spikesorting pipeline (Kilosort 4 + UnitMatch via SpikeInterface)
└── TaskFiles/               <- pyControl task files for the maze hardware
```

Per-folder READMEs cover specific subsystems in more detail:
[`Notebooks/`](Notebooks/README.md) · [`GridMaze/`](GridMaze/README.md) · [`GridMaze/analysis/processing/`](GridMaze/analysis/processing/README.md) · [`jobs/`](jobs/README.md).

---

## ▶️ Running code locally

After downloading `data.zip`, you'll need to generate `analysis_data/` from downloaded `processed_data/` – this is handled by [`GridMaze/analysis/processing`](./GridMaze/analysis/processing/README.md). `analysis_data` mirrors `processed_data` in structure but contains derivate data tables that are convient starting points for different analyses and are required to run most of the analyses in the Notebook summaries below (~50 GB).

**Build `analysis_data/`** from `processed_data/`:

```python
from GridMaze.analysis.processing import populate_analysis_data as pad
pad.populate_analysis_data()
```

> ⚠️ Slow to generate without multiprocessing. See [`GridMaze/analysis/processing/README.md`](GridMaze/analysis/processing/README.md) for the full recipe, table specs, partial builds, and gotchas.

**If you're interested in how `processed_data/` was generated from raw recordings**: see the `preprocessing/` section of [`GridMaze/README.md`](GridMaze/README.md).

```python
from GridMaze.preprocessing import populate_processed_data as ppd
ppd.populate_processed_data()
```

### Configuring paths (only if needed)

By default `GridMaze/paths.py` resolves `data/` and `results/` relative to `code/`, so the layout above just works. If you placed data elsewhere, edit:

```python
# code/GridMaze/paths.py
DATA_PATH    = Path("/absolute/path/to/your/data")
RESULTS_PATH = Path("/absolute/path/to/your/results")
```

> ℹ️ Scripts and notebooks assume CWD = `code/` (notebooks `os.chdir` to it automatically). If you run a script from a different working directory, prefer absolute paths in `paths.py`.

---

## 📓 Jumping into the analyses

The `Notebooks/` folder is the main entry point for reproducing each figure of the paper. Notebooks are organised by paper section and walk through the analysis from data loading to figure generation.

See [`Notebooks/README.md`](Notebooks/README.md) for the figure-by-figure index and recommended reading order.

> ‼️ if `results` was not downloaded from Zenodo, some analyses will start running compute heavy processing that takes a while. Consider downloaded cached results if you want to run things faster. 

> 💡 New to the dataset? The [companion data repo](https://github.com/peterdoohan/GridMaze-mFC-ephys-DATA) hosts simpler walked-through notebooks aimed at general data exploration, not paper-specific analyses.

---

## 🔗 Related repos

- ⚡ **Opto experiment code and results** — [github.com/peterdoohan/GridMaze-mFC-opto](https://github.com/peterdoohan/GridMaze-mFC-opto)
- 📊 **Companion data repo** — [github.com/peterdoohan/GridMaze-mFC-ephys-DATA](https://github.com/peterdoohan/GridMaze-mFC-ephys-DATA)
- 🌈 **Neural Embedding GLM** — *TODO:* see [`neGLM`](./neGLM/) for code, docs and separate repo to come.
 
---

## 📜 Citation

Please cite both the paper and the dataset:

```bibtex
@article{placeholder,
  title  = {Structured and flexible representations in medial-frontal cortex
            support goal-directed navigation},
  author = {Doohan, Peter T. Jensen, Jensen, Kristopher, T. Chen, Yaqing. Godinho, Beatriz. Burns, Charles D.G. Qin, Chongyu (Xiao). Emery, Josie. Cini, Ryan. Walton, Mark E. T. Behrens, Timothy E.J. Akam, Thomas E.},
  year   = {2026}
}

@dataset{doohan_2026_dataset,
  title     = {Data and results for: Structured and flexible representations in
               medial-frontal cortex support goal-directed navigation},
  author    = {Doohan, Peter T. Behrens, Timothy E.J. Akam, Thomas E.},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.7863716},
  url       = {https://doi.org/10.5281/zenodo.7863716}
}
```

---
