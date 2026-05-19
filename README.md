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

**This repo contains code for:**
- **Preprocessing** raw data into the GridMaze standard format
- **Analysis** codebase implementing all ephys analyses presented in the parent paper
- **Notebook summaries** of those analyses
- **Ephys spike sorting pipeline** (built with [SpikeInterface](https://spikeinterface.readthedocs.io/))
- **pyControl task files** for running experiments on GridMaze hardware

> 📦 **Note:** this repository does **not** contain experiment data or saved results — both are archived on Zenodo: [doi.org/10.5281/zenodo.7863716](https://doi.org/10.5281/zenodo.7863716). See [📥 Downloading data and results](#-downloading-data-and-results) below.

---
## 📁 Code, data, and results organisation

This code repo is designed to live inside a parent folder alongside its `data/` and
`results/` directories:

```
parent_folder/
├── 💻 code/      <- this repo
├── 📦 data/      <- raw, preprocessed, processed, and analysis data
└── 📈 results/   <- figures and saved analysis outputs
```

Instructions for downloading the accompanying `data/` and `results/` from Zenodo are in [📥 Downloading data and results](#-downloading-data-and-results) below.
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

Per-folder READMEs cover each section in more detail: [`Notebooks/`](neGLM/README.md) · [`GridMaze/`](GridMaze/README.md) · [`GridMaze/preprocessing/`](GridMaze/preprocessing/README.md) · [`GridMaze/analysis/`](GridMaze/analysis/README.md) · [`GridMaze/maze`](GridMaze/maze/README.md).

If you are interested in specific analyses from the paper we would recommend finding it `Notebooks/` and then investigating the relevant GridMaze code. 

---

### 📦 Data organisation

```
data/
├── raw_data/                <- data as it comes off the rig
│   ├── pycontrol/           <- behavioural task readout
│   ├── video/               <- top-down video of animals on the maze
│   └── ephys/               <- raw electrophysiology recordings
├── preprocessed_data/       <- outputs from raw-data preprocessing pipelines
│   ├── DeepLabCut/          <- video tracking
│   ├── spikesorting/        <- SpikeInterface + Kilosort 4 output
│   └── HERBS/               <- histology registered to the Allen Atlas
├── processed_data/          <- standardised, human-readable data (subject_ID/session_ID/)
├── analysis_data/           <- compressed, project-specific data optimised for analysis
└── experiment_info/         <- subject IDs, dates, maze configurations, etc.
```



### 📥 Downloading data and results

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
4. If you placed `data/` and `results/` somewhere other than next to `code/`, edit `code/GridMaze/paths.py` to point at your local copies (see [Configuring paths](#configuring-paths) below). Otherwise no further setup is needed.

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

By default `code/GridMaze/paths.py` resolves `data/` and `results/` relative to the `code/` directory, so this layout works out of the box. If you placed the data elsewhere, see [Configuring paths](#configuring-paths).

#### Option 3 — `download_data.sh` helper script

A helper script in the repo root handles download, unzip, optional exclusion of large files (e.g. LFP traces), and the choice of whether to pull `results/`.

> 🚧 **Coming in next commit** — usage will be documented here once `download_data.sh` lands. It will accept flags like `--no-results`, `--no-lfp`, and `--data-dir <path>`.

#### Configuring paths

By default, `GridMaze/paths.py` resolves `data/` and `results/` relative to the `code/` directory:

```python
# code/GridMaze/paths.py
DATA_PATH    = Path("../data")
RESULTS_PATH = Path("../results")
```

If you unzipped `data.zip` and `results.zip` inside the same `parent_folder/` that contains `code/`, **no edits are needed**.

If you downloaded data or results somewhere else, edit these two lines to point at your local copies:

```python
# code/GridMaze/paths.py
DATA_PATH    = Path("/absolute/path/to/your/data")
RESULTS_PATH = Path("/absolute/path/to/your/results")
```

> ℹ️ All scripts and notebooks assume CWD = `code/` (notebooks `os.chdir` to it automatically). If you run a script from a different working directory, prefer absolute paths in `paths.py`.

---

### Regenerating from source

The Zenodo bundle ships only `processed_data/` + `experiment_info/` (in `data.zip`) and saved results (in `results.zip`). Raw and preprocessed data are not redistributed because the files are large and non-standardised. Both upstream and downstream artefacts can be rebuilt locally:

**`processed_data/`** is built from `raw_data/` + `preprocessed_data/` via:

```python
from GridMaze.preprocessing import populate_processed_data as ppd
ppd.populate_processed_data()
```

**`analysis_data/`** is built from `processed_data/` via:

```python
from GridMaze.analysis.processing import populate_analysis_data as pad
pad.populate_analysis_data()
```

> ⚠️ `analysis_data/` is ~50 GB and slow to generate without multiprocessing.

**`results/`** is regenerated by re-running the analysis notebooks in `Notebooks/` against `processed_data/` + `analysis_data/`. The saved `results.zip` from Zenodo simply skips the compute-heavy steps (permutation tests, etc.).

---


## 🔗 Related repos

- ⚡ **Opto experiment code and results** — *(link TBD)*
- 🌈 **Neural Embedding GLM** — *(link TBD)*

---

## 📜 Citation

Please cite both the paper and the dataset:

```bibtex
@article{placeholder,
  title  = {Structured and flexible representations in medial-frontal cortex
            support goal-directed navigation},
  author = {Doohan, Peter T. and colleagues},
  year   = {2026}
}

@dataset{doohan_2026_dataset,
  title     = {Data and results for: Structured and flexible representations in
               medial-frontal cortex support goal-directed navigation},
  author    = {Doohan, Peter T. and colleagues},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.7863716},
  url       = {https://doi.org/10.5281/zenodo.7863716}
}
```

---
