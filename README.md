<div align="center">

### Code repository for  *Doohan et al., 2026* [electrophysiology experiment]

![Python](https://img.shields.io/badge/python-3.12-blue)
![Platform](https://img.shields.io/badge/platform-linux-lightgrey)
![Status](https://img.shields.io/badge/status-research-orange)
![License](https://img.shields.io/badge/license-BSD--style-green)

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

> 📦 **Note:** this repository does **not** contain experiment data, you can find that [here]()

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

Instructions for downloading the accompanying data and/or results can be found in the 📦 [data repo]().
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



### 📥 Data availability

The accompanying [data repo]() incldues only `processed_data/` and `experiment_info/`, which is all that is needed to reproduce analyses. 
We have decided not make `raw_data/` and `preprocessed_data/` publically available because the files are large and non-standardised. However, code that generates `processed_data/` can be inspected here:

**`processed_data/`** was build from `raw_data/` & `preprocessed_data/` using:

```python
from GridMaze.preprocessing import populate_processed_data as ppd
ppd.populate_processed_data()
```

The accompanying `analysis_data/` data is not available for download but can be rebuilt locally from `processed_data/`.

**Rebuild `analysis_data/`** from `processed_data/`:

```python
from GridMaze.analysis.processing import populate_analysis_data as pad
pad.populate_analysis_data()
```

> ⚠️ `analysis_data/` is ~50 GB and slow to generate without multiprocessing.

---


## 🔗 Related repos

- ⚡ **Opto experiment code and results** — *(link TBD)*
- 🌈 **Neural Embedding GLM** — *(link TBD)*

---

## 📜 Citation

```bibtex
@article{placeholder,
  title  = {Structured and flexible representations in medial-frontal cortex
            support goal-directed navigation},
  author = {Doohan, Peter T. and colleagues},
  year   = {2026}
}
```

---
