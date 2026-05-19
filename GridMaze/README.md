# `GridMaze/` — core package

The Python package containing preprocessing, analysis, and maze-representation code for the GridMaze ephys experiment. This README is the single documentation entry-point for everything inside `GridMaze/`. The one exception is [`analysis/processing/README.md`](analysis/processing/README.md), a deep reference for the `analysis_data/` generation pipeline.

---

## Quick map

```
GridMaze/
├── paths.py              <- DATA_PATH / RESULTS_PATH definitions
├── preprocessing/        <- raw → processed_data ETL (most users don't run this)
├── maze/                 <- maze graph representations + plotting
└── analysis/
    ├── core/             <- session-loading API (get_maze_sessions)
    ├── processing/       <- generates analysis_data/ (deep reference: ./analysis/processing/README.md)
    └── <16 themed subpackages>   <- the paper analyses
```

---

## `paths.py`

Defines `DATA_PATH` and `RESULTS_PATH` — used by every downstream module. Defaults resolve relative to the `code/` directory, so the `parent_folder/{code, data, results}` layout works out of the box:

```python
DATA_PATH    = Path("../data")
RESULTS_PATH = Path("../results")
```

Change these two lines if your data lives elsewhere — see [Configuring paths](../README.md#configuring-paths-only-if-needed) in the main README.

---

## Loading data: `get_maze_sessions`

The entry point for every analysis. Defined in [`analysis/core/get_sessions.py`](analysis/core/get_sessions.py). Filters sessions by metadata and returns `MazeSession` objects pre-loaded with requested data.

```python
from GridMaze.analysis.core import get_sessions as gs
```

### Signature

```python
def get_maze_sessions(
    subject_IDs="all",
    maze_names="all",
    days_on_maze="all",
    goal_subsets="all",
    with_data="all",
    must_have_data=True,
    verbose=False,
)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `subject_IDs` | `str` or `list[str]` | `"all"` | `"all"` loads every subject in `experiment_info/subject_IDs.json`; or pass a list like `["m2", "m6"]`. |
| `maze_names` | `str` or `list[str]` | `"all"` | `"all"` loads every maze in `experiment_info/maze_configs.json`; or pass a list like `["maze_1", "maze_2"]`. |
| `days_on_maze` | `str` or `list[int]` | `"all"` | `"all"` for every day; `"late"` for the last 7 days only; or a list of ints (e.g. `[1, 5, 10]`) for specific days. Invalid day numbers are silently skipped. |
| `goal_subsets` | `str` or `list[str]` | `"all"` | `"all"` expands to `["all", "subset_1", "subset_2"]`; or pass a custom list. |
| `with_data` | `str` or `list[str]` | `"all"` | `"all"` loads every available processed + analysis-data attribute (23 in total). Or a list — see the attribute tables below. |
| `must_have_data` | `bool` | `True` | If `True`, sessions missing any requested `with_data` attribute are dropped from the result. If `False`, they're retained with `None` placeholders. |
| `verbose` | `bool` | `False` | If `True`, prints a warning each time a requested file is missing. |

### Return type — single vs list

- **Multiple matches** → `list[MazeSession]`.
- **Exactly one match** → a single `MazeSession` (not wrapped in a list).
- **No matches** → `FileNotFoundError`.

> ⚠️ **Footgun.** Because of the dual single/list return, downstream code that always iterates can break on a one-session query. Use this idiom if your filter could match either count:
> ```python
> result = gs.get_maze_sessions(...)
> sessions = result if isinstance(result, list) else [result]
> ```

### `MazeSession` — metadata attributes

Always populated, regardless of `with_data`:

| Attribute | Type | Description |
|---|---|---|
| `subject_ID` | `str` | e.g. `"m2"` |
| `maze_name` | `str` | e.g. `"maze_1"` |
| `day_on_maze` | `int` | 1-indexed day count for this animal on this maze |
| `goal_subset` | `str` | one of `"all"`, `"subset_1"`, `"subset_2"` |
| `name` | `str` | formatted `"{subject}.{date}.{type}"` e.g. `"m2.2022-07-04.maze"` |
| `date` | `datetime.date` | session date |
| `late_session` | `bool` | auto-computed; `True` if within the last 7 days on this maze |
| `maze_structure` | `dict` | maze topology (nodes, edges, layout) |
| `session_info` | `dict` | raw contents of `session_info.json` |
| `has_data` | `list[str]` | names of `with_data` attributes that were successfully loaded |

### `MazeSession` — data attributes

**Loaded from `processed_data/`** (shipped on Zenodo):

| Attribute | Source file |
|---|---|
| `events_df` | `events.htsv` |
| `trials_df` | `trials.htsv` |
| `spike_times` | `spikes.times.npy` |
| `spike_clusters` | `spikes.clusters.npy` |
| `cluster_metrics` | `clusters.metrics.htsv` |
| `tracking_df` | `frames.tracking.htsv` |
| `trajectories_df` | `frames.trajectories.htsv` |
| `trial_info_df` | `frames.trialInfo.htsv` |
| `lfp_times` | `lfp.times.npy` |
| `lfp_signal` | `lfp.signal.npy` |
| `lfp_metrics` | `lfp.metrics.htsv` |

**Loaded from `analysis_data/`** (generated locally via `populate_analysis_data`):

| Attribute | Source file |
|---|---|
| `navigation_df` | `frames.navigation.parquet` |
| `navigation_spike_rates_df` | `frames.spikeRates.parquet` |
| `navigation_spike_counts_df` | `frames.spikeCounts.parquet` |
| `navigation_theta_spike_counts_df` | `frames.thetaSpikeCounts.parquet` |
| `trial_aligned_rates_df` | `trial_aligned_rates.parquet` |
| `event_aligned_rates_df` | `event_aligned_rates.parquet` |
| `trajectory_decisions_df` | `trajectory_decisions.parquet` |
| `cluster_distance_tuning_metrics` | `clusters.distanceTuningMetrics.parquet` |
| `cluster_place_direction_tuning_metrics` | `clusters.placeDirectionTuningMetrics.parquet` |
| `cluster_egocentric_action_tuning_metrics` | `clusters.egocentricActionTuningMetrics.parquet` |
| `cluster_movement_metrics` | `clusters.movementMetrics.parquet` |
| `cluster_theta_modulation_metrics` | `clusters.thetaModulationMetrics.parquet` |

> Missing files → the attribute exists but is `None`. Check `session.has_data` for the list that actually loaded.

### `MazeSession` — methods

- `get_clusters(single_units=True)` — returns `MazeCluster` objects for this session.
- `simple_maze()` — networkx graph of the maze with simplified topology.
- `skeleton_maze()` — networkx graph of just the connectivity skeleton.
- `get_navigation_activity_df(type="rates", with_routes=False, cluster_kwargs={})` — joins `navigation_df` with per-frame neural activity (rates or counts), filtered to single units by default.

### Examples

**Single session, everything loaded:**

```python
session = gs.get_maze_sessions(
    subject_IDs=["m6"], maze_names=["maze_1"], days_on_maze=[12], with_data="all"
)
print(session.has_data)
```

**Every late-stage session on `maze_1`, navigation + trials only:**

```python
sessions = gs.get_maze_sessions(
    maze_names=["maze_1"],
    days_on_maze="late",
    with_data=["navigation_df", "trials_df"],
)
```

**All sessions for a behavioural analysis (no spike data needed):**

```python
sessions = gs.get_maze_sessions(
    days_on_maze="all",
    with_data=["trials_df", "trajectories_df"],
)
```

---

## `analysis/`

The paper analyses, split into themed subpackages. Each is roughly one analysis topic. For the figure-by-figure walkthrough that calls into these subpackages, see [`Notebooks/README.md`](../Notebooks/README.md).

| Subpackage | Computes | Paper figure |
|---|---|---|
| [`core/`](analysis/core/) | Session loading API (above), filters, encoding/decoding utilities | — |
| [`processing/`](analysis/processing/) | Builds `analysis_data/` parquets — see [`analysis/processing/README.md`](analysis/processing/README.md) | — |
| `behaviour/` | Performance metrics, trajectory plotting, dimensionality reduction | Fig 1 |
| `navigation_strategies/` | Mixture-of-strategies behavioural model fits | Fig 1 |
| `event_aligned/` | Trial/event-locked population activity, reward representations | Fig 3 |
| `anatomy/` | Brain region distribution, anatomical coverage | Methods |
| `cluster_tuning/` | Per-neuron tuning curves (spatial, HD, distance, movement, ego-action) | Figs 3–5 |
| `place_direction/` | Place × head-direction tuning, decoding, RSA, efficient coding | Figs 3–4 |
| `goal_coding/` | Goal-location decoders + controls (place, pseudo-trial) | Fig 3 |
| `distance_to_goal/` | Distance-to-goal tuning + decoding, theta modulation | Figs 4, 7 |
| `ego_angle/` | Egocentric angle-to-goal population tuning + decoding | Fig 6 |
| `egocentric_action/` | Left / forward / right action coding dynamics | Fig 5 |
| `velocity/` | Speed / 2D velocity tuning | Supp |
| `unit_match/` | UnitMatch cross-session cell tracking | Methods |
| `neGLM/` | Neural-embedding GLM encoding models | Fig 6 |
| `lfp/` | Raw LFP + theta-phase neuron-level analyses | Fig 7 |
| `theta_HD/` | Theta-phase-dependent head-direction dynamics | Fig 7 |
| `theta_mod/` | Theta-phase-stratified decoding | Fig 7 |

---

## `maze/`

Maze representations and plotting helpers — networkx-based graph models of the gridworld plus utilities for rendering activity onto them.

- `representations.py` — build `simple_maze` and `skeleton_maze` networkx graphs from session maze structure
- `plotting.py` — render maze graphs with overlaid activity heatmaps and trajectory traces
- `metrics.py` — graph-derived metrics (shortest paths, RSA-style dissimilarity matrices)
- `partitions.py` — partition the maze into sub-regions for generalisation tests

Typical use is via a `MazeSession` (which calls the same functions internally):

```python
session = gs.get_maze_sessions(subject_IDs=["m6"], maze_names=["maze_1"], days_on_maze=[12])
G = session.simple_maze()        # networkx graph
```

> 📓 The [companion data repo](https://github.com/peterdoohan/GridMaze-mFC-ephys-DATA) hosts a walked-through notebook explaining the maze graph representations in more detail.

---

## `preprocessing/`

Raw-data ingestion: pyControl logs, top-down video, raw ephys, DeepLabCut tracking, and Kilosort outputs → standardised per-session `processed_data/`. **Most users won't run this** — `processed_data/` ships on Zenodo (see [Downloading data and results](../README.md#-downloading-data-and-results)). The code is here so the full preprocessing chain is inspectable.

Entry point (only if you have raw + preprocessed data):

```python
from GridMaze.preprocessing import populate_processed_data as ppd
ppd.populate_processed_data()
```

Relies on metadata files in `experiment_info/` (`maze_configs.json`, `subject_IDs.json`, `probe_depths.htsv`, …).

### `processed_data/` conventions

Per-session folder named `<YYYY-MM-DD>.<type>` where `<type>` is `maze` or `rest`. Filenames follow the [IBL convention](https://doi.org/10.1101/827873) `object.attribute.filetype`:

- All files for the same `object` share the same first dimension — e.g. every `frames.*` file has one row per video frame.
- `.npy` for numerical arrays (load with `numpy.load`).
- `.htsv` for tabular data with tab separation and a header row (load with `pandas.read_csv(..., sep="\t")`).
- `.json` for nested key/value metadata.

**Units:** times are in seconds from pycontrol session start; spatial measurements in SI units.

**File list per session:** `session_info.json`, `events.htsv`, `trials.htsv`, `spikes.times.npy`, `spikes.clusters.npy`, `clusters.metrics.htsv`, `frames.tracking.htsv`, `frames.trajectories.htsv`, `frames.trialInfo.htsv`, `lfp.signal.npy`, `lfp.times.npy`, `lfp.metrics.htsv`, plus the `UnitMatch/` subfolder for cross-session matching outputs.

---

## Where to next

- Reproducing a paper figure → [`Notebooks/README.md`](../Notebooks/README.md)
- Generating `analysis_data/` → [`analysis/processing/README.md`](analysis/processing/README.md)
- Running compute-heavy analyses on SLURM → [`jobs/README.md`](../jobs/README.md)
- Downloading data, env setup, repo layout → [main README](../README.md)
