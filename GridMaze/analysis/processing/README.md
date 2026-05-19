# Analysis data processing

This package turns `processed_data/` (one folder per session, same structures as `processed_data`) into the compressed, project-specific tables that all downstream analyses (notebooks, jobs, paper figures) load from. Everything written here lives in `data/analysis_data/`:

```
analysis_data/
├── analysis_info/                          <- experiment-wide constants (json)
└── <subject_ID>/<session_ID>/              <- per-session analysis tables (parquet)
```
>❓**What is analysis data for**❓Across analyses we end up doing the same computations many times (eg, aligning spike times with firing rates). Analysis data is designed to be convient for downstream analyses and can be loaded similar to processed data using MazeSession Objectts.
>

There are two streams:

| Entry point | What it builds | Where it writes |
|---|---|---|
| `save_analysis_info()` | Experiment-wide constants pooled across all sessions (movement threshold, average intra-trial-interval times, mean place occupancy, mean edge transition counts) | `analysis_data/analysis_info/*.json` |
| `populate_analysis_data()` | Per-session tables (navigation frames, spike rates/counts, time-aligned rates, trajectory decisions, cluster tuning metrics, …) | `analysis_data/<subject>/<session>/*.parquet` |

> ⚠️ **There is a chicken-and-egg ordering between these two.** Some `analysis_info` constants need the per-session parquet tables to exist, and some per-session tables need `analysis_info` constants to exist. The recipe below handles this.

---

## 🚀 Quick start — generate everything from scratch

After downloading `processed_data/` (see [parent README](../../../README.md) and sibling [companion data repo](https://github.com/peterdoohan/GridMaze-mFC-ephys-DATA)), run this once from inside the `GridMaze_mFC` conda env, with CWD set to your `code/` folder:

```python
from GridMaze.analysis.processing import populate_analysis_data as pad
from GridMaze.analysis.processing import get_analysis_info as gai

# 1. Build the two analysis_info constants that don't depend on analysis_data —
#    these are needed by populate_analysis_data in step 2.
gai.save_analysis_info(["movement_threshold", "intra_trial_interval_times"])

# 2. Build all per-session analysis tables (~50 GB, slow — use parallel_jobs).
#    also see code/jobs/analysis_data for SLURM submitable version of this
pad.populate_analysis_data(parallel_jobs=8)

# 3. Build the remaining two analysis_info constants, which need
#    frames.navigation.parquet and trajectory_decisions.parquet to exist.
gai.save_analysis_info(["maze_name2mean_occupancy", "maze_name2edge_transition_counts"])
```

Each session that finishes gets its parquet files in `analysis_data/<subject>/<session>/`. Re-running with `overwrite=False` (the default) skips files that already exist, so you can resume after a crash without redoing finished work.

### Building only a subset

```python
# one data structure across all sessions
pad.populate_analysis_data(
    data_structures=["frames.navigation.parquet"],
    parallel_jobs=8,
)
```

The list of valid `data_structures` names is the `filename` column of `ANALYSIS_DATA_STRUCTURES_DF` in [`populate_analysis_data.py`](populate_analysis_data.py) — same as the table further down this README.

---

## 📦 What ends up in `analysis_info/`

Four json files, all written by `save_analysis_info()`:

| File | Built from | Used by |
|---|---|---|
| `movement_threshold.json` | Per-frame speeds across all sessions, fit with a 2-component GMM in log-space → threshold separating "stationary" from "moving" | Stamps the `moving` column when building `frames.navigation.parquet`; downstream analyses that filter for `moving == True` |
| `intra_trial_interval_times.json` | Median cue→reward→end-reward-consumption→trial-end times across subjects | Trial warping in `trial_aligned_rates.parquet` |
| `maze_name2mean_occupancy.json` | `frames.navigation.parquet` for all late sessions on each maze | Normalising place / place-direction occupancy in downstream maps |
| `maze_name2edge_transition_counts.json` | `trajectory_decisions.parquet` for all late sessions on each maze | Normalising edge-transition heatmaps |

The first two are derived directly from `processed_data/` so can be built immediately. The last two require `frames.navigation.parquet` and `trajectory_decisions.parquet` to already exist — hence the two-stage recipe above. On the first `save_analysis_info()` call you will see:

```
Warning: maze_name2mean_occupancy not saved. Data structure is None.
Warning: maze_name2edge_transition_counts not saved. Data structure is None.
```

These warnings disappear on the second call.

---

## 📦 What ends up in each `<subject>/<session>/` folder

`populate_analysis_data()` loops over the rows of `ANALYSIS_DATA_STRUCTURES_DF` (defined in [`populate_analysis_data.py`](populate_analysis_data.py)) and saves one parquet per row. All are gzipped pandas DataFrames; load them with `pandas.read_parquet`.

The order in the table below is the order they are built — later rows depend on earlier ones existing.

### Frame-level tables (one row per video frame, 60 Hz)

| File | Built by | Contents | Main downstream consumers |
|---|---|---|---|
| `frames.navigation.parquet` | [`get_navigation_df.py`](get_navigation_df.py) | Trial info, maze position, head direction, velocity/speed, `moving` flag, egocentric action + choice degree, cardinal movement direction, distance / steps / progress / angle to goal (geodesic, euclidean, future, manhattan, allocentric, egocentric) | Almost everything — joined with spike-rate tables to build every per-frame tuning analysis (Notebooks 4, 5, 6, 7) |
| `frames.spikeRates.parquet` | [`get_navigation_spike_dfs.py`](get_navigation_spike_dfs.py) | Per-cluster firing rate at each video frame (Gaussian-smoothed) | Distance, place-direction, ego-action, movement and place-cell tuning metrics; neGLM (Notebook 6) |
| `frames.spikeCounts.parquet` | [`get_navigation_spike_dfs.py`](get_navigation_spike_dfs.py) | Per-cluster integer spike count within each video frame | Decoding analyses that need raw counts (Notebook 4, 5, 6); Poisson-likelihood models |
| `frames.thetaSpikeCounts.parquet` | [`get_lfp_aligned_spike_counts.py`](get_lfp_aligned_spike_counts.py) | Per-cluster spike counts stratified into 12 theta-phase bins per frame (theta filtered 7–11 Hz from shank-averaged LFP) | Theta-phase locking + theta-stratified decoding (Notebook 7) |

### Event-aligned tables (one row per trial × cluster)

| File | Built by | Contents | Main downstream consumers |
|---|---|---|---|
| `trial_aligned_rates.parquet` | [`get_time_aligned_rates_dfs.py`](get_time_aligned_rates_dfs.py) | Per-cluster firing rates warped onto a canonical cue→reward→end-reward-consumption→trial-end timebase (one row per trial × cluster) | Trial-warped population dynamics, sequence plots, cluster heatmaps |
| `event_aligned_rates.parquet` | [`get_time_aligned_rates_dfs.py`](get_time_aligned_rates_dfs.py) | Per-cluster firing rates aligned to `cue`, `reward`, and `end_reward_consumption` events (±10 s, 25 Hz) | Event-locked PSTHs, reward-time representation analyses |

### Trajectory table (one row per maze-position visit)

| File | Built by | Contents | Main downstream consumers |
|---|---|---|---|
| `trajectory_decisions.parquet` | [`get_trajectory_decisions_dfs.py`](get_trajectory_decisions_dfs.py) | One row per node (and optionally edge) visit, with backtracking corrected, plus action, egocentric action, choice degree, distance and steps to goal | Behavioural analyses (Notebook 1); habit / strategy fits; transition-count normalisation in `analysis_info` |

### Cluster-level tuning metric tables (one row per cluster)

All five of these are built from `frames.navigation.parquet` + `frames.spikeRates.parquet` (theta one also needs `frames.thetaSpikeCounts.parquet`). Each contains split-half significance tests as well as fitted tuning parameters.

| File | Built by | Contents | Main downstream consumers |
|---|---|---|---|
| `clusters.distanceTuningMetrics.parquet` | [`get_distance_tuning_metrics_df.py`](get_distance_tuning_metrics_df.py) | Split-half correlation + p-value for distance-to-goal tuning; gamma / gaussian / polynomial curve fits (CV and non-CV) | Distance-to-goal cell analyses (Notebook 4) |
| `clusters.placeDirectionTuningMetrics.parquet` | [`get_place_direction_metrics_df.py`](get_place_direction_metrics_df.py) | Split-half correlation + p-value for place × head-direction tuning, plus a boolean `place_direction_tuned` flag | Place/place-direction analyses (Notebook 4) |
| `clusters.egocentricActionTuningMetrics.parquet` | [`get_action_tuning_metrics_df.py`](get_action_tuning_metrics_df.py) | Preferred egocentric action (left / forward / right) + factor + t_max; split-half correlations for all / free / forced actions and for free-vs-forced contrasts | Egocentric action coding analyses (Notebook 5) |
| `clusters.movementMetrics.parquet` | [`get_movement_metrics_df.py`](get_movement_metrics_df.py) | Split-half correlations and tuning-curve extrema for speed and 2D velocity | Movement-tuning controls in cell-type classification |
| `clusters.thetaModulationMetrics.parquet` | [`get_theta_mod_metrics_df.py`](get_theta_mod_metrics_df.py) | Von Mises fit (baseline, amp, kappa, mu, phase max/min, modulation depth, r²), Rayleigh test, split-half correlation, mean firing rate, anatomical region/voxel | Theta modulation analyses (Notebook 7) |

---

## 🧠 Tips, gotchas and likely problems

- **Disk and time.** A full run produces ~50 GB across all sessions. Without `parallel_jobs`, expect many hours; with 8 workers, expect 1–2 hours on a reasonable machine (the curve-fitting steps in the cluster tuning tables are the slow part).
- **Memory per worker.** Each parallel worker loads one full session's spikes + LFP into memory. The LFP step (`frames.thetaSpikeCounts.parquet`) is the most memory-hungry; if you OOM, drop `parallel_jobs` rather than disabling files.
- **`save_analysis_info()` is split-able.** Pass a list of structure names (see the quick-start recipe) to build only those — the two that depend on parquets (`maze_name2mean_occupancy`, `maze_name2edge_transition_counts`) need `populate_analysis_data` to have run first. Calling with no argument builds all four.
- **Skipped sessions are normal.** Sessions in `processed_data/` whose folder name does not end in `.maze` (e.g. rest sessions) are silently skipped because `session_types=["maze"]` for every row in `ANALYSIS_DATA_STRUCTURES_DF`.
- **Missing prerequisite files.** If a session is missing a processed file (e.g. no LFP), the relevant function raises `FileNotFoundError`, which is caught and logged as `FileNotFoundError: <function_name> failed for <path>`. The rest of the session continues. Look through the print output to spot sessions that are partially built.
- **Re-running is idempotent.** `populate_analysis_data` defaults to `overwrite=False` and skips any file that already exists. To force a rebuild of a single table, either delete the parquet file or pass `overwrite=True` (which rebuilds **all** selected `data_structures` for **all** selected sessions — be deliberate).
- **Order matters within a session.** The cluster tuning tables load `frames.navigation.parquet` and `frames.spikeRates.parquet` from `analysis_data/`, so if you build only those tuning tables on a session that has no navigation/spike-rate parquets yet, they will fail. Either rebuild from the top of the table or always run the full default set first.
- **For compute-heavy reruns, use SLURM.** The `jobs/` folder at the repo root has submission scripts that wrap `populate_analysis_data` for cluster execution.
