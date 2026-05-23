"""
SLURM submission for hyperparameter sweeps over `get_sweep_update_df`.

Each entry in PARAM_SETS becomes one SLURM job. The `tag` is passed through to
`get_sweep_update_df(..., tag=...)`, which routes the cached parquet to
`RESULTS_PATH/theta_mod/sweep_update_tests/sweep_update_df{tag}.parquet`, so
runs never clobber each other or the canonical (`tag=None`) cache.

Usage:
    cd <repo root>
    python -c "from jobs.sweep_update.submit import submit_all; submit_all()"

@peterdoohan
"""

# %% Imports
import os
import subprocess
from pathlib import Path

from GridMaze.analysis.theta_mod.catch_update import RESULTS_DIR

# %% Param sets — edit this list to drive the sweep.
# Each item is (tag, kwargs_dict) where kwargs are forwarded to
# `get_sweep_update_df(...)` (which forwards them to
# `get_session_sweep_update_df(...)` per session).
PARAM_SETS = [
    ("_base", {}),
    ("_C1e-3", {"C": 1e-3}),
    ("_C1e-2", {"C": 1e-2}),
    ("_C1e-1", {"C": 1e-1}),
    ("_C1", {"C": 1.0}),
    ("_indep_pop", {"exclude_place_cells_from_decoder": True}),
    ("_C1e-2_indep", {"C": 1e-2, "exclude_place_cells_from_decoder": True}),
    ("_C1e-1_indep", {"C": 1e-1, "exclude_place_cells_from_decoder": True}),
    # max_steps_to_goal sweep at C=1e-1 (varies the on-task row filter in get_input_data)
    ("_C1e-1_msg10", {"C": 1e-1, "max_steps_to_goal": 10}),
    ("_C1e-1_msg16", {"C": 1e-1, "max_steps_to_goal": 16}),
    ("_C1e-1_msg24", {"C": 1e-1, "max_steps_to_goal": 24}),
    # phase-bin sensitivity sweep (place_trough_bins, distance_peak_bins, distance_trough_bins)
    # all phase-bin configs run with independent populations (exclude_place_cells_from_decoder=True)
    # so the two correlation axes use disjoint cells; tagged with `_indep` for clarity.
    # tags use hyphen-separated bin lists for uniform readability: pt<bins>, dp<bins>, dt<bins>
    # config A — current defaults
    (
        "_phases_pt1-2-3_dp3-4-5_dt9-10-11_indep",
        {
            "place_trough_bins": [1, 2, 3],
            "distance_peak_bins": [3, 4, 5],
            "distance_trough_bins": [9, 10, 11],
            "exclude_place_cells_from_decoder": True,
        },
    ),
    # config B — shifted +1, distance_trough wraps via the wrap-aware helper
    (
        "_phases_pt1-2-3_dp4-5-6_dt10-11-0_indep",
        {
            "place_trough_bins": [1, 2, 3],
            "distance_peak_bins": [4, 5, 6],
            "distance_trough_bins": [10, 11, 0],
            "exclude_place_cells_from_decoder": True,
        },
    ),
    # config C — place_trough shifted left by 1
    (
        "_phases_pt0-1-2_dp3-4-5_dt9-10-11_indep",
        {
            "place_trough_bins": [0, 1, 2],
            "distance_peak_bins": [3, 4, 5],
            "distance_trough_bins": [9, 10, 11],
            "exclude_place_cells_from_decoder": True,
        },
    ),
    # config D — 4-bin variant at n_training_phases=4; distance_trough wraps
    (
        "_phases_n4_pt0-1-2-3_dp3-4-5-6_dt9-10-11-0_indep",
        {
            "place_trough_bins": [0, 1, 2, 3],
            "distance_peak_bins": [3, 4, 5, 6],
            "distance_trough_bins": [9, 10, 11, 0],
            "n_training_phases": 4,
            "exclude_place_cells_from_decoder": True,
        },
    ),
    # short-distance × C-regularisation × indep grid:
    # max_steps_to_goal ∈ {10, 8}, C ∈ {1e-1, None, 1, 10, 100}, both indep + non-indep.
    # `C=None` forwards to `penalty=None` (unregularised LR).
    # `_C1e-1_msg10` already covered above; omitted here to avoid clobbering.
    ("_C1e-1_msg10_indep", {"C": 1e-1, "max_steps_to_goal": 10, "exclude_place_cells_from_decoder": True}),
    ("_C1_msg10", {"C": 1.0, "max_steps_to_goal": 10}),
    ("_C1_msg10_indep", {"C": 1.0, "max_steps_to_goal": 10, "exclude_place_cells_from_decoder": True}),
    ("_C10_msg10", {"C": 10.0, "max_steps_to_goal": 10}),
    ("_C10_msg10_indep", {"C": 10.0, "max_steps_to_goal": 10, "exclude_place_cells_from_decoder": True}),
    ("_C100_msg10", {"C": 100.0, "max_steps_to_goal": 10}),
    ("_C100_msg10_indep", {"C": 100.0, "max_steps_to_goal": 10, "exclude_place_cells_from_decoder": True}),
    ("_Cnone_msg10", {"C": None, "max_steps_to_goal": 10}),
    ("_Cnone_msg10_indep", {"C": None, "max_steps_to_goal": 10, "exclude_place_cells_from_decoder": True}),
    ("_C1e-1_msg8", {"C": 1e-1, "max_steps_to_goal": 8}),
    ("_C1e-1_msg8_indep", {"C": 1e-1, "max_steps_to_goal": 8, "exclude_place_cells_from_decoder": True}),
    ("_C1_msg8", {"C": 1.0, "max_steps_to_goal": 8}),
    ("_C1_msg8_indep", {"C": 1.0, "max_steps_to_goal": 8, "exclude_place_cells_from_decoder": True}),
    ("_C10_msg8", {"C": 10.0, "max_steps_to_goal": 8}),
    ("_C10_msg8_indep", {"C": 10.0, "max_steps_to_goal": 8, "exclude_place_cells_from_decoder": True}),
    ("_C100_msg8", {"C": 100.0, "max_steps_to_goal": 8}),
    ("_C100_msg8_indep", {"C": 100.0, "max_steps_to_goal": 8, "exclude_place_cells_from_decoder": True}),
    ("_Cnone_msg8", {"C": None, "max_steps_to_goal": 8}),
    ("_Cnone_msg8_indep", {"C": None, "max_steps_to_goal": 8, "exclude_place_cells_from_decoder": True}),
    ("_C1e-1_msg12", {"C": 1e-1, "max_steps_to_goal": 12}),
    ("_C1e-1_msg12_indep", {"C": 1e-1, "max_steps_to_goal": 12, "exclude_place_cells_from_decoder": True}),
    ("_C1_msg12", {"C": 1.0, "max_steps_to_goal": 12}),
    ("_C1_msg12_indep", {"C": 1.0, "max_steps_to_goal": 12, "exclude_place_cells_from_decoder": True}),
    ("_C10_msg12", {"C": 10.0, "max_steps_to_goal": 12}),
    ("_C10_msg12_indep", {"C": 10.0, "max_steps_to_goal": 12, "exclude_place_cells_from_decoder": True}),
    ("_C100_msg12", {"C": 100.0, "max_steps_to_goal": 12}),
    ("_C100_msg12_indep", {"C": 100.0, "max_steps_to_goal": 12, "exclude_place_cells_from_decoder": True}),
    ("_Cnone_msg12", {"C": None, "max_steps_to_goal": 12}),
    ("_Cnone_msg12_indep", {"C": None, "max_steps_to_goal": 12, "exclude_place_cells_from_decoder": True}),
]

# joblib-level parallelism inside each SLURM job (sessions processed concurrently).
# Stay well below the per-job CPU allocation below.
N_JOBS_INSIDE = 8

# SLURM resource defaults (override per-job by editing get_SLURM_script).
SBATCH_CPUS = 16
SBATCH_MEM = "128GB"
SBATCH_TIME = "24:00:00"
SBATCH_PARTITION = "gpu_lowp"

# %% Functions


def submit_all(force=False):
    """Submit one SLURM job per entry in PARAM_SETS.

    By default, skips any tag that is already on disk or already in the SLURM
    queue (running or pending). Cache-skip prevents recompute clobbering — the
    SLURM body calls `get_sweep_update_df(save=True, ...)`. Queue-skip prevents
    duplicate submissions for in-flight runs whose parquet hasn't landed yet.

    Pass `force=True` to (re)submit every entry regardless of state.
    """
    sweep_dir = RESULTS_DIR / "sweep_update_tests"
    queued = _running_or_pending_sweep_tags() if not force else set()
    submitted, cache_skipped, queue_skipped = [], [], []
    for tag, kwargs in PARAM_SETS:
        cache_path = sweep_dir / f"sweep_update_df{tag}.parquet"
        if cache_path.exists() and not force:
            print(f"skipping tag={tag!r} — cached at {cache_path.name}")
            cache_skipped.append(tag)
            continue
        if tag in queued:
            print(f"skipping tag={tag!r} — already in SLURM queue")
            queue_skipped.append(tag)
            continue
        script_path = get_SLURM_script(tag, kwargs)
        print(f"submitting tag={tag!r} kwargs={kwargs}")
        os.system(f"chmod +x {script_path}")
        os.system(f"sbatch {script_path}")
        submitted.append(tag)
    print(
        f"\nsubmitted {len(submitted)}; "
        f"skipped {len(cache_skipped)} cached, {len(queue_skipped)} queued"
    )
    return {"submitted": submitted, "cache_skipped": cache_skipped, "queue_skipped": queue_skipped}


def _running_or_pending_sweep_tags():
    """Return the set of tags whose `sweep_update{tag}` job is currently in the
    SLURM queue (R, PD, CG, etc.) for $USER. Parses `squeue` output by job name."""
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return set()
    tags = set()
    prefix = "sweep_update"
    for line in out.splitlines():
        name = line.strip()
        if name.startswith(prefix):
            # job_name is f"sweep_update{tag}" — recover tag (leading underscore included)
            tags.add(name[len(prefix):])
    return tags


def get_SLURM_script(tag, kwargs):
    """Build (and write to disk) a `.sh` that runs `get_sweep_update_df` with
    the given tag and kwargs. Returns the script path."""
    job_name = f"sweep_update{tag}"
    kwargs_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=jobs/sweep_update/out/{job_name}.out
#SBATCH --error=jobs/sweep_update/err/{job_name}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={SBATCH_CPUS}
#SBATCH -p {SBATCH_PARTITION}
#SBATCH --mem={SBATCH_MEM}
#SBATCH --time={SBATCH_TIME}

module load miniconda
conda deactivate
conda activate goalNav_mEC

python <<EOF
from GridMaze.analysis.theta_mod import catch_update as cu
cu.get_sweep_update_df(save=True, tag={tag!r}, n_jobs={N_JOBS_INSIDE}, {kwargs_str})
EOF
"""
    script_path = Path(f"jobs/sweep_update/slurm/{job_name}.sh")
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w") as f:
        f.write(script)
    return str(script_path)
