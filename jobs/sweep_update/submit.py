"""
SLURM submission for hyperparameter sweeps over `get_sweep_update_df`.

PARAM_SETS is the Cartesian product of four axes:

    C ∈ {0.1, 1, 10}                                    decoder regularisation
    moving_only ∈ {True, False}                         theta-locked row filter
    max_steps_to_goal ∈ {8, 12, 16, 20, 24, 28}         on-task row filter
    phase config ∈ {canonical, shifted}                 peak/trough readout bins

→ 3 × 2 × 6 × 2 = 72 jobs.

Tags follow the existing convention: `_C{val}_msg{N}_{move|nomove}[_shifted]`.
Each tag is passed through to `get_sweep_update_df(..., tag=...)`, which routes
the cached parquet to
`RESULTS_PATH/theta_mod/sweep_update_tests/sweep_update_df{tag}.parquet`.

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

# %% Sweep axes
C_VALUES = [1e-1, 1.0, 10.0]
MOVING_VALUES = [True, False]
MAX_STEPS_VALUES = [8, 12, 16, 20, 24, 28]
# (suffix, distance_peak_bins, distance_trough_bins). Empty suffix == canonical.
PHASE_CONFIGS = [
    ("", [3, 4, 5], [9, 10, 11]),                # canonical
    ("shifted", [4, 5, 6], [10, 11, 0]),         # +1 shifted, distance_trough wraps via wrap-aware helper
]


def _format_C(C):
    """Format C for tags: 0.1 → 'C1e-1', 1.0 → 'C1', 10.0 → 'C10'."""
    if C == 1e-1:
        return "C1e-1"
    if C == 1.0:
        return "C1"
    if C == 10.0:
        return "C10"
    raise ValueError(f"unrecognised C={C!r} — extend _format_C")


def _build_param_sets():
    out = []
    for C in C_VALUES:
        for moving in MOVING_VALUES:
            for msg in MAX_STEPS_VALUES:
                for phase_suffix, dp, dt in PHASE_CONFIGS:
                    tag_parts = [_format_C(C), f"msg{msg}", "move" if moving else "nomove"]
                    if phase_suffix:
                        tag_parts.append(phase_suffix)
                    tag = "_" + "_".join(tag_parts)
                    kwargs = {
                        "C": C,
                        "moving_only": moving,
                        "max_steps_to_goal": msg,
                        "distance_peak_bins": dp,
                        "distance_trough_bins": dt,
                    }
                    out.append((tag, kwargs))
    return out


PARAM_SETS = _build_param_sets()

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
