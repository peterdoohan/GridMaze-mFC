"""
SLURM submission for the catch_update2 robustness batch.

A SMALL robustness set (not a full sweep), so we can check the distance→place
result is stable across the two design choices most likely to matter:

    phase_method ∈ {hilbert, waveform}     non-sinusoidal-theta artifact check
    pool_method  ∈ {argmax, median}        pool-definition robustness
                                           (median ⇒ parity with catch_update)

→ 2 × 2 = 4 jobs. Each job runs `get_catch_update2_df(save=True, ...)` over
every subject × maze × late day and caches one parquet to
`RESULTS_PATH/theta_mod/catch_update2/runs/catch_update2_df_{phase}{tag}.parquet`.

Usage:
    cd <repo root>
    python -c "from jobs.catch_update2.submit import submit_all; submit_all()"

@peterdoohan (catch_update2 rebuild)
"""

# %% Imports
import os
import subprocess
from pathlib import Path

from GridMaze.analysis.theta_mod.catch_update2 import RESULTS_DIR

# %% Sweep axes (deliberately small)
PHASE_METHODS = ["hilbert", "waveform"]
POOL_METHODS = ["argmax", "median"]

# joblib-level parallelism inside each SLURM job (sessions processed concurrently)
N_JOBS_INSIDE = 8

# SLURM resource defaults
SBATCH_CPUS = 16
SBATCH_MEM = "128GB"
SBATCH_TIME = "24:00:00"
SBATCH_PARTITION = "gpu_lowp"


def _build_param_sets():
    """One entry per (phase_method, pool_method). `tag` encodes the pool method;
    the phase method is handled by `get_catch_update2_df`'s own filename suffix,
    so the cached parquet is catch_update2_df_{phase}{tag}.parquet."""
    out = []
    for phase_method in PHASE_METHODS:
        for pool_method in POOL_METHODS:
            tag = f"_{pool_method}"
            kwargs = {"phase_method": phase_method, "pool_method": pool_method, "tag": tag}
            out.append((phase_method, tag, kwargs))
    return out


PARAM_SETS = _build_param_sets()


def submit_all(force=False):
    """Submit one SLURM job per (phase_method, pool_method).

    Skips any condition already cached on disk or already in the SLURM queue,
    unless `force=True`.
    """
    runs_dir = RESULTS_DIR / "runs"
    queued = _running_or_pending_tags() if not force else set()
    submitted, cache_skipped, queue_skipped = [], [], []
    for phase_method, tag, kwargs in PARAM_SETS:
        job_name = f"catch_update2_{phase_method}{tag}"
        cache_path = runs_dir / f"catch_update2_df_{phase_method}{tag}.parquet"
        if cache_path.exists() and not force:
            print(f"skipping {job_name} — cached at {cache_path.name}")
            cache_skipped.append(job_name)
            continue
        if job_name in queued:
            print(f"skipping {job_name} — already in SLURM queue")
            queue_skipped.append(job_name)
            continue
        script_path = get_SLURM_script(job_name, kwargs)
        print(f"submitting {job_name} kwargs={kwargs}")
        os.system(f"chmod +x {script_path}")
        os.system(f"sbatch {script_path}")
        submitted.append(job_name)
    print(
        f"\nsubmitted {len(submitted)}; skipped {len(cache_skipped)} cached, {len(queue_skipped)} queued"
    )
    return {"submitted": submitted, "cache_skipped": cache_skipped, "queue_skipped": queue_skipped}


def _running_or_pending_tags():
    """Set of catch_update2 job names currently in the SLURM queue for $USER."""
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return set()
    return {line.strip() for line in out.splitlines() if line.strip().startswith("catch_update2_")}


def get_SLURM_script(job_name, kwargs):
    """Build (and write) a `.sh` that runs `get_catch_update2_df(save=True, ...)`
    with the given kwargs. Returns the script path."""
    kwargs_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=jobs/catch_update2/out/{job_name}.out
#SBATCH --error=jobs/catch_update2/err/{job_name}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={SBATCH_CPUS}
#SBATCH -p {SBATCH_PARTITION}
#SBATCH --mem={SBATCH_MEM}
#SBATCH --time={SBATCH_TIME}

module load miniconda
conda deactivate
conda activate goalNav_mEC

python <<EOF
from GridMaze.analysis.theta_mod import catch_update2 as cu2
cu2.get_catch_update2_df(save=True, n_jobs={N_JOBS_INSIDE}, {kwargs_str})
EOF
"""
    script_path = Path(f"jobs/catch_update2/slurm/{job_name}.sh")
    script_path.parent.mkdir(parents=True, exist_ok=True)
    for sub in ("out", "err"):
        Path(f"jobs/catch_update2/{sub}").mkdir(parents=True, exist_ok=True)
    with open(script_path, "w") as f:
        f.write(script)
    return str(script_path)
