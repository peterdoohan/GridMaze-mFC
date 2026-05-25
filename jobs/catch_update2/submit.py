"""
SLURM submission for the simplified catch_update2 pipeline.

The pipeline is now a single canonical configuration (Hilbert phase, argmax
tuned pools, per-cycle D3↔P4 metrics), so this submits ONE job that runs
`get_catch_update2_df(save=True)` over every subject × maze × late day and
caches the result to
`RESULTS_PATH/theta_mod/catch_update2/runs/catch_update2_df{tag}.parquet`.

Usage:
    cd <repo root>
    python -c "from jobs.catch_update2.submit import submit; submit()"

@peterdoohan
"""

# %% Imports
import os
import subprocess
from pathlib import Path

from GridMaze.analysis.theta_mod.catch_update2 import RESULTS_DIR

# joblib-level parallelism inside the SLURM job (sessions processed concurrently)
N_JOBS_INSIDE = 8

# SLURM resource defaults
SBATCH_CPUS = 16
SBATCH_MEM = "128GB"
SBATCH_TIME = "24:00:00"
SBATCH_PARTITION = "gpu_lowp"


def submit(tag="", force=False):
    """Submit the single canonical catch_update2 run.

    Skips if the parquet already exists or the job is already in the SLURM queue,
    unless `force=True`. `tag` (e.g. "_test") suffixes the cache filename and job
    name so an alternative run doesn't clobber the canonical one.
    """
    job_name = f"catch_update2{tag}"
    cache_path = RESULTS_DIR / "runs" / f"catch_update2_df{tag}.parquet"
    if cache_path.exists() and not force:
        print(f"skipping {job_name} — cached at {cache_path.name}")
        return
    if (not force) and job_name in _running_or_pending_jobs():
        print(f"skipping {job_name} — already in SLURM queue")
        return
    script_path = get_SLURM_script(job_name, tag)
    print(f"submitting {job_name}")
    os.system(f"chmod +x {script_path}")
    os.system(f"sbatch {script_path}")


def _running_or_pending_jobs():
    """Set of catch_update2 job names currently in the SLURM queue for $USER."""
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return set()
    return {line.strip() for line in out.splitlines() if line.strip().startswith("catch_update2")}


def get_SLURM_script(job_name, tag):
    """Build (and write) a `.sh` that runs `get_catch_update2_df(save=True)`.
    Returns the script path."""
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
cu2.get_catch_update2_df(save=True, n_jobs={N_JOBS_INSIDE}, tag={tag!r})
EOF
"""
    script_path = Path(f"jobs/catch_update2/slurm/{job_name}.sh")
    script_path.parent.mkdir(parents=True, exist_ok=True)
    for sub in ("out", "err"):
        Path(f"jobs/catch_update2/{sub}").mkdir(parents=True, exist_ok=True)
    with open(script_path, "w") as f:
        f.write(script)
    return str(script_path)
