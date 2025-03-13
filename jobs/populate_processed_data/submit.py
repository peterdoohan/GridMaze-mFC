"""
Script for parallelised populate processed data processing
@peterdoohan
"""

# %% Imports
import os
from GridMaze.preprocessing import get_data_directory as dd

# %% Global Variables


# %% Functions


def HPC_populate_processed_data(data_streams=["session_info", "pycontrol", "video", "spikes", "lfp"], overwrite=False):
    """ """
    data_directory = dd.get_sessions_data_directory()
    for _, session_dir in data_directory.iterrows():
        session_name = f"{session_dir.subject_ID}_{session_dir.date.isoformat()}.{session_dir.session_type}"
        script = get_SLURM_script(session_dir, data_streams, overwrite)
        print(f"submitting {session_name} to HPC")
        os.system(f"chmod +x {script}")
        os.system(f"sbatch {script}")
    return print("All sessions submitted to HPC")


def get_SLURM_script(session_dir, data_streams, overwrite):
    """"""
    session_name = f"{session_dir.subject_ID}_{session_dir.date.isoformat()}.{session_dir.session_type}"
    script = f"""#!/bin/bash
#SBATCH --job-name={session_name}
#SBATCH --output=jobs/populate_processed_data/out/{session_name}.out
#SBATCH --error=jobs/populate_processed_data/err/{session_name}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=00:30:00

module load miniconda
conda deactivate
conda deactivate
conda deactivate
conda activate goalNav_mEC

python <<EOF
try:
    from GridMaze.preprocessing import populate_processed_data as ppd
    ppd.populate_session_processed_data('{session_dir.subject_ID}', '{session_dir.date.isoformat()}', '{session_dir.session_type}', {data_streams}, {overwrite})
except:
    print('ERROR processing {session_name}')
EOF
"""
    script_path = f"jobs/populate_processed_data/slurm/{session_name}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    return script_path
