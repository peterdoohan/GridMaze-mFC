#!/bin/bash
#
#SBATCH --job-name=hyper
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -t 0-1:00
#SBATCH -o /ceph/behrens/peter_doohan/goalNav_mFC_refactor/code/GridMaze/analysis/xiao_explore/slurm_reports/slurm.%N.%j.out # STDOUT
#SBATCH -e /ceph/behrens/peter_doohan/goalNav_mFC_refactor/code/GridMaze/analysis/xiao_explore/slurm_reports/slurm.%N.%j.err # STDERR
#SBATCH --array=0-9

hostname
source ~/.bashrc
conda activate MazeRL
cd /ceph/behrens/peter_doohan/goalNav_mFC_refactor/code
export PYTHONPATH=/ceph/behrens/peter_doohan/goalNav_mFC_refactor/code:$PYTHONPATH
echo "Current directory: $(pwd)"

python /ceph/behrens/peter_doohan/goalNav_mFC_refactor/code/GridMaze/analysis/xiao_explore/fr_umap.py --session="$SLURM_ARRAY_TASK_ID"
