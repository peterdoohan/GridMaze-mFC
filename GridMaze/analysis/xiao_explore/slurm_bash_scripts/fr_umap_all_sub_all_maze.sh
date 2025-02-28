#!/bin/bash
#
#SBATCH --job-name=hyper
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -t 0-0:10
#SBATCH -o /ceph/behrens/peter_doohan/goalNav_mFC_refactor/code/GridMaze/analysis/xiao_explore/slurm_reports/slurm.%N.%j.out # STDOUT
#SBATCH -e /ceph/behrens/peter_doohan/goalNav_mFC_refactor/code/GridMaze/analysis/xiao_explore/slurm_reports/slurm.%N.%j.err # STDERR
#SBATCH --array=2-157

config=/ceph/behrens/peter_doohan/goalNav_mFC_refactor/code/GridMaze/analysis/xiao_explore/hyperparams/hyperparam_basic/maze_subject_session.txt
maze_number=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID 'NR==ArrayTaskID {print $1}' $config)
subject_ID=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID 'NR==ArrayTaskID {print $2}' $config)
session=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID 'NR==ArrayTaskID {print $3}' $config)

hostname
source ~/.bashrc
conda activate MazeRL
cd /ceph/behrens/peter_doohan/goalNav_mFC_refactor/code
export PYTHONPATH=/ceph/behrens/peter_doohan/goalNav_mFC_refactor/code:$PYTHONPATH

python /ceph/behrens/peter_doohan/goalNav_mFC_refactor/code/GridMaze/analysis/xiao_explore/fr_umap.py --maze=${maze_number} --subject=${subject_ID} --session=${session}
