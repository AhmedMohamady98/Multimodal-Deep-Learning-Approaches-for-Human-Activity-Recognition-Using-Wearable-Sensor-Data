#!/bin/bash
#SBATCH --job-name=har-cma-10s-f1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=./logs/cma_10s_fold1_%j.out
#SBATCH --error=./logs/cma_10s_fold1_%j.err

module load python/3.11.2

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
mkdir -p logs results/models

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo "Fold:    1  (test participants 8-14)"
echo ""

PYTHONPATH=. python jobs/train_cma_10s.py --fold 1

echo ""
echo "Finished: $(date)"
