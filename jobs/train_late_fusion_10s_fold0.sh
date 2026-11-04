#!/bin/bash
#SBATCH --job-name=har-lf-10s-f0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=./logs/late_fusion_10s_fold0_%j.out
#SBATCH --error=./logs/late_fusion_10s_fold0_%j.err

module load python/3.11.2

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
mkdir -p logs results/models

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo "Fold:    0  (test participants 1-7)"
echo ""

PYTHONPATH=. python jobs/train_late_fusion_10s.py --fold 0

echo ""
echo "Finished: $(date)"
