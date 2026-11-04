#!/bin/bash
#SBATCH --job-name=har-cls-f2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=./logs/cls_transformer_10s_fold2_%j.out
#SBATCH --error=./logs/cls_transformer_10s_fold2_%j.err

module load python/3.11.2

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
mkdir -p logs results/models

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo "Fold:    2  (test participants 15-20)"
echo ""

PYTHONPATH=. python jobs/train_cls_transformer_10s.py --fold 2

echo ""
echo "Finished: $(date)"
