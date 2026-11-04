#!/bin/bash
#SBATCH --job-name=har-eval-fusion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=./logs/evaluate_fusion_all_10s_%j.out
#SBATCH --error=./logs/evaluate_fusion_all_10s_%j.err

module load python/3.11.2

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
mkdir -p logs results/fusion/tables

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo ""

PYTHONPATH=. python jobs/evaluate_fusion_all_10s.py

echo ""
echo "Finished: $(date)"
