#!/bin/bash
#SBATCH --job-name=har-raw-preds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=./logs/save_raw_predictions_%j.out
#SBATCH --error=./logs/save_raw_predictions_%j.err

module load python/3.11.2

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
mkdir -p logs results/predictions

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo ""

PYTHONPATH=. python jobs/save_raw_predictions.py

echo ""
echo "Finished: $(date)"
