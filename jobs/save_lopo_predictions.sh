#!/bin/bash
#SBATCH --job-name=har-lopo-preds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=./logs/save_lopo_predictions_%j.out
#SBATCH --error=./logs/save_lopo_predictions_%j.err

# Run after evaluate_gmf_lopo.sh completes:
#   sbatch jobs/save_lopo_predictions.sh

module load python/3.11.2

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
mkdir -p logs results/lopo

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo ""

PYTHONPATH=. python jobs/save_lopo_predictions.py

echo ""
echo "Finished: $(date)"
