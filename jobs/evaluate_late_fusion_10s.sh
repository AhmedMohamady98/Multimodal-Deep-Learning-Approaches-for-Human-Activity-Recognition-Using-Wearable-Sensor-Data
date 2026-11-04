#!/bin/bash
#SBATCH --job-name=har-eval-lf-10s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=./logs/eval_late_fusion_10s_%j.out
#SBATCH --error=./logs/eval_late_fusion_10s_%j.err

module load python/3.11.2

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
mkdir -p logs results/figures/late_fusion results/tables

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo "Window:  10s (500 samples at 50 Hz, 50% overlap)"
echo "Method:  Late Fusion — TinyHAR + AST + TSMixer"
echo ""

PYTHONPATH=. python jobs/evaluate_late_fusion_10s.py

echo ""
echo "Finished: $(date)"
