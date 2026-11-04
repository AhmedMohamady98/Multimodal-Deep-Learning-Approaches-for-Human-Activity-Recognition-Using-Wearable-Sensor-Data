#!/bin/bash
#SBATCH --job-name=eval-cma-10s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=./logs/eval_cma_10s_%j.out
#SBATCH --error=./logs/eval_cma_10s_%j.err

module load python/3.11.2

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
mkdir -p logs results/cma/figures results/cma/tables

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo ""

PYTHONPATH=. python jobs/evaluate_cma_10s.py

echo ""
echo "Finished: $(date)"
