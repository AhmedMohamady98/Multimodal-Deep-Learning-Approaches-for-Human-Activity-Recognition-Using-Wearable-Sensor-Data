#!/bin/bash
#SBATCH --job-name=har-eval-10s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=./logs/eval_10s_%j.out
#SBATCH --error=./logs/eval_10s_%j.err

module load python/3.11.2

cd ~/multimodal-har-thesis
source .venv/bin/activate
mkdir -p logs results/v3/10s/figures results/v3/10s/tables

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo "Window:  10s (500 samples at 50 Hz, 50% overlap)"
echo ""

PYTHONPATH=. python jobs/evaluate_unimodal_10s.py

echo ""
echo "Finished: $(date)"
