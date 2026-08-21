#!/bin/bash
#SBATCH --job-name=eval-gmf-10s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=./logs/eval_gmf_10s_%j.out
#SBATCH --error=./logs/eval_gmf_10s_%j.err

module load python/3.11.2

cd ~/multimodal-har-thesis
source .venv/bin/activate
mkdir -p logs results/gmf/figures results/gmf/tables

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo ""

PYTHONPATH=. python jobs/evaluate_gmf_10s.py

echo ""
echo "Finished: $(date)"
