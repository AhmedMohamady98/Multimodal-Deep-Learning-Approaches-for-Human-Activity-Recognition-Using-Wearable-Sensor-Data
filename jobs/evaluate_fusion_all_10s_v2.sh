#!/bin/bash
#SBATCH --job-name=har-eval-fusion-v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=./logs/evaluate_fusion_all_10s_v2_%j.out
#SBATCH --error=./logs/evaluate_fusion_all_10s_v2_%j.err

module load python/3.11.2

cd ~/multimodal-har-thesis
source .venv/bin/activate
mkdir -p logs results/v3/10s/fusion/tables

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo ""

PYTHONPATH=. python jobs/evaluate_fusion_all_10s_v2.py

echo ""
echo "Finished: $(date)"
