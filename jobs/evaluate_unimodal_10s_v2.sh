#!/bin/bash
#SBATCH --job-name=har-eval-uni-v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=./logs/eval_unimodal_10s_v2_%j.out
#SBATCH --error=./logs/eval_unimodal_10s_v2_%j.err

module load python/3.11.2

cd ~/multimodal-har-thesis
source .venv/bin/activate
mkdir -p logs results/v3/10s/figures results/v3/10s/tables

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo "v2: AST audio SR bug fixed — resampling 44.1kHz → 16kHz"
echo ""

PYTHONPATH=. python jobs/evaluate_unimodal_10s_v2.py

echo ""
echo "Finished: $(date)"
