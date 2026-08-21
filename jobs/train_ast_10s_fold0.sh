#!/bin/bash
#SBATCH --job-name=har-ast-10s-f0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=./logs/ast_10s_fold0_%j.out
#SBATCH --error=./logs/ast_10s_fold0_%j.err

module load python/3.11.2

cd ~/multimodal-har-thesis
source .venv/bin/activate
mkdir -p logs results/v3/10s/models

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo "Fold:    0  (Participants 1–7)"
echo "Window:  10s (500 samples at 50 Hz, 50% overlap)"
echo "AST fill: ~100% spectrogram (matches AudioSet pretraining)"
echo ""

PYTHONPATH=. python jobs/train_ast_10s.py --fold 0

echo ""
echo "Finished: $(date)"
