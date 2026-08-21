#!/bin/bash
#SBATCH --job-name=har-latefusion-10s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=./logs/late_fusion_10s_%j.out
#SBATCH --error=./logs/late_fusion_10s_%j.err

module load python/3.11.2

cd ~/multimodal-har-thesis
source .venv/bin/activate
mkdir -p logs results/v3/10s/models

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo "Window:  10s (500 samples at 50 Hz, 50% overlap)"
echo "Method:  Late Fusion — concat 384-dim → MLP → 15 classes"
echo "Init:    Warm-started from results/v3/10s/models/tinyhar/tsmixer/ast_fold*.pt"
echo ""

PYTHONPATH=. python jobs/train_late_fusion_10s.py

echo ""
echo "Finished: $(date)"
