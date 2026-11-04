#!/bin/bash
#SBATCH --job-name=har-10s-imu-hum
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=./logs/tinyhar_tsmixer_10s_%j.out
#SBATCH --error=./logs/tinyhar_tsmixer_10s_%j.err

module load python/3.11.2

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
mkdir -p logs results/models

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo "Window:  10s (500 samples at 50 Hz, 50% overlap)"
echo ""

PYTHONPATH=. python jobs/train_tinyhar_tsmixer_10s.py

echo ""
echo "Finished: $(date)"
