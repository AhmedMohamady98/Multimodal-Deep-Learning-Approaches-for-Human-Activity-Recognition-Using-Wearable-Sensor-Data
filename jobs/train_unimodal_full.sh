#!/bin/bash
#SBATCH --job-name=har-unimodal-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=./logs/unimodal_full_%j.out
#SBATCH --error=./logs/unimodal_full_%j.err

module load python/3.11.2

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
mkdir -p logs results/models

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo ""

PYTHONPATH=. python jobs/train_unimodal_full.py

echo ""
echo "Finished: $(date)"
