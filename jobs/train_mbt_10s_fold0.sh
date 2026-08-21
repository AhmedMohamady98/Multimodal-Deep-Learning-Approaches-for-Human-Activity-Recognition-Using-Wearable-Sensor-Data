#!/bin/bash
#SBATCH --job-name=har-mbt-10s-f0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=./logs/mbt_10s_fold0_%j.out
#SBATCH --error=./logs/mbt_10s_fold0_%j.err

module load python/3.11.2

cd ~/multimodal-har-thesis
source .venv/bin/activate
mkdir -p logs results/v3/10s/models

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo "Fold:    0  (test participants 1-7)"
echo ""

PYTHONPATH=. python jobs/train_mbt_10s.py --fold 0

echo ""
echo "Finished: $(date)"
