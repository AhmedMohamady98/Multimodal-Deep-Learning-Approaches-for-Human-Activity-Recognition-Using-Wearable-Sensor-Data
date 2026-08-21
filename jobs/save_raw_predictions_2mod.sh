#!/bin/bash
#SBATCH --job-name=har-save-2mod-preds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=./logs/save_2mod_preds_%j.out
#SBATCH --error=./logs/save_2mod_preds_%j.err

# Dump per-window softmax for all 9 2-mod ablation checkpoints
# (Late Fusion / GMF / CMA × 3 folds). Run AFTER the ablation training
# array 11655400_[0-8] completes.

module load python/3.11.2

cd ~/multimodal-har-thesis
source .venv/bin/activate
mkdir -p logs results/v3/10s/predictions

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo ""

PYTHONPATH=. python jobs/save_raw_predictions_2mod.py

echo ""
echo "Finished: $(date)"
