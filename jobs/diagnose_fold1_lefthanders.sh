#!/bin/bash
#SBATCH --job-name=har-lh-diag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=./logs/lefthander_diag_%j.out
#SBATCH --error=./logs/lefthander_diag_%j.err

module load python/3.11.2

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
mkdir -p logs results/lefthander

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo "Left-hander analysis — fold 1 all models"
echo ""

PYTHONPATH=. python jobs/diagnose_fold1_lefthanders.py --models all

echo ""
echo "Finished: $(date)"
