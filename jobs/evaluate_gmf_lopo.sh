#!/bin/bash
#SBATCH --job-name=har-gmf-lopo-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=./logs/evaluate_gmf_lopo_%j.out
#SBATCH --error=./logs/evaluate_gmf_lopo_%j.err

# Run after all train_gmf_lopo array tasks finish:
#   sbatch --dependency=afterok:<array_job_id> jobs/evaluate_gmf_lopo.sh

module load python/3.11.2

cd ~/multimodal-har-thesis
source .venv/bin/activate
mkdir -p logs results/v3/10s/lopo

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo ""

PYTHONPATH=. python jobs/evaluate_gmf_lopo.py

echo ""
echo "Finished: $(date)"
