#!/bin/bash
#SBATCH --job-name=har-pipeline-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=./logs/pipeline_test_%j.out
#SBATCH --error=./logs/pipeline_test_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11.2

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate

mkdir -p logs

# ── Pipeline test on 3 participants ───────────────────────────────────────────
echo "=== Pipeline test: 3 participants ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo ""

PYTHONPATH=. python jobs/test_pipeline.py
