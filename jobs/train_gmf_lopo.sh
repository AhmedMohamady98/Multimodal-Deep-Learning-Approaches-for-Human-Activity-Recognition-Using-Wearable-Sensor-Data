#!/bin/bash
#SBATCH --job-name=har-gmf-lopo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=1-20
#SBATCH --output=./logs/gmf_lopo_p%a_%j.out
#SBATCH --error=./logs/gmf_lopo_p%a_%j.err

# Submit all 20 LOPO folds as a job array:
#   sbatch jobs/train_gmf_lopo.sh
# Each array task trains with one participant held out.
# SLURM_ARRAY_TASK_ID = participant ID (1–20).

module load python/3.11.2

cd ~/multimodal-har-thesis
source .venv/bin/activate
mkdir -p logs results/v3/10s/lopo/models

echo "Job ID:       $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Node:         $SLURMD_NODENAME"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started:      $(date)"
echo "Participant:  P$(printf '%02d' $SLURM_ARRAY_TASK_ID) held out"
echo ""

PYTHONPATH=. python jobs/train_gmf_lopo.py --participant $SLURM_ARRAY_TASK_ID

echo ""
echo "Finished: $(date)"
