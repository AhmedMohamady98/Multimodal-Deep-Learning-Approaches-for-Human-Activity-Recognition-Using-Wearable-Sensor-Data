#!/bin/bash
#SBATCH --job-name=har-abl-2mod
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-8%4
#SBATCH --ntasks-per-node=1
#SBATCH --output=./logs/abl_2mod_t%a_%j.out
#SBATCH --error=./logs/abl_2mod_t%a_%j.err

# --array=0-8%4    cap at 4 concurrent tasks (matches GPU-node count, avoids
#                  node sharing that triggered kernel OOM in run 11655400)
# --ntasks-per-node=1   one of our tasks per physical node

# Submit all 9 2-mod ablation runs as a SLURM array:
#   sbatch jobs/train_ablation_2mod_10s.sh
#
# Array task ID encodes (method, fold) as task_id = method_idx * 3 + fold:
#   task 0  → late_fusion fold 0
#   task 1  → late_fusion fold 1
#   task 2  → late_fusion fold 2
#   task 3  → gmf         fold 0
#   task 4  → gmf         fold 1
#   task 5  → gmf         fold 2
#   task 6  → cma         fold 0
#   task 7  → cma         fold 1
#   task 8  → cma         fold 2

module load python/3.11.2

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
mkdir -p logs results/models

TASK=$SLURM_ARRAY_TASK_ID
METHODS=(late_fusion gmf cma)
M_IDX=$(( TASK / 3 ))
FOLD=$(( TASK % 3 ))
METHOD=${METHODS[$M_IDX]}

echo "Job ID:      $SLURM_JOB_ID (array task $TASK)"
echo "Node:        $SLURMD_NODENAME"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Method:      $METHOD"
echo "Fold:        $FOLD"
echo "Started:     $(date)"
echo ""

PYTHONPATH=. python jobs/train_ablation_2mod_10s.py --method "$METHOD" --fold "$FOLD"

echo ""
echo "Finished: $(date)"
