#!/bin/bash
#SBATCH --job-name=gmf-gates
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=./logs/gmf_gates_%j.out
#SBATCH --error=./logs/gmf_gates_%j.err

module load python/3.11.2

# ~/.bashrc points HF_HOME at /work/ws-tmp/USER-sci-humanizer/hf_cache, a
# workspace that has since expired. Any job inheriting it dies when the AST
# backbone cannot be resolved. Pin the cache to a persistent home location so
# this job does not depend on the login shell's environment.
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1          # weights are already cached; never hit the network

cd ~/multimodal-har-thesis
source .venv/bin/activate
mkdir -p logs results/v3/10s/gates

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo ""

PYTHONPATH=. python jobs/analyze_gmf_gates.py

echo ""
echo "Finished: $(date)"
