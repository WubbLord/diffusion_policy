#!/bin/bash
#SBATCH --job-name=dp_adapter_oracle
#SBATCH --output=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%j.out
#SBATCH --error=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%j.err
#SBATCH -p csail-shared-h200
#SBATCH -t 01:30:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:h200:1
#SBATCH --qos=lab-free
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

REPO_DIR=/data/scratch/zhangbr/diffusion_policy
cd "$REPO_DIR"
mkdir -p slurm/logs

source /data/scratch/zhangbr/miniforge3/etc/profile.d/conda.sh
conda activate /data/scratch/zhangbr/.conda/envs/robodiff

CHECKPOINT="${1:?Usage: sbatch oracle_replay_adapter.sbatch CHECKPOINT DATASET OUTPUT_DIR DEMO_START DEMO_END [MAX_DEMOS]}"
DATASET="${2:?Usage: sbatch oracle_replay_adapter.sbatch CHECKPOINT DATASET OUTPUT_DIR DEMO_START DEMO_END [MAX_DEMOS]}"
OUTPUT_DIR="${3:?Usage: sbatch oracle_replay_adapter.sbatch CHECKPOINT DATASET OUTPUT_DIR DEMO_START DEMO_END [MAX_DEMOS]}"
DEMO_START="${4:?Usage: sbatch oracle_replay_adapter.sbatch CHECKPOINT DATASET OUTPUT_DIR DEMO_START DEMO_END [MAX_DEMOS]}"
DEMO_END="${5:?Usage: sbatch oracle_replay_adapter.sbatch CHECKPOINT DATASET OUTPUT_DIR DEMO_START DEMO_END [MAX_DEMOS]}"
MAX_DEMOS="${6:-50}"

python reverse_controller/oracle_replay.py \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --demo-start "$DEMO_START" \
    --demo-end "$DEMO_END" \
    --max-demos "$MAX_DEMOS" \
    --device cuda:0 \
    --overwrite
