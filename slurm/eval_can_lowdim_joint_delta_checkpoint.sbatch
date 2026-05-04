#!/bin/bash
#SBATCH --job-name=dp_eval_can_joint_ckpt
#SBATCH --output=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%j.out
#SBATCH --error=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%j.err
#SBATCH -p csail-shared-h200
#SBATCH -t 08:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:h200:1
#SBATCH --qos=lab-free
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: sbatch $0 CHECKPOINT OUTPUT_DIR [DEVICE]" >&2
    exit 2
fi

CHECKPOINT="$1"
OUTPUT_DIR="$2"
DEVICE="${3:-cuda:0}"

REPO_DIR=/data/scratch/zhangbr/diffusion_policy
cd "$REPO_DIR"

source /data/scratch/zhangbr/miniforge3/etc/profile.d/conda.sh
conda activate /data/scratch/zhangbr/.conda/envs/robodiff

if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

if [ -e "$OUTPUT_DIR" ]; then
    echo "Output directory already exists: $OUTPUT_DIR" >&2
    exit 1
fi

python eval.py \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"
