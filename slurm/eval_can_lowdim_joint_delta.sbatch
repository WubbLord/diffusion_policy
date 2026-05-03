#!/bin/bash
#SBATCH --job-name=dp_eval_can_joint_e3200
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

REPO_DIR=/data/scratch/zhangbr/diffusion_policy
cd "$REPO_DIR"

source /data/scratch/zhangbr/miniforge3/etc/profile.d/conda.sh
conda activate /data/scratch/zhangbr/.conda/envs/robodiff

CHECKPOINT=data/outputs/2026.05.03/12.20.41_train_diffusion_unet_lowdim_joint_delta_can_lowdim_joint_delta/checkpoints/latest.ckpt
OUTPUT_DIR=data/outputs/2026.05.03/12.20.41_train_diffusion_unet_lowdim_joint_delta_can_lowdim_joint_delta/eval_latest_epoch

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
    --device cuda:0
