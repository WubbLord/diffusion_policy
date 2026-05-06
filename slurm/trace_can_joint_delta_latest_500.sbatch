#!/bin/bash
#SBATCH --job-name=dp_trace_can_joint_500
#SBATCH --output=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%j.out
#SBATCH --error=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%j.err
#SBATCH -p csail-shared-h200
#SBATCH -t 12:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:h200:1
#SBATCH --qos=lab-free
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28

set -euo pipefail

REPO_DIR=/data/scratch/zhangbr/diffusion_policy
cd "$REPO_DIR"
mkdir -p slurm/logs

source /data/scratch/zhangbr/miniforge3/etc/profile.d/conda.sh
conda activate /data/scratch/zhangbr/.conda/envs/robodiff

CHECKPOINT="${1:-data/outputs/2026.05.03/12.20.41_train_diffusion_unet_lowdim_joint_delta_can_lowdim_joint_delta/checkpoints/latest.ckpt}"
OUTPUT_DIR="${2:-data/outputs/2026.05.03/12.20.41_train_diffusion_unet_lowdim_joint_delta_can_lowdim_joint_delta/joint_delta_trace_latest_500}"
N_ROLLOUTS="${3:-500}"
N_ENVS="${4:-28}"

python diffusion_policy/scripts/trace_joint_delta_rollouts.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --n-rollouts "$N_ROLLOUTS" \
    --n-envs "$N_ENVS" \
    --device cuda:0
