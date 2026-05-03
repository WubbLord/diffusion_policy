#!/bin/bash
#SBATCH --job-name=dp_can_joint_eval_mh_s42
#SBATCH --output=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%j.out
#SBATCH --error=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%j.err
#SBATCH -p csail-shared-h200
#SBATCH -t 12:00:00
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
: "${WANDB_API_KEY:?Set WANDB_API_KEY before submitting this job}"
export WANDB_API_KEY

python train.py \
    --config-name=train_diffusion_unet_lowdim_joint_delta_workspace \
    task=can_lowdim_joint_delta \
    task.dataset_type=mh \
    training.seed=42 \
    training.device=cuda:0 \
    training.num_epochs=5000 \
    training.rollout_every=50 \
    training.checkpoint_every=50 \
    checkpoint.topk.k=0 \
    checkpoint.save_last_ckpt=True
