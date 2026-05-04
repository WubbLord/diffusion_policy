#!/bin/bash
#SBATCH --job-name=dp_can_lowdim_mh_s42
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
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5000}"
RESUME_ARGS=()
if [ -n "${RESUME_CKPT:-}" ]; then
    if [ ! -f "$RESUME_CKPT" ]; then
        echo "Resume checkpoint not found: $RESUME_CKPT" >&2
        exit 1
    fi
    RESUME_ARGS=(training.resume=True training.resume_ckpt_path="$RESUME_CKPT")
fi

python train.py \
    --config-name=train_diffusion_unet_lowdim_workspace \
    task=can_lowdim \
    task.dataset_type=mh \
    training.seed=42 \
    training.device=cuda:0 \
    training.num_epochs="$TOTAL_EPOCHS" \
    training.rollout_every=50 \
    training.checkpoint_every=50 \
    checkpoint.topk.k=0 \
    checkpoint.save_last_ckpt=True \
    hydra.run.dir=data/outputs/2026.05.03/11.28.30_train_diffusion_unet_lowdim_can_lowdim \
    logging.id=2rokxbfh \
    logging.resume=True \
    "${RESUME_ARGS[@]}"
