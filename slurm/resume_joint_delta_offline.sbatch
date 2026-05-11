#!/bin/bash
#SBATCH --job-name=dp_jd_resume
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

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    echo "Usage: sbatch $0 TASK DATASET_TYPE RUN_DIR LOGGING_NAME [TOTAL_EPOCHS]" >&2
    exit 2
fi

REPO_DIR=/data/scratch/zhangbr/diffusion_policy
cd "$REPO_DIR"

source /data/scratch/zhangbr/miniforge3/etc/profile.d/conda.sh
conda activate /data/scratch/zhangbr/.conda/envs/robodiff

TASK="$1"
DATASET_TYPE="$2"
RUN_DIR="$3"
LOGGING_NAME="$4"
TOTAL_EPOCHS="${5:-5000}"
RESUME_CKPT="$RUN_DIR/checkpoints/latest.ckpt"

if [ ! -f "$RESUME_CKPT" ]; then
    echo "Resume checkpoint not found: $RESUME_CKPT" >&2
    exit 1
fi

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "start=$(date --iso-8601=seconds)"
echo "task=${TASK}"
echo "dataset_type=${DATASET_TYPE}"
echo "run_dir=${RUN_DIR}"
echo "resume_ckpt=${RESUME_CKPT}"
echo "total_epochs=${TOTAL_EPOCHS}"

python train.py \
    --config-name=train_diffusion_unet_lowdim_joint_delta_workspace \
    task="$TASK" \
    task.dataset_type="$DATASET_TYPE" \
    training.seed=42 \
    training.device=cuda:0 \
    training.num_epochs="$TOTAL_EPOCHS" \
    training.checkpoint_every=50 \
    checkpoint.save_last_ckpt=True \
    logging.mode=offline \
    logging.name="$LOGGING_NAME" \
    training.resume=True \
    training.resume_ckpt_path="$RESUME_CKPT" \
    hydra.run.dir="$RUN_DIR"
