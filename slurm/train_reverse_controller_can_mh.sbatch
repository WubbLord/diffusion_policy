#!/bin/bash
#SBATCH --job-name=dp_reverse_can_mh
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

DATASET="${1:-data/robomimic/datasets/can/mh/low_dim.hdf5}"
RUN_DIR="${2:-data/reverse_controller/can_mh_joint_position_s0.25_n32}"
JOINT_DELTA_SCALE="${3:-0.25}"
SAMPLES_PER_STEP="${4:-32}"
N_ENVS="${5:-28}"
EPOCHS="${6:-100}"

COLLECT_DIR="$RUN_DIR/probes"
TRAIN_DIR="$RUN_DIR/f_mlp"
EVAL_DIR="$RUN_DIR/eval_one_step"

python reverse_controller/collect_inverse_dataset.py \
    --dataset "$DATASET" \
    --output-dir "$COLLECT_DIR" \
    --joint-delta-scale "$JOINT_DELTA_SCALE" \
    --samples-per-step "$SAMPLES_PER_STEP" \
    --n-envs "$N_ENVS" \
    --seed 42

python reverse_controller/train_inverse_model.py \
    --dataset-dir "$COLLECT_DIR" \
    --output-dir "$TRAIN_DIR" \
    --epochs "$EPOCHS" \
    --batch-size 8192 \
    --hidden-dims 512,512,512 \
    --device cuda:0 \
    --num-workers 4 \
    --overwrite

python reverse_controller/eval_inverse_model.py \
    --checkpoint "$TRAIN_DIR/best.pt" \
    --dataset "$DATASET" \
    --output-dir "$EVAL_DIR" \
    --joint-delta-scale "$JOINT_DELTA_SCALE" \
    --n-envs "$N_ENVS" \
    --max-demos 50 \
    --device cuda:0 \
    --overwrite
