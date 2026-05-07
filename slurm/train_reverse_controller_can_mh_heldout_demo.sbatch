#!/bin/bash
#SBATCH --job-name=dp_reverse_can_heldout
#SBATCH --output=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%j.out
#SBATCH --error=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%j.err
#SBATCH -p csail-shared-h200
#SBATCH -t 04:00:00
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
PROBE_DIR="${2:-data/reverse_controller/can_mh_joint_position_s0.25_n32/probes}"
RUN_DIR="${3:-data/reverse_controller/can_mh_joint_position_s0.25_n32_heldout_demo}"
JOINT_DELTA_SCALE="${4:-0.25}"
N_ENVS="${5:-28}"
EPOCHS="${6:-100}"

TRAIN_DIR="$RUN_DIR/f_mlp_train000_249_val250_299"
EVAL_DIR="$RUN_DIR/eval_one_step_demo250_299"

if [ ! -d "$PROBE_DIR/shards" ]; then
    echo "Probe shards not found at $PROBE_DIR/shards" >&2
    echo "Run slurm/train_reverse_controller_can_mh.sbatch first, or pass PROBE_DIR as arg 2." >&2
    exit 1
fi

python reverse_controller/train_inverse_model.py \
    --dataset-dir "$PROBE_DIR" \
    --output-dir "$TRAIN_DIR" \
    --train-demo-start 0 \
    --train-demo-end 250 \
    --val-demo-start 250 \
    --val-demo-end 300 \
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
    --demo-start 250 \
    --demo-end 300 \
    --max-demos 50 \
    --n-envs "$N_ENVS" \
    --device cuda:0 \
    --overwrite
