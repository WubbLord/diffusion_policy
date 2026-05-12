#!/bin/bash
#SBATCH --job-name=residual_osc
#SBATCH --output=/data/scratch/sour/DiffusionProject/diffusion_policy/data/outputs/slurm-%j.out
#SBATCH --error=/data/scratch/sour/DiffusionProject/diffusion_policy/data/outputs/slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00

TASK=${1:?need task arg}
KP=${2:-1000}
RESIDUAL_CLIP=${3:-0.3}
set -uo pipefail
export HOME=/data/scratch/sour
source /data/scratch/sour/miniforge3/etc/profile.d/conda.sh
conda activate robodiff
cd /data/scratch/sour/DiffusionProject/diffusion_policy

DATASET="data/robomimic/datasets/${TASK}/ph/low_dim.hdf5"
OBS_KEYS="object,robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos,robot0_joint_pos"
OUT="data/reverse_controller_osc_residual/${TASK}_ph"
ADAPTER_DIR="$OUT/inverse_mlp"

# 1. Collect residual targets (= demo_command - FK->OSC).
python collect_demo_residual_osc.py \
  --dataset "$DATASET" --output-dir "$OUT" \
  --obs-keys "$OBS_KEYS" \
  --joint-key robot0_joint_pos \
  --eef-quat-key robot0_eef_quat \
  --max-demos 200 --overwrite \
  || { echo "collect failed"; exit 1; }

# 2. Train MLP on residual targets.
python -m reverse_controller.train_inverse_model \
  --dataset-dir "$OUT" --output-dir "$ADAPTER_DIR" \
  --epochs 200 --batch-size 2048 \
  --hidden-dims 512,512,512 --activation silu \
  --val-ratio 0.05 --overwrite \
  || { echo "train failed"; exit 1; }

# 3. Full-pipeline eval with FK->OSC + residual NN.
DP_RUN=$(ls -dt data/outputs/2026.05.1[01]/*${TASK}_lowdim_joint_delta_joint5k 2>/dev/null | head -1)
if [ -n "$DP_RUN" ] && [ -f "$DP_RUN/checkpoints/latest.ckpt" ]; then
  EVAL_OUT="$DP_RUN/eval_latest_fk_osc_residual_kp${KP}_c${RESIDUAL_CLIP}"
  python eval_fk_residual.py \
    --checkpoint "$DP_RUN/checkpoints/latest.ckpt" --output_dir "$EVAL_OUT" \
    --residual_adapter "$ADAPTER_DIR/best.pt" \
    --residual_clip "$RESIDUAL_CLIP" \
    --osc_kp "$KP" --device cuda:0 \
    || echo "eval failed"
  if [ -f "$EVAL_OUT/eval_log.json" ]; then
    echo "=== $TASK residual kp=$KP clip=$RESIDUAL_CLIP ==="
    grep -oE '"test/mean_score": [0-9.]+' "$EVAL_OUT/eval_log.json" | head -1
  fi
fi

echo "ALL DONE residual $TASK"
