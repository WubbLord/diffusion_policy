#!/bin/bash
#SBATCH --job-name=demosup_ablate
#SBATCH --output=/data/scratch/sour/DiffusionProject/diffusion_policy/data/outputs/slurm-%j.out
#SBATCH --error=/data/scratch/sour/DiffusionProject/diffusion_policy/data/outputs/slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

TASK=${1:?need task arg}    # lift|can|square
set -uo pipefail
export HOME=/data/scratch/sour
source /data/scratch/sour/miniforge3/etc/profile.d/conda.sh
conda activate robodiff
cd /data/scratch/sour/DiffusionProject/diffusion_policy

DATASET="data/robomimic/datasets/${TASK}/ph/low_dim.hdf5"
OBS_KEYS="object,robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos,robot0_joint_pos"
DP_RUN=$(ls -dt data/outputs/2026.05.1[01]/*${TASK}_lowdim_joint_delta_joint5k 2>/dev/null | head -1)
DP_CKPT="$DP_RUN/checkpoints/latest.ckpt"

# Variant A: 50 demos, default arch & epochs
# Variant B: 100 demos, default arch & epochs
# Variant C: 200 demos, smaller MLP (128x2)
# Variant D: 200 demos, only 50 epochs
for VARIANT in d50 d100 h128 e50; do
  OUT="data/reverse_controller_osc_demosup_${VARIANT}/${TASK}_ph"
  ADAPTER_DIR="$OUT/inverse_mlp"
  if [ -f "$ADAPTER_DIR/best.pt" ]; then echo "[skip] $VARIANT exists"; continue; fi

  case "$VARIANT" in
    d50)   MAX_DEMOS=50;  HIDDEN=512,512,512; EPOCHS=200 ;;
    d100)  MAX_DEMOS=100; HIDDEN=512,512,512; EPOCHS=200 ;;
    h128)  MAX_DEMOS=200; HIDDEN=128,128;     EPOCHS=200 ;;
    e50)   MAX_DEMOS=200; HIDDEN=512,512,512; EPOCHS=50  ;;
  esac

  echo "==== $VARIANT (demos=$MAX_DEMOS hidden=$HIDDEN epochs=$EPOCHS) ===="
  python collect_demo_only_osc.py \
    --dataset "$DATASET" --output-dir "$OUT" \
    --obs-keys "$OBS_KEYS" \
    --max-demos $MAX_DEMOS --overwrite \
    || { echo "collect $VARIANT failed"; continue; }

  python -m reverse_controller.train_inverse_model \
    --dataset-dir "$OUT" --output-dir "$ADAPTER_DIR" \
    --epochs $EPOCHS --batch-size 2048 \
    --hidden-dims "$HIDDEN" --activation silu \
    --val-ratio 0.05 --overwrite \
    || { echo "train $VARIANT failed"; continue; }

  if [ -f "$DP_CKPT" ]; then
    EVAL_OUT="$DP_RUN/eval_latest_nn_osc_demosup_${VARIANT}"
    python eval_nn_osc.py \
      --checkpoint "$DP_CKPT" --output_dir "$EVAL_OUT" \
      --adapter "$ADAPTER_DIR/best.pt" --device cuda:0 \
      || echo "eval $VARIANT failed"
  fi
done

echo "ALL DONE demosup ablate $TASK"
