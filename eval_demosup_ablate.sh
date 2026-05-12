#!/bin/bash
#SBATCH --job-name=demosup_ablate_eval
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
set -uo pipefail
export HOME=/data/scratch/sour
source /data/scratch/sour/miniforge3/etc/profile.d/conda.sh
conda activate robodiff
cd /data/scratch/sour/DiffusionProject/diffusion_policy

DP_RUN=$(ls -dt data/outputs/2026.05.1[01]/*${TASK}_lowdim_joint_delta_joint5k 2>/dev/null | head -1)
DP_CKPT="$DP_RUN/checkpoints/latest.ckpt"
if [ ! -f "$DP_CKPT" ]; then echo "missing $DP_CKPT"; exit 1; fi

for VARIANT in d50 d100 h128 e50; do
  ADAPTER="data/reverse_controller_osc_demosup_${VARIANT}/${TASK}_ph/inverse_mlp/best.pt"
  if [ ! -f "$ADAPTER" ]; then echo "[skip] no adapter for $VARIANT"; continue; fi
  EVAL_OUT="$DP_RUN/eval_latest_nn_osc_demosup_${VARIANT}"
  if [ -f "$EVAL_OUT/eval_log.json" ]; then echo "[skip] $VARIANT already evaluated"; continue; fi
  echo "==== eval $TASK $VARIANT ===="
  python eval_nn_osc.py \
    --checkpoint "$DP_CKPT" --output_dir "$EVAL_OUT" \
    --adapter "$ADAPTER" --device cuda:0 \
    || echo "[fail] $VARIANT"
done

echo "=== ablate eval results $TASK ==="
for VARIANT in d50 d100 h128 e50; do
  f="$DP_RUN/eval_latest_nn_osc_demosup_${VARIANT}/eval_log.json"
  [ -f "$f" ] && echo "$TASK $VARIANT: $(grep -oE '\"test/mean_score\": [0-9.]+' $f | head -1)"
done
echo "ALL DONE ablate eval $TASK"
