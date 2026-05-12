#!/bin/bash
#SBATCH --job-name=sweep_actsteps
#SBATCH --output=/data/scratch/sour/DiffusionProject/diffusion_policy/data/outputs/slurm-%j.out
#SBATCH --error=/data/scratch/sour/DiffusionProject/diffusion_policy/data/outputs/slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

set -uo pipefail
export HOME=/data/scratch/sour
source /data/scratch/sour/miniforge3/etc/profile.d/conda.sh
conda activate robodiff
cd /data/scratch/sour/DiffusionProject/diffusion_policy

# Experiment 2: n_action_steps sweep on lift + can + square 5k-epoch checkpoints
# under the FK→OSC adapter.
# Default trained value: n_action_steps = 8 (matches existing eval_latest_fk_osc_kp1000).
# Sweep over {1, 2, 4, 8, 12}.

LIFT_CKPT=$(ls -dt data/outputs/2026.05.11/*lift_lowdim_joint_delta_joint5k 2>/dev/null | head -1)/checkpoints/latest.ckpt
CAN_CKPT=$(ls -dt data/outputs/2026.05.11/*can_lowdim_joint_delta_joint5k 2>/dev/null | head -1)/checkpoints/latest.ckpt
SQR_CKPT=$(ls -dt data/outputs/2026.05.11/*square_lowdim_joint_delta_joint5k 2>/dev/null | head -1)/checkpoints/latest.ckpt

for STEPS in 1 2 4 8 12; do
  for pair in "lift:$LIFT_CKPT:1000" "can:$CAN_CKPT:1000" "square:$SQR_CKPT:3000"; do
    T="${pair%%:*}"
    REST="${pair#*:}"
    CKPT="${REST%:*}"
    KP="${REST##*:}"
    OUTDIR="$(dirname $(dirname $CKPT))/eval_actsteps_${STEPS}_kp${KP}"
    if [ -f "$OUTDIR/eval_log.json" ]; then echo "[skip] $T steps=$STEPS exists"; continue; fi
    echo "[run] $T steps=$STEPS kp=$KP -> $OUTDIR"
    python eval_fk.py --checkpoint "$CKPT" --output_dir "$OUTDIR" \
      --osc_kp "$KP" --n_action_steps "$STEPS" --device cuda:0 \
      || echo "[fail] $T steps=$STEPS"
  done
done

echo "=== sweep results ==="
for T in lift can square; do
  for STEPS in 1 2 4 8 12; do
    for KP in 1000 3000; do
      d=$(ls -dt data/outputs/2026.05.11/*${T}_lowdim_joint_delta_joint5k/eval_actsteps_${STEPS}_kp${KP} 2>/dev/null | head -1)
      [ -f "$d/eval_log.json" ] && echo "$T steps=$STEPS kp=$KP: $(grep -oE '\"test/mean_score\": [0-9.]+' $d/eval_log.json | head -1)"
    done
  done
done
echo "ALL DONE actsteps sweep"
