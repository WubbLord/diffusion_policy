#!/bin/bash
#SBATCH --job-name=jp_sweep
#SBATCH --output=/data/scratch/sour/DiffusionProject/diffusion_policy/data/outputs/slurm-%j.out
#SBATCH --error=/data/scratch/sour/DiffusionProject/diffusion_policy/data/outputs/slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

TASK=${1:?need task arg}
set -uo pipefail
export HOME=/data/scratch/sour
source /data/scratch/sour/miniforge3/etc/profile.d/conda.sh
conda activate robodiff
cd /data/scratch/sour/DiffusionProject/diffusion_policy

DP_RUN=$(ls -dt data/outputs/2026.05.1[01]/*${TASK}_lowdim_joint_delta_joint5k 2>/dev/null | head -1)
DP_CKPT="$DP_RUN/checkpoints/latest.ckpt"
if [ ! -f "$DP_CKPT" ]; then echo "missing $DP_CKPT"; exit 1; fi

# JP controller kp sweep. Default robosuite JP kp=50 (way too low).
# Higher kp = stiffer joint tracking. damping_ratio=2.0 from the prior best.
for KP in 300 1000 3000 5000; do
  OUT="$DP_RUN/eval_latest_jp_kp${KP}_dr2.0"
  if [ -f "$OUT/eval_log.json" ]; then echo "[skip] kp=$KP exists"; continue; fi
  echo "==== JP $TASK kp=$KP ===="
  yes y | python eval_jp.py --checkpoint "$DP_CKPT" --output_dir "$OUT" \
    --kp "$KP" --damping_ratio 2.0 \
    || echo "[fail] $TASK kp=$KP"
done

echo "=== JP results $TASK ==="
for KP in 300 1000 3000 5000; do
  f="$DP_RUN/eval_latest_jp_kp${KP}_dr2.0/eval_log.json"
  [ -f "$f" ] && echo "$TASK kp=$KP: $(grep -oE '\"test/mean_score\": [0-9.]+' $f | head -1)"
done
echo "ALL DONE jp_sweep $TASK"
