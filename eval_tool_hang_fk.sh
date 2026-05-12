#!/bin/bash
#SBATCH --job-name=tool_hang_fk_eval
#SBATCH --output=/data/scratch/sour/DiffusionProject/diffusion_policy/data/outputs/slurm-%j.out
#SBATCH --error=/data/scratch/sour/DiffusionProject/diffusion_policy/data/outputs/slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00

set -uo pipefail
export HOME=/data/scratch/sour
source /data/scratch/sour/miniforge3/etc/profile.d/conda.sh
conda activate robodiff
cd /data/scratch/sour/DiffusionProject/diffusion_policy

# tool_hang DP is still training (job 818757), but its latest.ckpt updates
# in-place as epochs complete. Eval the current latest at three kp settings.
DP_RUN=$(ls -dt data/outputs/2026.05.11/*tool_hang_lowdim_joint_delta_joint5k 2>/dev/null \
         | xargs -I{} sh -c 'if [ -f "{}/checkpoints/latest.ckpt" ]; then echo {}; fi' \
         | head -1)
DP_CKPT="$DP_RUN/checkpoints/latest.ckpt"
echo "using $DP_CKPT (size $(stat -c %s $DP_CKPT 2>/dev/null))"

for KP in 1000 3000; do
  OUT="$DP_RUN/eval_latest_fk_osc_kp${KP}_inprogress"
  if [ -f "$OUT/eval_log.json" ]; then echo "[skip] kp=$KP exists"; continue; fi
  echo "==== eval tool_hang FK kp=$KP ===="
  python eval_fk.py --checkpoint "$DP_CKPT" --output_dir "$OUT" \
    --osc_kp "$KP" --device cuda:0 \
    || echo "[fail] tool_hang kp=$KP"
done

# Also: re-attempt demosup tool_hang full eval since adapter exists from job 827301
DEMOSUP_ADAPTER="data/reverse_controller_osc_demosup/tool_hang_ph/inverse_mlp/best.pt"
if [ -f "$DEMOSUP_ADAPTER" ]; then
  OUT="$DP_RUN/eval_latest_nn_osc_demosup"
  if [ ! -f "$OUT/eval_log.json" ]; then
    echo "==== eval tool_hang NN-demosup ===="
    python eval_nn_osc.py --checkpoint "$DP_CKPT" --output_dir "$OUT" \
      --adapter "$DEMOSUP_ADAPTER" --device cuda:0 \
      || echo "[fail] tool_hang demosup"
  fi
fi

echo "ALL DONE tool_hang fk+demosup"
