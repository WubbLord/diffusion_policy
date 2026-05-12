#!/bin/bash
#SBATCH --job-name=demosup_count
#SBATCH --output=/data/scratch/sour/DiffusionProject/diffusion_policy/data/outputs/slurm-%j.out
#SBATCH --error=/data/scratch/sour/DiffusionProject/diffusion_policy/data/outputs/slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

TASK=${1:?need task arg}
set -uo pipefail
export HOME=/data/scratch/sour
source /data/scratch/sour/miniforge3/etc/profile.d/conda.sh
conda activate robodiff
cd /data/scratch/sour/DiffusionProject/diffusion_policy

DATASET="data/robomimic/datasets/${TASK}/ph/low_dim.hdf5"
OBS_KEYS="object,robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos,robot0_joint_pos"
DP_RUN=$(ls -dt data/outputs/2026.05.1[01]/*${TASK}_lowdim_joint_delta_joint5k 2>/dev/null | head -1)
DP_CKPT="$DP_RUN/checkpoints/latest.ckpt"

# Demo-count scaling: how few demos does demosup need to beat probe?
for D in 10 20; do
  OUT="data/reverse_controller_osc_demosup_d${D}/${TASK}_ph"
  ADAPTER_DIR="$OUT/inverse_mlp"
  if [ -f "$ADAPTER_DIR/best.pt" ]; then echo "[skip] d$D trained"; else
    python collect_demo_only_osc.py \
      --dataset "$DATASET" --output-dir "$OUT" \
      --obs-keys "$OBS_KEYS" \
      --max-demos $D --overwrite || { echo "collect d$D failed"; continue; }
    python -m reverse_controller.train_inverse_model \
      --dataset-dir "$OUT" --output-dir "$ADAPTER_DIR" \
      --epochs 200 --batch-size 2048 \
      --hidden-dims 512,512,512 --activation silu \
      --val-ratio 0.05 --overwrite || { echo "train d$D failed"; continue; }
  fi
  EVAL_OUT="$DP_RUN/eval_latest_nn_osc_demosup_d${D}"
  if [ -f "$EVAL_OUT/eval_log.json" ]; then echo "[skip] d$D eval"; else
    python eval_nn_osc.py \
      --checkpoint "$DP_CKPT" --output_dir "$EVAL_OUT" \
      --adapter "$ADAPTER_DIR/best.pt" --device cuda:0 || echo "eval d$D failed"
  fi
done

echo "=== demo-count results $TASK ==="
for D in 10 20 50 100 200; do
  f="$DP_RUN/eval_latest_nn_osc_demosup_d${D}/eval_log.json"
  [ -f "$f" ] && echo "$TASK d$D: $(grep -oE '\"test/mean_score\": [0-9.]+' $f | head -1)"
done
echo "ALL DONE demosup count $TASK"
