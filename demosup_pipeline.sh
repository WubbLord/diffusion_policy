#!/bin/bash
#SBATCH --job-name=demosup_osc
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
OUT="data/reverse_controller_osc_demosup/${TASK}_ph"
ADAPTER_DIR="$OUT/inverse_mlp"

# 1. collect demo-supervised pairs (no env probing, fast)
python collect_demo_only_osc.py \
  --dataset "$DATASET" --output-dir "$OUT" \
  --obs-keys "object,robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos,robot0_joint_pos" \
  --max-demos 200 --overwrite || { echo "collect failed"; exit 1; }

# 2. train (no joint_vel drop; obs_keys already exclude joint_vel)
python -m reverse_controller.train_inverse_model \
  --dataset-dir "$OUT" --output-dir "$ADAPTER_DIR" \
  --epochs 200 --batch-size 2048 \
  --hidden-dims 512,512,512 --activation silu \
  --val-ratio 0.05 --overwrite \
  || { echo "train failed"; exit 1; }

# 3. oracle replay (adapter alone)
ORC="$OUT/oracle_replay"
python oracle_replay_osc.py \
  --task "${TASK}_ph" --adapter "$ADAPTER_DIR/best.pt" \
  --output-dir "$ORC" --demo-start 150 --demo-end 200 \
  || echo "oracle failed"

# 4. full pipeline eval (DP + demo-supervised adapter)
DP_RUN=$(ls -dt data/outputs/2026.05.1[01]/*${TASK}_lowdim_joint_delta_joint5k 2>/dev/null | head -1)
if [ -n "$DP_RUN" ] && [ -f "$DP_RUN/checkpoints/latest.ckpt" ]; then
  DP_CKPT="$DP_RUN/checkpoints/latest.ckpt"
  EVAL_OUT="$DP_RUN/eval_latest_nn_osc_demosup"
  python eval_nn_osc.py \
    --checkpoint "$DP_CKPT" --output_dir "$EVAL_OUT" \
    --adapter "$ADAPTER_DIR/best.pt" --device cuda:0 \
    || echo "full eval failed"
fi

echo "ALL DONE demosup $TASK"
