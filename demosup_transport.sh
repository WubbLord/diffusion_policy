#!/bin/bash
#SBATCH --job-name=demosup_transport
#SBATCH --output=/data/scratch/sour/DiffusionProject/diffusion_policy/data/outputs/slurm-%j.out
#SBATCH --error=/data/scratch/sour/DiffusionProject/diffusion_policy/data/outputs/slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -uo pipefail
export HOME=/data/scratch/sour
source /data/scratch/sour/miniforge3/etc/profile.d/conda.sh
conda activate robodiff
cd /data/scratch/sour/DiffusionProject/diffusion_policy

TASK=transport
DATASET="data/robomimic/datasets/${TASK}/ph/low_dim.hdf5"
OUT="data/reverse_controller_osc_demosup/${TASK}_ph"
ADAPTER_DIR="$OUT/inverse_mlp"

OBS_KEYS="object,robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos,robot0_joint_pos,robot1_eef_pos,robot1_eef_quat,robot1_gripper_qpos,robot1_joint_pos"
JOINT_KEYS="robot0_joint_pos,robot1_joint_pos"

# 1. demo-supervised collect (dual-arm)
python collect_demo_only_osc.py \
  --dataset "$DATASET" --output-dir "$OUT" \
  --obs-keys "$OBS_KEYS" \
  --joint-keys "$JOINT_KEYS" \
  --max-demos 200 --overwrite \
  || { echo "collect failed"; exit 1; }

# 2. train (input dim 73 obs + 14 dq = 87; output 14 OSC commands)
python -m reverse_controller.train_inverse_model \
  --dataset-dir "$OUT" --output-dir "$ADAPTER_DIR" \
  --epochs 200 --batch-size 2048 \
  --hidden-dims 512,512,512 --activation silu \
  --val-ratio 0.05 --overwrite \
  || { echo "train failed"; exit 1; }

# 3. Full-pipeline eval requires dual-arm support in
# RobomimicJointBrianOSCRunner which doesn't exist yet. Mark adapter ready;
# eval is a follow-up code change.
echo "ALL DONE demosup transport (collect+train); eval pending dual-arm runner support"
