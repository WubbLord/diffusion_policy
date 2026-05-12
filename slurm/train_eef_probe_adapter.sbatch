#!/bin/bash
#SBATCH --job-name=dp_eef_probe_adapter
#SBATCH --output=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%j.out
#SBATCH --error=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%j.err
#SBATCH -p csail-shared-h200
#SBATCH -t 12:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:h200:1
#SBATCH --qos=lab-free
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28

set -euo pipefail

REPO_DIR=/data/scratch/zhangbr/diffusion_policy
cd "$REPO_DIR"
mkdir -p slurm/logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source /data/scratch/zhangbr/miniforge3/etc/profile.d/conda.sh
conda activate /data/scratch/zhangbr/.conda/envs/robodiff

TASK="${1:?Usage: sbatch slurm/train_eef_probe_adapter.sbatch TASK DATASET_TYPE [SAMPLES_PER_STEP] [N_ENVS] [EPOCHS] [RUN_DIR]}"
DATASET_TYPE="${2:?Usage: sbatch slurm/train_eef_probe_adapter.sbatch TASK DATASET_TYPE [SAMPLES_PER_STEP] [N_ENVS] [EPOCHS] [RUN_DIR]}"
SAMPLES_PER_STEP="${3:-32}"
N_ENVS="${4:-28}"
EPOCHS="${5:-100}"
RUN_DIR="${6:-data/reverse_controller_osc_probe/${TASK}_${DATASET_TYPE}_osc_pose_n${SAMPLES_PER_STEP}_heldout_demo}"

DATASET="data/robomimic/datasets/${TASK}/${DATASET_TYPE}/low_dim.hdf5"
if [ ! -f "$DATASET" ]; then
    echo "Dataset not found: $DATASET" >&2
    exit 1
fi

if [ "$TASK" = "transport" ]; then
    OBS_KEYS="object,robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos,robot0_joint_pos,robot0_joint_vel,robot1_eef_pos,robot1_eef_quat,robot1_gripper_qpos,robot1_joint_pos,robot1_joint_vel"
    JOINT_KEYS="robot0_joint_pos,robot1_joint_pos"
else
    OBS_KEYS="object,robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos,robot0_joint_pos,robot0_joint_vel"
    JOINT_KEYS="robot0_joint_pos"
fi

if [ "$DATASET_TYPE" = "ph" ]; then
    TRAIN_START=0
    TRAIN_END=150
    VAL_START=150
    VAL_END=200
elif [ "$DATASET_TYPE" = "mh" ]; then
    TRAIN_START=0
    TRAIN_END=250
    VAL_START=250
    VAL_END=300
else
    echo "Unsupported dataset type: $DATASET_TYPE" >&2
    exit 1
fi

printf -v TRAIN_START_PAD "%03d" "$TRAIN_START"
printf -v TRAIN_LAST_PAD "%03d" "$((TRAIN_END - 1))"
printf -v VAL_START_PAD "%03d" "$VAL_START"
printf -v VAL_LAST_PAD "%03d" "$((VAL_END - 1))"

COLLECT_DIR="$RUN_DIR/probes"
TRAIN_DIR="$RUN_DIR/f_mlp_train${TRAIN_START_PAD}_${TRAIN_LAST_PAD}_val${VAL_START_PAD}_${VAL_LAST_PAD}"

echo "task=${TASK}"
echo "dataset_type=${DATASET_TYPE}"
echo "dataset=${DATASET}"
echo "run_dir=${RUN_DIR}"
echo "collect_dir=${COLLECT_DIR}"
echo "train_dir=${TRAIN_DIR}"
echo "obs_keys=${OBS_KEYS}"
echo "joint_keys=${JOINT_KEYS}"
echo "samples_per_step=${SAMPLES_PER_STEP}"
echo "n_envs=${N_ENVS}"
echo "epochs=${EPOCHS}"
echo "train_demos=[${TRAIN_START}, ${TRAIN_END})"
echo "val_demos=[${VAL_START}, ${VAL_END})"

python - "$DATASET" "$SAMPLES_PER_STEP" <<'PY'
import h5py
import sys

dataset = sys.argv[1]
samples_per_step = int(sys.argv[2])
with h5py.File(dataset, "r") as f:
    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
    n_steps = sum(int(f[f"data/{demo}/actions"].shape[0]) for demo in demos)
    action_dim = int(f[f"data/{demos[0]}/actions"].shape[-1])
print(f"n_demos={len(demos)}")
print(f"n_demo_steps={n_steps}")
print(f"action_dim={action_dim}")
print(f"synthetic_probe_steps={n_steps * samples_per_step}")
PY

python reverse_controller/collect_inverse_dataset_osc.py \
    --dataset "$DATASET" \
    --output-dir "$COLLECT_DIR" \
    --obs-keys "$OBS_KEYS" \
    --joint-keys "$JOINT_KEYS" \
    --samples-per-step "$SAMPLES_PER_STEP" \
    --n-envs "$N_ENVS" \
    --seed 42

python reverse_controller/train_inverse_model.py \
    --dataset-dir "$COLLECT_DIR" \
    --output-dir "$TRAIN_DIR" \
    --train-demo-start "$TRAIN_START" \
    --train-demo-end "$TRAIN_END" \
    --val-demo-start "$VAL_START" \
    --val-demo-end "$VAL_END" \
    --epochs "$EPOCHS" \
    --batch-size 8192 \
    --hidden-dims 512,512,512 \
    --device cuda:0 \
    --num-workers 4 \
    --overwrite

echo "EEF probe adapter training complete: ${TRAIN_DIR}/best.pt"
