#!/bin/bash
#SBATCH --job-name=dp_jd_adapter_eval50
#SBATCH --output=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%j.out
#SBATCH --error=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%j.err
#SBATCH -p csail-shared-h200
#SBATCH -t 02:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:h200:1
#SBATCH --qos=lab-free
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 5 ]; then
    echo "Usage: sbatch $0 DP_CHECKPOINT ADAPTER_CHECKPOINT OUTPUT_DIR [N_TEST] [N_ENVS]" >&2
    exit 2
fi

REPO_DIR=/data/scratch/zhangbr/diffusion_policy
cd "$REPO_DIR"
mkdir -p slurm/logs

source /data/scratch/zhangbr/miniforge3/etc/profile.d/conda.sh
conda activate /data/scratch/zhangbr/.conda/envs/robodiff

DP_CHECKPOINT="$1"
ADAPTER_CHECKPOINT="$2"
OUTPUT_DIR="$3"
N_TEST="${4:-50}"
N_ENVS="${5:-28}"

if [ ! -f "$DP_CHECKPOINT" ]; then
    echo "DP checkpoint not found: $DP_CHECKPOINT" >&2
    exit 1
fi

if [ ! -f "$ADAPTER_CHECKPOINT" ]; then
    echo "Adapter checkpoint not found: $ADAPTER_CHECKPOINT" >&2
    exit 1
fi

if [ -e "$OUTPUT_DIR" ] && [ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
    echo "Output directory already exists and is not empty: $OUTPUT_DIR" >&2
    exit 1
fi

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "start=$(date --iso-8601=seconds)"
echo "dp_checkpoint=${DP_CHECKPOINT}"
echo "adapter_checkpoint=${ADAPTER_CHECKPOINT}"
echo "output_dir=${OUTPUT_DIR}"
echo "n_test=${N_TEST}"
echo "n_envs=${N_ENVS}"

python scripts/eval_joint_delta_with_adapter.py \
    --checkpoint "$DP_CHECKPOINT" \
    --adapter-checkpoint "$ADAPTER_CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda:0 \
    --adapter-device cpu \
    --n-test "$N_TEST" \
    --n-test-vis 4 \
    --n-train 6 \
    --n-train-vis 2 \
    --n-envs "$N_ENVS"
