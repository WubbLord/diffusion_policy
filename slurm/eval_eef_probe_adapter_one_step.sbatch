#!/bin/bash
#SBATCH --job-name=dp_eef_adapt_1step
#SBATCH --output=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%j.out
#SBATCH --error=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%j.err
#SBATCH -p csail-shared-h200
#SBATCH -t 01:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:h200:1
#SBATCH --qos=lab-free
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28

set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 7 ]; then
    echo "Usage: sbatch $0 ADAPTER_CHECKPOINT DATASET OUTPUT_DIR [DEMO_START] [DEMO_END] [N_ENVS] [MAX_DEMOS]" >&2
    exit 2
fi

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

ADAPTER_CHECKPOINT="$1"
DATASET="$2"
OUTPUT_DIR="$3"
DEMO_START="${4:-}"
DEMO_END="${5:-}"
N_ENVS="${6:-28}"
MAX_DEMOS="${7:-50}"

echo "adapter_checkpoint=${ADAPTER_CHECKPOINT}"
echo "dataset=${DATASET}"
echo "output_dir=${OUTPUT_DIR}"
echo "demo_start=${DEMO_START}"
echo "demo_end=${DEMO_END}"
echo "n_envs=${N_ENVS}"
echo "max_demos=${MAX_DEMOS}"

CMD=(
    python scripts/eval_osc_adapter_one_step.py
    --checkpoint "$ADAPTER_CHECKPOINT"
    --dataset "$DATASET"
    --output-dir "$OUTPUT_DIR"
    --n-envs "$N_ENVS"
    --max-demos "$MAX_DEMOS"
    --device cuda:0
    --overwrite
)
if [ -n "$DEMO_START" ]; then
    CMD+=(--demo-start "$DEMO_START")
fi
if [ -n "$DEMO_END" ]; then
    CMD+=(--demo-end "$DEMO_END")
fi

"${CMD[@]}"
