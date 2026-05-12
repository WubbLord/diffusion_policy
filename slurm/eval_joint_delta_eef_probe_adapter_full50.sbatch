#!/bin/bash
#SBATCH --job-name=dp_eef_adapter_full
#SBATCH --output=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%j.out
#SBATCH --error=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%j.err
#SBATCH -p csail-shared-h200
#SBATCH -t 01:00:00
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

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source /data/scratch/zhangbr/miniforge3/etc/profile.d/conda.sh
conda activate /data/scratch/zhangbr/.conda/envs/robodiff

DP_CHECKPOINT="$1"
ADAPTER_CHECKPOINT="$2"
OUTPUT_DIR="$3"
N_TEST="${4:-50}"
N_ENVS="${5:-28}"

echo "dp_checkpoint=${DP_CHECKPOINT}"
echo "adapter_checkpoint=${ADAPTER_CHECKPOINT}"
echo "output_dir=${OUTPUT_DIR}"
echo "n_test=${N_TEST}"
echo "n_envs=${N_ENVS}"

python scripts/eval_joint_delta_with_osc_adapter.py \
    --checkpoint "$DP_CHECKPOINT" \
    --adapter-checkpoint "$ADAPTER_CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda:0 \
    --adapter-device cpu \
    --n-test "$N_TEST" \
    --n-test-vis 4 \
    --n-train 6 \
    --n-train-vis 2 \
    --n-envs "$N_ENVS" \
    --overwrite

python - <<'PY' "$OUTPUT_DIR"
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1]) / "eval_log.json"
log = json.load(open(path))
runner_log = log.get("runner_log", {})
test_scores = [
    float(value)
    for key, value in runner_log.items()
    if key.startswith("test/sim_max_reward_")
]
train_scores = [
    float(value)
    for key, value in runner_log.items()
    if key.startswith("train/sim_max_reward_")
]
summary = {
    "test_success_count": int(sum(test_scores)),
    "test_rollouts": len(test_scores),
    "test_mean_score": runner_log.get("test/mean_score"),
    "train_success_count": int(sum(train_scores)),
    "train_rollouts": len(train_scores),
    "train_mean_score": runner_log.get("train/mean_score"),
}
with open(path.parent / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
PY
