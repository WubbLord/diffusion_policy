#!/bin/bash
#SBATCH --job-name=dp_jd_adapt_cl
#SBATCH --output=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%A_%a.out
#SBATCH --error=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%A_%a.err
#SBATCH -p csail-shared-h200
#SBATCH -t 02:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:h200:1
#SBATCH --qos=lab-free
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-8%1

set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 5 ]; then
    echo "Usage: sbatch $0 DP_CHECKPOINT ADAPTER_CHECKPOINT RUN_ROOT [N_TEST] [N_ENVS]" >&2
    exit 2
fi

REPO_DIR=/data/scratch/zhangbr/diffusion_policy
cd "$REPO_DIR"
mkdir -p slurm/logs

source /data/scratch/zhangbr/miniforge3/etc/profile.d/conda.sh
conda activate /data/scratch/zhangbr/.conda/envs/robodiff
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

DP_CHECKPOINT="$1"
ADAPTER_CHECKPOINT="$2"
RUN_ROOT="$3"
N_TEST="${4:-50}"
N_ENVS="${5:-28}"
K="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"
OUT_DIR="$RUN_ROOT/k${K}"

if [ ! -f "$DP_CHECKPOINT" ]; then
    echo "DP checkpoint not found: $DP_CHECKPOINT" >&2
    exit 1
fi

if [ ! -f "$ADAPTER_CHECKPOINT" ]; then
    echo "Adapter checkpoint not found: $ADAPTER_CHECKPOINT" >&2
    exit 1
fi

mkdir -p "$RUN_ROOT"

echo "job_id=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "host=$(hostname)"
echo "start=$(date --iso-8601=seconds)"
echo "dp_checkpoint=${DP_CHECKPOINT}"
echo "adapter_checkpoint=${ADAPTER_CHECKPOINT}"
echo "run_root=${RUN_ROOT}"
echo "output_dir=${OUT_DIR}"
echo "n_test=${N_TEST}"
echo "n_envs=${N_ENVS}"
echo "adapter_execution_mode=closed_loop"
echo "adapter_inner_steps=${K}"
echo "omp_num_threads=${OMP_NUM_THREADS}"

python scripts/eval_joint_delta_with_adapter.py \
    --checkpoint "$DP_CHECKPOINT" \
    --adapter-checkpoint "$ADAPTER_CHECKPOINT" \
    --output-dir "$OUT_DIR" \
    --device cuda:0 \
    --adapter-device cpu \
    --n-test "$N_TEST" \
    --n-test-vis 4 \
    --n-train 6 \
    --n-train-vis 2 \
    --n-envs "$N_ENVS" \
    --adapter-execution-mode closed_loop \
    --adapter-inner-steps "$K" \
    --overwrite

python - <<'PY' "$RUN_ROOT"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("k*/eval_log.json"), key=lambda p: int(p.parent.name[1:])):
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
    rows.append({
        "k": int(path.parent.name[1:]),
        "output_dir": str(path.parent),
        "test_success_count": int(sum(test_scores)),
        "test_rollouts": len(test_scores),
        "test_mean_score": runner_log.get("test/mean_score"),
        "train_success_count": int(sum(train_scores)),
        "train_rollouts": len(train_scores),
        "train_mean_score": runner_log.get("train/mean_score"),
    })

with open(root / "sweep_summary.json", "w") as f:
    json.dump(rows, f, indent=2, sort_keys=True)

print(json.dumps(rows, indent=2, sort_keys=True), flush=True)
PY
