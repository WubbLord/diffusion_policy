#!/bin/bash
#SBATCH --job-name=dp_oracle_cl_array
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

if [ "$#" -lt 5 ] || [ "$#" -gt 6 ]; then
    echo "Usage: sbatch $0 CHECKPOINT DATASET RUN_ROOT DEMO_START DEMO_END [MAX_DEMOS]" >&2
    exit 2
fi

REPO_DIR=/data/scratch/zhangbr/diffusion_policy
cd "$REPO_DIR"
mkdir -p slurm/logs

source /data/scratch/zhangbr/miniforge3/etc/profile.d/conda.sh
conda activate /data/scratch/zhangbr/.conda/envs/robodiff

CHECKPOINT="$1"
DATASET="$2"
RUN_ROOT="$3"
DEMO_START="$4"
DEMO_END="$5"
MAX_DEMOS="${6:-50}"
K="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Adapter checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

if [ ! -f "$DATASET" ]; then
    echo "Dataset not found: $DATASET" >&2
    exit 1
fi

OUT_DIR="$RUN_ROOT/k${K}"
mkdir -p "$RUN_ROOT"

echo "job_id=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "host=$(hostname)"
echo "start=$(date --iso-8601=seconds)"
echo "checkpoint=${CHECKPOINT}"
echo "dataset=${DATASET}"
echo "run_root=${RUN_ROOT}"
echo "output_dir=${OUT_DIR}"
echo "demo_start=${DEMO_START}"
echo "demo_end=${DEMO_END}"
echo "max_demos=${MAX_DEMOS}"
echo "inner_steps=${K}"

python reverse_controller/oracle_replay.py \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" \
    --output-dir "$OUT_DIR" \
    --demo-start "$DEMO_START" \
    --demo-end "$DEMO_END" \
    --max-demos "$MAX_DEMOS" \
    --execution-mode closed_loop \
    --inner-steps "$K" \
    --device cuda:0 \
    --overwrite

python - <<'PY' "$RUN_ROOT"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("k*/summary.json"), key=lambda p: int(p.parent.name[1:])):
    summary = json.load(open(path))
    rows.append({
        "k": int(path.parent.name[1:]),
        "output_dir": str(path.parent),
        "success_rate": summary.get("success_rate"),
        "delta_tracking_mae": summary.get("delta_tracking_mae"),
        "delta_tracking_rmse": summary.get("delta_tracking_rmse"),
        "mean_q_abs_error": summary.get("mean_q_abs_error"),
        "mean_q_l2_error": summary.get("mean_q_l2_error"),
        "mean_final_q_abs_error": summary.get("mean_final_q_abs_error"),
        "mean_final_q_l2_error": summary.get("mean_final_q_l2_error"),
        "mean_eef_pos_l2_error": summary.get("mean_eef_pos_l2_error"),
        "mean_object_l2_error": summary.get("mean_object_l2_error"),
        "n_steps": summary.get("n_steps"),
        "n_controller_steps": summary.get("n_controller_steps"),
    })

with open(root / "sweep_summary.json", "w") as f:
    json.dump(rows, f, indent=2, sort_keys=True)

print(json.dumps(rows, indent=2, sort_keys=True), flush=True)
PY
