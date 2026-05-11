#!/bin/bash
#SBATCH --job-name=dp_oracle_cl_sweep
#SBATCH --output=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%j.out
#SBATCH --error=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%j.err
#SBATCH -p csail-shared-h200
#SBATCH -t 04:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:h200:1
#SBATCH --qos=lab-free
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

if [ "$#" -lt 6 ] || [ "$#" -gt 7 ]; then
    echo "Usage: sbatch $0 CHECKPOINT DATASET RUN_ROOT DEMO_START DEMO_END 'KS' [MAX_DEMOS]" >&2
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
KS="$6"
MAX_DEMOS="${7:-50}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Adapter checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

if [ ! -f "$DATASET" ]; then
    echo "Dataset not found: $DATASET" >&2
    exit 1
fi

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "start=$(date --iso-8601=seconds)"
echo "checkpoint=${CHECKPOINT}"
echo "dataset=${DATASET}"
echo "run_root=${RUN_ROOT}"
echo "demo_start=${DEMO_START}"
echo "demo_end=${DEMO_END}"
echo "max_demos=${MAX_DEMOS}"
echo "ks=${KS}"

mkdir -p "$RUN_ROOT"

for K in $KS; do
    OUT_DIR="$RUN_ROOT/k${K}"
    echo "Running closed-loop oracle replay with inner_steps=${K}"
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
done

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
