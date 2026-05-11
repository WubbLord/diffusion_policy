#!/bin/bash
#SBATCH --job-name=dp_oracle_cl_sweep
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

REPO_DIR=/data/scratch/zhangbr/diffusion_policy
cd "$REPO_DIR"
mkdir -p slurm/logs

source /data/scratch/zhangbr/miniforge3/etc/profile.d/conda.sh
conda activate /data/scratch/zhangbr/.conda/envs/robodiff

KS="${1:?Usage: sbatch oracle_replay_can_mh_closed_loop_sweep.sbatch '1 2 3 4' [RUN_ROOT]}"
RUN_ROOT="${2:-data/reverse_controller/can_mh_joint_position_s0.25_n32_heldout_demo/oracle_replay_closed_loop_sweep_demo250_299}"

CHECKPOINT="data/reverse_controller/can_mh_joint_position_s0.25_n32_heldout_demo/f_mlp_train000_249_val250_299/best.pt"
DATASET="data/robomimic/datasets/can/mh/low_dim.hdf5"
DEMO_START=250
DEMO_END=300
MAX_DEMOS=50

mkdir -p "$RUN_ROOT"

for K in $KS; do
    OUT_DIR="$RUN_ROOT/k${K}"
    echo "Running Can MH closed-loop oracle replay with inner_steps=${K}"
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
