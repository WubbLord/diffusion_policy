#!/bin/bash
# Submit FK->OSC oracle replay for all 5 tasks at their best kp.
# Run from repo root: bash eval_fk_oracle_all5.sh
set -e
cd "$(dirname "$0")"

echo "submitting 5 oracle-replay jobs (one per task)..."
sbatch eval_fk_oracle.sh lift      1000 ph | tail -1
sbatch eval_fk_oracle.sh can       1000 ph | tail -1
sbatch eval_fk_oracle.sh square    3000 ph | tail -1
sbatch eval_fk_oracle.sh tool_hang 5000 ph | tail -1
sbatch eval_fk_oracle.sh transport 1000 ph | tail -1
echo "done. squeue:"
squeue -u sour --format="%.10i %.30j %.2t %.10M %.10L" | head
