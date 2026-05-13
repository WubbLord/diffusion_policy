#!/bin/bash
#SBATCH --job-name=eval_fk_oracle
#SBATCH --output=/data/scratch/sour/DiffusionProject/diffusion_policy/data/outputs/slurm-%j.out
#SBATCH --error=/data/scratch/sour/DiffusionProject/diffusion_policy/data/outputs/slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# Usage: sbatch eval_fk_oracle.sh <task> <kp> [split]
# task in {lift, can, square, tool_hang, transport}
# kp e.g. 1000, 3000, 5000
# split defaults to ph

set -uo pipefail
TASK="${1:?task required}"
KP="${2:?osc_kp required}"
SPLIT="${3:-ph}"

export HOME=/data/scratch/sour
source /data/scratch/sour/miniforge3/etc/profile.d/conda.sh
conda activate robodiff

cd /data/scratch/sour/DiffusionProject/diffusion_policy

OUT="data/outputs/fk_oracle/${TASK}_${SPLIT}_kp${KP}"
mkdir -p "$OUT"
echo "=== FK->OSC oracle replay: task=$TASK split=$SPLIT kp=$KP ==="
python eval_fk_oracle.py --task "$TASK" --split "$SPLIT" --osc_kp "$KP" --output_dir "$OUT" 2>&1
echo "ALL DONE oracle $TASK kp=$KP"
