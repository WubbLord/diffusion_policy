#!/bin/bash
#SBATCH --job-name=dp_test_nvidia_smi
#SBATCH --output=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%j.out
#SBATCH --error=/data/scratch/zhangbr/diffusion_policy/slurm/logs/%x-%j.err
#SBATCH -p csail-shared-h200
#SBATCH -t 12:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:h200:1
#SBATCH --qos=lab-free
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -euo pipefail

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "start=$(date --iso-8601=seconds)"
watch -n 0.1 nvidia-smi
