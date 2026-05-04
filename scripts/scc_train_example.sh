#!/usr/bin/env bash
#SBATCH --job-name=wiag-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate wiag

DATA_ARGS=()
if [[ -n "${TRAIN_MANIFEST:-}" ]]; then
  DATA_ARGS+=(--train-manifest "${TRAIN_MANIFEST}")
else
  DATA_ARGS+=(--train-dir "${TRAIN_DIR:-data/train}")
fi

if [[ -n "${VAL_MANIFEST:-}" ]]; then
  DATA_ARGS+=(--val-manifest "${VAL_MANIFEST}")
elif [[ -n "${VAL_DIR:-}" ]]; then
  DATA_ARGS+=(--val-dir "${VAL_DIR}")
fi

python scripts/train.py \
  --config configs/default.yaml \
  --output-dir "${OUTPUT_DIR:-runs/scc_vit_baseline}" \
  --workers "${SLURM_CPUS_PER_TASK:-8}" \
  --amp \
  "${DATA_ARGS[@]}"
