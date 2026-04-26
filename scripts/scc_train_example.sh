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
conda activate jigsaw

python scripts/train.py \
  --config configs/default.yaml \
  --train-dir "${TRAIN_DIR:-data/train}" \
  --val-dir "${VAL_DIR:-data/val}" \
  --output-dir "${OUTPUT_DIR:-runs/scc_vit_baseline}" \
  --workers "${SLURM_CPUS_PER_TASK:-8}" \
  --amp
