#!/bin/bash
#SBATCH --job-name=techjam-manifests
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm/logs/techjam-manifests-%j.out
#SBATCH --error=slurm/logs/techjam-manifests-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/TechJam}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/Dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/Dataset/manifests}"
cd "$PROJECT_DIR"
mkdir -p slurm/logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR"

MANIFEST_EXTRA_ARGS=()
if [[ "${ALLOW_DECODE_FAILURES:-0}" == "1" ]]; then
  MANIFEST_EXTRA_ARGS+=(--allow-decode-failures)
fi

python3 -u "$PROJECT_DIR/cluster/build_dataset_manifests.py" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --seed 42 \
    --val-fraction 0.05 \
    --calibration-fraction 0.05 \
    --workers "${SLURM_CPUS_PER_TASK:-4}" \
    "${MANIFEST_EXTRA_ARGS[@]}"
