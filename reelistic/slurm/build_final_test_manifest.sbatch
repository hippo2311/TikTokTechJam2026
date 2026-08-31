#!/bin/bash
#SBATCH --job-name=techjam-final-manifest
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:nv:1
#SBATCH --output=slurm/logs/techjam-final-manifest-%j.out
#SBATCH --error=slurm/logs/techjam-final-manifest-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/TechJam}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/Dataset}"
TRAINING_MANIFEST_DIR="${TRAINING_MANIFEST_DIR:-$DATA_DIR/manifests}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_DIR/WildFake/final_test/manifests}"
CONTAINER="${CONTAINER:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"

cd "$PROJECT_DIR"
mkdir -p slurm/logs "$OUTPUT_DIR" "$PROJECT_DIR/.apptainer-cache"
test -s "$TRAINING_MANIFEST_DIR/audit.json"
export APPTAINER_CACHEDIR="$PROJECT_DIR/.apptainer-cache"
export APPTAINERENV_PYTHONPATH="$PROJECT_DIR"

apptainer exec \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  "$CONTAINER" \
  python3 -u "$PROJECT_DIR/cluster/build_final_test_manifest.py" \
    --data-dir "$DATA_DIR" \
    --training-manifest-dir "$TRAINING_MANIFEST_DIR" \
    --output-dir "$OUTPUT_DIR"
