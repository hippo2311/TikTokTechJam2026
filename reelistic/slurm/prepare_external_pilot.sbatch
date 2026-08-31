#!/bin/bash
#SBATCH --job-name=techjam-prepare-data
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:nv:1
#SBATCH --output=slurm/logs/techjam-prepare-data-%j.out
#SBATCH --error=slurm/logs/techjam-prepare-data-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/TechJam}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/Dataset/external_pilot}"
MANIFEST_DIR="${MANIFEST_DIR:-$PROJECT_DIR/data_manifests/external_pilot}"
CONTAINER="${CONTAINER:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"

cd "$PROJECT_DIR"
mkdir -p slurm/logs "$OUTPUT_DIR" "$MANIFEST_DIR" "$PROJECT_DIR/.apptainer-cache"
export APPTAINER_CACHEDIR="$PROJECT_DIR/.apptainer-cache"

JOB_TMP="${SLURM_TMPDIR:-/tmp/$USER-techjam-$SLURM_JOB_ID}"
PYTHON_PACKAGES="$JOB_TMP/techjam-python"
mkdir -p "$PYTHON_PACKAGES"

apptainer exec --nv \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --bind "$JOB_TMP:$JOB_TMP" \
  "$CONTAINER" \
  python3 -m pip install --quiet --no-cache-dir \
    --target "$PYTHON_PACKAGES" \
    "datasets>=3,<5" "modelscope>=1.20,<2" addict oss2 simplejson

export APPTAINERENV_PYTHONPATH="$PYTHON_PACKAGES:$PROJECT_DIR"
[[ -n "${HF_TOKEN:-}" ]] && export APPTAINERENV_HF_TOKEN="$HF_TOKEN"

apptainer exec --nv \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --bind "$JOB_TMP:$JOB_TMP" \
  "$CONTAINER" \
  python3 -u "$PROJECT_DIR/cluster/prepare_external_pilot.py" \
    --output-dir "$OUTPUT_DIR" \
    --manifest-dir "$MANIFEST_DIR" \
    "$@"
