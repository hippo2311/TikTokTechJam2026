#!/bin/bash
#SBATCH --job-name=techjam-final-data
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:nv:1
#SBATCH --array=0-1%1
#SBATCH --output=slurm/logs/techjam-final-data-%A_%a.out
#SBATCH --error=slurm/logs/techjam-final-data-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/TechJam}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/Dataset/WildFake}"
CONTAINER="${CONTAINER:-docker://python:3.11-slim}"
JOB_TMP="${SLURM_TMPDIR:-/tmp/$USER-techjam-final-data-$SLURM_JOB_ID}"
PYTHON_PACKAGES="$JOB_TMP/python-packages"

cd "$PROJECT_DIR"
mkdir -p slurm/logs "$OUTPUT_DIR" "$PYTHON_PACKAGES" "$PROJECT_DIR/.apptainer-cache"
export APPTAINER_CACHEDIR="$PROJECT_DIR/.apptainer-cache"

apptainer exec \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --bind "$JOB_TMP:$JOB_TMP" \
  "$CONTAINER" \
  python3 -m pip install --quiet --no-cache-dir \
    --target "$PYTHON_PACKAGES" "modelscope>=1.20,<2.0"

export APPTAINERENV_PYTHONPATH="$PYTHON_PACKAGES:$PROJECT_DIR"
export APPTAINERENV_PREPEND_PATH="$PYTHON_PACKAGES/bin"

echo "[final-data] index=$SLURM_ARRAY_TASK_ID output=$OUTPUT_DIR scratch=$JOB_TMP"
apptainer exec \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --bind "$JOB_TMP:$JOB_TMP" \
  "$CONTAINER" \
  python3 -u "$PROJECT_DIR/cluster/download_wildfake_archive.py" \
    --archive-set final-test \
    --archive-index "$SLURM_ARRAY_TASK_ID" \
    --output-dir "$OUTPUT_DIR" \
    --scratch-dir "$JOB_TMP"
