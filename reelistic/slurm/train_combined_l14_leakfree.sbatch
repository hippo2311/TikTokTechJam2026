#!/bin/bash
#SBATCH --job-name=techjam-l14-clean
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nv:1
#SBATCH --output=slurm/logs/techjam-l14-clean-%j.out
#SBATCH --error=slurm/logs/techjam-l14-clean-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/TechJam}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/Dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/cluster_outputs/combined_l14_leakfree_reproduction}"
CONTAINER="${CONTAINER:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"

cd "$PROJECT_DIR"
mkdir -p slurm/logs "$OUTPUT_DIR" "$PROJECT_DIR/.apptainer-cache"
export APPTAINER_CACHEDIR="$PROJECT_DIR/.apptainer-cache"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"

JOB_TMP="${SLURM_TMPDIR:-/tmp/$USER-techjam-$SLURM_JOB_ID}"
PYTHON_PACKAGES="$JOB_TMP/techjam-python"
mkdir -p "$PYTHON_PACKAGES"

apptainer exec --nv \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --bind "$JOB_TMP:$JOB_TMP" \
  "$CONTAINER" \
  python3 -m pip install --quiet --no-cache-dir --no-deps \
    --target "$PYTHON_PACKAGES" "timm==1.0.28"
apptainer exec --nv \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --bind "$JOB_TMP:$JOB_TMP" \
  "$CONTAINER" \
  python3 -m pip install --quiet --no-cache-dir \
    --target "$PYTHON_PACKAGES" "huggingface_hub>=0.34,<1.0"

export APPTAINERENV_PYTHONPATH="$PYTHON_PACKAGES:$PROJECT_DIR"
export APPTAINERENV_TORCH_HOME="$TORCH_HOME"
export APPTAINERENV_HF_HOME="$HF_HOME"

echo "[leakfree] validating split and calibration safeguards"
apptainer exec --nv \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --bind "$JOB_TMP:$JOB_TMP" \
  "$CONTAINER" \
  python3 -m unittest tests.test_calibration tests.test_core.DataSplitTests

echo "[leakfree] training from pretrained backbones without a project checkpoint"
echo "[leakfree] physical_batch=$BATCH_SIZE accumulation=$GRAD_ACCUM_STEPS effective=$((BATCH_SIZE * GRAD_ACCUM_STEPS))"
nvidia-smi
apptainer exec --nv \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --bind "$JOB_TMP:$JOB_TMP" \
  "$CONTAINER" \
  python3 -u "$PROJECT_DIR/cluster/train_cluster.py" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --num-workers 8 \
    --lr 5e-5 \
    --device cuda \
    --semantic-backbone clip_vit_l14 \
    --selection-metric mean_source_auc \
    --skip-test \
    --log-interval 100
