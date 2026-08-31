#!/bin/bash
#SBATCH --job-name=techjam-l14-three
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nv:1
#SBATCH --output=slurm/logs/techjam-l14-three-%j.out
#SBATCH --error=slurm/logs/techjam-l14-three-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/TechJam}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/Dataset}"
MANIFEST_DIR="${MANIFEST_DIR:-$DATA_DIR/manifests}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/cluster_outputs/three_source_l14}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$PROJECT_DIR/cluster_outputs/combined_l14_leakfree/best_ensemble_calibrated.pt}"
RESUME="${RESUME:-}"
CONTAINER="${CONTAINER:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-30000}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-10000}"
MAX_CALIBRATION_SAMPLES="${MAX_CALIBRATION_SAMPLES:-10000}"
EPOCH_SAMPLES="${EPOCH_SAMPLES:-30000}"
SEED="${SEED:-42}"
SEMANTIC_BACKBONE="${SEMANTIC_BACKBONE:-clip_vit_l14}"
TEXTURE_BACKBONE="${TEXTURE_BACKBONE:-resnet18}"

cd "$PROJECT_DIR"
mkdir -p slurm/logs "$OUTPUT_DIR" "$PROJECT_DIR/.apptainer-cache"
export APPTAINER_CACHEDIR="$PROJECT_DIR/.apptainer-cache"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"

JOB_TMP="${SLURM_TMPDIR:-/tmp/$USER-techjam-three-$SLURM_JOB_ID}"
PYTHON_PACKAGES="$JOB_TMP/techjam-python"
mkdir -p "$PYTHON_PACKAGES"

apptainer exec --nv --bind "$PROJECT_DIR:$PROJECT_DIR" --bind "$JOB_TMP:$JOB_TMP" \
  "$CONTAINER" python3 -m pip install --quiet --no-cache-dir --no-deps \
  --target "$PYTHON_PACKAGES" "timm==1.0.28"
apptainer exec --nv --bind "$PROJECT_DIR:$PROJECT_DIR" --bind "$JOB_TMP:$JOB_TMP" \
  "$CONTAINER" python3 -m pip install --quiet --no-cache-dir \
  --target "$PYTHON_PACKAGES" "huggingface_hub>=0.34,<1.0"

export APPTAINERENV_PYTHONPATH="$PYTHON_PACKAGES:$PROJECT_DIR"
export APPTAINERENV_TORCH_HOME="$TORCH_HOME"
export APPTAINERENV_HF_HOME="$HF_HOME"

test -f "$MANIFEST_DIR/audit.json"
if [[ -n "$RESUME" ]]; then
  test -f "$RESUME"
  CHECKPOINT_ARGS=(--resume "$RESUME")
else
  test -f "$INIT_CHECKPOINT"
  CHECKPOINT_ARGS=(--init-checkpoint "$INIT_CHECKPOINT")
  if [[ -e "$OUTPUT_DIR/best_ensemble.pt" || -e "$OUTPUT_DIR/latest_training.pt" ]]; then
    echo "Refusing to reuse a non-empty training output: $OUTPUT_DIR" >&2
    exit 2
  fi
fi
nvidia-smi

apptainer exec --nv --bind "$PROJECT_DIR:$PROJECT_DIR" --bind "$JOB_TMP:$JOB_TMP" \
  "$CONTAINER" python3 -u "$PROJECT_DIR/cluster/train_cluster.py" \
  --data-dir "$DATA_DIR" \
  --manifest-dir "$MANIFEST_DIR" \
  --balanced-sampling \
  --epoch-samples "$EPOCH_SAMPLES" \
  --output-dir "$OUTPUT_DIR" \
  "${CHECKPOINT_ARGS[@]}" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --max-train-samples "$MAX_TRAIN_SAMPLES" \
  --max-val-samples "$MAX_VAL_SAMPLES" \
  --max-calibration-samples "$MAX_CALIBRATION_SAMPLES" \
  --seed "$SEED" \
  --num-workers 8 \
  --lr 2e-5 \
  --clean-probability 0.25 \
  --robustness-focus \
  --fusion-reliability-weight 0.25 \
  --device cuda \
  --semantic-backbone "$SEMANTIC_BACKBONE" \
  --texture-backbone "$TEXTURE_BACKBONE" \
  --selection-metric robust_source_auc \
  --skip-test \
  --log-interval 100
