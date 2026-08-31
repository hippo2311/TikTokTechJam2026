#!/bin/bash
#SBATCH --job-name=techjam-final-eval
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nv:1
#SBATCH --output=slurm/logs/techjam-final-eval-%j.out
#SBATCH --error=slurm/logs/techjam-final-eval-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/TechJam}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/Dataset}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_DIR/cluster_outputs/disagreement_full_seed43_20260829/best_ensemble_calibrated.pt}"
MANIFEST="${MANIFEST:-$DATA_DIR/WildFake/final_test/manifests/final_test.jsonl}"
UNIQUE_MANIFEST="${UNIQUE_MANIFEST:-$DATA_DIR/WildFake/final_test/manifests/final_test_unique.jsonl}"
OUTPUT_JSON="${OUTPUT_JSON:-$PROJECT_DIR/cluster_outputs/final_test/final_test_metrics.json}"
UNIQUE_OUTPUT_JSON="${UNIQUE_OUTPUT_JSON:-$PROJECT_DIR/cluster_outputs/final_test/final_test_unique_metrics.json}"
CONTAINER="${CONTAINER:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"

if [[ "${CONFIRM_FINAL_TEST:-NO}" != "YES" ]]; then
  echo "Refusing to evaluate the final holdout without CONFIRM_FINAL_TEST=YES." >&2
  exit 2
fi

cd "$PROJECT_DIR"
mkdir -p slurm/logs "$(dirname "$OUTPUT_JSON")" "$PROJECT_DIR/.apptainer-cache"
test -s "$CHECKPOINT"
test -s "$MANIFEST"
test -s "$UNIQUE_MANIFEST"
test ! -e "$OUTPUT_JSON"
test ! -e "$UNIQUE_OUTPUT_JSON"
export APPTAINER_CACHEDIR="$PROJECT_DIR/.apptainer-cache"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"

JOB_TMP="${SLURM_TMPDIR:-/tmp/$USER-techjam-final-eval-$SLURM_JOB_ID}"
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

apptainer exec --nv --bind "$PROJECT_DIR:$PROJECT_DIR" --bind "$JOB_TMP:$JOB_TMP" \
  "$CONTAINER" python3 -u -m aigc_detector.evaluate_external_holdout \
  --data-dir "$DATA_DIR" \
  --manifest "$MANIFEST" \
  --checkpoint "$CHECKPOINT" \
  --output-json "$OUTPUT_JSON" \
  --batch-size 16 \
  --num-workers 8 \
  --device cuda \
  --conditions clean

apptainer exec --nv --bind "$PROJECT_DIR:$PROJECT_DIR" --bind "$JOB_TMP:$JOB_TMP" \
  "$CONTAINER" python3 -u -m aigc_detector.evaluate_external_holdout \
  --data-dir "$DATA_DIR" \
  --manifest "$UNIQUE_MANIFEST" \
  --checkpoint "$CHECKPOINT" \
  --output-json "$UNIQUE_OUTPUT_JSON" \
  --batch-size 16 \
  --num-workers 8 \
  --device cuda \
  --conditions clean

echo "[done] final holdout evaluated once: $OUTPUT_JSON"
echo "[done] duplicate-aware supplementary result: $UNIQUE_OUTPUT_JSON"
