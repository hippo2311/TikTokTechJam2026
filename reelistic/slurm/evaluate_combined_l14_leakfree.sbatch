#!/bin/bash
#SBATCH --job-name=techjam-l14-robust
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nv:1
#SBATCH --output=slurm/logs/techjam-l14-robust-%j.out
#SBATCH --error=slurm/logs/techjam-l14-robust-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/TechJam}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_DIR/cluster_outputs/combined_l14_leakfree/best_ensemble_calibrated.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/cluster_outputs/combined_l14_leakfree/evaluation}"
CONTAINER="${CONTAINER:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
CONDITIONS=(
  clean color_shift center_crop_80
  downsample_50 downsample_25
  blur_0.5 blur_1.0 blur_2.0
  jpeg_90 jpeg_70 jpeg_50 jpeg_30
  noise_0.02 noise_0.05 noise_0.10
)

cd "$PROJECT_DIR"
mkdir -p slurm/logs "$OUTPUT_DIR" "$PROJECT_DIR/.apptainer-cache"
test -s "$CHECKPOINT"
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

echo "[evaluation] validating exact source filtering"
apptainer exec --nv \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --bind "$JOB_TMP:$JOB_TMP" \
  "$CONTAINER" \
  python3 -m unittest \
    tests.test_core.DataSplitTests.test_robustness_source_filter_is_balanced_and_deterministic

for SOURCE in birdy654 external_pilot; do
  echo "[evaluation] source=$SOURCE checkpoint=$CHECKPOINT"
  apptainer exec --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --bind "$JOB_TMP:$JOB_TMP" \
    "$CONTAINER" \
    python3 -u -m aigc_detector.robustness \
      --data-dir "$PROJECT_DIR/Dataset" \
      --checkpoint "$CHECKPOINT" \
      --source "$SOURCE" \
      --split val \
      --max-samples 1000 \
      --batch-size 16 \
      --num-workers 8 \
      --device cuda \
      --conditions "${CONDITIONS[@]}" \
      --output-json "$OUTPUT_DIR/${SOURCE}_validation.json"
done

echo "[done] checkpoint=$CHECKPOINT output=$OUTPUT_DIR"
