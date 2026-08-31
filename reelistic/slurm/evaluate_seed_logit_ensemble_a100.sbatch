#!/bin/bash
#SBATCH --job-name=techjam-logit-ens
#SBATCH --partition=gpu
#SBATCH --nodelist=xgph0
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100-80:1
#SBATCH --output=slurm/logs/techjam-logit-ens-%j.out
#SBATCH --error=slurm/logs/techjam-logit-ens-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/TechJam}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/Dataset}"
MANIFEST_DIR="${MANIFEST_DIR:-$DATA_DIR/manifests}"
ENV_DIR="${ENV_DIR:-$PROJECT_DIR/.envs/a100-cu121-py312}"
MATCHED_ROOT="${MATCHED_ROOT:-$PROJECT_DIR/cluster_outputs/matched_three_seed_robustness_20260829}"
OUTPUT_DIR="${OUTPUT_DIR:-$MATCHED_ROOT/logit_ensemble}"
EVAL_SEED="${EVAL_SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
MAX_CALIBRATION_SAMPLES="${MAX_CALIBRATION_SAMPLES:-6000}"
BATCH_SIZE="${BATCH_SIZE:-64}"

CHECKPOINTS=(
  "$PROJECT_DIR/cluster_outputs/three_source_l14_full_20260828/best_ensemble_calibrated.pt"
  "$PROJECT_DIR/cluster_outputs/three_source_l14_seed43_full_20260829/best_ensemble_calibrated.pt"
  "$PROJECT_DIR/cluster_outputs/three_source_l14_seed44_full_20260829/best_ensemble_calibrated.pt"
)
SOURCES=(birdy654 external_pilot WildFake)
CONDITIONS=(
  clean color_shift center_crop_80
  downsample_50 downsample_25
  blur_0.5 blur_1.0 blur_2.0
  jpeg_90 jpeg_70 jpeg_50 jpeg_30
  noise_0.02 noise_0.05 noise_0.10
)

cd "$PROJECT_DIR"
test -x "$ENV_DIR/bin/python"
test -f "$ENV_DIR/.ready"
test -d "$MATCHED_ROOT/seed43"
for checkpoint in "${CHECKPOINTS[@]}"; do test -s "$checkpoint"; done
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to reuse ensemble output: $OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p slurm/logs
export PYTHONPATH="$PROJECT_DIR"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi
"$ENV_DIR/bin/python" -u cluster/evaluate_seed_logit_ensemble.py \
  --data-dir "$DATA_DIR" \
  --manifest-dir "$MANIFEST_DIR" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --sources "${SOURCES[@]}" \
  --conditions "${CONDITIONS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$EVAL_SEED" \
  --max-samples "$MAX_SAMPLES" \
  --max-calibration-samples "$MAX_CALIBRATION_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --num-workers 8 \
  --device cuda

"$ENV_DIR/bin/python" -u cluster/compare_matched_robustness.py \
  --root "$MATCHED_ROOT" \
  --candidates seed43 logit_ensemble \
  --sources "${SOURCES[@]}" \
  --reference seed43 \
  --expected-seed "$EVAL_SEED" \
  --output-json "$MATCHED_ROOT/seed43_vs_logit_ensemble.json"
