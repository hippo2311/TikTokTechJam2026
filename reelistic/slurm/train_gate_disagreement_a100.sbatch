#!/bin/bash
#SBATCH --job-name=techjam-disagree
#SBATCH --partition=gpu
#SBATCH --nodelist=xgph0
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100-80:1
#SBATCH --output=slurm/logs/techjam-disagree-%j.out
#SBATCH --error=slurm/logs/techjam-disagree-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/TechJam}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/Dataset}"
MANIFEST_DIR="${MANIFEST_DIR:-$DATA_DIR/manifests}"
ENV_DIR="${ENV_DIR:-$PROJECT_DIR/.envs/a100-cu121-py312}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$PROJECT_DIR/cluster_outputs/three_source_l14_seed43_full_20260829/best_ensemble_calibrated.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/cluster_outputs/gate_disagreement_seed43_20260829}"
MATCHED_ROOT="${MATCHED_ROOT:-$PROJECT_DIR/cluster_outputs/matched_three_seed_robustness_20260829}"
CANDIDATE_NAME="${CANDIDATE_NAME:-gate_disagreement}"
CANDIDATE_DIR="$MATCHED_ROOT/$CANDIDATE_NAME"

cd "$PROJECT_DIR"
test -x "$ENV_DIR/bin/python"
test -f "$ENV_DIR/.ready"
test -s "$INIT_CHECKPOINT"
test -s "$MANIFEST_DIR/validation.jsonl"
if [[ -e "$OUTPUT_DIR" || -e "$CANDIDATE_DIR" ]]; then
  echo "Refusing to reuse Gate-4 output directories." >&2
  exit 2
fi
mkdir -p slurm/logs
export PYTHONPATH="$PROJECT_DIR"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$ENV_DIR/bin/python" -m unittest \
  tests.test_phases_1_to_4.ImprovedArchitectureTests.test_explicit_disagreement_is_bounded_and_zero_initialized \
  tests.test_operating_metrics

"$ENV_DIR/bin/python" -u cluster/train_cluster.py \
  --data-dir "$DATA_DIR" \
  --manifest-dir "$MANIFEST_DIR" \
  --balanced-sampling \
  --epoch-samples 100000 \
  --output-dir "$OUTPUT_DIR" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --epochs 2 \
  --batch-size 64 \
  --grad-accum-steps 1 \
  --max-train-samples 300000 \
  --max-val-samples 15000 \
  --max-calibration-samples 6000 \
  --seed 43 \
  --num-workers 8 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --clean-probability 0.25 \
  --robustness-focus \
  --fusion-reliability-weight 0.25 \
  --device cuda \
  --semantic-backbone clip_vit_l14 \
  --selection-metric robust_source_auc \
  --explicit-gate-disagreement \
  --fusion-only-warmup \
  --skip-test \
  --log-interval 100

CHECKPOINT="$OUTPUT_DIR/best_ensemble_calibrated.pt"
test -s "$CHECKPOINT"
mkdir -p "$CANDIDATE_DIR"
SOURCES=(birdy654 external_pilot WildFake)
CONDITIONS=(
  clean color_shift center_crop_80
  downsample_50 downsample_25
  blur_0.5 blur_1.0 blur_2.0
  jpeg_90 jpeg_70 jpeg_50 jpeg_30
  noise_0.02 noise_0.05 noise_0.10
)
for source in "${SOURCES[@]}"; do
  echo "[matched-eval] candidate=$CANDIDATE_NAME source=$source"
  "$ENV_DIR/bin/python" -u -m aigc_detector.robustness \
    --data-dir "$DATA_DIR" \
    --manifest-dir "$MANIFEST_DIR" \
    --checkpoint "$CHECKPOINT" \
    --source "$source" \
    --split val \
    --max-samples 1000 \
    --batch-size 64 \
    --num-workers 8 \
    --device cuda \
    --seed 42 \
    --conditions "${CONDITIONS[@]}" \
    --output-json "$CANDIDATE_DIR/${source}_validation.json"
done

"$ENV_DIR/bin/python" -u cluster/compare_matched_robustness.py \
  --root "$MATCHED_ROOT" \
  --candidates seed43 "$CANDIDATE_NAME" \
  --sources "${SOURCES[@]}" \
  --reference seed43 \
  --expected-seed 42 \
  --output-json "$MATCHED_ROOT/seed43_vs_${CANDIDATE_NAME}.json"
