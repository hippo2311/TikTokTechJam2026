#!/bin/bash
#SBATCH --job-name=techjam-dis-eval
#SBATCH --partition=gpu
#SBATCH --nodelist=xgph0
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100-80:1
#SBATCH --output=slurm/logs/techjam-dis-eval-%j.out
#SBATCH --error=slurm/logs/techjam-dis-eval-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/TechJam}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/Dataset}"
MANIFEST_DIR="${MANIFEST_DIR:-$DATA_DIR/manifests}"
ENV_DIR="${ENV_DIR:-$PROJECT_DIR/.envs/a100-cu121-py312}"
MATCHED_ROOT="${MATCHED_ROOT:-$PROJECT_DIR/cluster_outputs/matched_three_seed_robustness_20260829}"
EVAL_SEED="${EVAL_SEED:-42}"

CANDIDATES=(disagreement_seed42 disagreement_seed43 disagreement_seed44)
CHECKPOINTS=(
  "$PROJECT_DIR/cluster_outputs/disagreement_full_seed42_20260829/best_ensemble_calibrated.pt"
  "$PROJECT_DIR/cluster_outputs/disagreement_full_seed43_20260829/best_ensemble_calibrated.pt"
  "$PROJECT_DIR/cluster_outputs/disagreement_full_seed44_20260829/best_ensemble_calibrated.pt"
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
for checkpoint in "${CHECKPOINTS[@]}"; do test -s "$checkpoint"; done
for candidate in "${CANDIDATES[@]}"; do
  if [[ -e "$MATCHED_ROOT/$candidate" ]]; then
    echo "Refusing to reuse matched output: $MATCHED_ROOT/$candidate" >&2
    exit 2
  fi
done
mkdir -p slurm/logs
export PYTHONPATH="$PROJECT_DIR"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for index in "${!CANDIDATES[@]}"; do
  candidate="${CANDIDATES[$index]}"
  checkpoint="${CHECKPOINTS[$index]}"
  mkdir -p "$MATCHED_ROOT/$candidate"
  for source in "${SOURCES[@]}"; do
    echo "[matched-eval] candidate=$candidate source=$source"
    "$ENV_DIR/bin/python" -u -m aigc_detector.robustness \
      --data-dir "$DATA_DIR" \
      --manifest-dir "$MANIFEST_DIR" \
      --checkpoint "$checkpoint" \
      --source "$source" \
      --split val \
      --max-samples 1000 \
      --batch-size 64 \
      --num-workers 8 \
      --device cuda \
      --seed "$EVAL_SEED" \
      --conditions "${CONDITIONS[@]}" \
      --output-json "$MATCHED_ROOT/$candidate/${source}_validation.json"
  done
done

"$ENV_DIR/bin/python" -u cluster/compare_matched_robustness.py \
  --root "$MATCHED_ROOT" \
  --candidates seed43 "${CANDIDATES[@]}" \
  --sources "${SOURCES[@]}" \
  --reference seed43 \
  --expected-seed "$EVAL_SEED" \
  --output-json "$MATCHED_ROOT/seed43_vs_disagreement_full_seeds.json"
