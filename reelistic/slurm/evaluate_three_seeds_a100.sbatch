#!/bin/bash
#SBATCH --job-name=techjam-seeds-a100
#SBATCH --partition=gpu
#SBATCH --nodelist=xgph0
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100-80:1
#SBATCH --output=slurm/logs/techjam-seeds-a100-%j.out
#SBATCH --error=slurm/logs/techjam-seeds-a100-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/TechJam}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/Dataset}"
MANIFEST_DIR="${MANIFEST_DIR:-$DATA_DIR/manifests}"
ENV_DIR="${ENV_DIR:-$PROJECT_DIR/.envs/a100-cu121-py312}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/cluster_outputs/matched_three_seed_robustness}"
EVAL_SEED="${EVAL_SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-64}"

CANDIDATES=(seed42 seed43 seed44)
SOURCES=(birdy654 external_pilot WildFake)
CONDITIONS=(
  clean color_shift center_crop_80
  downsample_50 downsample_25
  blur_0.5 blur_1.0 blur_2.0
  jpeg_90 jpeg_70 jpeg_50 jpeg_30
  noise_0.02 noise_0.05 noise_0.10
)
declare -A CHECKPOINTS=(
  [seed42]="$PROJECT_DIR/cluster_outputs/three_source_l14_full_20260828/best_ensemble_calibrated.pt"
  [seed43]="$PROJECT_DIR/cluster_outputs/three_source_l14_seed43_full_20260829/best_ensemble_calibrated.pt"
  [seed44]="$PROJECT_DIR/cluster_outputs/three_source_l14_seed44_full_20260829/best_ensemble_calibrated.pt"
)

cd "$PROJECT_DIR"
test -x "$ENV_DIR/bin/python"
test -f "$ENV_DIR/.ready"
test -s "$MANIFEST_DIR/validation.jsonl"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to reuse matched-evaluation output: $OUTPUT_DIR" >&2
  exit 2
fi
for candidate in "${CANDIDATES[@]}"; do
  test -s "${CHECKPOINTS[$candidate]}"
done

mkdir -p slurm/logs "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi
"$ENV_DIR/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available()
print(f"[runtime] torch={torch.__version__}")
print(f"[runtime] device={torch.cuda.get_device_name(0)}")
print(f"[runtime] vram_gib={torch.cuda.get_device_properties(0).total_memory / 2**30:.1f}")
PY

for candidate in "${CANDIDATES[@]}"; do
  candidate_dir="$OUTPUT_DIR/$candidate"
  mkdir -p "$candidate_dir"
  for source in "${SOURCES[@]}"; do
    echo "[evaluation] candidate=$candidate source=$source seed=$EVAL_SEED"
    "$ENV_DIR/bin/python" -u -m aigc_detector.robustness \
      --data-dir "$DATA_DIR" \
      --manifest-dir "$MANIFEST_DIR" \
      --checkpoint "${CHECKPOINTS[$candidate]}" \
      --source "$source" \
      --split val \
      --max-samples "$MAX_SAMPLES" \
      --batch-size "$BATCH_SIZE" \
      --num-workers 8 \
      --device cuda \
      --seed "$EVAL_SEED" \
      --conditions "${CONDITIONS[@]}" \
      --output-json "$candidate_dir/${source}_validation.json"
  done
done

"$ENV_DIR/bin/python" -u cluster/compare_matched_robustness.py \
  --root "$OUTPUT_DIR" \
  --candidates "${CANDIDATES[@]}" \
  --sources "${SOURCES[@]}" \
  --reference seed42 \
  --expected-seed "$EVAL_SEED" \
  --output-json "$OUTPUT_DIR/comparison.json"
