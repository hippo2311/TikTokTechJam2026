#!/bin/bash
#SBATCH --job-name=techjam-l14-a100
#SBATCH --partition=gpu
#SBATCH --nodelist=xgph0
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100-80:1
#SBATCH --output=slurm/logs/techjam-l14-a100-%j.out
#SBATCH --error=slurm/logs/techjam-l14-a100-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/TechJam}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/Dataset}"
MANIFEST_DIR="${MANIFEST_DIR:-$DATA_DIR/manifests}"
ENV_DIR="${ENV_DIR:-$PROJECT_DIR/.envs/a100-cu121-py312}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/cluster_outputs/a100_migration_smoke}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$PROJECT_DIR/cluster_outputs/combined_l14_leakfree/best_ensemble_calibrated.pt}"
RESUME="${RESUME:-}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-4000}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-1000}"
MAX_CALIBRATION_SAMPLES="${MAX_CALIBRATION_SAMPLES:-1000}"
EPOCH_SAMPLES="${EPOCH_SAMPLES:-4000}"
SEED="${SEED:-45}"
SEMANTIC_BACKBONE="${SEMANTIC_BACKBONE:-clip_vit_l14}"
TEXTURE_BACKBONE="${TEXTURE_BACKBONE:-resnet18}"

cd "$PROJECT_DIR"
mkdir -p slurm/logs "$OUTPUT_DIR" "$PROJECT_DIR/.cache/torch" "$PROJECT_DIR/.cache/huggingface"

test -x "$ENV_DIR/bin/python"
test -f "$ENV_DIR/.ready"
test -f "$MANIFEST_DIR/audit.json"
if [[ -n "$RESUME" ]]; then
  test -f "$RESUME"
  CHECKPOINT_ARGS=(--resume "$RESUME")
else
  test -f "$INIT_CHECKPOINT"
  CHECKPOINT_ARGS=(--init-checkpoint "$INIT_CHECKPOINT")
fi
if [[ -z "$RESUME" && ( -e "$OUTPUT_DIR/best_ensemble.pt" || -e "$OUTPUT_DIR/latest_training.pt" ) ]]; then
  echo "Refusing to reuse non-empty training output: $OUTPUT_DIR" >&2
  exit 2
fi

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

"$ENV_DIR/bin/python" -u "$PROJECT_DIR/cluster/train_cluster.py" \
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
  --log-interval 20
