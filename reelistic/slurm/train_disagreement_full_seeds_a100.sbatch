#!/bin/bash
#SBATCH --job-name=techjam-dis-full
#SBATCH --partition=gpu
#SBATCH --nodelist=xgph0
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100-80:1
#SBATCH --array=0-2%1
#SBATCH --output=slurm/logs/techjam-dis-full-%A_%a.out
#SBATCH --error=slurm/logs/techjam-dis-full-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/TechJam}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/Dataset}"
MANIFEST_DIR="${MANIFEST_DIR:-$DATA_DIR/manifests}"
ENV_DIR="${ENV_DIR:-$PROJECT_DIR/.envs/a100-cu121-py312}"

SEEDS=(42 43 44)
INIT_CHECKPOINTS=(
  "$PROJECT_DIR/cluster_outputs/three_source_l14_full_20260828/best_ensemble_calibrated.pt"
  "$PROJECT_DIR/cluster_outputs/three_source_l14_seed43_full_20260829/best_ensemble_calibrated.pt"
  "$PROJECT_DIR/cluster_outputs/three_source_l14_seed44_full_20260829/best_ensemble_calibrated.pt"
)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
INIT_CHECKPOINT="${INIT_CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="$PROJECT_DIR/cluster_outputs/disagreement_full_seed${SEED}_20260829"

cd "$PROJECT_DIR"
test -x "$ENV_DIR/bin/python"
test -f "$ENV_DIR/.ready"
test -s "$INIT_CHECKPOINT"
test -s "$MANIFEST_DIR/validation.jsonl"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to reuse full-seed output: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p slurm/logs
export PYTHONPATH="$PROJECT_DIR"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[full-seed] task=$SLURM_ARRAY_TASK_ID seed=$SEED init=$INIT_CHECKPOINT"
"$ENV_DIR/bin/python" -u cluster/train_cluster.py \
  --data-dir "$DATA_DIR" \
  --manifest-dir "$MANIFEST_DIR" \
  --balanced-sampling \
  --epoch-samples 100000 \
  --output-dir "$OUTPUT_DIR" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --epochs 5 \
  --batch-size 64 \
  --grad-accum-steps 1 \
  --max-train-samples 0 \
  --max-val-samples 15000 \
  --max-calibration-samples 6000 \
  --seed "$SEED" \
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

test -s "$OUTPUT_DIR/best_ensemble_calibrated.pt"
echo "[done-full-seed] seed=$SEED checkpoint=$OUTPUT_DIR/best_ensemble_calibrated.pt"
