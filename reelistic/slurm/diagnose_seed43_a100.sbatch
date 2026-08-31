#!/bin/bash
#SBATCH --job-name=techjam-seed43-diag
#SBATCH --partition=gpu
#SBATCH --nodelist=xgph0
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100-80:1
#SBATCH --output=slurm/logs/techjam-seed43-diag-%j.out
#SBATCH --error=slurm/logs/techjam-seed43-diag-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/TechJam}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/Dataset}"
MANIFEST_DIR="${MANIFEST_DIR:-$DATA_DIR/manifests}"
ENV_DIR="${ENV_DIR:-$PROJECT_DIR/.envs/a100-cu121-py312}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_DIR/cluster_outputs/three_source_l14_seed43_full_20260829/best_ensemble_calibrated.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/cluster_outputs/seed43_gate2_gate3_20260829}"

cd "$PROJECT_DIR"
test -x "$ENV_DIR/bin/python"
test -f "$ENV_DIR/.ready"
test -s "$CHECKPOINT"
test -s "$MANIFEST_DIR/validation.jsonl"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to reuse diagnostic output: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p slurm/logs
export PYTHONPATH="$PROJECT_DIR"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$ENV_DIR/bin/python" -m unittest \
  tests.test_operating_metrics \
  tests.test_core.DataSplitTests.test_robustness_source_filter_is_balanced_and_deterministic

"$ENV_DIR/bin/python" -u cluster/diagnose_selected_checkpoint.py \
  --data-dir "$DATA_DIR" \
  --manifest-dir "$MANIFEST_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output-json "$OUTPUT_DIR/diagnosis.json" \
  --sources birdy654 external_pilot WildFake \
  --seed 42 \
  --source-samples 5000 \
  --family-samples-per-class 1000 \
  --min-family-samples 100 \
  --weak-family-count 5 \
  --bootstrap-repetitions 300 \
  --batch-size 64 \
  --num-workers 8 \
  --device cuda
