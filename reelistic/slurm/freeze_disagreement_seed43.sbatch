#!/bin/bash
#SBATCH --job-name=techjam-freeze
#SBATCH --partition=gpu
#SBATCH --nodelist=xgph0
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100-80:1
#SBATCH --output=slurm/logs/techjam-freeze-%j.out
#SBATCH --error=slurm/logs/techjam-freeze-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/TechJam}"
ENV_DIR="${ENV_DIR:-$PROJECT_DIR/.envs/a100-cu121-py312}"
MODEL_DIR="$PROJECT_DIR/cluster_outputs/disagreement_full_seed43_20260829"
MATCHED_DIR="$PROJECT_DIR/cluster_outputs/matched_three_seed_robustness_20260829"
DIAGNOSIS_DIR="$PROJECT_DIR/cluster_outputs/disagreement_seed43_gate2_gate3_20260829"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/cluster_outputs/frozen_disagreement_seed43_final_20260829}"

cd "$PROJECT_DIR"
test -x "$ENV_DIR/bin/python"
test -s "$MODEL_DIR/best_ensemble_calibrated.pt"
test -s "$MODEL_DIR/best_ensemble.pt"
test -s "$MATCHED_DIR/seed43_vs_disagreement_full_seeds.json"
test -s "$DIAGNOSIS_DIR/diagnosis.json"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to reuse frozen output: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p slurm/logs
export PYTHONPATH="$PROJECT_DIR"

"$ENV_DIR/bin/python" -m unittest tests.test_freeze_package
"$ENV_DIR/bin/python" -u cluster/freeze_development_package.py \
  --project-dir "$PROJECT_DIR" \
  --checkpoint "$MODEL_DIR/best_ensemble_calibrated.pt" \
  --uncalibrated-checkpoint "$MODEL_DIR/best_ensemble.pt" \
  --manifest-dir "$PROJECT_DIR/Dataset/manifests" \
  --output-dir "$OUTPUT_DIR" \
  --selection-name disagreement_full_seed43 \
  --threshold 0.5 \
  --parameter-count 315129796 \
  --report "matched_comparison=$MATCHED_DIR/seed43_vs_disagreement_full_seeds.json" \
  --report "selected_diagnosis=$DIAGNOSIS_DIR/diagnosis.json" \
  --report "bounded_gate_comparison=$MATCHED_DIR/seed43_vs_gate_disagreement.json" \
  --report "rejected_logit_ensemble=$MATCHED_DIR/seed43_vs_logit_ensemble.json"
