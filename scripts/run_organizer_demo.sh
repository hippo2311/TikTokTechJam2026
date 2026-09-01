#!/usr/bin/env bash
# One-command Reelistic ONNX demo for organisers.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <image-or-directory> [results-json]" >&2
  echo "Example: $0 reelistic/data/example/Gemini_Generated_Image_ff8c3jff8c3jff8c.png" >&2
  exit 2
fi

INPUT_PATH="$1"
OUTPUT_PATH="${2:-}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="$PROJECT_ROOT/models/reelistic_dino/checkpoints/reelistic_dinov3.onnx"
VENV_PATH="$PROJECT_ROOT/.venv"

if [[ ! -f "$MODEL_PATH" ]]; then
  if ! command -v git-lfs >/dev/null 2>&1; then
    echo "Git LFS is required to retrieve the ONNX model. Install it, then rerun this script." >&2
    exit 1
  fi
  echo "Retrieving the ONNX model with Git LFS..."
  (cd "$PROJECT_ROOT" && git lfs pull --include "models/reelistic_dino/checkpoints/reelistic_dinov3.onnx")
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ONNX model was not found after Git LFS pull: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -d "$VENV_PATH" ]]; then python3 -m venv "$VENV_PATH"; fi
"$VENV_PATH/bin/python" -m pip install --quiet --upgrade pip
"$VENV_PATH/bin/python" -m pip install --quiet -r "$PROJECT_ROOT/requirements-onnx.txt"
if [[ -n "$OUTPUT_PATH" ]]; then
  "$VENV_PATH/bin/python" "$PROJECT_ROOT/scripts/run_onnx_demo.py" "$INPUT_PATH" --output "$OUTPUT_PATH"
else
  "$VENV_PATH/bin/python" "$PROJECT_ROOT/scripts/run_onnx_demo.py" "$INPUT_PATH"
fi
