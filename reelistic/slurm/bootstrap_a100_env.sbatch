#!/bin/bash
#SBATCH --job-name=techjam-a100-env
#SBATCH --partition=gpu
#SBATCH --nodelist=xgph0
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100-80:1
#SBATCH --output=slurm/logs/techjam-a100-env-%j.out
#SBATCH --error=slurm/logs/techjam-a100-env-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/TechJam}"
ENV_DIR="${ENV_DIR:-$PROJECT_DIR/.envs/a100-cu121-py312}"
READY_MARKER="$ENV_DIR/.ready"
BUILD_DIR="$PROJECT_DIR/.envs/a100-cu121-py312.build-$SLURM_JOB_ID"

cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/.envs" slurm/logs

if [[ -f "$READY_MARKER" ]]; then
  echo "[environment] reusing ready environment at $ENV_DIR"
  "$ENV_DIR/bin/python" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda={torch.cuda.is_available()}")
print(f"device={torch.cuda.get_device_name(0)}")
PY
  exit 0
fi

if [[ -e "$ENV_DIR" ]]; then
  echo "Refusing to overwrite incomplete environment: $ENV_DIR" >&2
  exit 2
fi
if [[ -e "$BUILD_DIR" ]]; then
  echo "Refusing to overwrite build directory: $BUILD_DIR" >&2
  exit 2
fi

echo "[environment] building in $BUILD_DIR"
python3 -m venv "$BUILD_DIR"
"$BUILD_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$BUILD_DIR/bin/python" -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1 torchvision==0.19.1
"$BUILD_DIR/bin/python" -m pip install --no-cache-dir \
  timm==1.0.28 "huggingface_hub>=0.34,<1.0" \
  "numpy>=1.26" "pillow>=10.0" "scikit-learn>=1.5"

"$BUILD_DIR/bin/python" - <<'PY'
import sklearn
import timm
import torch
import torchvision

assert torch.cuda.is_available(), "CUDA is not visible from the A100 environment"
print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"timm={timm.__version__}")
print(f"sklearn={sklearn.__version__}")
print(f"device={torch.cuda.get_device_name(0)}")
print(
    "vram_gib="
    f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f}"
)
PY

mv "$BUILD_DIR" "$ENV_DIR"
"$ENV_DIR/bin/python" -c "import sklearn, timm, torch, torchvision"
touch "$READY_MARKER"
echo "[done] environment=$ENV_DIR"
