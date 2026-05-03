#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

pick_python() {
  if command -v python3.11 >/dev/null 2>&1; then
    echo "python3.11"
    return
  fi
  if command -v python3.10 >/dev/null 2>&1; then
    echo "python3.10"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return
  fi
  echo "python3 is required but was not found." >&2
  exit 1
}

PYTHON_BIN="$(pick_python)"

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" --version

cd "$ROOT_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
python -m pip install -r requirements.txt

mkdir -p "$ROOT_DIR/.cache/hf" "$ROOT_DIR/.cache/datasets" "$ROOT_DIR/data" "$ROOT_DIR/outputs"

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
PY
