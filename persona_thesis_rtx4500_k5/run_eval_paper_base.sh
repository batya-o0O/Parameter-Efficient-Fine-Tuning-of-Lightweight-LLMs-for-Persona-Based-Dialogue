#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

cd "$ROOT_DIR"
source "$VENV_DIR/bin/activate"

export PYTHONPATH="$ROOT_DIR"
export HF_HUB_DISABLE_XET=1
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT_DIR/.cache/datasets}"

python src/eval/evaluate_rolebench_paper_routed.py \
  --eval-mode base \
  --output "${OUTPUT_JSON:-outputs/rolebench_paper_eval_base_partial.json}" \
  --model-id "${MODEL_ID:-$ROOT_DIR/.cache/hf/models--microsoft--Phi-3-mini-4k-instruct/snapshots/f39ac1d28e925b323eae81227eaba4464caced4e}" \
  --dataset-repo "${DATASET_REPO:-ZenMoore/RoleBench}" \
  --benchmark-root "${BENCHMARK_ROOT:-rolebench-eng/instruction-generalization}" \
  --raw-instructions-file "${RAW_INSTRUCTIONS_FILE:-instructions-eng/instructions-general.jsonl}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-128}" \
  --max-input-length "${MAX_INPUT_LENGTH:-1024}" \
  --max-examples-per-split "${MAX_EXAMPLES_PER_SPLIT:-60}" \
  --save-every "${SAVE_EVERY:-25}" \
  --use-4bit \
  --trust-remote-code
