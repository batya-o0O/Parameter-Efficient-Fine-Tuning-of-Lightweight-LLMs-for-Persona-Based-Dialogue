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

CHARACTERBENCH_DIR="${CHARACTERBENCH_DIR:-$ROOT_DIR/CharacterBench}"
BASE_MODEL_ID="${BASE_MODEL_ID:-microsoft/Phi-3-mini-4k-instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
MAX_SESSIONS_PER_FILE="${MAX_SESSIONS_PER_FILE:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-256}"

python src/eval/generate_characterbench_responses.py \
  --characterbench-dir "$CHARACTERBENCH_DIR" \
  --base-model-id "$BASE_MODEL_ID" \
  --model-label "${BASE_LABEL:-phi3_base}" \
  --max-sessions-per-file "$MAX_SESSIONS_PER_FILE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --use-4bit \
  --trust-remote-code

python src/eval/evaluate_characterbench_judge.py \
  --characterbench-dir "$CHARACTERBENCH_DIR" \
  --responses-dir "$CHARACTERBENCH_DIR/eval_data/response_data_${BASE_LABEL:-phi3_base}" \
  --output-dir "$ROOT_DIR/outputs/characterbench_${BASE_LABEL:-phi3_base}" \
  --max-new-tokens "$JUDGE_MAX_NEW_TOKENS" \
  --use-4bit

if [[ -n "$ADAPTER_PATH" ]]; then
  python src/eval/generate_characterbench_responses.py \
    --characterbench-dir "$CHARACTERBENCH_DIR" \
    --base-model-id "$BASE_MODEL_ID" \
    --adapter-path "$ADAPTER_PATH" \
    --model-label "${ADAPTER_LABEL:-phi3_adapter}" \
    --max-sessions-per-file "$MAX_SESSIONS_PER_FILE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --use-4bit \
    --trust-remote-code

  python src/eval/evaluate_characterbench_judge.py \
    --characterbench-dir "$CHARACTERBENCH_DIR" \
    --responses-dir "$CHARACTERBENCH_DIR/eval_data/response_data_${ADAPTER_LABEL:-phi3_adapter}" \
    --output-dir "$ROOT_DIR/outputs/characterbench_${ADAPTER_LABEL:-phi3_adapter}" \
    --max-new-tokens "$JUDGE_MAX_NEW_TOKENS" \
    --use-4bit
fi
