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
ADAPTERS_ROOT="${ADAPTERS_ROOT:-$ROOT_DIR/outputs/phi3_rolebench_phase2_clusters_k5}"
CLUSTERS_FILE="${CLUSTERS_FILE:-$ROOT_DIR/data/clusters_k5.json}"
ROUTED_LABEL="${ROUTED_LABEL:-phi3_routed_k5}"
MAX_SESSIONS_PER_FILE="${MAX_SESSIONS_PER_FILE:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-256}"
ROUTING_EMBED_MODEL="${ROUTING_EMBED_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

EXTRA_ARGS=()
if [[ -f "$CLUSTERS_FILE" ]]; then
  EXTRA_ARGS+=(--clusters-file "$CLUSTERS_FILE")
fi

python src/eval/generate_characterbench_responses.py \
  --characterbench-dir "$CHARACTERBENCH_DIR" \
  --base-model-id "$BASE_MODEL_ID" \
  --adapters-root "$ADAPTERS_ROOT" \
  --routing-embed-model "$ROUTING_EMBED_MODEL" \
  --model-label "$ROUTED_LABEL" \
  --max-sessions-per-file "$MAX_SESSIONS_PER_FILE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --use-4bit \
  --trust-remote-code \
  "${EXTRA_ARGS[@]}"

python src/eval/evaluate_characterbench_judge.py \
  --characterbench-dir "$CHARACTERBENCH_DIR" \
  --responses-dir "$CHARACTERBENCH_DIR/eval_data/response_data_${ROUTED_LABEL}" \
  --output-dir "$ROOT_DIR/outputs/characterbench_${ROUTED_LABEL}" \
  --max-new-tokens "$JUDGE_MAX_NEW_TOKENS" \
  --use-4bit
