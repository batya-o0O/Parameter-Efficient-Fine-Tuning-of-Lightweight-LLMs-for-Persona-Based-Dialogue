#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

cd "$ROOT_DIR"
source "$VENV_DIR/bin/activate"

export PYTHONPATH="$ROOT_DIR"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT_DIR/.cache/datasets}"

python src/eval/evaluate_rolebench_paper_routed.py \
  --clusters "${CLUSTERS_FILE:-data/clusters_k5.json}" \
  --adapters-root "${ADAPTERS_ROOT:-outputs/phi3_rolebench_phase2_clusters_k5}" \
  --output "${OUTPUT_JSON:-outputs/rolebench_paper_eval_k5.json}" \
  --model-id "${MODEL_ID:-microsoft/Phi-3-mini-4k-instruct}" \
  --dataset-repo "${DATASET_REPO:-ZenMoore/RoleBench}" \
  --benchmark-root "${BENCHMARK_ROOT:-rolebench-eng/role-generalization}" \
  --raw-instructions-file "${RAW_INSTRUCTIONS_FILE:-instructions-eng/instructions-general.jsonl}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
  --max-input-length "${MAX_INPUT_LENGTH:-2048}" \
  --use-4bit \
  --trust-remote-code
