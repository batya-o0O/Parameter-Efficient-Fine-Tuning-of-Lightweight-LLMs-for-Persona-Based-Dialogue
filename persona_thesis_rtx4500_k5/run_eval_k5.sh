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

python src/eval/evaluate_rolebench_routed.py \
  --clusters "${CLUSTERS_FILE:-data/clusters_k5.json}" \
  --adapters-root "${ADAPTERS_ROOT:-outputs/phi3_rolebench_phase2_clusters_k5}" \
  --output "${OUTPUT_JSON:-outputs/rolebench_routed_eval_k5.json}" \
  --model-id "${MODEL_ID:-microsoft/Phi-3-mini-4k-instruct}" \
  --dataset-repo "${DATASET_REPO:-ZenMoore/RoleBench}" \
  --train-file "${TRAIN_FILE:-rolebench-eng/role-generalization/general/train.jsonl}" \
  --eval-file "${EVAL_FILE:-rolebench-eng/role-generalization/general/test.jsonl}" \
  --global-adapter-path "${GLOBAL_ADAPTER_PATH:-}" \
  --global-label "${GLOBAL_LABEL:-phase1_global}" \
  --paper-results "${PAPER_RESULTS:-}" \
  --max-length "${MAX_LENGTH:-1024}" \
  --batch-size "${EVAL_BATCH_SIZE:-1}" \
  --use-4bit \
  --trust-remote-code
