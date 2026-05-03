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

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$ROOT_DIR/data" "$ROOT_DIR/outputs"

python src/cluster/extract_rolebench_personas.py \
  --dataset_repo "${DATASET_REPO:-ZenMoore/RoleBench}" \
  --train_file "${TRAIN_FILE:-rolebench-eng/role-generalization/general/train.jsonl}" \
  --mode "${PERSONA_MODE:-sampled_qa}" \
  --examples_per_role "${PERSONA_EXAMPLES_PER_ROLE:-8}" \
  --max_question_chars "${PERSONA_MAX_QUESTION_CHARS:-180}" \
  --max_answer_chars "${PERSONA_MAX_ANSWER_CHARS:-220}" \
  --seed "${PERSONA_SEED:-42}" \
  --out data/personas_rolebench.json

python src/cluster/cluster_personas.py \
  --personas data/personas_rolebench.json \
  --k 5 \
  --embed_model "${EMBED_MODEL:-sentence-transformers/all-MiniLM-L6-v2}" \
  --out data/clusters_k5.json

EXTRA_ARGS=()
if [[ "${USE_BF16:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--use_bf16)
fi
if [[ "${ENABLE_PACKING:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--packing)
fi
if [[ -n "${CLUSTER_IDS:-}" ]]; then
  EXTRA_ARGS+=(--cluster_ids "$CLUSTER_IDS")
fi

python src/train/train_cluster_adapters.py \
  --clusters data/clusters_k5.json \
  --output_root outputs/phi3_rolebench_phase2_clusters_k5 \
  --model_id "${MODEL_ID:-microsoft/Phi-3-mini-4k-instruct}" \
  --dataset_repo "${DATASET_REPO:-ZenMoore/RoleBench}" \
  --train_file "${TRAIN_FILE:-rolebench-eng/role-generalization/general/train.jsonl}" \
  --eval_file "${EVAL_FILE:-rolebench-eng/role-generalization/general/test.jsonl}" \
  --epochs "${EPOCHS:-1}" \
  --max_length "${MAX_LENGTH:-1024}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --grad_accum "${GRAD_ACCUM:-16}" \
  --lr "${LR:-2e-4}" \
  --save_steps "${SAVE_STEPS:-1000}" \
  --eval_steps "${EVAL_STEPS:-1000}" \
  --logging_steps "${LOGGING_STEPS:-25}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
  --dataloader_workers "${DATALOADER_WORKERS:-2}" \
  --group_by_length \
  --report_to "${REPORT_TO:-tensorboard}" \
  "${EXTRA_ARGS[@]}"
