#!/usr/bin/env bash
set -euo pipefail
BASE_MODEL="microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH="outputs/phi3_role_lora"
ITEMS="data/mpi_ipip_items_sample.jsonl"
mkdir -p results
for MODEL_TAG in base lora; do
  for TRAIT in O C E A N; do
    for LEVEL in high low; do
      EXTRA=""; if [ "$MODEL_TAG" = "lora" ]; then EXTRA="--adapter_path $ADAPTER_PATH"; fi
      python src/mpi_eval.py --backend hf --base_model "$BASE_MODEL" $EXTRA --items_path "$ITEMS" --model_tag "$MODEL_TAG" --prompt_name "${LEVEL}_${TRAIT}" --persona_prompt "${LEVEL} ${TRAIT} persona" --output_prefix "results/${MODEL_TAG}_${LEVEL}_${TRAIT}"
    done
  done
done
