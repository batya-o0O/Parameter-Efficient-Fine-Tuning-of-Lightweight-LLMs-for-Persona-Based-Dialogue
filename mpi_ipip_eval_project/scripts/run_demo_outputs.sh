#!/usr/bin/env bash
set -euo pipefail
ITEMS="data/mpi_ipip_items_sample.jsonl"
mkdir -p example_outputs example_outputs/summary
python src/mpi_eval.py --backend demo --items_path "$ITEMS" --model_tag base --prompt_name target_profile --output_prefix example_outputs/base_target_profile
python src/mpi_eval.py --backend demo --items_path "$ITEMS" --model_tag lora --prompt_name target_profile --output_prefix example_outputs/lora_target_profile
for MODEL_TAG in base lora; do
  for TRAIT in O C E A N; do
    for LEVEL in high low; do
      python src/mpi_eval.py --backend demo --items_path "$ITEMS" --model_tag "$MODEL_TAG" --prompt_name "${LEVEL}_${TRAIT}" --output_prefix "example_outputs/${MODEL_TAG}_${LEVEL}_${TRAIT}"
    done
  done
done
python src/compare_mpi_results.py --results_dir example_outputs --target_profile configs/target_profile_example.json --output_dir example_outputs/summary
