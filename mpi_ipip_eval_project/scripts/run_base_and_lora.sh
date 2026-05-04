#!/usr/bin/env bash
set -euo pipefail
BASE_MODEL="microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH="outputs/phi3_role_lora"
ITEMS="data/mpi_ipip_items_sample.jsonl"
mkdir -p results
python src/mpi_eval.py --backend hf --base_model "$BASE_MODEL" --items_path "$ITEMS" --model_tag base --prompt_name target_profile --persona_prompt "You are a persona with high openness, high conscientiousness, low extraversion, high agreeableness, and low neuroticism. Answer the questionnaire as this persona." --output_prefix results/base_target_profile
python src/mpi_eval.py --backend hf --base_model "$BASE_MODEL" --adapter_path "$ADAPTER_PATH" --items_path "$ITEMS" --model_tag lora --prompt_name target_profile --persona_prompt "You are a persona with high openness, high conscientiousness, low extraversion, high agreeableness, and low neuroticism. Answer the questionnaire as this persona." --output_prefix results/lora_target_profile
