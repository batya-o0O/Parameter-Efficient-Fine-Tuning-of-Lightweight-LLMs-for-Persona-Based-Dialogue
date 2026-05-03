#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score CharacterBench responses with CharacterJudge.")
    parser.add_argument("--characterbench-dir", required=True)
    parser.add_argument("--responses-dir", required=True, help="Directory created by generate_characterbench_responses.py")
    parser.add_argument("--judge-model-id", default="thu-coai/CharacterJudge")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--use-4bit", action="store_true")
    return parser.parse_args()


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def make_quant_config(use_4bit: bool):
    if not use_4bit or not torch.cuda.is_available():
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=pick_dtype(),
    )


def load_causal_lm(model_id: str, use_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "device_map": "auto",
        "dtype": pick_dtype(),
        "trust_remote_code": False,
        "attn_implementation": "eager",
    }
    quant_config = make_quant_config(use_4bit)
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False
    model.eval()
    return tokenizer, model


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if hasattr(tokenizer, "apply_chat_template") else prompt
    inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=3072)
    device = getattr(model, "device", None) or next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_eval_data(path: Path) -> list[dict]:
    data = read_json(path)
    rows = []
    for index, item in enumerate(data):
        rows.append({"id": index, "input": item["instruction"], "output": item["output"]})
    return sorted(rows, key=lambda row: len(row["input"]))


def normalize_generated(generated: str) -> float:
    first_line = generated.split("\n")[0].strip()
    match = re.search(r"(\d+(?:\.\d{1,3})?)", first_line)
    return float(match.group(1)) if match else 3.0


def aggregate_scores(result_dir: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    average = 0.0
    files = sorted(result_dir.glob("*.jsonl"))
    for file_path in files:
        rows = [json.loads(line) for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        predictions = [normalize_generated(row["generated"]) for row in rows]
        gold_scores = [float(row["output"]) for row in rows]
        max_gold = max(gold_scores)
        metric = (float(np.mean(predictions)) / max_gold) * 5
        key = file_path.stem.replace(file_path.stem.split("_")[0] + "_", "")
        result[key] = metric
        average += metric
    result["average"] = average / len(files) if files else 0.0
    return result


def main() -> int:
    args = parse_args()
    characterbench_dir = Path(args.characterbench_dir).resolve()
    responses_dir = Path(args.responses_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_dir = output_dir / "evaluation_prompts"
    construct_dir = characterbench_dir / "construct_prompts"
    subprocess.run(
        [
            "python",
            str(construct_dir / "process_wo_context_en_all.py"),
            "--data_path",
            str(responses_dir),
            "--output_path",
            str(evaluation_dir),
            "--model_name",
            responses_dir.name,
        ],
        check=True,
        cwd=str(construct_dir),
    )

    tokenizer, model = load_causal_lm(args.judge_model_id, args.use_4bit)
    judge_results_dir = output_dir / "judge_results"
    judge_results_dir.mkdir(parents=True, exist_ok=True)

    for eval_file in sorted(evaluation_dir.glob("*.json")):
        rows = read_eval_data(eval_file)
        out_path = judge_results_dir / f"{eval_file.stem}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for row in tqdm(rows, desc=eval_file.stem, leave=True):
                generated = generate_text(model, tokenizer, row["input"], args.max_new_tokens)
                handle.write(json.dumps({**row, "generated": generated}, ensure_ascii=False) + "\n")

    summary = aggregate_scores(judge_results_dir)
    write_json(output_dir / "characterbench_summary.json", summary)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
