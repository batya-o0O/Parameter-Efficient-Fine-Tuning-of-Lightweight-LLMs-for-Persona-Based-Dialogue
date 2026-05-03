#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CharacterBench English responses with Phi-3 base or LoRA.")
    p.add_argument("--characterbench-dir", required=True, help="Path to the CharacterBench repo root.")
    p.add_argument("--base-model-id", default="microsoft/Phi-3-mini-4k-instruct")
    p.add_argument("--adapter-path", default=None, help="Optional LoRA adapter path.")
    p.add_argument("--model-label", required=True, help="Label used in output directories.")
    p.add_argument("--raw-data-dir", default=None, help="Defaults to <characterbench-dir>/eval_data/raw_data")
    p.add_argument("--output-dir", default=None, help="Defaults to <characterbench-dir>/eval_data/response_data_<model_label>")
    p.add_argument("--only-files", nargs="*", default=None, help="Optional list of raw_data filenames to process.")
    p.add_argument("--max-sessions-per-file", type=int, default=None, help="Optional cap for quick pilots.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--use-4bit", action="store_true", help="Load Phi-3 in 4-bit mode.")
    return p.parse_args()


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def make_quant_config(use_4bit: bool) -> Optional[BitsAndBytesConfig]:
    if not use_4bit or not torch.cuda.is_available():
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=pick_dtype(),
    )


def load_model_and_tokenizer(base_model_id: str, adapter_path: Optional[str], use_4bit: bool):
    tokenizer_source = adapter_path if adapter_path and Path(adapter_path).exists() else base_model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=False)
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

    model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def norm_session_en(session: dict) -> dict:
    session_en = session["translation_en"]
    messages = []
    for msg in session_en["dialogue"]:
        speaker = msg["speaker"].lstrip()
        if speaker in {"user", "User"}:
            role = "user"
        elif speaker in {"character", "Character"}:
            role = "assistant"
        else:
            if speaker == session_en["user_name"].lstrip():
                role = "user"
            elif speaker == session_en["character_name"].lstrip() or speaker in session_en["character_name"].lstrip():
                role = "assistant"
            elif messages and messages[-1]["role"] == "user":
                role = "assistant"
            else:
                raise ValueError(f"invalid role = {speaker}")
        messages.append({"role": role, "content": msg["utterance"]})

    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Expected final English dialogue turn to be a user turn.")

    greeting = session_en.get("greeting", "")
    if greeting == "N/A":
        greeting = ""
    if greeting:
        if messages[0]["role"] != "user":
            raise ValueError("Expected first role to be user when greeting is present.")
        messages.insert(0, {"role": "assistant", "content": greeting})

    if messages[0]["role"] == "assistant":
        messages.insert(0, {"role": "user", "content": ""})

    return {
        "dialogue_setting": {
            "user_name": session_en["user_name"],
            "bot_name": session_en["character_name"],
            "user_profile": session_en.get("user_profile", ""),
            "bot_profile": session_en["character_profile"],
        },
        "messages": messages,
    }


def load_roleplay_prompt(characterbench_dir: Path) -> str:
    roleplay_path = characterbench_dir / "roleplay_prompt.py"
    text = roleplay_path.read_text(encoding="utf-8")
    start = text.index('Role_Play_PROMPT_EN =')
    prompt_block = text[start:]
    first_triple = prompt_block.index('"""')
    remainder = prompt_block[first_triple + 3:]
    second_triple = remainder.index('"""')
    return remainder[:second_triple]


def build_prompt(session: dict, roleplay_prompt_en: str) -> str:
    dialogue = "\n".join(f"{msg['role']}: {msg['content']}" for msg in session["messages"])
    return roleplay_prompt_en.format(
        character_profile=session["dialogue_setting"]["bot_profile"],
        user_profile=session["dialogue_setting"]["user_profile"],
        dialogue=dialogue,
    )


def extract_response(text: str) -> str:
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict) and "response" in payload:
            return str(payload["response"]).strip()
    except Exception:
        pass
    return cleaned.strip()


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        rendered = prompt

    inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=3072)
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def iter_files(raw_data_dir: Path, only_files: Optional[Iterable[str]]) -> List[Path]:
    if only_files:
        return [raw_data_dir / name for name in only_files]
    return sorted(raw_data_dir.glob("*.json"))


def main() -> int:
    args = parse_args()
    characterbench_dir = Path(args.characterbench_dir).resolve()
    raw_data_dir = Path(args.raw_data_dir).resolve() if args.raw_data_dir else characterbench_dir / "eval_data" / "raw_data"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else characterbench_dir / f"eval_data/response_data_{args.model_label}"
    output_dir.mkdir(parents=True, exist_ok=True)

    roleplay_prompt_en = load_roleplay_prompt(characterbench_dir)
    tokenizer, model = load_model_and_tokenizer(args.base_model_id, args.adapter_path, args.use_4bit)

    files = iter_files(raw_data_dir, args.only_files)
    for file_path in files:
        if not file_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {file_path}")

        out_path = output_dir / file_path.name
        data = json.loads(file_path.read_text(encoding="utf-8"))
        output = []
        for idx, session in enumerate(data):
            if args.max_sessions_per_file is not None and idx >= args.max_sessions_per_file:
                break

            try:
                session_en = norm_session_en(session)
                prompt_en = build_prompt(session_en, roleplay_prompt_en)
                response_en = extract_response(generate_text(model, tokenizer, prompt_en, args.max_new_tokens))
            except Exception as exc:
                print(f"Skipping session {idx} in {file_path.name}: {exc}")
                continue

            session_copy = json.loads(json.dumps(session, ensure_ascii=False))
            session_copy.setdefault("translation_en", {})
            session_copy["translation_en"]["response_en"] = response_en
            output.append(session_copy)

        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(output)} sessions to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
