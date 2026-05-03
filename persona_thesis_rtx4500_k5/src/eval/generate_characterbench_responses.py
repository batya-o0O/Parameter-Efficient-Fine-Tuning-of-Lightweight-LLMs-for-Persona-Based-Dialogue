#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils.io import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CharacterBench English responses with a base model, one LoRA adapter, or routed cluster adapters."
    )
    parser.add_argument("--characterbench-dir", required=True, help="Path to the CharacterBench repo root.")
    parser.add_argument("--base-model-id", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--adapter-path", default=None, help="Optional single LoRA adapter directory.")
    parser.add_argument(
        "--adapters-root",
        default=None,
        help="Optional root directory containing cluster_<id> adapter folders for routed generation.",
    )
    parser.add_argument(
        "--clusters-file",
        default=None,
        help="Optional clusters_k5.json used for routed generation. If omitted, cluster_meta.json files are used.",
    )
    parser.add_argument(
        "--routing-embed-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model used to route CharacterBench personas to the nearest cluster.",
    )
    parser.add_argument("--model-label", required=True, help="Label used in output directories.")
    parser.add_argument("--raw-data-dir", default=None, help="Defaults to <characterbench-dir>/eval_data/raw_data")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <characterbench-dir>/eval_data/response_data_<model_label>",
    )
    parser.add_argument("--only-files", nargs="*", default=None, help="Optional list of raw_data filenames to process.")
    parser.add_argument("--max-sessions-per-file", type=int, default=None, help="Optional cap for quick pilots.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--use-4bit", action="store_true", help="Load the model in 4-bit mode.")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


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


def load_model_and_tokenizer(
    base_model_id: str,
    adapter_path: Optional[str],
    use_4bit: bool,
    trust_remote_code: bool,
):
    base_path = Path(base_model_id)
    base_is_local = base_path.exists()
    model_source = str(base_path.resolve()) if base_is_local else base_model_id

    # LoRA adapters should normally reuse the base model tokenizer.
    # Some exported adapter folders contain partial or incompatible tokenizer
    # metadata, which can break AutoTokenizer loading for otherwise valid
    # adapters. Prefer the base model tokenizer unless the caller explicitly
    # points base_model_id at a local tokenizer snapshot.
    tokenizer_source = model_source
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=trust_remote_code,
        local_files_only=base_is_local,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "device_map": "auto",
        "dtype": pick_dtype(),
        "trust_remote_code": trust_remote_code,
        "attn_implementation": "eager",
        "local_files_only": base_is_local,
    }
    quant_config = make_quant_config(use_4bit)
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        if hasattr(model, "config"):
            model.config.use_cache = False
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.use_cache = False
    model.eval()
    return tokenizer, model


def norm_session_en(session: dict) -> dict:
    session_en = session["translation_en"]
    messages = []
    for message in session_en["dialogue"]:
        speaker = message["speaker"].lstrip()
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
                raise ValueError(f"Invalid role marker: {speaker}")
        messages.append({"role": role, "content": message["utterance"]})

    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Expected the final English dialogue turn to be a user turn.")

    greeting = session_en.get("greeting", "")
    if greeting == "N/A":
        greeting = ""
    if greeting:
        if messages[0]["role"] != "user":
            raise ValueError("Expected the first role to be user when a greeting exists.")
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
    remainder = prompt_block[first_triple + 3 :]
    second_triple = remainder.index('"""')
    return remainder[:second_triple]


def build_prompt(session: dict, roleplay_prompt_en: str) -> str:
    dialogue = "\n".join(f"{message['role']}: {message['content']}" for message in session["messages"])
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
    rendered = (
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if hasattr(tokenizer, "apply_chat_template")
        else prompt
    )

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


def iter_files(raw_data_dir: Path, only_files: Optional[Iterable[str]]) -> list[Path]:
    if only_files:
        return [raw_data_dir / name for name in only_files]
    return sorted(raw_data_dir.glob("*.json"))


def normalize_name(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def load_cluster_specs(clusters_file: Optional[str], adapters_root: Path) -> dict[str, dict]:
    specs: dict[str, dict] = {}

    if clusters_file and Path(clusters_file).exists():
        clusters_payload = load_json(clusters_file)
        raw_clusters = clusters_payload["clusters"]
        for cluster_id, members in raw_clusters.items():
            roles = []
            descriptions = []
            for member in members:
                if isinstance(member, dict):
                    role = member.get("persona_id") or member.get("role") or member.get("description")
                    desc = member.get("description") or role
                else:
                    role = str(member)
                    desc = role
                roles.append(role)
                descriptions.append(desc)
            specs[str(cluster_id)] = {"roles": roles, "descriptions": descriptions}

    for cluster_dir in sorted(adapters_root.glob("cluster_*")):
        cluster_id = cluster_dir.name.replace("cluster_", "")
        meta_path = cluster_dir / "cluster_meta.json"
        if meta_path.exists():
            meta = load_json(str(meta_path))
            roles = list(meta.get("roles", []))
        else:
            roles = []
        spec = specs.setdefault(cluster_id, {"roles": roles, "descriptions": roles[:]})
        if not spec["roles"] and roles:
            spec["roles"] = roles
        if not spec["descriptions"]:
            spec["descriptions"] = spec["roles"][:]
        spec["adapter_path"] = str(cluster_dir.resolve())

    missing = [cluster_id for cluster_id, spec in specs.items() if "adapter_path" not in spec]
    if missing:
        raise FileNotFoundError(f"Missing adapter directories for clusters: {missing}")

    if not specs:
        raise FileNotFoundError(f"No cluster adapters found under {adapters_root}")

    return specs


def build_router(clusters_file: Optional[str], adapters_root: str, embed_model: str):
    adapters_root_path = Path(adapters_root).resolve()
    specs = load_cluster_specs(clusters_file, adapters_root_path)

    exact_map: dict[str, str] = {}
    cluster_ids = sorted(specs.keys(), key=lambda value: int(value) if value.isdigit() else value)

    embedder = SentenceTransformer(embed_model)
    centroids = []
    ordered_specs = []
    for cluster_id in cluster_ids:
        spec = specs[cluster_id]
        ordered_specs.append((cluster_id, spec))
        for role in spec["roles"]:
            exact_map[normalize_name(role)] = cluster_id

        descs = [desc for desc in spec["descriptions"] if desc]
        if not descs:
            descs = spec["roles"]
        embeddings = embedder.encode(descs, normalize_embeddings=True, show_progress_bar=False)
        centroid = np.asarray(embeddings).mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids.append(centroid)

    centroid_matrix = np.vstack(centroids)
    return {
        "specs": specs,
        "exact_map": exact_map,
        "embedder": embedder,
        "cluster_ids": [cluster_id for cluster_id, _ in ordered_specs],
        "centroid_matrix": centroid_matrix,
    }


def route_character(router, character_name: str, character_profile: str) -> dict:
    normalized = normalize_name(character_name)
    if normalized in router["exact_map"]:
        cluster_id = router["exact_map"][normalized]
        return {
            "cluster_id": cluster_id,
            "reason": "exact_name_match",
            "score": 1.0,
            "matched_role": character_name,
        }

    query = f"{character_name}\n{character_profile}".strip()
    query_embedding = router["embedder"].encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = np.dot(router["centroid_matrix"], np.asarray(query_embedding))
    best_index = int(np.argmax(scores))
    cluster_id = router["cluster_ids"][best_index]
    return {
        "cluster_id": cluster_id,
        "reason": "nearest_centroid_profile",
        "score": float(scores[best_index]),
        "matched_role": None,
    }


def make_output_session(session: dict, response_en: str, routing: Optional[dict]) -> dict:
    session_copy = json.loads(json.dumps(session, ensure_ascii=False))
    session_copy.setdefault("translation_en", {})
    session_copy["translation_en"]["response_en"] = response_en
    if routing is not None:
        session_copy["routing_info"] = routing
    return session_copy


def run_single_model_mode(args, raw_files: list[Path], roleplay_prompt_en: str, output_dir: Path) -> None:
    tokenizer, model = load_model_and_tokenizer(
        args.base_model_id,
        args.adapter_path,
        args.use_4bit,
        args.trust_remote_code,
    )

    for file_path in raw_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {file_path}")

        out_path = output_dir / file_path.name
        data = json.loads(file_path.read_text(encoding="utf-8"))
        output = []
        total = min(len(data), args.max_sessions_per_file) if args.max_sessions_per_file is not None else len(data)
        progress = tqdm(data[:total] if args.max_sessions_per_file is not None else data, desc=file_path.name, leave=True)

        for index, session in enumerate(progress):
            try:
                session_en = norm_session_en(session)
                prompt_en = build_prompt(session_en, roleplay_prompt_en)
                response_en = extract_response(generate_text(model, tokenizer, prompt_en, args.max_new_tokens))
            except Exception as exc:
                print(f"Skipping session {index} in {file_path.name}: {exc}")
                continue

            output.append(make_output_session(session, response_en, routing=None))
            progress.set_postfix(saved=len(output))

        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(output)} sessions to {out_path}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_routed_mode(args, raw_files: list[Path], roleplay_prompt_en: str, output_dir: Path) -> None:
    router = build_router(args.clusters_file, args.adapters_root, args.routing_embed_model)
    tasks_by_cluster: dict[str, list[dict]] = defaultdict(list)
    results_by_file: dict[str, list[tuple[int, dict]]] = defaultdict(list)

    for file_path in raw_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {file_path}")

        data = json.loads(file_path.read_text(encoding="utf-8"))
        total = min(len(data), args.max_sessions_per_file) if args.max_sessions_per_file is not None else len(data)
        progress = tqdm(
            data[:total] if args.max_sessions_per_file is not None else data,
            desc=f"route:{file_path.name}",
            leave=True,
        )
        for index, session in enumerate(progress):
            try:
                session_en = norm_session_en(session)
                prompt_en = build_prompt(session_en, roleplay_prompt_en)
                routing = route_character(
                    router,
                    session_en["dialogue_setting"]["bot_name"],
                    session_en["dialogue_setting"]["bot_profile"],
                )
            except Exception as exc:
                print(f"Skipping session {index} in {file_path.name}: {exc}")
                continue

            task = {
                "file_name": file_path.name,
                "index": index,
                "session": session,
                "prompt": prompt_en,
                "routing": routing,
            }
            tasks_by_cluster[routing["cluster_id"]].append(task)
            progress.set_postfix(cluster=routing["cluster_id"])

    for cluster_id in router["cluster_ids"]:
        cluster_tasks = tasks_by_cluster.get(cluster_id, [])
        if not cluster_tasks:
            continue

        adapter_path = router["specs"][cluster_id]["adapter_path"]
        print(f"[cluster {cluster_id}] loading adapter from {adapter_path} for {len(cluster_tasks)} CharacterBench sessions")
        tokenizer, model = load_model_and_tokenizer(
            args.base_model_id,
            adapter_path,
            args.use_4bit,
            args.trust_remote_code,
        )

        progress = tqdm(cluster_tasks, desc=f"cluster_{cluster_id}", leave=True)
        for task in progress:
            try:
                response_en = extract_response(generate_text(model, tokenizer, task["prompt"], args.max_new_tokens))
            except Exception as exc:
                print(f"Skipping routed session {task['file_name']}#{task['index']} in cluster {cluster_id}: {exc}")
                continue

            results_by_file[task["file_name"]].append(
                (
                    task["index"],
                    make_output_session(task["session"], response_en, routing=task["routing"]),
                )
            )
            progress.set_postfix(saved=len(results_by_file[task["file_name"]]))

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for file_path in raw_files:
        rows = results_by_file.get(file_path.name, [])
        rows.sort(key=lambda item: item[0])
        output = [row for _, row in rows]
        out_path = output_dir / file_path.name
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(output)} routed sessions to {out_path}")


def main() -> int:
    args = parse_args()
    if args.adapter_path and args.adapters_root:
        raise ValueError("Use either --adapter-path or --adapters-root, not both.")

    characterbench_dir = Path(args.characterbench_dir).resolve()
    raw_data_dir = (
        Path(args.raw_data_dir).resolve() if args.raw_data_dir else characterbench_dir / "eval_data" / "raw_data"
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else characterbench_dir / f"eval_data/response_data_{args.model_label}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    roleplay_prompt_en = load_roleplay_prompt(characterbench_dir)
    raw_files = iter_files(raw_data_dir, args.only_files)

    if args.adapters_root:
        run_routed_mode(args, raw_files, roleplay_prompt_en, output_dir)
    else:
        run_single_model_mode(args, raw_files, roleplay_prompt_en, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
