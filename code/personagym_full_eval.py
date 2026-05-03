#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


TASK_ORDER = [
    "Expected Action",
    "Toxicity",
    "Linguistic Habits",
    "Persona Consistency",
    "Action Justification",
]

TASK_CANONICAL_MAP = {
    "Expected Action": "expected_action",
    "Toxicity": "toxicity_control",
    "Linguistic Habits": "linguistic_habits",
    "Persona Consistency": "persona_consistency",
    "Action Justification": "action_justification",
}

TASK_NAME_MAP = {
    "expected_action": "Expected Action",
    "toxicity_control": "Toxicity",
    "linguistic_habits": "Linguistic Habits",
    "persona_consistency": "Persona Consistency",
    "action_justification": "Action Justification",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and optionally evaluate full PersonaGym runs for a base model, "
            "a single full adapter, or routed clustered adapters."
        )
    )
    parser.add_argument("--mode", required=True, choices=["base", "adapter", "routed"])
    parser.add_argument(
        "--benchmark-dir",
        default="PersonaGym-master/PersonaGym-master/questions/benchmark-v1",
        help="PersonaGym benchmark-v1 directory.",
    )
    parser.add_argument(
        "--personagym-code-dir",
        default="PersonaGym-master/PersonaGym-master/code",
        help="PersonaGym code directory containing run.py and api_keys.py.",
    )
    parser.add_argument("--base-model-id", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--adapter-path", default=None, help="Single full-dataset LoRA adapter path.")
    parser.add_argument(
        "--adapters-root",
        default=None,
        help="Directory containing cluster_<id> LoRA adapters for routed mode.",
    )
    parser.add_argument(
        "--clusters-file",
        default="persona_thesis_rtx4500_k5/data/clusters_k5.json",
        help="Cluster file used to build routing centroids.",
    )
    parser.add_argument(
        "--routing-embed-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model used for routed cluster selection.",
    )
    parser.add_argument("--output-root", default="responses", help="Root directory for PersonaGym outputs.")
    parser.add_argument("--model-label", default=None, help="Optional explicit label for outputs.")
    parser.add_argument("--persona-start", type=int, default=0)
    parser.add_argument("--persona-end", type=int, default=None)
    parser.add_argument("--max-personas", type=int, default=None)
    parser.add_argument(
        "--max-questions-per-task",
        type=int,
        default=10,
        help="Full PersonaGym uses 10. Lower this for cheaper pilots.",
    )
    parser.add_argument("--max-input-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Run PersonaGym rubric scoring after generation.")
    parser.add_argument(
        "--judge-model",
        default="gpt-5-nano",
        help="Default PersonaGym GPT judge model used unless environment overrides it.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse already-generated JSONL rows and skip completed examples.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


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

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=trust_remote_code,
        local_files_only=base_is_local,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": pick_dtype(),
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


def generate_text(
    model,
    tokenizer,
    persona_text: str,
    prompt: str,
    max_input_length: int,
    max_new_tokens: int,
) -> str:
    system_prompt = (
        f"Adopt the identity of {persona_text}. "
        "Answer the questions while staying in strict accordance with the nature of this identity."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    rendered = (
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if hasattr(tokenizer, "apply_chat_template")
        else f"{system_prompt}\n\n{prompt}"
    )

    inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=max_input_length)
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


def normalize_name(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def load_benchmark_rows(
    benchmark_dir: Path,
    max_questions_per_task: int,
    persona_start: int,
    persona_end: Optional[int],
    max_personas: Optional[int],
) -> List[dict]:
    persona_files = sorted(benchmark_dir.glob("*.json"))
    selected = persona_files
    if max_personas is not None and max_personas > 0:
        selected = selected[:max_personas]
    start = max(persona_start, 0)
    end = persona_end if persona_end is not None else None
    selected = selected[start:end]

    rows: List[dict] = []
    for persona_path in selected:
        payload = load_json(persona_path)
        persona_text = persona_path.stem
        for task_name in TASK_ORDER:
            prompts = list(payload.get(task_name, []))[:max_questions_per_task]
            for index, prompt in enumerate(prompts):
                canonical = TASK_CANONICAL_MAP[task_name]
                rows.append(
                    {
                        "example_id": f"{persona_text}-{canonical}-{index}",
                        "persona_id": persona_text,
                        "persona_text": persona_text,
                        "task": canonical,
                        "prompt": str(prompt),
                        "source_file": str(persona_path.resolve()),
                    }
                )
    return rows


def load_cluster_specs(clusters_file: Optional[str], adapters_root: Path) -> dict[str, dict]:
    specs: dict[str, dict] = {}

    if clusters_file and Path(clusters_file).exists():
        clusters_payload = load_json(Path(clusters_file))
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
                if role:
                    roles.append(role)
                if desc:
                    descriptions.append(desc)
            specs[str(cluster_id)] = {"roles": roles, "descriptions": descriptions}

    for cluster_dir in sorted(adapters_root.glob("cluster_*")):
        cluster_id = cluster_dir.name.replace("cluster_", "")
        meta_path = cluster_dir / "cluster_meta.json"
        roles = []
        if meta_path.exists():
            meta = load_json(meta_path)
            roles = list(meta.get("roles", []))
        spec = specs.setdefault(cluster_id, {"roles": roles, "descriptions": roles[:]})
        if not spec["roles"] and roles:
            spec["roles"] = roles
        if not spec["descriptions"]:
            spec["descriptions"] = spec["roles"][:]
        spec["adapter_path"] = str(cluster_dir.resolve())

    if not specs:
        raise FileNotFoundError(f"No cluster adapters found under {adapters_root}")

    missing = [cluster_id for cluster_id, spec in specs.items() if "adapter_path" not in spec]
    if missing:
        raise FileNotFoundError(f"Missing adapter directories for clusters: {missing}")
    return specs


def build_router(clusters_file: Optional[str], adapters_root: Path, embed_model: str) -> dict:
    specs = load_cluster_specs(clusters_file, adapters_root)
    exact_map: dict[str, str] = {}
    cluster_ids = sorted(specs.keys(), key=lambda value: int(value) if value.isdigit() else value)

    embedder = SentenceTransformer(embed_model)
    centroids = []
    for cluster_id in cluster_ids:
        spec = specs[cluster_id]
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

    return {
        "specs": specs,
        "exact_map": exact_map,
        "embedder": embedder,
        "cluster_ids": cluster_ids,
        "centroid_matrix": np.vstack(centroids),
    }


def route_persona(router: dict, persona_text: str) -> dict:
    normalized = normalize_name(persona_text)
    if normalized in router["exact_map"]:
        cluster_id = router["exact_map"][normalized]
        return {
            "cluster_id": cluster_id,
            "reason": "exact_name_match",
            "score": 1.0,
        }

    query_embedding = router["embedder"].encode([persona_text], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = np.dot(router["centroid_matrix"], np.asarray(query_embedding))
    best_index = int(np.argmax(scores))
    cluster_id = router["cluster_ids"][best_index]
    return {
        "cluster_id": cluster_id,
        "reason": "nearest_centroid_persona_text",
        "score": float(scores[best_index]),
    }


def load_existing_example_ids(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    return {row["example_id"] for row in read_jsonl(jsonl_path)}


def infer_model_label(args: argparse.Namespace) -> str:
    if args.model_label:
        return args.model_label
    if args.mode == "base":
        return "personagym_base_full"
    if args.mode == "adapter":
        adapter_name = Path(args.adapter_path).name if args.adapter_path else "adapter"
        return f"personagym_{adapter_name}_full"
    return "personagym_clustered_routed_full"


def summarize_payload(rows: List[dict], model_label: str, mode: str) -> dict:
    task_counts = defaultdict(int)
    personas = set()
    for row in rows:
        task_counts[TASK_NAME_MAP[row["task"]]] += 1
        personas.add(row["persona_id"])
    return {
        "model_label": model_label,
        "mode": mode,
        "persona_count": len(personas),
        "task_counts": dict(task_counts),
        "personas": sorted(personas),
    }


def convert_rows(rows: List[dict]) -> Dict[str, Dict[str, List[List[str]]]]:
    persona_task_pairs: Dict[str, Dict[str, List[tuple[int, List[str]]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        example_id = row["example_id"]
        _, _, question_idx = example_id.rsplit("-", 2)
        task_name = TASK_NAME_MAP[row["task"]]
        persona_task_pairs[row["persona_id"]][task_name].append((int(question_idx), [row["prompt"], row["response"]]))

    converted: Dict[str, Dict[str, List[List[str]]]] = {}
    for persona, task_map in persona_task_pairs.items():
        converted[persona] = {}
        for task_name, qa_items in task_map.items():
            qa_items.sort(key=lambda item: item[0])
            converted[persona][task_name] = [qa for _, qa in qa_items]
    return converted


def write_personagym_files(persona_payloads: Dict[str, Dict[str, List[List[str]]]], output_dir: Path) -> None:
    ensure_dir(output_dir)
    for persona, payload in persona_payloads.items():
        (output_dir / f"{persona}_qa.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def write_json_atomic(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def load_api_key_file(api_key_file: Path) -> str:
    if not api_key_file.exists():
        return "missing"
    text = api_key_file.read_text(encoding="utf-8")
    if "Insert OpenAI key here" in text or "Insert Claude key here" in text or "Insert Llama key here" in text:
        return "placeholder"
    return "configured"


def summarize_scores(results: Dict[str, dict], model_label: str, mode: str) -> dict:
    by_task = defaultdict(list)
    for persona_scores in results.values():
        for key, value in persona_scores.items():
            by_task[key].append(value)
    averages = {
        key: (sum(values) / len(values) if values else 0.0)
        for key, values in by_task.items()
    }
    return {
        "model_label": model_label,
        "mode": mode,
        "persona_count": len(results),
        "average_scores": averages,
        "per_persona_scores": results,
    }


def summarize_progress(
    results: Dict[str, dict],
    partial_batches: Dict[str, Dict[str, List[float]]],
    model_label: str,
    mode: str,
) -> dict:
    payload = summarize_scores(results, model_label, mode)
    payload["partial_batches"] = partial_batches
    return payload


def evaluate_personagym(
    personagym_code_dir: Path,
    saved_responses_dir: Path,
    personas: List[str],
    progress_path: Path,
    model_label: str,
    mode: str,
    judge_model: str,
) -> Dict[str, dict]:
    personagym_code_dir = personagym_code_dir.resolve()
    saved_responses_dir = saved_responses_dir.resolve()
    progress_path = progress_path.resolve()

    os.environ.setdefault("PERSONAGYM_SETTINGS_MODEL", judge_model)
    os.environ.setdefault("PERSONAGYM_QUESTION_MODEL", judge_model)
    os.environ.setdefault("PERSONAGYM_EXAMPLE_MODEL", judge_model)
    os.environ.setdefault("PERSONAGYM_EVAL_MODEL", judge_model)

    api_status = load_api_key_file(personagym_code_dir / "api_keys.py")
    if api_status != "configured":
        raise RuntimeError(
            "PersonaGym API keys are not configured. Fill PersonaGym-master/PersonaGym-master/code/api_keys.py first."
        )

    existing_results: Dict[str, dict] = {}
    partial_batches: Dict[str, Dict[str, List[float]]] = {}
    if progress_path.exists():
        try:
            existing_payload = load_json(progress_path)
            existing_results = existing_payload.get("per_persona_scores", {}) or {}
            partial_batches = existing_payload.get("partial_batches", {}) or {}
            print(f"Found existing evaluation progress for {len(existing_results)} personas at {progress_path}")
        except Exception as exc:
            print(f"Could not load existing progress from {progress_path}: {exc}")

    completed_personas = set(existing_results.keys())
    original_cwd = Path.cwd()
    sys.path.insert(0, str(personagym_code_dir))
    os.chdir(personagym_code_dir)
    try:
        run_path = personagym_code_dir / "run.py"
        spec = importlib.util.spec_from_file_location("personagym_run", run_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load PersonaGym runner from {run_path}")
        personagym_run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(personagym_run)

        results: Dict[str, dict] = dict(existing_results)
        for persona in personas:
            if persona in completed_personas:
                continue

            task_to_qa = personagym_run.load_responses(persona, str(saved_responses_dir))
            persona_partial = partial_batches.get(persona, {})
            persona_scores: Dict[str, float] = {}

            for task in task_to_qa:
                task_batch_scores = list(persona_partial.get(task, []))
                total_batches = len(range(0, len(task_to_qa[task]), 5))

                for batch_index, i in enumerate(range(0, len(task_to_qa[task]), 5)):
                    if batch_index < len(task_batch_scores):
                        continue
                    selected_qa = task_to_qa[task][i : i + 5]
                    rubric = Path(f"../rubrics/{task}.txt").read_text(encoding="utf-8")
                    sys_prompt, scoring_prompt = personagym_run.format_rubrics(persona, rubric, selected_qa)
                    batch_score = personagym_run.score_rubrics(sys_prompt, scoring_prompt)
                    task_batch_scores.append(batch_score)

                    persona_partial[task] = task_batch_scores
                    partial_batches[persona] = persona_partial
                    write_json_atomic(progress_path, summarize_progress(results, partial_batches, model_label, mode))
                    print(
                        f"Saved batch progress: persona {persona!r}, task {task!r}, "
                        f"batch {batch_index + 1}/{total_batches}"
                    )

                if task_batch_scores:
                    persona_scores[task] = sum(task_batch_scores) / len(task_batch_scores)

            persona_scores["PersonaScore"] = (
                sum(persona_scores.values()) / len(persona_scores) if persona_scores else 0.0
            )
            results[persona] = persona_scores
            if persona in partial_batches:
                del partial_batches[persona]
            write_json_atomic(progress_path, summarize_progress(results, partial_batches, model_label, mode))
            print(f"Saved evaluation progress: {len(results)}/{len(personas)} personas")
        return results
    finally:
        os.chdir(original_cwd)


def run_single_model_mode(args: argparse.Namespace, rows: List[dict], jsonl_path: Path) -> None:
    existing_ids = load_existing_example_ids(jsonl_path) if args.resume_existing else set()
    pending_rows = [row for row in rows if row["example_id"] not in existing_ids]

    tokenizer, model = load_model_and_tokenizer(
        args.base_model_id,
        args.adapter_path if args.mode == "adapter" else None,
        args.use_4bit,
        args.trust_remote_code,
    )

    progress = tqdm(pending_rows, desc=args.mode, leave=True)
    for row in progress:
        response = generate_text(
            model,
            tokenizer,
            row["persona_text"],
            row["prompt"],
            args.max_input_length,
            args.max_new_tokens,
        )
        out_row = dict(row)
        out_row["model_label"] = infer_model_label(args)
        out_row["base_model_id"] = args.base_model_id
        out_row["adapter_path"] = args.adapter_path if args.mode == "adapter" else None
        out_row["response"] = response
        append_jsonl(jsonl_path, out_row)
        progress.set_postfix(saved=len(existing_ids) + progress.n + 1)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_routed_mode(args: argparse.Namespace, rows: List[dict], jsonl_path: Path) -> None:
    if not args.adapters_root:
        raise ValueError("--adapters-root is required for routed mode.")

    existing_ids = load_existing_example_ids(jsonl_path) if args.resume_existing else set()
    pending_rows = [row for row in rows if row["example_id"] not in existing_ids]

    router = build_router(args.clusters_file, Path(args.adapters_root).resolve(), args.routing_embed_model)
    rows_by_cluster: dict[str, List[dict]] = defaultdict(list)
    for row in pending_rows:
        routing = route_persona(router, row["persona_text"])
        routed_row = dict(row)
        routed_row["routing"] = routing
        rows_by_cluster[routing["cluster_id"]].append(routed_row)

    for cluster_id in router["cluster_ids"]:
        cluster_rows = rows_by_cluster.get(cluster_id, [])
        if not cluster_rows:
            continue

        adapter_path = router["specs"][cluster_id]["adapter_path"]
        print(f"[cluster {cluster_id}] loading adapter from {adapter_path} for {len(cluster_rows)} PersonaGym prompts")
        tokenizer, model = load_model_and_tokenizer(
            args.base_model_id,
            adapter_path,
            args.use_4bit,
            args.trust_remote_code,
        )

        progress = tqdm(cluster_rows, desc=f"cluster_{cluster_id}", leave=True)
        for row in progress:
            response = generate_text(
                model,
                tokenizer,
                row["persona_text"],
                row["prompt"],
                args.max_input_length,
                args.max_new_tokens,
            )
            out_row = dict(row)
            out_row["model_label"] = infer_model_label(args)
            out_row["base_model_id"] = args.base_model_id
            out_row["adapter_path"] = adapter_path
            out_row["cluster_id"] = cluster_id
            out_row["routing_reason"] = row["routing"]["reason"]
            out_row["routing_score"] = row["routing"]["score"]
            out_row["response"] = response
            del out_row["routing"]
            append_jsonl(jsonl_path, out_row)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    if args.mode == "adapter" and not args.adapter_path:
        raise ValueError("--adapter-path is required when --mode adapter is used.")

    model_label = infer_model_label(args)
    output_root = Path(args.output_root).resolve()
    run_dir = output_root / model_label
    ensure_dir(run_dir)

    benchmark_dir = Path(args.benchmark_dir).resolve()
    personagym_code_dir = Path(args.personagym_code_dir).resolve()
    rows = load_benchmark_rows(
        benchmark_dir=benchmark_dir,
        max_questions_per_task=args.max_questions_per_task,
        persona_start=args.persona_start,
        persona_end=args.persona_end,
        max_personas=args.max_personas,
    )
    if not rows:
        raise RuntimeError("No PersonaGym benchmark rows were selected.")

    jsonl_path = run_dir / f"{model_label}.jsonl"
    if args.mode == "routed":
        run_routed_mode(args, rows, jsonl_path)
    else:
        run_single_model_mode(args, rows, jsonl_path)

    generated_rows = read_jsonl(jsonl_path)
    qa_output_dir = run_dir / "saved_responses"
    persona_payloads = convert_rows(generated_rows)
    write_personagym_files(persona_payloads, qa_output_dir)

    personas = sorted(persona_payloads.keys())
    save_json(run_dir / "personas.json", personas)
    save_json(run_dir / "summary.json", summarize_payload(generated_rows, model_label, args.mode))

    if args.evaluate:
        progress_path = run_dir / "evaluation_progress.json"
        results = evaluate_personagym(
            personagym_code_dir=personagym_code_dir,
            saved_responses_dir=qa_output_dir,
            personas=personas,
            progress_path=progress_path,
            model_label=model_label,
            mode=args.mode,
            judge_model=args.judge_model,
        )
        final_summary = summarize_scores(results, model_label, args.mode)
        final_summary["judge_model"] = args.judge_model
        final_summary["benchmark_persona_count"] = len(personas)
        final_summary["max_questions_per_task"] = args.max_questions_per_task
        save_json(run_dir / "evaluation_summary.json", final_summary)
        print(json.dumps(final_summary, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(summarize_payload(generated_rows, model_label, args.mode), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
