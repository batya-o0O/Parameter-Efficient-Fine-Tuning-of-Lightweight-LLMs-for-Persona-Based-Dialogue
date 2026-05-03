#!/usr/bin/env python3
import argparse
import gc
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils.io import load_json, save_json


def parse_args():
    ap = argparse.ArgumentParser(
        description="Paper-style RoleBench evaluation with routed cluster adapters and Rouge-L."
    )
    ap.add_argument("--eval-mode", default="routed", choices=["routed", "base", "adapter"])
    ap.add_argument("--clusters", default=None, help="Path to clusters_k*.json")
    ap.add_argument("--adapters-root", default=None, help="Directory containing cluster_<id> adapter folders")
    ap.add_argument("--adapter-path", default=None, help="Path to a single LoRA adapter directory for --eval-mode adapter")
    ap.add_argument("--adapter-label", default="single_adapter")
    ap.add_argument("--output", required=True, help="Path to summary JSON output")
    ap.add_argument("--model-id", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--dataset-repo", default="ZenMoore/RoleBench")
    ap.add_argument("--benchmark-root", default="rolebench-eng/role-generalization")
    ap.add_argument("--general-test-file", default=None)
    ap.add_argument("--role-specific-test-file", default=None)
    ap.add_argument("--general-baseline-file", default=None)
    ap.add_argument("--role-specific-baseline-file", default=None)
    ap.add_argument("--raw-instructions-file", default="instructions-eng/instructions-general.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-input-length", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--use-4bit", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--max-examples-per-split", type=int, default=None)
    ap.add_argument("--cluster-ids", default="", help="Optional comma-separated subset, for example: 0,2")
    ap.add_argument("--save-every", type=int, default=50, help="Write progress snapshot every N generated rows")
    ap.add_argument("--skip-missing-adapters", action="store_true")
    return ap.parse_args()


def parse_cluster_ids(value):
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def first_generated(example):
    generated = example.get("generated", [])
    if isinstance(generated, list) and generated:
        return str(generated[0]).strip()
    if isinstance(generated, str):
        return generated.strip()
    return ""


def normalize_text(text):
    return " ".join(str(text).strip().split())


def normalize_key(text):
    return normalize_text(text).lower()


def lcs_length(a_tokens, b_tokens):
    if not a_tokens or not b_tokens:
        return 0
    prev = [0] * (len(b_tokens) + 1)
    for a_tok in a_tokens:
        curr = [0]
        for j, b_tok in enumerate(b_tokens, start=1):
            if a_tok == b_tok:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(prediction, reference):
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def pick_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def make_quant_config(use_4bit):
    if not use_4bit or not torch.cuda.is_available():
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=pick_dtype(),
    )


def load_base_model_and_tokenizer(model_id, use_4bit, trust_remote_code):
    model_path = Path(model_id)
    local_files_only = model_path.exists()
    model_source = str(model_path.resolve()) if local_files_only else model_id

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "device_map": "auto",
        "dtype": pick_dtype(),
        "trust_remote_code": trust_remote_code,
        "attn_implementation": "eager",
    }
    quant_config = make_quant_config(use_4bit)
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        local_files_only=local_files_only,
        **model_kwargs,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False
    model.eval()
    return tokenizer, model


def build_prompt(role, question):
    return (
        f"<|system|>\nYou are role-playing as {role}. Stay in character.\n<|end|>\n"
        f"<|user|>\n{question}\n<|end|>\n"
        f"<|assistant|>\n"
    )


def generate_response(model, tokenizer, role, question, max_input_length, max_new_tokens, temperature):
    prompt = build_prompt(role, question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    device = getattr(model, "device", None) or next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    do_sample = temperature > 0
    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = 0.9

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)

    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return normalize_text(text)


def build_cluster_router(clusters_payload):
    embed_model_name = clusters_payload["embed_model"]
    clusters = clusters_payload["clusters"]

    role_to_cluster = {}
    cluster_texts = {}
    for cluster_id, members in clusters.items():
        texts = []
        for member in members:
            role = member["persona_id"]
            desc = member.get("description", role)
            role_to_cluster[role] = cluster_id
            texts.append(desc)
        cluster_texts[cluster_id] = texts

    embedder = SentenceTransformer(embed_model_name)
    centroid_map = {}
    for cluster_id, texts in cluster_texts.items():
        embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        centroid = embeddings.mean(axis=0)
        norm = float((centroid ** 2).sum() ** 0.5)
        centroid_map[cluster_id] = centroid / norm if norm > 0 else centroid

    ordered_cluster_ids = sorted(
        centroid_map.keys(),
        key=lambda value: int(value) if str(value).isdigit() else str(value),
    )

    def route(role_text):
        if role_text in role_to_cluster:
            return role_to_cluster[role_text], "known_member"
        embedding = embedder.encode([role_text], normalize_embeddings=True, show_progress_bar=False)[0]
        best_cluster = None
        best_score = None
        for cluster_id in ordered_cluster_ids:
            score = float((embedding * centroid_map[cluster_id]).sum())
            if best_score is None or score > best_score:
                best_score = score
                best_cluster = cluster_id
        return best_cluster, "nearest_centroid"

    return route


def load_jsonl_from_repo(dataset_repo, relative_path):
    ds = load_dataset(dataset_repo, data_files={"data": relative_path})
    return list(ds["data"])


def default_path(root, suffix):
    return f"{root}/{suffix}"


def load_raw_reference_map(dataset_repo, raw_instructions_file):
    rows = load_jsonl_from_repo(dataset_repo, raw_instructions_file)
    mapping = {}
    for row in rows:
        instruction = row.get("instruction") or row.get("question")
        answer = row.get("answer") or first_generated(row)
        if instruction and answer:
            mapping[normalize_key(instruction)] = normalize_text(answer)
    return mapping


def prepare_general_rows(rows, raw_map, route_fn, max_examples=None):
    prepared = []
    missing_raw = 0
    for row in rows:
        role = row.get("role", "UnknownRole")
        question = row.get("question", "")
        cus_ref = first_generated(row)
        raw_ref = raw_map.get(normalize_key(question))
        if not question or not cus_ref:
            continue
        if not raw_ref:
            missing_raw += 1
            continue
        if route_fn is None:
            cluster_id, route_type = "base", "base_model"
        else:
            cluster_id, route_type = route_fn(role)
        prepared.append(
            {
                "split": "general",
                "role": role,
                "question": question,
                "cluster_id": cluster_id,
                "route_type": route_type,
                "refs": {"RAW": raw_ref, "CUS": normalize_text(cus_ref)},
            }
        )
        if max_examples is not None and len(prepared) >= max_examples:
            break
    return prepared, missing_raw


def prepare_role_specific_rows(rows, route_fn, max_examples=None):
    prepared = []
    for row in rows:
        role = row.get("role", "UnknownRole")
        question = row.get("question", "")
        spe_ref = first_generated(row)
        if not question or not spe_ref:
            continue
        if route_fn is None:
            cluster_id, route_type = "base", "base_model"
        else:
            cluster_id, route_type = route_fn(role)
        prepared.append(
            {
                "split": "role_specific",
                "role": role,
                "question": question,
                "cluster_id": cluster_id,
                "route_type": route_type,
                "refs": {"SPE": normalize_text(spe_ref)},
            }
        )
        if max_examples is not None and len(prepared) >= max_examples:
            break
    return prepared


def group_rows_by_cluster(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["cluster_id"]].append(row)
    return grouped


def score_predictions(rows, predictions, label):
    metric_values = defaultdict(list)
    for row, pred in zip(rows, predictions):
        for metric_name, ref in row["refs"].items():
            metric_values[metric_name].append(rouge_l_f1(pred, ref))

    summary = {
        "label": label,
        "counts": {key: len(values) for key, values in metric_values.items()},
        "metrics": {
            key: (sum(values) / len(values) if values else None)
            for key, values in metric_values.items()
        },
    }

    metric_means = [value for value in summary["metrics"].values() if value is not None]
    summary["metrics"]["AVG"] = (sum(metric_means) / len(metric_means)) if metric_means else None
    return summary


def build_progress_snapshot(cluster_metrics, route_stats, generated_rows, total_rows):
    ordered = {}
    for cluster_id, payload in sorted(
        cluster_metrics.items(),
        key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]),
    ):
        ordered[str(cluster_id)] = payload
    return {
        "generated_rows": generated_rows,
        "total_rows": total_rows,
        "routing_counts": dict(route_stats),
        "clusters": ordered,
    }


def evaluate_routed_adapters(base_model, tokenizer, rows, adapters_root, args):
    grouped = group_rows_by_cluster(rows)
    route_stats = defaultdict(int)
    total_rows = len(rows)
    generated_rows = 0
    progress_path = Path(args.output).with_suffix(Path(args.output).suffix + ".progress.json")
    selected_cluster_ids = set(parse_cluster_ids(args.cluster_ids))

    aggregate_sums = defaultdict(float)
    aggregate_counts = defaultdict(int)
    cluster_metrics = {}

    for cluster_id, cluster_rows in sorted(
        grouped.items(),
        key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]),
    ):
        if selected_cluster_ids and str(cluster_id) not in selected_cluster_ids:
            continue

        adapter_dir = Path(adapters_root) / f"cluster_{cluster_id}"
        if not adapter_dir.exists():
            if args.skip_missing_adapters:
                print(f"[skip] adapter not found for cluster {cluster_id}: {adapter_dir}")
                continue
            raise FileNotFoundError(f"Cluster adapter directory was not found: {adapter_dir}")

        print(f"[cluster {cluster_id}] loading adapter from {adapter_dir}")
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        model.eval()
        per_cluster_sums = defaultdict(float)
        per_cluster_counts = defaultdict(int)

        progress_bar = tqdm(cluster_rows, desc=f"cluster_{cluster_id}", leave=True)
        for row in progress_bar:
            prediction = generate_response(
                model,
                tokenizer,
                role=row["role"],
                question=row["question"],
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            for metric_name, ref in row["refs"].items():
                score = rouge_l_f1(prediction, ref)
                per_cluster_sums[metric_name] += score
                per_cluster_counts[metric_name] += 1
                aggregate_sums[metric_name] += score
                aggregate_counts[metric_name] += 1

            route_stats[f"{row['route_type']}:{cluster_id}"] += 1
            generated_rows += 1
            progress_bar.set_postfix(done=f"{generated_rows}/{total_rows}")

            if args.save_every > 0 and (generated_rows % args.save_every == 0):
                cluster_metrics[str(cluster_id)] = {
                    "counts": {key: per_cluster_counts[key] for key in per_cluster_counts},
                    "metrics": {
                        key: (per_cluster_sums[key] / per_cluster_counts[key]) if per_cluster_counts[key] else None
                        for key in per_cluster_counts
                    },
                }
                save_json(
                    str(progress_path),
                    build_progress_snapshot(cluster_metrics, route_stats, generated_rows, total_rows),
                )

        cluster_summary = {
            "counts": {key: per_cluster_counts[key] for key in per_cluster_counts},
            "metrics": {
                key: (per_cluster_sums[key] / per_cluster_counts[key]) if per_cluster_counts[key] else None
                for key in per_cluster_counts
            },
        }
        cluster_metric_values = [value for value in cluster_summary["metrics"].values() if value is not None]
        cluster_summary["metrics"]["AVG"] = (
            sum(cluster_metric_values) / len(cluster_metric_values) if cluster_metric_values else None
        )
        cluster_metrics[str(cluster_id)] = cluster_summary
        save_json(
            str(progress_path),
            build_progress_snapshot(cluster_metrics, route_stats, generated_rows, total_rows),
        )
        print(f"[cluster {cluster_id}] completed {len(cluster_rows)} rows")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "label": "routed_cluster_adapters",
        "counts": {key: aggregate_counts[key] for key in aggregate_counts},
        "metrics": {
            key: (aggregate_sums[key] / aggregate_counts[key]) if aggregate_counts[key] else None
            for key in aggregate_counts
        },
        "by_cluster": cluster_metrics,
        "progress_file": str(progress_path),
    }
    metric_means = [value for value in summary["metrics"].values() if value is not None]
    summary["metrics"]["AVG"] = (sum(metric_means) / len(metric_means)) if metric_means else None
    return summary, dict(route_stats)


def evaluate_base_model(base_model, tokenizer, rows, args):
    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)
    route_stats = defaultdict(int)

    progress_path = Path(args.output).with_suffix(Path(args.output).suffix + ".progress.json")
    total_rows = len(rows)
    generated_rows = 0

    progress_bar = tqdm(rows, desc="base_model", leave=True)
    for row in progress_bar:
        prediction = generate_response(
            base_model,
            tokenizer,
            role=row["role"],
            question=row["question"],
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        for metric_name, ref in row["refs"].items():
            score = rouge_l_f1(prediction, ref)
            metric_sums[metric_name] += score
            metric_counts[metric_name] += 1

        route_stats[row["route_type"]] += 1
        generated_rows += 1
        progress_bar.set_postfix(done=f"{generated_rows}/{total_rows}")

        if args.save_every > 0 and (generated_rows % args.save_every == 0):
            snapshot = {
                "generated_rows": generated_rows,
                "total_rows": total_rows,
                "routing_counts": dict(route_stats),
                "metrics": {
                    key: (metric_sums[key] / metric_counts[key]) if metric_counts[key] else None
                    for key in metric_counts
                },
            }
            save_json(str(progress_path), snapshot)

    summary = {
        "label": "base_model",
        "counts": {key: metric_counts[key] for key in metric_counts},
        "metrics": {
            key: (metric_sums[key] / metric_counts[key]) if metric_counts[key] else None
            for key in metric_counts
        },
        "progress_file": str(progress_path),
    }
    metric_means = [value for value in summary["metrics"].values() if value is not None]
    summary["metrics"]["AVG"] = (sum(metric_means) / len(metric_means)) if metric_means else None
    save_json(
        str(progress_path),
        {
            "generated_rows": generated_rows,
            "total_rows": total_rows,
            "routing_counts": dict(route_stats),
            "metrics": summary["metrics"],
        },
    )
    return summary, dict(route_stats)


def evaluate_single_adapter(base_model, tokenizer, rows, args):
    if not args.adapter_path:
        raise ValueError("--adapter-path is required for --eval-mode adapter")

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory was not found: {adapter_path}")

    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)
    route_stats = defaultdict(int)

    progress_path = Path(args.output).with_suffix(Path(args.output).suffix + ".progress.json")
    total_rows = len(rows)
    generated_rows = 0

    progress_bar = tqdm(rows, desc=args.adapter_label, leave=True)
    for row in progress_bar:
        prediction = generate_response(
            model,
            tokenizer,
            role=row["role"],
            question=row["question"],
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        for metric_name, ref in row["refs"].items():
            score = rouge_l_f1(prediction, ref)
            metric_sums[metric_name] += score
            metric_counts[metric_name] += 1

        route_stats[args.adapter_label] += 1
        generated_rows += 1
        progress_bar.set_postfix(done=f"{generated_rows}/{total_rows}")

        if args.save_every > 0 and (generated_rows % args.save_every == 0):
            snapshot = {
                "generated_rows": generated_rows,
                "total_rows": total_rows,
                "routing_counts": dict(route_stats),
                "metrics": {
                    key: (metric_sums[key] / metric_counts[key]) if metric_counts[key] else None
                    for key in metric_counts
                },
            }
            save_json(str(progress_path), snapshot)

    summary = {
        "label": args.adapter_label,
        "adapter_path": str(adapter_path),
        "counts": {key: metric_counts[key] for key in metric_counts},
        "metrics": {
            key: (metric_sums[key] / metric_counts[key]) if metric_counts[key] else None
            for key in metric_counts
        },
        "progress_file": str(progress_path),
    }
    metric_means = [value for value in summary["metrics"].values() if value is not None]
    summary["metrics"]["AVG"] = (sum(metric_means) / len(metric_means)) if metric_means else None
    save_json(
        str(progress_path),
        {
            "generated_rows": generated_rows,
            "total_rows": total_rows,
            "routing_counts": dict(route_stats),
            "metrics": summary["metrics"],
        },
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary, dict(route_stats)


def load_prediction_rows(rows, prediction_rows):
    lookup = {}
    for pred_row in prediction_rows:
        role = pred_row.get("role", "UnknownRole")
        question = pred_row.get("question", "")
        pred = first_generated(pred_row)
        if question and pred:
            lookup[(role, normalize_key(question))] = normalize_text(pred)

    aligned_rows = []
    predictions = []
    missing = 0
    for row in rows:
        key = (row["role"], normalize_key(row["question"]))
        pred = lookup.get(key)
        if pred is None:
            missing += 1
            continue
        aligned_rows.append(row)
        predictions.append(pred)

    return aligned_rows, predictions, missing


def main():
    args = parse_args()
    route_fn = None
    if args.eval_mode == "routed":
        if not args.clusters or not args.adapters_root:
            raise ValueError("--clusters and --adapters-root are required for --eval-mode routed")
        clusters_payload = load_json(args.clusters)
        route_fn = build_cluster_router(clusters_payload)

    general_test_file = args.general_test_file or default_path(args.benchmark_root, "general/test.jsonl")
    role_specific_test_file = args.role_specific_test_file or default_path(args.benchmark_root, "role_specific/test.jsonl")
    general_baseline_file = args.general_baseline_file or default_path(args.benchmark_root, "general/rolegpt_baseline.jsonl")
    role_specific_baseline_file = args.role_specific_baseline_file or default_path(args.benchmark_root, "role_specific/rolegpt_baseline.jsonl")

    raw_map = load_raw_reference_map(args.dataset_repo, args.raw_instructions_file)
    general_rows_raw = load_jsonl_from_repo(args.dataset_repo, general_test_file)
    role_specific_rows_raw = load_jsonl_from_repo(args.dataset_repo, role_specific_test_file)

    general_rows, missing_raw = prepare_general_rows(
        general_rows_raw,
        raw_map=raw_map,
        route_fn=route_fn,
        max_examples=args.max_examples_per_split,
    )
    role_specific_rows = prepare_role_specific_rows(
        role_specific_rows_raw,
        route_fn=route_fn,
        max_examples=args.max_examples_per_split,
    )
    all_rows = general_rows + role_specific_rows

    tokenizer, base_model = load_base_model_and_tokenizer(
        args.model_id,
        use_4bit=args.use_4bit,
        trust_remote_code=args.trust_remote_code,
    )

    if args.eval_mode == "routed":
        model_summary, route_stats = evaluate_routed_adapters(
            base_model,
            tokenizer,
            all_rows,
            adapters_root=args.adapters_root,
            args=args,
        )
        result_key = "routed_cluster_adapters"
    elif args.eval_mode == "adapter":
        model_summary, route_stats = evaluate_single_adapter(
            base_model,
            tokenizer,
            all_rows,
            args=args,
        )
        result_key = "single_adapter"
    else:
        model_summary, route_stats = evaluate_base_model(
            base_model,
            tokenizer,
            all_rows,
            args=args,
        )
        result_key = "base_model"

    output = {
        "metadata": {
            "eval_mode": args.eval_mode,
            "model_id": args.model_id,
            "dataset_repo": args.dataset_repo,
            "benchmark_root": args.benchmark_root,
            "general_test_file": general_test_file,
            "role_specific_test_file": role_specific_test_file,
            "raw_instructions_file": args.raw_instructions_file,
        },
        "paper_metric_definition": {
            "metric": "Rouge-L",
            "categories": {
                "RAW": "general instructions without role-playing",
                "CUS": "customized general instruction responses with role-playing",
                "SPE": "role-specific instruction responses",
                "AVG": "average of available RAW/CUS/SPE means",
            },
        },
        "evaluation_data": {
            "general_examples": len(general_rows),
            "role_specific_examples": len(role_specific_rows),
            "missing_raw_matches": missing_raw,
            "routing_counts": route_stats,
        },
        "results": {
            result_key: model_summary,
        },
    }

    baseline_available = True
    try:
        general_baseline_rows = load_jsonl_from_repo(args.dataset_repo, general_baseline_file)
        role_specific_baseline_rows = load_jsonl_from_repo(args.dataset_repo, role_specific_baseline_file)
    except Exception as exc:
        baseline_available = False
        output["baseline_note"] = f"Could not load RoleGPT baseline files automatically: {exc}"

    if baseline_available:
        general_aligned, general_predictions, general_missing = load_prediction_rows(general_rows, general_baseline_rows)
        role_aligned, role_predictions, role_missing = load_prediction_rows(role_specific_rows, role_specific_baseline_rows)
        baseline_rows = general_aligned + role_aligned
        baseline_predictions_all = general_predictions + role_predictions
        output["results"]["rolegpt_baseline"] = score_predictions(
            baseline_rows,
            baseline_predictions_all,
            "rolegpt_baseline",
        )
        output["baseline_alignment"] = {
            "general_missing_predictions": general_missing,
            "role_specific_missing_predictions": role_missing,
        }

    save_json(args.output, output)
    print(json.dumps(output, indent=2, ensure_ascii=False))

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
