#!/usr/bin/env python3
import argparse
import gc
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils.io import load_json, save_json


def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate RoleBench test rows by routing each persona to its nearest cluster adapter."
    )
    ap.add_argument("--clusters", required=True, help="Path to clusters_k*.json")
    ap.add_argument("--adapters-root", required=True, help="Directory containing cluster_<id> adapter folders")
    ap.add_argument("--output", required=True, help="Path to summary JSON output")
    ap.add_argument("--model-id", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--dataset-repo", default="ZenMoore/RoleBench")
    ap.add_argument("--train-file", default="rolebench-eng/role-generalization/general/train.jsonl")
    ap.add_argument("--eval-file", default="rolebench-eng/role-generalization/general/test.jsonl")
    ap.add_argument("--global-adapter-path", default=None, help="Optional single adapter to score on the same test rows")
    ap.add_argument("--global-label", default="global_adapter")
    ap.add_argument("--paper-results", default=None, help="Optional thesis_reported_metrics.json path")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--use-4bit", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    return ap.parse_args()


def pick_answer(ex):
    gen = ex.get("generated", [])
    if isinstance(gen, list) and len(gen) > 0:
        return gen[0]
    return ""


def build_prefix(role, question):
    return (
        f"<|system|>\nYou are role-playing as {role}. Stay in character.\n<|end|>\n"
        f"<|user|>\n{question}\n<|end|>\n"
        f"<|assistant|>\n"
    )


def build_full_text(role, question, answer):
    return build_prefix(role, question) + f"{answer}\n<|end|>\n"


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
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        "device_map": "auto",
        "dtype": pick_dtype(),
        "trust_remote_code": trust_remote_code,
        "attn_implementation": "eager",
    }
    quant_config = make_quant_config(use_4bit)
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    return tokenizer, model


def load_adapter_model(base_model, adapter_path):
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model


def encode_example(tokenizer, role, question, answer, max_length):
    prefix = build_prefix(role, question)
    full = build_full_text(role, question, answer)

    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if len(prefix_ids) >= len(full_ids):
        return None

    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
    if len(full_ids) > max_length:
        full_ids = full_ids[-max_length:]
        labels = labels[-max_length:]

    answer_token_count = sum(1 for x in labels if x != -100)
    if answer_token_count == 0:
        return None

    return {
        "input_ids": full_ids,
        "labels": labels,
        "answer_token_count": answer_token_count,
    }


def collate_batch(rows, pad_token_id):
    max_len = max(len(row["input_ids"]) for row in rows)
    input_ids = []
    attention_mask = []
    labels = []
    for row in rows:
        pad_len = max_len - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(row["input_ids"]) + [0] * pad_len)
        labels.append(row["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def score_batch(model, batch, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.inference_mode():
        outputs = model(**batch)

    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = batch["labels"][:, 1:].contiguous()
    valid_mask = shift_labels != -100

    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    flat_mask = valid_mask.view(-1)

    valid_logits = flat_logits[flat_mask]
    valid_labels = flat_labels[flat_mask]

    token_count = int(valid_labels.numel())
    if token_count == 0:
        return {"nll_sum": 0.0, "correct": 0, "tokens": 0}

    token_losses = F.cross_entropy(valid_logits, valid_labels, reduction="none")
    predictions = valid_logits.argmax(dim=-1)
    correct = int((predictions == valid_labels).sum().item())

    return {
        "nll_sum": float(token_losses.sum().item()),
        "correct": correct,
        "tokens": token_count,
    }


def aggregate_metrics(name, token_count, nll_sum, correct, example_count):
    mean_nll = (nll_sum / token_count) if token_count else None
    perplexity = math.exp(mean_nll) if mean_nll is not None else None
    token_accuracy = (correct / token_count) if token_count else None
    return {
        "label": name,
        "examples": example_count,
        "answer_tokens": token_count,
        "mean_nll": mean_nll,
        "perplexity": perplexity,
        "token_accuracy": token_accuracy,
    }


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

    sorted_cluster_ids = sorted(
        centroid_map.keys(),
        key=lambda value: int(value) if str(value).isdigit() else str(value),
    )

    def route(role_text):
        if role_text in role_to_cluster:
            return role_to_cluster[role_text], "known_member"
        embedding = embedder.encode([role_text], normalize_embeddings=True, show_progress_bar=False)[0]
        best_cluster = None
        best_score = None
        for cluster_id in sorted_cluster_ids:
            score = float((embedding * centroid_map[cluster_id]).sum())
            if best_score is None or score > best_score:
                best_score = score
                best_cluster = cluster_id
        return best_cluster, "nearest_centroid"

    return route


def build_eval_rows(dataset, route_fn, max_examples=None):
    rows = []
    routing_counts = defaultdict(int)
    route_cache = {}

    for example in dataset["test"]:
        role = example.get("role", "UnknownRole")
        question = example.get("question", "")
        answer = pick_answer(example)
        if not question or not answer:
            continue

        if role not in route_cache:
            cluster_id, route_type = route_fn(role)
            route_cache[role] = (cluster_id, route_type)
        cluster_id, route_type = route_cache[role]

        rows.append(
            {
                "role": role,
                "question": question,
                "answer": answer,
                "cluster_id": cluster_id,
                "route_type": route_type,
            }
        )
        routing_counts[f"{route_type}:{cluster_id}"] += 1
        if max_examples is not None and len(rows) >= max_examples:
            break

    return rows, dict(routing_counts)


def group_rows_by_cluster(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["cluster_id"]].append(row)
    return grouped


def evaluate_rows(model, tokenizer, rows, batch_size, max_length):
    device = getattr(model, "device", None) or next(model.parameters()).device
    encoded = []
    for row in rows:
        payload = encode_example(tokenizer, row["role"], row["question"], row["answer"], max_length)
        if payload is not None:
            encoded.append(payload)

    total_nll = 0.0
    total_correct = 0
    total_tokens = 0

    for start in range(0, len(encoded), batch_size):
        batch_rows = encoded[start : start + batch_size]
        batch = collate_batch(batch_rows, tokenizer.pad_token_id)
        score = score_batch(model, batch, device)
        total_nll += score["nll_sum"]
        total_correct += score["correct"]
        total_tokens += score["tokens"]

    return aggregate_metrics("eval", total_tokens, total_nll, total_correct, len(encoded))


def evaluate_routed_clusters(base_model, tokenizer, grouped_rows, adapters_root, batch_size, max_length):
    adapters_root = Path(adapters_root)
    total_nll = 0.0
    total_correct = 0
    total_tokens = 0
    total_examples = 0
    by_cluster = {}

    for cluster_id, rows in sorted(
        grouped_rows.items(),
        key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]),
    ):
        adapter_dir = adapters_root / f"cluster_{cluster_id}"
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Cluster adapter directory was not found: {adapter_dir}")

        model = load_adapter_model(base_model, str(adapter_dir))
        metrics = evaluate_rows(model, tokenizer, rows, batch_size=batch_size, max_length=max_length)
        metrics["label"] = f"cluster_{cluster_id}"
        metrics["adapter_path"] = str(adapter_dir)
        by_cluster[str(cluster_id)] = metrics

        total_nll += metrics["mean_nll"] * metrics["answer_tokens"] if metrics["mean_nll"] is not None else 0.0
        total_correct += int((metrics["token_accuracy"] or 0.0) * metrics["answer_tokens"]) if metrics["token_accuracy"] is not None else 0
        total_tokens += metrics["answer_tokens"]
        total_examples += metrics["examples"]

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = aggregate_metrics(
        "routed_cluster_adapters",
        total_tokens,
        total_nll,
        total_correct,
        total_examples,
    )
    summary["by_cluster"] = by_cluster
    return summary


def evaluate_global_adapter(base_model, tokenizer, rows, adapter_path, label, batch_size, max_length):
    model = load_adapter_model(base_model, adapter_path)
    metrics = evaluate_rows(model, tokenizer, rows, batch_size=batch_size, max_length=max_length)
    metrics["label"] = label
    metrics["adapter_path"] = adapter_path
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def compare_metrics(left, right):
    result = {"left": left["label"], "right": right["label"]}
    for key in ["mean_nll", "perplexity", "token_accuracy"]:
        if left.get(key) is not None and right.get(key) is not None:
            result[f"delta_{key}"] = left[key] - right[key]
    return result


def main():
    args = parse_args()

    clusters_payload = load_json(args.clusters)
    route_fn = build_cluster_router(clusters_payload)

    dataset = load_dataset(
        args.dataset_repo,
        data_files={"train": args.train_file, "test": args.eval_file},
        cache_dir=os.environ.get("HF_DATASETS_CACHE", None),
    )

    eval_rows, routing_counts = build_eval_rows(dataset, route_fn, max_examples=args.max_examples)
    grouped_rows = group_rows_by_cluster(eval_rows)

    tokenizer, base_model = load_base_model_and_tokenizer(
        args.model_id,
        use_4bit=args.use_4bit,
        trust_remote_code=args.trust_remote_code,
    )

    routed_summary = evaluate_routed_clusters(
        base_model,
        tokenizer,
        grouped_rows,
        adapters_root=args.adapters_root,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    comparisons = []
    outputs = {
        "metadata": {
            "model_id": args.model_id,
            "dataset_repo": args.dataset_repo,
            "eval_file": args.eval_file,
            "clusters": args.clusters,
            "adapters_root": args.adapters_root,
            "max_examples": args.max_examples,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
        },
        "routing": {
            "examples_evaluated": len(eval_rows),
            "unique_roles": len({row["role"] for row in eval_rows}),
            "cluster_counts": {str(cluster_id): len(rows) for cluster_id, rows in grouped_rows.items()},
            "route_type_counts": routing_counts,
        },
        "rolebench_eval": {
            "routed_cluster_adapters": routed_summary,
        },
    }

    if args.global_adapter_path:
        global_summary = evaluate_global_adapter(
            base_model,
            tokenizer,
            eval_rows,
            adapter_path=args.global_adapter_path,
            label=args.global_label,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        outputs["rolebench_eval"]["global_adapter"] = global_summary
        comparisons.append(compare_metrics(routed_summary, global_summary))

    if args.paper_results:
        paper_payload = load_json(args.paper_results)
        outputs["paper_reference"] = {
            "source": args.paper_results,
            "note": (
                "These are Chapter 5 benchmark results from the thesis. "
                "They are useful as context, but they are not directly comparable "
                "to RoleBench test-set loss or token-accuracy metrics."
            ),
            "personagym": paper_payload.get("personagym", []),
            "characterbench": paper_payload.get("characterbench", []),
            "mpi_variance": paper_payload.get("mpi_variance", []),
            "contradiction_rate": paper_payload.get("contradiction_rate", []),
        }

    if comparisons:
        outputs["comparisons"] = comparisons

    save_json(args.output, outputs)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
