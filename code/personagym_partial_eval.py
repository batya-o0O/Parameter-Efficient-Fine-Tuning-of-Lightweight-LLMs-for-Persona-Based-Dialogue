#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


TASK_NAME_MAP = {
    "expected_action": "Expected Action",
    "toxicity_control": "Toxicity",
    "linguistic_habits": "Linguistic Habits",
    "persona_consistency": "Persona Consistency",
    "action_justification": "Action Justification",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert chunked JSONL model outputs to PersonaGym saved_responses format and optionally evaluate them."
    )
    p.add_argument("--inputs", nargs="+", required=True, help="Input JSONL chunk files.")
    p.add_argument("--output-dir", required=True, help="Directory for PersonaGym-compatible *_qa.json files.")
    p.add_argument(
        "--persona-list-out",
        default=None,
        help="Optional path to save the evaluated persona list as JSON.",
    )
    p.add_argument(
        "--summary-out",
        default=None,
        help="Optional path to save conversion or evaluation summary as JSON.",
    )
    p.add_argument(
        "--personagym-code-dir",
        default="PersonaGym-master/PersonaGym-master/code",
        help="Path to PersonaGym code directory.",
    )
    p.add_argument(
        "--evaluate",
        action="store_true",
        help="Run PersonaGym scoring after conversion. Requires API keys configured in PersonaGym code/api_keys.py.",
    )
    p.add_argument(
        "--model-label",
        default=None,
        help="Optional label recorded in the summary output.",
    )
    p.add_argument(
        "--max-questions-per-task",
        type=int,
        default=5,
        help="Maximum number of question-answer pairs to score per task.",
    )
    p.add_argument(
        "--max-personas",
        type=int,
        default=None,
        help="If set, evaluate only the first N personas after sorting.",
    )
    return p.parse_args()


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return rows


def parse_example_id(example_id: str) -> Tuple[str, str, int]:
    persona, task, idx = example_id.rsplit("-", 2)
    return persona, task, int(idx)


def convert_rows(rows: List[dict]) -> Dict[str, Dict[str, List[List[str]]]]:
    persona_task_pairs: Dict[str, Dict[str, List[Tuple[int, List[str]]]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        example_id = row["example_id"]
        persona_from_id, task_from_id, question_idx = parse_example_id(example_id)
        persona = row.get("persona_id") or persona_from_id
        task_key = row.get("task") or task_from_id
        task_name = TASK_NAME_MAP[task_key]
        prompt = row["prompt"]
        response = row["response"]
        persona_task_pairs[persona][task_name].append((question_idx, [prompt, response]))

    converted: Dict[str, Dict[str, List[List[str]]]] = {}
    for persona, task_map in persona_task_pairs.items():
        converted[persona] = {}
        for task_name, qa_items in task_map.items():
            qa_items.sort(key=lambda item: item[0])
            converted[persona][task_name] = [qa for _, qa in qa_items]
    return converted


def write_personagym_files(persona_payloads: Dict[str, Dict[str, List[List[str]]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for persona, payload in persona_payloads.items():
        with (output_dir / f"{persona}_qa.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    temp_path.replace(path)


def load_api_key_file(api_key_file: Path) -> str:
    if not api_key_file.exists():
        return "missing"
    text = api_key_file.read_text(encoding="utf-8")
    if 'Insert OpenAI key here' in text or 'Insert Claude key here' in text or 'Insert Llama key here' in text:
        return "placeholder"
    return "configured"


def evaluate_personagym(
    personagym_code_dir: Path,
    saved_responses_dir: Path,
    personas: List[str],
    progress_path: Path,
    model_label: str | None,
) -> Dict[str, dict]:
    personagym_code_dir = personagym_code_dir.resolve()
    saved_responses_dir = saved_responses_dir.resolve()
    progress_path = progress_path.resolve()
    api_status = load_api_key_file(personagym_code_dir / "api_keys.py")
    if api_status != "configured":
        raise RuntimeError(
            "PersonaGym API keys are not configured. Fill PersonaGym-master/PersonaGym-master/code/api_keys.py first."
        )

    existing_results: Dict[str, dict] = {}
    partial_batches: Dict[str, Dict[str, List[float]]] = {}
    if progress_path.exists():
        try:
            existing_payload = json.loads(progress_path.read_text(encoding="utf-8"))
            existing_results = existing_payload.get("per_persona_scores", {}) or {}
            partial_batches = existing_payload.get("partial_batches", {}) or {}
            print(
                f"Found existing evaluation progress for {len(existing_results)} personas at {progress_path}"
            )
        except Exception as exc:
            print(f"Could not load existing progress from {progress_path}: {exc}")

    completed_personas = set(existing_results.keys())
    print(f"Saving progress to: {progress_path}")

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

                    selected_qa = task_to_qa[task][i:i + 5]
                    rubric = open(f'../rubrics/{task}.txt').read()
                    sys_prompt, scoring_prompt = personagym_run.format_rubrics(persona, rubric, selected_qa)
                    batch_score = personagym_run.score_rubrics(sys_prompt, scoring_prompt)
                    task_batch_scores.append(batch_score)

                    persona_partial[task] = task_batch_scores
                    partial_batches[persona] = persona_partial
                    write_json_atomic(progress_path, summarize_progress(results, partial_batches, model_label))
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
            write_json_atomic(progress_path, summarize_progress(results, partial_batches, model_label))
            print(f"Saved evaluation progress: {len(results)}/{len(personas)} personas")
        return results
    finally:
        os.chdir(original_cwd)


def summarize_payload(persona_payloads: Dict[str, Dict[str, List[List[str]]]], model_label: str | None) -> dict:
    task_counts = defaultdict(int)
    for task_map in persona_payloads.values():
        for task_name, qa in task_map.items():
            task_counts[task_name] += len(qa)

    return {
        "model_label": model_label,
        "persona_count": len(persona_payloads),
        "task_counts": dict(task_counts),
        "personas": sorted(persona_payloads.keys()),
    }


def summarize_scores(results: Dict[str, dict], model_label: str | None) -> dict:
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
        "persona_count": len(results),
        "average_scores": averages,
        "per_persona_scores": results,
    }


def summarize_progress(
    results: Dict[str, dict],
    partial_batches: Dict[str, Dict[str, List[float]]],
    model_label: str | None,
) -> dict:
    payload = summarize_scores(results, model_label)
    payload["partial_batches"] = partial_batches
    return payload


def limit_questions_per_task(
    persona_payloads: Dict[str, Dict[str, List[List[str]]]],
    max_questions_per_task: int,
) -> Dict[str, Dict[str, List[List[str]]]]:
    if max_questions_per_task <= 0:
        return persona_payloads

    limited: Dict[str, Dict[str, List[List[str]]]] = {}
    for persona, task_map in persona_payloads.items():
        limited[persona] = {}
        for task_name, qa_items in task_map.items():
            limited[persona][task_name] = qa_items[:max_questions_per_task]
    return limited


def limit_personas(
    persona_payloads: Dict[str, Dict[str, List[List[str]]]],
    max_personas: int | None,
) -> Dict[str, Dict[str, List[List[str]]]]:
    if max_personas is None or max_personas <= 0:
        return persona_payloads

    selected_personas = sorted(persona_payloads.keys())[:max_personas]
    return {persona: persona_payloads[persona] for persona in selected_personas}


def main() -> int:
    args = parse_args()
    input_paths = [Path(p) for p in args.inputs]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    all_rows: List[dict] = []
    for path in input_paths:
        all_rows.extend(read_jsonl(path))

    persona_payloads = convert_rows(all_rows)
    persona_payloads = limit_questions_per_task(persona_payloads, args.max_questions_per_task)
    persona_payloads = limit_personas(persona_payloads, args.max_personas)
    output_dir = Path(args.output_dir)
    write_personagym_files(persona_payloads, output_dir)

    personas = sorted(persona_payloads.keys())
    if args.persona_list_out:
        persona_list_path = Path(args.persona_list_out)
        persona_list_path.parent.mkdir(parents=True, exist_ok=True)
        with persona_list_path.open("w", encoding="utf-8") as f:
            json.dump(personas, f, indent=2, ensure_ascii=False)

    if args.evaluate:
        progress_path = Path(args.summary_out) if args.summary_out else output_dir / "evaluation_progress.json"
        results = evaluate_personagym(
            Path(args.personagym_code_dir),
            output_dir,
            personas,
            progress_path,
            args.model_label,
        )
        summary = summarize_scores(results, args.model_label)
    else:
        summary = summarize_payload(persona_payloads, args.model_label)

    if args.summary_out:
        summary_path = Path(args.summary_out)
        write_json_atomic(summary_path, summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
