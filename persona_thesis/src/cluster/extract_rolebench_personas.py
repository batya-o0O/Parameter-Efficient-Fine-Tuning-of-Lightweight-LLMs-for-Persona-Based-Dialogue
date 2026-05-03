import argparse
import random
import re
from collections import defaultdict

from datasets import load_dataset

from src.utils.io import ensure_dir, save_json


def clean_text(value: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def pick_answer(example) -> str:
    generated = example.get("generated", [])
    if isinstance(generated, list) and generated:
        return generated[0]
    if isinstance(generated, str):
        return generated
    return ""


def build_description(role: str, examples: list[dict], examples_per_role: int, max_question_chars: int, max_answer_chars: int, seed: int) -> str:
    if not examples:
        return role

    rng = random.Random(f"{seed}:{role}")
    if len(examples) > examples_per_role:
        chosen = rng.sample(examples, examples_per_role)
    else:
        chosen = list(examples)

    chosen.sort(key=lambda item: (len(str(item.get("question", ""))), str(item.get("question", ""))))

    parts = [
        f"Role: {role}",
        f"Representative in-character examples from RoleBench training ({len(chosen)} sampled from {len(examples)} total):",
    ]
    for index, example in enumerate(chosen, start=1):
        question = clean_text(example.get("question", ""), max_question_chars)
        answer = clean_text(pick_answer(example), max_answer_chars)
        parts.append(f"{index}. User: {question}")
        parts.append(f"   Assistant: {answer}")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_repo", default="ZenMoore/RoleBench")
    ap.add_argument("--train_file", default="rolebench-eng/role-generalization/general/train.jsonl")
    ap.add_argument("--out", default="data/personas_rolebench.json")
    ap.add_argument(
        "--mode",
        default="sampled_qa",
        choices=["role_name", "sampled_qa"],
        help="How to build persona descriptions for clustering.",
    )
    ap.add_argument("--examples_per_role", type=int, default=8)
    ap.add_argument("--max_question_chars", type=int, default=180)
    ap.add_argument("--max_answer_chars", type=int, default=220)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_dir("data")
    raw = load_dataset(args.dataset_repo, data_files={"train": args.train_file})
    train_rows = raw["train"]
    roles = sorted(list(set(train_rows["role"])))

    if args.mode == "role_name":
        personas = [{"persona_id": role, "description": role} for role in roles]
    else:
        by_role: dict[str, list[dict]] = defaultdict(list)
        for row in train_rows:
            by_role[row["role"]].append(row)

        personas = []
        for role in roles:
            description = build_description(
                role=role,
                examples=by_role.get(role, []),
                examples_per_role=args.examples_per_role,
                max_question_chars=args.max_question_chars,
                max_answer_chars=args.max_answer_chars,
                seed=args.seed,
            )
            personas.append({"persona_id": role, "description": description})

    save_json(args.out, personas)
    print("Saved", args.out, "count:", len(personas), "mode:", args.mode)


if __name__ == "__main__":
    main()
