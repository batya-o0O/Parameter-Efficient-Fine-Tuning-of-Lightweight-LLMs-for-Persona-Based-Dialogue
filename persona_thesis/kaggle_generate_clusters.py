#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_step(command: list[str], cwd: Path) -> None:
    print("\n[run]", " ".join(command))
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(cwd) if not prev else f"{cwd}:{prev}"
    subprocess.run(command, check=True, cwd=str(cwd), env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kaggle-friendly wrapper to regenerate RoleBench persona files and cluster assignments without retraining."
    )
    parser.add_argument("--dataset_repo", default="ZenMoore/RoleBench")
    parser.add_argument("--train_file", default="rolebench-eng/role-generalization/general/train.jsonl")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--persona_mode", default="sampled_qa", choices=["role_name", "sampled_qa"])
    parser.add_argument("--examples_per_role", type=int, default=8)
    parser.add_argument("--max_question_chars", type=int, default=180)
    parser.add_argument("--max_answer_chars", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--persona_out", default="data/personas_rolebench_profiles.json")
    parser.add_argument("--cluster_out", default="data/clusters_k5_profiles.json")
    parser.add_argument(
        "--also_generate_names_only",
        action="store_true",
        help="Also generate a names-only persona file and cluster file.",
    )
    parser.add_argument("--names_only_persona_out", default="data/personas_rolebench_names_only.json")
    parser.add_argument("--names_only_cluster_out", default="data/clusters_k5_names_only.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    python = sys.executable

    Path(root / "data").mkdir(parents=True, exist_ok=True)

    run_step(
        [
            python,
            "src/cluster/extract_rolebench_personas.py",
            "--dataset_repo",
            args.dataset_repo,
            "--train_file",
            args.train_file,
            "--mode",
            args.persona_mode,
            "--examples_per_role",
            str(args.examples_per_role),
            "--max_question_chars",
            str(args.max_question_chars),
            "--max_answer_chars",
            str(args.max_answer_chars),
            "--seed",
            str(args.seed),
            "--out",
            args.persona_out,
        ],
        cwd=root,
    )

    run_step(
        [
            python,
            "src/cluster/cluster_personas.py",
            "--personas",
            args.persona_out,
            "--k",
            str(args.k),
            "--embed_model",
            args.embed_model,
            "--out",
            args.cluster_out,
        ],
        cwd=root,
    )

    if args.also_generate_names_only:
        run_step(
            [
                python,
                "src/cluster/extract_rolebench_personas.py",
                "--dataset_repo",
                args.dataset_repo,
                "--train_file",
                args.train_file,
                "--mode",
                "role_name",
                "--out",
                args.names_only_persona_out,
            ],
            cwd=root,
        )

        run_step(
            [
                python,
                "src/cluster/cluster_personas.py",
                "--personas",
                args.names_only_persona_out,
                "--k",
                str(args.k),
                "--embed_model",
                args.embed_model,
                "--out",
                args.names_only_cluster_out,
            ],
            cwd=root,
        )

    print("\nDone.")
    print("Persona file:", args.persona_out)
    print("Cluster file:", args.cluster_out)
    if args.also_generate_names_only:
        print("Names-only persona file:", args.names_only_persona_out)
        print("Names-only cluster file:", args.names_only_cluster_out)


if __name__ == "__main__":
    main()
